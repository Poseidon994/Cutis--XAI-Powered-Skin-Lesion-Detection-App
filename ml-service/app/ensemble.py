"""Image-network ensemble: loads the fine-tuned fold networks and scores one lesion crop exactly like the submission notebook."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .artifacts import Artifacts
from .config import Settings


@dataclass
class Net:
    backbone: str
    fold: int
    module: object
    img: object
    cl: bool
    res: int
    bspec: dict


def _softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


class Ensemble:
    def __init__(self, art: Artifacts, settings: Settings, log=print):
        import torch
        self.torch = torch
        self.art, self.cfg, self.log = art, settings, log
        self.L = art.lib
        self.nets: list[Net] = []
        self.stats = art.stats
        img = art.image
        views = settings.tta_views or int(img['views'])
        self.views = self.L.VIEW_SETS[views]
        self.n_views = views
        self.bs = int(img.get('bs_infer', 32))
        self.store_res = int(img['store_res'])
        self.n_aux = int(img['n_aux'])
        self.lock = threading.Lock()
        if settings.torch_threads > 0:
            torch.set_num_threads(settings.torch_threads)
        want = settings.device
        if want == 'auto':
            want = 'cuda' if torch.cuda.is_available() else 'cpu'
        if want == 'cuda' and not torch.cuda.is_available():
            log('ML_DEVICE=cuda requested but CUDA is not available; falling back to CPU')
            want = 'cpu'
        self.dev = torch.device(want)

    @property
    def backbones(self) -> list[str]:
        return sorted({n.backbone for n in self.nets})

    def load(self) -> None:
        torch, L = self.torch, self.L
        img = self.art.image
        k_trained = int(img['folds'])
        k_use = min(k_trained, self.cfg.max_folds) if self.cfg.max_folds > 0 else k_trained
        names = sorted(img['backbones'])
        if self.cfg.backbones:
            names = [n for n in names if n in self.cfg.backbones]
            if not names:
                raise RuntimeError(f'ML_BACKBONES={self.cfg.backbones} matches none of {sorted(img["backbones"])}')
        t0 = time.time()
        for n in names:
            bspec = dict(img['backbones'][n])
            res = int(bspec['res'])
            for k in range(k_use):
                p = self.art.weight_path(n, k)
                if not p.exists():
                    raise FileNotFoundError(f'Missing fine-tuned weights {p}. Was weight saving on in the training run, and is the backbones/ folder copied?')
                net = L.make_net(bspec, False, res, self.n_aux)
                sd = torch.load(str(p), map_location='cpu')
                net.load_state_dict({kk: v.float() for kk, v in sd.items()})
                net = net.to(self.dev).eval()
                cl = bool(bspec.get('cl')) and self.dev.type == 'cuda'
                if cl:
                    net = net.to(memory_format=torch.channels_last)
                for prm in net.parameters():
                    prm.requires_grad_(False)
                self.nets.append(Net(n, k, net, L.Img(self.dev, res, net.mean, net.std), cl, res, bspec))
            self.log(f'  {n}: {k_use} fold network(s) loaded')
        self.log(f'ensemble ready: {len(self.nets)} networks on {self.dev} in {time.time() - t0:.1f}s | views={self.n_views}')

    def explain_nets(self) -> list[Net]:
        k = self.cfg.xai_folds
        return [n for n in self.nets if n.fold < k]

    def raw_logits(self, u8: np.ndarray) -> np.ndarray:
        arr = np.asarray(u8)[None]
        out = np.zeros((len(self.nets), self.L.N_HEADS), np.float64)
        for i, n in enumerate(self.nets):
            out[i] = self.L.predict_logits(n.module, n.img, arr, np.arange(1), self.views, self.bs, n.cl)[0]
        return out

    def standardise(self, net: Net, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64).copy()
        for j in (0, 4):
            mu, sd = self.stats[net.backbone][f'f{net.fold}_h{j}']
            z[j] = (z[j] - mu) / (sd if sd else 1.0)
        return z

    def predict(self, u8: np.ndarray) -> dict:
        t0 = time.time()
        raw = self.raw_logits(u8)
        per_fold: dict[str, list[np.ndarray]] = {}
        all_z0 = []
        for i, n in enumerate(self.nets):
            z = self.standardise(n, raw[i])
            per_fold.setdefault(n.backbone, []).append(z)
            all_z0.append(z[0])
        zb = {b: np.mean(v, axis=0) for b, v in per_fold.items()}
        base = self.L.image_base_scores({b: z[None, :] for b, z in zb.items()})
        lsm = np.array([float(base[f'type_{t}'][0]) for t in self.L.TYPES])
        probs = _softmax(lsm)
        per_backbone = []
        for b in sorted(zb):
            zs = [z[0] for z in per_fold[b]]
            per_backbone.append({'name': b, 'z': float(zb[b][0]), 'fold_sd': float(np.std(zs)) if len(zs) > 1 else None, 'n_folds': len(zs)})
        return {
            'score': float(base['img_mean'][0]),
            'suspicious': float(base['susp_mean'][0]),
            'per_backbone': per_backbone,
            'type_probs': {t: float(p) for t, p in zip(self.L.TYPES, probs)},
            'network_sd': float(np.std(all_z0)) if len(all_z0) > 1 else None,
            'n_networks': len(self.nets),
            'views': self.n_views,
            'runtime_s': round(time.time() - t0, 3),
        }
