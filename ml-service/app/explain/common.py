"""Shared pieces for the three explanation methods: the scalar score being explained, colour overlays, region summaries."""
from __future__ import annotations

import math

import numpy as np
from matplotlib import colormaps
from PIL import Image


# ---------------------------------------------------------------------------------------------------------- scoring
class Scorer:
    """Mean over the explanation networks of the z-scored malignant logit, i.e. the quantity the app's risk score is built from.

    Takes crops at the store resolution (HxW uint8, or float tensors in [0, 1]) and applies the same resize + normalisation as
    the training library (bicubic to the model resolution, then the backbone's mean / std)."""

    def __init__(self, ens, nets=None):
        import torch
        self.torch = torch
        self.ens = ens
        self.nets = list(nets) if nets is not None else ens.explain_nets()
        if not self.nets:
            raise RuntimeError('no networks available for explanations')
        self.dev = ens.dev
        self.mu = [float(ens.stats[n.backbone][f'f{n.fold}_h0'][0]) for n in self.nets]
        self.sd = [float(ens.stats[n.backbone][f'f{n.fold}_h0'][1]) or 1.0 for n in self.nets]

    def prep(self, net, x01):
        import torch.nn.functional as F
        if x01.shape[-1] != net.res or x01.shape[-2] != net.res:
            x01 = F.interpolate(x01, size=(net.res, net.res), mode='bicubic', align_corners=False).clamp(0, 1)
        x = net.img.norm(x01)
        return x.contiguous(memory_format=self.torch.channels_last) if net.cl else x

    def forward_t(self, x01, autocast=False):
        """x01: (B, 3, S, S) float in [0, 1] on the device -> (B,) mean z over the networks. Differentiable."""
        zs = []
        for net, mu, sd in zip(self.nets, self.mu, self.sd):
            x = self.prep(net, x01)
            if autocast and self.dev.type == 'cuda':
                with self.torch.autocast(device_type='cuda', dtype=self.torch.float16):
                    o = net.module(x)[:, 0].float()
                if not bool(self.torch.isfinite(o).all()):
                    o = net.module(x)[:, 0].float()
            else:
                o = net.module(x)[:, 0].float()
            zs.append((o - mu) / sd)
        return self.torch.stack(zs, 0).mean(0)

    def to_tensor(self, u8):
        t = self.torch.from_numpy(np.ascontiguousarray(u8)).to(self.dev)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t.permute(0, 3, 1, 2).float().div(255.0)

    def score_u8(self, batch_u8, chunk=32) -> np.ndarray:
        batch_u8 = np.asarray(batch_u8)
        out = []
        with self.torch.no_grad():
            for a in range(0, len(batch_u8), chunk):
                x = self.to_tensor(batch_u8[a:a + chunk])
                out.append(self.forward_t(x, autocast=True).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------------------------------------- imaging
def skin_reference(u8: np.ndarray, ring: int = 6) -> np.ndarray:
    """Median colour of the crop's border ring, a stand-in for the surrounding skin (used as the 'lesion removed' reference)."""
    h, w, _ = u8.shape
    m = np.ones((h, w), bool)
    m[ring:h - ring, ring:w - ring] = False
    return np.median(u8[m].reshape(-1, 3), axis=0)


def resize_float(m: np.ndarray, size: int, resample=Image.BILINEAR) -> np.ndarray:
    return np.asarray(Image.fromarray(np.asarray(m, dtype=np.float32), mode='F').resize((size, size), resample), dtype=np.float32)


def _base(u8: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(u8).resize((size, size), Image.BICUBIC), dtype=np.float32)


def overlay_sequential(u8: np.ndarray, heat: np.ndarray, size: int = 384, alpha_max: float = 0.7, cmap: str = 'inferno') -> np.ndarray:
    h = np.clip(resize_float(heat, size), 0, 1)
    rgb = colormaps[cmap](h)[..., :3] * 255.0
    a = (np.power(h, 0.8) * alpha_max)[..., None]
    return np.clip(_base(u8, size) * (1 - a) + rgb * a, 0, 255).astype(np.uint8)


def normalise_signed(m: np.ndarray, pct: float = 99.0) -> np.ndarray:
    s = float(np.percentile(np.abs(m), pct))
    if s <= 1e-12:
        return np.zeros_like(m, dtype=np.float32)
    return np.clip(m / s, -1, 1).astype(np.float32)


def overlay_diverging(u8: np.ndarray, signed01: np.ndarray, size: int = 384, alpha_max: float = 0.72, cmap: str = 'RdBu_r') -> np.ndarray:
    s = np.clip(resize_float(signed01, size), -1, 1)
    rgb = colormaps[cmap](0.5 + 0.5 * s)[..., :3] * 255.0
    a = (np.power(np.abs(s), 0.9) * alpha_max)[..., None]
    return np.clip(_base(u8, size) * (1 - a) + rgb * a, 0, 255).astype(np.uint8)


def draw_boundaries(img: np.ndarray, labels: np.ndarray, color=(255, 255, 255), alpha: float = 0.55) -> np.ndarray:
    from skimage.segmentation import find_boundaries
    size = img.shape[0]
    lab = np.asarray(Image.fromarray(labels.astype(np.int32), mode='I').resize((size, size), Image.NEAREST))
    edge = find_boundaries(lab, mode='inner')
    out = img.astype(np.float32).copy()
    out[edge] = out[edge] * (1 - alpha) + np.array(color, np.float32) * alpha
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------- summaries
_DIRS = ['right', 'upper right', 'upper', 'upper left', 'left', 'lower left', 'lower', 'lower right']


def direction_of(dx: float, dy: float, tol: float = 0.1) -> str:
    if math.hypot(dx, dy) < tol:
        return 'centre'
    ang = math.degrees(math.atan2(-dy, dx)) % 360
    return _DIRS[int(((ang + 22.5) % 360) // 45)]


def region_summary(pos: np.ndarray) -> dict:
    """pos: non-negative evidence map (H, W) pointing towards higher risk."""
    pos = np.clip(np.asarray(pos, dtype=np.float64), 0, None)
    h, w = pos.shape
    tot = float(pos.sum())
    if tot <= 1e-12:
        return {'focus': 'none', 'direction': 'centre', 'center_share': 0.0, 'edge_share': 0.0, 'concentration': None}
    yy, xx = np.mgrid[0:h, 0:w]
    nx, ny = (xx - (w - 1) / 2) / (w / 2), (yy - (h - 1) / 2) / (h / 2)
    r = np.hypot(nx, ny)
    center_share = float(pos[r < 0.5].sum() / tot)
    edge_share = float(pos[r > 0.85].sum() / tot)
    cx, cy = float((pos * nx).sum() / tot), float((pos * ny).sum() / tot)
    srt = np.sort(pos.ravel())[::-1]
    k80 = int(np.searchsorted(np.cumsum(srt), 0.8 * tot)) + 1
    area_center = float((r < 0.5).mean())
    focus = 'central' if center_share >= 0.45 else ('peripheral' if edge_share >= 0.35 else 'intermediate')
    return {'focus': focus, 'direction': direction_of(cx, cy), 'center_share': round(center_share, 3), 'edge_share': round(edge_share, 3),
            'center_vs_area': round(center_share / max(area_center, 1e-9), 2), 'concentration': round(k80 / pos.size, 3)}


def downsample(m: np.ndarray, n: int = 32) -> list:
    d = np.asarray(Image.fromarray(np.asarray(m, dtype=np.float32), mode='F').resize((n, n), Image.BOX), dtype=np.float32)
    return np.round(d, 4).tolist()
