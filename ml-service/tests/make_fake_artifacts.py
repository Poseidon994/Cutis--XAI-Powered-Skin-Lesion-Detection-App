"""Builds a miniature stand-in for the training notebook's artifacts/ folder: the real network architectures with random
weights, a meta.json in the v4 schema and a synthetic out-of-fold file. For wiring tests only: the scores mean nothing.

    python -m tests.make_fake_artifacts /tmp/fake_artifacts --folds 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parents[1] / 'app' / 'vendor' / 'isic_v4_lib.py'


def _lib():
    spec = importlib.util.spec_from_file_location('isic_v4_lib', LIB)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['isic_v4_lib'] = mod
    spec.loader.exec_module(mod)
    return mod


def build(out: Path, folds: int = 1, backbones=('eva02_small', 'edgenext_base', 'convnext_tiny', 'swin_t'), seed: int = 0) -> Path:
    import torch
    L = _lib()
    out = Path(out)
    (out / 'backbones').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_aux = len(L.AUX_COLS)
    bbs, weights, stats = {}, {}, {}
    for n in backbones:
        spec = dict(L.BACKBONES[n])
        bbs[n] = spec
        weights[n] = []
        stats[n] = {}
        for k in range(folds):
            net = L.make_net(spec, False, int(spec['res']), n_aux)
            with torch.no_grad():
                net.head.weight.mul_(8.0)           # random heads are tiny; widen the logit range a little
            rel = f'backbones/{n}_fold{k}.pth'
            L.save_half(net, out / rel)
            weights[n].append(rel)
            for j in (0, 4):
                stats[n][f'f{k}_h{j}'] = [float(rng.normal(0, 0.1)), float(rng.uniform(0.5, 1.5))]
    meta = dict(
        version='v4', lib_version=L.LIB_VERSION, created=time.strftime('%Y-%m-%d %H:%M:%S'), profile='full', cfg={},
        data=dict(n_lesions=401059, n_patients=1042, n_malignant=393, labels={'Benign': 400666, 'Melanoma': 157, 'BCC': 200, 'SCC': 36}),
        external_data={'used': False},
        features=dict(cat_levels={}, feats_tab=[], feats_fusion=[], img_cols=[], all_cats=[], noise_sd=0.0, aux_stats={}),
        image=dict(backbones=bbs, folds=folds, store_res=128, views=2, bs_infer=32, n_aux=n_aux, weights=weights, oof_standardisation_stats=stats,
                   standardisation='fake'),
        plan=dict(bench={}), calibration=dict(platt=[1.0, -7.0], nested={}), thresholds={},
        results=dict(threshold_free=[{'model': 'fusion', 'AUROC': 0.96}], image_only=[{'model': 'image only: mean of backbones', 'AUROC': 0.93}]),
        versions=dict(python=sys.version.split()[0], torch=torch.__version__),
    )
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    (out / 'isic_v4_lib.py').write_text(LIB.read_text())
    n = 30000
    y = (rng.random(n) < 0.012).astype(int)
    lab = np.where(y == 0, 'Benign', rng.choice(['Melanoma', 'BCC', 'SCC'], n, p=[0.4, 0.5, 0.1]))
    df = pd.DataFrame({'isic_id': [f'ISIC_{i:07d}' for i in range(n)], 'patient_id': 'IP0', 'target': y, 'label': lab})
    base = np.where(y == 1, rng.normal(2.4, 1.0, n), rng.normal(-0.03, 0.9, n))
    for b in backbones:
        df[f's_img_{b}'] = base + rng.normal(0, 0.5, n)
    df['s_img_mean'] = np.mean([df[f's_img_{b}'] for b in backbones], axis=0)
    df.to_csv(out / 'oof_predictions.csv.gz', index=False)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--folds', type=int, default=1)
    a = ap.parse_args()
    print(build(Path(a.out), a.folds))
