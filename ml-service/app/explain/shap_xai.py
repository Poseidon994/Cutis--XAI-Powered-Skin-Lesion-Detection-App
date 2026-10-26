"""SHAP via expected gradients (shap.GradientExplainer): pixel-level Shapley-style attributions for the ensemble score.

Reference ('background') images are plain skin: the crop's border colour plus faint noise. The attributions therefore explain
the shift from "nothing but skin" to this lesion's score, and (approximately) sum to that shift (completeness), which the
response reports so the numbers can be checked. If the shap package is unavailable the same algorithm is run directly.
"""
from __future__ import annotations

import time

import numpy as np

from .common import Scorer, downsample, normalise_signed, overlay_diverging, region_summary, skin_reference


class _Wrap:
    pass


def _backgrounds(sc: Scorer, skin: np.ndarray, S: int, n: int = 8, seed: int = 0):
    torch = sc.torch
    g = torch.Generator().manual_seed(seed)
    base = torch.from_numpy((skin / 255.0).astype(np.float32)).view(1, 3, 1, 1).expand(n, 3, S, S)
    noise = torch.randn((n, 3, S, S), generator=g) * 0.02
    return (base + noise).clamp(0, 1).to(sc.dev)


def _expected_gradients(sc: Scorer, x, bg, n_samples: int, bs: int, seed: int = 0):
    torch = sc.torch
    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros_like(x)
    done = 0
    while done < n_samples:
        m = min(bs, n_samples - done)
        b = bg[torch.randint(len(bg), (m,), generator=g).to(sc.dev)]
        a = torch.rand((m, 1, 1, 1), generator=g).to(sc.dev)
        xi = (b + a * (x - b)).detach().requires_grad_(True)
        with torch.enable_grad():
            out = sc.forward_t(xi).sum()
            grad, = torch.autograd.grad(out, xi)
        acc += ((x - b) * grad).sum(0, keepdim=True)
        done += m
    return acc / n_samples


def explain_shap(ens, u8, n_samples: int, batch_size: int = 4, seed: int = 0, smooth_sigma: float = 2.0) -> dict:
    import torch.nn as nn
    from scipy.ndimage import gaussian_filter
    t0 = time.time()
    sc = Scorer(ens)
    torch = sc.torch
    S = u8.shape[0]
    skin = skin_reference(u8)
    x = sc.to_tensor(u8)
    bg = _backgrounds(sc, skin, S, seed=seed)
    engine = 'shap.GradientExplainer'
    attr = None
    try:
        import shap

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.mods = nn.ModuleList([n.module for n in sc.nets])

            def forward(self, inp):
                return sc.forward_t(inp).unsqueeze(1)

        model = Model().eval()
        expl = shap.GradientExplainer(model, bg, batch_size=batch_size)
        sv = expl.shap_values(x, nsamples=int(n_samples), rseed=seed)
        arr = np.asarray(sv[0] if isinstance(sv, (list, tuple)) else sv, dtype=np.float64)
        arr = np.squeeze(arr)
        if arr.shape == (S, S, 3):
            arr = arr.transpose(2, 0, 1)
        if arr.shape != (3, S, S):
            raise ValueError(f'unexpected SHAP output shape {arr.shape}')
        attr = arr
    except Exception:  # noqa: BLE001
        attr = None
    if attr is None or not np.isfinite(attr).all():
        engine = 'expected gradients (direct implementation)'
        attr = _expected_gradients(sc, x, bg, int(n_samples), batch_size, seed)[0].detach().cpu().numpy().astype(np.float64)
    with torch.no_grad():
        out_z = float(sc.forward_t(x).item())
        base_z = float(sc.forward_t(bg).mean().item())
    raw_map = attr.sum(0)
    total = float(raw_map.sum())
    smooth = gaussian_filter(raw_map, smooth_sigma) if smooth_sigma > 0 else raw_map
    signed = normalise_signed(smooth)
    pos = np.clip(smooth, 0, None)
    absum = float(np.abs(raw_map).sum()) or 1.0
    summ = region_summary(pos)
    summ.update({'n_samples': int(n_samples), 'n_networks': len(sc.nets), 'engine': engine,
                 'positive_share': round(float(np.clip(raw_map, 0, None).sum() / absum), 3),
                 'output_z': round(out_z, 3), 'reference_z': round(base_z, 3), 'attribution_sum': round(total, 3),
                 'completeness_gap': round(out_z - base_z - total, 3)})
    return {
        'method': 'shap', 'title': 'SHAP', 'signed': True, 'overlay': overlay_diverging(u8, signed), 'map': smooth,
        'map32': downsample(pos / (pos.max() + 1e-12) if pos.max() > 0 else pos),
        'summary': summ, 'explained_score': {'subset_z': round(out_z, 3), 'n_networks': len(sc.nets)},
        'params': {'engine': engine, 'background': 'surrounding-skin colour + noise (8 images)', 'display_smoothing_sigma_px': smooth_sigma},
        'runtime_s': round(time.time() - t0, 2),
    }
