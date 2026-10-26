"""LIME (superpixel, model-agnostic): which regions of the lesion crop push the ensemble score up or down.

Superpixels are SLIC cells; 'switching a cell off' paints it with the surrounding-skin colour, so each coefficient answers
"what would the score be if this part of the picture were plain skin?". The regression target is the ensemble's z-scored
malignant logit (continuous, so the local linear model is not squashed by a sigmoid).
"""
from __future__ import annotations

import time

import numpy as np

from .common import (Scorer, downsample, draw_boundaries, normalise_signed, overlay_diverging, region_summary, skin_reference,
                     direction_of)


def explain_lime(ens, u8, n_samples: int, seed: int = 0, n_segments: int = 40) -> dict:
    from lime import lime_image
    from skimage.segmentation import slic
    t0 = time.time()
    sc = Scorer(ens)
    skin = skin_reference(u8)
    S = u8.shape[0]

    def seg_fn(im):
        return slic(im, n_segments=n_segments, compactness=10, sigma=1, start_label=0, channel_axis=-1)

    def clf(batch):
        z = sc.score_u8(np.asarray(batch, dtype=np.uint8))
        return np.stack([np.zeros_like(z), z], axis=1)

    import inspect
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    kw = dict(labels=(1,), hide_color=tuple(float(v) for v in skin), top_labels=None, num_samples=int(n_samples), batch_size=32,
              segmentation_fn=seg_fn, random_seed=seed)
    if 'progress_bar' in inspect.signature(explainer.explain_instance).parameters:      # newer LIME builds only
        kw['progress_bar'] = False
    exp = explainer.explain_instance(u8, clf, **kw)
    seg = exp.segments
    weights = dict(exp.local_exp[1])
    wmap = np.zeros(seg.shape, np.float32)
    for k, w in weights.items():
        wmap[seg == k] = w
    r2 = None
    try:
        r2 = float(exp.score[1]) if isinstance(exp.score, dict) else float(exp.score)
    except Exception:  # noqa: BLE001
        pass
    signed = normalise_signed(wmap, pct=100.0)
    overlay = draw_boundaries(overlay_diverging(u8, signed), seg)
    pos = np.clip(wmap, 0, None)
    yy, xx = np.mgrid[0:S, 0:S]
    top = []
    for k, w in sorted(weights.items(), key=lambda kv: -kv[1])[:3]:
        if w <= 0:
            break
        m = seg == k
        dx, dy = (xx[m].mean() - (S - 1) / 2) / (S / 2), (yy[m].mean() - (S - 1) / 2) / (S / 2)
        top.append({'weight': round(float(w), 4), 'area_share': round(float(m.mean()), 3), 'direction': direction_of(float(dx), float(dy))})
    summ = region_summary(pos)
    summ.update({'n_segments': int(len(np.unique(seg))), 'n_samples': int(n_samples), 'local_r2': None if r2 is None else round(r2, 3),
                 'positive_weight_sum': round(float(sum(w for w in weights.values() if w > 0)), 4),
                 'negative_weight_sum': round(float(sum(w for w in weights.values() if w < 0)), 4), 'top_regions': top})
    subset_z = float(sc.score_u8(u8[None])[0])
    return {
        'method': 'lime', 'title': 'LIME', 'signed': True, 'overlay': overlay, 'map': wmap, 'map32': downsample(np.clip(wmap, 0, None) / (wmap.max() + 1e-12) if wmap.max() > 0 else wmap),
        'summary': summ, 'explained_score': {'subset_z': round(subset_z, 3), 'n_networks': len(sc.nets)},
        'params': {'segmentation': f'SLIC, about {n_segments} superpixels', 'hide_colour': [int(v) for v in skin], 'num_samples': int(n_samples)},
        'runtime_s': round(time.time() - t0, 2),
    }
