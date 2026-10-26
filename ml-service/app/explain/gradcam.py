"""Grad-CAM (primary) and Score-CAM (gradient-free fallback / option) on the last feature stage of every explanation network.

Why Grad-CAM first: all four backbones expose a well-defined last spatial stage (ConvNeXt / EdgeNeXt stages, Swin's last
block group, EVA02's last transformer block whose patch tokens form a 16x16 grid), so one method covers the whole ensemble. Score-CAM needs no
gradients (more stable for the transformer backbones when gradients are noisy) but costs k extra forward passes per network; it is used
when XAI_CAM_METHOD=scorecam, or automatically when a Grad-CAM map comes out empty.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from .common import Scorer, downsample, overlay_sequential, region_summary, resize_float, skin_reference


def _to_nchw(a):
    """Activations -> (B, C, h, w). Handles NCHW (CNN stages), NHWC (torchvision Swin) and token sequences (ViT blocks)."""
    if a.dim() == 3:
        n = a.shape[1]
        g = math.isqrt(n)
        a = a[:, n - g * g:, :]                 # drop the class / prefix tokens
        return a.reshape(a.shape[0], g, g, a.shape[2]).permute(0, 3, 1, 2)
    if a.dim() == 4:
        _, d1, d2, d3 = a.shape
        if d2 == d3:                              # (B, C, h, w): square spatial grid in the last two dims
            return a
        if d1 == d2:                              # (B, h, w, C): channels last
            return a.permute(0, 3, 1, 2)
        return a if d1 >= max(d2, d3) else a.permute(0, 3, 1, 2)
    raise ValueError(f'unexpected activation shape {tuple(a.shape)}')


def target_layer(net):
    b = net.module.backbone
    for attr in ('stages', 'blocks', 'layer4', 'features'):
        if hasattr(b, attr):
            seq = getattr(b, attr)
            try:
                return seq[-1]
            except TypeError:
                return seq
    raise RuntimeError(f'no known feature stage for backbone {net.backbone}')


def _capture(layer):
    owner = threading.get_ident()
    store: dict = {}

    def hook(_m, _i, out):
        if threading.get_ident() == owner:     # other threads' forward passes through the same module are ignored
            store['a'] = out
    return store, layer.register_forward_hook(hook)


def gradcam_one(sc: Scorer, net, x01):
    torch = sc.torch
    layer = target_layer(net)
    store, h = _capture(layer)
    try:
        with torch.enable_grad():
            x = sc.prep(net, x01.detach()).detach().requires_grad_(True)
            out = net.module(x)
            act = store['a']
            grad, = torch.autograd.grad(out[0, 0], act)
    finally:
        h.remove()
    a, g = _to_nchw(act.detach()), _to_nchw(grad)
    w = g.mean(dim=(2, 3), keepdim=True)
    return torch.relu((w * a).sum(1, keepdim=True))[0, 0].float().cpu().numpy()


def scorecam_one(sc: Scorer, net, x01, base01, k: int = 24):
    torch = sc.torch
    import torch.nn.functional as F
    layer = target_layer(net)
    store, h = _capture(layer)
    try:
        with torch.no_grad():
            x = sc.prep(net, x01)
            net.module(x)
            act = store['a']
    finally:
        h.remove()
    a = _to_nchw(act.detach())[0]
    idx = a.flatten(1).var(1).topk(min(k, a.shape[0])).indices
    maps = F.interpolate(a[idx][None], size=x01.shape[-2:], mode='bilinear', align_corners=False)[0]
    mn, mx = maps.flatten(1).min(1).values.view(-1, 1, 1), maps.flatten(1).max(1).values.view(-1, 1, 1)
    maps = (maps - mn) / (mx - mn + 1e-8)
    masked = base01 * (1 - maps[:, None]) + x01 * maps[:, None]
    with torch.no_grad():
        s = []
        for i in range(0, len(masked), 16):
            xm = sc.prep(net, masked[i:i + 16])
            s.append(net.module(xm)[:, 0].float())
        s = torch.cat(s)
        s0 = net.module(sc.prep(net, base01))[:, 0].float()
        cic = s - s0
        spread = cic.std() if cic.numel() > 1 else cic.new_tensor(1.0)
        wts = torch.softmax((cic - cic.mean()) / (spread + 1e-6), 0)
    return torch.relu((wts.view(-1, 1, 1) * maps).sum(0)).float().cpu().numpy()


def _valid(cam) -> bool:
    return cam is not None and bool(np.isfinite(cam).all()) and float(cam.max()) > 1e-12


def explain_cam(ens, u8, method: str = 'gradcam') -> dict:
    t0 = time.time()
    sc = Scorer(ens)
    x01 = sc.to_tensor(u8)
    skin = skin_reference(u8)
    base01 = sc.torch.from_numpy(np.broadcast_to(skin / 255.0, u8.shape).copy()).to(sc.dev).permute(2, 0, 1).float().unsqueeze(0)
    S = u8.shape[0]
    maps, used, fallbacks = [], [], 0
    for net in sc.nets:
        cam, how = None, None
        if method == 'gradcam':
            try:
                cam, how = gradcam_one(sc, net, x01), 'gradcam'
            except Exception:  # noqa: BLE001
                cam = None
            if not _valid(cam):
                cam, how = scorecam_one(sc, net, x01, base01), 'scorecam'
                fallbacks += 1
        else:
            cam, how = scorecam_one(sc, net, x01, base01), 'scorecam'
        if _valid(cam):
            up = resize_float(cam, S)
            maps.append(up / (up.max() + 1e-12))
            used.append(how)
    agree = None
    if maps:
        heat = np.mean(maps, 0)
        heat = (heat / (heat.max() + 1e-12)).astype(np.float32)
        if len(maps) > 1:
            c = np.corrcoef(np.stack([m.ravel() for m in maps]))
            agree = float(c[np.triu_indices(len(maps), 1)].mean())
    else:
        heat = np.zeros((S, S), np.float32)
    summ = region_summary(heat)
    summ.update({'n_networks': len(sc.nets), 'network_agreement': None if agree is None else round(agree, 3),
                 'gradcam_networks': used.count('gradcam'), 'scorecam_networks': used.count('scorecam'), 'scorecam_fallbacks': fallbacks})
    subset_z = float(sc.score_u8(u8[None])[0])
    cam_method = 'gradcam' if used.count('gradcam') >= used.count('scorecam') and used else 'scorecam'
    return {
        'method': 'gradcam', 'title': 'Grad-CAM' if cam_method == 'gradcam' else 'Score-CAM', 'signed': False, 'cam_method': cam_method,
        'overlay': overlay_sequential(u8, heat), 'map': heat, 'map32': downsample(heat),
        'summary': summ, 'explained_score': {'subset_z': round(subset_z, 3), 'n_networks': len(sc.nets)},
        'params': {'layers': {f'{n.backbone}_f{n.fold}': type(target_layer(n)).__name__ for n in sc.nets}},
        'runtime_s': round(time.time() - t0, 2),
    }
