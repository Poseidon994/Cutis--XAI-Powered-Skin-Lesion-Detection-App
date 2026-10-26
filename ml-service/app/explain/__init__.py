"""Three parallel explanation methods for the image ensemble: Grad-CAM (or Score-CAM), LIME and SHAP."""
from __future__ import annotations

METHODS = ('gradcam', 'lime', 'shap')


def default_budget(device_type: str) -> dict:
    if device_type == 'cuda':
        return {'lime_samples': 500, 'shap_samples': 64, 'shap_batch': 8}
    return {'lime_samples': 150, 'shap_samples': 24, 'shap_batch': 4}


def run(method: str, ens, u8, settings) -> dict:
    budget = default_budget(ens.dev.type)
    if method == 'gradcam':
        from .gradcam import explain_cam
        return explain_cam(ens, u8, 'scorecam' if settings.cam_method == 'scorecam' else 'gradcam')
    if method == 'lime':
        from .lime_xai import explain_lime
        return explain_lime(ens, u8, settings.lime_samples or budget['lime_samples'])
    if method == 'shap':
        from .shap_xai import explain_shap
        return explain_shap(ens, u8, settings.shap_samples or budget['shap_samples'], batch_size=budget['shap_batch'])
    raise ValueError(f'unknown method {method}')
