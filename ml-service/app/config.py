from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path
    service_key: str
    device: str
    max_folds: int
    backbones: tuple
    tta_views: int
    xai_folds: int
    lime_samples: int
    shap_samples: int
    cam_method: str
    max_upload_bytes: int
    torch_threads: int
    preload: bool

    @property
    def max_upload_mb(self) -> float:
        return self.max_upload_bytes / 1024 / 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    bbs = tuple(b.strip() for b in os.environ.get('ML_BACKBONES', '').split(',') if b.strip())
    return Settings(
        artifacts_dir=Path(os.environ.get('ARTIFACTS_DIR', '/artifacts')),
        service_key=os.environ.get('ML_SERVICE_KEY', ''),
        device=os.environ.get('ML_DEVICE', 'auto').lower(),
        max_folds=_int('ML_MAX_FOLDS', 0),
        backbones=bbs,
        tta_views=_int('ML_TTA_VIEWS', 0),
        xai_folds=max(1, _int('XAI_FOLDS', 1)),
        lime_samples=_int('XAI_LIME_SAMPLES', 0),
        shap_samples=_int('XAI_SHAP_SAMPLES', 0),
        cam_method=os.environ.get('XAI_CAM_METHOD', 'gradcam').lower(),
        max_upload_bytes=_int('MAX_UPLOAD_MB', 10) * 1024 * 1024,
        torch_threads=_int('TORCH_THREADS', 0),
        preload=os.environ.get('ML_PRELOAD', '1') != '0',
    )
