"""Locate, validate and load the artifacts written by isic2024_v4_train.ipynb.

Expected layout (the training notebook's `artifacts/` folder):
    meta.json                     run description, image-network config, standardisation stats, thresholds
    isic_v4_lib.py                the exact library the networks were trained with
    backbones/<name>_fold<k>.pth  fine-tuned weights (half precision state dicts)
    oof_predictions.csv.gz        out-of-fold scores (used to calibrate the image-only score)
    tabular_gbdt.joblib, fusion_gbdt.joblib, multiclass.joblib   optional, only for the TBP batch script
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

VENDORED_LIB = Path(__file__).parent / 'vendor' / 'isic_v4_lib.py'
OLD_LAYOUT_FILES = ('frozen_stack.joblib', 'stage2.joblib', 'type_cascade.joblib')


class ArtifactError(RuntimeError):
    pass


@dataclass
class Artifacts:
    root: Path
    meta: dict
    lib: ModuleType
    lib_source: str

    @property
    def image(self) -> dict:
        return self.meta['image']

    @property
    def stats(self) -> dict:
        return self.image['oof_standardisation_stats']

    def weight_path(self, backbone: str, fold: int) -> Path:
        rels = self.image['weights'][backbone]
        return self.root / rels[fold]

    def optional(self, name: str) -> Path | None:
        p = self.root / name
        return p if p.exists() else None


def looks_old(root: Path) -> bool:
    """Files only an earlier pipeline wrote (and no fusion model from the final v4 notebook)."""
    return any((root / f).exists() for f in OLD_LAYOUT_FILES) and not (root / 'fusion_gbdt.joblib').exists()


OLD_MESSAGE = (
    'The artifacts folder holds the output of an earlier training run (it contains frozen_stack / stage2 / type_cascade files), not of the final '
    'isic2024_v4_train.ipynb. This service needs that notebook\'s output: meta.json (version "v4") with an image section, isic_v4_lib.py, '
    'fusion_gbdt.joblib and backbones/*.pth. Re-run isic2024_v4_train.ipynb (profile "full") with weight saving on, then point ARTIFACTS_DIR at its artifacts/ folder.'
)


def _candidates(base: Path) -> list[Path]:
    out = []
    if (base / 'meta.json').exists():
        out.append(base)
    for pat in ('artifacts/meta.json', '*/meta.json', '*/artifacts/meta.json', '*/*/artifacts/meta.json'):
        out += [p.parent for p in base.glob(pat)]
    seen, uniq = set(), []
    for c in out:
        if c.resolve() not in seen:
            seen.add(c.resolve())
            uniq.append(c)
    return uniq


def locate(base: Path) -> Path:
    base = Path(base)
    if not base.exists():
        raise ArtifactError(f'Artifacts folder {base} does not exist. Copy the training notebook output there (see README).')
    good, others = [], []
    for c in _candidates(base):
        try:
            m = json.loads((c / 'meta.json').read_text())
        except Exception:
            continue
        (good if m.get('version') == 'v4' and not looks_old(c) and 'image' in m else others).append((m.get('created', ''), c))
    if good:
        return sorted(good)[-1][1]
    old = [p for p in base.rglob('*') if p.name in OLD_LAYOUT_FILES][:1]
    if old or others:
        raise ArtifactError(f'Found artifacts under {base}, but not in the v4 layout. {OLD_MESSAGE}')
    raise ArtifactError(f'No meta.json found under {base}. Copy the artifacts/ folder written by isic2024_v4_train.ipynb there.')


def _load_module(path: Path, name: str = 'isic_v4_lib') -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def load_library(root: Path) -> tuple[ModuleType, str]:
    art_lib = root / 'isic_v4_lib.py'
    if art_lib.exists():
        return _load_module(art_lib), f'artifacts ({art_lib})'
    if VENDORED_LIB.exists():
        return _load_module(VENDORED_LIB), 'vendored copy (artifacts/isic_v4_lib.py not found)'
    raise ArtifactError('isic_v4_lib.py not found in the artifacts and no vendored copy is available.')


def load_artifacts(base: Path) -> Artifacts:
    root = locate(base)
    meta = json.loads((root / 'meta.json').read_text())
    for key in ('image', 'features', 'calibration'):
        if key not in meta:
            raise ArtifactError(f'meta.json is missing "{key}"; it was not written by the v4 training notebook.')
    img = meta['image']
    for key in ('backbones', 'weights', 'oof_standardisation_stats', 'store_res', 'folds'):
        if key not in img:
            raise ArtifactError(f'meta.json image section is missing "{key}".')
    if not img['backbones']:
        raise ArtifactError('The training run finished without any image backbone (meta.image.backbones is empty).')
    no_weights = [n for n in img['backbones'] if not img['weights'].get(n)]
    if no_weights:
        raise ArtifactError(f'meta.json lists no fine-tuned weights for {", ".join(no_weights)}. Weight saving must be on in the training run (CFG.save_weights), '
                            'and the backbones/ folder must be part of the artifacts you copy.')
    lib, src = load_library(root)
    return Artifacts(root=root, meta=meta, lib=lib, lib_source=src)


def describe_lib_difference(root: Path) -> str | None:
    art_lib = root / 'isic_v4_lib.py'
    if art_lib.exists() and VENDORED_LIB.exists() and _md5(art_lib) != _md5(VENDORED_LIB):
        return 'artifacts/isic_v4_lib.py differs from the vendored copy; the artifact copy is used (it matches the trained weights).'
    return None
