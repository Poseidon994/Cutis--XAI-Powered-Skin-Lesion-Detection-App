"""Cutis ML service: image-ensemble inference + three explanation methods behind a small FastAPI app."""
from __future__ import annotations

import hmac
import math
from contextlib import asynccontextmanager
import threading
import time
import traceback

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import explain as xai
from .artifacts import ArtifactError, describe_lib_difference, load_artifacts
from .calibration import ImageCalibration
from .config import get_settings
from .imaging import ImageError, png_b64, prepare

LIMITATIONS = [
    'Trained on ISIC 2024 (SLICE-3D): 15 x 15 mm lesion crops cut from 3D total-body photography of about 1,000 patients at nine centres. '
    'Phone photos and dermoscopy images look different, so accuracy on them is unknown and probably lower.',
    'The web app uses the image branch only. The full pipeline also uses 3D-TBP measurements and comparisons with the patient\'s other lesions, '
    'which a single photo does not provide.',
    'Only 393 of 401,059 training lesions were cancers, so the statistics for rare types (especially squamous cell carcinoma) are uncertain.',
    'A low score never rules out skin cancer, and a high score is not a diagnosis. Histopathology is the reference standard.',
    'Research prototype, not a medical device.',
]


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


class State:
    def __init__(self):
        self.status = 'loading'
        self.error: str | None = None
        self.art = None
        self.ens = None
        self.calib: ImageCalibration | None = None
        self.notes: list[str] = []
        self.started = time.time()
        self.ready_at: float | None = None

    def log(self, msg: str):
        print(f'[ml] {msg}', flush=True)


state = State()
_predict_sem = threading.BoundedSemaphore(4)
_explain_sem = threading.BoundedSemaphore(3)


def load_everything():
    cfg = get_settings()
    try:
        state.log(f'loading artifacts from {cfg.artifacts_dir}')
        art = load_artifacts(cfg.artifacts_dir)
        state.art = art
        diff = describe_lib_difference(art.root)
        if diff:
            state.notes.append(diff)
        state.log(f'artifacts: {art.root} | v{art.meta["version"]} created {art.meta.get("created")} profile {art.meta.get("profile")} | library from {art.lib_source}')
        if art.meta.get('profile') != 'full':
            state.notes.append(f'artifacts come from a "{art.meta.get("profile")}" training run, not a full one; scores are not meaningful')
        from .ensemble import Ensemble
        ens = Ensemble(art, cfg, log=state.log)
        ens.load()
        state.ens = ens
        oof = art.optional('oof_predictions.csv.gz')
        if oof is not None:
            try:
                state.calib = ImageCalibration.from_oof(oof, art.lib)
                state.log(f'image-only calibration from {oof.name}: {state.calib.thresholds}')
            except Exception as e:  # noqa: BLE001
                state.notes.append(f'could not calibrate from {oof.name} ({e}); using fallback thresholds')
        if state.calib is None:
            state.calib = ImageCalibration.fallback()
            state.notes.append('oof_predictions.csv.gz missing: risk tiers use fallback thresholds')
        state.status, state.ready_at = 'ready', time.time()
        state.log('ready')
    except ArtifactError as e:
        state.status, state.error = 'error', str(e)
        state.log(f'ARTIFACT ERROR: {e}')
    except Exception as e:  # noqa: BLE001
        state.status, state.error = 'error', f'{type(e).__name__}: {e}'
        state.log('LOAD FAILED\n' + traceback.format_exc())


@asynccontextmanager
async def lifespan(_app):
    if get_settings().preload:
        threading.Thread(target=load_everything, daemon=True).start()
    yield


app = FastAPI(title='Cutis ML service', version='1.0.0', lifespan=lifespan)


def require_key(x_service_key: str | None = Header(default=None)):
    key = get_settings().service_key
    if key and not hmac.compare_digest(x_service_key or '', key):
        raise HTTPException(status_code=401, detail='invalid service key')


def require_ready():
    if state.status != 'ready':
        raise HTTPException(status_code=503, detail=state.error or 'model is still loading')


@app.get('/health')
def health():
    ens = state.ens
    return {'status': state.status, 'ready': state.status == 'ready', 'error': state.error, 'notes': state.notes,
            'device': str(ens.dev) if ens else None, 'networks': len(ens.nets) if ens else 0,
            'backbones': ens.backbones if ens else [], 'uptime_s': round(time.time() - state.started, 1),
            'calibration': state.calib.source if state.calib else None,
            'artifact_version': state.art.meta.get('created') if state.art else None}


@app.get('/model-card', dependencies=[Depends(require_key), Depends(require_ready)])
def model_card():
    art, ens = state.art, state.ens
    m = art.meta
    res = m.get('results', {})
    return clean({
        'name': 'Cutis image ensemble', 'version': m.get('version'), 'lib_version': m.get('lib_version'), 'created': m.get('created'), 'profile': m.get('profile'),
        'data': m.get('data'), 'external_data': m.get('external_data'),
        'backbones': {n: {'resolution': b.get('res'), 'source': b.get('src'), 'id': b.get('id')} for n, b in art.image['backbones'].items()},
        'folds_trained': art.image['folds'], 'networks_loaded': len(ens.nets), 'tta_views': ens.n_views, 'device': str(ens.dev),
        'image_only_calibration': state.calib.summary(),
        'results': {'image_only': res.get('image_only'), 'threshold_free': res.get('threshold_free')},
        'versions': m.get('versions'), 'notes': state.notes, 'limitations': LIMITATIONS,
    })


def _read(file: UploadFile) -> bytes:
    cfg = get_settings()
    data = file.file.read(cfg.max_upload_bytes + 1)
    if len(data) > cfg.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f'image larger than {cfg.max_upload_mb:.0f} MB')
    return data


def _prepare(file: UploadFile):
    data = _read(file)
    try:
        return prepare(data, state.ens.store_res)
    except ImageError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/predict', dependencies=[Depends(require_key), Depends(require_ready)])
def predict(file: UploadFile = File(...)):
    t0 = time.time()
    u8, info = _prepare(file)
    with _predict_sem:
        r = state.ens.predict(u8)
    calib = state.calib
    desc = calib.describe(r['score'])
    probs = r['type_probs']
    top = max(probs, key=probs.get)
    bb = []
    for b in r['per_backbone']:
        tier = calib.backbone_tier(b['name'], b['z'])
        bb.append({**b, 'tier': tier})
    tiers = [b['tier'] for b in bb if b['tier']]
    agreement = None
    if tiers:
        agreement = {'backbones': len(tiers), 'same_tier_as_ensemble': sum(t == desc['tier'] for t in tiers),
                     'flagging_elevated_or_high': sum(t in ('elevated', 'high') for t in tiers)}
    warnings = list(info['warnings'])
    if state.notes:
        warnings += [{'code': 'service_note', 'text': n} for n in state.notes if 'fallback' in n or 'run, not a full' in n]
    return clean({
        'ok': True,
        'prediction': {**desc, 'suspicious': r['suspicious'], 'type_probs': probs, 'top_type': top, 'per_backbone': bb,
                       'network_sd': r['network_sd'], 'n_networks': r['n_networks'], 'tta_views': r['views'], 'agreement': agreement},
        'quality': {**{k: v for k, v in info.items() if k != 'warnings'}, 'warnings': warnings},
        'preview_png_b64': png_b64(u8, 256),
        'model': {'version': state.art.meta.get('version'), 'created': state.art.meta.get('created'), 'device': str(state.ens.dev),
                  'backbones': state.ens.backbones, 'networks': len(state.ens.nets)},
        'runtime_s': round(time.time() - t0, 3),
    })


@app.post('/explain/{method}', dependencies=[Depends(require_key), Depends(require_ready)])
def explain(method: str, file: UploadFile = File(...)):
    if method not in xai.METHODS:
        raise HTTPException(status_code=404, detail=f'unknown method {method}; use one of {", ".join(xai.METHODS)}')
    u8, _ = _prepare(file)
    try:
        with _explain_sem:
            out = xai.run(method, state.ens, u8, get_settings())
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        state.log(f'explain {method} failed\n' + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f'{method} failed: {type(e).__name__}: {e}')
    overlay = out.pop('overlay')
    out.pop('map', None)
    return JSONResponse(clean({'ok': True, **out, 'overlay_png_b64': png_b64(overlay)}))
