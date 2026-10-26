"""API-level tests against fake artifacts (real architectures, random weights): wiring, shapes, contracts, concurrency."""
import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image, ImageDraw

KEY = 'test-key'


def lesion_jpeg(size=(240, 210)) -> bytes:
    img = Image.new('RGB', size, (205, 160, 140))
    d = ImageDraw.Draw(img)
    d.ellipse((70, 55, 170, 155), fill=(95, 58, 42))
    d.ellipse((105, 85, 135, 120), fill=(45, 28, 22))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=92)
    return buf.getvalue()


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    from tests.make_fake_artifacts import build
    art = tmp_path_factory.mktemp('art')
    build(art, folds=2, backbones=('convnext_tiny', 'swin_t'))
    os.environ.update(ARTIFACTS_DIR=str(art), ML_SERVICE_KEY=KEY, XAI_LIME_SAMPLES='30', XAI_SHAP_SAMPLES='6', TQDM_DISABLE='1')
    from app.config import get_settings
    get_settings.cache_clear()
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        for _ in range(180):
            if c.get('/health').json()['status'] != 'loading':
                break
            time.sleep(1)
        assert c.get('/health').json()['ready'], c.get('/health').json()
        c.art_dir = art
        yield c


H = {'X-Service-Key': KEY}


def post(client, path, data=None, name='lesion.jpg'):
    return client.post(path, files={'file': (name, data or lesion_jpeg(), 'image/jpeg')}, headers=H)


def test_health_open_and_ready(client):
    h = client.get('/health').json()
    assert h['ready'] and h['networks'] == 4 and h['backbones'] == ['convnext_tiny', 'swin_t'] and h['calibration'] == 'oof'


def test_key_required(client):
    assert client.post('/predict', files={'file': ('a.jpg', lesion_jpeg(), 'image/jpeg')}).status_code == 401
    assert client.get('/model-card').status_code == 401


def test_predict_contract(client):
    r = post(client, '/predict')
    assert r.status_code == 200, r.text
    j = r.json()
    p = j['prediction']
    assert p['tier'] in ('low', 'guarded', 'elevated', 'high')
    assert abs(sum(p['type_probs'].values()) - 1) < 1e-6 and p['top_type'] in p['type_probs']
    assert [b['name'] for b in p['per_backbone']] == ['convnext_tiny', 'swin_t'] and all(b['n_folds'] == 2 for b in p['per_backbone'])
    assert p['n_networks'] == 4 and p['agreement']['backbones'] == 2
    assert 0 <= p['percentile_benign'] <= 100 and p['enrichment'] is not None
    png = Image.open(io.BytesIO(base64.b64decode(j['preview_png_b64'])))
    assert png.size == (256, 256)
    assert j['quality']['cropped_to_square'] is True and any(w['code'] == 'center_cropped' for w in j['quality']['warnings'])


def test_prediction_matches_library_computation(client):
    """Recompute the score from scratch with the training library's own functions and compare."""
    import app.main as m
    ens, L = m.state.ens, m.state.art.lib
    from app.imaging import prepare
    u8, _ = prepare(lesion_jpeg(), 128)
    per = {}
    for n in ens.nets:
        lg = L.predict_logits(n.module, n.img, u8[None], np.arange(1), L.VIEW_SETS[2], 32, n.cl)[0].astype(np.float64)
        z = lg.copy()
        for j in (0, 4):
            mu, sd = m.state.art.stats[n.backbone][f'f{n.fold}_h{j}']
            z[j] = (z[j] - mu) / sd
        per.setdefault(n.backbone, []).append(z)
    base = L.image_base_scores({b: np.mean(v, 0)[None, :] for b, v in per.items()})
    r = post(client, '/predict').json()['prediction']
    assert r['score'] == pytest.approx(float(base['img_mean'][0]), abs=1e-3)
    assert r['type_probs']['Melanoma'] == pytest.approx(float(np.exp(base['type_Melanoma'][0]) / sum(np.exp(base[f'type_{t}'][0]) for t in L.TYPES)), abs=1e-3)


def test_bad_inputs(client):
    assert post(client, '/predict', b'not an image').status_code == 400
    tiny = io.BytesIO()
    Image.new('RGB', (20, 20)).save(tiny, 'PNG')
    assert post(client, '/predict', tiny.getvalue()).status_code == 400
    assert post(client, '/explain/nonsense').status_code == 404


def test_png_with_alpha_and_tall_image(client):
    buf = io.BytesIO()
    Image.new('RGBA', (180, 400), (200, 150, 130, 255)).save(buf, 'PNG')
    r = client.post('/predict', files={'file': ('a.png', buf.getvalue(), 'image/png')}, headers=H)
    assert r.status_code == 200


def test_three_explanations_in_parallel(client):
    data = lesion_jpeg()

    def run(m):
        t = time.time()
        r = post(client, f'/explain/{m}', data)
        return m, r, time.time() - t

    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(run, ['gradcam', 'lime', 'shap']))
    for m, r, dt in res:
        assert r.status_code == 200, (m, r.text)
        j = r.json()
        assert j['method'] == m and j['title'] in ('Grad-CAM', 'Score-CAM', 'LIME', 'SHAP')
        ov = Image.open(io.BytesIO(base64.b64decode(j['overlay_png_b64'])))
        assert ov.size == (384, 384)
        assert len(j['map32']) == 32 and len(j['map32'][0]) == 32
        assert j['summary']['focus'] in ('central', 'peripheral', 'intermediate', 'none')
        assert j['explained_score']['n_networks'] == 2          # XAI_FOLDS=1 -> one fold per backbone
    s = {m: r.json() for m, r, _ in res}
    assert s['gradcam']['summary']['gradcam_networks'] == 2 and s['gradcam']['summary']['scorecam_fallbacks'] == 0
    assert s['lime']['summary']['n_samples'] == 30 and s['lime']['summary']['local_r2'] is not None
    assert s['shap']['summary']['engine'] in ('shap.GradientExplainer', 'expected gradients (direct implementation)')


def test_model_card(client):
    j = client.get('/model-card', headers=H).json()
    assert j['version'] == 'v4' and j['networks_loaded'] == 4 and len(j['limitations']) >= 4
    assert j['image_only_calibration']['operating_points'][0]['rule'] == 'sens95'
