"""Check that a training run's artifacts/ folder is complete and that the service reproduces the training-time scores.

    python -m scripts.verify_artifacts --artifacts ../artifacts
    python -m scripts.verify_artifacts --artifacts ../artifacts --isic-dir /path/to/isic-2024-challenge --n 300

With --isic-dir (the competition data: train-metadata.csv + train-image.hdf5 or train-image/image/*.jpg) the script scores a sample of
training lesions through the exact code path the web app uses and compares the result with the out-of-fold scores stored in
oof_predictions.csv.gz. Each lesion was scored out-of-fold by ONE fold network per backbone, while the app averages all folds, so the two
are not identical, but their rank correlation should be high (the AUROC on the sample should also be in the region of the reported one).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.artifacts import ArtifactError, VENDORED_LIB, load_artifacts, locate  # noqa: E402

OK, WARN, FAIL = 'ok  ', 'WARN', 'FAIL'
problems = {'fail': 0, 'warn': 0}


def say(level, msg):
    if level == FAIL:
        problems['fail'] += 1
    elif level == WARN:
        problems['warn'] += 1
    print(f'[{level}] {msg}')


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', default='../artifacts')
    ap.add_argument('--isic-dir', default=None)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--device', default='auto')
    a = ap.parse_args()

    try:
        root = locate(Path(a.artifacts))
    except ArtifactError as e:
        say(FAIL, str(e))
        sys.exit(1)
    art = load_artifacts(root)
    m = art.meta
    say(OK, f'artifacts at {root} | version {m["version"]} | profile {m.get("profile")} | created {m.get("created")}')
    if m.get('profile') != 'full':
        say(WARN, 'this is not a "full" training run')
    img = m['image']
    say(OK, f'image config: {len(img["backbones"])} backbones x {img["folds"]} folds, store_res {img["store_res"]}, {img["views"]} TTA views')

    missing, size = [], 0
    for n, rels in img['weights'].items():
        for rel in rels:
            p = root / rel
            if p.exists():
                size += p.stat().st_size
            else:
                missing.append(rel)
    if missing:
        say(FAIL, f'{len(missing)} weight file(s) missing, e.g. {missing[:3]}. Copy the whole backbones/ folder.')
    else:
        say(OK, f'all {sum(len(v) for v in img["weights"].values())} weight files present ({size / 1e6:.0f} MB)')
    for name, why in (('oof_predictions.csv.gz', 'image-only calibration'), ('isic_v4_lib.py', 'exact training library'),
                      ('tabular_gbdt.joblib', 'TBP batch scoring only'), ('fusion_gbdt.joblib', 'TBP batch scoring only'), ('multiclass.joblib', 'TBP batch scoring only')):
        if (root / name).exists():
            say(OK, f'{name} present')
        else:
            say(WARN if name in ('oof_predictions.csv.gz', 'isic_v4_lib.py') else OK, f'{name} not found ({why})')
    if (root / 'isic_v4_lib.py').exists() and md5(root / 'isic_v4_lib.py') != md5(VENDORED_LIB):
        say(WARN, 'artifacts/isic_v4_lib.py differs from the copy bundled with the service; the artifact copy is used')

    for pkg, want in (m.get('versions') or {}).items():
        if pkg == 'python':
            have = sys.version.split()[0]
        else:
            try:
                have = importlib.import_module(pkg).__version__
            except Exception:  # noqa: BLE001
                have = None
        if have is None:
            say(WARN, f'{pkg} {want} was used for training and is not installed here')
        elif have.split('+')[0] != str(want).split('+')[0]:
            say(OK if pkg in ('python', 'torch') else WARN, f'{pkg}: trained with {want}, installed {have}')

    import numpy as np
    from app.calibration import ImageCalibration
    from app.config import get_settings
    from app.ensemble import Ensemble
    import os
    os.environ['ML_DEVICE'] = a.device
    get_settings.cache_clear()
    ens = Ensemble(art, get_settings())
    t0 = time.time()
    try:
        ens.load()
        say(OK, f'{len(ens.nets)} networks loaded in {time.time() - t0:.1f}s on {ens.dev}')
    except Exception as e:  # noqa: BLE001
        say(FAIL, f'could not load the networks: {type(e).__name__}: {e}')
        sys.exit(1)
    r = ens.predict(np.full((ens.store_res, ens.store_res, 3), 160, np.uint8))
    say(OK if np.isfinite(r['score']) else FAIL, f'dummy forward pass: score {r["score"]:.3f} in {r["runtime_s"]}s')

    oof_path = art.optional('oof_predictions.csv.gz')
    cal = None
    if oof_path:
        cal = ImageCalibration.from_oof(oof_path, art.lib)
        say(OK, f'image-only calibration: AUROC {cal.metrics["auroc"]:.4f}, pAUC(TPR>=80%) {cal.metrics["pauc_tpr80"]:.4f}, '
                f'thresholds {({k: round(v, 2) for k, v in cal.thresholds.items()})}')

    if a.isic_dir and oof_path:
        import pandas as pd
        from app.imaging import prepare
        d = Path(a.isic_dir)
        meta = pd.read_csv(d / 'train-metadata.csv', usecols=['isic_id', 'target'], low_memory=False)
        oof = pd.read_csv(oof_path, usecols=['isic_id', 's_img_mean', 'target'])
        df = meta.merge(oof[['isic_id', 's_img_mean']], on='isic_id')
        pos, neg = df[df.target == 1], df[df.target == 0]
        pick = pd.concat([pos.sample(min(len(pos), a.n // 4), random_state=0), neg.sample(min(len(neg), a.n - a.n // 4), random_state=0)])
        h5 = None
        if (d / 'train-image.hdf5').exists():
            import h5py
            h5 = h5py.File(d / 'train-image.hdf5', 'r')
        scores, ref, ys = [], [], []
        t0 = time.time()
        for _, row in pick.iterrows():
            raw = bytes(h5[row.isic_id][()]) if h5 is not None else (d / 'train-image' / 'image' / f'{row.isic_id}.jpg').read_bytes()
            u8, _ = prepare(raw, ens.store_res)
            scores.append(ens.predict(u8)['score'])
            ref.append(row.s_img_mean)
            ys.append(row.target)
        scores, ref, ys = np.array(scores), np.array(ref), np.array(ys)
        from scipy.stats import spearmanr
        rho = float(spearmanr(scores, ref).statistic)
        auc = float(art.lib.auroc(ys, scores))
        say(OK if rho > 0.8 else FAIL, f'{len(pick)} lesions scored in {time.time() - t0:.0f}s | Spearman with the stored out-of-fold image score: {rho:.3f} (expect > 0.8)')
        say(OK if auc > 0.85 else WARN, f'AUROC on this sample: {auc:.3f} (stored out-of-fold image-only AUROC: {cal.metrics["auroc"]:.3f}; small samples vary)')
    elif a.isic_dir:
        say(WARN, 'oof_predictions.csv.gz missing, cannot compare scores')

    print()
    print('RESULT:', 'FAILED' if problems['fail'] else ('passed with warnings' if problems['warn'] else 'all checks passed'))
    sys.exit(1 if problems['fail'] else 0)


if __name__ == '__main__':
    main()
