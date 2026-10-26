"""ISIC 2024 (SLICE-3D) pipeline v4: shared library.

Used by the training notebook (main process and GPU worker processes) and by the submission notebook, so that
features, image preprocessing and models are computed by exactly the same code at training and at test time.

Run as a script for a GPU worker:   python isic_v4_lib.py worker --run-dir DIR --device cuda|cpu --tag gpu0
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

LIB_VERSION = 'v4.0'

# ============================================================================================================
# 1. Column groups and labels
# ============================================================================================================
BASE_NUM = ['age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext',
            'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio',
            'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
            'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
            'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt', 'tbp_lv_symm_2axis',
            'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z']
CAT_COLS = ['sex', 'anatom_site_general', 'tbp_lv_location', 'tbp_lv_location_simple']
SITE_ID_COLS = ['attribution', 'tbp_tile_type']          # institution / capture-rig identifiers: ablation only
DEMO_COLS = ['age_approx', 'sex', 'anatom_site_general', 'tbp_lv_location', 'tbp_lv_location_simple']
LABEL_ONLY_COLS = ['target', 'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',
                   'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence']   # absent from the test metadata
LABELS = ['Benign', 'Melanoma', 'BCC', 'SCC', 'OtherMalignant', 'Indeterminate']
TYPES = ['Melanoma', 'BCC', 'SCC']
# TBP measurements the image network also regresses (dense auxiliary supervision on every lesion, as in the 2nd-place solution)
AUX_COLS = ['tbp_lv_nevi_confidence', 'tbp_lv_deltaLBnorm', 'tbp_lv_color_std_mean', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
            'tbp_lv_symm_2axis', 'tbp_lv_H', 'tbp_lv_L', 'tbp_lv_A', 'tbp_lv_B', 'tbp_lv_areaMM2', 'tbp_lv_eccentricity',
            'tbp_lv_radial_color_std_max', 'tbp_lv_stdL']
N_HEADS = 5          # image-network outputs used downstream: [malignant | Melanoma, BCC, SCC | suspicious]


def derive_labels(meta: pd.DataFrame) -> dict:
    """Binary target, lesion type, and an auxiliary 'suspicious' flag from the iddx_* hierarchy (training metadata only)."""
    n = len(meta)
    y = pd.to_numeric(meta['target'], errors='coerce').fillna(0).astype(int).values
    cols = [c for c in ['iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5'] if c in meta.columns]
    if cols:
        txt = meta[cols[0]].fillna('').astype(str)
        for c in cols[1:]:
            txt = txt.str.cat(meta[c].fillna('').astype(str), sep='|')
        txt = txt.str.lower()
    else:
        txt = pd.Series([''] * n, index=meta.index)
    has = lambda pat: np.asarray(txt.str.contains(pat, regex=False), dtype=bool)
    is_mal = y == 1
    label = np.full(n, 'Benign', dtype=object)
    label[is_mal & has('melanoma')] = 'Melanoma'
    label[is_mal & has('basal cell') & (label == 'Benign')] = 'BCC'
    label[is_mal & has('squamous') & (label == 'Benign')] = 'SCC'
    label[is_mal & (label == 'Benign')] = 'OtherMalignant'
    ind = np.zeros(n, dtype=bool)
    if 'iddx_1' in meta.columns:
        ind = np.asarray(meta['iddx_1'].fillna('').astype(str).str.lower().str.startswith('indeterminate'), dtype=bool) & ~is_mal
        label[ind] = 'Indeterminate'
    # 'suspicious' = malignant, indeterminate, or carrying a specific diagnosis below the top level (biopsied / expert-reviewed).
    susp = is_mal | ind
    if 'iddx_2' in meta.columns:
        d2 = meta['iddx_2'].fillna('').astype(str).str.strip()
        susp = susp | np.asarray(d2 != '', dtype=bool)
    type_code = np.array([TYPES.index(l) if l in TYPES else -1 for l in label], dtype=np.int8)
    label_code = np.array([LABELS.index(l) for l in label], dtype=np.int8)
    return dict(y=y.astype(np.int8), label=label, label_code=label_code, type_code=type_code, susp=susp.astype(np.int8),
                indeterminate=ind)


# ============================================================================================================
# 2. Tabular (TBP) features + intra-patient context
# ============================================================================================================
KEY_CTX = ['clin_size_long_diam_mm', 'tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'tbp_lv_L', 'tbp_lv_A', 'tbp_lv_B', 'tbp_lv_C', 'tbp_lv_H',
           'tbp_lv_deltaL', 'tbp_lv_deltaLBnorm', 'tbp_lv_color_std_mean', 'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_symm_2axis',
           'tbp_lv_eccentricity', 'tbp_lv_nevi_confidence', 'tbp_lv_stdL', 'tbp_lv_radial_color_std_max', 'lesion_color_difference',
           'border_complexity', 'visibility_score', 'severity_index', 'color_contrast_index', 'lesion_shape_index']
KEY_SITE = ['tbp_lv_nevi_confidence', 'tbp_lv_L', 'tbp_lv_H', 'clin_size_long_diam_mm', 'severity_index', 'tbp_lv_deltaLBnorm',
            'tbp_lv_color_std_mean', 'tbp_lv_norm_color', 'tbp_lv_norm_border', 'lesion_color_difference']
LOF_COLS = ['tbp_lv_H', 'hue_contrast', 'age_size_norm', 'tbp_lv_deltaB', 'color_uniformity', 'tbp_lv_z', 'clin_size_long_diam_mm',
            'tbp_lv_y', 'dist_3d', 'tbp_lv_stdLExt', 'mean_hue', 'age_norm_nevi_conf', 'visibility_score', 'tbp_lv_minorAxisMM',
            'tbp_lv_Hext']
PHENO_COLS = ['clin_size_long_diam_mm', 'tbp_lv_nevi_confidence', 'tbp_lv_H', 'tbp_lv_color_std_mean']


def _sdiv(a, b, eps=1e-6):
    return a / (b + eps)


def _engineered(meta: pd.DataFrame) -> pd.DataFrame:
    g = lambda c: (pd.to_numeric(meta[c], errors='coerce').astype(np.float64) if c in meta.columns
                   else pd.Series(np.nan, index=meta.index, dtype=np.float64))
    d = pd.DataFrame({c: g(c) for c in BASE_NUM}, index=meta.index)
    area, per, size, minor, age = d['tbp_lv_areaMM2'], d['tbp_lv_perimeterMM'], d['clin_size_long_diam_mm'], d['tbp_lv_minorAxisMM'], d['age_approx']
    dA, dB, dL = d['tbp_lv_deltaA'], d['tbp_lv_deltaB'], d['tbp_lv_deltaL']
    nb, nc, sym, ecc = d['tbp_lv_norm_border'], d['tbp_lv_norm_color'], d['tbp_lv_symm_2axis'], d['tbp_lv_eccentricity']
    H, Hx, L, Lx, A, Ax, B, Bx = (d[c] for c in ['tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext'])
    cstd, rad, stdL, stdLx, dLBn = d['tbp_lv_color_std_mean'], d['tbp_lv_radial_color_std_max'], d['tbp_lv_stdL'], d['tbp_lv_stdLExt'], d['tbp_lv_deltaLBnorm']
    x, yy, z, apr = d['tbp_lv_x'], d['tbp_lv_y'], d['tbp_lv_z'], d['tbp_lv_area_perim_ratio']
    e = {}
    e['lesion_size_ratio'] = _sdiv(minor, size)
    e['lesion_shape_index'] = _sdiv(area, per ** 2)
    e['perimeter_to_area'] = _sdiv(per, area)
    e['area_to_perimeter'] = _sdiv(area, per)
    e['hue_contrast'] = (H - Hx).abs()
    e['luminance_contrast'] = (L - Lx).abs()
    e['signed_dA'] = A - Ax
    e['signed_dB'] = B - Bx
    e['signed_dC'] = d['tbp_lv_C'] - d['tbp_lv_Cext']
    e['std_L_diff'] = stdL - stdLx
    e['lesion_color_difference'] = np.sqrt(dA ** 2 + dB ** 2 + dL ** 2)
    e['border_complexity'] = nb + sym
    e['color_uniformity'] = _sdiv(cstd, rad)
    e['dist_3d'] = np.sqrt(x ** 2 + yy ** 2 + z ** 2)
    e['orientation_3d'] = np.arctan2(yy, x)
    e['visibility_score'] = dLBn + nc
    e['symmetry_border'] = sym * nb
    e['consistency_symmetry_border'] = _sdiv(sym * nb, sym + nb)
    e['color_consistency'] = _sdiv(stdL, Lx)
    e['consistency_color'] = _sdiv(stdL * Lx, stdL + Lx)
    e['size_age'] = size * age
    e['norm_size_age'] = _sdiv(size, age)
    e['severity_index'] = (nb + nc + ecc) / 3
    e['shape_complexity'] = e['border_complexity'] + e['lesion_shape_index']
    e['color_contrast_index'] = dA + dB + dL + dLBn
    e['log_area'] = np.log1p(area.clip(lower=0))
    e['mean_hue'] = (H + Hx) / 2
    e['std_dev_contrast'] = np.sqrt((dA ** 2 + dB ** 2 + dL ** 2) / 3)
    e['color_shape_composite'] = (cstd + apr + sym) / 3
    e['overall_color_difference'] = (dA + dB + dL) / 3
    e['sym_perimeter'] = sym * per
    e['comprehensive_lesion_index'] = (apr + ecc + nc + sym) / 4
    e['color_variance_ratio'] = _sdiv(cstd, stdLx)
    e['border_color'] = nb * nc
    e['border_color_2'] = _sdiv(nb * nc, nb + nc)
    e['size_color_contrast_ratio'] = _sdiv(size, dLBn)
    e['age_norm_nevi_conf'] = _sdiv(d['tbp_lv_nevi_confidence'], age)
    e['age_size_norm'] = np.sqrt(size ** 2 + age ** 2)
    e['color_asymmetry_index'] = sym * rad
    e['volume_approx_3d'] = area * e['dist_3d']
    e['color_range'] = (L - Lx).abs() + (A - Ax).abs() + (B - Bx).abs()
    e['shape_color_consistency'] = ecc * cstd
    e['border_length_ratio'] = _sdiv(per, 2 * np.pi * np.sqrt(area.clip(lower=0) / np.pi))
    e['age_size_symmetry'] = age * size * sym
    e['age_area_symmetry'] = age * area * sym
    e['hue_color_std'] = H * cstd
    return pd.concat([d, pd.DataFrame(e, index=meta.index)], axis=1)


def _lof_per_patient(Z: np.ndarray, codes: np.ndarray, k_max=30, min_n=6) -> np.ndarray:
    """Local outlier factor of every lesion among the lesions of the same patient (a learned 'ugly duckling' sign)."""
    from sklearn.neighbors import LocalOutlierFactor
    out = np.full(len(codes), np.nan, dtype=np.float32)
    order = np.argsort(codes, kind='stable')
    bounds = np.r_[0, np.cumsum(np.bincount(codes))]
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    for g in range(len(bounds) - 1):
        rows = order[bounds[g]:bounds[g + 1]]
        if len(rows) < min_n:
            continue
        try:
            lof = LocalOutlierFactor(n_neighbors=min(k_max, len(rows) - 1))
            lof.fit(Z[rows])
            out[rows] = lof.negative_outlier_factor_
        except Exception:
            pass
    return out


def build_tabular(meta: pd.DataFrame, cat_levels: dict | None = None, with_lof: bool = True):
    """All tabular features. Returns (X, groups, cat_levels). Every feature uses only the lesion itself and the other lesions of the
    SAME patient, so it is computed identically on the hidden test set (patients are never split between train and test)."""
    d = _engineered(meta)
    num_all = list(d.columns)
    pid = meta['patient_id'].astype(str).values
    codes = pd.factorize(pid)[0]
    grp = d.groupby(codes)
    mu, sd = grp.transform('mean'), grp.transform('std')
    pz = (d - mu) / (sd + 1e-6)
    pz.columns = [f'pz_{c}' for c in num_all]
    prk = d[KEY_CTX].groupby(codes).rank(pct=True)
    prk.columns = [f'prk_{c}' for c in KEY_CTX]
    site = meta['tbp_lv_location_simple'].astype(str).values if 'tbp_lv_location_simple' in meta.columns else np.zeros(len(meta), dtype=str)
    ps_codes = pd.factorize(pd.Series(pid).str.cat(pd.Series(site).astype(str), sep='|').values)[0]
    gs = d[KEY_SITE].groupby(ps_codes)
    psz = (d[KEY_SITE] - gs.transform('mean')) / (gs.transform('std') + 1e-6)
    psz.columns = [f'psz_{c}' for c in KEY_SITE]
    pm = d[PHENO_COLS].groupby(codes).transform('mean')
    pm.columns = [f'pm_{c}' for c in PHENO_COLS]
    parts = [d, pz, prk, psz, pm]
    ctx_cols = list(pz.columns) + list(prk.columns) + list(psz.columns) + list(pm.columns)
    if with_lof:
        lof = _lof_per_patient(pz[[f'pz_{c}' for c in LOF_COLS]].values.astype(np.float32), codes)
        parts.append(pd.DataFrame({'lof_patient': lof}, index=meta.index))
        ctx_cols.append('lof_patient')
    # lesion burden of the patient: ablation only (lesion counts also depend on the capture protocol / lighting of each site)
    cnt = pd.Series(codes).map(pd.Series(codes).value_counts()).values.astype(np.float64)
    area = d['tbp_lv_areaMM2']
    burden = pd.DataFrame({'n_lesions_patient': cnt,
                           'area_sum_patient': area.groupby(codes).transform('sum').values,
                           'area_sum_patient_site': area.groupby(ps_codes).transform('sum').values}, index=meta.index)
    parts.append(burden)
    X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    cats = CAT_COLS + [c for c in SITE_ID_COLS]
    cat_levels = dict(cat_levels or {})
    for c in cats:
        vals = meta[c].astype(str).fillna('NA') if c in meta.columns else pd.Series(['NA'] * len(meta), index=meta.index)
        vals = vals.where(vals.str.lower() != 'nan', 'NA')
        if c not in cat_levels:
            cat_levels[c] = sorted(vals.unique().tolist())
        X[c] = pd.Categorical(vals, categories=cat_levels[c])
    groups = dict(num=num_all, base=list(BASE_NUM), ctx=ctx_cols, burden=list(burden.columns), cat=list(CAT_COLS),
                  site=list(SITE_ID_COLS), demo=list(DEMO_COLS))
    groups['default'] = num_all + ctx_cols + list(CAT_COLS)
    groups['noctx'] = num_all + list(CAT_COLS)
    return X, groups, cat_levels


def aux_targets(meta: pd.DataFrame, stats: dict | None = None):
    """Standardised TBP measurements used as dense auxiliary regression targets for the image network (NaN = ignored)."""
    cols = [c for c in AUX_COLS if c in meta.columns]
    A = np.full((len(meta), len(AUX_COLS)), np.nan, dtype=np.float32)
    stats = dict(stats or {})
    for j, c in enumerate(AUX_COLS):
        if c not in cols:
            continue
        v = pd.to_numeric(meta[c], errors='coerce').astype(np.float64).values
        if c == 'tbp_lv_areaMM2':
            v = np.log1p(np.clip(v, 0, None))
        if c not in stats:
            stats[c] = (float(np.nanmean(v)), float(np.nanstd(v) + 1e-6))
        A[:, j] = np.clip((v - stats[c][0]) / stats[c][1], -5, 5)
    return A, stats


# ============================================================================================================
# 3. Image-score features (shared by the training and the submission notebook)
# ============================================================================================================
def standardize(v, mu=None, sd=None):
    v = np.asarray(v, dtype=np.float64)
    mu = float(np.nanmean(v)) if mu is None else mu
    sd = float(np.nanstd(v) + 1e-9) if sd is None else sd
    return (v - mu) / sd, (mu, sd)


def _log_softmax(a):
    a = np.asarray(a, dtype=np.float64)
    m = a.max(axis=1, keepdims=True)
    return a - m - np.log(np.exp(a - m).sum(axis=1, keepdims=True))


def image_base_scores(per_backbone: dict) -> dict:
    """per_backbone: name -> (N, 5) array of STANDARDISED malignant / suspicious scores and raw type logits:
    columns = [z_malignant, type_logit_Mel, type_logit_BCC, type_logit_SCC, z_suspicious].
    Returns the base image scores that enter the GBDT (before patient context)."""
    names = sorted(per_backbone)
    out = {}
    for n in names:
        out[f'img_{n}'] = per_backbone[n][:, 0]
    out['img_mean'] = np.mean([per_backbone[n][:, 0] for n in names], axis=0)
    out['susp_mean'] = np.mean([per_backbone[n][:, 4] for n in names], axis=0)
    lsm = np.mean([_log_softmax(per_backbone[n][:, 1:4]) for n in names], axis=0)
    for j, t in enumerate(TYPES):
        out[f'type_{t}'] = lsm[:, j]
    return out


def image_feature_frame(base: dict, pid, site=None) -> pd.DataFrame:
    """Base image scores + their intra-patient context (z-score, percentile rank, ratio to the patient mean, gap to the patient
    maximum, z-score inside patient x body site). The image version of the 'ugly duckling' features."""
    pid = np.asarray(pid).astype(str)
    codes = pd.factorize(pid)[0]
    f = pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in base.items()})
    ctx = {}
    for c in [k for k in base if k.startswith('img_')] + ['susp_mean']:
        s = f[c]
        g = s.groupby(codes)
        ctx[f'{c}_pz'] = (s - g.transform('mean')) / (g.transform('std') + 1e-6)
        ctx[f'{c}_prk'] = g.rank(pct=True)
    s = f['img_mean']
    g = s.groupby(codes)
    p = 1 / (1 + np.exp(-s))
    ctx['img_mean_ratio'] = p / (p.groupby(codes).transform('mean') + 1e-9)       # 1st-place trick: score / patient mean score
    ctx['img_mean_gap'] = s - g.transform('max')
    if site is not None:
        sc = pd.factorize(pd.Series(pid).str.cat(pd.Series(np.asarray(site)).astype(str), sep='|').values)[0]
        gs = s.groupby(sc)
        ctx['img_mean_psz'] = (s - gs.transform('mean')) / (gs.transform('std') + 1e-6)
    out = pd.concat([f, pd.DataFrame(ctx)], axis=1)
    return out.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def image_feature_groups(cols):
    base = [c for c in cols if not c.endswith(('_pz', '_prk', '_ratio', '_gap', '_psz'))]
    ctx = [c for c in cols if c not in base]
    return dict(img_all=list(cols), img_base=base, img_ctx=ctx)


# ============================================================================================================
# 4. GBDT helpers (models are pickled by the training notebook and only used for prediction at submission time)
# ============================================================================================================
def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def make_bag(idx, y, seed, neg_frac, floor=30):
    rng = np.random.default_rng(seed)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    n_neg = int(min(len(neg), max(neg_frac * len(neg), floor * max(len(pos), 1))))
    return np.sort(np.r_[pos, rng.choice(neg, size=n_neg, replace=False)])


DEFAULT_PARAMS = dict(
    lgb=dict(learning_rate=0.03, num_leaves=31, max_depth=5, min_child_samples=60, subsample=0.75, subsample_freq=1,
             colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=2.5),
    xgb=dict(learning_rate=0.05, max_depth=5, subsample=0.7, colsample_bytree=0.5, min_child_weight=5, reg_lambda=2.0, scale_pos_weight=2.5),
    cat=dict(depth=6, learning_rate=0.05, l2_leaf_reg=5.0, scale_pos_weight=2.5),
)


def gbdt_model(kind, seed, n_est, threads, params=None, cat_cols=()):
    p = dict(DEFAULT_PARAMS[kind])
    p.update(params or {})
    n_est = int(p.pop('n_estimators', n_est))
    if kind == 'lgb':
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=n_est, random_state=seed, n_jobs=threads, verbosity=-1, **p)
    if kind == 'xgb':
        import xgboost as xgb
        kw = dict(n_estimators=n_est, tree_method='hist', enable_categorical=True, random_state=seed, n_jobs=threads, eval_metric='auc',
                  max_cat_to_onehot=1, **p)
        try:
            return xgb.XGBClassifier(device='cpu', **kw)
        except TypeError:
            return xgb.XGBClassifier(**kw)
    if kind == 'cat':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=int(n_est * 1.3), random_seed=seed, verbose=0, thread_count=threads,
                                  cat_features=list(cat_cols), allow_writing_files=False, **p)
    raise ValueError(kind)


def _cat_frame(X, cat_cols):
    X = X.copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str)
    return X


def gbdt_fit(kind, X, yb, seed, n_est, threads, params=None, cat_cols=()):
    cc = [c for c in cat_cols if c in X.columns]
    m = gbdt_model(kind, seed, n_est, threads, params, cc)
    if kind == 'cat':
        m.fit(_cat_frame(X, cc), yb)
    else:
        m.fit(X, yb)
    return m


def gbdt_logit(kind, model, X, cat_cols=()):
    cc = [c for c in cat_cols if c in X.columns]
    Xp = _cat_frame(X, cc) if kind == 'cat' else X
    return logit(model.predict_proba(Xp)[:, 1])


def predict_ensemble(models: dict, X: pd.DataFrame, cols: list, cat_cols=()) -> np.ndarray:
    """models: kind -> list of fitted models (all folds x seeds). Score = mean over kinds of the mean logit over that kind's models."""
    out = []
    for kind, ms in models.items():
        if ms:
            out.append(np.mean([gbdt_logit(kind, m, X[cols], cat_cols) for m in ms], axis=0))
    return np.mean(out, axis=0)


# ============================================================================================================
# 5. Metrics
# ============================================================================================================
def pauc_above_tpr(y_true, y_score, min_tpr=0.80):
    """Official ISIC-2024 metric: partial AUC above `min_tpr` (range 0..0.20; random = 0.02)."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.min() == y_true.max():
        return float('nan')
    max_fpr = 1.0 - min_tpr
    p_scaled = roc_auc_score(1 - y_true, -y_score, max_fpr=max_fpr)
    return float(0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (p_scaled - 0.5))


def pauc_manual(y, s, min_tpr=0.80):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    grid = np.linspace(min_tpr, 1.0, 4001)
    spec = 1.0 - np.interp(grid, tpr, fpr)
    return float(np.sum((spec[1:] + spec[:-1]) / 2 * np.diff(grid)))


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float('nan')


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float('nan')


def spec_at_sens(y, s, levels=(0.80, 0.90, 0.95)):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    return {f'spec@sens{int(l * 100)}': float(1 - np.interp(l, tpr, fpr)) for l in levels}


def nnt_at_sens(y, s, sens):
    """Lesions flagged per cancer found at the lowest threshold reaching `sens` (in-sample, as defined in the ISIC'24 challenge paper)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind='stable')
    tp = np.cumsum(y[order])
    need = int(math.ceil(sens * y.sum() - 1e-9))
    i = int(np.searchsorted(tp, need, side='left'))
    i = min(i, len(order) - 1)
    thr = s[order[i]]
    flagged = int((s >= thr).sum())
    tp_at = int(y[s >= thr].sum())
    return flagged / max(tp_at, 1)


def se_topk(pid, y, s, k=15):
    """Patient-level triage metrics. `SE_top15` = mean over cancer patients of the share of their cancers ranked in their top k
    (each diseased patient weighted equally, the ISIC'24 paper's definition)."""
    d = pd.DataFrame({'pid': np.asarray(pid), 'y': np.asarray(y).astype(int), 's': np.asarray(s, dtype=np.float64)})
    d['rk'] = d.groupby('pid')['s'].rank(ascending=False, method='first')
    mal = d[d.y == 1]
    if len(mal) == 0:
        return dict(SE_top15=float('nan'), patient_hit_top15=float('nan'), lesion_recall_top15=float('nan'))
    per_patient = (mal.rk <= k).groupby(mal.pid).mean()
    return dict(SE_top15=float(per_patient.mean()), patient_hit_top15=float((mal.groupby('pid')['rk'].min() <= k).mean()),
                lesion_recall_top15=float((mal.rk <= k).mean()))


def summarize(y, s, name='', pid=None):
    y = np.asarray(y).astype(int)
    out = dict(model=name, n=int(len(y)), n_pos=int(y.sum()), pAUC80=pauc_above_tpr(y, s), AUROC=auroc(y, s), AUPRC=auprc(y, s))
    out.update(spec_at_sens(y, s))
    if pid is not None:
        out['SE_top15'] = se_topk(pid, y, s)['SE_top15']
    return out


def pick_threshold(y, s, rule):
    """rule: ('sens', 0.9) -> lowest threshold with sensitivity >= 0.9; 'youden' -> max(sens + spec - 1); 'maxf1' -> max F1."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    if rule == 'youden':
        fpr, tpr, thr = roc_curve(y, s)
        return float(thr[np.argmax(tpr - fpr)])
    if rule == 'maxf1':
        p, r, thr = precision_recall_curve(y, s)
        f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
        return float(thr[np.argmax(f1)])
    if isinstance(rule, (tuple, list)) and rule[0] == 'sens':
        ss = np.sort(s[y == 1])[::-1]
        need = int(math.ceil(rule[1] * len(ss) - 1e-9))
        return float(ss[max(need - 1, 0)])
    raise ValueError(rule)


def nested_predictions(y, s, folds, rule):
    """The threshold applied to fold k is chosen on the OTHER folds only -> honest operating point."""
    y = np.asarray(y)
    pred = np.zeros(len(y), dtype=np.int8)
    thr = {}
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        thr[int(k)] = pick_threshold(y[tr], s[tr], rule)
        pred[va] = (s[va] >= thr[int(k)]).astype(np.int8)
    return pred, thr


def counts(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def op_metrics_from_counts(tp, fp, tn, fn, prev=None):
    """All threshold metrics. prev=None -> natural prevalence of the data; prev=0.5 -> the same classifier on a 50/50 test set."""
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    if prev is None:
        n = tp + fp + tn + fn
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        acc = (tp + tn) / max(n, 1)
    else:
        ppv = prev * sens / max(prev * sens + (1 - prev) * (1 - spec), 1e-12)
        npv = (1 - prev) * spec / max((1 - prev) * spec + prev * (1 - sens), 1e-12)
        acc = prev * sens + (1 - prev) * spec
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    if prev is None:
        den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
        mcc = (tp * tn - fp * fn) / den
    else:
        p, q = prev, 1 - prev
        tp_, fn_, tn_, fp_ = p * sens, p * (1 - sens), q * spec, q * (1 - spec)
        den = math.sqrt(max((tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_), 1e-12))
        mcc = (tp_ * tn_ - fp_ * fn_) / den
    lr_pos = sens / max(1 - spec, 1e-12)
    lr_neg = (1 - sens) / max(spec, 1e-12)
    return dict(sensitivity=sens, specificity=spec, precision=ppv, NPV=npv, accuracy=acc, balanced_accuracy=(sens + spec) / 2,
                F1=f1, MCC=mcc, LR_plus=lr_pos, LR_minus=lr_neg,
                flagged_per_cancer=((tp + fp) / tp) if (prev is None and tp) else float('nan'))


class ClusterBootstrap:
    """Resamples PATIENTS with replacement (lesions of one patient are correlated, so lesion-level bootstraps are too narrow)."""

    def __init__(self, pid, seed=0):
        codes, uniq = pd.factorize(np.asarray(pid))
        self.order = np.argsort(codes, kind='stable')
        self.cnt = np.bincount(codes)
        self.start = np.r_[0, np.cumsum(self.cnt)[:-1]]
        self.G = len(uniq)
        self.rng = np.random.default_rng(seed)

    def sample(self):
        pick = self.rng.integers(0, self.G, self.G)
        lens = self.cnt[pick]
        out_start = np.r_[0, np.cumsum(lens)[:-1]]
        pos = np.arange(lens.sum()) - np.repeat(out_start, lens) + np.repeat(self.start[pick], lens)
        return self.order[pos]

    def ci(self, fn, n_boot=200, y=None, min_pos=3, alpha=0.05):
        """fn(idx) -> float or dict of floats. Returns {key: (lo, hi)} (or (lo, hi))."""
        vals = []
        for _ in range(n_boot):
            idx = self.sample()
            if y is not None and y[idx].sum() < min_pos:
                continue
            vals.append(fn(idx))
        if not vals:
            return None
        if isinstance(vals[0], dict):
            return {k: (float(np.nanpercentile([v[k] for v in vals], 100 * alpha / 2)),
                        float(np.nanpercentile([v[k] for v in vals], 100 * (1 - alpha / 2)))) for k in vals[0]}
        return (float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2))))


def _trapz(yv, xv):
    return float(np.sum((xv[1:] - xv[:-1]) * (yv[1:] + yv[:-1]) / 2.0))


def roc_metrics(y, s, min_tpr=0.80):
    """AUROC, the official pAUC above `min_tpr` and AUPRC from ONE sort (identical to sklearn; used inside the bootstraps)."""
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    o = np.argsort(-s, kind='mergesort')
    ss, ys = s[o], y[o]
    idx = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tps = np.cumsum(ys)[idx].astype(np.float64)
    fps = (idx + 1) - tps
    P, Nn = tps[-1], fps[-1]
    if P == 0 or Nn == 0:
        return dict(AUROC=float('nan'), pAUC80=float('nan'), AUPRC=float('nan'))
    tpr, fpr = np.r_[0.0, tps / P], np.r_[0.0, fps / Nn]
    auc = _trapz(tpr, fpr)
    xf, yf = (1 - tpr)[::-1], (1 - fpr)[::-1]          # the competition scores the flipped problem up to FPR' = 1 - min_tpr
    mf = 1.0 - min_tpr
    stop = int(np.searchsorted(xf, mf, side='right'))
    yi = np.interp(mf, xf[stop - 1:stop + 1], yf[stop - 1:stop + 1])
    pauc = _trapz(np.r_[yf[:stop], yi], np.r_[xf[:stop], mf])
    ap = float(np.sum(np.diff(np.r_[0.0, tps / P]) * (tps / (tps + fps))))
    return dict(AUROC=auc, pAUC80=pauc, AUPRC=ap)


class CountBootstrap:
    """Fast patient bootstrap for decision metrics: per-patient TP/FP/TN/FN, resampled patients -> summed counts."""

    def __init__(self, pid, seed=0):
        self.codes, uniq = pd.factorize(np.asarray(pid))
        self.G = len(uniq)
        self.seed = seed

    def per_patient(self, y, pred):
        y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
        cols = [(pred == 1) & (y == 1), (pred == 1) & (y == 0), (pred == 0) & (y == 0), (pred == 0) & (y == 1)]
        return np.stack([np.bincount(self.codes, weights=c.astype(float), minlength=self.G) for c in cols], axis=1)

    def ci(self, y, pred, fn, n_boot=200, alpha=0.05):
        M = self.per_patient(y, pred)
        rng = np.random.default_rng(self.seed)
        vals = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, self.G, self.G), minlength=self.G)
            c = w @ M
            vals.append(fn(*c))
        keys = vals[0].keys()
        return {k: (float(np.nanpercentile([v[k] for v in vals], 100 * alpha / 2)),
                    float(np.nanpercentile([v[k] for v in vals], 100 * (1 - alpha / 2)))) for k in keys}


def calibration_stats(y, p, bins=20):
    """Brier score and expected calibration error over equal-mass bins of the predicted probability."""
    y = np.asarray(y).astype(float); p = np.asarray(p, dtype=np.float64)
    o = np.argsort(p)
    parts = np.array_split(o, bins)
    ece = sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in parts if len(b))
    return dict(Brier=float(np.mean((p - y) ** 2)), ECE=float(ece),
                observed_top_bin=float(y[parts[-1]].mean()), predicted_top_bin=float(p[parts[-1]].mean()))


def delong_test(y, s1, s2):
    """Paired DeLong test for two correlated AUROCs (fast O(n log n) version of Sun & Xu 2014). Assumes independent lesions."""
    from scipy.stats import rankdata, norm
    y = np.asarray(y).astype(bool)
    P = np.vstack([np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)])
    pos, neg = P[:, y], P[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.vstack([rankdata(r) for r in pos])
    ty = np.vstack([rankdata(r) for r in neg])
    tz = np.vstack([rankdata(np.r_[pos[i], neg[i]]) for i in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    zz = (aucs[0] - aucs[1]) / math.sqrt(max(var, 1e-18))
    return dict(AUROC_1=float(aucs[0]), AUROC_2=float(aucs[1]), diff=float(aucs[0] - aucs[1]), z=float(zz), p=float(2 * norm.sf(abs(zz))))


def multiclass_report(y_true, y_pred, names):
    """Per-class recall (sensitivity), specificity, precision, F1 + accuracy, balanced accuracy, macro-F1, Cohen's kappa."""
    k = len(names)
    yt = np.asarray(y_true).astype(int); yp = np.asarray(y_pred).astype(int)
    cm = np.bincount(yt * k + yp, minlength=k * k).reshape(k, k)
    n = cm.sum()
    rows = []
    for j, nm in enumerate(names):
        tp = cm[j, j]; fn = cm[j].sum() - tp; fp = cm[:, j].sum() - tp; tn = n - tp - fn - fp
        rec = tp / max(tp + fn, 1); prc = tp / max(tp + fp, 1)
        rows.append(dict(cls=nm, support=int(cm[j].sum()), recall=rec, specificity=tn / max(tn + fp, 1), precision=prc,
                         F1=2 * rec * prc / max(rec + prc, 1e-12)))
    per = pd.DataFrame(rows)
    po = np.trace(cm) / max(n, 1)
    pe = float(cm.sum(0) @ cm.sum(1)) / max(n, 1) ** 2
    present = per.support > 0
    overall = dict(accuracy=float(po), balanced_accuracy=float(per.recall[present].mean()), macro_F1=float(per.F1[present].mean()),
                   kappa=float((po - pe) / (1 - pe)) if pe < 1 else float('nan'))
    return per, overall, cm


def tune_offsets(S, y, n_cls, grid=None, rounds=3, w=None):
    """Additive per-class offsets (class 0 fixed at 0) on a score matrix S (n, n_cls), maximising balanced accuracy."""
    grid = np.arange(-12, 12.01, 0.25) if grid is None else grid
    b = np.zeros(n_cls)
    cnt = np.bincount(y, minlength=n_cls).astype(np.float64)

    def bal(bb):
        pr = np.argmax(S + bb, axis=1)
        hit = np.bincount(y[pr == y], minlength=n_cls)
        return float(np.mean(hit[cnt > 0] / cnt[cnt > 0]))

    best = bal(b)
    for _ in range(rounds):
        for c in range(1, n_cls):
            for g in grid:
                bb = b.copy(); bb[c] = g
                v = bal(bb)
                if v > best + 1e-12:
                    best, b = v, bb
    return b, best


# ============================================================================================================
# 6. Image backbones (timm / torchvision) and the multi-task network
# ============================================================================================================
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)

BACKBONES = {
    # name: source, model id, input resolution, backbone LR multiplier, weight decay, channels_last, pass img_size to timm
    'eva02_small':     dict(src='timm', id='eva02_small_patch14_336.mim_in22k_ft_in1k', res=224, lr_mult=0.5, wd=0.05, cl=False, img_size_arg=True),
    'edgenext_base':   dict(src='timm', id='edgenext_base.in21k_ft_in1k', res=256, lr_mult=1.0, wd=0.05, cl=True, img_size_arg=False),
    'convnext_tiny':   dict(src='timm', id='convnext_tiny.in12k_ft_in1k', res=224, lr_mult=0.5, wd=0.05, cl=True, img_size_arg=False),
    'swin_t':          dict(src='tv', id='swin_t', res=224, lr_mult=0.5, wd=0.05, cl=False, img_size_arg=False),
    'efficientnet_b3': dict(src='timm', id='efficientnet_b3.ra2_in1k', res=256, lr_mult=1.0, wd=1e-5, cl=True, img_size_arg=False),
    'resnet18':        dict(src='timm', id='resnet18.a1_in1k', res=224, lr_mult=1.0, wd=1e-4, cl=True, img_size_arg=False),
}


def build_backbone(bspec: dict, pretrained: bool, res: int):
    """Returns (module returning pooled features, feature dim, mean, std)."""
    import torch.nn as nn
    if bspec['src'] == 'timm':
        import timm
        kw = dict(bspec.get('kw', {}))
        if bspec.get('img_size_arg'):
            kw['img_size'] = res
        m = timm.create_model(bspec['id'], pretrained=pretrained, num_classes=0, **kw)
        cfg = getattr(m, 'pretrained_cfg', None) or {}
        mean, std = tuple(cfg.get('mean', IMNET_MEAN)), tuple(cfg.get('std', IMNET_STD))
        return m, int(m.num_features), mean, std
    from torchvision import models
    w = 'DEFAULT' if pretrained else None
    nm = bspec['id']
    if nm == 'swin_t':
        m = models.swin_t(weights=w); dim = m.head.in_features; m.head = nn.Identity()
    elif nm == 'convnext_tiny':
        m = models.convnext_tiny(weights=w); dim = m.classifier[2].in_features; m.classifier[2] = nn.Identity()
    elif nm == 'resnet18':
        m = models.resnet18(weights=w); dim = m.fc.in_features; m.fc = nn.Identity()
    elif nm == 'efficientnet_v2_s':
        m = models.efficientnet_v2_s(weights=w); dim = m.classifier[1].in_features; m.classifier[1] = nn.Identity()
    else:
        raise ValueError(nm)
    return m, int(dim), IMNET_MEAN, IMNET_STD


def make_net(bspec, pretrained, res, n_aux):
    import torch.nn as nn

    class FTNet(nn.Module):
        """Backbone + one linear head: [malignant | Melanoma, BCC, SCC | suspicious | aux TBP regressions]."""

        def __init__(self):
            super().__init__()
            self.backbone, dim, self.mean, self.std = build_backbone(bspec, pretrained, res)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(dim, N_HEADS + n_aux)

        def forward(self, x):
            return self.head(self.drop(self.backbone(x)))

    return FTNet()


VIEW_SETS = {1: [0], 2: [0, 4], 4: [0, 2, 4, 6], 8: list(range(8))}


class Img:
    """GPU-side preprocessing: uint8 crops (B, S, S, 3) -> normalised float tensors at the model resolution."""

    def __init__(self, device, res, mean, std):
        import torch
        self.torch = torch
        self.dev = device
        self.res = res
        self.mean = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    def to_float(self, u8):
        t = self.torch.from_numpy(np.ascontiguousarray(u8))
        if self.dev.type == 'cuda':
            t = t.pin_memory().to(self.dev, non_blocking=True)
        return t.permute(0, 3, 1, 2).float().div_(255.0)

    def resize(self, x):
        import torch.nn.functional as F
        if x.shape[-1] != self.res or x.shape[-2] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode='bicubic', align_corners=False).clamp_(0, 1)
        return x

    def norm(self, x):
        return (x - self.mean) / self.std

    def prep(self, u8):
        return self.norm(self.resize(self.to_float(u8)))

    @staticmethod
    def d4(x, v):
        """v in 0..7: horizontal flip (v>=4) then rotation by 90*(v%4) degrees = the dihedral group D4."""
        import torch
        if v >= 4:
            x = torch.flip(x, dims=(3,))
        if v % 4:
            x = torch.rot90(x, v % 4, dims=(2, 3))
        return x

    def augment(self, u8, gen, strength=1.0, rot_max=math.pi, scale=(0.80, 1.0), shift=0.08, jitter=0.2,
                p_drop=0.5, drop_frac=0.35, p_noise=0.3, noise_sd=0.03, p_blur=0.25, vflip=False):
        """Random rotation / mirror / zoom / shift + brightness-contrast-saturation jitter + coarse dropout + noise + blur, on the GPU."""
        import torch
        import torch.nn.functional as F
        x = self.to_float(u8)
        B, dev = x.shape[0], x.device
        r = lambda *s: torch.rand(*s, device=dev, generator=gen)
        ang = (r(B) * 2 - 1) * rot_max * strength
        sc = scale[0] + (scale[1] - scale[0]) * r(B)
        flip = (r(B) < 0.5).float() * 2 - 1
        vfl = ((r(B) < 0.5).float() * 2 - 1) if vflip else torch.ones(B, device=dev)
        tx, ty = (r(B) - 0.5) * 2 * shift * strength, (r(B) - 0.5) * 2 * shift * strength
        c, s = torch.cos(ang), torch.sin(ang)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = sc * c * flip, -sc * s * vfl, tx
        theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sc * s * flip, sc * c * vfl, ty
        grid = F.affine_grid(theta, size=(B, 3, self.res, self.res), align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        j = lambda: 1 + (r(B, 1, 1, 1) - 0.5) * 2 * jitter * strength
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - m) * j() + m) * j()
        gmean = x.mean(dim=1, keepdim=True)
        x = ((x - gmean) * j() + gmean).clamp_(0, 1)
        if p_blur > 0:
            bl = r(B) < p_blur * strength
            if bool(bl.any()):
                xb = F.avg_pool2d(F.pad(x[bl], (1, 1, 1, 1), mode='reflect'), 3, stride=1)
                x = x.clone(); x[bl] = xb
        if p_noise > 0:
            nm = (r(B, 1, 1, 1) < p_noise * strength).float()
            x = (x + nm * noise_sd * torch.randn(x.shape, device=dev, generator=gen)).clamp_(0, 1)
        if p_drop > 0:
            dm = r(B) < p_drop * strength
            if bool(dm.any()):
                L = max(2, int(self.res * drop_frac))
                cx = (r(B) * (self.res - L)).long(); cy = (r(B) * (self.res - L)).long()
                ar = torch.arange(self.res, device=dev)
                mx = (ar.view(1, -1) >= cx.view(-1, 1)) & (ar.view(1, -1) < (cx + L).view(-1, 1))
                my = (ar.view(1, -1) >= cy.view(-1, 1)) & (ar.view(1, -1) < (cy + L).view(-1, 1))
                hole = (my.view(B, 1, -1, 1) & mx.view(B, 1, 1, -1)) & dm.view(B, 1, 1, 1)
                x = torch.where(hole, torch.full_like(x, 0.5), x)
        return self.norm(x)


# ============================================================================================================
# 7. Training / inference primitives (run inside the GPU worker processes)
# ============================================================================================================
class EMA:
    """Exponential moving average of the weights (the evaluated network = the EMA network; no checkpoint picking on validation data)."""

    def __init__(self, model, decay):
        sd = model.state_dict()
        self.keys = list(sd.keys())
        self.live = [sd[k] for k in self.keys]
        self.shadow = [v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone() for v in self.live]
        self.decay = decay
        self.n = 0

    def update(self):
        import torch
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        with torch.no_grad():
            for s, v in zip(self.shadow, self.live):
                if v.dtype.is_floating_point:
                    s.mul_(d).add_(v.detach().float(), alpha=1 - d)
                else:
                    s.copy_(v)

    def state_dict(self):
        return {k: s for k, s in zip(self.keys, self.shadow)}


class Prefetch:
    """Reads the next batches of uint8 crops from the memory-mapped store in a background thread."""

    def __init__(self, imgs, batches, depth=4):
        self.q = queue.Queue(depth)
        self.stop = False
        self.t = threading.Thread(target=self._run, args=(imgs, batches), daemon=True)
        self.t.start()

    def _run(self, imgs, batches):
        try:
            for b in batches:
                if self.stop:
                    break
                bs = np.sort(b)
                self.q.put((bs, np.asarray(imgs[bs])))
        except BaseException as e:      # surface reader errors in the consumer
            self.q.put(e)
        self.q.put(None)

    def __iter__(self):
        while True:
            it = self.q.get()
            if it is None:
                return
            if isinstance(it, BaseException):
                raise it
            yield it

    def close(self):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


def _autocast(dev, enabled=True):
    import torch
    return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=bool(enabled and dev.type == 'cuda'))


def _grad_scaler(dev):
    import torch
    try:
        return torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=dev.type == 'cuda')


def _param_groups(net, bspec, lr, lr_head):
    dec, nodec = [], []
    for n, p in net.backbone.named_parameters():
        if not p.requires_grad:
            continue
        (nodec if (p.ndim <= 1 or n.endswith('.bias') or 'pos_embed' in n or 'cls_token' in n or 'rel_pos' in n) else dec).append(p)
    lb = lr * bspec.get('lr_mult', 1.0)
    return [dict(params=dec, lr=lb, weight_decay=bspec.get('wd', 1e-4)), dict(params=nodec, lr=lb, weight_decay=0.0),
            dict(params=list(net.head.parameters()), lr=lr_head, weight_decay=1e-4)]


def predict_logits(net, img: Img, arr, idx, views, bs, cl=False, log=None):
    """Mean of the head outputs over the TTA views -> (n, N_HEADS). CUDA OOM halves the batch; non-finite fp16 output is redone in fp32."""
    import torch
    dev = img.dev
    net.eval()
    out = np.empty((len(idx), N_HEADS), np.float32)
    a, bs_cur, n_fp32, n_bad = 0, bs, 0, 0
    with torch.no_grad():
        while a < len(idx):
            b = idx[a:a + bs_cur]
            try:
                x = img.prep(np.asarray(arr[b]))
                xv = torch.cat([Img.d4(x, v) for v in views], 0)
                if cl:
                    xv = xv.contiguous(memory_format=torch.channels_last)
                with _autocast(dev):
                    o = net(xv)[:, :N_HEADS].float()
                if not bool(torch.isfinite(o).all()):
                    o = net(xv)[:, :N_HEADS].float()     # fp32 retry
                    n_fp32 += 1
                o = o.view(len(views), len(b), -1).mean(0)
                bad = ~torch.isfinite(o).all(dim=1)
                if bool(bad.any()):
                    n_bad += int(bad.sum())
                    o = torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)
                out[a:a + len(b)] = o.cpu().numpy()
                a += len(b)
            except RuntimeError as e:
                if 'out of memory' not in str(e).lower() or bs_cur <= 4:
                    raise
                torch.cuda.empty_cache()
                bs_cur = max(4, bs_cur // 2)
                if log:
                    log(f'  CUDA out of memory in inference -> batch {bs_cur}')
    if log and n_fp32:
        log(f'  fp16 overflow in {n_fp32} batch(es) -> recomputed in fp32')
    if n_bad > 0.005 * max(len(idx), 1):
        raise FloatingPointError(f'{n_bad}/{len(idx)} rows non-finite after the fp32 retry')
    return out


def _loss(out, yb, tb, sb, ab, mm, w):
    import torch
    import torch.nn.functional as F
    zero = out.sum() * 0.0
    l_mal = F.binary_cross_entropy_with_logits(out[mm, 0], yb[mm]) if bool(mm.any()) else zero
    tm = tb >= 0
    l_type = F.cross_entropy(out[tm, 1:4], tb[tm].long()) if (w['w_type'] > 0 and bool(tm.any())) else zero
    l_susp = F.binary_cross_entropy_with_logits(out[:, 4], sb) if w['w_susp'] > 0 else zero
    l_aux = zero
    if w['w_aux'] > 0 and ab is not None and ab.shape[1] > 0:
        am = torch.isfinite(ab)
        if bool(am.any()):
            l_aux = ((out[:, N_HEADS:][am] - ab[am]) ** 2).mean()
    tot = l_mal + w['w_type'] * l_type + w['w_susp'] * l_susp + w['w_aux'] * l_aux
    return tot, torch.stack([l_mal.detach(), l_type.detach(), l_susp.detach(), l_aux.detach()])


def train_net(dev, bspec, res, data, rows, ft, seed, n_aux, pretrained=True, init=None, log=print, tag=''):
    """Fine-tune one network on `rows` of `data` (dict: imgs, y, type_code, susp, aux, mal_mask). Returns the EMA network."""
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    net = make_net(bspec, pretrained and init is None, res, n_aux)
    if init is not None:
        sd = torch.load(init, map_location='cpu')
        missing, unexpected = net.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        log(f'  {tag} initialised from {Path(init).name} (missing {len(missing)}, unexpected {len(unexpected)} keys)')
    net = net.to(dev)
    cl = bool(bspec.get('cl')) and dev.type == 'cuda'
    if cl:
        net = net.to(memory_format=torch.channels_last)
    img = Img(dev, res, net.mean, net.std)
    y, tcode, susp, aux, mm_all = data['y'], data['type_code'], data['susp'], data.get('aux'), data['mal_mask']
    pos = rows[y[rows] == 1]
    sneg = rows[(y[rows] == 0) & (susp[rows] == 1)]
    neg = rows[(y[rows] == 0) & (susp[rows] == 0)]
    bs = int(ft['bs'])
    n_neg_ep = int(min(len(neg), ft['neg_per_epoch']))
    n_s_ep = int(min(len(sneg), ft['susp_per_epoch']))
    per_ep = len(pos) * int(ft['pos_mult']) + n_s_ep + n_neg_ep
    steps_ep = max(1, per_ep // bs)
    total = max(1, int(ft['epochs']) * steps_ep)
    warm = max(1, int(ft.get('warmup_frac', 0.05) * total))
    opt = torch.optim.AdamW(_param_groups(net, bspec, ft['lr'], ft['lr_head']))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    scaler = _grad_scaler(dev)
    ema = EMA(net, 0.5 ** (1.0 / max(1.0, ft.get('ema_halflife', 0.1) * total)))
    w = dict(w_type=ft['w_type'], w_susp=ft['w_susp'] if data.get('use_susp', True) else 0.0, w_aux=ft['w_aux'] if aux is not None else 0.0)
    neg_perm, ptr = rng.permutation(neg), 0
    t0 = time.time()
    seen = 0
    for ep in range(int(ft['epochs'])):
        if n_neg_ep:
            if ptr + n_neg_ep > len(neg_perm):            # cycle through ALL benign lesions across epochs (fresh order each pass)
                neg_perm, ptr = rng.permutation(neg), 0
            negs = neg_perm[ptr:ptr + n_neg_ep]
            ptr += n_neg_ep
        else:
            negs = np.array([], dtype=rows.dtype)
        idx = np.r_[np.repeat(pos, int(ft['pos_mult'])), rng.choice(sneg, n_s_ep, replace=False) if n_s_ep else np.array([], dtype=rows.dtype), negs]
        rng.shuffle(idx)
        batches = [idx[a:a + bs] for a in range(0, len(idx), bs) if len(idx[a:a + bs]) >= max(4, bs // 4)]
        net.train()
        acc = torch.zeros(4, device=dev)
        nb, nbad = 0, 0
        pf = Prefetch(data['imgs'], batches)
        try:
            for b, u8 in pf:
                x = img.augment(u8, gen, strength=ft.get('aug_strength', 1.0))
                if cl:
                    x = x.contiguous(memory_format=torch.channels_last)
                yb = torch.from_numpy(y[b].astype(np.float32)).to(dev)
                tb = torch.from_numpy(tcode[b].astype(np.int64)).to(dev)
                sb = torch.from_numpy(susp[b].astype(np.float32)).to(dev)
                mm = torch.from_numpy(mm_all[b].astype(bool)).to(dev)
                ab = torch.from_numpy(aux[b].astype(np.float32)).to(dev) if aux is not None else None
                with _autocast(dev):
                    out = net(x)
                loss, parts = _loss(out.float(), yb, tb, sb, ab, mm, w)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), ft.get('clip', 2.0))
                scaler.step(opt)
                scaler.update()
                sched.step()
                ema.update()
                ok = bool(torch.isfinite(loss.detach()))
                if ok:
                    acc += parts
                else:
                    nbad += 1
                nb += 1
                seen += len(b)
        finally:
            pf.close()
        a = (acc / max(nb - nbad, 1)).tolist()
        log(f'    {tag} epoch {ep + 1}/{ft["epochs"]}: loss mal {a[0]:.4f} | type {a[1]:.3f} | susp {a[2]:.3f} | aux {a[3]:.3f} '
            f'| {seen / max(time.time() - t0, 1e-9):.0f} img/s ({time.time() - t0:.0f}s)' + (f' | {nbad} non-finite batch(es) skipped' if nbad else ''))
        if nbad > 0.2 * max(nb, 1):
            raise FloatingPointError(f'{nbad}/{nb} training batches non-finite - training diverged')
    net.load_state_dict(ema.state_dict())
    return net, img, cl


def save_half(net, path):
    import torch
    sd = {k: (v.detach().half().cpu() if v.dtype.is_floating_point else v.detach().cpu()) for k, v in net.state_dict().items()}
    tmp = str(path) + '.tmp'
    torch.save(sd, tmp)
    os.replace(tmp, path)


def _savez_atomic(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


# ============================================================================================================
# 8. Worker jobs
# ============================================================================================================
class WorkerCtx:
    def __init__(self, spec, dev, log):
        self.spec, self.dev, self.log = spec, dev, log
        self._data, self._ext = None, None

    @property
    def data(self):
        if self._data is None:
            z = np.load(self.spec['arrays'])
            d = {k: z[k] for k in z.files}
            d['imgs'] = np.load(self.spec['imgs'], mmap_mode='r')
            self._data = d
        return self._data

    @property
    def ext(self):
        if self._ext is None:
            e = self.spec['ext']
            z = np.load(e['arrays'])
            d = {k: z[k] for k in z.files}
            d['imgs'] = np.load(e['imgs'], mmap_mode='r')
            d['use_susp'] = False
            self._ext = d
        return self._ext


def _bspec(spec, name):
    b = dict(spec['backbones'][name])
    return b, int(b['res'])


def job_bench(ctx, job):
    """Speed of inference (image-views/s) and of a training step (images/s) for each backbone, with the real code path."""
    import torch
    spec, dev, log = ctx.spec, ctx.dev, ctx.log
    S, views, ft = spec['store_res'], VIEW_SETS[spec['ft']['views']], spec['ft']
    res_all = {}
    for name in job['backbones']:
        r = {}
        try:
            bspec, res = _bspec(spec, name)
            t0 = time.time()
            net = make_net(bspec, spec.get('pretrained', True), res, len(AUX_COLS)).to(dev)
            r['build_s'] = time.time() - t0
            cl = bool(bspec.get('cl')) and dev.type == 'cuda'
            if cl:
                net = net.to(memory_format=torch.channels_last)
            img = Img(dev, res, net.mean, net.std)
            bsi = int(ft['bs_infer'])
            fake = np.random.randint(0, 255, (bsi * 3, S, S, 3), dtype=np.uint8)
            predict_logits(net, img, fake, np.arange(bsi), views, bsi, cl)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t = time.time()
            predict_logits(net, img, fake, np.arange(bsi, 3 * bsi), views, bsi, cl)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            r['infer_views_per_s'] = 2 * bsi * len(views) / (time.time() - t)
            bs = int(ft['bs'])
            opt = torch.optim.AdamW(_param_groups(net, bspec, ft['lr'], ft['lr_head']))
            scaler = _grad_scaler(dev)
            gen = torch.Generator(device=dev); gen.manual_seed(0)
            w = dict(w_type=ft['w_type'], w_susp=ft['w_susp'], w_aux=ft['w_aux'])
            while True:
                try:
                    fk = np.random.randint(0, 255, (bs, S, S, 3), dtype=np.uint8)
                    yb = torch.zeros(bs, device=dev); tb = torch.zeros(bs, dtype=torch.long, device=dev)
                    sb = torch.zeros(bs, device=dev); mm = torch.ones(bs, dtype=torch.bool, device=dev)
                    ab = torch.zeros(bs, len(AUX_COLS), device=dev)

                    def step():
                        x = img.augment(fk, gen)
                        if cl:
                            x = x.contiguous(memory_format=torch.channels_last)
                        with _autocast(dev):
                            out = net(x)
                        loss, _ = _loss(out.float(), yb, tb, sb, ab, mm, w)
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()

                    net.train()
                    step(); step()
                    if dev.type == 'cuda':
                        torch.cuda.synchronize()
                    t = time.time()
                    n_st = 4 if dev.type == 'cuda' else 2
                    for _ in range(n_st):
                        step()
                    if dev.type == 'cuda':
                        torch.cuda.synchronize()
                    r['train_img_per_s'] = bs * n_st / (time.time() - t)
                    r['bs'] = bs
                    break
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and bs > 8:
                        torch.cuda.empty_cache(); bs //= 2
                        log(f'  bench {name}: CUDA OOM -> training batch {bs}')
                        continue
                    raise
            if dev.type == 'cuda':
                r['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
            log(f'  bench {name}@{res}: {r["infer_views_per_s"]:.0f} img-views/s inference | {r["train_img_per_s"]:.0f} img/s training '
                f'(batch {r["bs"]}) | build {r["build_s"]:.0f}s')
            del net, opt
        except Exception as e:
            r = dict(error=f'{type(e).__name__}: {str(e)[:400]}')
            log(f'  bench {name} FAILED: {r["error"]}')
        gc.collect()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        res_all[name] = r
    _write_json_atomic(job['out'], res_all)


def job_fold(ctx, job):
    """Cross-fitted fine-tuning: train on the other folds, predict every lesion of the held-out fold."""
    import torch
    spec, dev, log = ctx.spec, ctx.dev, ctx.log
    name, k = job['backbone'], int(job['fold'])
    bspec, res = _bspec(spec, name)
    data = ctx.data
    fold = data['fold']
    rows = np.where((fold != k) & data['trainable'])[0]
    ft = dict(spec['ft'])
    ft['bs'] = int(job.get('bs', ft['bs']))
    t0 = time.time()
    for attempt in range(3):
        try:
            net, img, cl = train_net(dev, bspec, res, data, rows, ft, seed=int(job['seed']), n_aux=data['aux'].shape[1],
                                     pretrained=spec.get('pretrained', True), init=job.get('init'), log=log, tag=f'{name} f{k}')
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and ft['bs'] > 8:
                gc.collect(); torch.cuda.empty_cache()
                ft['bs'] = ft['bs'] // 2
                log(f'  {name} f{k}: CUDA OOM -> training batch {ft["bs"]}, restarting the fold')
                continue
            raise
    t_train = time.time() - t0
    va = np.where(fold == k)[0]
    lg = predict_logits(net, img, data['imgs'], va, VIEW_SETS[spec['ft']['views']], int(spec['ft']['bs_infer']), cl, log)
    if not np.isfinite(lg).all():
        raise FloatingPointError('non-finite logits')
    if job.get('weights'):
        save_half(net, job['weights'])
        Path(job['weights']).with_suffix('.sig').write_text(Path(job['out']).name)   # which fold run these weights belong to
    msg = ''
    yv = data['y'][va]
    if 0 < yv.sum() < len(yv):
        msg = f' | held-out AUROC {auroc(yv, lg[:, 0]):.4f} pAUC80 {pauc_above_tpr(yv, lg[:, 0]):.4f}'
    _savez_atomic(job['out'], va=va.astype(np.int64), lg=lg)
    log(f'  [{name}] fold {k} done: train {t_train / 60:.1f} min + inference {(time.time() - t0 - t_train) / 60:.1f} min{msg}')
    del net
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()


def job_pretrain(ctx, job):
    """Optional: pre-train a backbone on EXTERNAL dermoscopy images before the cross-fitted fine-tuning."""
    import torch
    spec, dev, log = ctx.spec, ctx.dev, ctx.log
    name = job['backbone']
    bspec, res = _bspec(spec, name)
    e = ctx.ext
    ft = dict(spec['ft'])
    ft.update(spec['ext'].get('ft', {}))
    rows = np.arange(len(e['y']))
    net, img, cl = train_net(dev, bspec, res, e, rows, ft, seed=int(job['seed']), n_aux=len(AUX_COLS), pretrained=spec.get('pretrained', True),
                             log=log, tag=f'{name} external')
    save_half(net, job['out'])
    del net
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()


def audit_augment(img: Img, u8, gen):
    """The augmentations a typical 'augment, then split' paper applies (rotation +-30 deg, flips, zoom/crop, colour jitter, noise)."""
    return img.augment(u8, gen, strength=1.0, rot_max=math.radians(30), scale=(0.8, 1.0), shift=0.05, jitter=0.2, p_drop=0.0,
                       p_noise=1.0, noise_sd=0.02, p_blur=0.0, vflip=True)


def job_audit(ctx, job):
    """Frozen ImageNet embeddings of each audit lesion and of `n_aug` augmented copies (for the leakage audit)."""
    import torch
    import torch.nn.functional as F
    spec, dev, log = ctx.spec, ctx.dev, ctx.log
    bspec, res = _bspec(spec, job['backbone'])
    m, dim, mean, std = build_backbone(bspec, spec.get('pretrained', True), res)
    m = m.to(dev).eval()
    img = Img(dev, res, mean, std)
    rows = np.asarray(job['rows'], dtype=np.int64)
    srt = np.argsort(rows)
    rows_s = rows[srt]
    imgs = ctx.data['imgs']
    n_aug, bs = int(job['n_aug']), int(spec['ft']['bs_infer'])
    E = np.zeros((n_aug + 1, len(rows), dim), np.float16)
    gen = torch.Generator(device=dev)
    with torch.no_grad():
        for a in range(n_aug + 1):
            gen.manual_seed(1000 + a)
            for s in range(0, len(rows_s), bs):
                b = rows_s[s:s + bs]
                u8 = np.asarray(imgs[b])
                x = img.prep(u8) if a == 0 else audit_augment(img, u8, gen)
                with _autocast(dev):
                    f = m(x).float()
                E[a, srt[s:s + bs]] = F.normalize(torch.nan_to_num(f), dim=1).cpu().numpy().astype(np.float16)
    _savez_atomic(job['out'], E=E, rows=rows)
    log(f'  audit embeddings: {len(rows)} lesions x {n_aug + 1} copies ({dim}-d)')


def _read_raw(h5, jdir, iid):
    if h5 is not None:
        r = h5[iid][()]
        return r.tobytes() if hasattr(r, 'tobytes') else bytes(r)
    return (Path(jdir) / f'{iid}.jpg').read_bytes()


def decode_crop(raw, S):
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert('RGB')
    if im.size != (S, S):
        im = im.resize((S, S), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def _open_src(src):
    src = Path(src)
    if src.suffix in ('.hdf5', '.h5'):
        import h5py
        return h5py.File(src, 'r'), None
    return None, src


def job_test(ctx, job):
    """Submission-time inference: all fold networks of all backbones on a range of test lesions (decoded here, block by block)."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    spec, dev, log = ctx.spec, ctx.dev, ctx.log
    ids = np.load(job['ids'], allow_pickle=False)[job['a']:job['b']]
    S, views, bsi = int(spec['store_res']), VIEW_SETS[int(spec['views'])], int(spec['bs_infer'])
    nets = []
    for m in job['models']:
        bspec, res = _bspec(spec, m['backbone'])
        net = make_net(bspec, False, res, int(spec['n_aux']))
        sd = torch.load(m['path'], map_location='cpu')
        net.load_state_dict({k: v.float() for k, v in sd.items()})
        net = net.to(dev).eval()
        cl = bool(bspec.get('cl')) and dev.type == 'cuda'
        if cl:
            net = net.to(memory_format=torch.channels_last)
        nets.append((net, Img(dev, res, net.mean, net.std), cl))
    log(f'  test job {job["id"]}: {len(nets)} networks loaded, {len(ids):,} lesions')
    out = np.zeros((len(ids), len(nets), N_HEADS), np.float32)
    h5, jdir = _open_src(job['src'])
    BLOCK = int(job.get('block', 4096))
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(spec.get('threads', 2)))) as ex:
            nxt = None
            for a in range(0, len(ids), BLOCK):
                raws = [_read_raw(h5, jdir, i) for i in ids[a:a + BLOCK]]
                u8 = np.stack(list(ex.map(lambda r: decode_crop(r, S), raws)))
                for mi, (net, img, cl) in enumerate(nets):
                    out[a:a + len(u8), mi] = predict_logits(net, img, u8, np.arange(len(u8)), views, bsi, cl, log)
                done = min(a + BLOCK, len(ids))
                if (a // BLOCK) % 5 == 0 or done == len(ids):
                    log(f'  test job {job["id"]}: {done:,}/{len(ids):,} lesions ({done / max(time.time() - t0, 1e-9):.0f} lesions/s)')
    finally:
        if h5 is not None:
            h5.close()
    _savez_atomic(job['out'], lg=out, a=np.int64(job['a']), b=np.int64(job['b']))


JOBS = dict(bench=job_bench, fold=job_fold, pretrain=job_pretrain, audit=job_audit, test=job_test)


# ============================================================================================================
# 9. Worker process: claims jobs from a shared list (lock files), respects the deadline, survives job failures
# ============================================================================================================
def _status(run_dir, job):
    if Path(job['out']).exists():
        return 'done'
    sd = Path(run_dir) / 'status'
    for st in ('failed', 'skipped'):
        if (sd / f'{job["id"]}.{st}').exists():
            return st
    if (sd / f'{job["id"]}.lock').exists():
        return 'locked'
    return None


def _mark(run_dir, job_id, st, msg=''):
    p = Path(run_dir) / 'status' / f'{job_id}.{st}'
    p.write_text(msg)


def _try_lock(run_dir, job, tag):
    p = Path(run_dir) / 'status' / f'{job["id"]}.lock'
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, tag.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def worker_main(run_dir, device, tag):
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / 'spec.json').read_text())
    jobs = json.loads((run_dir / 'jobs.json').read_text())
    logf = open(run_dir / f'log_{tag}.txt', 'a', buffering=1)
    t_start = time.time()

    def log(msg):
        logf.write(f'[{time.strftime("%H:%M:%S")} {tag}] {msg}\n')

    import torch
    torch.set_num_threads(max(1, int(spec.get('threads', 1))))
    dev = torch.device('cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        log(f'worker up on {torch.cuda.get_device_name(0)} | torch {torch.__version__}')
    else:
        log(f'worker up on CPU | torch {torch.__version__}')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    ctx = WorkerCtx(spec, dev, log)
    deadline = float(spec.get('deadline', 1e18))
    n_workers = max(1, int(spec.get('n_workers', 1)))
    st_dir = run_dir / 'status'

    def bb_failed(name):
        return (st_dir / f'bb_{name}.failed').exists()

    while True:
        progressed, waiting = False, False
        for job in jobs:
            st = _status(run_dir, job)
            if st in ('done', 'failed', 'skipped'):
                continue
            if st == 'locked':
                waiting = True
                continue
            deps = [d for d in jobs if d['id'] in job.get('deps', [])]
            dst = [_status(run_dir, d) for d in deps]
            if any(s in ('failed', 'skipped') for s in dst) and not job.get('soft_deps'):
                _mark(run_dir, job['id'], 'skipped', 'dependency failed')
                log(f'skip {job["id"]}: a dependency failed or was skipped')
                progressed = True
                continue
            if any(s not in ('done', 'failed', 'skipped') for s in dst):
                waiting = True
                continue
            if job.get('soft_deps'):
                job = dict(job)
                if job.get('init') and not Path(job['init']).exists():
                    job['init'] = None
            if job['kind'] == 'fold' and bb_failed(job['backbone']):
                _mark(run_dir, job['id'], 'skipped', 'backbone failed')
                progressed = True
                continue
            est = float(job.get('est_s', 0))
            if job['kind'] == 'fold':
                rem = [j for j in jobs if j['kind'] == 'fold' and j['backbone'] == job['backbone'] and _status(run_dir, j) in (None, 'locked')]
                need = math.ceil(len(rem) / n_workers) * est
                if time.time() + 1.05 * need > deadline:
                    for j in rem:
                        if _status(run_dir, j) is None:
                            _mark(run_dir, j['id'], 'skipped', 'time budget')
                    log(f'skip backbone {job["backbone"]}: {len(rem)} fold(s) x {est / 60:.0f} min do not fit the remaining budget '
                        f'({max(deadline - time.time(), 0) / 60:.0f} min)')
                    progressed = True
                    continue
            elif est and time.time() + est > deadline:
                _mark(run_dir, job['id'], 'skipped', 'time budget')
                log(f'skip {job["id"]}: does not fit the remaining budget')
                progressed = True
                continue
            if not _try_lock(run_dir, job, tag):
                continue
            log(f'start {job["id"]} (estimate {est / 60:.1f} min)')
            t0 = time.time()
            try:
                JOBS[job['kind']](ctx, job)
                log(f'done {job["id"]} in {(time.time() - t0) / 60:.1f} min')
            except Exception as ex:
                tb = traceback.format_exc()
                log(f'FAILED {job["id"]}: {type(ex).__name__}: {str(ex)[:300]}')
                logf.write(tb + '\n')
                _mark(run_dir, job['id'], 'failed', tb)
                if job['kind'] == 'fold':
                    (st_dir / f'bb_{job["backbone"]}.failed').write_text(tb)
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            progressed = True
            break                                  # re-scan from the top: the job list is in priority order
        if not progressed:
            if waiting:
                time.sleep(3)
                continue
            break
    log(f'worker finished after {(time.time() - t_start) / 60:.1f} min')
    logf.close()


# ============================================================================================================
# 10. Main-process helpers: worker pool, decoding into a memory-mapped store, external data discovery
# ============================================================================================================
_LIVE_WORKERS = globals().get('_LIVE_WORKERS', [])          # survives importlib.reload: workers of earlier cell runs


class WorkerPool:
    """One worker process per GPU (no DataParallel). The notebook process keeps the CPU for GBDT work while the GPUs train."""

    def __init__(self, run_dir, lib_path, devices, threads=1, log=print):
        self.run_dir = Path(run_dir)
        self.lib_path = str(lib_path)
        self.devices = list(devices)
        self.threads = threads
        self.log = log
        self.procs = []
        self.offsets = {}

    def start(self, spec, jobs):
        import subprocess
        for p in _LIVE_WORKERS:                              # a cell re-run must not leave a second set of workers on the same jobs
            if p.poll() is None:
                p.kill()
                p.wait()
        _LIVE_WORKERS.clear()
        for f in self.run_dir.glob('log_*.txt'):            # print only what this session's workers write
            self.offsets[f.name] = f.stat().st_size
        (self.run_dir / 'status').mkdir(parents=True, exist_ok=True)
        for f in (self.run_dir / 'status').glob('*'):
            if f.suffix in ('.lock', '.failed', '.skipped'):
                f.unlink()                                   # stale state from an earlier session
        spec = dict(spec, n_workers=len(self.devices), threads=self.threads)
        _write_json_atomic(self.run_dir / 'spec.json', spec)
        _write_json_atomic(self.run_dir / 'jobs.json', jobs)
        self.procs = []
        for i, dev in enumerate(self.devices):
            tag = f'gpu{i}' if dev == 'cuda' else f'cpu{i}'
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = str(i) if dev == 'cuda' else ''
            env['OMP_NUM_THREADS'] = str(self.threads)
            env['MKL_NUM_THREADS'] = str(self.threads)
            env['PYTHONUNBUFFERED'] = '1'
            out = open(self.run_dir / f'stdout_{tag}.txt', 'ab')
            p = subprocess.Popen([sys.executable, '-u', self.lib_path, 'worker', '--run-dir', str(self.run_dir), '--device', dev,
                                  '--tag', tag], stdout=out, stderr=subprocess.STDOUT, env=env)
            self.procs.append((tag, p, out))
            _LIVE_WORKERS.append(p)
        import atexit
        atexit.register(self.kill)

    def pump(self, prefix='   '):
        """Print the worker log lines written since the last call."""
        for f in sorted(self.run_dir.glob('log_*.txt')):
            off = self.offsets.get(f.name, 0)
            try:
                with open(f, 'r') as fh:
                    fh.seek(off)
                    txt = fh.read()
                    self.offsets[f.name] = fh.tell()
            except FileNotFoundError:
                continue
            for line in txt.splitlines():
                if line.strip():
                    print(prefix + line, flush=True)

    def running(self):
        return [t for t, p, _ in self.procs if p.poll() is None]

    def wait(self, poll_s=20, timeout_s=None):
        t0 = time.time()
        while self.running():
            self.pump()
            if timeout_s is not None and time.time() - t0 > timeout_s:
                self.log('worker timeout -> stopping workers')
                self.kill()
                break
            time.sleep(poll_s)
        self.pump()
        for tag, p, out in self.procs:
            out.close()
            if p.returncode not in (0, None):
                txt = (self.run_dir / f'stdout_{tag}.txt').read_text(errors='ignore')[-3000:]
                self.log(f'worker {tag} exited with code {p.returncode}; last output:\n{txt}')

    def kill(self):
        for _, p, _ in self.procs:
            if p.poll() is None:
                p.kill()


def _decode_task(src, ids, start, S, out_path):
    out = np.load(out_path, mmap_mode='r+')
    h5, jdir = _open_src(src)
    try:
        for j, iid in enumerate(ids):
            out[start + j] = decode_crop(_read_raw(h5, jdir, iid), S)
        out.flush()
    finally:
        if h5 is not None:
            h5.close()
    return len(ids)


def decode_to_memmap(src, ids, S, out_path, n_proc=4, log=print, chunk=4096):
    """Decode every crop ONCE into a disk-backed uint8 array (N, S, S, 3); all networks then read pixels, never JPEGs."""
    import concurrent.futures as cf
    import multiprocessing as mp
    out_path = Path(out_path)
    if out_path.exists():
        return np.load(out_path, mmap_mode='r')
    part = out_path.with_name(out_path.stem + '.part.npy')
    mm = np.lib.format.open_memmap(part, mode='w+', dtype=np.uint8, shape=(len(ids), S, S, 3))
    del mm
    ids = [str(i) for i in ids]
    tasks = [(a, min(a + chunk, len(ids))) for a in range(0, len(ids), chunk)]
    t0, done, nxt = time.time(), 0, 0.1
    if n_proc <= 1:
        for a, b in tasks:
            done += _decode_task(str(src), ids[a:b], a, S, str(part))
    else:
        with cf.ProcessPoolExecutor(max_workers=n_proc, mp_context=mp.get_context('spawn')) as ex:
            futs = [ex.submit(_decode_task, str(src), ids[a:b], a, S, str(part)) for a, b in tasks]
            for f in cf.as_completed(futs):
                done += f.result()
                if done / len(ids) >= nxt:
                    log(f'  decoded {done:,}/{len(ids):,} ({done / max(time.time() - t0, 1e-9):,.0f} crops/s)')
                    nxt += 0.2
    os.replace(part, out_path)
    return np.load(out_path, mmap_mode='r')


def find_external_sources(root='/kaggle/input', exclude=()):
    """Find ISIC-archive style datasets attached to the notebook: a CSV with an image-id column and a diagnosis / malignancy
    column, next to a folder with <id>.jpg files (e.g. the 'jpeg-isic2019-*' / 'jpeg-melanoma-*' Kaggle datasets)."""
    root = Path(root)
    found = []
    if not root.exists():
        return found
    for csv in list(root.glob('*/*.csv')) + list(root.glob('*/*/*.csv')):
        if any(str(e) in str(csv) for e in exclude) or 'test' in csv.name.lower() or 'sample' in csv.name.lower():
            continue
        try:
            head = pd.read_csv(csv, nrows=50)
        except Exception:
            continue
        idc = next((c for c in ['image_name', 'isic_id', 'image', 'image_id'] if c in head.columns), None)
        lab = [c for c in ['target', 'benign_malignant', 'diagnosis', 'diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'dx'] if c in head.columns]
        if idc is None or not lab:
            continue
        for d in [csv.parent / 'train', csv.parent / 'images', csv.parent / 'image', csv.parent / 'train-image' / 'image', csv.parent]:
            if d.is_dir() and sum((d / f'{i}.jpg').exists() for i in head[idc].astype(str).head(20)) >= 15:
                found.append(dict(csv=str(csv), img_dir=str(d), id_col=idc))
                break
    return found


def load_external(sources, max_per_source=None, seed=0):
    """Labels for external images: malignant = melanoma / basal cell / squamous cell carcinoma (from the diagnosis text, or target)."""
    frames = []
    for s in sources:
        df = pd.read_csv(s['csv'])
        txt = pd.Series([''] * len(df), index=df.index)
        for c in ['diagnosis', 'diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'dx', 'benign_malignant']:
            if c in df.columns:
                txt = txt.str.cat(df[c].fillna('').astype(str), sep='|')
        txt = txt.str.lower()
        mel = txt.str.contains('melanoma', regex=False) | txt.str.contains(r'\bmel\b', regex=True)
        bcc = txt.str.contains('basal cell', regex=False) | txt.str.contains(r'\bbcc\b', regex=True)
        scc = txt.str.contains('squamous', regex=False) | txt.str.contains(r'\bscc\b', regex=True)
        mal = mel | bcc | scc
        if 'target' in df.columns:
            mal = mal | (pd.to_numeric(df['target'], errors='coerce').fillna(0) == 1)
        tcode = np.full(len(df), -1, dtype=np.int8)
        tcode[np.asarray(mel)] = 0
        tcode[np.asarray(bcc & ~mel)] = 1
        tcode[np.asarray(scc & ~mel & ~bcc)] = 2
        f = pd.DataFrame(dict(id=df[s['id_col']].astype(str), path=[str(Path(s['img_dir']) / f'{i}.jpg') for i in df[s['id_col']].astype(str)],
                              y=mal.astype(int).values, type_code=tcode, source=Path(s['csv']).parent.name))
        f = f[[Path(p).exists() for p in f.path]]
        if max_per_source and len(f) > max_per_source:
            f = pd.concat([f[f.y == 1], f[f.y == 0].sample(max(0, max_per_source - int(f.y.sum())), random_state=seed)])
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=['id', 'path', 'y', 'type_code', 'source'])
    return pd.concat(frames).drop_duplicates('id').reset_index(drop=True)


def _decode_files_task(paths, start, S, out_path):
    out = np.load(out_path, mmap_mode='r+')
    for j, p in enumerate(paths):
        try:
            out[start + j] = decode_crop(Path(p).read_bytes(), S)
        except Exception:
            out[start + j] = 0
    out.flush()
    return len(paths)


def decode_files_to_memmap(paths, S, out_path, n_proc=4, log=print, chunk=2048):
    import concurrent.futures as cf
    import multiprocessing as mp
    out_path = Path(out_path)
    if out_path.exists():
        return np.load(out_path, mmap_mode='r')
    part = out_path.with_name(out_path.stem + '.part.npy')
    mm = np.lib.format.open_memmap(part, mode='w+', dtype=np.uint8, shape=(len(paths), S, S, 3))
    del mm
    paths = [str(p) for p in paths]
    tasks = [(a, min(a + chunk, len(paths))) for a in range(0, len(paths), chunk)]
    if n_proc <= 1:
        for a, b in tasks:
            _decode_files_task(paths[a:b], a, S, str(part))
    else:
        with cf.ProcessPoolExecutor(max_workers=n_proc, mp_context=mp.get_context('spawn')) as ex:
            futs = [ex.submit(_decode_files_task, paths[a:b], a, S, str(part)) for a, b in tasks]
            for f in cf.as_completed(futs):
                f.result()
    os.replace(part, out_path)
    log(f'  decoded {len(paths):,} external images')
    return np.load(out_path, mmap_mode='r')


def file_md5(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['worker'])
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='w0')
    a = ap.parse_args()
    if a.mode == 'worker':
        worker_main(a.run_dir, a.device, a.tag)


if __name__ == '__main__':
    main()