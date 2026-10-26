"""Turns the image-only ensemble score into something interpretable.

The training notebook calibrates the *fused* score (tabular + image + patient context). A single uploaded photo has no
patient context, so the web app uses the image branch alone. Its score is the mean (over backbones) of the z-scored
malignant logit, exactly the quantity stored per lesion as `s_img_mean` in oof_predictions.csv.gz. This module reads that
out-of-fold file and derives, for the image-only score:
  * operating points (the thresholds that keep 95 / 90 / 80 % of the cancers), reusing the notebook's own picker,
  * the percentile among benign lesions,
  * the observed cancer rate among out-of-fold lesions scoring at least as high (enrichment over the base rate),
  * a Platt calibration at the training-set prevalence (~0.1 %), reported for clinicians only.
Tiers are defined by the operating points, so they carry a known meaning (e.g. "at or above the 90 % sensitivity point").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TIERS = ['low', 'guarded', 'elevated', 'high']
TIER_LABELS = {'low': 'Low concern', 'guarded': 'Some concern', 'elevated': 'Elevated concern', 'high': 'High concern'}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


@dataclass
class ImageCalibration:
    source: str
    thresholds: dict                       # {'sens95': t, 'sens90': t, 'sens80': t}
    benign_sorted: np.ndarray = field(default_factory=lambda: np.array([]))
    malignant_sorted: np.ndarray = field(default_factory=lambda: np.array([]))
    all_sorted_desc: np.ndarray = field(default_factory=lambda: np.array([]))
    all_labels_by_desc: np.ndarray = field(default_factory=lambda: np.array([]))
    platt: tuple = (1.0, -7.0)
    bb_thresholds: dict = field(default_factory=dict)
    n: int = 0
    n_malignant: int = 0
    metrics: dict = field(default_factory=dict)
    class_medians: dict = field(default_factory=dict)

    @classmethod
    def fallback(cls) -> 'ImageCalibration':
        return cls(source='fallback', thresholds={'sens95': 0.8, 'sens90': 1.4, 'sens80': 2.2})

    @classmethod
    def from_oof(cls, path: Path, lib, score_col: str = 's_img_mean') -> 'ImageCalibration':
        import pandas as pd
        df = pd.read_csv(path, usecols=lambda c: c in ('target', 'label', score_col) or c.startswith('s_img_'))
        if score_col not in df.columns or 'target' not in df.columns:
            raise ValueError(f'{path.name} has no {score_col} / target columns')
        df = df.dropna(subset=[score_col, 'target'])
        s = df[score_col].to_numpy(np.float64)
        y = df['target'].to_numpy().astype(int)
        if y.sum() < 20:
            raise ValueError('too few malignant lesions in the out-of-fold file to calibrate')
        th = {f'sens{int(r * 100)}': float(lib.pick_threshold(y, s, ('sens', r))) for r in (0.95, 0.90, 0.80)}
        order = np.argsort(-s)
        platt = (1.0, -7.0)
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1e6, max_iter=2000).fit(s.reshape(-1, 1), y)
            platt = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        except Exception:  # noqa: BLE001
            pass
        medians = {}
        if 'label' in df.columns:
            for k, g in df.groupby('label')[score_col]:
                medians[str(k)] = float(np.median(g))
        metrics = {'auroc': float(lib.auroc(y, s)), 'pauc_tpr80': float(lib.pauc_above_tpr(y, s, 0.80))}
        bb_thr = {}
        for c in df.columns:
            if c.startswith('s_img_') and c != score_col:
                sb = df[c].to_numpy(np.float64)
                ok = np.isfinite(sb)
                if ok.sum() > 100 and y[ok].sum() >= 20:
                    bb_thr[c[len('s_img_'):]] = {f'sens{int(r * 100)}': float(lib.pick_threshold(y[ok], sb[ok], ('sens', r))) for r in (0.95, 0.90, 0.80)}
        return cls(source='oof', thresholds=th, benign_sorted=np.sort(s[y == 0]), malignant_sorted=np.sort(s[y == 1]),
                   all_sorted_desc=s[order], all_labels_by_desc=y[order], platt=platt, bb_thresholds=bb_thr, n=len(s), n_malignant=int(y.sum()),
                   metrics=metrics, class_medians=medians)

    def tier_of(self, z: float) -> str:
        t = self.thresholds
        if z >= t['sens80']:
            return 'high'
        if z >= t['sens90']:
            return 'elevated'
        if z >= t['sens95']:
            return 'guarded'
        return 'low'

    def backbone_tier(self, name: str, z: float) -> str | None:
        t = self.bb_thresholds.get(name)
        if not t:
            return None
        if z >= t['sens80']:
            return 'high'
        if z >= t['sens90']:
            return 'elevated'
        if z >= t['sens95']:
            return 'guarded'
        return 'low'

    def operating_points(self) -> list[dict]:
        out = []
        for k, label in (('sens95', 'keeps 95% of cancers'), ('sens90', 'keeps 90% of cancers'), ('sens80', 'keeps 80% of cancers')):
            t = self.thresholds[k]
            spec = float(np.searchsorted(self.benign_sorted, t, side='left') / len(self.benign_sorted)) if len(self.benign_sorted) else None
            out.append({'rule': k, 'label': label, 'threshold': round(t, 3), 'specificity': None if spec is None else round(spec, 4)})
        return out

    def describe(self, z: float) -> dict:
        z = float(z)
        tier = self.tier_of(z)
        out = {'score': round(z, 3), 'tier': tier, 'tier_label': TIER_LABELS[tier], 'tier_rank': TIERS.index(tier),
               'source': self.source, 'thresholds': {k: round(v, 3) for k, v in self.thresholds.items()}}
        if self.source == 'oof' and len(self.benign_sorted):
            pct = float(np.searchsorted(self.benign_sorted, z, side='right') / len(self.benign_sorted) * 100)
            sens_ge = float(1 - np.searchsorted(self.malignant_sorted, z, side='left') / len(self.malignant_sorted))
            # observed cancer rate among out-of-fold lesions scoring at least this high (smoothed: at least 200 lesions)
            k = int(np.sum(self.all_sorted_desc >= z))
            k_eff = min(max(k, 200), len(self.all_sorted_desc))
            rate = float(self.all_labels_by_desc[:k_eff].mean())
            base = self.n_malignant / self.n
            a, b = self.platt
            p = 1 / (1 + math.exp(-(a * z + b)))
            out.update({
                'percentile_benign': round(pct, 1),
                'sensitivity_at_or_above': round(sens_ge, 4),
                'cancer_rate_at_or_above': round(rate, 5),
                'base_rate': round(base, 5),
                'enrichment': round(rate / base, 1) if base > 0 else None,
                'enrichment_n': k_eff,
                'probability_at_training_prevalence': round(p, 5),
            })
        else:
            out['percentile_benign'] = round(_norm_cdf(z) * 100, 1)
            out['note'] = 'Calibration file (oof_predictions.csv.gz) not found; tiers and percentiles are approximate.'
        return out

    def summary(self) -> dict:
        return {'source': self.source, 'n_lesions': self.n, 'n_malignant': self.n_malignant, 'metrics': self.metrics,
                'operating_points': self.operating_points() if self.source == 'oof' else [], 'class_median_score': self.class_medians}
