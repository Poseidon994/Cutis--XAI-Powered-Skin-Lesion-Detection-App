import { BODY_SITES, DURATIONS, SEX, SYMPTOMS, TIERS } from '../constants.js';

function daysBetween(a, b) {
  return Math.round((new Date(b).getTime() - new Date(a).getTime()) / 86400000);
}

// Earlier scans of the same lesion: same lesion label, or (no label) same patient reference and body site, or same body site.
export function relatedHistory(store, scan) {
  const label = (scan.lesionLabel || '').trim().toLowerCase();
  const ref = (scan.patientRef || '').trim().toLowerCase();
  const all = store.listScans(scan.userId, { limit: 500 });
  return all
    .filter((s) => s.id !== scan.id && s.createdAt < scan.createdAt)
    .filter((s) => {
      const sl = (s.lesionLabel || '').trim().toLowerCase();
      const sr = (s.patientRef || '').trim().toLowerCase();
      if (label) return sl === label && (!ref || sr === ref);
      if (ref) return sr === ref && s.bodySite === scan.bodySite;
      return s.bodySite === scan.bodySite && s.bodySite !== 'unknown' && !sl;
    })
    .slice(0, 6);
}

export function trendOf(scan, history) {
  if (!history.length) return { label: 'no_earlier_scans', n_scans: 1 };
  const prev = history[0];
  const delta = Number(((scan.score ?? 0) - (prev.score ?? 0)).toFixed(2));
  const tierChange = (TIERS[scan.tier]?.rank ?? 0) - (TIERS[prev.tier]?.rank ?? 0);
  let label = 'stable';
  if (delta >= 0.5 || tierChange > 0) label = 'rising';
  else if (delta <= -0.5 || tierChange < 0) label = 'falling';
  return { label, score_change_since_previous: delta, previous_tier: prev.tier, days_since_previous: daysBetween(prev.createdAt, scan.createdAt), n_scans: history.length + 1 };
}

function plainComparison(pred) {
  if (!pred || pred.source !== 'oof' || pred.percentile_benign === undefined) return null;
  const pct = Math.round(pred.percentile_benign);
  const out = { versus_harmless_spots: `The AI scored this photo higher than about ${pct} out of every 100 harmless spots it was tested on.` };
  if (pred.cancer_rate_at_or_above !== undefined && pred.base_rate !== undefined) {
    const per1000 = (pred.cancer_rate_at_or_above * 1000).toFixed(pred.cancer_rate_at_or_above * 1000 < 10 ? 1 : 0);
    const base = (pred.base_rate * 1000).toFixed(1);
    out.cancer_rate_text = `Among test photos scoring this high or higher, about ${per1000} in every 1,000 were skin cancers, compared with about ${base} in 1,000 overall.`;
  }
  return out;
}

export function buildContext({ scan, history, role }) {
  const pred = scan.prediction || {};
  const clinician = role === 'clinician';
  const xai = scan.xai || {};
  const xs = (m) => (xai[m]?.status === 'done' ? xai[m] : null);
  const explain = {};
  for (const m of ['gradcam', 'lime', 'shap']) {
    const e = xs(m);
    if (!e) continue;
    const s = e.summary || {};
    explain[m] = {
      method: e.title,
      where_the_evidence_lies: s.focus,
      direction_in_photo: s.direction,
      share_of_evidence_in_central_area: s.center_share,
      ...(m === 'gradcam' ? { agreement_between_networks: s.network_agreement } : {}),
      ...(m === 'lime' ? { local_fit_r2: s.local_r2, strongest_regions: s.top_regions } : {}),
      ...(m === 'shap' ? { share_of_attribution_pointing_to_higher_risk: s.positive_share } : {}),
    };
  }
  const result = {
    tier: pred.tier,
    tier_label: pred.tier_label,
    ...(clinician
      ? {
          score_z: pred.score,
          percentile_among_benign_lesions: pred.percentile_benign,
          sensitivity_at_or_above_this_score: pred.sensitivity_at_or_above,
          observed_cancer_rate_at_or_above: pred.cancer_rate_at_or_above,
          base_rate_in_training_data: pred.base_rate,
          enrichment_over_base_rate: pred.enrichment,
          probability_at_training_prevalence: pred.probability_at_training_prevalence,
          operating_point_thresholds: pred.thresholds,
          calibration_source: pred.source,
          per_backbone: (pred.per_backbone || []).map((b) => ({ name: b.name, z: Number(b.z.toFixed(2)), tier: b.tier })),
          network_agreement: pred.agreement,
          disagreement_between_networks_sd: pred.network_sd,
        }
      : { plain_language: plainComparison(pred) }),
    ...(clinician
      ? {
          type_distribution_if_it_is_a_skin_cancer: pred.type_probs ? Object.fromEntries(Object.entries(pred.type_probs).map(([k, v]) => [k, Number(v.toFixed(3))])) : null,
          most_likely_type_if_cancer: pred.top_type,
        }
      : {}),
    image_quality_warnings: (scan.quality?.warnings || []).map((w) => w.text),
  };
  return {
    audience: role,
    today: new Date().toISOString().slice(0, 10),
    case: {
      scan_date: scan.createdAt.slice(0, 10),
      body_site: BODY_SITES[scan.bodySite] || 'Not specified',
      lesion_label: scan.lesionLabel || null,
      age: scan.age ?? null,
      sex: scan.sex ? SEX[scan.sex] : null,
      symptoms_reported: (scan.symptoms || []).map((s) => SYMPTOMS[s] || s),
      duration: scan.duration ? DURATIONS[scan.duration] || scan.duration : null,
      free_text_notes_from_user: scan.notes || null,
    },
    tool: {
      name: 'Cutis',
      what_it_is: 'Research prototype that rates how concerning a close-up photo of a skin lesion looks, using an ensemble of image networks (four architectures, several folds each).',
      trained_on: 'ISIC 2024 (SLICE-3D): 15 x 15 mm lesion crops from 3D total-body photography, about 401,000 lesions of which 393 were cancers.',
      how_to_read_the_tier:
        'Tiers are cut at operating points on out-of-fold validation scores: low = below the 95%-sensitivity cut-off; some concern = between the 95% and 90% cut-offs; elevated = between the 90% and 80% cut-offs; high = at or above the 80% cut-off. A low tier never rules out cancer.',
      ...(clinician ? { type_distribution_note: 'The type distribution (Melanoma / BCC / SCC) is conditional on the lesion being a skin cancer; it is not the chance that it is cancer.' } : {}),
      known_limits: [
        'Image-only branch: no 3D-TBP measurements or comparison with the patient\'s other lesions.',
        'Not validated on phone photos or dermoscopy images; accuracy there is unknown.',
        clinician ? 'Rare cancer types, especially squamous cell carcinoma, are poorly represented in training data.' : 'Rarer kinds of skin cancer are poorly represented in the training data.',
      ],
    },
    result,
    explainability: { methods: explain, concordance: scan.xai?.agreement || null, how_to_read: 'Grad-CAM: brighter = more attention. LIME and SHAP: red = pushes the score up (towards higher concern), blue = pushes it down.' },
    history: history.map((h) => ({
      date: h.createdAt.slice(0, 10),
      days_before_this_scan: daysBetween(h.createdAt, scan.createdAt),
      tier: h.tier,
      score_z: clinician ? h.score : undefined,
      ...(clinician ? { most_likely_type_if_cancer: h.topType } : {}),
    })),
    trend: trendOf(scan, history),
  };
}
