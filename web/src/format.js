export const TIERS = {
  low: { label: 'Low concern', rank: 0 },
  guarded: { label: 'Some concern', rank: 1 },
  elevated: { label: 'Elevated concern', rank: 2 },
  high: { label: 'High concern', rank: 3 },
};

export const BODY_SITES = [
  ['head_neck', 'Head or neck'],
  ['upper_extremity', 'Arm or hand'],
  ['lower_extremity', 'Leg or foot'],
  ['anterior_torso', 'Chest or abdomen'],
  ['posterior_torso', 'Back'],
  ['other', 'Other'],
  ['unknown', 'Not sure'],
];
export const siteLabel = (k) => (BODY_SITES.find(([v]) => v === k) || [0, 'Not specified'])[1];

export const SYMPTOMS = [
  ['itching', 'Itching'],
  ['bleeding', 'Bleeding'],
  ['crusting', 'Crusting or scabbing'],
  ['pain', 'Pain or tenderness'],
  ['recent_growth', 'Recent growth'],
  ['colour_change', 'Colour change'],
  ['shape_change', 'Shape or border change'],
  ['new_lesion', 'New spot'],
  ['non_healing', 'Sore that will not heal'],
];
export const symptomLabel = (k) => (SYMPTOMS.find(([v]) => v === k) || [0, k])[1];

export const DURATIONS = [
  ['lt_1m', 'Less than 1 month'],
  ['m1_6', '1 to 6 months'],
  ['m6_12', '6 to 12 months'],
  ['gt_1y', 'More than 1 year'],
  ['unknown', 'Not sure'],
];
export const durationLabel = (k) => (DURATIONS.find(([v]) => v === k) || [0, ''])[1];

export const SEX = [
  ['female', 'Female'],
  ['male', 'Male'],
  ['other', 'Other'],
  ['unspecified', 'Prefer not to say'],
];
export const sexLabel = (k) => (SEX.find(([v]) => v === k) || [0, ''])[1];

export const fmtDate = (iso) => new Date(iso).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
export const fmtDateTime = (iso) => new Date(iso).toLocaleString(undefined, { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
export const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
export const firstName = (n) => (n || '').trim().split(/\s+/)[0] || 'there';
