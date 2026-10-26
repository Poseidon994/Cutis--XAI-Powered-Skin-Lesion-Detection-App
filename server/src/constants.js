export const BODY_SITES = {
  head_neck: 'Head or neck',
  upper_extremity: 'Arm or hand',
  lower_extremity: 'Leg or foot',
  anterior_torso: 'Chest or abdomen',
  posterior_torso: 'Back',
  other: 'Other',
  unknown: 'Not specified',
};

export const SYMPTOMS = {
  itching: 'Itching',
  bleeding: 'Bleeding',
  crusting: 'Crusting or scabbing',
  pain: 'Pain or tenderness',
  recent_growth: 'Recent growth',
  colour_change: 'Colour change',
  shape_change: 'Shape or border change',
  new_lesion: 'New spot (was not there before)',
  non_healing: 'Sore that does not heal',
};

export const DURATIONS = {
  lt_1m: 'Less than 1 month',
  m1_6: '1 to 6 months',
  m6_12: '6 to 12 months',
  gt_1y: 'More than 1 year',
  unknown: 'Not sure',
};

export const SEX = { female: 'Female', male: 'Male', other: 'Other', unspecified: 'Not specified' };

export const TIERS = {
  low: { rank: 0, label: 'Low concern' },
  guarded: { rank: 1, label: 'Some concern' },
  elevated: { rank: 2, label: 'Elevated concern' },
  high: { rank: 3, label: 'High concern' },
};

// Deterministic guidance that the language model can add to but never replace or soften.
export const PATIENT_URGENCY = {
  low: {
    level: 'routine',
    timeframe: 'Routine',
    text: 'This result does not suggest an urgent visit. Photograph the spot again in 4 weeks and mention it at your next regular skin check. See a doctor sooner if it changes, bleeds or worries you.',
  },
  guarded: {
    level: 'soon',
    timeframe: 'Within 1 to 2 months',
    text: 'Please have this spot looked at by a doctor within the next 1 to 2 months, and photograph it again in 4 weeks.',
  },
  elevated: {
    level: 'prompt',
    timeframe: 'Within 2 to 4 weeks',
    text: 'Please book a skin check with a doctor or dermatologist within the next 2 to 4 weeks.',
  },
  high: {
    level: 'urgent',
    timeframe: 'As soon as you can, ideally within 1 to 2 weeks',
    text: 'Please see a doctor or dermatologist as soon as you can, ideally within 1 to 2 weeks. Do not wait to see whether the spot changes.',
  },
};

export const CLINICIAN_DISPOSITION = {
  low: 'Below the 95%-sensitivity operating point. Routine surveillance with clinical and dermoscopic correlation; a low score does not exclude malignancy.',
  guarded: 'Between the 95% and 90% sensitivity operating points. Dermoscopic review; short-interval digital monitoring (about 3 months) or biopsy if clinically or dermoscopically atypical.',
  elevated: 'At or above the 90%-sensitivity operating point. Expedited specialist review with dermoscopy and a low threshold for biopsy or excision.',
  high: 'At or above the 80%-sensitivity operating point (the strictest cut-off). Urgent specialist review; biopsy or excision for histopathology is usually warranted unless dermoscopy is clearly benign.',
};

export const DISCLAIMER = {
  patient:
    'Cutis is a research prototype. It cannot diagnose skin cancer and it can miss cancers, including when the result is low. Only a doctor, and usually a skin sample (biopsy), can tell you what a spot is. If you are worried about a spot, see a doctor whatever this result says.',
  clinician:
    'Cutis is a research prototype and not a medical device. Outputs are decision support for a qualified clinician, are not a diagnosis, and must be correlated with history, examination and dermoscopy. Histopathology is the reference standard. Management options listed are general and must be checked against current local guidelines.',
};

export const XAI_METHODS = ['gradcam', 'lime', 'shap'];
export const FILE_NAMES = ['original', 'model_view', 'gradcam', 'lime', 'shap'];
