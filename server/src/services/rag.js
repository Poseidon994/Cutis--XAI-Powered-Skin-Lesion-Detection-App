// Guideline retrieval for the report agent. Never throws on service problems: callers get
// { status: 'unavailable', reason } and carry on with the standard (ungrounded) text.

const DEFAULT_TIMEOUT_MS = 8000;

function topicOf(type) {
  const s = String(type ?? '').toLowerCase();
  if (s.includes('melanoma')) return 'melanoma';
  if (s.includes('basal') || s === 'bcc') return 'bcc';
  if (s.includes('squamous') || s === 'scc') return 'scc';
  return null;
}

export { topicOf };

function words(value, limit) {
  return String(value ?? '')
    .replace(/[^\p{L}\p{N}\s-]/gu, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, limit);
}

export function symptomWords(symptoms) {
  const list = Array.isArray(symptoms) ? symptoms : [symptoms];
  return list
    .flatMap((s) => words(typeof s === 'string' ? s : (s?.label ?? s?.name ?? ''), 4))
    .slice(0, 8);
}

const MANAGEMENT = {
  melanoma: 'melanoma surgical excision margins sentinel lymph node biopsy staging',
  bcc: 'basal cell carcinoma treatment options surgical excision Mohs micrographic surgery radiation',
  scc: 'cutaneous squamous cell carcinoma treatment surgical excision high-risk features radiation',
};
const MEDICATION = {
  melanoma: 'melanoma systemic therapy adjuvant immunotherapy BRAF targeted therapy',
  bcc: 'basal cell carcinoma topical imiquimod fluorouracil hedgehog pathway inhibitor vismodegib sonidegib',
  scc: 'cutaneous squamous cell carcinoma systemic therapy PD-1 inhibitor cemiplimab',
};

// Patients: only generic questions (warning signs, what a skin check involves, sun safety). The predicted type
// and the scores never go into a patient query, so what is retrieved cannot depend on them.
// Clinicians: one query per report section; the predicted type steers management and medication.
export function buildQueries({ role, tier = 'low', symptoms = [], topType = null } = {}) {
  if (role === 'patient') {
    const low = String(tier).toLowerCase() === 'low';
    return {
      queries: [
        {
          id: 'warning_signs',
          text: ['when to see a doctor about a changing mole or skin spot warning signs', ...symptomWords(symptoms)].join(' '),
          k: 3,
        },
        low
          ? { id: 'next_steps', text: 'checking your skin regularly self-examination ABCDE what to watch for', k: 2 }
          : { id: 'next_steps', text: 'what happens at a skin check skin biopsy examination of a suspicious mole', k: 3 },
        { id: 'sun_safety', text: 'protecting your skin from the sun sunscreen clothing shade', k: 2 },
      ],
      topics: [],
    };
  }
  if (role !== 'clinician') throw new TypeError(`unknown role: ${role}`);
  const type = topicOf(topType);
  const wantsMedication = /elevated|high/.test(String(tier).toLowerCase());
  const queries = [
    { id: 'workup', text: 'diagnostic work-up of a suspicious pigmented skin lesion dermoscopy biopsy technique', k: 3 },
    {
      id: 'differential',
      text: 'differential diagnosis of pigmented skin lesions melanoma basal cell carcinoma squamous cell carcinoma clinical features',
      k: 2,
    },
    { id: 'management', text: MANAGEMENT[type] ?? 'management of suspected skin cancer referral biopsy excision follow-up', k: 3 },
  ];
  if (wantsMedication) {
    queries.push({ id: 'medication', text: MEDICATION[type] ?? 'systemic and topical therapy options for skin cancer', k: 3 });
  }
  queries.push({ id: 'follow_up', text: 'follow-up and surveillance after treatment of skin cancer skin examination', k: 2 });
  return { queries, topics: type ? [type] : [] };
}

// One query per chat question: the question itself plus the case topic. Patients never get the predicted type
// in their query (their own words are their own words; the role rules and the scrubber still apply downstream).
const TYPE_TEXT = { melanoma: 'melanoma', bcc: 'basal cell carcinoma', scc: 'squamous cell carcinoma' };

export function buildChatQuery({ role, message = '', topType = null } = {}) {
  if (role !== 'clinician' && role !== 'patient') throw new TypeError(`unknown role: ${role}`);
  const text = words(message, 40).join(' ').trim();
  if (role === 'patient') {
    return { queries: [{ id: 'chat', text: text || 'what this skin check result means', k: 3 }], topics: [] };
  }
  const type = topicOf(topType);
  const typeText = type ? TYPE_TEXT[type] : '';
  return {
    queries: [{ id: 'chat', text: [text, typeText].filter(Boolean).join(' ') || 'management of a suspicious skin lesion', k: 3 }],
    topics: type ? [type] : [],
  };
}

function unavailable(reason) {
  return { status: 'unavailable', reason, embedder: null, dense: false, corpus_sha: null, built_at: null, passages: [], sections: {} };
}

export async function retrieve({
  role,
  queries,
  topics = [],
  maxPassages,
  baseUrl = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000',
  key = process.env.ML_SERVICE_KEY || '',
  timeoutMs = Number(process.env.RAG_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS,
  fetchImpl = globalThis.fetch,
} = {}) {
  if (role !== 'clinician' && role !== 'patient') throw new TypeError(`unknown role: ${role}`);
  if (process.env.RAG_ENABLED === '0') return unavailable('disabled');
  try {
    const res = await fetchImpl(`${String(baseUrl).replace(/\/$/, '')}/rag/retrieve`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-service-key': key },
      body: JSON.stringify({ audience: role, queries, topics, ...(maxPassages ? { max_passages: maxPassages } : {}) }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return unavailable(`ml service answered ${res.status}`);
    const data = await res.json();
    if (!data || !Array.isArray(data.passages)) return unavailable('unexpected response');
    return {
      status: data.status ?? (data.passages.length ? 'ok' : 'empty'),
      reason: data.reason ?? null,
      embedder: data.embedder ?? null,
      dense: Boolean(data.dense),
      corpus_sha: data.corpus_sha ?? null,
      built_at: data.built_at ?? null,
      passages: data.passages,
      sections: data.sections ?? {},
    };
  } catch (err) {
    return unavailable(err?.name === 'TimeoutError' ? 'timeout' : `request failed: ${err?.message ?? err}`);
  }
}
