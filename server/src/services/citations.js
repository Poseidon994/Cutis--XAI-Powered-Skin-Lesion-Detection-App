// Turns retrieved passages into a prompt block, and checks the model's answer against them.
// The model is asked to mark each supported statement with [P1], [P2], ... ; this code, not the model,
// decides which markers are real, which statements count as grounded, and what is shown as a source.

const MARK = /\[\s*(P\d{1,2}(?:\s*[,;]\s*P\d{1,2})*)\s*\]/g;
const LABEL = /P\d{1,2}/g;

export const RAG_RULES = [
  'You are given PASSAGES, numbered [P1], [P2], ... They are excerpts from public clinical guidance.',
  'Support every statement about work-up, treatment, medicines or follow-up with the passage that says it: put its marker at the end of that statement, for example "Margins depend on thickness [P2]."',
  'Use only markers that appear in the PASSAGES. Never invent a marker, a drug, a dose or a figure that the passages do not contain.',
  'If the passages do not cover something, leave it out or say the supplied sources do not cover it. Do not fill the gap from memory.',
  'Put the markers inside the existing text fields. Do not add new fields for them.',
].join('\n');

export const PATIENT_RAG_RULES = [
  RAG_RULES,
  'This is for a patient: never name a specific type of skin cancer and never mention medicines or treatments. Say "skin cancer" or "a skin problem" and use the passages only for warning signs, what a skin check involves, and sun safety.',
].join('\n');

export function passagesBlock(passages) {
  if (!passages?.length) return '';
  const body = passages
    .map((p) => `[${p.label}] ${p.title} (${p.publisher})${p.section ? ` - ${p.section}` : ''}\n${p.text}`)
    .join('\n\n');
  return `PASSAGES\n${body}`;
}

const FIELDS = {
  clinician: {
    cited: ['supporting_evidence', 'differential', 'recommended_workup', 'management_options', 'medication_options'],
    strict: ['medication_options'],
  },
  patient: {
    cited: ['next_steps', 'see_a_doctor_sooner_if', 'keep_an_eye_on_it'],
    strict: [],
  },
};

// Inside a strict field's item, these string lists hold individual statements that must each carry a marker.
const NESTED_STRICT = { medication_options: ['agents'] };

function mapStrings(value, fn) {
  if (typeof value === 'string') return fn(value);
  if (Array.isArray(value)) return value.map((v) => mapStrings(v, fn));
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, mapStrings(v, fn)]));
  }
  return value;
}

function joinStrings(value) {
  const parts = [];
  mapStrings(value, (s) => {
    parts.push(s);
    return s;
  });
  return parts.join(' ');
}

export function citedLabels(text, valid) {
  const found = [];
  for (const m of String(text).matchAll(MARK)) {
    for (const label of m[1].match(LABEL)) if (valid.has(label) && !found.includes(label)) found.push(label);
  }
  return found;
}

function sourceOf(p, { excerpt = true } = {}) {
  const out = {
    label: p.label,
    chunk_id: p.chunk_id,
    source_id: p.source_id,
    title: p.title,
    publisher: p.publisher,
    url: p.url ?? null,
    licence: p.licence ?? null,
    section: p.section ?? '',
    page: p.page ?? null,
  };
  // Patients never see cancer-type names anywhere in their report JSON; the excerpt of a patient-audience
  // page can contain them, and the patient UI does not show excerpts, so none is stored for that role.
  if (excerpt) out.excerpt = p.text.length > 600 ? `${p.text.slice(0, 597)}...` : p.text;
  return out;
}

function normaliseMarkers(text, valid) {
  if (!text.includes('[')) return text;
  let changed = false;
  const out = text.replace(MARK, (_m, inner) => {
    changed = true;
    const labels = [...new Set(inner.match(LABEL))].filter((l) => valid.has(l));
    return labels.map((l) => `[${l}]`).join('');
  });
  if (!changed) return text;
  return out.replace(/\s+([.,;:!?])/g, '$1').replace(/[ \t]{2,}/g, ' ').trim();
}

// Patients never see a named cancer type (the app's rule). Retrieved education text names them,
// so the names are replaced in code instead of trusting the model to avoid them.
const TYPE_NAMES = /\b(?:non-?melanoma(?: skin cancers?)?|melanomas?|basal[- ]cell(?: carcinomas?| cancers?| skin cancers?)?|squamous[- ]cell(?: carcinomas?| cancers?| skin cancers?)?|c?BCC|c?SCC)\b/gi;

export function scrubTypeNames(text) {
  let count = 0;
  const out = text
    .replace(TYPE_NAMES, (_match, offset, whole) => {
      count += 1;
      return /(^|[.!?]\s+|\n)$/.test(whole.slice(0, offset)) ? 'Skin cancer' : 'skin cancer';
    })
    .replace(/\b(skin cancer)(?:\s+skin cancer)+/gi, '$1');
  return { text: out, count };
}

export function validateCitations(report, passages, { role, onUnavailable = 'keep' } = {}) {
  const cfg = FIELDS[role];
  if (!cfg) throw new TypeError(`unknown role: ${role}`);
  const valid = new Set((passages ?? []).map((p) => p.label));
  const unavailable = valid.size === 0;
  const out = structuredClone(report ?? {});
  const byField = {};
  const dropped = [];
  let scrubbed = 0;

  const clean = (s) => {
    let text = normaliseMarkers(s, valid);
    if (role === 'patient') {
      const r = scrubTypeNames(text);
      scrubbed += r.count;
      text = r.text;
    }
    return text;
  };

  for (const field of cfg.cited) {
    if (out[field] === undefined || out[field] === null) continue;
    const isArray = Array.isArray(out[field]);
    const kept = [];
    let items = 0;
    let cited = 0;
    for (let item of isArray ? out[field] : [out[field]]) {
      const text = joinStrings(item);
      if (!text.trim()) {
        kept.push(item);
        continue;
      }
      const grounded = citedLabels(text, valid).length > 0;
      let dropIt = cfg.strict.includes(field) && !grounded && !(unavailable && onUnavailable === 'keep');
      // Strict statement lists nested inside an item (the agents of a medication scenario): each named
      // medicine must carry a marker itself, or that statement is dropped; a scenario with nothing left goes too.
      if (!dropIt && !unavailable && item && typeof item === 'object' && !Array.isArray(item)) {
        for (const key of NESTED_STRICT[field] || []) {
          if (!Array.isArray(item[key])) continue;
          for (const v of item[key]) {
            if (typeof v === 'string' && v.trim() && citedLabels(v, valid).length === 0) {
              dropped.push({ field, text: v.slice(0, 200) });
            }
          }
          const survivors = item[key].filter((v) => typeof v !== 'string' || !v.trim() || citedLabels(v, valid).length > 0);
          if (!survivors.length && item[key].length) dropIt = true;
          else item = { ...item, [key]: survivors };
        }
      }
      if (dropIt) {
        if (!dropped.some((d) => d.field === field && d.text === text.slice(0, 200))) dropped.push({ field, text: text.slice(0, 200) });
        continue;
      }
      items += 1;
      if (grounded) cited += 1;
      kept.push(mapStrings(item, clean));
    }
    out[field] = isArray ? kept : (kept[0] ?? '');
    byField[field] = { items, cited };
  }

  for (const key of Object.keys(out)) {
    if (!cfg.cited.includes(key)) out[key] = mapStrings(out[key], clean);
  }

  const used = new Set();
  mapStrings(out, (s) => {
    for (const label of citedLabels(s, valid)) used.add(label);
    return s;
  });
  const sources = (passages ?? []).filter((p) => used.has(p.label)).map((p) => sourceOf(p, { excerpt: role === 'clinician' }));

  const items = Object.values(byField).reduce((n, f) => n + f.items, 0);
  const cited = Object.values(byField).reduce((n, f) => n + f.cited, 0);
  let status = 'grounded';
  if (unavailable) status = 'unavailable';
  else if (items === 0) status = 'no_claims';
  else if (cited === 0) status = 'ungrounded';
  else if (cited < items) status = 'partial';

  return {
    report: out,
    sources,
    grounding: { status, items, cited, ratio: items ? cited / items : null, by_field: byField, dropped, scrubbed },
  };
}

// The same check for a single chat answer: markers validated, patient type names scrubbed, sources listed.
export function validateChatCitations(text, passages, { role } = {}) {
  if (role !== 'clinician' && role !== 'patient') throw new TypeError(`unknown role: ${role}`);
  const valid = new Set((passages ?? []).map((p) => p.label));
  let out = String(text ?? '');
  if (out.includes('[')) out = normaliseMarkers(out, valid);
  let scrubbed = 0;
  if (role === 'patient') {
    const r = scrubTypeNames(out);
    scrubbed = r.count;
    out = r.text;
  }
  const used = new Set(citedLabels(out, valid));
  const sources = (passages ?? []).filter((p) => used.has(p.label)).map((p) => sourceOf(p, { excerpt: role === 'clinician' }));
  const status = valid.size === 0 ? 'unavailable' : used.size ? 'grounded' : 'ungrounded';
  return { text: out, sources, grounding: { status, cited: used.size, scrubbed } };
}

// Removes every [P#] marker from a report (used when there are no passages to cite against, so stray
// markers a model left behind can never reach the user without a source list to point at).
export function stripMarkers(value) {
  return mapStrings(value, (s) => normaliseMarkers(String(s), new Set()));
}
