import test from 'node:test';
import assert from 'node:assert/strict';
import { buildChatQuery, buildQueries, retrieve, symptomWords } from '../src/services/rag.js';
import {
  passagesBlock,
  scrubTypeNames,
  stripMarkers,
  validateCitations,
  validateChatCitations,
  citedLabels,
  RAG_RULES,
  PATIENT_RAG_RULES,
} from '../src/services/citations.js';

const passages = [
  { label: 'P1', chunk_id: 'S01-001', source_id: 'S01', title: 'Melanoma Treatment', publisher: 'NCI', url: 'https://example.org/s01', licence: 'free', section: 'Surgery', page: null, text: 'Margins depend on tumour thickness.' },
  { label: 'P2', chunk_id: 'S02-004', source_id: 'S02', title: 'Skin Cancer Treatment', publisher: 'NCI', url: null, licence: 'free', section: '', page: 3, text: 'x'.repeat(700) },
  { label: 'P3', chunk_id: 'S03-002', source_id: 'S03', title: 'Never cited', publisher: 'NCI', url: null, licence: 'free', section: '', page: null, text: 'unused' },
];

// ---- queries ----------------------------------------------------------------------------------------------------

test('patient queries never mention a cancer type or scores, whatever the prediction was', () => {
  const { queries, topics } = buildQueries({ role: 'patient', tier: 'high', topType: 'Melanoma', symptoms: ['bleeding', 'itching!'] });
  assert.deepEqual(topics, []);
  const all = queries.map((q) => q.text).join(' ').toLowerCase();
  for (const word of ['melanoma', 'basal', 'squamous', 'carcinoma']) assert.ok(!all.includes(word), word);
  assert.ok(all.includes('bleeding') && all.includes('itching') && !all.includes('!'));
  assert.deepEqual(queries.map((q) => q.id), ['warning_signs', 'next_steps', 'sun_safety']);
});

test('patient next-step query depends on the tier only', () => {
  const low = buildQueries({ role: 'patient', tier: 'low' }).queries.find((q) => q.id === 'next_steps').text;
  const high = buildQueries({ role: 'patient', tier: 'high' }).queries.find((q) => q.id === 'next_steps').text;
  assert.notEqual(low, high);
  assert.ok(high.includes('biopsy'));
});

test('clinician queries follow the predicted type and only ask about medicines at elevated or high tiers', () => {
  const mel = buildQueries({ role: 'clinician', tier: 'high', topType: 'Melanoma' });
  assert.deepEqual(mel.topics, ['melanoma']);
  assert.ok(mel.queries.find((q) => q.id === 'management').text.includes('sentinel'));
  assert.ok(mel.queries.some((q) => q.id === 'medication'));

  const bcc = buildQueries({ role: 'clinician', tier: 'elevated', topType: 'BCC' });
  assert.deepEqual(bcc.topics, ['bcc']);
  assert.ok(bcc.queries.find((q) => q.id === 'medication').text.includes('imiquimod'));

  const low = buildQueries({ role: 'clinician', tier: 'low', topType: null });
  assert.deepEqual(low.topics, []);
  assert.ok(!low.queries.some((q) => q.id === 'medication'));
  assert.ok(low.queries.length <= 8 && new Set(low.queries.map((q) => q.id)).size === low.queries.length);
});

test('unknown roles are rejected', () => {
  assert.throws(() => buildQueries({ role: 'admin' }), TypeError);
});

test('symptom words are sanitised and capped', () => {
  assert.deepEqual(symptomWords(['a sore that will not heal at all!']), ['a', 'sore', 'that', 'will']);
  assert.equal(symptomWords('x y z; DROP TABLE').join(' '), 'x y z DROP');
  assert.equal(symptomWords(Array(20).fill('itch')).length, 8);
  assert.deepEqual(symptomWords(undefined), []);
});

// ---- retrieval client -------------------------------------------------------------------------------------------

const okResponse = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });

test('retrieve posts the audience, queries and key, and passes the answer through', async () => {
  let seen;
  const fetchImpl = async (url, init) => {
    seen = { url, init };
    return okResponse({ status: 'ok', embedder: 'bge', dense: true, passages, sections: { a: ['P1'] } });
  };
  const out = await retrieve({ role: 'clinician', queries: [{ id: 'a', text: 'melanoma margins' }], topics: ['melanoma'], maxPassages: 4, baseUrl: 'http://ml:8000/', key: 'k', fetchImpl });
  assert.equal(seen.url, 'http://ml:8000/rag/retrieve');
  assert.equal(seen.init.method, 'POST');
  assert.equal(seen.init.headers['x-service-key'], 'k');
  assert.deepEqual(JSON.parse(seen.init.body), { audience: 'clinician', queries: [{ id: 'a', text: 'melanoma margins' }], topics: ['melanoma'], max_passages: 4 });
  assert.equal(out.status, 'ok');
  assert.equal(out.passages.length, 3);
  assert.equal(out.dense, true);
});

test('retrieve never throws on service trouble: it reports unavailable with a reason', async () => {
  const args = { role: 'patient', queries: [{ id: 'a', text: 'when to see a doctor' }] };
  const http500 = await retrieve({ ...args, fetchImpl: async () => okResponse({}, 503) });
  assert.deepEqual([http500.status, http500.passages.length], ['unavailable', 0]);
  assert.match(http500.reason, /503/);

  const badBody = await retrieve({ ...args, fetchImpl: async () => okResponse({ nothing: true }) });
  assert.equal(badBody.status, 'unavailable');

  const refused = await retrieve({ ...args, fetchImpl: async () => { throw new Error('ECONNREFUSED'); } });
  assert.match(refused.reason, /ECONNREFUSED/);

  const slow = await retrieve({
    ...args,
    timeoutMs: 20,
    fetchImpl: (_url, { signal }) =>
      new Promise((_res, rej) => {
        const keepAlive = setTimeout(() => {}, 2000);
        signal.addEventListener('abort', () => {
          clearTimeout(keepAlive);
          rej(signal.reason);
        });
      }),
  });
  assert.equal(slow.reason, 'timeout');
});

test('retrieve reports an empty library answer as empty, and can be switched off', async () => {
  const args = { role: 'clinician', queries: [{ id: 'a', text: 'pizza dough recipe' }] };
  const empty = await retrieve({ ...args, fetchImpl: async () => okResponse({ status: 'empty', reason: 'no relevant passages found', passages: [], sections: { a: [] } }) });
  assert.equal(empty.status, 'empty');
  process.env.RAG_ENABLED = '0';
  try {
    let called = false;
    const off = await retrieve({ ...args, fetchImpl: async () => { called = true; return okResponse({}); } });
    assert.equal(off.reason, 'disabled');
    assert.equal(called, false);
  } finally {
    delete process.env.RAG_ENABLED;
  }
  await assert.rejects(() => retrieve({ role: 'admin', queries: [] }), TypeError);
});

// ---- prompt block -----------------------------------------------------------------------------------------------

test('the prompt block lists every passage under its label, and is empty without passages', () => {
  const block = passagesBlock(passages.slice(0, 2));
  assert.ok(block.startsWith('PASSAGES\n[P1] Melanoma Treatment (NCI) - Surgery\nMargins depend'));
  assert.ok(block.includes('[P2] Skin Cancer Treatment (NCI)\n'));
  assert.equal(passagesBlock([]), '');
  assert.ok(RAG_RULES.includes('Never invent a marker'));
  assert.ok(PATIENT_RAG_RULES.includes('never name a specific type of skin cancer'));
});

// ---- citation validation ----------------------------------------------------------------------------------------

const clinicianReport = () => ({
  summary: 'Evidence [P1, P2] and [P9].',
  management_options: ['Wide excision with margins by thickness [P1].', 'Radiotherapy may be used [P7].', 'Close follow-up.'],
  medication_options: ['Topical therapy for superficial disease [P2].', 'A drug nobody supplied a source for.'],
  disclaimer: 'Fixed text.',
});

test('markers are normalised, unknown ones removed, and uncited medicines dropped', () => {
  const { report, grounding } = validateCitations(clinicianReport(), passages, { role: 'clinician' });
  assert.ok(report.summary.includes('[P1][P2]') && !report.summary.includes('P9'));
  assert.deepEqual(report.management_options, ['Wide excision with margins by thickness [P1].', 'Radiotherapy may be used.', 'Close follow-up.']);
  assert.deepEqual(report.medication_options, ['Topical therapy for superficial disease [P2].']);
  assert.equal(report.disclaimer, 'Fixed text.');
  assert.deepEqual(grounding.dropped, [{ field: 'medication_options', text: 'A drug nobody supplied a source for.' }]);
  assert.deepEqual(grounding.by_field.management_options, { items: 3, cited: 1 });
  assert.deepEqual(grounding.by_field.medication_options, { items: 1, cited: 1 });
  assert.equal(grounding.status, 'partial');
  assert.equal(grounding.ratio, 2 / 4);
});

test('sources lists only passages the final report actually cites, with an excerpt', () => {
  const { sources } = validateCitations(clinicianReport(), passages, { role: 'clinician' });
  assert.deepEqual(sources.map((s) => s.label), ['P1', 'P2']);
  assert.equal(sources[0].url, 'https://example.org/s01');
  assert.equal(sources[0].excerpt, 'Margins depend on tumour thickness.');
  assert.ok(sources[1].excerpt.length <= 600 && sources[1].excerpt.endsWith('...'));
  assert.equal(sources[1].page, 3);
});

test('the input report is not modified', () => {
  const input = clinicianReport();
  const copy = structuredClone(input);
  validateCitations(input, passages, { role: 'clinician' });
  assert.deepEqual(input, copy);
});

test('without passages every marker is stripped and the report is flagged, not blocked', () => {
  const kept = validateCitations(clinicianReport(), [], { role: 'clinician' });
  assert.equal(kept.grounding.status, 'unavailable');
  assert.ok(!JSON.stringify(kept.report).includes('[P'));
  assert.equal(kept.report.medication_options.length, 2);
  assert.deepEqual(kept.sources, []);

  const strict = validateCitations(clinicianReport(), [], { role: 'clinician', onUnavailable: 'drop_strict' });
  assert.deepEqual(strict.report.medication_options, []);
  assert.equal(strict.grounding.dropped.length, 2);
});

test('statements the model left uncited are counted as ungrounded', () => {
  const { grounding } = validateCitations({ management_options: ['No marker here.'] }, passages, { role: 'clinician' });
  assert.equal(grounding.status, 'ungrounded');
  assert.equal(validateCitations({ summary: 'only prose' }, passages, { role: 'clinician' }).grounding.status, 'no_claims');
  const full = validateCitations({ management_options: ['A [P1].', 'B [P2].'] }, passages, { role: 'clinician' });
  assert.equal(full.grounding.status, 'grounded');
});

test('patient reports have cancer type names replaced in code, markers kept, and object items handled', () => {
  const report = {
    next_steps: [{ step: 'Book a skin check: ABCDE signs of melanoma are described in the guidance [P1].', when: 'soon' }],
    see_a_doctor_sooner_if: 'A basal cell carcinoma or a squamous cell carcinoma can bleed [P2].',
    keep_an_eye_on_it: 'Non-melanoma skin cancer can be slow growing; check your skin monthly.',
    headline: 'Your result',
  };
  const { report: out, grounding, sources } = validateCitations(report, passages, { role: 'patient' });
  assert.equal(out.next_steps[0].step, 'Book a skin check: ABCDE signs of skin cancer are described in the guidance [P1].');
  assert.equal(out.next_steps[0].when, 'soon');
  assert.equal(out.see_a_doctor_sooner_if, 'A skin cancer or a skin cancer can bleed [P2].');
  assert.equal(out.keep_an_eye_on_it, 'Skin cancer can be slow growing; check your skin monthly.');
  assert.equal(grounding.scrubbed, 4);
  assert.deepEqual(sources.map((s) => s.label), ['P1', 'P2']);
  assert.equal(scrubTypeNames('cSCC and BCC and Melanomas').text, 'Skin cancer and skin cancer and skin cancer');
  assert.equal(scrubTypeNames('It is rare. Melanoma is serious.').text, 'It is rare. Skin cancer is serious.');
});

test('helpers behave on odd input', () => {
  assert.deepEqual(citedLabels('a [P1, P2] b [P2] c [P12]', new Set(['P1', 'P2'])), ['P1', 'P2']);
  assert.deepEqual(citedLabels('no markers', new Set(['P1'])), []);
  assert.throws(() => validateCitations({}, [], { role: 'admin' }), TypeError);
  assert.equal(validateCitations(undefined, passages, { role: 'patient' }).grounding.status, 'no_claims');
  const single = validateCitations({ medication_options: 'Single string without a source' }, passages, { role: 'clinician' });
  assert.equal(single.report.medication_options, '');
});

// ---- chat queries -----------------------------------------------------------------------------------------------

test('chat queries use the question itself, and only clinicians add the predicted type', () => {
  const patient = buildChatQuery({ role: 'patient', message: 'Is bleeding normal??', topType: 'Melanoma' });
  assert.deepEqual(patient.topics, []);
  assert.equal(patient.queries.length, 1);
  assert.equal(patient.queries[0].id, 'chat');
  assert.equal(patient.queries[0].text, 'Is bleeding normal');
  assert.ok(!patient.queries[0].text.toLowerCase().includes('melanoma'));

  const clinician = buildChatQuery({ role: 'clinician', message: 'What margins??', topType: 'Melanoma' });
  assert.deepEqual(clinician.topics, ['melanoma']);
  assert.equal(clinician.queries[0].text, 'What margins melanoma');

  const bare = buildChatQuery({ role: 'clinician', message: '' });
  assert.ok(bare.queries[0].text.length > 0, 'an empty question still produces a usable query');
  assert.throws(() => buildChatQuery({ role: 'admin' }), TypeError);
});

// ---- chat citation check ----------------------------------------------------------------------------------------

test('chat answers keep valid markers, lose invalid ones, and list their sources', () => {
  const { text, sources, grounding } = validateChatCitations('Margins depend on thickness [P1] and [P7].', passages, { role: 'clinician' });
  assert.equal(text, 'Margins depend on thickness [P1] and.');
  assert.deepEqual(sources.map((s) => s.label), ['P1']);
  assert.equal(sources[0].excerpt, 'Margins depend on tumour thickness.');
  assert.equal(grounding.status, 'grounded');
});

test('chat answers for patients are scrubbed of cancer-type names and carry no excerpt', () => {
  const { text, sources, grounding } = validateChatCitations('A basal cell carcinoma can bleed [P2].', passages, { role: 'patient' });
  assert.equal(text, 'A skin cancer can bleed [P2].');
  assert.equal(grounding.scrubbed, 1);
  assert.deepEqual(sources.map((s) => s.label), ['P2']);
  assert.equal(sources[0].excerpt, undefined);
  assert.equal(validateChatCitations('nothing cited', [], { role: 'patient' }).grounding.status, 'unavailable');
  assert.equal(validateChatCitations('nothing cited', passages, { role: 'clinician' }).grounding.status, 'ungrounded');
});

// ---- marker stripping --------------------------------------------------------------------------------------------

test('stripMarkers removes every marker from a report when no passages exist', () => {
  const report = { summary: 'See [P1] and [P9].', steps: [{ a: 'one [P2]' }, 'two [P3]'], plain: 'no markers' };
  const out = stripMarkers(report);
  assert.deepEqual(out, { summary: 'See and.', steps: [{ a: 'one' }, 'two'], plain: 'no markers' });
  assert.ok(!JSON.stringify(out).includes('[P'));
});

// ---- nested strict statements ------------------------------------------------------------------------------------

test('each named medicine inside a medication scenario must carry its own marker', () => {
  const report = {
    medication_options: [
      { scenario: 'Superficial disease', agents: ['Imiquimod 5% cream [P2].', 'Vismodegib with no source.'], notes: 'Formulary.' },
      { scenario: 'Advanced disease, nothing cited', agents: ['Some unapproved drug.'], notes: 'No.' },
    ],
  };
  const { report: out, grounding } = validateCitations(report, passages, { role: 'clinician' });
  assert.equal(out.medication_options.length, 1);
  assert.deepEqual(out.medication_options[0].agents, ['Imiquimod 5% cream [P2].']);
  assert.equal(grounding.dropped.length, 2);
  assert.ok(grounding.dropped.some((d) => /Vismodegib/.test(d.text)));
  assert.ok(grounding.dropped.some((d) => /unapproved/.test(d.text)));
});
