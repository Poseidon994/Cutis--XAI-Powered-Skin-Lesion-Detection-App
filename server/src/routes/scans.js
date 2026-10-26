import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import multer from 'multer';
import rateLimit from 'express-rate-limit';
import { z } from 'zod';
import { config } from '../config.js';
import { BODY_SITES, CLINICIAN_DISPOSITION, DURATIONS, FILE_NAMES, PATIENT_URGENCY, SEX, SYMPTOMS, TIERS, XAI_METHODS } from '../constants.js';
import { requireAuth } from '../middleware/auth.js';
import { agentConfigured, chatReply, generateReport } from '../services/agent.js';
import { buildContext, relatedHistory } from '../services/context.js';
import { ml, MlError } from '../services/ml.js';
import { buildReportPdf } from '../services/report.js';
import { agreement } from '../services/xai.js';

const fields = z.object({
  bodySite: z.enum(Object.keys(BODY_SITES)).default('unknown'),
  lesionLabel: z.string().trim().max(80).optional(),
  patientRef: z.string().trim().max(80).optional(),
  age: z.coerce.number().int().min(0).max(120).optional(),
  sex: z.enum(Object.keys(SEX)).optional(),
  duration: z.enum(Object.keys(DURATIONS)).optional(),
  notes: z.string().trim().max(1000).optional(),
  symptoms: z.string().max(500).optional(),
});

function parseSymptoms(raw) {
  if (!raw) return [];
  let list = [];
  try {
    const v = JSON.parse(raw);
    list = Array.isArray(v) ? v : [];
  } catch {
    list = raw.split(',');
  }
  return [...new Set(list.map((s) => String(s).trim()).filter((s) => s in SYMPTOMS))];
}

function sniff(buf) {
  if (buf.length > 12 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return { mime: 'image/jpeg', ext: 'jpg' };
  if (buf.length > 8 && buf.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return { mime: 'image/png', ext: 'png' };
  if (buf.length > 12 && buf.subarray(0, 4).toString() === 'RIFF' && buf.subarray(8, 12).toString() === 'WEBP') return { mime: 'image/webp', ext: 'webp' };
  return null;
}

const dir = (id) => path.join(config.dataDir, 'scans', id);
const fileUrl = (id, key) => `/api/scans/${id}/files/${key}`;

function findOriginal(id) {
  const d = dir(id);
  if (!fs.existsSync(d)) return null;
  const f = fs.readdirSync(d).find((n) => n.startsWith('original.'));
  return f ? path.join(d, f) : null;
}

function filePathFor(id, key) {
  if (key === 'original') return findOriginal(id);
  const p = path.join(dir(id), `${key}.png`);
  return fs.existsSync(p) ? p : null;
}

function xaiDto(scan) {
  const out = {};
  for (const m of XAI_METHODS) {
    const e = scan.xai?.[m];
    out[m] = e
      ? { status: e.status, title: e.title, signed: e.signed, summary: e.summary, explainedScore: e.explained_score, params: e.params, runtimeS: e.runtime_s, error: e.error, url: e.status === 'done' ? `${fileUrl(scan.id, m)}?v=${e.at || ''}` : null }
      : { status: 'pending' };
  }
  out.agreement = scan.xai?.agreement || null;
  return out;
}

function agentDto(scan) {
  if (!scan.agentReport) return { status: 'none' };
  return { status: 'ready', source: scan.agentMeta?.source, model: scan.agentMeta?.model, reason: scan.agentMeta?.reason, guard: scan.agentMeta?.guard, generatedAt: scan.agentMeta?.generatedAt, report: scan.agentReport };
}

// Patients never receive raw scores, per-network numbers or cancer-type estimates, not even in the JSON.
const PATIENT_PREDICTION_FIELDS = ['tier', 'tier_label', 'tier_rank', 'source', 'note', 'percentile_benign'];
const forRole = (prediction, role) =>
  role === 'patient' && prediction ? Object.fromEntries(PATIENT_PREDICTION_FIELDS.filter((k) => prediction[k] !== undefined).map((k) => [k, prediction[k]])) : prediction;

function scanDto(scan, role) {
  return {
    id: scan.id, createdAt: scan.createdAt, bodySite: scan.bodySite, lesionLabel: scan.lesionLabel, patientRef: scan.patientRef, age: scan.age, sex: scan.sex,
    symptoms: scan.symptoms, duration: scan.duration, notes: scan.notes, tier: scan.tier, score: role === 'patient' ? null : scan.score, topType: role === 'patient' ? null : scan.topType,
    prediction: forRole(scan.prediction, role), quality: scan.quality, model: scan.modelInfo, xai: xaiDto(scan), agent: agentDto(scan),
    files: { original: fileUrl(scan.id, 'original'), modelView: fileUrl(scan.id, 'model_view') },
    guidance: role === 'patient' ? { kind: 'urgency', ...PATIENT_URGENCY[scan.tier] } : { kind: 'disposition', text: CLINICIAN_DISPOSITION[scan.tier] },
  };
}

function summaryDto(scan, role) {
  return {
    id: scan.id, createdAt: scan.createdAt, bodySite: scan.bodySite, lesionLabel: scan.lesionLabel, patientRef: scan.patientRef, tier: scan.tier, tierLabel: TIERS[scan.tier]?.label,
    score: role === 'patient' ? null : scan.score, topType: role === 'patient' ? null : scan.topType, hasReport: Boolean(scan.agentReport), thumb: fileUrl(scan.id, 'model_view'),
  };
}

export function scansRouter(store) {
  const r = express.Router();
  r.use(requireAuth(store));
  const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: config.maxUploadMb * 1024 * 1024, files: 1, fields: 20, fieldSize: 4096 } });
  const agentLimiter = config.rateLimit
    ? rateLimit({ windowMs: 60 * 1000, limit: 20, standardHeaders: 'draft-7', legacyHeaders: false, keyGenerator: (req) => `u${req.user.id}`, validate: { keyGeneratorIpFallback: false }, message: { error: 'rate_limited', message: 'Too many requests. Please wait a minute.' } })
    : (_q, _s, n) => n();

  const owned = (req, res) => {
    const scan = /^[0-9a-f-]{36}$/.test(req.params.id) ? store.scan(req.params.id) : null;
    if (!scan || scan.userId !== req.user.id) {
      res.status(404).json({ error: 'not_found', message: 'Scan not found.' });
      return null;
    }
    return scan;
  };

  // ------------------------------------------------------------------------------------------------ list / create
  r.get('/', (req, res) => {
    const limit = Math.min(Math.max(Number(req.query.limit) || 100, 1), 500);
    const offset = Math.max(Number(req.query.offset) || 0, 0);
    const search = String(req.query.search || '').slice(0, 80);
    res.json({ scans: store.listScans(req.user.id, { search, limit, offset }).map((x) => summaryDto(x, req.user.role)), total: store.countScans(req.user.id) });
  });

  r.post('/', upload.single('image'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'no_image', message: 'Please choose a photo of the lesion.' });
    const kind = sniff(req.file.buffer);
    if (!kind) return res.status(400).json({ error: 'bad_image', message: 'Only JPEG, PNG or WebP images are accepted.' });
    const parsed = fields.safeParse(req.body || {});
    if (!parsed.success) return res.status(400).json({ error: 'invalid_input', message: parsed.error.issues?.[0]?.message || 'Invalid details.' });
    const f = parsed.data;
    let pred;
    try {
      pred = await ml.predict(req.file.buffer, `lesion.${kind.ext}`, kind.mime);
    } catch (e) {
      const status = e instanceof MlError ? e.status : 502;
      return res.status(status).json({ error: 'analysis_failed', message: e.message });
    }
    const id = crypto.randomUUID();
    fs.mkdirSync(dir(id), { recursive: true });
    fs.writeFileSync(path.join(dir(id), `original.${kind.ext}`), req.file.buffer);
    if (pred.preview_png_b64) fs.writeFileSync(path.join(dir(id), 'model_view.png'), Buffer.from(pred.preview_png_b64, 'base64'));
    const p = pred.prediction;
    const scan = store.insertScan({
      id, userId: req.user.id, patientRef: req.user.role === 'clinician' ? f.patientRef : undefined, lesionLabel: f.lesionLabel, bodySite: f.bodySite, age: f.age, sex: f.sex,
      symptoms: parseSymptoms(f.symptoms), duration: f.duration, notes: f.notes, tier: p.tier, score: p.score, topType: p.top_type, prediction: p, quality: pred.quality, modelInfo: pred.model,
    });
    res.status(201).json({ scan: scanDto(scan, req.user.role) });
  });

  // -------------------------------------------------------------------------------------------------- one scan
  r.get('/:id', (req, res) => {
    const scan = owned(req, res);
    if (scan) res.json({ scan: scanDto(scan, req.user.role), messages: store.messages(scan.id), history: relatedHistory(store, scan).map((x) => summaryDto(x, req.user.role)) });
  });

  r.delete('/:id', (req, res) => {
    const scan = owned(req, res);
    if (!scan) return;
    store.deleteScan(scan.id);
    fs.rmSync(dir(scan.id), { recursive: true, force: true });
    res.json({ ok: true });
  });

  r.get('/:id/files/:name', (req, res) => {
    const scan = owned(req, res);
    if (!scan) return;
    if (!FILE_NAMES.includes(req.params.name)) return res.status(404).json({ error: 'not_found', message: 'No such file.' });
    const p = filePathFor(scan.id, req.params.name);
    if (!p) return res.status(404).json({ error: 'not_found', message: 'File not available yet.' });
    res.setHeader('Cache-Control', 'private, max-age=3600');
    res.sendFile(p);
  });

  // ---------------------------------------------------------------------------------------------- explanations
  const inflight = new Map();
  async function runExplain(scan, method) {
    const orig = findOriginal(scan.id);
    if (!orig) throw new MlError('The original image is missing.', 404);
    const buf = fs.readFileSync(orig);
    const ext = path.extname(orig).slice(1);
    try {
      const out = await ml.explain(method, buf, `lesion.${ext}`, ext === 'jpg' ? 'image/jpeg' : `image/${ext}`);
      fs.writeFileSync(path.join(dir(scan.id), `${method}.png`), Buffer.from(out.overlay_png_b64, 'base64'));
      const fresh = store.scan(scan.id); // read-modify-write without an await in between, so parallel explanations cannot overwrite each other
      if (!fresh) return;
      fresh.xai[method] = { status: 'done', title: out.title, signed: out.signed, summary: out.summary, explained_score: out.explained_score, params: out.params, runtime_s: out.runtime_s, map32: out.map32, at: Date.now() };
      const maps = Object.fromEntries(XAI_METHODS.filter((m) => fresh.xai[m]?.status === 'done').map((m) => [m, fresh.xai[m].map32]));
      fresh.xai.agreement = agreement(maps);
      store.setXai(scan.id, fresh.xai);
    } catch (e) {
      const fresh = store.scan(scan.id);
      if (fresh) {
        fresh.xai[method] = { status: 'failed', error: String(e.message || e).slice(0, 300), at: Date.now() };
        store.setXai(scan.id, fresh.xai);
      }
      throw e;
    }
  }

  r.post('/:id/explain/:method', async (req, res) => {
    const scan = owned(req, res);
    if (!scan) return;
    const { method } = req.params;
    if (!XAI_METHODS.includes(method)) return res.status(404).json({ error: 'unknown_method', message: `Unknown method. Use one of ${XAI_METHODS.join(', ')}.` });
    if (scan.xai?.[method]?.status === 'done' && req.query.force !== '1') return res.json({ xai: xaiDto(scan) });
    const key = `${scan.id}:${method}`;
    let job = inflight.get(key);
    if (!job) {
      job = runExplain(scan, method).finally(() => inflight.delete(key));
      inflight.set(key, job);
    }
    try {
      await job;
    } catch (e) {
      const status = e instanceof MlError ? e.status : 502;
      return res.status(status).json({ error: 'explain_failed', message: e.message, xai: xaiDto(store.scan(scan.id) || scan) });
    }
    res.json({ xai: xaiDto(store.scan(scan.id)) });
  });

  // ---------------------------------------------------------------------------------------------------- agent
  const reportJobs = new Map();
  async function ensureReport(scan, user, force = false) {
    const fresh = store.scan(scan.id);
    if (fresh.agentReport && !force) return fresh;
    let job = reportJobs.get(scan.id);
    if (!job) {
      job = (async () => {
        const s = store.scan(scan.id);
        const ctx = buildContext({ scan: s, history: relatedHistory(store, s), role: user.role });
        const { report, meta } = await generateReport(user.role, ctx);
        store.setAgent(scan.id, report, meta);
      })().finally(() => reportJobs.delete(scan.id));
      reportJobs.set(scan.id, job);
    }
    await job;
    return store.scan(scan.id);
  }

  r.post('/:id/agent/report', agentLimiter, async (req, res) => {
    const scan = owned(req, res);
    if (!scan) return;
    const updated = await ensureReport(scan, req.user, Boolean(req.body?.force));
    res.json({ agent: agentDto(updated) });
  });

  r.get('/:id/messages', (req, res) => {
    const scan = owned(req, res);
    if (scan) res.json({ messages: store.messages(scan.id) });
  });

  r.post('/:id/agent/chat', agentLimiter, async (req, res) => {
    const scan = owned(req, res);
    if (!scan) return;
    const msg = z.string().trim().min(1).max(1000).safeParse(req.body?.message);
    if (!msg.success) return res.status(400).json({ error: 'invalid_input', message: 'Type a question (up to 1000 characters).' });
    if (!agentConfigured()) return res.status(503).json({ error: 'agent_unavailable', message: 'The AI assistant is not configured on this server (GROQ_API_KEY).' });
    const history = store.messages(scan.id);
    const ctx = buildContext({ scan, history: relatedHistory(store, scan), role: req.user.role });
    try {
      const { reply } = await chatReply(req.user.role, ctx, history, msg.data);
      store.addMessage(scan.id, 'user', msg.data);
      store.addMessage(scan.id, 'assistant', reply);
      res.json({ reply, messages: store.messages(scan.id) });
    } catch (e) {
      res.status(e.status === 401 ? 503 : 502).json({ error: 'agent_failed', message: 'The AI assistant could not answer right now. Please try again.' });
    }
  });

  // ---------------------------------------------------------------------------------------------------- report
  r.get('/:id/report.pdf', async (req, res) => {
    const scan = owned(req, res);
    if (!scan) return;
    const full = await ensureReport(scan, req.user);
    const history = relatedHistory(store, full);
    const doc = buildReportPdf({ scan: full, user: req.user, role: req.user.role, report: full.agentReport, meta: full.agentMeta, history, filePath: (k) => filePathFor(full.id, k), model: full.modelInfo });
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename="cutis-report-${full.createdAt.slice(0, 10)}-${full.id.slice(0, 8)}.pdf"`);
    res.setHeader('Cache-Control', 'private, no-store');
    doc.pipe(res);
    doc.end();
  });

  return r;
}
