import fs from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';

const SCHEMA = `
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT NOT NULL UNIQUE COLLATE NOCASE,
  name TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('patient','clinician')),
  organisation TEXT,
  license_no TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS scans (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  patient_ref TEXT,
  lesion_label TEXT,
  body_site TEXT NOT NULL DEFAULT 'unknown',
  age INTEGER,
  sex TEXT,
  symptoms TEXT NOT NULL DEFAULT '[]',
  duration TEXT,
  notes TEXT,
  tier TEXT,
  score REAL,
  top_type TEXT,
  prediction TEXT,
  quality TEXT,
  model_info TEXT,
  xai TEXT NOT NULL DEFAULT '{}',
  agent_report TEXT,
  agent_meta TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_scans_user ON scans(user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scan_id TEXT NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_scan ON messages(scan_id, id);
`;

const j = (v) => (v === undefined || v === null ? null : JSON.stringify(v));
const p = (s, fallback = null) => {
  if (s === null || s === undefined) return fallback;
  try {
    return JSON.parse(s);
  } catch {
    return fallback;
  }
};

export class Store {
  constructor(dataDir, file = 'cutis.db') {
    fs.mkdirSync(dataDir, { recursive: true });
    this.dataDir = dataDir;
    this.db = new DatabaseSync(file === ':memory:' ? file : path.join(dataDir, file));
    this.db.exec('PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 5000;');
    this.db.exec(SCHEMA);
  }

  // ------------------------------------------------------------------ users
  createUser({ email, name, passwordHash, role, organisation, licenseNo }) {
    const r = this.db
      .prepare('INSERT INTO users (email, name, password_hash, role, organisation, license_no) VALUES (?,?,?,?,?,?)')
      .run(email, name, passwordHash, role, organisation ?? null, licenseNo ?? null);
    return this.userById(Number(r.lastInsertRowid));
  }

  userByEmail(email) {
    return this.db.prepare('SELECT * FROM users WHERE email = ?').get(email) ?? null;
  }

  userById(id) {
    return this.db.prepare('SELECT * FROM users WHERE id = ?').get(id) ?? null;
  }

  deleteUser(id) {
    this.db.prepare('DELETE FROM users WHERE id = ?').run(id);
  }

  // ------------------------------------------------------------------ scans
  insertScan(s) {
    this.db
      .prepare(`INSERT INTO scans (id, user_id, patient_ref, lesion_label, body_site, age, sex, symptoms, duration, notes,
                                   tier, score, top_type, prediction, quality, model_info)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`)
      .run(s.id, s.userId, s.patientRef ?? null, s.lesionLabel ?? null, s.bodySite ?? 'unknown', s.age ?? null, s.sex ?? null,
        JSON.stringify(s.symptoms ?? []), s.duration ?? null, s.notes ?? null, s.tier, s.score, s.topType, j(s.prediction), j(s.quality), j(s.modelInfo));
    return this.scan(s.id);
  }

  scan(id) {
    const r = this.db.prepare('SELECT * FROM scans WHERE id = ?').get(id);
    return r ? this._scan(r) : null;
  }

  _scan(r) {
    return {
      id: r.id, userId: r.user_id, patientRef: r.patient_ref, lesionLabel: r.lesion_label, bodySite: r.body_site, age: r.age, sex: r.sex,
      symptoms: p(r.symptoms, []), duration: r.duration, notes: r.notes, tier: r.tier, score: r.score, topType: r.top_type,
      prediction: p(r.prediction), quality: p(r.quality), modelInfo: p(r.model_info), xai: p(r.xai, {}), agentReport: p(r.agent_report),
      agentMeta: p(r.agent_meta), createdAt: r.created_at, updatedAt: r.updated_at,
    };
  }

  listScans(userId, { search = '', limit = 100, offset = 0 } = {}) {
    const like = `%${search.replace(/[%_]/g, (m) => `\\${m}`)}%`;
    const rows = search
      ? this.db
          .prepare(`SELECT * FROM scans WHERE user_id = ? AND (lesion_label LIKE ? ESCAPE '\\' OR patient_ref LIKE ? ESCAPE '\\' OR body_site LIKE ? ESCAPE '\\')
                    ORDER BY created_at DESC LIMIT ? OFFSET ?`)
          .all(userId, like, like, like, limit, offset)
      : this.db.prepare('SELECT * FROM scans WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?').all(userId, limit, offset);
    return rows.map((r) => this._scan(r));
  }

  countScans(userId) {
    return this.db.prepare('SELECT COUNT(*) AS n FROM scans WHERE user_id = ?').get(userId).n;
  }

  setXai(id, xai) {
    this.db.prepare("UPDATE scans SET xai = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?").run(JSON.stringify(xai), id);
  }

  setAgent(id, report, meta) {
    this.db
      .prepare("UPDATE scans SET agent_report = ?, agent_meta = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?")
      .run(j(report), j(meta), id);
  }

  deleteScan(id) {
    this.db.prepare('DELETE FROM scans WHERE id = ?').run(id);
  }

  scanIdsForUser(userId) {
    return this.db.prepare('SELECT id FROM scans WHERE user_id = ?').all(userId).map((r) => r.id);
  }

  // --------------------------------------------------------------- messages
  messages(scanId) {
    return this.db.prepare('SELECT id, role, content, created_at AS createdAt FROM messages WHERE scan_id = ? ORDER BY id').all(scanId);
  }

  addMessage(scanId, role, content) {
    this.db.prepare('INSERT INTO messages (scan_id, role, content) VALUES (?,?,?)').run(scanId, role, content);
  }

  close() {
    this.db.close();
  }
}
