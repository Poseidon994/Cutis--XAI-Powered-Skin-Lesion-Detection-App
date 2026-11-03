// MySQL data layer (mysql2). The app previously used the built-in node:sqlite; it now needs a MySQL 8
// (or MariaDB 10.6+) server, configured through DB_* variables. All methods are async: callers await them.
//
// Conventions kept from the SQLite version so the rest of the app did not have to change:
//   - created_at / updated_at travel as UTC ISO strings ("2026-10-19T10:00:00.000Z").
//     We write them ourselves as UTC "YYYY-MM-DD HH:MM:SS.mmm" and read with dateStrings: true,
//     so no session-timezone or JS-Date conversion can shift a date.
//   - JSON-shaped columns (symptoms, prediction, xai, agent_report, ...) are stored as JSON and
//     returned as parsed objects; the reader also accepts plain strings for MariaDB LONGTEXT rows.
import mysql from 'mysql2/promise';

const SCHEMA = `
CREATE TABLE IF NOT EXISTS users (
  id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  email VARCHAR(190) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL UNIQUE,
  name VARCHAR(80) NOT NULL,
  password_hash VARCHAR(100) NOT NULL,
  role ENUM('patient','clinician') NOT NULL,
  organisation VARCHAR(120) NULL,
  license_no VARCHAR(60) NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE TABLE IF NOT EXISTS scans (
  id CHAR(36) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL PRIMARY KEY,
  user_id INT UNSIGNED NOT NULL,
  patient_ref VARCHAR(80) NULL,
  lesion_label VARCHAR(80) NULL,
  body_site VARCHAR(30) NOT NULL DEFAULT 'unknown',
  age SMALLINT UNSIGNED NULL,
  sex VARCHAR(20) NULL,
  symptoms JSON NOT NULL,
  duration VARCHAR(20) NULL,
  notes TEXT NULL,
  tier VARCHAR(20) NULL,
  score DOUBLE NULL,
  top_type VARCHAR(20) NULL,
  prediction JSON NULL,
  quality JSON NULL,
  model_info JSON NULL,
  xai JSON NOT NULL,
  rag JSON NULL,
  agent_report JSON NULL,
  agent_meta JSON NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  CONSTRAINT fk_scans_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  KEY idx_scans_user (user_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
CREATE TABLE IF NOT EXISTS messages (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  scan_id CHAR(36) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  role ENUM('user','assistant') NOT NULL,
  content TEXT NOT NULL,
  sources JSON NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  CONSTRAINT fk_messages_scan FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE,
  KEY idx_messages_scan (scan_id, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
`;

const j = (v) => (v === undefined || v === null ? null : JSON.stringify(v));
const p = (s, fallback = null) => {
  if (s === null || s === undefined) return fallback;
  if (typeof s === 'object') return s; // mysql2 parses JSON columns; MariaDB returns strings
  try {
    return JSON.parse(s);
  } catch {
    return fallback;
  }
};

// "2026-10-19 10:00:00.123" (UTC literal from MySQL) -> "2026-10-19T10:00:00.123Z"
const iso = (s) => {
  if (s === null || s === undefined) return null;
  const m = String(s).match(/^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z?$/);
  if (!m) return s instanceof Date ? s.toISOString() : s;
  const frac = m[2].includes('.') ? m[2] : `${m[2]}.000`;
  return `${m[1]}T${frac}Z`;
};
const now = () => new Date().toISOString().slice(0, 23); // UTC "YYYY-MM-DD HH:MM:SS.mmm"

async function createSchema(pool) {
  // mysql2 runs one statement per query (multipleStatements is off, as it should be), so split the DDL.
  for (const stmt of SCHEMA.split(/;\s*(?:\n|$)/).map((s) => s.trim()).filter(Boolean)) {
    await pool.query(stmt);
  }
}

export class Store {
  constructor(opts = {}) {
    this.opts = {
      host: opts.host ?? '127.0.0.1',
      port: opts.port ?? 3306,
      user: opts.user ?? 'cutis',
      password: opts.password ?? '',
      database: opts.database ?? 'cutis',
      connectionLimit: opts.connectionLimit ?? 10,
      waitForConnections: true,
      dateStrings: true, // keep DATETIME as "YYYY-MM-DD HH:MM:SS.mmm" strings, never JS Dates
      supportBigNumbers: true,
      namedPlaceholders: false,
    };
    this.db = mysql.createPool(this.opts);
    this.db.on('connection', (conn) => {
      // All timestamps are written by the app as UTC literals; keep NOW()/CURRENT_TIMESTAMP in UTC too.
      conn.promise().query("SET time_zone = '+00:00'").catch(() => {});
    });
    this.ready = createSchema(this.db).catch((e) => {
      this.readyError = e;
      throw e;
    });
  }

  async _q(sql, params = []) {
    await this.ready;
    const [rows] = await this.db.query(sql, params);
    return rows;
  }

  // ------------------------------------------------------------------ users
  async createUser({ email, name, passwordHash, role, organisation, licenseNo }) {
    try {
      const r = await this._q('INSERT INTO users (email, name, password_hash, role, organisation, license_no) VALUES (?,?,?,?,?,?)', [
        email, name, passwordHash, role, organisation ?? null, licenseNo ?? null,
      ]);
      return this.userById(Number(r.insertId));
    } catch (e) {
      if (e?.code === 'ER_DUP_ENTRY') return null; // raced another registration for the same email
      throw e;
    }
  }

  async userByEmail(email) {
    return this._user((await this._q('SELECT * FROM users WHERE email = ?', [email]))[0]);
  }

  async userById(id) {
    return this._user((await this._q('SELECT * FROM users WHERE id = ?', [id]))[0]);
  }

  _user(r) {
    return r ? { ...r, created_at: iso(r.created_at) } : null;
  }

  async deleteUser(id) {
    await this._q('DELETE FROM users WHERE id = ?', [id]);
  }

  // ------------------------------------------------------------------ scans
  async insertScan(s) {
    await this._q(
      `INSERT INTO scans (id, user_id, patient_ref, lesion_label, body_site, age, sex, symptoms, duration, notes,
                          tier, score, top_type, prediction, quality, model_info, xai, created_at, updated_at)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
      [s.id, s.userId, s.patientRef ?? null, s.lesionLabel ?? null, s.bodySite ?? 'unknown', s.age ?? null, s.sex ?? null,
        j(s.symptoms ?? []), s.duration ?? null, s.notes ?? null, s.tier, s.score, s.topType, j(s.prediction), j(s.quality), j(s.modelInfo), j({}), now(), now()],
    );
    return this.scan(s.id);
  }

  async scan(id) {
    const r = (await this._q('SELECT * FROM scans WHERE id = ?', [id]))[0];
    return r ? this._scan(r) : null;
  }

  _scan(r) {
    return {
      id: r.id, userId: r.user_id, patientRef: r.patient_ref, lesionLabel: r.lesion_label, bodySite: r.body_site, age: r.age, sex: r.sex,
      symptoms: p(r.symptoms, []), duration: r.duration, notes: r.notes, tier: r.tier, score: r.score, topType: r.top_type,
      prediction: p(r.prediction), quality: p(r.quality), modelInfo: p(r.model_info), xai: p(r.xai, {}), rag: p(r.rag),
      agentReport: p(r.agent_report), agentMeta: p(r.agent_meta), createdAt: iso(r.created_at), updatedAt: iso(r.updated_at),
    };
  }

  async listScans(userId, { search = '', limit = 100, offset = 0 } = {}) {
    const like = `%${search.replace(/[!%_]/g, (m) => `!${m}`)}%`;
    const rows = search
      ? await this._q(
          `SELECT * FROM scans WHERE user_id = ? AND (lesion_label LIKE ? ESCAPE '!' OR patient_ref LIKE ? ESCAPE '!' OR body_site LIKE ? ESCAPE '!')
           ORDER BY created_at DESC LIMIT ? OFFSET ?`,
          [userId, like, like, like, limit, offset],
        )
      : await this._q('SELECT * FROM scans WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?', [userId, limit, offset]);
    return rows.map((r) => this._scan(r));
  }

  async countScans(userId) {
    const r = (await this._q('SELECT COUNT(*) AS n FROM scans WHERE user_id = ?', [userId]))[0];
    return Number(r.n);
  }

  async setXai(id, xai) {
    await this._q('UPDATE scans SET xai = ?, updated_at = ? WHERE id = ?', [j(xai), now(), id]);
  }

  async setRag(id, rag) {
    await this._q('UPDATE scans SET rag = ?, updated_at = ? WHERE id = ?', [j(rag), now(), id]);
  }

  async setAgent(id, report, meta) {
    await this._q('UPDATE scans SET agent_report = ?, agent_meta = ?, updated_at = ? WHERE id = ?', [j(report), j(meta), now(), id]);
  }

  async deleteScan(id) {
    await this._q('DELETE FROM scans WHERE id = ?', [id]);
  }

  async scanIdsForUser(userId) {
    return (await this._q('SELECT id FROM scans WHERE user_id = ?', [userId])).map((r) => r.id);
  }

  // --------------------------------------------------------------- messages
  async messages(scanId) {
    const rows = await this._q('SELECT id, role, content, sources, created_at FROM messages WHERE scan_id = ? ORDER BY id', [scanId]);
    return rows.map((r) => ({ id: Number(r.id), role: r.role, content: r.content, sources: p(r.sources, []), createdAt: iso(r.created_at) }));
  }

  async addMessage(scanId, role, content, sources = null) {
    const r = await this._q('INSERT INTO messages (scan_id, role, content, sources) VALUES (?,?,?,?)', [scanId, role, content, j(sources)]);
    return Number(r.insertId);
  }

  // --------------------------------------------------------------- lifecycle
  async close() {
    await this.ready.catch(() => {});
    await this.db.end();
  }

  // Test helper: drop everything so a fresh schema can be created for the next run.
  async dropAll() {
    await this._q('SET FOREIGN_KEY_CHECKS = 0');
    await this._q('DROP TABLE IF EXISTS messages, scans, users');
    await this._q('SET FOREIGN_KEY_CHECKS = 1');
  }

  // Test helper: drop and re-create the schema, leaving the database empty.
  async reset() {
    await this.dropAll();
    await createSchema(this.db);
  }
}
