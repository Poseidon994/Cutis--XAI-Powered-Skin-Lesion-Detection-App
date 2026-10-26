import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import bcrypt from 'bcryptjs';
import express from 'express';
import rateLimit from 'express-rate-limit';
import { z } from 'zod';
import { config } from '../config.js';
import { clearSession, issueSession, publicUser, requireAuth } from '../middleware/auth.js';

const registerSchema = z.object({
  name: z.string().trim().min(1, 'Please enter your name').max(80),
  email: z.string().trim().toLowerCase().email('Please enter a valid email address').max(160),
  password: z.string().min(8, 'Use at least 8 characters').max(72, 'Use at most 72 characters'),
  role: z.enum(['patient', 'clinician']),
  inviteCode: z.string().max(100).optional(),
  organisation: z.string().trim().max(120).optional(),
  licenseNo: z.string().trim().max(60).optional(),
});
const loginSchema = z.object({ email: z.string().trim().toLowerCase().email().max(160), password: z.string().min(1).max(200) });

const DUMMY_HASH = bcrypt.hashSync('not-a-real-password', 4);

function safeEqual(a, b) {
  const x = Buffer.from(String(a));
  const y = Buffer.from(String(b));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}

function zodMessage(err) {
  return err.issues?.[0]?.message || 'Invalid input';
}

export function authRouter(store) {
  const r = express.Router();
  const limiter = config.rateLimit
    ? rateLimit({ windowMs: 15 * 60 * 1000, limit: 30, standardHeaders: 'draft-7', legacyHeaders: false, message: { error: 'rate_limited', message: 'Too many attempts. Please wait a few minutes.' } })
    : (_q, _s, n) => n();

  r.post('/register', limiter, async (req, res) => {
    const parsed = registerSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: 'invalid_input', message: zodMessage(parsed.error) });
    const d = parsed.data;
    if (d.role === 'clinician') {
      if (!config.clinicianInvite) {
        if (config.production) return res.status(403).json({ error: 'clinician_signup_disabled', message: 'Clinician sign-up is not enabled on this server.' });
      } else if (!safeEqual(d.inviteCode || '', config.clinicianInvite)) {
        return res.status(403).json({ error: 'invalid_invite', message: 'That clinician invite code is not valid.' });
      }
    }
    if (store.userByEmail(d.email)) return res.status(409).json({ error: 'email_taken', message: 'An account with this email already exists.' });
    const passwordHash = await bcrypt.hash(d.password, config.bcryptRounds);
    const user = store.createUser({ email: d.email, name: d.name, passwordHash, role: d.role, organisation: d.organisation, licenseNo: d.licenseNo });
    issueSession(res, user);
    res.status(201).json({ user: publicUser(user) });
  });

  r.post('/login', limiter, async (req, res) => {
    const parsed = loginSchema.safeParse(req.body);
    if (!parsed.success) return res.status(400).json({ error: 'invalid_input', message: 'Enter your email and password.' });
    const user = store.userByEmail(parsed.data.email);
    const ok = await bcrypt.compare(parsed.data.password, user ? user.password_hash : DUMMY_HASH);
    if (!user || !ok) return res.status(401).json({ error: 'bad_credentials', message: 'Email or password is incorrect.' });
    issueSession(res, user);
    res.json({ user: publicUser(user) });
  });

  r.post('/logout', (_req, res) => {
    clearSession(res);
    res.json({ ok: true });
  });

  r.get('/me', requireAuth(store), (req, res) => res.json({ user: publicUser(req.user) }));

  // Right to erasure: removes the account, every scan and every stored image.
  r.delete('/me', requireAuth(store), (req, res) => {
    for (const id of store.scanIdsForUser(req.user.id)) fs.rmSync(path.join(config.dataDir, 'scans', id), { recursive: true, force: true });
    store.deleteUser(req.user.id);
    clearSession(res);
    res.json({ ok: true });
  });

  return r;
}
