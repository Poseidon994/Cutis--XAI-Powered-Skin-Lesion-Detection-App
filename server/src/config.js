import crypto from 'node:crypto';
import path from 'node:path';

const env = process.env;
const num = (v, d) => {
  const n = Number(v);
  return v !== undefined && v !== '' && Number.isFinite(n) ? n : d;
};

const production = env.NODE_ENV === 'production';

let jwtSecret = env.JWT_SECRET || '';
if (!jwtSecret) {
  if (production) {
    console.error('FATAL: JWT_SECRET must be set in production (at least 32 random characters).');
    process.exit(1);
  }
  jwtSecret = crypto.randomBytes(32).toString('hex');
  console.warn('[config] JWT_SECRET not set: using a random secret for this run (sessions end when the server restarts).');
}
if (production && jwtSecret.length < 32) {
  console.error('FATAL: JWT_SECRET is too short (use at least 32 characters).');
  process.exit(1);
}

export const config = {
  production,
  port: num(env.PORT, 4000),
  dataDir: path.resolve(env.DATA_DIR || './data'),
  jwtSecret,
  jwtDays: num(env.JWT_DAYS, 7),
  bcryptRounds: Math.max(4, num(env.BCRYPT_ROUNDS, 12)),
  rateLimit: env.RATE_LIMIT !== '0',
  cookieSecure: env.COOKIE_SECURE ? env.COOKIE_SECURE === '1' : production,
  clinicianInvite: env.CLINICIAN_INVITE_CODE || '',
  maxUploadMb: num(env.MAX_UPLOAD_MB, 10),
  trustProxy: env.TRUST_PROXY ? env.TRUST_PROXY : production ? 1 : false,
  staticDir: env.STATIC_DIR || '',
  ml: {
    url: (env.ML_SERVICE_URL || 'http://localhost:8000').replace(/\/$/, ''),
    key: env.ML_SERVICE_KEY || '',
    timeoutMs: num(env.ML_TIMEOUT_MS, 300000),
  },
  groq: {
    key: env.GROQ_API_KEY || '',
    baseUrl: (env.GROQ_BASE_URL || 'https://api.groq.com/openai/v1').replace(/\/$/, ''),
    model: env.GROQ_MODEL || 'llama-3.3-70b-versatile',
    fallbackModels: (env.GROQ_FALLBACK_MODELS || '').split(',').map((s) => s.trim()).filter(Boolean),
    timeoutMs: num(env.GROQ_TIMEOUT_MS, 60000),
  },
};
