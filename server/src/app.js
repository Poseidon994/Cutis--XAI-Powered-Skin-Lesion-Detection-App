import fs from 'node:fs';
import path from 'node:path';
import cookieParser from 'cookie-parser';
import express from 'express';
import helmet from 'helmet';
import { config } from './config.js';
import { Store } from './db.js';
import { agentConfigured } from './services/agent.js';
import { ml, MlError } from './services/ml.js';
import { requireAuth, requireCustomHeader } from './middleware/auth.js';
import { authRouter } from './routes/auth.js';
import { scansRouter } from './routes/scans.js';

export function createApp(store = new Store(config.dataDir)) {
  const app = express();
  if (config.trustProxy) app.set('trust proxy', config.trustProxy);
  app.disable('x-powered-by');
  app.use(
    helmet({
      contentSecurityPolicy: { directives: { defaultSrc: ["'self'"], imgSrc: ["'self'", 'data:', 'blob:'], styleSrc: ["'self'", "'unsafe-inline'"], connectSrc: ["'self'"], objectSrc: ["'none'"], frameAncestors: ["'none'"] } },
      crossOriginResourcePolicy: { policy: 'same-origin' },
    }),
  );
  app.use(cookieParser());
  app.use(express.json({ limit: '100kb' }));
  app.use('/api', requireCustomHeader);

  app.get('/api/health', async (_req, res) => {
    const m = await ml.health();
    res.json({ server: 'ok', ml: { status: m.status, ready: Boolean(m.ready), device: m.device, networks: m.networks, notes: m.notes, error: m.error }, agent: { configured: agentConfigured(), model: agentConfigured() ? config.groq.model : null }, clinicianSignup: Boolean(config.clinicianInvite) || !config.production, clinicianInviteRequired: Boolean(config.clinicianInvite) });
  });

  app.get('/api/model-card', requireAuth(store), async (_req, res) => {
    try {
      res.json(await ml.modelCard());
    } catch (e) {
      res.status(e instanceof MlError ? e.status : 502).json({ error: 'model_card_unavailable', message: e.message });
    }
  });

  app.use('/api/auth', authRouter(store));
  app.use('/api/scans', scansRouter(store));
  app.use('/api', (_req, res) => res.status(404).json({ error: 'not_found', message: 'No such endpoint.' }));

  if (config.staticDir && fs.existsSync(path.join(config.staticDir, 'index.html'))) {
    app.use(express.static(config.staticDir, { index: false, maxAge: '1h' }));
    app.use((req, res, next) => (req.method === 'GET' ? res.sendFile(path.resolve(config.staticDir, 'index.html')) : next()));
  }

  // eslint-disable-next-line no-unused-vars
  app.use((err, _req, res, _next) => {
    if (err?.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ error: 'too_large', message: `The image is larger than ${config.maxUploadMb} MB.` });
    if (err?.type === 'entity.parse.failed') return res.status(400).json({ error: 'bad_json', message: 'The request body is not valid JSON.' });
    if (err?.name === 'MulterError') return res.status(400).json({ error: 'upload_error', message: err.message });
    console.error('[server] unhandled error:', err);
    res.status(500).json({ error: 'server_error', message: 'Something went wrong on the server.' });
  });
  return app;
}
