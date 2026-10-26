import fs from 'node:fs';
import path from 'node:path';
import { createApp } from './app.js';
import { config } from './config.js';
import { Store } from './db.js';
import { agentConfigured } from './services/agent.js';

fs.mkdirSync(path.join(config.dataDir, 'scans'), { recursive: true });
const store = new Store(config.dataDir);
const app = createApp(store);
const server = app.listen(config.port, () => {
  console.log(`[server] Cutis API listening on :${config.port} | data ${config.dataDir} | ML ${config.ml.url}`);
  console.log(`[server] AI agent: ${agentConfigured() ? `Groq (${config.groq.model})` : 'no GROQ_API_KEY, template explanations will be used'}`);
  if (!config.clinicianInvite) console.warn(`[server] CLINICIAN_INVITE_CODE is not set: ${config.production ? 'clinician sign-up is disabled' : 'anyone can register as a clinician (development only)'}`);
});
const stop = () => server.close(() => { store.close(); process.exit(0); });
process.on('SIGTERM', stop);
process.on('SIGINT', stop);
