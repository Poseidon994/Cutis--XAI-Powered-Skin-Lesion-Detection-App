import fs from 'node:fs';
import path from 'node:path';
import { createApp } from './app.js';
import { config } from './config.js';
import { Store } from './db.js';
import { agentConfigured } from './services/agent.js';

fs.mkdirSync(path.join(config.dataDir, 'scans'), { recursive: true });
const store = new Store(config.db);
const app = createApp(store);
try {
  await store.ready;
} catch (e) {
  console.error(`[server] FATAL: cannot open the MySQL database ${config.db.user}@${config.db.host}:${config.db.port}/${config.db.database}: ${e?.message || e}`);
  process.exit(1);
}
const server = app.listen(config.port, () => {
  console.log(`[server] Cutis API listening on :${config.port} | data ${config.dataDir} | MySQL ${config.db.host}:${config.db.port}/${config.db.database} | ML ${config.ml.url}`);
  console.log(`[server] AI agent: ${agentConfigured() ? `Groq (${config.groq.model})` : 'no GROQ_API_KEY, template explanations will be used'}`);
  console.log(`[server] Guideline library (RAG): ${config.rag.enabled ? 'enabled' : 'disabled (RAG_ENABLED=0); reports will say they are not checked against guidelines'}`);
  if (!config.clinicianInvite) console.warn(`[server] CLINICIAN_INVITE_CODE is not set: ${config.production ? 'clinician sign-up is disabled' : 'anyone can register as a clinician (development only)'}`);
});
const stop = () => server.close(() => { store.close().finally(() => process.exit(0)); });
process.on('SIGTERM', stop);
process.on('SIGINT', stop);
