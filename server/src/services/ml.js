import { config } from '../config.js';

export class MlError extends Error {
  constructor(message, status = 502, detail = null) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function call(path, { method = 'GET', body, timeoutMs = config.ml.timeoutMs } = {}) {
  const headers = {};
  if (config.ml.key) headers['X-Service-Key'] = config.ml.key;
  let res;
  try {
    res = await fetch(`${config.ml.url}${path}`, { method, body, headers, signal: AbortSignal.timeout(timeoutMs) });
  } catch (e) {
    const timeout = e?.name === 'TimeoutError' || e?.name === 'AbortError';
    throw new MlError(timeout ? 'The analysis service took too long to respond.' : 'The analysis service is not reachable.', 504, String(e?.message || e));
  }
  let data = null;
  const text = await res.text();
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    const detail = data?.detail || text.slice(0, 300);
    const status = res.status === 400 || res.status === 413 ? res.status : res.status === 503 ? 503 : 502;
    throw new MlError(typeof detail === 'string' ? detail : 'The analysis service returned an error.', status, detail);
  }
  return data;
}

function form(buffer, filename, mime) {
  const f = new FormData();
  f.append('file', new Blob([buffer], { type: mime || 'application/octet-stream' }), filename || 'lesion.jpg');
  return f;
}

export const ml = {
  health: async () => {
    try {
      return await call('/health', { timeoutMs: 5000 });
    } catch (e) {
      return { status: 'unreachable', ready: false, error: e.message };
    }
  },
  modelCard: () => call('/model-card', { timeoutMs: 15000 }),
  predict: (buffer, filename, mime) => call('/predict', { method: 'POST', body: form(buffer, filename, mime), timeoutMs: 120000 }),
  explain: (method, buffer, filename, mime) => call(`/explain/${method}`, { method: 'POST', body: form(buffer, filename, mime) }),
};
