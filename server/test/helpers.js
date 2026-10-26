import http from 'node:http';
import zlib from 'node:zlib';

export function png(w, h, fn) {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 3 + 1)] = 0;
    for (let x = 0; x < w; x++) {
      const [r, g, b] = fn(x, y);
      const o = y * (w * 3 + 1) + 1 + x * 3;
      raw[o] = r;
      raw[o + 1] = g;
      raw[o + 2] = b;
    }
  }
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(zlib.crc32(td) >>> 0);
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

export const lesionPng = (w = 128, h = 128) =>
  png(w, h, (x, y) => {
    const d = Math.hypot(x - w / 2, y - h / 2);
    return d < w * 0.28 ? [88, 52, 38] : [206, 160, 140];
  });

export function listen(handler) {
  return new Promise((resolve) => {
    const server = http.createServer(handler);
    server.listen(0, '127.0.0.1', () => resolve({ server, url: `http://127.0.0.1:${server.address().port}` }));
  });
}

export async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return Buffer.concat(chunks);
}

export const KEY = 'ml-test-key';

export function map32(cx, cy, s) {
  return Array.from({ length: 32 }, (_, y) => Array.from({ length: 32 }, (_, x) => Number(Math.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))).toFixed(4))));
}

export function prediction(tier = 'elevated', score = 2.31) {
  return {
    ok: true,
    prediction: {
      score, tier, tier_label: { low: 'Low concern', guarded: 'Some concern', elevated: 'Elevated concern', high: 'High concern' }[tier], tier_rank: ['low', 'guarded', 'elevated', 'high'].indexOf(tier),
      source: 'oof', thresholds: { sens95: 0.61, sens90: 1.04, sens80: 1.54 }, percentile_benign: 96.4, sensitivity_at_or_above: 0.81, cancer_rate_at_or_above: 0.021, base_rate: 0.00098,
      enrichment: 21.4, enrichment_n: 612, probability_at_training_prevalence: 0.012, suspicious: 0.4, type_probs: { Melanoma: 0.58, BCC: 0.31, SCC: 0.11 }, top_type: 'Melanoma',
      per_backbone: [
        { name: 'convnext_tiny', z: 2.1, fold_sd: 0.3, n_folds: 5, tier: 'elevated' }, { name: 'edgenext_base', z: 2.6, fold_sd: 0.4, n_folds: 5, tier: 'high' },
        { name: 'eva02_small', z: 1.9, fold_sd: 0.2, n_folds: 5, tier: 'elevated' }, { name: 'swin_t', z: 2.4, fold_sd: 0.5, n_folds: 5, tier: 'elevated' },
      ],
      network_sd: 0.8, n_networks: 20, tta_views: 2, agreement: { backbones: 4, same_tier_as_ensemble: 3, flagging_elevated_or_high: 4 },
    },
    quality: { original_size: [400, 400], cropped_to_square: false, mean_luminance: 120, sharpness: 30, warnings: [] },
    preview_png_b64: lesionPng(256, 256).toString('base64'),
    model: { version: 'v4', created: '2026-10-19 10:00:00', device: 'cpu', backbones: ['convnext_tiny', 'edgenext_base', 'eva02_small', 'swin_t'], networks: 20 },
    runtime_s: 1.2,
  };
}

export function explanation(method) {
  const m = { gradcam: [16, 16, 5], lime: [14, 15, 6], shap: [17, 14, 5] }[method];
  return {
    ok: true, method, title: { gradcam: 'Grad-CAM', lime: 'LIME', shap: 'SHAP' }[method], signed: method !== 'gradcam', map32: map32(...m),
    summary: { focus: 'central', direction: method === 'lime' ? 'upper left' : 'centre', center_share: 0.62, edge_share: 0.05, concentration: 0.18, n_networks: 4, network_agreement: 0.71, local_r2: 0.86, n_samples: 150, positive_share: 0.64, top_regions: [{ weight: 0.12, area_share: 0.05, direction: 'upper left' }] },
    explained_score: { subset_z: 2.2, n_networks: 4 }, params: {}, runtime_s: 2.5,
    overlay_png_b64: png(384, 384, (x, y) => [Math.min(255, x), Math.min(255, y), 120]).toString('base64'),
  };
}
