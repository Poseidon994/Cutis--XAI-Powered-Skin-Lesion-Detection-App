// Agreement between the three explanation maps (each arrives as a 32x32 grid of non-negative "evidence for higher risk").

function flat(grid) {
  return grid.flat().map(Number);
}

function topMask(v, frac = 0.2) {
  const sorted = [...v].sort((a, b) => b - a);
  const k = Math.max(1, Math.round(v.length * frac));
  const cut = sorted[k - 1];
  if (!(cut > 0)) return v.map(() => false);
  return v.map((x) => x >= cut);
}

function iou(a, b) {
  let inter = 0;
  let uni = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] && b[i]) inter++;
    if (a[i] || b[i]) uni++;
  }
  return uni ? inter / uni : 0;
}

function pearson(a, b) {
  const n = a.length;
  const ma = a.reduce((s, x) => s + x, 0) / n;
  const mb = b.reduce((s, x) => s + x, 0) / n;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0;
}

export function agreement(maps) {
  const names = Object.keys(maps).filter((k) => Array.isArray(maps[k]) && maps[k].length);
  if (names.length < 2) return null;
  const vecs = Object.fromEntries(names.map((n) => [n, flat(maps[n])]));
  const masks = Object.fromEntries(names.map((n) => [n, topMask(vecs[n])]));
  const pairs = [];
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      pairs.push({
        a: names[i],
        b: names[j],
        iou_top20: Number(iou(masks[names[i]], masks[names[j]]).toFixed(3)),
        correlation: Number(pearson(vecs[names[i]], vecs[names[j]]).toFixed(3)),
      });
    }
  }
  const meanIou = pairs.reduce((s, p) => s + p.iou_top20, 0) / pairs.length;
  // Two random maps that each mark the top 20% of cells overlap with an IoU of about 0.11.
  const level = meanIou >= 0.4 ? 'high' : meanIou >= 0.25 ? 'moderate' : 'low';
  return { methods: names, pairs, mean_iou_top20: Number(meanIou.toFixed(3)), level, chance_iou_top20: 0.11 };
}
