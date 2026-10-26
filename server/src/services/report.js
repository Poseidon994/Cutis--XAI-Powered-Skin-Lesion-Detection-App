// PDF report (pdfkit). Layout differs by role: the patient report is a plain-language action sheet, the clinician report is a
// results sheet with the supporting evidence, options and history.
import fs from 'node:fs';
import PDFDocument from 'pdfkit';
import { BODY_SITES, DISCLAIMER, DURATIONS, PATIENT_URGENCY, SEX, SYMPTOMS, TIERS } from '../constants.js';

const M = 42;
const INK = '#16252B';
const MUTED = '#5B6B74';
const LINE = '#D5DDE2';
const ACCENT = '#34389B';
const TIER_COLOR = { low: '#2E7D6B', guarded: '#B7791F', elevated: '#D2571F', high: '#B3243B' };

// pdfkit's built-in fonts use WinAnsi; map or drop everything else so no glyph is silently lost.
const MAP = { '≥': '>=', '≤': '<=', '→': '->', '←': '<-', '≈': '~', '−': '-', '‑': '-', '…': '...', '×': 'x', '\u00A0': ' ', '‐': '-' };
export function clean(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/[≥≤→←≈−‑…×\u00A0‐]/g, (c) => MAP[c])
    .replace(/[^\x20-\x7E\u00A1-\u00FF\u2013\u2014\u2018\u2019\u201C\u201D\u2022\u20AC\n]/g, '');
}

const fmtDate = (iso) => new Date(iso).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });

export function buildReportPdf({ scan, user, role, report, meta, history, filePath, model }) {
  const doc = new PDFDocument({ size: 'A4', margins: { top: M, bottom: M + 14, left: M, right: M }, bufferPages: true, info: { Title: 'Cutis skin lesion report', Author: 'Cutis', Subject: 'AI-assisted skin lesion analysis' } });
  const W = doc.page.width - 2 * M;
  const tier = scan.tier || 'guarded';
  const color = TIER_COLOR[tier];

  const space = (h) => {
    if (doc.y + h > doc.page.height - M - 20) doc.addPage();
  };
  const h2 = (t) => {
    space(48);
    doc.moveDown(0.6);
    doc.font('Helvetica-Bold').fontSize(11.5).fillColor(ACCENT).text(clean(t), M, doc.y, { width: W });
    const y = doc.y + 2;
    doc.moveTo(M, y).lineTo(M + W, y).lineWidth(0.6).strokeColor(LINE).stroke();
    doc.y = y + 6;
  };
  const para = (t, o = {}) => {
    if (!t) return;
    space(30);
    doc.font(o.bold ? 'Helvetica-Bold' : 'Helvetica').fontSize(o.size || 10).fillColor(o.color || INK).text(clean(t), M, doc.y, { width: W, lineGap: 2.2 });
    doc.moveDown(0.35);
  };
  const bullet = (t, o = {}) => {
    space(22);
    const y = doc.y;
    doc.font('Helvetica').fontSize(10).fillColor(o.color || INK).text(o.marker || '\u2022', M + 2, y, { width: 14, lineBreak: false });
    doc.text(clean(t), M + 16, y, { width: W - 16, lineGap: 2 });
    doc.moveDown(0.25);
  };
  const kv = (rows) => {
    for (const [k, v] of rows) {
      if (v === null || v === undefined || v === '') continue;
      space(18);
      const y = doc.y;
      doc.font('Helvetica-Bold').fontSize(9.5).fillColor(MUTED).text(clean(k), M, y, { width: 120, lineBreak: false });
      doc.font('Helvetica').fontSize(10).fillColor(INK).text(clean(v), M + 124, y, { width: W - 124 });
      doc.moveDown(0.15);
    }
  };

  // ------------------------------------------------------------------------------------------------ header
  doc.font('Times-Bold').fontSize(24).fillColor(INK).text('Cutis', M, M - 4, { continued: false });
  doc.font('Helvetica').fontSize(10).fillColor(MUTED).text('AI-assisted skin lesion report', M, M + 24);
  doc.font('Helvetica').fontSize(9).fillColor(MUTED).text(`${role === 'patient' ? 'Patient report' : 'Clinician report'}`, M, M, { width: W, align: 'right' });
  doc.text(`${fmtDate(scan.createdAt)}  |  Scan ${scan.id.slice(0, 8)}`, M, M + 12, { width: W, align: 'right' });
  doc.y = M + 46;
  doc.moveTo(M, doc.y).lineTo(M + W, doc.y).lineWidth(1.2).strokeColor(INK).stroke();
  doc.y += 10;

  const dis = clean(report.disclaimer || DISCLAIMER[role]);
  const boxH = doc.font('Helvetica').fontSize(8.5).heightOfString(dis, { width: W - 16 }) + 12;
  doc.roundedRect(M, doc.y, W, boxH, 4).fillColor('#FFF6DB').fill();
  doc.fillColor('#6B5300').font('Helvetica').fontSize(8.5).text(dis, M + 8, doc.y + 6, { width: W - 16 });
  doc.y += boxH - 6 + 8;

  // ------------------------------------------------------------------------------------------------ headline / result
  const pred = scan.prediction || {};
  const headline = role === 'patient' ? report.headline : report.summary;
  doc.roundedRect(M, doc.y, 6, 54, 2).fillColor(color).fill();
  doc.font('Helvetica-Bold').fontSize(17).fillColor(color).text(clean(TIERS[tier]?.label || tier), M + 16, doc.y + 2, { width: W - 16 });
  if (role === 'patient') {
    doc.font('Helvetica').fontSize(10.5).fillColor(INK).text(clean(headline), M + 16, doc.y + 2, { width: W - 16 });
  } else {
    doc.font('Helvetica').fontSize(9.5).fillColor(INK).text(clean(`Image-ensemble score z = ${pred.score}  |  percentile among benign lesions ${pred.percentile_benign ?? 'n/a'}  |  enrichment ${pred.enrichment ?? 'n/a'}x`), M + 16, doc.y + 3, { width: W - 16 });
  }
  doc.y = Math.max(doc.y, doc.y) + 14;

  // ------------------------------------------------------------------------------------------------ case details
  h2('Case details');
  kv([
    ['Scan date', fmtDate(scan.createdAt)],
    ['Body site', BODY_SITES[scan.bodySite]],
    ['Lesion label', scan.lesionLabel],
    role === 'clinician' ? ['Patient reference', scan.patientRef] : null,
    ['Age / sex', [scan.age != null ? `${scan.age} years` : null, scan.sex ? SEX[scan.sex] : null].filter(Boolean).join(', ')],
    ['Symptoms reported', (scan.symptoms || []).map((s) => SYMPTOMS[s] || s).join(', ')],
    ['Duration', scan.duration ? DURATIONS[scan.duration] : null],
    ['Notes', scan.notes],
  ].filter(Boolean));

  // ------------------------------------------------------------------------------------------------ images
  h2('What the AI looked at');
  space(170);
  const cols = [
    ['model_view', 'Image analysed', 'Centre-cropped lesion photo as seen by the model'],
    ['gradcam', scan.xai?.gradcam?.title || 'Grad-CAM', 'Brighter = more attention'],
    ['lime', 'LIME', 'Red raises, blue lowers the rating'],
    ['shap', 'SHAP', 'Red raises, blue lowers the rating'],
  ];
  const gap = 10;
  const iw = (W - gap * 3) / 4;
  const y0 = doc.y;
  cols.forEach(([key, title, cap], i) => {
    const x = M + i * (iw + gap);
    const p = filePath(key);
    if (p && fs.existsSync(p)) {
      doc.image(p, x, y0, { width: iw, height: iw });
    } else {
      doc.rect(x, y0, iw, iw).fillColor('#EEF1F3').fill();
      doc.font('Helvetica').fontSize(8).fillColor(MUTED).text('not available', x, y0 + iw / 2 - 4, { width: iw, align: 'center' });
    }
    doc.rect(x, y0, iw, iw).lineWidth(0.5).strokeColor(LINE).stroke();
    doc.font('Helvetica-Bold').fontSize(9).fillColor(INK).text(clean(title), x, y0 + iw + 4, { width: iw });
    doc.font('Helvetica').fontSize(7.5).fillColor(MUTED).text(clean(cap), x, doc.y, { width: iw });
  });
  doc.y = y0 + iw + 38;
  if (role === 'patient') para(report.what_the_images_show);

  // ------------------------------------------------------------------------------------------------ role content
  if (role === 'patient') {
    h2('What we found');
    para(report.what_we_found);
    h2('What this means');
    para(report.what_it_means);
    const urg = report.urgency || PATIENT_URGENCY[tier];
    space(70);
    const th = doc.font('Helvetica').fontSize(10.5).heightOfString(clean(urg.text), { width: W - 20 }) + 36;
    doc.roundedRect(M, doc.y, W, th, 5).lineWidth(1.2).strokeColor(color).stroke();
    doc.font('Helvetica-Bold').fontSize(10).fillColor(color).text(clean(`Suggested timing: ${urg.timeframe}`), M + 10, doc.y + 8, { width: W - 20 });
    doc.font('Helvetica').fontSize(10.5).fillColor(INK).text(clean(urg.text), M + 10, doc.y + 3, { width: W - 20 });
    doc.y += 14;
    h2('What to do next');
    (report.next_steps || []).forEach((s, i) => {
      space(30);
      const y = doc.y;
      doc.font('Helvetica-Bold').fontSize(10).fillColor(ACCENT).text(`${i + 1}.`, M, y, { width: 18, lineBreak: false });
      doc.font('Helvetica').fontSize(10).fillColor(INK).text(clean(s.step), M + 20, y, { width: W - 20, lineGap: 2 });
      if (s.when) doc.font('Helvetica-Oblique').fontSize(9).fillColor(MUTED).text(clean(s.when), M + 20, doc.y, { width: W - 20 });
      doc.moveDown(0.35);
    });
    h2('See a doctor sooner if');
    (report.see_a_doctor_sooner_if || []).forEach((t) => bullet(t));
    h2('Keeping an eye on it');
    (report.keep_an_eye_on_it || []).forEach((t) => bullet(t));
    h2('Questions you could ask your doctor');
    (report.questions_for_your_doctor || []).forEach((t) => bullet(t, { marker: '?' }));
    if (report.history_note) {
      h2('Earlier scans of this spot');
      para(report.history_note);
    }
  } else {
    h2('Model findings');
    para(report.model_findings);
    const types = pred.type_probs || {};
    if (Object.keys(types).length) {
      space(90);
      doc.font('Helvetica-Bold').fontSize(9.5).fillColor(MUTED).text('Type distribution if the lesion is a skin cancer', M, doc.y);
      doc.moveDown(0.3);
      Object.entries(types).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
        const y = doc.y;
        doc.font('Helvetica').fontSize(9.5).fillColor(INK).text(k, M, y, { width: 70, lineBreak: false });
        doc.rect(M + 74, y + 1, 200, 9).fillColor('#EEF1F3').fill();
        doc.rect(M + 74, y + 1, 200 * v, 9).fillColor(ACCENT).fill();
        doc.fillColor(INK).text(`${Math.round(v * 100)}%`, M + 282, y, { width: 40, lineBreak: false });
        doc.y = y + 15;
      });
      doc.moveDown(0.2);
    }
    if ((pred.per_backbone || []).length) {
      space(60);
      doc.font('Helvetica-Bold').fontSize(9.5).fillColor(MUTED).text('Per-architecture scores (z)', M, doc.y);
      doc.moveDown(0.2);
      doc.font('Helvetica').fontSize(9.5).fillColor(INK).text(clean(pred.per_backbone.map((b) => `${b.name}: ${Number(b.z).toFixed(2)}${b.tier ? ` (${b.tier})` : ''}`).join('   |   ')), M, doc.y, { width: W });
      doc.moveDown(0.4);
    }
    h2('Supporting evidence');
    (report.supporting_evidence || []).forEach((t) => bullet(t));
    h2('Differential considerations');
    (report.differential || []).forEach((d) => {
      space(34);
      doc.font('Helvetica-Bold').fontSize(10).fillColor(INK).text(clean(d.entity), M, doc.y, { width: W });
      para(d.rationale, { size: 9.5, color: MUTED });
    });
    h2('Recommended work-up');
    (report.recommended_workup || []).forEach((t) => bullet(t));
    h2('Management options (conditional on histopathology)');
    (report.management_options || []).forEach((m) => {
      space(40);
      doc.font('Helvetica-Bold').fontSize(10).fillColor(INK).text(clean(m.scenario), M, doc.y, { width: W });
      doc.moveDown(0.15);
      m.options.forEach((o) => bullet(o));
    });
    h2('Medication options (general, no doses)');
    (report.medication_options || []).forEach((m) => {
      space(40);
      doc.font('Helvetica-Bold').fontSize(10).fillColor(INK).text(clean(m.scenario), M, doc.y, { width: W });
      doc.moveDown(0.15);
      m.agents.forEach((o) => bullet(o));
      if (m.notes) para(m.notes, { size: 9, color: MUTED });
    });
    h2('Advisory');
    (report.advisory || []).forEach((t) => bullet(t));
    if (report.history_and_trend) {
      h2('History and trend');
      para(report.history_and_trend);
    }
    h2('Limitations');
    (report.limitations || []).forEach((t) => bullet(t));
  }

  // ------------------------------------------------------------------------------------------------ scan history table
  if (history.length) {
    h2('Scan history for this lesion');
    const cw = [92, 110, 70, 120];
    const hdr = ['Date', 'Rating', role === 'clinician' ? 'Score z' : '', role === 'clinician' ? 'Type if cancer' : ''];
    space(24 + history.length * 15);
    let y = doc.y;
    doc.font('Helvetica-Bold').fontSize(9).fillColor(MUTED);
    let x = M;
    hdr.forEach((t, i) => {
      doc.text(t, x, y, { width: cw[i], lineBreak: false });
      x += cw[i];
    });
    y += 14;
    doc.font('Helvetica').fontSize(9.5).fillColor(INK);
    [{ ...scan, current: true }, ...history].forEach((h) => {
      x = M;
      const cells = [`${fmtDate(h.createdAt)}${h.current ? ' (this scan)' : ''}`, TIERS[h.tier]?.label || h.tier, role === 'clinician' ? (h.score != null ? Number(h.score).toFixed(2) : '') : '', role === 'clinician' ? h.topType || '' : ''];
      cells.forEach((t, i) => {
        doc.fillColor(i === 1 ? TIER_COLOR[h.tier] || INK : INK).text(clean(t), x, y, { width: cw[i], lineBreak: false });
        x += cw[i];
      });
      y += 15;
    });
    doc.y = y + 4;
  }

  // ------------------------------------------------------------------------------------------------ provenance
  h2('About this report');
  const x = scan.xai || {};
  para(
    `Model: ${model?.version ? `Cutis image ensemble ${model.version}` : 'Cutis image ensemble'}${model?.created ? ` (trained ${model.created})` : ''}; ${model?.networks ?? '?'} networks on ${(model?.backbones || []).join(', ') || 'n/a'}. ` +
      `Explanations: ${x.gradcam?.title || 'Grad-CAM'}, LIME${x.lime?.summary?.n_samples ? ` (${x.lime.summary.n_samples} samples)` : ''}, SHAP${x.shap?.summary?.n_samples ? ` (${x.shap.summary.n_samples} samples)` : ''}. ` +
      `Explanatory text: ${meta?.source === 'groq' ? `written by an AI language model (${meta.model}) from the results above and checked against fixed safety rules` : 'standard template text (no language model was used)'}. ` +
      `Prepared for ${user.name}${user.organisation ? `, ${user.organisation}` : ''}.`,
    { size: 8.5, color: MUTED },
  );

  // ------------------------------------------------------------------------------------------------ footer
  const range = doc.bufferedPageRange();
  for (let i = 0; i < range.count; i++) {
    doc.switchToPage(i);
    const b = doc.page.margins.bottom;
    doc.page.margins.bottom = 0;
    doc.font('Helvetica').fontSize(8).fillColor(MUTED).text(`Cutis research prototype, not a medical device  |  Page ${i + 1} of ${range.count}`, M, doc.page.height - 30, { width: W, align: 'center', lineBreak: false });
    doc.page.margins.bottom = b;
  }
  return doc;
}
