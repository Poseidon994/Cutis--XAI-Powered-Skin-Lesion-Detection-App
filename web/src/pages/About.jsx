import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import { Banner, Spinner } from '../components/Bits.jsx';
import { useHealth, useTitle } from '../hooks.js';

const fmt = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : String(v ?? ''));

export default function About() {
  const { user } = useAuth();
  const clinician = user.role === 'clinician';
  useTitle('About');
  const health = useHealth();
  const [card, setCard] = useState(null);
  const [err, setErr] = useState('');
  useEffect(() => {
    api.get('/model-card').then(setCard).catch((e) => setErr(e.message));
  }, []);
  const rows = card?.results?.image_only;
  const cols = rows?.length ? Object.keys(rows[0]).filter((k) => rows.some((r) => typeof r[k] === 'number' || k === 'model' || k === 'name')).slice(0, 6) : [];
  const cal = card?.image_only_calibration;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>About this tool</h1>
          <p>What Cutis does, how it was built, and where it can go wrong.</p>
        </div>
      </div>
      <div className="prose">
        <h3>What it does</h3>
        <p>Cutis looks at one close-up photo of a skin lesion and rates how concerning it looks, from low to high. It then shows which parts of the photo drove that rating, three different ways, and a writing assistant explains the result {clinician ? 'for a clinician' : 'in plain language'}.</p>
        <h3>How it was built</h3>
        <p>The rating comes from an ensemble of image networks (four different architectures, several copies each) trained on the ISIC 2024 dataset: about 401,000 small lesion crops from 3D total-body photography at nine centres, of which only 393 were skin cancers. The full research pipeline also uses body-mapping measurements and comparisons with the patient's other lesions. A single photo has neither, so this app uses the image part only.</p>
        <h3>The three explanation methods</h3>
        <ul>
          <li><strong>Grad-CAM</strong> follows the network's internal activations to show where it paid attention. It works on all four architectures here, so it is the main attention map. Where gradients vanish, Score-CAM is used instead.</li>
          <li><strong>LIME</strong> hides parts of the photo (replacing them with surrounding skin colour) and measures how the rating moves, then fits a simple model to those changes.</li>
          <li><strong>SHAP</strong> shares out the difference between plain skin and this lesion across the pixels, in a way that adds up to the rating.</li>
        </ul>
        <h3>Where it can go wrong</h3>
        <ul>{(card?.limitations || ['Research prototype. Not validated on phone photos. It can miss cancers.']).map((t, i) => <li key={i}>{t}</li>)}</ul>
      </div>
      {err && <Banner kind="warn">{err}</Banner>}
      {!card && !err && <p className="muted"><Spinner /> Loading model details...</p>}
      {card && clinician && (
        <section className="section">
          <h2>Model card</h2>
          <dl className="kv" style={{ marginTop: 12 }}>
            <dt>Version</dt><dd>{card.version} (library {card.lib_version}), trained {card.created}, profile {card.profile}</dd>
            <dt>Backbones</dt><dd>{Object.entries(card.backbones || {}).map(([n, b]) => `${n} (${b.resolution}px)`).join(', ')}</dd>
            <dt>Networks loaded</dt><dd>{card.networks_loaded} of {Object.keys(card.backbones || {}).length * card.folds_trained}, {card.tta_views} views each, on {card.device}</dd>
            {card.data && <><dt>Training data</dt><dd>{card.data.n_lesions?.toLocaleString()} lesions, {card.data.n_patients?.toLocaleString()} patients, {card.data.n_malignant} malignant</dd></>}
            {cal?.metrics && <><dt>Image-only score (out-of-fold)</dt><dd>AUROC {cal.metrics.auroc?.toFixed(4)}, partial AUC above 80% TPR {cal.metrics.pauc_tpr80?.toFixed(4)}</dd></>}
            <dt>Calibration</dt><dd>{cal?.source === 'oof' ? `from ${cal.n_lesions?.toLocaleString()} out-of-fold lesions` : 'fallback thresholds (calibration file missing)'}</dd>
          </dl>
          {cal?.operating_points?.length > 0 && (
            <div className="scroll-x" style={{ marginTop: 18 }}>
              <h3 style={{ marginBottom: 6 }}>Operating points behind the rating tiers</h3>
              <table className="data">
                <thead><tr><th>Cut-off</th><th>Score z at or above</th><th>Share of benign lesions below it</th></tr></thead>
                <tbody>{cal.operating_points.map((o) => <tr key={o.rule}><td>{o.label}</td><td className="num">{o.threshold}</td><td className="num">{o.specificity != null ? `${(o.specificity * 100).toFixed(1)}%` : 'n/a'}</td></tr>)}</tbody>
              </table>
              <p className="tiny" style={{ marginTop: 6 }}>Low = below the first cut-off. Some concern = between the first and second. Elevated = between the second and third. High = at or above the third.</p>
            </div>
          )}
          {rows?.length > 0 && (
            <div className="scroll-x" style={{ marginTop: 22 }}>
              <h3 style={{ marginBottom: 6 }}>Image-only results from the training run</h3>
              <table className="data">
                <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
                <tbody>{rows.slice(0, 8).map((r, i) => <tr key={i}>{cols.map((c) => <td key={c} className="num">{fmt(r[c])}</td>)}</tr>)}</tbody>
              </table>
            </div>
          )}
        </section>
      )}
      {health && (
        <p className="footer-note">
          Analysis engine: {health.ml?.ready ? `ready (${health.ml.networks} networks on ${health.ml.device})` : health.ml?.status}. AI writer: {health.agent?.configured ? `Groq, ${health.agent.model}` : 'not configured, standard text is used'}.
        </p>
      )}
    </>
  );
}
