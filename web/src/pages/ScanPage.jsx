import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import { AgentPanel } from '../components/AgentPanel.jsx';
import { Banner, Spinner, TierChip, TrendChart, TypeBars } from '../components/Bits.jsx';
import { Chat } from '../components/Chat.jsx';
import { Lens } from '../components/Lens.jsx';
import { XaiPanel } from '../components/XaiPanel.jsx';
import { TIERS, durationLabel, fmtDate, fmtDateTime, siteLabel, sexLabel, symptomLabel } from '../format.js';
import { useTitle } from '../hooks.js';
import { ScanRow } from './Home.jsx';

const METHODS = ['gradcam', 'lime', 'shap'];
const DEFAULT_LEAD = {
  low: 'This spot looks lower concern on this photo.',
  guarded: 'This spot shows some features that are worth having checked.',
  elevated: 'This spot looks higher concern and should be checked by a doctor.',
  high: 'This spot looks high concern and should be checked soon.',
};

export default function ScanPage() {
  const { id } = useParams();
  const { user } = useAuth();
  const nav = useNavigate();
  const role = user.role;
  const clinician = role === 'clinician';
  const [data, setData] = useState(null);
  const [err, setErr] = useState('');
  const [running, setRunning] = useState({});
  const [agentBusy, setAgentBusy] = useState(false);
  const [agentErr, setAgentErr] = useState('');
  const started = useRef({});
  const agentStarted = useRef(false);
  const scan = data?.scan;
  useTitle(scan ? TIERS[scan.tier]?.label : 'Result');

  useEffect(() => {
    started.current = {};
    agentStarted.current = false;
    setData(null);
    setErr('');
    api.get(`/scans/${id}`).then(setData).catch((e) => setErr(e.message));
  }, [id]);

  const runExplain = useCallback(async (m, force = false) => {
    setRunning((r) => ({ ...r, [m]: true }));
    const merge = (x, fallback) => setData((d) => d && { ...d, scan: { ...d.scan, xai: { ...d.scan.xai, [m]: x?.[m] ?? fallback, agreement: x?.agreement ?? d.scan.xai.agreement } } });
    try {
      const r = await api.post(`/scans/${id}/explain/${m}${force ? '?force=1' : ''}`);
      merge(r.xai);
    } catch (e) {
      merge(e.data?.xai, { status: 'failed', error: e.message });
    } finally {
      setRunning((r) => ({ ...r, [m]: false }));
    }
  }, [id]);

  const generate = useCallback(async (force = false) => {
    setAgentBusy(true);
    setAgentErr('');
    try {
      const r = await api.post(`/scans/${id}/agent/report`, { force });
      setData((d) => d && { ...d, scan: { ...d.scan, agent: r.agent } });
    } catch (e) {
      setAgentErr(e.message);
    } finally {
      setAgentBusy(false);
    }
  }, [id]);

  // The three explanation methods start together, as soon as the scan is on screen.
  useEffect(() => {
    if (!scan) return;
    METHODS.forEach((m) => {
      if (scan.xai[m].status !== 'done' && !started.current[m]) {
        started.current[m] = true;
        runExplain(m);
      }
    });
  }, [scan?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const settled = scan && METHODS.every((m) => ['done', 'failed'].includes(scan.xai[m].status)) && !Object.values(running).some(Boolean);
  useEffect(() => {
    if (settled && scan.agent.status === 'none' && !agentStarted.current) {
      agentStarted.current = true;
      generate(false);
    }
  }, [settled]); // eslint-disable-line react-hooks/exhaustive-deps

  if (err) return <><Banner kind="error">{err}</Banner><Link to="/history">Back to history</Link></>;
  if (!scan) return <p className="muted"><Spinner /> Loading...</p>;

  const p = scan.prediction;
  const rank = TIERS[scan.tier].rank;
  const fraction = (p.percentile_benign ?? [25, 50, 75, 100][rank]) / 100;
  const report = scan.agent.status === 'ready' ? scan.agent.report : null;
  const lead = clinician ? report?.summary || `Image-ensemble score z = ${p.score}. Higher than ${p.percentile_benign}% of benign lesions in the validation data.` : report?.headline || DEFAULT_LEAD[scan.tier];
  const g = scan.guidance;
  const history = data.history || [];
  const points = [...history, { createdAt: scan.createdAt, score: scan.score, tier: scan.tier }].reverse().map((h) => ({ date: h.createdAt, value: clinician ? h.score : TIERS[h.tier].rank, tier: h.tier }));
  const remove = async () => {
    if (!window.confirm('Delete this scan and its images permanently?')) return;
    await api.del(`/scans/${id}`);
    nav('/history');
  };

  return (
    <>
      <div className={`verdict tier ${scan.tier}`}>
        <Lens src={scan.files.modelView} size={clinician ? 230 : 270} tier={scan.tier} fraction={fraction} alt={`${TIERS[scan.tier].label}. The ring shows how this photo's score compares with harmless spots.`} />
        <div>
          <h1>{TIERS[scan.tier].label}</h1>
          <p className="lead">{lead}</p>
          <div className="guidance">
            <strong>{g.kind === 'urgency' ? `Suggested timing: ${g.timeframe}` : 'Disposition rule for this tier'}</strong>
            <div>{g.text}</div>
          </div>
          <div className="actions">
            <a className="btn" href={`/api/scans/${id}/report.pdf`} download>Download report (PDF)</a>
            <Link className="btn secondary" to="/scan/new">{clinician ? 'New analysis' : 'Check another spot'}</Link>
          </div>
          <p className="tiny" style={{ marginTop: 12 }}>The ring shows how this photo scored compared with harmless spots the AI was tested on. It is not the chance of cancer. {scan.tier === 'low' || scan.tier === 'guarded' ? 'A lower rating never rules out skin cancer.' : ''}</p>
        </div>
      </div>

      <div style={{ marginTop: 26 }}>
        {p.source === 'fallback' && <Banner kind="warn">{p.note || 'Calibration data was not found, so the rating is approximate.'}</Banner>}
        {(scan.quality?.warnings || []).map((w) => <Banner key={w.code} kind="warn">{w.text}</Banner>)}
      </div>

      {clinician && (
        <section className="section">
          <h2>Model output</h2>
          <div className="cols2" style={{ marginTop: 14 }}>
            <dl className="kv num">
              <dt>Rating</dt><dd><TierChip tier={scan.tier} /></dd>
              <dt>Image-ensemble score</dt><dd>z = {p.score}</dd>
              <dt>Percentile among benign</dt><dd>{p.percentile_benign}</dd>
              {p.sensitivity_at_or_above !== undefined && <><dt>Sensitivity at or above</dt><dd>{(p.sensitivity_at_or_above * 100).toFixed(1)}%</dd></>}
              {p.cancer_rate_at_or_above !== undefined && <><dt>Observed cancer rate at or above</dt><dd>{(p.cancer_rate_at_or_above * 1000).toFixed(1)} per 1,000 (base rate {(p.base_rate * 1000).toFixed(2)} per 1,000, {p.enrichment}x)</dd></>}
              <dt>Operating points (z)</dt><dd>95% sens {p.thresholds.sens95}, 90% sens {p.thresholds.sens90}, 80% sens {p.thresholds.sens80}</dd>
              <dt>Networks</dt><dd>{p.n_networks} ({p.tta_views} views each), spread SD {p.network_sd?.toFixed(2) ?? 'n/a'}</dd>
              {p.agreement && <><dt>Backbone agreement</dt><dd>{p.agreement.same_tier_as_ensemble} of {p.agreement.backbones} share this tier</dd></>}
            </dl>
            <div>
              <h3 style={{ marginBottom: 10 }}>Type if it is a skin cancer</h3>
              <TypeBars probs={p.type_probs} />
              <p className="tiny" style={{ marginTop: 8 }}>Conditional on malignancy. It is not the probability that the lesion is cancer, and SCC estimates are the least reliable.</p>
              <div className="scroll-x" style={{ marginTop: 18 }}>
                <table className="data">
                  <thead><tr><th>Architecture</th><th>z</th><th>Tier</th><th>Fold SD</th></tr></thead>
                  <tbody>{p.per_backbone.map((b) => <tr key={b.name}><td>{b.name}</td><td className="num">{b.z.toFixed(2)}</td><td>{b.tier ? <TierChip tier={b.tier} /> : 'n/a'}</td><td className="num">{b.fold_sd?.toFixed(2) ?? 'n/a'}</td></tr>)}</tbody>
                </table>
              </div>
            </div>
          </div>
        </section>
      )}

      <section className="section">
        <h2>What the AI looked at</h2>
        <p className="lede">{clinician ? 'Three independent explanation methods run on the same photo at the same time.' : 'Three different methods show which parts of your photo mattered most. They run side by side so you can see whether they agree.'}</p>
        <XaiPanel scan={scan} role={role} running={running} onRetry={(m) => runExplain(m, true)} />
        {!clinician && report?.what_the_images_show && <p className="prose" style={{ marginTop: 18 }}>{report.what_the_images_show}</p>}
      </section>

      <section className="section">
        <h2>{clinician ? 'Clinical summary' : 'Your explanation'}</h2>
        <p className="lede">{clinician ? 'Written from the results above by the AI assistant, with fixed safety wording added by the server.' : 'Written in plain language from your results. It does not replace advice from a doctor.'}</p>
        <AgentPanel scan={scan} role={role} busy={agentBusy} waiting={!report && !settled} error={agentErr} onGenerate={generate} />
      </section>

      {history.length > 0 && (
        <section className="section">
          <h2>{clinician ? 'History of this lesion' : 'Earlier photos of this spot'}</h2>
          <p className="lede">{history.length} earlier scan{history.length > 1 ? 's' : ''} with the same name{clinician ? ' and patient reference' : ''}.</p>
          <TrendChart points={points} mode={clinician ? 'score' : 'tier'} />
          <div className="scan-list" style={{ marginTop: 12 }}>{history.map((h) => <ScanRow key={h.id} s={h} clinician={clinician} />)}</div>
        </section>
      )}

      <section className="section">
        <h2>Ask about this result</h2>
        <p className="lede">{clinician ? 'Follow-up questions about this scan, answered from its results.' : 'Not sure what something means? Ask.'}</p>
        <Chat scanId={id} role={role} initial={data.messages} />
      </section>

      <section className="section">
        <h2>About this scan</h2>
        <dl className="kv" style={{ marginTop: 12 }}>
          <dt>Scanned</dt><dd>{fmtDateTime(scan.createdAt)}</dd>
          <dt>Body site</dt><dd>{siteLabel(scan.bodySite)}</dd>
          {scan.lesionLabel && <><dt>Name</dt><dd>{scan.lesionLabel}</dd></>}
          {clinician && scan.patientRef && <><dt>Patient reference</dt><dd>{scan.patientRef}</dd></>}
          {(scan.age != null || scan.sex) && <><dt>Age and sex</dt><dd>{[scan.age != null ? `${scan.age} years` : null, scan.sex ? sexLabel(scan.sex) : null].filter(Boolean).join(', ')}</dd></>}
          {scan.symptoms.length > 0 && <><dt>Symptoms</dt><dd>{scan.symptoms.map(symptomLabel).join(', ')}</dd></>}
          {scan.duration && <><dt>Duration</dt><dd>{durationLabel(scan.duration)}</dd></>}
          {scan.notes && <><dt>Notes</dt><dd>{scan.notes}</dd></>}
          <dt>Model</dt><dd>{scan.model?.networks} networks, {scan.model?.backbones?.join(', ')} (trained {scan.model?.created})</dd>
        </dl>
        <p style={{ marginTop: 22 }}><button className="btn danger" onClick={remove}>Delete this scan</button></p>
      </section>
    </>
  );
}
