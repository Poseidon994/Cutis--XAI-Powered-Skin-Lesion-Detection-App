import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import { Banner, TierChip } from '../components/Bits.jsx';
import { Lens } from '../components/Lens.jsx';
import { fmtDate, firstName, siteLabel } from '../format.js';
import { useHealth, useTitle } from '../hooks.js';

export function ScanRow({ s, clinician }) {
  return (
    <Link to={`/scan/${s.id}`} className="scan-row">
      <Lens src={s.thumb} size={56} ring={4} alt="" />
      <div>
        <div className="title">{s.lesionLabel || siteLabel(s.bodySite)}</div>
        <div className="meta">{fmtDate(s.createdAt)}{clinician && s.patientRef ? `, ${s.patientRef}` : ''}{s.lesionLabel ? `, ${siteLabel(s.bodySite)}` : ''}</div>
      </div>
      <div><TierChip tier={s.tier} /></div>
      <div className="score num">{clinician && s.score != null ? `z ${s.score.toFixed(2)}` : ''}</div>
    </Link>
  );
}

export default function Home() {
  const { user } = useAuth();
  const clinician = user.role === 'clinician';
  useTitle('Home');
  const health = useHealth();
  const [data, setData] = useState(null);
  useEffect(() => {
    api.get('/scans?limit=8').then(setData).catch(() => setData({ scans: [], total: 0 }));
  }, []);
  return (
    <>
      <div className="page-head">
        <div>
          <h1>{clinician ? 'Clinic workspace' : `Hello, ${firstName(user.name)}`}</h1>
          <p>{clinician ? 'Analyse a lesion photo, review the evidence behind the score, and keep a history per patient.' : 'Take a close-up photo of a spot you are unsure about and see how it rates. You can bring the report to your doctor.'}</p>
        </div>
        <Link className="btn big" to="/scan/new">{clinician ? 'New analysis' : 'Check a spot'}</Link>
      </div>
      {health && !health.ml?.ready && <Banner kind="warn">The analysis engine is {health.ml?.status === 'loading' ? 'still loading' : 'not reachable'}. New scans will work as soon as it is ready.</Banner>}
      {clinician && health && !health.agent?.configured && <Banner kind="info">AI-written summaries are off because no Groq API key is configured. Standard template text is used instead.</Banner>}
      <section className="section" style={{ marginTop: 10 }}>
        <h2>{clinician ? 'Recent analyses' : 'Your recent checks'}</h2>
        {!data ? <p className="muted">Loading...</p> : data.scans.length === 0 ? (
          <p className="empty">Nothing here yet. {clinician ? 'Run your first analysis' : 'Check your first spot'} to start a history you can compare over time.</p>
        ) : (
          <>
            <div className="scan-list">{data.scans.map((s) => <ScanRow key={s.id} s={s} clinician={clinician} />)}</div>
            {data.total > data.scans.length && <p style={{ marginTop: 14 }}><Link to="/history">See all {data.total} scans</Link></p>}
          </>
        )}
      </section>
      <p className="footer-note">Cutis is a research prototype and not a medical device. {clinician ? 'Outputs support, and never replace, clinical judgement.' : 'It cannot tell you whether a spot is cancer. If you are worried, see a doctor.'}</p>
    </>
  );
}
