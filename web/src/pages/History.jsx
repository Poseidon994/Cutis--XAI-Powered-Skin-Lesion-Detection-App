import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import { TrendChart } from '../components/Bits.jsx';
import { TIERS, siteLabel } from '../format.js';
import { useTitle } from '../hooks.js';
import { ScanRow } from './Home.jsx';

export default function History() {
  const { user } = useAuth();
  const clinician = user.role === 'clinician';
  useTitle('History');
  const [search, setSearch] = useState('');
  const [data, setData] = useState(null);
  const [pick, setPick] = useState('');
  useEffect(() => {
    const t = setTimeout(() => api.get(`/scans?limit=500&search=${encodeURIComponent(search)}`).then(setData).catch(() => setData({ scans: [], total: 0 })), 200);
    return () => clearTimeout(t);
  }, [search]);

  const groups = useMemo(() => {
    const m = new Map();
    (data?.scans || []).forEach((s) => {
      const key = `${(s.patientRef || '').trim().toLowerCase()}|${(s.lesionLabel || '').trim().toLowerCase() || `site:${s.bodySite}`}`;
      if (!m.has(key)) m.set(key, { key, name: [clinician ? s.patientRef : null, s.lesionLabel || siteLabel(s.bodySite)].filter(Boolean).join(', '), scans: [] });
      m.get(key).scans.push(s);
    });
    return [...m.values()].filter((g) => g.scans.length >= 2);
  }, [data, clinician]);
  const group = groups.find((g) => g.key === pick) || groups[0];
  const points = group ? [...group.scans].reverse().map((s) => ({ date: s.createdAt, value: clinician ? s.score : TIERS[s.tier].rank, tier: s.tier })) : [];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>History</h1>
          <p>{clinician ? 'Every analysis you have run. Search by patient reference, lesion name or body site.' : 'Every spot you have checked. Photos of the same spot, saved under the same name, are linked so you can see whether it changes.'}</p>
        </div>
        <Link className="btn" to="/scan/new">{clinician ? 'New analysis' : 'Check a spot'}</Link>
      </div>
      <div className="toolbar">
        <label className="sr" htmlFor="q">Search scans</label>
        <input id="q" type="search" placeholder={clinician ? 'Search patient reference or lesion' : 'Search by name or body part'} value={search} onChange={(e) => setSearch(e.target.value)} />
        {data && <span className="muted small">{data.scans.length} shown</span>}
      </div>
      {!data ? <p className="muted">Loading...</p> : data.scans.length === 0 ? (
        <p className="empty">{search ? 'No scans match that search.' : 'No scans yet.'}</p>
      ) : (
        <div className="scan-list">{data.scans.map((s) => <ScanRow key={s.id} s={s} clinician={clinician} />)}</div>
      )}
      {group && (
        <section className="section">
          <h2>Change over time</h2>
          <div className="toolbar" style={{ marginTop: 12 }}>
            <label htmlFor="g" className="small muted">Spot</label>
            <select id="g" style={{ maxWidth: 340 }} value={group.key} onChange={(e) => setPick(e.target.value)}>{groups.map((g) => <option key={g.key} value={g.key}>{g.name} ({g.scans.length} scans)</option>)}</select>
          </div>
          <TrendChart points={points} mode={clinician ? 'score' : 'tier'} />
          <p className="tiny" style={{ marginTop: 8 }}>{clinician ? 'Image-ensemble score (z) per scan. Differences below about 0.5 z are within normal photo-to-photo variation.' : 'Concern level of each photo. A spot that keeps moving up is a good reason to see a doctor.'}</p>
        </section>
      )}
    </>
  );
}
