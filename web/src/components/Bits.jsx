import { useEffect, useRef } from 'react';
import { TIERS, fmtDate } from '../format.js';

export const Spinner = () => <span className="spinner" aria-hidden="true" />;

export function Banner({ kind = 'info', children }) {
  return (
    <div className={`banner ${kind}`} role={kind === 'error' ? 'alert' : 'status'}>
      {children}
    </div>
  );
}

export const TierChip = ({ tier }) => <span className={`tier tier-chip ${tier}`}>{TIERS[tier]?.label || tier}</span>;

export function TypeBars({ probs }) {
  const rows = Object.entries(probs || {}).sort((a, b) => b[1] - a[1]);
  return (
    <div className="bars" role="list">
      {rows.map(([k, v]) => (
        <div className="bar" role="listitem" key={k}>
          <span>{k}</span>
          <span className="track" aria-hidden="true">
            <span className="fill" style={{ width: `${Math.round(v * 100)}%`, display: 'block' }} />
          </span>
          <span className="v num">{Math.round(v * 100)}%</span>
        </div>
      ))}
    </div>
  );
}

export function Lightbox({ open, onClose, children, label }) {
  const ref = useRef(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog ref={ref} className="light" onClose={onClose} onClick={(e) => e.target === ref.current && onClose()} aria-label={label}>
      <div className="light-body">
        {children}
        <button className="btn secondary" onClick={onClose}>Close</button>
      </div>
    </dialog>
  );
}

// Concern level (patients) or score (clinicians) across scans of the same lesion.
export function TrendChart({ points, mode = 'tier' }) {
  if (!points || points.length < 2) return <p className="muted small">A trend appears here once there are at least two scans of the same lesion.</p>;
  const W = 620;
  const H = 210;
  const pad = { l: 72, r: 18, t: 14, b: 34 };
  const t0 = new Date(points[0].date).getTime();
  const t1 = new Date(points[points.length - 1].date).getTime();
  const span = Math.max(t1 - t0, 1);
  const vals = points.map((p) => p.value);
  const lo = mode === 'tier' ? 0 : Math.min(...vals, 0) - 0.3;
  const hi = mode === 'tier' ? 3 : Math.max(...vals) + 0.3;
  const x = (d) => pad.l + ((new Date(d).getTime() - t0) / span) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  const ticks = mode === 'tier' ? [0, 1, 2, 3].map((v) => [v, Object.values(TIERS)[v].label]) : [lo, (lo + hi) / 2, hi].map((v) => [v, v.toFixed(1)]);
  const colour = { low: 'var(--low)', guarded: 'var(--guarded)', elevated: 'var(--elevated)', high: 'var(--high)' };
  return (
    <svg className="trend" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Trend over ${points.length} scans, from ${fmtDate(points[0].date)} to ${fmtDate(points[points.length - 1].date)}`}>
      {ticks.map(([v, label]) => (
        <g key={v}>
          <line className="grid" x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} />
          <text x={pad.l - 8} y={y(v) + 4} textAnchor="end">{label}</text>
        </g>
      ))}
      <polyline className="line" points={points.map((p) => `${x(p.date)},${y(p.value)}`).join(' ')} />
      {points.map((p, i) => (
        <g key={i}>
          <circle cx={x(p.date)} cy={y(p.value)} r="5" fill={colour[p.tier] || 'var(--accent)'} stroke="#fff" strokeWidth="2" />
          {(i === 0 || i === points.length - 1 || points.length < 6) && (
            <text x={x(p.date)} y={H - 10} textAnchor={i === 0 ? 'start' : i === points.length - 1 ? 'end' : 'middle'}>{fmtDate(p.date)}</text>
          )}
        </g>
      ))}
    </svg>
  );
}
