import { Fragment, useState } from 'react';
import { Lightbox, Spinner } from './Bits.jsx';

const ORDER = ['gradcam', 'lime', 'shap'];

const PATIENT_CAPTION = {
  gradcam: 'Brighter areas are where the AI paid the most attention.',
  lime: 'Red areas pushed the rating up. Blue areas pushed it down.',
  shap: 'A second way of showing what pushed the rating up (red) or down (blue).',
};

function clinicianCaption(m, e) {
  const s = e.summary || {};
  if (m === 'gradcam') return `Last-stage activation map, averaged over ${s.n_networks} networks. Network agreement ${s.network_agreement ?? 'n/a'}.`;
  if (m === 'lime') return `SLIC superpixels, ${s.n_samples} perturbations. Local surrogate R2 ${s.local_r2 ?? 'n/a'}.`;
  return `Expected gradients (${s.engine || 'SHAP'}), ${s.n_samples} samples. Completeness gap ${s.completeness_gap ?? 'n/a'} z.`;
}

function Legend({ signed }) {
  return (
    <>
      <div className={`legend-bar ${signed ? 'div' : 'seq'}`} aria-hidden="true" />
      <div className="legend-ends"><span>{signed ? 'lowers' : 'less'}</span><span>{signed ? 'raises' : 'more'}</span></div>
    </>
  );
}

export function XaiPanel({ scan, role, running, onRetry }) {
  const [open, setOpen] = useState(null);
  const x = scan.xai || {};
  return (
    <>
      <div className="lens-row">
        <div className="lens-cell">
          <div className="lens-box"><img src={scan.files.modelView} alt="The lesion photo exactly as the AI saw it" /></div>
          <h3>Photo analysed</h3>
          <p>Centre of your photo, as the AI saw it.</p>
        </div>
        {ORDER.map((m) => {
          const e = x[m] || { status: 'pending' };
          const busy = running[m] || (e.status === 'pending' && !e.error);
          const title = e.title || { gradcam: 'Grad-CAM', lime: 'LIME', shap: 'SHAP' }[m];
          return (
            <div className="lens-cell" key={m}>
              {e.status === 'done' ? (
                <button type="button" className="lens-box" onClick={() => setOpen(m)} aria-label={`Enlarge the ${title} explanation`}>
                  <img src={e.url} alt={`${title} explanation map over the lesion`} />
                </button>
              ) : (
                <div className="lens-box" aria-live="polite">
                  {busy ? <span><Spinner /><br />Running {title}</span> : <span>Could not run<br />{title}</span>}
                </div>
              )}
              <h3>{title}</h3>
              {e.status === 'done' ? (
                <>
                  <p>{role === 'patient' ? PATIENT_CAPTION[m] : clinicianCaption(m, e)}</p>
                  <Legend signed={Boolean(e.signed)} />
                  {role === 'clinician' && e.summary?.focus && e.summary.focus !== 'none' && (
                    <p className="tiny">Evidence is {e.summary.focus}{e.summary.direction && e.summary.direction !== 'centre' ? `, towards the ${e.summary.direction}` : ''}.</p>
                  )}
                </>
              ) : !busy ? (
                <>
                  <p className="small" style={{ color: 'var(--high)' }}>{e.error || 'This explanation did not finish.'}</p>
                  <button className="btn secondary" onClick={() => onRetry(m)}>Try again</button>
                </>
              ) : (
                <p>This can take up to a minute on a computer without a graphics card.</p>
              )}
            </div>
          );
        })}
      </div>
      {role === 'clinician' && x.agreement && (
        <p className="small muted" style={{ marginTop: 20 }}>
          Agreement between the three maps: <strong>{x.agreement.level}</strong> (mean overlap of the top 20% areas {x.agreement.mean_iou_top20}; two unrelated maps would score about {x.agreement.chance_iou_top20}).
        </p>
      )}
      <p className="tiny" style={{ marginTop: 14, maxWidth: '68ch' }}>
        These maps show what the AI attended to. They help you judge whether it looked at the lesion rather than at hair, ruler marks or the photo edge. They are not proof of what the lesion is.
      </p>
      <Lightbox open={Boolean(open)} onClose={() => setOpen(null)} label="Explanation map">
        {open && x[open]?.url && (
          <>
            <h3>{x[open].title}</h3>
            <img src={x[open].url} alt={`${x[open].title} explanation map, enlarged`} />
            <p className="small muted">{role === 'patient' ? PATIENT_CAPTION[open] : clinicianCaption(open, x[open])}</p>
            {role === 'clinician' && (
              <dl className="kv small">
                {Object.entries(x[open].summary || {}).filter(([, v]) => typeof v !== 'object').map(([k, v]) => (
                  <Fragment key={k}><dt>{k.replaceAll('_', ' ')}</dt><dd className="num">{String(v)}</dd></Fragment>
                ))}
              </dl>
            )}
          </>
        )}
      </Lightbox>
    </>
  );
}
