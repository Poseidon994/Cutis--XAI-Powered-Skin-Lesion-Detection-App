import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import { Banner, Spinner } from '../components/Bits.jsx';
import { LensCropper } from '../components/LensCropper.jsx';
import { BODY_SITES, DURATIONS, SEX, SYMPTOMS } from '../format.js';
import { useHealth, useTitle } from '../hooks.js';

export default function NewScan() {
  const { user } = useAuth();
  const clinician = user.role === 'clinician';
  useTitle(clinician ? 'New analysis' : 'Check a spot');
  const nav = useNavigate();
  const health = useHealth();
  const cropper = useRef(null);
  const [file, setFile] = useState(null);
  const [f, setF] = useState({ bodySite: 'unknown', lesionLabel: '', patientRef: '', age: '', sex: '', duration: '', notes: '' });
  const [symptoms, setSymptoms] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  const toggle = (k) => setSymptoms((s) => (s.includes(k) ? s.filter((x) => x !== k) : [...s, k]));

  const submit = async (e) => {
    e.preventDefault();
    if (!file) {
      setErr('Choose a photo first.');
      return;
    }
    setBusy(true);
    setErr('');
    try {
      const blob = await cropper.current.getBlob();
      const form = new FormData();
      form.append('image', blob, 'lesion.jpg');
      Object.entries({ ...f, symptoms: JSON.stringify(symptoms) }).forEach(([k, v]) => v !== '' && form.append(k, v));
      const r = await api.upload('/scans', form);
      nav(`/scan/${r.scan.id}`);
    } catch (ex) {
      setErr(ex.message);
      setBusy(false);
    }
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>{clinician ? 'New analysis' : 'Check a spot'}</h1>
          <p>One lesion per photo. Move and zoom the photo so the lesion sits inside the circle and fills most of it.</p>
        </div>
      </div>
      {health && !health.ml?.ready && <Banner kind="warn">The analysis engine is not ready yet, so the result may take a moment or fail. Try again in a minute if it does.</Banner>}
      <form className="scan-new" onSubmit={submit}>
        <div>
          {file ? <LensCropper ref={cropper} file={file} /> : (
            <div className="cropper cropper-empty">
              <div>
                <p>Choose a close-up photo of the spot.</p>
                <label className="btn big" htmlFor="photo" style={{ cursor: 'pointer' }}>Choose photo</label>
              </div>
            </div>
          )}
          <input id="photo" className={file ? undefined : 'sr'} type="file" accept="image/jpeg,image/png,image/webp" onChange={(e) => e.target.files[0] && setFile(e.target.files[0])} style={file ? { marginTop: 12 } : undefined} aria-label="Photo of the lesion" />
          <ul className="tips">
            <li>Use daylight or bright, even light. Avoid flash glare and shadows.</li>
            <li>Hold the camera steady, close enough that the spot fills the circle, and wait for it to focus.</li>
            <li>One spot per photo. Move hair away, and do not put a filter on the photo.</li>
            <li>The AI was trained on small, tightly framed crops from body-mapping scans. Photos that look different may be rated less reliably.</li>
          </ul>
        </div>
        <div>
          <div className="field">
            <label htmlFor="site">Where on the body is it?</label>
            <select id="site" value={f.bodySite} onChange={set('bodySite')}>{BODY_SITES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
          </div>
          <div className="row">
            <div className="field">
              <label htmlFor="label">Name for this spot</label>
              <input id="label" type="text" maxLength={80} value={f.lesionLabel} onChange={set('lesionLabel')} placeholder="e.g. left shin mole" />
              <span className="hint">Use the same name each time to link scans of the same spot.</span>
            </div>
            {clinician && (
              <div className="field">
                <label htmlFor="ref">Patient reference</label>
                <input id="ref" type="text" maxLength={80} value={f.patientRef} onChange={set('patientRef')} placeholder="e.g. MRN-0042" />
                <span className="hint">An ID, not a name. It is kept on this server and never sent to the AI writer.</span>
              </div>
            )}
          </div>
          <div className="row">
            <div className="field">
              <label htmlFor="age">Age (optional)</label>
              <input id="age" type="number" min="0" max="120" value={f.age} onChange={set('age')} />
            </div>
            <div className="field">
              <label htmlFor="sex">Sex (optional)</label>
              <select id="sex" value={f.sex} onChange={set('sex')}><option value="">Not given</option>{SEX.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
            </div>
          </div>
          <fieldset>
            <legend className="legend">Anything you have noticed? (optional)</legend>
            <div className="chips" style={{ marginTop: 8 }}>
              {SYMPTOMS.map(([v, l]) => (
                <label className="chip-toggle" key={v}><input type="checkbox" checked={symptoms.includes(v)} onChange={() => toggle(v)} /><span>{l}</span></label>
              ))}
            </div>
          </fieldset>
          <div className="field">
            <label htmlFor="dur">How long has it been there?</label>
            <select id="dur" value={f.duration} onChange={set('duration')}><option value="">Not given</option>{DURATIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
          </div>
          <div className="field">
            <label htmlFor="notes">Notes (optional)</label>
            <textarea id="notes" maxLength={1000} value={f.notes} onChange={set('notes')} />
            <span className="hint">Do not enter names or contact details. These notes are shared with the AI writer (Groq) to personalise the explanation. Your photo is not.</span>
          </div>
          {err && <Banner kind="error">{err}</Banner>}
          <button className="btn big" disabled={busy || !file}>{busy ? <><Spinner /> Analysing, this can take a little while...</> : 'Analyse this spot'}</button>
        </div>
      </form>
    </>
  );
}
