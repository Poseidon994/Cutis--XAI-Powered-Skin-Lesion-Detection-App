import { useState } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth.jsx';
import { Banner, Spinner } from '../components/Bits.jsx';
import { Brand } from '../components/Shell.jsx';
import { useHealth, useTitle } from '../hooks.js';

export default function AuthPage() {
  useTitle('Sign in');
  const { user, login, register } = useAuth();
  const nav = useNavigate();
  const health = useHealth();
  const [mode, setMode] = useState('login');
  const [role, setRole] = useState('patient');
  const [f, setF] = useState({ name: '', email: '', password: '', inviteCode: '', organisation: '' });
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  if (user) return <Navigate to="/" replace />;
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  const clinicianOk = health ? health.clinicianSignup !== false : true;

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setErr('');
    try {
      if (mode === 'login') await login(f.email, f.password);
      else await register({ name: f.name, email: f.email, password: f.password, role, inviteCode: f.inviteCode || undefined, organisation: f.organisation || undefined });
      nav('/', { replace: true });
    } catch (ex) {
      setErr(ex.message);
      setBusy(false);
    }
  };

  return (
    <div className="auth">
      <section className="auth-hero">
        <Brand />
        <div>
          <div className="hero-lens" aria-hidden="true" />
          <h1>A closer look at a spot on your skin.</h1>
          <ul>
            <li>Upload a close-up photo of one lesion.</li>
            <li>See what the AI looked at, in three different ways.</li>
            <li>Get an explanation written for you, and a report to take to a doctor.</li>
          </ul>
        </div>
        <p className="tiny" style={{ maxWidth: '52ch' }}>Cutis is a research prototype. It cannot diagnose skin cancer and it can miss cancers. It does not replace a doctor.</p>
      </section>
      <section className="auth-panel">
        <h2>{mode === 'login' ? 'Sign in' : 'Create your account'}</h2>
        <p className="muted">{mode === 'login' ? 'Welcome back.' : 'Choose the account that fits how you will use Cutis.'}</p>
        {health && !health.ml?.ready && <Banner kind="warn">The analysis engine is not ready yet ({health.ml?.status === 'loading' ? 'it is still loading' : 'it cannot be reached'}). You can sign in, but new scans may fail for a minute.</Banner>}
        <form onSubmit={submit} noValidate>
          {mode === 'register' && (
            <>
              <div className="role-pick" role="radiogroup" aria-label="Account type">
                <label>
                  <input type="radio" name="role" value="patient" checked={role === 'patient'} onChange={() => setRole('patient')} />
                  <strong>Patient</strong>
                  <small>Check a spot, get a plain-language explanation and next steps.</small>
                </label>
                <label style={clinicianOk ? undefined : { opacity: 0.55 }}>
                  <input type="radio" name="role" value="clinician" checked={role === 'clinician'} onChange={() => clinicianOk && setRole('clinician')} disabled={!clinicianOk} />
                  <strong>Clinician</strong>
                  <small>Scores, evidence, differential and management options.</small>
                </label>
              </div>
              <div className="field">
                <label htmlFor="name">Full name</label>
                <input id="name" type="text" autoComplete="name" value={f.name} onChange={set('name')} required />
              </div>
            </>
          )}
          <div className="field">
            <label htmlFor="email">Email</label>
            <input id="email" type="email" autoComplete="email" value={f.email} onChange={set('email')} required />
          </div>
          <div className="field">
            <label htmlFor="password">Password</label>
            <input id="password" type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} value={f.password} onChange={set('password')} required minLength={mode === 'register' ? 8 : undefined} />
            {mode === 'register' && <span className="hint">At least 8 characters.</span>}
          </div>
          {mode === 'register' && role === 'clinician' && (
            <>
              <div className="field">
                <label htmlFor="org">Clinic or hospital (optional)</label>
                <input id="org" type="text" value={f.organisation} onChange={set('organisation')} />
              </div>
              {health?.clinicianInviteRequired && (
                <div className="field">
                  <label htmlFor="invite">Clinician invite code</label>
                  <input id="invite" type="text" autoComplete="off" value={f.inviteCode} onChange={set('inviteCode')} required />
                  <span className="hint">Ask the person who runs this installation.</span>
                </div>
              )}
            </>
          )}
          {err && <Banner kind="error">{err}</Banner>}
          <button className="btn big" disabled={busy} style={{ width: '100%' }}>{busy ? <Spinner /> : mode === 'login' ? 'Sign in' : 'Create account'}</button>
        </form>
        <p className="auth-switch">
          {mode === 'login' ? 'New here? ' : 'Already have an account? '}
          <a href="#switch" onClick={(e) => { e.preventDefault(); setErr(''); setMode(mode === 'login' ? 'register' : 'login'); }}>{mode === 'login' ? 'Create an account' : 'Sign in'}</a>
        </p>
      </section>
    </div>
  );
}
