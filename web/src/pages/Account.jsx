import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api.js';
import { useAuth } from '../auth.jsx';
import { Banner } from '../components/Bits.jsx';
import { useTitle } from '../hooks.js';

export default function Account() {
  const { user, setUser } = useAuth();
  const nav = useNavigate();
  useTitle('Account');
  const [text, setText] = useState('');
  const [err, setErr] = useState('');
  const del = async () => {
    try {
      await api.del('/auth/me');
      setUser(null);
      nav('/login', { replace: true });
    } catch (e) {
      setErr(e.message);
    }
  };
  return (
    <>
      <div className="page-head"><h1>Account</h1></div>
      <dl className="kv">
        <dt>Name</dt><dd>{user.name}</dd>
        <dt>Email</dt><dd>{user.email}</dd>
        <dt>Account type</dt><dd>{user.role === 'patient' ? 'Patient' : 'Clinician'}{user.organisation ? `, ${user.organisation}` : ''}</dd>
      </dl>
      <section className="section">
        <h2>Delete my data</h2>
        <p className="lede">This removes your account, every scan, and every stored image from the server. It cannot be undone.</p>
        {err && <Banner kind="error">{err}</Banner>}
        <div className="field" style={{ maxWidth: 360 }}>
          <label htmlFor="del">Type DELETE to confirm</label>
          <input id="del" type="text" value={text} onChange={(e) => setText(e.target.value)} autoComplete="off" />
        </div>
        <button className="btn danger" disabled={text !== 'DELETE'} onClick={del}>Delete my account and data</button>
      </section>
    </>
  );
}
