import { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';
import { Banner, Spinner } from './Bits.jsx';

const SUGGEST = {
  patient: ['What does this result mean?', 'What should I do next?', 'Do I need a biopsy?', 'How do I take a better photo?'],
  clinician: ['Which dermoscopic features would support the leading type here?', 'When would you biopsy this lesion?', 'How reliable are these maps for this image?', 'What follow-up interval fits this tier?'],
};

export function Chat({ scanId, role, initial }) {
  const [msgs, setMsgs] = useState(initial || []);
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const end = useRef(null);
  useEffect(() => setMsgs(initial || []), [initial]);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {          // do not pull the page down to the chat when the result opens
      first.current = false;
      return;
    }
    end.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' });
  }, [msgs.length, busy]);

  const send = async (q) => {
    const message = (q ?? text).trim();
    if (!message || busy) return;
    setBusy(true);
    setErr('');
    setMsgs((m) => [...m, { id: `tmp-${Date.now()}`, role: 'user', content: message }]);
    setText('');
    try {
      const r = await api.post(`/scans/${scanId}/agent/chat`, { message });
      setMsgs(r.messages);
    } catch (e) {
      setMsgs((m) => m.slice(0, -1));
      setText(message);
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="chat">
      {!msgs.length && (
        <div className="suggest">
          {SUGGEST[role].map((s) => <button key={s} type="button" onClick={() => send(s)}>{s}</button>)}
        </div>
      )}
      <div className="msgs" aria-live="polite">
        {msgs.map((m) => <div key={m.id} className={`msg ${m.role}`}>{m.content}</div>)}
        {busy && <div className="status-line"><Spinner /> Thinking...</div>}
        <div ref={end} />
      </div>
      {err && <Banner kind="error">{err}</Banner>}
      <form onSubmit={(e) => { e.preventDefault(); send(); }}>
        <label className="sr" htmlFor="chat-q">Your question</label>
        <input id="chat-q" type="text" value={text} maxLength={1000} onChange={(e) => setText(e.target.value)} placeholder={role === 'patient' ? 'Ask anything about this result' : 'Ask about this scan or its management'} />
        <button className="btn" disabled={busy || !text.trim()}>Ask</button>
      </form>
      <p className="tiny" style={{ marginTop: 10 }}>The assistant sees your scan results and notes, not your photo. It cannot diagnose and can be wrong.</p>
    </div>
  );
}
