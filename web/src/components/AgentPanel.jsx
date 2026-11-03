import { Banner, RichText, Sources, Spinner } from './Bits.jsx';

function Source({ agent }) {
  if (agent.source === 'groq') return <span className="badge">Written by an AI model ({agent.model}) from the results above</span>;
  const why = agent.reason === 'agent_error' ? 'the AI service was not reachable' : agent.reason === 'no_api_key' ? 'AI writing is not set up on this server' : agent.guard ? 'a safety check replaced the AI text' : '';
  return <span className="badge">Standard text, no AI model used{why ? ` (${why})` : ''}</span>;
}

// How many checkable statements carry a real source marker: computed by the server, not the model.
function Grounding({ grounding, ragStatus }) {
  if (!grounding) return null;
  const label = {
    grounded: `Guideline check: every checkable statement is cited`,
    partial: `Guideline check: ${grounding.cited} of ${grounding.items} statements are cited`,
    ungrounded: 'Guideline check: no statement carried a valid source',
    no_claims: 'Guideline check: no checkable statements',
    unavailable: ragStatus === 'empty' ? 'Guideline check: the library had nothing relevant for this case' : 'Not checked against the guideline library',
  }[grounding.status];
  if (!label) return null;
  const kind = grounding.status === 'grounded' ? 'ok' : grounding.status === 'partial' ? 'warn' : 'muted';
  return <span className={`badge grounding ${kind}`}>{label}</span>;
}

const List = ({ items, sources }) => (items?.length ? <ul>{items.map((t, i) => <li key={i}><RichText text={t} sources={sources} /></li>)}</ul> : null);

function PatientReport({ r, sources }) {
  return (
    <div className="prose">
      <h3>What we found</h3>
      <p><RichText text={r.what_we_found} sources={sources} /></p>
      <h3>What this means</h3>
      <p><RichText text={r.what_it_means} sources={sources} /></p>
      <h3>What to do next</h3>
      <ol className="steps">
        {r.next_steps.map((s, i) => (
          <li key={i}>
            <div>
              <RichText text={s.step} sources={sources} />
              {s.when && <span className="when">{s.when}</span>}
            </div>
          </li>
        ))}
      </ol>
      <h3>See a doctor sooner if</h3>
      <List items={r.see_a_doctor_sooner_if} sources={sources} />
      <h3>Keeping an eye on it</h3>
      <List items={r.keep_an_eye_on_it} sources={sources} />
      <h3>Questions you could ask your doctor</h3>
      <List items={r.questions_for_your_doctor} sources={sources} />
      {r.history_note && (
        <>
          <h3>Earlier photos of this spot</h3>
          <p><RichText text={r.history_note} sources={sources} /></p>
        </>
      )}
    </div>
  );
}

function ClinicianReport({ r, sources }) {
  return (
    <div className="prose">
      <p style={{ fontSize: 17 }}><RichText text={r.summary} sources={sources} /></p>
      <div className="dispo"><strong>Disposition rule for this tier.</strong> {r.disposition?.rule}</div>
      <h3>Model findings</h3>
      <p><RichText text={r.model_findings} sources={sources} /></p>
      <h3>Supporting evidence</h3>
      <List items={r.supporting_evidence} sources={sources} />
      <h3>Differential considerations</h3>
      <div className="scroll-x">
        <table className="data">
          <thead><tr><th>Entity</th><th>Rationale</th></tr></thead>
          <tbody>{r.differential.map((d, i) => <tr key={i}><td><strong>{d.entity}</strong></td><td><RichText text={d.rationale} sources={sources} /></td></tr>)}</tbody>
        </table>
      </div>
      <h3>Recommended work-up</h3>
      <List items={r.recommended_workup} sources={sources} />
      <h3>Management options, conditional on histopathology</h3>
      {r.management_options.map((m, i) => (
        <div key={i}>
          <p style={{ margin: '10px 0 4px' }}><strong>{m.scenario}</strong></p>
          <List items={m.options} sources={sources} />
        </div>
      ))}
      <h3>Medication options (general, no doses)</h3>
      {r.medication_options.map((m, i) => (
        <div key={i}>
          <p style={{ margin: '10px 0 4px' }}><strong>{m.scenario}</strong></p>
          <List items={m.agents} sources={sources} />
          {m.notes && <p className="small muted"><RichText text={m.notes} sources={sources} /></p>}
        </div>
      ))}
      <h3>Advisory</h3>
      <List items={r.advisory} sources={sources} />
      {r.history_and_trend && (
        <>
          <h3>History and trend</h3>
          <p><RichText text={r.history_and_trend} sources={sources} /></p>
        </>
      )}
      <h3>Limitations</h3>
      <List items={r.limitations} sources={sources} />
    </div>
  );
}

export function AgentPanel({ scan, role, busy, waiting, error, onGenerate }) {
  const a = scan.agent;
  if (a.status !== 'ready') {
    return (
      <div>
        {busy || waiting ? (
          <p className="status-line" aria-live="polite"><Spinner /> {waiting && !busy ? 'Waiting for the three explanation maps to finish, then writing the ' : 'Writing the '}{role === 'patient' ? 'explanation in plain language' : 'clinical summary'}...</p>
        ) : (
          <>
            {error && <Banner kind="error">{error}</Banner>}
            <button className="btn" onClick={() => onGenerate(false)}>Write the {role === 'patient' ? 'explanation' : 'clinical summary'}</button>
          </>
        )}
      </div>
    );
  }
  const sources = a.report?.sources || [];
  const ragStatus = a.rag?.status;
  return (
    <div className={`tier ${scan.tier}`}>
      <div className="agent-head">
        <Source agent={a} />
        <Grounding grounding={a.report?.grounding} ragStatus={ragStatus} />
        <button className="btn quiet" onClick={() => onGenerate(true)} disabled={busy}>{busy ? 'Writing...' : 'Write it again'}</button>
      </div>
      {error && <Banner kind="error">{error}</Banner>}
      {ragStatus === 'unavailable' && (
        <Banner kind="info">The guideline library was not available when this report was written, so its statements are not backed by cited sources.</Banner>
      )}
      {role === 'patient' ? <PatientReport r={a.report} sources={sources} /> : <ClinicianReport r={a.report} sources={sources} />}
      <div className="disclaimer" role="note">{a.report.disclaimer}</div>
      <Sources sources={sources} role={role} rag={a.rag} />
    </div>
  );
}
