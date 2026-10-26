import { Banner, Spinner } from './Bits.jsx';

function Source({ agent }) {
  if (agent.source === 'groq') return <span className="badge">Written by an AI model ({agent.model}) from the results above</span>;
  const why = agent.reason === 'agent_error' ? 'the AI service was not reachable' : agent.reason === 'no_api_key' ? 'AI writing is not set up on this server' : agent.guard ? 'a safety check replaced the AI text' : '';
  return <span className="badge">Standard text, no AI model used{why ? ` (${why})` : ''}</span>;
}

const List = ({ items }) => (items?.length ? <ul>{items.map((t, i) => <li key={i}>{t}</li>)}</ul> : null);

function PatientReport({ r }) {
  return (
    <div className="prose">
      <h3>What we found</h3>
      <p>{r.what_we_found}</p>
      <h3>What this means</h3>
      <p>{r.what_it_means}</p>
      <h3>What to do next</h3>
      <ol className="steps">
        {r.next_steps.map((s, i) => (
          <li key={i}>
            <div>
              {s.step}
              {s.when && <span className="when">{s.when}</span>}
            </div>
          </li>
        ))}
      </ol>
      <h3>See a doctor sooner if</h3>
      <List items={r.see_a_doctor_sooner_if} />
      <h3>Keeping an eye on it</h3>
      <List items={r.keep_an_eye_on_it} />
      <h3>Questions you could ask your doctor</h3>
      <List items={r.questions_for_your_doctor} />
      {r.history_note && (
        <>
          <h3>Earlier photos of this spot</h3>
          <p>{r.history_note}</p>
        </>
      )}
    </div>
  );
}

function ClinicianReport({ r }) {
  return (
    <div className="prose">
      <p style={{ fontSize: 17 }}>{r.summary}</p>
      <div className="dispo"><strong>Disposition rule for this tier.</strong> {r.disposition?.rule}</div>
      <h3>Model findings</h3>
      <p>{r.model_findings}</p>
      <h3>Supporting evidence</h3>
      <List items={r.supporting_evidence} />
      <h3>Differential considerations</h3>
      <div className="scroll-x">
        <table className="data">
          <thead><tr><th>Entity</th><th>Rationale</th></tr></thead>
          <tbody>{r.differential.map((d, i) => <tr key={i}><td><strong>{d.entity}</strong></td><td>{d.rationale}</td></tr>)}</tbody>
        </table>
      </div>
      <h3>Recommended work-up</h3>
      <List items={r.recommended_workup} />
      <h3>Management options, conditional on histopathology</h3>
      {r.management_options.map((m, i) => (
        <div key={i}>
          <p style={{ margin: '10px 0 4px' }}><strong>{m.scenario}</strong></p>
          <List items={m.options} />
        </div>
      ))}
      <h3>Medication options (general, no doses)</h3>
      {r.medication_options.map((m, i) => (
        <div key={i}>
          <p style={{ margin: '10px 0 4px' }}><strong>{m.scenario}</strong></p>
          <List items={m.agents} />
          {m.notes && <p className="small muted">{m.notes}</p>}
        </div>
      ))}
      <h3>Advisory</h3>
      <List items={r.advisory} />
      {r.history_and_trend && (
        <>
          <h3>History and trend</h3>
          <p>{r.history_and_trend}</p>
        </>
      )}
      <h3>Limitations</h3>
      <List items={r.limitations} />
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
  return (
    <div className={`tier ${scan.tier}`}>
      <div className="agent-head">
        <Source agent={a} />
        <button className="btn quiet" onClick={() => onGenerate(true)} disabled={busy}>{busy ? 'Writing...' : 'Write it again'}</button>
      </div>
      {error && <Banner kind="error">{error}</Banner>}
      {role === 'patient' ? <PatientReport r={a.report} /> : <ClinicianReport r={a.report} />}
      <div className="disclaimer" role="note">{a.report.disclaimer}</div>
    </div>
  );
}
