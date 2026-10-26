import { NavLink, Outlet, Link } from 'react-router-dom';
import { useAuth } from '../auth.jsx';

const Icon = ({ d }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {d.map((p, i) => <path key={i} d={p} />)}
  </svg>
);
const icons = {
  home: ['M3 11l9-8 9 8', 'M5 10v10h14V10'],
  plus: ['M12 5v14', 'M5 12h14'],
  list: ['M8 6h13', 'M8 12h13', 'M8 18h13', 'M3 6h.01', 'M3 12h.01', 'M3 18h.01'],
  info: ['M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z', 'M12 11v5', 'M12 8h.01'],
};

export function Brand() {
  return (
    <Link to="/" className="brand" aria-label="Cutis home">
      <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
        <circle cx="16" cy="16" r="13" fill="none" stroke="var(--ink)" strokeWidth="2.4" />
        <path d="M16 3a13 13 0 0 1 11.3 6.5" fill="none" stroke="var(--accent)" strokeWidth="4" strokeLinecap="round" />
        <circle cx="16" cy="16" r="5" fill="var(--ink)" />
      </svg>
      Cutis
    </Link>
  );
}

export function Shell() {
  const { user, logout } = useAuth();
  return (
    <div className="shell">
      <header className="topbar"><Brand /></header>
      <nav className="rail" aria-label="Main">
        <Brand />
        <div className="nav">
          <NavLink to="/" end><Icon d={icons.home} />Home</NavLink>
          <NavLink to="/scan/new"><Icon d={icons.plus} />{user.role === 'patient' ? 'Check a spot' : 'New analysis'}</NavLink>
          <NavLink to="/history"><Icon d={icons.list} />History</NavLink>
          <NavLink to="/about"><Icon d={icons.info} />About</NavLink>
        </div>
        <div className="rail-foot">
          <div className="who">{user.name}</div>
          <div className="muted small">{user.role === 'patient' ? 'Patient account' : 'Clinician account'}</div>
          <div style={{ marginTop: 8, display: 'flex', gap: 4 }}>
            <Link className="btn quiet" to="/account">Account</Link>
            <button className="btn quiet" onClick={logout}>Sign out</button>
          </div>
        </div>
      </nav>
      <main className="main" id="main"><Outlet /></main>
    </div>
  );
}
