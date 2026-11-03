import { useEffect, useState } from 'react';
import { NavLink, Outlet, Link } from 'react-router-dom';
import { useAuth } from '../auth.jsx';
import { currentTheme, savedTheme, systemTheme, applyTheme, toggleTheme } from '../theme.js';

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
  sun: ['M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10z', 'M12 1v2', 'M12 21v2', 'M4.22 4.22l1.42 1.42', 'M18.36 18.36l1.42 1.42', 'M1 12h2', 'M21 12h2', 'M4.22 19.78l1.42-1.42', 'M18.36 5.64l1.42-1.42'],
  moon: ['M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'],
};

// Presentational only: shows the live theme and flips it. While the visitor has never made an
// explicit choice (nothing stored), it also keeps following OS-level changes live.
function ThemeToggle() {
  const [theme, setShown] = useState(() => currentTheme());
  useEffect(() => {
    if (savedTheme() || !window.matchMedia) return undefined;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => { applyTheme(systemTheme()); setShown(systemTheme()); };
    mq.addEventListener?.('change', onChange);
    return () => mq.removeEventListener?.('change', onChange);
  }, []);
  const flip = () => setShown(toggleTheme());
  const dark = theme === 'dark';
  return (
    <button className="btn quiet theme-toggle" onClick={flip} title={dark ? 'Switch to light mode' : 'Switch to dark mode'} aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}>
      <Icon d={dark ? icons.sun : icons.moon} />
      <span>{dark ? 'Light mode' : 'Dark mode'}</span>
    </button>
  );
}

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
      <header className="topbar"><Brand /><ThemeToggle /></header>
      <nav className="rail" aria-label="Main">
        <Brand />
        <div className="nav">
          <NavLink to="/" end><Icon d={icons.home} />Home</NavLink>
          <NavLink to="/scan/new"><Icon d={icons.plus} />{user.role === 'patient' ? 'Check a spot' : 'New analysis'}</NavLink>
          <NavLink to="/history"><Icon d={icons.list} />History</NavLink>
          <NavLink to="/about"><Icon d={icons.info} />About</NavLink>
        </div>
        <div className="rail-foot">
          <ThemeToggle />
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
