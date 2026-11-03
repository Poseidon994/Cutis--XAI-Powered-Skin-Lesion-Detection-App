// Theme handling for Cutis: light / dark with "follow the OS" as the default.
//
// How it works (presentational only — no app logic depends on this file):
//   * <head> inline script in index.html sets document.documentElement.dataset.theme before first
//     paint, so there is no flash of the wrong theme. It mirrors the logic in initialTheme().
//   * While the visitor has never toggled (no 'cutis-theme' in localStorage) the theme follows the
//     OS preference, including live changes; Shell.jsx wires a matchMedia listener for that.
//   * The first explicit toggle stores the choice and it wins from then on.
// Everything is wrapped so it also works in jsdom (no matchMedia) and private-mode (no localStorage).

const KEY = 'cutis-theme';

const hasMM = () => typeof window !== 'undefined' && typeof window.matchMedia === 'function';

export const systemTheme = () => {
  try {
    if (hasMM() && window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
  } catch { /* matchMedia unavailable */ }
  return 'light';
};

export const savedTheme = () => {
  try {
    const v = localStorage.getItem(KEY);
    return v === 'dark' || v === 'light' ? v : null;
  } catch { return null; }
};

/** The theme that should be active right now: explicit choice, else the OS preference. */
export const currentTheme = () => savedTheme() ?? systemTheme();

export const prefersReducedMotion = () => {
  try {
    return hasMM() && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch { return false; }
};

let animTimer = 0;

/** Apply a theme to the document. With { animate: true } a short-lived class enables smooth colour
 *  transitions (added on toggles only, never on first paint, and skipped for reduced motion). */
export function applyTheme(theme, { animate = false } = {}) {
  const dark = theme === 'dark';
  const root = document.documentElement;
  if (animate && !prefersReducedMotion()) {
    root.classList.add('theme-anim');
    clearTimeout(animTimer);
    animTimer = setTimeout(() => root.classList.remove('theme-anim'), 450);
  }
  root.dataset.theme = dark ? 'dark' : 'light';
  root.style.colorScheme = dark ? 'dark' : 'light';
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute('content', dark ? '#0f1523' : '#f2f4f6');
}

/** Make an explicit choice (persisted). Choosing what the OS already prefers clears the stored
 *  value, so the app keeps following the OS afterwards. */
export function setTheme(theme) {
  const t = theme === 'dark' ? 'dark' : 'light';
  try {
    if (t === systemTheme()) localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, t);
  } catch { /* storage unavailable */ }
  applyTheme(t, { animate: true });
  return t;
}

export const toggleTheme = () => setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
