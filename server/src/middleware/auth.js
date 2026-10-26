import jwt from 'jsonwebtoken';
import { config } from '../config.js';

export const COOKIE = 'cutis_token';

export function issueSession(res, user) {
  const token = jwt.sign({ sub: user.id, role: user.role }, config.jwtSecret, { expiresIn: `${config.jwtDays}d` });
  res.cookie(COOKIE, token, {
    httpOnly: true,
    sameSite: 'lax',
    secure: config.cookieSecure,
    maxAge: config.jwtDays * 24 * 3600 * 1000,
    path: '/',
  });
}

export function clearSession(res) {
  res.clearCookie(COOKIE, { httpOnly: true, sameSite: 'lax', secure: config.cookieSecure, path: '/' });
}

export function publicUser(u) {
  return { id: u.id, name: u.name, email: u.email, role: u.role, organisation: u.organisation ?? null, createdAt: u.created_at };
}

export function requireAuth(store) {
  return (req, res, next) => {
    const token = req.cookies?.[COOKIE];
    if (!token) return res.status(401).json({ error: 'not_authenticated', message: 'Please sign in.' });
    try {
      const payload = jwt.verify(token, config.jwtSecret);
      const user = store.userById(Number(payload.sub));
      if (!user) throw new Error('unknown user');
      req.user = user;
      next();
    } catch {
      clearSession(res);
      res.status(401).json({ error: 'not_authenticated', message: 'Your session has expired. Please sign in again.' });
    }
  };
}

export function requireRole(...roles) {
  return (req, res, next) => {
    if (!roles.includes(req.user.role)) return res.status(403).json({ error: 'forbidden', message: 'This is not available for your account type.' });
    next();
  };
}

// Defence in depth on top of SameSite=Lax: state-changing requests must carry a custom header (browsers will not add it cross-site).
export function requireCustomHeader(req, res, next) {
  if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) return next();
  if (req.get('x-requested-with') !== 'cutis-web') return res.status(403).json({ error: 'bad_request_origin', message: 'Missing X-Requested-With header.' });
  next();
}
