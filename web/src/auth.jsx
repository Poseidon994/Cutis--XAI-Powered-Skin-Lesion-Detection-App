import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { api, setUnauthorizedHandler } from './api.js';

const Ctx = createContext(null);
export const useAuth = () => useContext(Ctx);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(undefined); // undefined = still checking, null = signed out
  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null));
    api.get('/auth/me').then((r) => setUser(r.user)).catch(() => setUser(null));
  }, []);
  const login = useCallback(async (email, password) => setUser((await api.post('/auth/login', { email, password })).user), []);
  const register = useCallback(async (body) => setUser((await api.post('/auth/register', body)).user), []);
  const logout = useCallback(async () => {
    await api.post('/auth/logout').catch(() => {});
    setUser(null);
  }, []);
  const value = useMemo(() => ({ user, login, register, logout, setUser }), [user, login, register, logout]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
