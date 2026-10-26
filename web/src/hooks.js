import { useEffect, useState } from 'react';
import { api } from './api.js';

export function useHealth() {
  const [health, setHealth] = useState(null);
  useEffect(() => {
    let alive = true;
    api.get('/health').then((h) => alive && setHealth(h)).catch(() => alive && setHealth({ ml: { ready: false, status: 'unreachable' }, agent: {} }));
    return () => {
      alive = false;
    };
  }, []);
  return health;
}

export function useTitle(t) {
  useEffect(() => {
    document.title = t ? `${t} | Cutis` : 'Cutis: skin lesion check';
  }, [t]);
}
