import { useEffect, useState } from 'react';

// The signature element: a circular, dermoscope-style viewport. With `tier` and `fraction` it draws the risk ring around the photo.
export function Lens({ src, size = 200, tier, fraction = 0, alt = '', ring = 9 }) {
  const [f, setF] = useState(0);
  useEffect(() => {
    const t = requestAnimationFrame(() => setF(fraction));
    return () => cancelAnimationFrame(t);
  }, [fraction]);
  const r = size / 2 - ring / 2;
  const c = 2 * Math.PI * r;
  const inset = ring + 6;
  const inner = size - inset * 2;
  return (
    <div className={`lens tier ${tier || ''}`} style={{ width: size, height: size }} role="img" aria-label={alt}>
      {src && <img className="lens-img" src={src} alt="" width={inner} height={inner} style={{ left: inset, top: inset, width: inner, height: inner }} />}
      <svg viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <circle className="track" cx={size / 2} cy={size / 2} r={r} strokeWidth={ring} />
        {tier && <circle className="arc" cx={size / 2} cy={size / 2} r={r} strokeWidth={ring} strokeDasharray={c} strokeDashoffset={c * (1 - f)} transform={`rotate(-90 ${size / 2} ${size / 2})`} />}
      </svg>
    </div>
  );
}
