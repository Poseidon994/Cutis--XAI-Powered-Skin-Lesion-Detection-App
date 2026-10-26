import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';

const OUT = 512;

// Pan and zoom a photo inside a square window; the circle shows where the lesion should sit. Exports the visible square as a JPEG.
export const LensCropper = forwardRef(function LensCropper({ file }, ref) {
  const canvas = useRef(null);
  const [img, setImg] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [center, setCenter] = useState({ x: 0, y: 0 });
  const drag = useRef(null);

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    const im = new Image();
    im.onload = () => {
      setImg(im);
      setZoom(1);
      setCenter({ x: im.naturalWidth / 2, y: im.naturalHeight / 2 });
    };
    im.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const side = img ? Math.min(img.naturalWidth, img.naturalHeight) / zoom : 1;
  const clamp = useCallback(
    (c, s = side) => ({ x: Math.min(Math.max(c.x, s / 2), img.naturalWidth - s / 2), y: Math.min(Math.max(c.y, s / 2), img.naturalHeight - s / 2) }),
    [img, side],
  );

  useEffect(() => {
    if (!img || !canvas.current) return;
    const c = clamp(center);
    const ctx = canvas.current.getContext('2d');
    ctx.fillStyle = '#dfe5e9';
    ctx.fillRect(0, 0, OUT, OUT);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, c.x - side / 2, c.y - side / 2, side, side, 0, 0, OUT, OUT);
  }, [img, center, side, clamp]);

  useImperativeHandle(
    ref,
    () => ({
      getBlob: () =>
        img
          ? new Promise((resolve, reject) => canvas.current.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not prepare the image'))), 'image/jpeg', 0.92))
          : Promise.reject(new Error('The photo is still loading. Please try again in a moment.')),
    }),
    [img],
  );

  const onDown = (e) => {
    if (!img) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, c: clamp(center) };
  };
  const onMove = (e) => {
    if (!drag.current) return;
    const k = side / e.currentTarget.getBoundingClientRect().width;
    setCenter(clamp({ x: drag.current.c.x - (e.clientX - drag.current.x) * k, y: drag.current.c.y - (e.clientY - drag.current.y) * k }));
  };
  const onUp = () => {
    drag.current = null;
  };
  const onKey = (e) => {
    const step = side * 0.05;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
    if (d && img) {
      e.preventDefault();
      setCenter(clamp({ x: center.x + d[0], y: center.y + d[1] }));
    }
  };
  const onWheel = (e) => {
    if (!img) return;
    setZoom((z) => Math.min(4, Math.max(1, z * (e.deltaY < 0 ? 1.08 : 0.92))));
  };

  // wheel must be non-passive to stop the page from scrolling while zooming
  useEffect(() => {
    const el = canvas.current;
    if (!el) return undefined;
    const h = (e) => {
      e.preventDefault();
      onWheel(e);
    };
    el.addEventListener('wheel', h, { passive: false });
    return () => el.removeEventListener('wheel', h);
  });

  return (
    <div>
      <div className="cropper">
        <canvas ref={canvas} width={OUT} height={OUT} tabIndex={0} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} onKeyDown={onKey} aria-label="Photo preview. Drag to move the photo, or use the arrow keys." />
        <div className="mask" />
      </div>
      <div className="zoom">
        <label htmlFor="zoom" className="small muted">Zoom</label>
        <input id="zoom" type="range" min="1" max="4" step="0.02" value={zoom} onChange={(e) => setZoom(Number(e.target.value))} disabled={!img} />
      </div>
    </div>
  );
});
