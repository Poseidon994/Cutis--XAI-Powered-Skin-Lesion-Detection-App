"""Decode uploads and reproduce the training-time crop pipeline (square crop -> store_res x store_res uint8, bicubic)."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 80_000_000


class ImageError(ValueError):
    pass


def decode_upload(data: bytes) -> Image.Image:
    if not data:
        raise ImageError('Empty file.')
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise ImageError('The file is not a readable JPEG, PNG or WebP image.') from e
    img = ImageOps.exif_transpose(img)
    if img.mode in ('RGBA', 'LA', 'P'):
        rgba = img.convert('RGBA')
        bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return img.convert('RGB')


def square_crop(img: Image.Image) -> tuple[Image.Image, bool]:
    w, h = img.size
    if w == h:
        return img, False
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)), True


def to_store_res(img: Image.Image, store_res: int) -> np.ndarray:
    """Same operation as the training library's decode_crop: RGB, bicubic resize to store_res if needed, uint8 HxWx3."""
    if img.size != (store_res, store_res):
        img = img.resize((store_res, store_res), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8).copy()


def laplacian_variance(u8: np.ndarray) -> float:
    g = u8.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def prepare(data: bytes, store_res: int) -> tuple[np.ndarray, dict]:
    img = decode_upload(data)
    w, h = img.size
    if min(w, h) < 48:
        raise ImageError(f'The image is too small ({w}x{h}px). Use a close-up photo of at least 128x128 pixels.')
    sq, was_cropped = square_crop(img)
    u8 = to_store_res(sq, store_res)
    warnings = []
    if was_cropped:
        warnings.append({'code': 'center_cropped', 'text': 'The image was not square, so its centre was cropped. Crop tightly around the lesion before uploading for the most reliable result.'})
    if min(w, h) < 128:
        warnings.append({'code': 'low_resolution', 'text': f'The image is small ({w}x{h}px) and was enlarged; fine detail is missing.'})
    lum = float(u8.astype(np.float32).mean())
    if lum < 45:
        warnings.append({'code': 'too_dark', 'text': 'The image is very dark. Retake it in even daylight.'})
    elif lum > 225:
        warnings.append({'code': 'overexposed', 'text': 'The image is very bright or washed out. Retake it without flash glare.'})
    sharp = laplacian_variance(u8)
    if sharp < 4.0:
        warnings.append({'code': 'blurry', 'text': 'The image looks blurry or has very little detail. Hold the camera steady and focus on the lesion.'})
    info = {'original_size': [w, h], 'cropped_to_square': was_cropped, 'mean_luminance': round(lum, 1), 'sharpness': round(sharp, 2), 'warnings': warnings}
    return u8, info


def png_b64(arr: np.ndarray, size: int | None = None) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    if size and img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=False)
    return base64.b64encode(buf.getvalue()).decode('ascii')
