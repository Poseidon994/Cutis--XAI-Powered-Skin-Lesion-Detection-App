from __future__ import annotations

import re

from .textclean import SKIP_HEADINGS, looks_like_reference

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PAGE = re.compile(r"^\[\[page (\d+)\]\]$")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_BULLETS = ("- ", "* ", "\u2022 ")


def _wc(text: str) -> int:
    return len(text.split())


def parse_sections(md: str) -> list[dict]:
    """Split markdown into sections: {'trail': [headings], 'paras': [...], 'page': int|None}."""
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []
    paras: list[str] = []
    buf: list[str] = []
    page = None
    sec_page = None

    def flush_buf() -> None:
        if buf:
            paras.append(" ".join(buf).strip())
            buf.clear()

    def close_section() -> None:
        nonlocal paras
        flush_buf()
        if paras:
            sections.append({"trail": [t for _, t in stack], "paras": paras, "page": sec_page})
        paras = []

    for raw in md.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        m = _PAGE.match(stripped)
        if m:
            page = int(m.group(1))
            continue
        h = _HEADING.match(line)
        if h:
            close_section()
            level, text = len(h.group(1)), h.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            sec_page = page
            continue
        if not stripped:
            flush_buf()
            continue
        if stripped.startswith(_BULLETS):
            flush_buf()
        if not paras and not buf and sec_page is None:
            sec_page = page
        buf.append(stripped)
    close_section()
    return sections


def _split_long(par: str, target: int) -> list[str]:
    if _wc(par) <= int(target * 1.4):
        return [par]
    out: list[str] = []
    cur: list[str] = []
    n = 0
    for sentence in _SENT.split(par):
        w = _wc(sentence)
        if cur and n + w > target:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(sentence)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def _tail(text: str, words: int) -> str:
    parts = text.split()
    return " ".join(parts[-words:]) if words > 0 and len(parts) > words else ""


def _pack(units: list[str], target: int, overlap: int) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    n = 0
    fresh = 0
    for unit in units:
        w = _wc(unit)
        if cur and fresh and n + w > target:
            text = "\n".join(cur)
            out.append(text)
            tail = _tail(text, overlap)
            cur = [tail] if tail else []
            n = _wc(tail)
            fresh = 0
        cur.append(unit)
        n += w
        fresh += 1
    if cur and fresh:
        out.append("\n".join(cur))
    return out


def chunk_document(
    source_id: str,
    md: str,
    *,
    target_words: int = 220,
    overlap_words: int = 30,
    min_words: int = 12,
) -> list[dict]:
    chunks: list[dict] = []
    ordinal = 0
    for sec in parse_sections(md):
        if any(SKIP_HEADINGS.match(t) for t in sec["trail"]):
            continue
        units: list[str] = []
        for par in sec["paras"]:
            if looks_like_reference(par):
                continue
            units.extend(_split_long(par, target_words))
        for text in _pack(units, target_words, overlap_words):
            if _wc(text) < min_words:
                continue
            ordinal += 1
            chunks.append(
                {
                    "chunk_id": f"{source_id}-{ordinal:03d}",
                    "source_id": source_id,
                    "ordinal": ordinal,
                    "section": " > ".join(sec["trail"]),
                    "page": sec["page"],
                    "text": text,
                    "n_words": _wc(text),
                }
            )
    return chunks
