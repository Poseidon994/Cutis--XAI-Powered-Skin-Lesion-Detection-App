from __future__ import annotations

import io
import re
from collections import Counter

_WS = re.compile(r"[ \t\u00a0\u2009\u200a\u202f]+")
_ZERO = re.compile(r"[\u200b\u200c\u200d\ufeff]")

SKIP_HEADINGS = re.compile(
    r"^(?:references?|bibliography|related summaries|related pages|table of contents|contents|"
    r"on this page|share|print|acknowledg(?:e)?ments?|disclosures?|disclaimer|copyright|"
    r"permission to use|suggested citation)\s*:?\s*$"
    r"|^(?:changes to this summary|about this pdq summary|purpose of this summary|"
    r"editorial board|get more information from nci|questions about cancer|to learn more about)\b",
    re.I,
)

_REF_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_REF_JOURNAL = re.compile(
    r"\d{1,4}\s?\(\d{1,3}\):\s*\d+|\d{4};\s*\d{1,4}(?:\(\d+\))?:\s*\d+|;\s*\d{1,4}(?:\(\d+\))?:\s*\d+"
)
_REF_LEAD = re.compile(r"^\s*\[?\d{1,3}[\].)]\s")

_DROP_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer", "aside", "form",
    "svg", "iframe", "button", "select", "dialog", "template",
]
_BLOCKS = {
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "dt", "dd",
    "td", "th", "caption", "figcaption", "div", "section", "article", "main", "body",
}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def looks_like_reference(par: str) -> bool:
    """True for bibliography-style paragraphs (citation strings carry no clinical content)."""
    p = re.sub(r"^[-*\u2022]\s+", "", par.strip())
    if not p or len(p) > 700:
        return False
    low = p.lower()
    score = 0
    if "doi:" in low or "doi.org" in low or "pmid" in low:
        score += 2
    if _REF_JOURNAL.search(p):
        score += 1
    if _REF_YEAR.search(p):
        score += 1
    if re.search(r"\bet al\b", p):
        score += 1
    if _REF_LEAD.match(p):
        score += 1
    return score >= 3


def html_title(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return _WS.sub(" ", soup.title.string).strip()
    h1 = soup.find("h1")
    return _WS.sub(" ", h1.get_text(" ")).strip() if h1 else ""


def html_to_markdown(html: str, selector: str | None = None) -> str:
    """Page chrome removed, headings kept as '#', list items as '- ', everything else as paragraphs."""
    from bs4 import BeautifulSoup, Comment, NavigableString

    soup = BeautifulSoup(html, "html.parser")
    for el in soup(_DROP_TAGS):
        el.decompose()
    root = soup.select_one(selector) if selector else None
    root = root or soup.find("main") or soup.find("article") or soup.body or soup

    # a table row becomes one line ("Less than 1 mm | 1 cm"), so a value stays next to its label
    for tr in root.find_all("tr"):
        cells = [_WS.sub(" ", c.get_text(" ")).strip() for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if not cells:
            tr.decompose()
            continue
        line = soup.new_tag("p")
        line.string = " | ".join(cells)
        tr.replace_with(line)

    lines: list[str] = []
    state = {"block": None, "buf": []}

    def flush() -> None:
        text = _WS.sub(" ", "".join(state["buf"])).strip()
        block = state["block"]
        state["buf"] = []
        if not text or block is None:
            return
        name = getattr(block, "name", "")
        if name in _HEADING_TAGS:
            lines.append("#" * int(name[1]) + " " + text)
        elif name == "li":
            lines.append("- " + text)
        else:
            lines.append(text)

    for node in root.descendants:
        if not isinstance(node, NavigableString) or isinstance(node, Comment):
            continue
        block = None
        for parent in node.parents:
            if parent.name in _BLOCKS:
                block = parent
                break
            if parent is root:
                break
        if block is None:
            block = root
        if block is not state["block"]:
            flush()
            state["block"] = block
        state["buf"].append(str(node))
    flush()
    return "\n\n".join(lines)


_PDF_NUM_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})[.)]?\s+([A-Z][^.]{2,70})$")


def _pdf_heading_level(line: str) -> int:
    """0 when the line is not heading-like, else the markdown level (2-4)."""
    s = line.strip()
    if not (4 <= len(s) <= 80) or s.endswith((".", ",", ";", ":")):
        return 0
    m = _PDF_NUM_HEADING.match(s)
    if m:
        return min(2 + m.group(1).count("."), 4)
    if s.isupper() and len(s.split()) >= 2 and re.search(r"[A-Z]{3}", s):
        return 2
    return 0


def pdf_to_markdown(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        out = []
        for line in text.splitlines():
            level = _pdf_heading_level(line)
            out.append(("#" * level + " " + line.strip()) if level else line)
        pages.append(f"[[page {number}]]\n" + "\n".join(out))
    return "\n\n".join(pages)


def to_markdown(data: bytes, kind: str, selector: str | None = None) -> str:
    kind = (kind or "html").lower()
    if kind == "pdf":
        return pdf_to_markdown(data)
    text = data.decode("utf-8", errors="replace")
    if kind in {"html", "htm"}:
        return html_to_markdown(text, selector)
    return text


def drop_repeated_lines(md: str, min_repeat: int = 4, max_words: int = 8) -> str:
    """Remove short non-bullet lines that repeat many times (menus, 'Skip to content', ...)."""
    lines = md.split("\n")
    counts = Counter(
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.lstrip().startswith(("#", "- ", "[[page")) and len(ln.split()) <= max_words
    )
    bad = {ln for ln, n in counts.items() if n >= min_repeat}
    if not bad:
        return md
    return "\n".join(ln for ln in lines if ln.strip() not in bad)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO.sub("", text)
    text = "\n".join(_WS.sub(" ", ln).rstrip() for ln in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
