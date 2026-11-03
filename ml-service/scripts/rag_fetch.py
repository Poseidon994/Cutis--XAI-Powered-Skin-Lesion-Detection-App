"""Download the pages listed in rag/sources.yaml and turn them into clean text files.

    python -m scripts.rag_fetch                  # every enabled source
    python -m scripts.rag_fetch --only S01,S02   # just these
    python -m scripts.rag_fetch --force          # download again even if a copy exists

Files you saved yourself (PDF, HTML, MD, TXT) go into rag/corpus/manual/ and are listed in
sources.yaml with `file: name.pdf` instead of `url:`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.rag.config import RagConfig
from app.rag.store import read_manifest
from app.rag.textclean import clean_text, drop_repeated_lines, to_markdown

USER_AGENT = "Mozilla/5.0 (compatible; CutisResearchPrototype/0.1; student research project)"
MIN_WORDS = 300


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _kind_for(entry: dict, content_type, name: str) -> str:
    if entry.get("kind"):
        return str(entry["kind"]).lower()
    ct = (content_type or "").lower()
    lower = name.lower()
    if "pdf" in ct or lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith(".md"):
        return "md"
    if lower.endswith(".txt") or "text/plain" in ct:
        return "txt"
    return "html"


def fetch_source(entry: dict, cfg: RagConfig, client: httpx.Client, force: bool = False) -> dict:
    sid = entry["id"]
    result = {
        "id": sid,
        "title": entry["title"],
        "source": entry.get("url") or entry.get("file"),
        "status": "ok",
        "bytes": 0,
        "words": 0,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        if entry.get("file"):
            src = cfg.manual_dir / entry["file"]
            if not src.exists():
                raise FileNotFoundError(f"{src} not found")
            data = src.read_bytes()
            kind = _kind_for(entry, None, src.name)
        else:
            cached = sorted(cfg.raw_dir.glob(f"{sid}.*"))
            if cached and not force:
                data = cached[0].read_bytes()
                kind = cached[0].suffix.lstrip(".")
                result["status"] = "cached"
            else:
                resp = client.get(entry["url"], follow_redirects=True)
                resp.raise_for_status()
                data = resp.content
                kind = _kind_for(entry, resp.headers.get("content-type"), resp.url.path)
                cfg.raw_dir.mkdir(parents=True, exist_ok=True)
                for old in cached:
                    old.unlink()
                (cfg.raw_dir / f"{sid}.{kind}").write_bytes(data)
        markdown = clean_text(drop_repeated_lines(to_markdown(data, kind, entry.get("selector"))))
        cfg.text_dir.mkdir(parents=True, exist_ok=True)
        (cfg.text_dir / f"{sid}.md").write_text(markdown, encoding="utf-8")
        result["bytes"] = len(data)
        result["words"] = len(markdown.split())
        if result["words"] < MIN_WORDS:
            result["warning"] = "very little text: the page may be blocked or empty, or needs a 'selector' in sources.yaml"
    except Exception as exc:  # one bad source must not stop the others
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main(argv=None, client: httpx.Client | None = None) -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated source ids, e.g. S01,S02")
    parser.add_argument("--force", action="store_true", help="download again even if a raw copy exists")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds to wait between downloads")
    parser.add_argument("--rag-dir", help="override RAG_DIR")
    args = parser.parse_args(argv)

    cfg = RagConfig.from_env()
    if args.rag_dir:
        cfg = replace(cfg, rag_dir=Path(args.rag_dir))
    entries = [s for s in read_manifest(cfg.manifest_path) if s["enabled"]]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        entries = [s for s in entries if s["id"] in wanted]
    if not entries:
        print("nothing to fetch")
        return 1

    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=httpx.Timeout(30.0))
    results = []
    try:
        for entry in entries:
            print(f"{entry['id']}  {entry['title'][:70]}")
            res = fetch_source(entry, cfg, client, force=args.force)
            results.append(res)
            if res["status"] == "error":
                print(f"    ERROR   {res['error']}")
            else:
                note = f"    {res['status']:<7} {res['words']:>6} words"
                if res.get("warning"):
                    note += f"   WARNING: {res['warning']}"
                print(note)
            if res["status"] == "ok" and entry.get("url") and args.delay > 0:
                time.sleep(args.delay)
    finally:
        if own_client:
            client.close()

    cfg.rag_dir.joinpath("corpus").mkdir(parents=True, exist_ok=True)
    (cfg.rag_dir / "corpus" / "fetch_log.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    failed = [r["id"] for r in results if r["status"] == "error"]
    thin = [r["id"] for r in results if r.get("warning")]
    print(f"\n{len(results) - len(failed)} of {len(results)} sources ready in {cfg.text_dir}")
    if failed:
        print("failed:", ", ".join(failed), "(save the page from your browser into corpus/manual/ and use `file:`)")
    if thin:
        print("thin text:", ", ".join(thin))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
