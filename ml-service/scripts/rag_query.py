"""Ask the index a question and see which passages come back, and why.

    python -m scripts.rag_query "excision margins for a thin melanoma" --audience clinician
    python -m scripts.rag_query                      # interactive: type questions, empty line to quit
    python -m scripts.rag_query "..." --all          # also show passages below the relevance floor

How to read a hit:
  rrf      combined score (higher is better); it only uses the ranks from the two searches below
  cos      meaning similarity from the embedding model (about 0.3 unrelated, 0.6+ related)
  bm25 #n  rank in the keyword search (blank = not in its top 30)
  dense #n rank in the meaning search (blank = not in its top 30)
  terms    how many distinct words of your question the passage contains
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

from app.rag.config import RagConfig
from app.rag.embedders import get_embedder
from app.rag.retriever import Retriever
from app.rag.store import IndexMissing, load_index


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def show(hits: list, width: int = 100) -> None:
    if not hits:
        print("  (nothing relevant found)")
        return
    for n, h in enumerate(hits, 1):
        cos = "   -  " if h["cosine"] is None else f"{h['cosine']:.3f}"
        bm = f"#{h['rank_bm25']}" if h["rank_bm25"] else "-"
        dn = f"#{h['rank_dense']}" if h["rank_dense"] else "-"
        flag = "" if h["relevant"] else "  [below relevance floor]"
        print(f"[{n}] {h['chunk_id']}  rrf={h['score']:.4f}  cos={cos}  bm25 {bm}  dense {dn}  terms={h['matched_terms']}{flag}")
        print(f"    {h['publisher']}: {h['title']}")
        if h["section"]:
            print(f"    section: {h['section']}")
        snippet = " ".join(h["text"].split())[:420]
        print(textwrap.indent(textwrap.fill(snippet, width), "    "))
        print()


def main(argv=None) -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?")
    parser.add_argument("--audience", choices=["clinician", "patient"], default="clinician")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--topics", default="", help="comma-separated topic tags to favour, e.g. melanoma")
    parser.add_argument("--all", action="store_true", help="include passages below the relevance floor")
    parser.add_argument("--no-dense", action="store_true", help="keyword search only")
    parser.add_argument("--rag-dir", help="override RAG_DIR")
    args = parser.parse_args(argv)

    cfg = RagConfig.from_env()
    if args.rag_dir:
        cfg = replace(cfg, rag_dir=Path(args.rag_dir))
    embedder = None
    if not args.no_dense:
        embedder, warning = get_embedder(cfg, offline=True)
        if warning:
            print("WARNING:", warning)
    try:
        index, warnings = load_index(cfg, embedder)
    except IndexMissing as exc:
        print("ERROR:", exc, "- run scripts.rag_build first")
        return 1
    for w in warnings:
        print("WARNING:", w)
    retriever = Retriever(index, embedder if index.emb is not None else None, cfg)
    mode = f"keyword + {embedder.name}" if retriever.dense_enabled else "keyword only"
    print(f"index: {index.meta['n_chunks']} chunks from {index.meta['n_sources']} sources, search = {mode}\n")
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    def run(q: str) -> None:
        print(f"audience={args.audience}  query: {q}\n")
        show(retriever.search(q, args.audience, k=args.k, topics=topics, include_irrelevant=args.all))

    if args.query:
        run(args.query)
        return 0
    while True:
        try:
            q = input("question> ").strip()
        except EOFError:
            break
        if not q:
            break
        run(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
