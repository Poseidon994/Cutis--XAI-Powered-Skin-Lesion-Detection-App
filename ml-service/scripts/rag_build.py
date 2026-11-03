"""Chunk the cleaned text, embed it, and write the index to rag/index/.

    python -m scripts.rag_build              # keyword + embedding index (downloads the model on first run)
    python -m scripts.rag_build --no-dense   # keyword-only index, no model needed

Run it again whenever sources.yaml or the text files change. The running service picks up a new index by itself.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from app.rag.config import RagConfig
from app.rag.embedders import get_embedder
from app.rag.store import RagBuildError, build_index


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None, embedder_factory=get_embedder) -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-dense", action="store_true", help="skip the embedding model (keyword search only)")
    parser.add_argument("--rag-dir", help="override RAG_DIR")
    args = parser.parse_args(argv)

    cfg = RagConfig.from_env()
    if args.rag_dir:
        cfg = replace(cfg, rag_dir=Path(args.rag_dir))

    embedder = None
    if not args.no_dense:
        print(f"loading embedding model {cfg.model_name} (the first run downloads about 70 MB)...")
        embedder, warning = embedder_factory(cfg)
        if warning:
            print("WARNING:", warning)
    try:
        meta = build_index(cfg, embedder)
    except RagBuildError as exc:
        print("ERROR:", exc)
        return 1

    print(f"\nindex written to {cfg.index_dir}")
    print(f"  sources : {meta['n_sources']}")
    print(f"  chunks  : {meta['n_chunks']}")
    print(f"  search  : {'keyword + ' + meta['embedder']['name'] if meta['embedder'] else 'keyword only'}")
    print("\n  per source:")
    for sid, info in meta["per_source"].items():
        print(f"    {sid}  {info['chunks']:>4} chunks  {info['words']:>6} words")
    if meta["embedder"] is None and not args.no_dense:
        print("\nNOTE: no embedding model was available, so this is a keyword-only index (see the warning above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
