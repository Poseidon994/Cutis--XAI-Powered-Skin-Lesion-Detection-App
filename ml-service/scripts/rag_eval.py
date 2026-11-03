"""Measure how well retrieval finds answer-bearing passages, and where the relevance floor should sit.

    python -m scripts.rag_eval                # uses rag/eval/gold_questions.yaml, k=5
    python -m scripts.rag_eval -k 3 --no-dense

Each gold question lists `expect` keywords. A retrieved passage counts as a hit when it contains any of them,
so the test keeps working whatever exact documents you ingested. Questions marked `oos: true` are out of scope
(nothing in the library should answer them) and must come back empty.

  hit@k   share of in-scope questions with at least one answer-bearing passage in the top k
  MRR     mean of 1/rank of the first answer-bearing passage (1.0 = always first)
  clean   share of out-of-scope questions that correctly returned nothing
"""
from __future__ import annotations

import argparse
import statistics
import sys
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


def load_gold(path: Path) -> list:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = data.get("questions") if isinstance(data, dict) else data
    if not items:
        raise ValueError(f"no questions in {path}")
    return items


def run_eval(retriever: Retriever, gold: list, k: int = 5) -> list:
    rows = []
    for g in gold:
        audience = g.get("audience", "clinician")
        hits = retriever.search(g["q"], audience, k=k, include_irrelevant=True)
        relevant = [h for h in hits if h["relevant"]]
        top_cos = next((h["cosine"] for h in hits if h["cosine"] is not None), None)
        row = {
            "id": g["id"],
            "q": g["q"],
            "oos": bool(g.get("oos")),
            "n_relevant": len(relevant),
            "top_cos": top_cos,
            "rank": None,
        }
        if not row["oos"]:
            keywords = [x.lower() for x in g.get("expect", [])]
            for rank, h in enumerate(relevant, 1):
                if any(word in h["text"].lower() for word in keywords):
                    row["rank"] = rank
                    break
        rows.append(row)
    return rows


def summarise(rows: list) -> dict:
    inscope = [r for r in rows if not r["oos"]]
    oos = [r for r in rows if r["oos"]]
    hit = sum(1 for r in inscope if r["rank"] is not None)
    mrr = sum(1.0 / r["rank"] for r in inscope if r["rank"]) / len(inscope) if inscope else 0.0
    clean = sum(1 for r in oos if r["n_relevant"] == 0)
    return {
        "n_inscope": len(inscope),
        "n_oos": len(oos),
        "hit_at_k": hit / len(inscope) if inscope else None,
        "mrr": mrr,
        "oos_clean": clean / len(oos) if oos else None,
    }


def main(argv=None) -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--gold", help="path to the gold questions (default rag/eval/gold_questions.yaml)")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--rag-dir", help="override RAG_DIR")
    args = parser.parse_args(argv)

    cfg = RagConfig.from_env()
    if args.rag_dir:
        cfg = replace(cfg, rag_dir=Path(args.rag_dir))
    gold = load_gold(Path(args.gold) if args.gold else cfg.rag_dir / "eval" / "gold_questions.yaml")
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
    print(f"search = {mode}, k = {args.k}, floor: cosine >= {cfg.min_cosine}, or >= {cfg.min_terms} query terms\n")

    rows = run_eval(retriever, gold, k=args.k)
    for r in rows:
        if r["oos"]:
            verdict = "ok  (nothing returned)" if r["n_relevant"] == 0 else f"LEAK ({r['n_relevant']} passages returned)"
        else:
            verdict = f"hit at rank {r['rank']}" if r["rank"] else "MISS"
        cos = "  -  " if r["top_cos"] is None else f"{r['top_cos']:.3f}"
        print(f"  {r['id']:<14} top cos {cos}  {verdict:<28} {r['q'][:60]}")

    s = summarise(rows)
    print()
    if s["hit_at_k"] is not None:
        print(f"hit@{args.k}: {s['hit_at_k']:.2f}   MRR: {s['mrr']:.2f}   ({s['n_inscope']} in-scope questions)")
    if s["oos_clean"] is not None:
        print(f"clean on out-of-scope: {s['oos_clean']:.2f}   ({s['n_oos']} questions)")

    ins = [r["top_cos"] for r in rows if not r["oos"] and r["top_cos"] is not None]
    out = [r["top_cos"] for r in rows if r["oos"] and r["top_cos"] is not None]
    if ins and out:
        low_in, high_out = min(ins), max(out)
        print(f"\ncalibration: in-scope top cosine {low_in:.2f} to {max(ins):.2f} (median {statistics.median(ins):.2f}); "
              f"out-of-scope top cosine up to {high_out:.2f}")
        if low_in > high_out:
            print(f"  the two groups separate: RAG_MIN_COSINE around {(low_in + high_out) / 2:.2f} would work")
        else:
            print("  the groups overlap: raise RAG_MIN_COSINE above the out-of-scope maximum and accept some misses,")
            print("  or add more documents that cover the missed questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
