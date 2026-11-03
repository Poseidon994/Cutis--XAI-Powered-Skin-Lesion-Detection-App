from __future__ import annotations

import math
import threading

import numpy as np

from .bm25 import tokenize
from .config import RagConfig
from .store import Index

RRF_K = 60          # reciprocal-rank-fusion constant: larger values flatten the difference between ranks
TOPIC_BOOST = 1.25  # gentle nudge for passages whose source is tagged with a requested topic


class Retriever:
    def __init__(self, index: Index, embedder, cfg: RagConfig):
        self.index = index
        self.embedder = embedder
        self.cfg = cfg
        audiences = [index.sources[c["source_id"]]["audience"] for c in index.chunks]
        self._masks = {
            a: np.array([x in (a, "both") for x in audiences], dtype=bool) for a in ("clinician", "patient")
        }
        self._topics = [set(index.sources[c["source_id"]].get("topics") or []) for c in index.chunks]
        self._embed_lock = threading.Lock()

    @property
    def dense_enabled(self) -> bool:
        return self.index.emb is not None and self.embedder is not None

    def search(
        self,
        query: str,
        audience: str,
        k: int = 5,
        topics=None,
        include_irrelevant: bool = False,
        n_candidates: int = 30,
    ) -> list:
        if audience not in self._masks:
            raise ValueError(f"audience must be 'clinician' or 'patient', got {audience!r}")
        q_tokens = tokenize(query)
        q_terms = set(q_tokens)
        mask = self._masks[audience]

        bm_hits = self.index.bm25.search(q_tokens, n=n_candidates, mask=mask)
        bm_rank = {i: r for r, (i, _) in enumerate(bm_hits, 1)}
        bm_score = dict(bm_hits)

        cos = None
        dense_rank: dict = {}
        if self.dense_enabled and query.strip():
            with self._embed_lock:
                qv = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
            cos = self.index.emb @ qv
            masked = np.where(mask, cos, -np.inf)
            order = np.argsort(-masked, kind="stable")[:n_candidates]
            dense_rank = {int(i): r for r, i in enumerate(order, 1) if np.isfinite(masked[i])}

        wanted = set(topics or [])
        fused: dict = {}
        for i in set(bm_rank) | set(dense_rank):
            score = 0.0
            if i in bm_rank:
                score += 1.0 / (RRF_K + bm_rank[i])
            if i in dense_rank:
                score += 1.0 / (RRF_K + dense_rank[i])
            if wanted and self._topics[i] & wanted:
                score *= TOPIC_BOOST
            fused[i] = score

        needed = 1
        if q_terms:
            needed = min(self.cfg.min_terms, len(q_terms), max(1, math.ceil(len(q_terms) / 2)))

        hits: list = []
        per_source: dict = {}
        for i in sorted(fused, key=lambda j: (-fused[j], j)):
            if len(hits) >= k:
                break
            chunk = self.index.chunks[i]
            sid = chunk["source_id"]
            if per_source.get(sid, 0) >= self.cfg.max_per_source:
                continue
            matched = self.index.bm25.matched(i, q_terms)
            cosine = float(cos[i]) if cos is not None else None
            relevant = (cosine is not None and cosine >= self.cfg.min_cosine) or matched >= needed
            if not relevant and not include_irrelevant:
                continue
            per_source[sid] = per_source.get(sid, 0) + 1
            src = self.index.sources[sid]
            hits.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source_id": sid,
                    "title": src["title"],
                    "publisher": src["publisher"],
                    "url": src.get("url"),
                    "licence": src["licence"],
                    "section": chunk["section"],
                    "page": chunk.get("page"),
                    "text": chunk["text"],
                    "score": round(fused[i], 6),
                    "cosine": None if cosine is None else round(cosine, 4),
                    "bm25": round(bm_score.get(i, 0.0), 3),
                    "matched_terms": matched,
                    "rank_bm25": bm_rank.get(i),
                    "rank_dense": dense_rank.get(i),
                    "relevant": bool(relevant),
                }
            )
        return hits

    def retrieve(self, queries: list, audience: str, topics=None, max_passages: int | None = None) -> dict:
        """Run several queries (one per report section) and merge into labelled passages P1, P2, ..."""
        cap = max_passages or self.cfg.max_passages
        per_section: dict = {}
        best: dict = {}
        for q in queries:
            k = q.get("k") or self.cfg.section_k
            hits = self.search(q["text"], audience, k=k, topics=topics)
            per_section[q["id"]] = [h["chunk_id"] for h in hits]
            for h in hits:
                prev = best.get(h["chunk_id"])
                if prev is None or h["score"] > prev["score"]:
                    best[h["chunk_id"]] = h

        chosen: list = []
        depth = max((len(ids) for ids in per_section.values()), default=0)
        for rank in range(depth):  # round-robin so every report section gets its best passage first
            for ids in per_section.values():
                if rank < len(ids) and ids[rank] not in chosen and len(chosen) < cap:
                    chosen.append(ids[rank])
        labels = {cid: f"P{n}" for n, cid in enumerate(chosen, 1)}
        passages = [dict(best[cid], label=labels[cid]) for cid in chosen]
        sections = {qid: [labels[c] for c in ids if c in labels] for qid, ids in per_section.items()}
        return {"status": "ok" if passages else "empty", "passages": passages, "sections": sections}
