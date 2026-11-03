from __future__ import annotations

import threading

from .config import RagConfig
from .embedders import get_embedder
from .retriever import Retriever
from .store import IndexMissing, load_index, read_meta

_UNSET = object()


class RagService:
    """Loads the index on first use, and again whenever scripts.rag_build writes a new one."""

    def __init__(self, embedder_factory=lambda cfg: get_embedder(cfg, offline=True)):
        self._factory = embedder_factory
        self._lock = threading.Lock()
        self._retriever: Retriever | None = None
        self._sig = _UNSET
        self._cfg: RagConfig | None = None
        self._embedder_cache = None
        self._state: dict = {"enabled": True, "ready": False, "reason": "not loaded", "warnings": []}

    @staticmethod
    def _signature(cfg: RagConfig):
        try:
            return (cfg.index_dir / "meta.json").stat().st_mtime_ns
        except OSError:
            return None

    def _load(self, cfg: RagConfig, sig) -> None:
        self._retriever = None
        self._sig = sig
        self._cfg = cfg
        if not cfg.enabled:
            self._state = {"enabled": False, "ready": False, "reason": "RAG_ENABLED is off", "warnings": []}
            return
        meta = read_meta(cfg) if sig is not None else None
        if meta is None:
            self._state = {
                "enabled": True,
                "ready": False,
                "reason": f"index not built in {cfg.index_dir}",
                "warnings": [],
            }
            return
        warnings: list = []
        embedder = None
        if meta.get("embedder"):
            if self._embedder_cache is None:
                self._embedder_cache = self._factory(cfg)
            embedder, warning = self._embedder_cache
            if warning:
                warnings.append(warning)
        try:
            index, extra = load_index(cfg, embedder)
        except IndexMissing as exc:
            self._state = {"enabled": True, "ready": False, "reason": str(exc), "warnings": warnings}
            return
        warnings.extend(extra)
        self._retriever = Retriever(index, embedder if index.emb is not None else None, cfg)
        self._state = {
            "enabled": True,
            "ready": True,
            "reason": None,
            "n_chunks": index.meta["n_chunks"],
            "n_sources": index.meta["n_sources"],
            "built_at": index.meta["built_at"],
            "embedder": embedder.name if self._retriever.dense_enabled else None,
            "dense": self._retriever.dense_enabled,
            "warnings": warnings,
        }

    def get(self) -> Retriever | None:
        cfg = RagConfig.from_env()
        with self._lock:
            sig = self._signature(cfg)
            if sig != self._sig or cfg != self._cfg:
                self._load(cfg, sig)
            return self._retriever

    def reload(self) -> None:
        with self._lock:
            self._embedder_cache = None
            self._sig = _UNSET
        self.get()

    def status(self) -> dict:
        self.get()
        return dict(self._state)

    def retrieve(self, audience: str, queries: list, topics=None, max_passages: int | None = None) -> dict:
        retriever = self.get()
        if retriever is None:
            return {
                "status": "unavailable",
                "reason": self._state.get("reason"),
                "embedder": None,
                "dense": False,
                "corpus_sha": None,
                "built_at": None,
                "passages": [],
                "sections": {q["id"]: [] for q in queries},
            }
        out = retriever.retrieve(queries, audience, topics, max_passages)
        out["reason"] = None if out["passages"] else "no relevant passages found"
        out["embedder"] = retriever.embedder.name if retriever.dense_enabled else None
        out["dense"] = retriever.dense_enabled
        out["corpus_sha"] = retriever.index.meta.get("corpus_sha")
        out["built_at"] = retriever.index.meta.get("built_at")
        return out
