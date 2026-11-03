from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class RagConfig:
    rag_dir: Path
    enabled: bool = True
    embedder: str = "auto"          # auto | fastembed | st | none
    model_name: str = DEFAULT_MODEL
    min_cosine: float = 0.55        # dense relevance floor (calibrate with scripts/rag_eval.py)
    min_terms: int = 3              # keyword relevance floor: distinct query terms found in the passage
    max_per_source: int = 2         # no single document may fill the answer
    section_k: int = 3              # passages per query (report section)
    max_passages: int = 6           # passages sent to the LLM per report (keeps the prompt small)
    chunk_words: int = 220
    chunk_overlap: int = 30

    @classmethod
    def from_env(cls) -> "RagConfig":
        default_dir = Path(__file__).resolve().parents[3] / "rag"
        return cls(
            rag_dir=Path(os.environ.get("RAG_DIR") or default_dir),
            enabled=_flag("RAG_ENABLED", True),
            embedder=(os.environ.get("RAG_EMBEDDER") or "auto").strip().lower(),
            model_name=os.environ.get("RAG_MODEL") or DEFAULT_MODEL,
            min_cosine=_float("RAG_MIN_COSINE", 0.55),
            min_terms=_int("RAG_MIN_TERMS", 3),
            max_per_source=_int("RAG_MAX_PER_SOURCE", 2),
            section_k=_int("RAG_SECTION_K", 3),
            max_passages=_int("RAG_MAX_PASSAGES", 6),
            chunk_words=_int("RAG_CHUNK_WORDS", 220),
            chunk_overlap=_int("RAG_CHUNK_OVERLAP", 30),
        )

    @property
    def manifest_path(self) -> Path:
        return self.rag_dir / "sources.yaml"

    @property
    def raw_dir(self) -> Path:
        return self.rag_dir / "corpus" / "raw"

    @property
    def manual_dir(self) -> Path:
        return self.rag_dir / "corpus" / "manual"

    @property
    def text_dir(self) -> Path:
        return self.rag_dir / "corpus" / "text"

    @property
    def index_dir(self) -> Path:
        return self.rag_dir / "index"

    @property
    def models_dir(self) -> Path:
        return self.rag_dir / "models"
