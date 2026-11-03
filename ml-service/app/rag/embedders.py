from __future__ import annotations

import importlib.util

import numpy as np


def _norm(matrix) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(n, 1e-12)


class FastEmbedEmbedder:
    """ONNX runtime on CPU. No torch needed, so it cannot disturb the torch/timm versions of the image models."""

    def __init__(self, model_name: str, cache_dir, offline: bool = False):
        from fastembed import TextEmbedding

        self.name = model_name
        extra = {"local_files_only": True} if offline else {}
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), **extra)
        self.dim = int(np.asarray(next(iter(self._model.embed(["probe"])))).shape[0])

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _norm(np.stack([np.asarray(v) for v in self._model.embed(texts, batch_size=batch_size)]))

    def embed_query(self, text: str) -> np.ndarray:
        return _norm(np.asarray(next(iter(self._model.embed([text])))))


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, cache_dir, offline: bool = False):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        extra = {"local_files_only": True} if offline else {}
        self._model = SentenceTransformer(model_name, device="cpu", cache_folder=str(cache_dir), **extra)
        self.dim = int(self._model.encode(["probe"], show_progress_bar=False, convert_to_numpy=True).shape[1])

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _norm(self._model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True))

    def embed_query(self, text: str) -> np.ndarray:
        return _norm(self._model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0])


_PACKAGES = {"fastembed": "fastembed", "st": "sentence_transformers"}
_CLASSES = {"fastembed": FastEmbedEmbedder, "st": SentenceTransformerEmbedder}


def get_embedder(cfg, offline: bool = False):
    """Return (embedder or None, warning or None). Never raises: no model means keyword-only search.

    offline=True only uses a model that is already downloaded, so a service start cannot hang on the network.
    """
    mode = cfg.embedder
    if mode in {"none", "off", "bm25"}:
        return None, None
    order = {
        "auto": ["fastembed", "st"],
        "fastembed": ["fastembed"],
        "st": ["st"],
        "sentence-transformers": ["st"],
    }.get(mode)
    if order is None:
        return None, f"unknown RAG_EMBEDDER '{mode}'; using keyword search only"
    problems = []
    for kind in order:
        package = _PACKAGES[kind]
        if importlib.util.find_spec(package) is None:
            problems.append(f"{package} is not installed")
            continue
        try:
            cfg.models_dir.mkdir(parents=True, exist_ok=True)
            return _CLASSES[kind](cfg.model_name, cfg.models_dir, offline), None
        except Exception as exc:  # model download or runtime problems must not stop the service
            problems.append(f"{package}: {type(exc).__name__}: {exc}")
    return None, "no embedding model available (" + "; ".join(problems) + "); using keyword search only"
