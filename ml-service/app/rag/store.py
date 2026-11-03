from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize
from .chunking import chunk_document
from .config import RagConfig

INDEX_VERSION = 1
AUDIENCES = {"clinician", "patient", "both"}
_ID = re.compile(r"^S\d{2,3}$")


class ManifestError(ValueError):
    pass


class RagBuildError(RuntimeError):
    pass


class IndexMissing(RuntimeError):
    pass


@dataclass
class Index:
    chunks: list
    sources: dict
    bm25: BM25
    emb: "np.ndarray | None"
    meta: dict


def read_manifest(path: Path) -> list:
    import yaml

    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get("sources") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ManifestError("manifest has no 'sources' list")
    seen: set = set()
    out = []
    for n, raw in enumerate(items, 1):
        if not isinstance(raw, dict):
            raise ManifestError(f"entry {n} is not a mapping")
        s = dict(raw)
        sid = str(s.get("id", "")).strip()
        if not _ID.match(sid):
            raise ManifestError(f"entry {n}: id must look like S01 (got {sid!r})")
        if sid in seen:
            raise ManifestError(f"duplicate id {sid}")
        seen.add(sid)
        for key in ("title", "publisher", "audience", "licence"):
            if not str(s.get(key, "")).strip():
                raise ManifestError(f"{sid}: missing '{key}'")
        if s["audience"] not in AUDIENCES:
            raise ManifestError(f"{sid}: audience must be one of {sorted(AUDIENCES)}")
        if not (s.get("url") or s.get("file")):
            raise ManifestError(f"{sid}: needs a 'url' or a 'file'")
        s["id"] = sid
        s["enabled"] = bool(s.get("enabled", True))
        s["topics"] = [str(t).strip().lower() for t in (s.get("topics") or [])]
        out.append(s)
    return out


def search_text(chunk: dict, source: dict) -> str:
    return f"{source['title']} {chunk['section']} {chunk['text']}"


def embed_text(chunk: dict, source: dict) -> str:
    head = source["title"] + (" > " + chunk["section"] if chunk["section"] else "")
    return f"{head}\n{chunk['text']}"


def _atomic_write(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, str):
        tmp.write_text(data, encoding="utf-8")
    else:
        tmp.write_bytes(data)
    os.replace(tmp, path)


def build_index(cfg: RagConfig, embedder=None, log=print) -> dict:
    sources = [s for s in read_manifest(cfg.manifest_path) if s["enabled"]]
    by_id = {s["id"]: s for s in sources}
    chunks: list = []
    per_source: dict = {}
    for s in sources:
        path = cfg.text_dir / f"{s['id']}.md"
        if not path.exists():
            log(f"  {s['id']}: no text file, skipped (run rag_fetch first)")
            continue
        found = chunk_document(
            s["id"],
            path.read_text(encoding="utf-8"),
            target_words=cfg.chunk_words,
            overlap_words=cfg.chunk_overlap,
        )
        if not found:
            log(f"  {s['id']}: the text produced no chunks, skipped")
            continue
        chunks.extend(found)
        per_source[s["id"]] = {"chunks": len(found), "words": sum(c["n_words"] for c in found)}
    if not chunks:
        raise RagBuildError(f"no text to index in {cfg.text_dir}; run rag_fetch first")

    used = {sid: by_id[sid] for sid in per_source}
    emb = None
    emb_meta = None
    if embedder is not None:
        texts = [embed_text(c, used[c["source_id"]]) for c in chunks]
        emb = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
        if emb.shape != (len(chunks), embedder.dim):
            raise RagBuildError(f"embedder returned shape {emb.shape}, expected {(len(chunks), embedder.dim)}")
        emb_meta = {"name": embedder.name, "dim": int(embedder.dim)}

    digest = hashlib.sha256()
    for c in chunks:
        digest.update(c["text"].encode("utf-8"))
    meta = {
        "version": INDEX_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_chunks": len(chunks),
        "n_sources": len(per_source),
        "per_source": per_source,
        "embedder": emb_meta,
        "chunk_words": cfg.chunk_words,
        "chunk_overlap": cfg.chunk_overlap,
        "corpus_sha": digest.hexdigest()[:16],
    }

    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(cfg.index_dir / "chunks.jsonl", "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n")
    public = {
        sid: {k: s.get(k) for k in ("id", "title", "publisher", "url", "audience", "licence", "topics")}
        for sid, s in used.items()
    }
    _atomic_write(cfg.index_dir / "sources.json", json.dumps(public, ensure_ascii=False, indent=1))
    emb_path = cfg.index_dir / "embeddings.npy"
    if emb is not None:
        buf = io.BytesIO()
        np.save(buf, emb)
        _atomic_write(emb_path, buf.getvalue())
    elif emb_path.exists():
        emb_path.unlink()
    _atomic_write(cfg.index_dir / "meta.json", json.dumps(meta, indent=1))  # last: signals "new build"
    return meta


def read_meta(cfg: RagConfig):
    path = cfg.index_dir / "meta.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_index(cfg: RagConfig, embedder=None):
    """Return (Index, warnings). Raises IndexMissing when there is no usable index."""
    d = cfg.index_dir
    meta = read_meta(cfg)
    if meta is None or not (d / "chunks.jsonl").exists() or not (d / "sources.json").exists():
        raise IndexMissing(f"index not built in {d}")
    if meta.get("version") != INDEX_VERSION:
        raise IndexMissing("index was built by a different version; rebuild it with scripts.rag_build")
    chunks = [json.loads(ln) for ln in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    sources = json.loads((d / "sources.json").read_text(encoding="utf-8"))
    if len(chunks) != meta.get("n_chunks"):
        raise IndexMissing("index files disagree (chunk count); rebuild it with scripts.rag_build")
    missing = {c["source_id"] for c in chunks} - set(sources)
    if missing:
        raise IndexMissing(f"index refers to unknown sources {sorted(missing)}; rebuild it")

    warnings: list = []
    emb = None
    want = meta.get("embedder")
    if want:
        emb_path = d / "embeddings.npy"
        if not emb_path.exists():
            warnings.append("embeddings file is missing; using keyword search only")
        elif embedder is None:
            warnings.append("embedding model is unavailable; using keyword search only")
        elif embedder.name != want["name"] or int(embedder.dim) != int(want["dim"]):
            warnings.append(
                f"index was built with {want['name']} but {embedder.name} is loaded; using keyword search only"
            )
        else:
            emb = np.load(emb_path, allow_pickle=False)
            if emb.shape != (len(chunks), int(want["dim"])):
                warnings.append("embedding matrix does not match the chunks; using keyword search only")
                emb = None

    docs = [tokenize(search_text(c, sources[c["source_id"]])) for c in chunks]
    return Index(chunks=chunks, sources=sources, bm25=BM25(docs), emb=emb, meta=meta), warnings
