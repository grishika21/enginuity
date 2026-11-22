"""Embeddings and the vector index used by the knowledge-base agents.

Qdrant (`QDRANT_URL`) is the stand-in for a managed vector search service. Without it, a small
file-backed NumPy index is used so the demo still runs; both share one interface.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import get_settings

log = logging.getLogger("agentkit.vectors")


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embed_model, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    vecs = _model().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


@dataclass
class Hit:
    score: float
    payload: dict[str, Any]


class VectorStore(Protocol):
    def upsert(self, collection: str, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None: ...
    def search(self, collection: str, vector: np.ndarray, top_k: int = 8, doc_id: str | None = None) -> list[Hit]: ...
    def delete_doc(self, collection: str, doc_id: str) -> None: ...
    def chunks(self, collection: str, doc_id: str) -> list[dict]: ...


class LocalStore:
    """Cosine search over a NumPy matrix on disk. Safe across processes through a file lock."""

    def __init__(self, root: Path | None = None):
        self.root = root or get_settings().storage_dir / "vectors"
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[float, np.ndarray, list[dict]]] = {}
        self._lock = threading.Lock()

    def _dir(self, collection: str) -> Path:
        d = self.root / collection
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load(self, collection: str) -> tuple[np.ndarray, list[dict]]:
        d = self._dir(collection)
        vec_path, meta_path = d / "vectors.npy", d / "payloads.json"
        if not vec_path.exists():
            return np.zeros((0, 0), dtype=np.float32), []
        mtime = vec_path.stat().st_mtime
        cached = self._cache.get(collection)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
        vectors = np.load(vec_path)
        payloads = json.loads(meta_path.read_text(encoding="utf-8"))
        self._cache[collection] = (mtime, vectors, payloads)
        return vectors, payloads

    def _write(self, collection: str, vectors: np.ndarray, payloads: list[dict]) -> None:
        d = self._dir(collection)
        tmp_vec, tmp_meta = d / "vectors.tmp.npy", d / "payloads.tmp.json"
        np.save(tmp_vec, vectors)
        tmp_meta.write_text(json.dumps(payloads), encoding="utf-8")
        os.replace(tmp_meta, d / "payloads.json")
        os.replace(tmp_vec, d / "vectors.npy")

    def _locked(self, collection: str):
        handle = open(self._dir(collection) / ".lock", "w")
        fcntl.flock(handle, fcntl.LOCK_EX)
        return handle

    def upsert(self, collection: str, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        with self._lock:
            handle = self._locked(collection)
            try:
                old_vecs, old_payloads = self._load(collection)
                keep = [i for i, p in enumerate(old_payloads) if p["id"] not in set(ids)]
                merged_payloads = [old_payloads[i] for i in keep] + [{**p, "id": i} for i, p in zip(ids, payloads)]
                parts = ([old_vecs[keep]] if keep else []) + [vectors]
                self._write(collection, np.vstack(parts).astype(np.float32), merged_payloads)
            finally:
                handle.close()

    def delete_doc(self, collection: str, doc_id: str) -> None:
        with self._lock:
            handle = self._locked(collection)
            try:
                vecs, payloads = self._load(collection)
                keep = [i for i, p in enumerate(payloads) if p.get("doc_id") != doc_id]
                if len(keep) != len(payloads):
                    self._write(collection, vecs[keep] if keep else np.zeros((0, vecs.shape[1]), np.float32),
                                [payloads[i] for i in keep])
            finally:
                handle.close()

    def search(self, collection: str, vector: np.ndarray, top_k: int = 8, doc_id: str | None = None) -> list[Hit]:
        vecs, payloads = self._load(collection)
        if not payloads:
            return []
        scores = vecs @ vector.astype(np.float32)
        order = np.argsort(-scores)
        hits = []
        for i in order:
            if doc_id and payloads[i].get("doc_id") != doc_id:
                continue
            hits.append(Hit(float(scores[i]), payloads[i]))
            if len(hits) >= top_k:
                break
        return hits

    def chunks(self, collection: str, doc_id: str) -> list[dict]:
        _vecs, payloads = self._load(collection)
        return [p for p in payloads if p.get("doc_id") == doc_id]


class QdrantStore:
    def __init__(self, url: str):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url, timeout=30)

    def _ensure(self, collection: str, dim: int) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            self.client.create_payload_index(collection, "doc_id", PayloadSchemaType.KEYWORD)

    @staticmethod
    def _pid(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert(self, collection: str, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        self._ensure(collection, vectors.shape[1])
        points = [PointStruct(id=self._pid(i), vector=v.tolist(), payload={**p, "id": i})
                  for i, v, p in zip(ids, vectors, payloads)]
        for start in range(0, len(points), 128):
            self.client.upsert(collection, points=points[start:start + 128])

    def _doc_filter(self, doc_id: str):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])

    def delete_doc(self, collection: str, doc_id: str) -> None:
        from qdrant_client.models import FilterSelector

        if self.client.collection_exists(collection):
            self.client.delete(collection, points_selector=FilterSelector(filter=self._doc_filter(doc_id)))

    def search(self, collection: str, vector: np.ndarray, top_k: int = 8, doc_id: str | None = None) -> list[Hit]:
        if not self.client.collection_exists(collection):
            return []
        res = self.client.query_points(collection, query=vector.tolist(), limit=top_k,
                                       query_filter=self._doc_filter(doc_id) if doc_id else None, with_payload=True)
        return [Hit(float(p.score), dict(p.payload or {})) for p in res.points]

    def chunks(self, collection: str, doc_id: str) -> list[dict]:
        if not self.client.collection_exists(collection):
            return []
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(collection, scroll_filter=self._doc_filter(doc_id), limit=256,
                                                offset=offset, with_payload=True)
            out.extend(dict(p.payload or {}) for p in points)
            if offset is None:
                return out


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    url = get_settings().qdrant_url
    if url:
        log.info("vector index: Qdrant at %s", url)
        return QdrantStore(url)
    log.info("vector index: local NumPy store")
    return LocalStore()
