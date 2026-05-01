from __future__ import annotations

import importlib
import json
import os
import warnings
from pathlib import Path
from time import perf_counter
from time import sleep
from typing import Any, Protocol, cast

from .config import AppConfig
from .embedding import EMBEDDING_PREPROCESS_VERSION, EmbeddingProvider
from .storage import Storage
from .utils import stable_content_hash


class _ChromaCollection(Protocol):
    name: str

    def count(self) -> int: ...

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None: ...

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
        include: list[str],
    ) -> dict[str, Any]: ...

    def delete(self, *, ids: list[str]) -> None: ...


class _ChromaClient(Protocol):
    def get_or_create_collection(self, *, name: str) -> _ChromaCollection: ...

    def get_collection(self, *, name: str) -> _ChromaCollection: ...

    def get_max_batch_size(self) -> int: ...


class _JsonVectorCollection:
    """Small persistent vector backend used when chromadb is unavailable.

    It implements the tiny subset of the Chroma collection API used by
    DysonSpherain. This keeps the CLI usable in clean/offline environments and
    makes smoke tests deterministic. Chroma remains the preferred backend for
    larger corpora.
    """

    def __init__(self, client: "_JsonVectorClient", name: str) -> None:
        self._client = client
        self.name = name
        self._client._state.setdefault(name, {})

    def count(self) -> int:
        return len(self._client._state.get(self.name, {}))

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        bucket = self._client._state.setdefault(self.name, {})
        for item_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            bucket[str(item_id)] = {
                "document": str(document),
                "metadata": dict(metadata or {}),
                "embedding": [float(v) for v in embedding],
            }
        self._client._flush()

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
        include: list[str],
    ) -> dict[str, Any]:
        query_embedding = [float(v) for v in (query_embeddings[0] if query_embeddings else [])]
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for item_id, item in self._client._state.get(self.name, {}).items():
            metadata = dict(item.get("metadata") or {})
            if not self._matches_where(metadata, where):
                continue
            distance = self._squared_l2(query_embedding, [float(v) for v in item.get("embedding") or []])
            scored.append((distance, str(item_id), item))
        scored.sort(key=lambda row: row[0])
        limited = scored[: max(1, int(n_results))]
        return {
            "ids": [[item_id for _, item_id, _ in limited]],
            "documents": [[str(item.get("document") or "") for _, _, item in limited]],
            "metadatas": [[dict(item.get("metadata") or {}) for _, _, item in limited]],
            "distances": [[float(distance) for distance, _, _ in limited]],
        }

    def delete(self, *, ids: list[str]) -> None:
        bucket = self._client._state.setdefault(self.name, {})
        for item_id in ids:
            bucket.pop(str(item_id), None)
        self._client._flush()

    @staticmethod
    def _squared_l2(a: list[float], b: list[float]) -> float:
        width = min(len(a), len(b))
        if width == 0:
            return 1e9
        total = 0.0
        for idx in range(width):
            diff = a[idx] - b[idx]
            total += diff * diff
        if len(a) != len(b):
            total += abs(len(a) - len(b))
        return total

    @classmethod
    def _matches_where(cls, metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(cls._matches_where(metadata, dict(child or {})) for child in list(where.get("$and") or []))
        for key, expected in where.items():
            if key == "$and":
                continue
            actual = metadata.get(key)
            if isinstance(expected, dict):
                if "$eq" in expected and str(actual) != str(expected["$eq"]):
                    return False
                if "$in" in expected and str(actual) not in {str(v) for v in expected["$in"]}:
                    return False
                if "$ne" in expected and str(actual) == str(expected["$ne"]):
                    return False
                continue
            if str(actual) != str(expected):
                return False
        return True


class _JsonVectorClient:
    def __init__(self, path: str) -> None:
        warnings.warn(
            "JSON vector backend is O(N) scan and intended for small/offline tests only.",
            RuntimeWarning,
            stacklevel=2,
        )
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.state_path = self.path / "json_vector_store.json"
        if self.state_path.exists():
            try:
                self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                self._state = {}
        else:
            self._state: dict[str, dict[str, Any]] = {}

    def get_or_create_collection(self, *, name: str) -> _JsonVectorCollection:
        self._state.setdefault(name, {})
        self._flush()
        return _JsonVectorCollection(self, name)

    def get_collection(self, *, name: str) -> _JsonVectorCollection:
        if name not in self._state:
            raise RuntimeError(f"Collection {name!r} does not exist")
        return _JsonVectorCollection(self, name)

    def get_max_batch_size(self) -> int:
        return 2048

    def close(self) -> None:
        self._flush()

    def _flush(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)


def _build_chroma_client(path: str, backend: str = "auto") -> _ChromaClient:
    backend = (backend or "auto").strip().lower()
    if backend in {"json", "simple", "local"}:
        return cast(_ChromaClient, _JsonVectorClient(path))
    try:
        chromadb_module = importlib.import_module("chromadb")
    except ModuleNotFoundError as exc:
        if backend == "auto":
            return cast(_ChromaClient, _JsonVectorClient(path))
        raise RuntimeError("chromadb is required for vector storage; SPHERE_VECTOR_BACKEND=chroma cannot fall back to JSON") from exc
    persistent_client = getattr(chromadb_module, "PersistentClient", None)
    if persistent_client is None:
        if backend == "auto":
            return cast(_ChromaClient, _JsonVectorClient(path))
        raise RuntimeError("chromadb.PersistentClient is not available; SPHERE_VECTOR_BACKEND=chroma cannot fall back to JSON")
    return cast(_ChromaClient, persistent_client(path=path))


class VectorStore:
    def __init__(self, config: AppConfig, storage: Storage | None = None) -> None:
        self.config = config
        self.storage = storage
        self.embedder = EmbeddingProvider(
            config.embedding_model_name,
            config.embedding_dim,
            cache_size=config.embedding_cache_memory_size,
            cache_path=config.embedding_cache_path,
            fail_fast=config.embedding_fail_fast,
        )
        self.client = _build_chroma_client(str(config.vector_dir), backend=config.vector_backend)
        self._json_scan_warning = ""
        self._vector_backend_resolved = "json" if isinstance(self.client, _JsonVectorClient) else "chroma"
        self._fallback_in_use = self._vector_backend_resolved == "json" and str(config.vector_backend or "auto").lower() == "auto"
        self._stats: dict[str, dict[str, float | int]] = {}
        self._counters: dict[str, int] = {
            "vector_dedup_hit_count": 0,
            "vector_dedup_miss_count": 0,
        }
        self.raw_collection = self.client.get_or_create_collection(name=config.vector_collection_name)
        if config.vector_collection_name != "memory_chunks" and self.raw_collection.count() == 0:
            try:
                legacy = self.client.get_collection(name="memory_chunks")
                if legacy.count() > 0:
                    self.raw_collection = legacy
            except Exception:
                pass
        self.object_collection = self.client.get_or_create_collection(name=config.object_collection_name)
        self.proxy_collection = self.client.get_or_create_collection(name=config.proxy_collection_name)
        self._reconcile_sync_state()
        self._enforce_json_backend_guard()

    def _refresh_collections(self) -> None:
        self.client = _build_chroma_client(str(self.config.vector_dir), backend=self.config.vector_backend)
        self._vector_backend_resolved = "json" if isinstance(self.client, _JsonVectorClient) else "chroma"
        self._fallback_in_use = self._vector_backend_resolved == "json" and str(self.config.vector_backend or "auto").lower() == "auto"
        self.raw_collection = self.client.get_or_create_collection(name=self.config.vector_collection_name)
        self.object_collection = self.client.get_or_create_collection(name=self.config.object_collection_name)
        self.proxy_collection = self.client.get_or_create_collection(name=self.config.proxy_collection_name)
        self._enforce_json_backend_guard()

    def _enforce_json_backend_guard(self) -> None:
        if self._vector_backend_resolved != "json":
            self._json_scan_warning = ""
            return
        vector_count = int(self.raw_collection.count()) + int(self.object_collection.count()) + int(self.proxy_collection.count())
        max_items = max(1, int(getattr(self.config, "json_vector_max_items", 5000) or 5000))
        message = "JSON vector backend is O(N) scan and intended for small/offline tests only."
        if vector_count > max_items:
            message = f"{message} vector_count={vector_count} exceeds SPHERE_JSON_VECTOR_MAX_ITEMS={max_items}."
            if bool(getattr(self.config, "vector_fail_fast_on_fallback", False)):
                raise RuntimeError(message)
        self._json_scan_warning = message
        if bool(getattr(self.config, "warn_on_json_vector_backend", True)):
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    @staticmethod
    def _is_retryable_query_error(exc: Exception) -> bool:
        lowered = str(exc).lower()
        return "nothing found on disk" in lowered or "creating hnsw segment reader" in lowered

    def _query_with_retry(
        self,
        *,
        collection_name: str,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None,
        include: list[str],
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                if collection_name == "raw":
                    collection = self.raw_collection
                elif collection_name == "object":
                    collection = self.object_collection
                else:
                    collection = self.proxy_collection
                if self._vector_backend_resolved == "json":
                    self._enforce_json_backend_guard()
                return collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where,
                    include=include,
                )
            except Exception as exc:
                if not self._is_retryable_query_error(exc) or attempt == 3:
                    raise
                last_exc = exc
                sleep(0.2 * (attempt + 1))
                self._refresh_collections()
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("query retry loop exited unexpectedly")

    def _query_many_with_retry(
        self,
        *,
        collection_name: str,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None,
        include: list[str],
    ) -> dict[str, Any]:
        if self._vector_backend_resolved == "json":
            ids: list[list[str]] = []
            documents: list[list[str]] = []
            metadatas: list[list[dict[str, Any]]] = []
            distances: list[list[float]] = []
            for query_embedding in query_embeddings:
                result = self._query_with_retry(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    n_results=n_results,
                    where=where,
                    include=include,
                )
                ids.append(list(result.get("ids", [[]])[0]))
                documents.append(list(result.get("documents", [[]])[0]))
                metadatas.append(list(result.get("metadatas", [[]])[0]))
                distances.append(list(result.get("distances", [[]])[0]))
            return {"ids": ids, "documents": documents, "metadatas": metadatas, "distances": distances}
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                if collection_name == "raw":
                    collection = self.raw_collection
                elif collection_name == "object":
                    collection = self.object_collection
                else:
                    collection = self.proxy_collection
                return collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where,
                    include=include,
                )
            except Exception as exc:
                if not self._is_retryable_query_error(exc) or attempt == 3:
                    raise
                last_exc = exc
                sleep(0.2 * (attempt + 1))
                self._refresh_collections()
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("batch query retry loop exited unexpectedly")

    def _record_stat(self, operation: str, elapsed_ms: float, rows: int | None = None) -> None:
        bucket = self._stats.setdefault(operation, {"calls": 0, "total_ms": 0.0, "rows": 0})
        bucket["calls"] = int(bucket["calls"]) + 1
        bucket["total_ms"] = float(bucket["total_ms"]) + float(elapsed_ms)
        if rows is not None:
            bucket["rows"] = int(bucket["rows"]) + int(rows)

    def _max_batch_size(self) -> int:
        try:
            value = int(getattr(self.client, "get_max_batch_size")())
        except Exception:
            value = 5000
        return max(1, value)

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        distance = max(0.0, float(distance or 0.0))
        # Chroma returns squared-L2 distances for the default collection metric.
        # Our embeddings are normalized, so cosine ~= 1 - d/2. Falling back to
        # a monotonic inverse keeps the score usable if a backend returns a
        # wider distance range instead of squared-L2.
        if distance <= 4.0:
            return max(0.0, 1.0 - (distance / 2.0))
        return 1.0 / (1.0 + distance)

    def _increment_counter(self, key: str, value: int) -> None:
        self._counters[key] = int(self._counters.get(key, 0)) + int(value)

    @staticmethod
    def _merge_where_filters(base: dict[str, Any] | None, extra: dict[str, Any] | None) -> dict[str, Any] | None:
        if not base:
            return dict(extra or {}) or None
        if not extra:
            return dict(base)
        return {"$and": [dict(base), dict(extra)]}

    def embed_query(self, query: str) -> list[float]:
        started = perf_counter()
        embedding = self.embedder.embed(query)
        self._record_stat("embed_query", (perf_counter() - started) * 1000.0, rows=1)
        return embedding

    def snapshot_stats(self, reset: bool = False) -> dict[str, Any]:
        ops = {
            name: {
                "calls": int(values.get("calls", 0)),
                "rows": int(values.get("rows", 0)),
                "total_ms": round(float(values.get("total_ms", 0.0)), 2),
            }
            for name, values in self._stats.items()
        }
        snapshot = {
            "total_ms": round(sum(float(values["total_ms"]) for values in ops.values()), 2),
            "calls": sum(int(values["calls"]) for values in ops.values()),
            "rows": sum(int(values["rows"]) for values in ops.values()),
            "ops": ops,
            "counters": dict(self._counters),
            "embedding_cache": self.embedder.snapshot_stats(reset=reset),
        }
        if reset:
            self._stats = {}
            self._counters = {
                "vector_dedup_hit_count": 0,
                "vector_dedup_miss_count": 0,
            }
        return snapshot

    def _format_query_rows(
        self,
        *,
        id_key: str,
        ids: list[Any],
        docs: list[Any],
        metas: list[Any],
        dists: list[Any],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item_id, doc, meta, dist in zip(ids, docs, metas, dists):
            distance = float(dist or 0.0)
            similarity = self._distance_to_similarity(distance)
            items.append(
                {
                    id_key: str(item_id),
                    "document": doc,
                    "metadata": meta,
                    "distance": distance,
                    "similarity": round(similarity, 4),
                }
            )
        return items

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        chunks = [chunk for chunk in chunks if not chunk.get("skip_vector")]
        if not chunks:
            return
        pending = self._filter_pending_items(
            collection_name=getattr(self.raw_collection, "name", self.config.vector_collection_name),
            items=chunks,
            item_id_key="chunk_id",
            text_key="text",
        )
        if not pending:
            return

        max_batch_size = self._max_batch_size()
        for start_index in range(0, len(pending), max_batch_size):
            batch = pending[start_index : start_index + max_batch_size]
            ids = [str(chunk["chunk_id"]) for chunk in batch]
            documents = [str(chunk["text"]) for chunk in batch]
            metadatas = [
                {
                    "node_id": chunk["node_id"],
                    "shell": int(chunk["shell"]),
                    "sector": chunk["sector"],
                    "zone": chunk["zone"],
                    "cell": chunk["cell"],
                    "chunk_index": int(chunk["chunk_index"]),
                    "grain": str(chunk.get("grain") or "micro"),
                    "scope": str(chunk.get("scope") or "global"),
                    "workspace": str(chunk.get("workspace") or ""),
                    "project": str(chunk.get("project") or ""),
                    "session_id": str(chunk.get("session_id") or ""),
                    "source_path": str(chunk.get("source_path") or ""),
                    "source_type": str(chunk.get("source_type") or ""),
                    "source_ref": str(chunk.get("source_ref") or ""),
                    # Keep these fields in vector metadata so retrieval can avoid
                    # storage fallbacks when ranking large candidate pools.
                    "summary": str(chunk.get("summary") or ""),
                    "retrieval_summary": str(chunk.get("retrieval_summary") or ""),
                    "structured_summary": str(chunk.get("structured_summary") or ""),
                    "retrieval_signature": str(chunk.get("retrieval_signature") or ""),
                    "time_bucket": str(chunk.get("time_bucket") or ""),
                    "entity_tags": str(chunk.get("entity_tags") or ""),
                    "task_type_tag": str(chunk.get("task_type_tag") or ""),
                    "content_ref": str(chunk.get("content_ref") or ""),
                    "access_count": int(chunk.get("access_count") or 0),
                    "neighbor_count": int(chunk.get("neighbor_count") or 0),
                    "created_at": str(chunk.get("created_at") or ""),
                    "timestamp": str(chunk.get("timestamp") or chunk.get("created_at") or ""),
                    "benchmark_name": str(chunk.get("benchmark_name") or ""),
                    "benchmark_adapter_version": str(chunk.get("benchmark_adapter_version") or ""),
                    "source_doc_id": str(chunk.get("source_doc_id") or ""),
                    "source_segment_id": str(chunk.get("source_segment_id") or ""),
                    "sample_id": str(chunk.get("sample_id") or ""),
                    "conversation_id": str(chunk.get("conversation_id") or ""),
                    "turn_id": str(chunk.get("turn_id") or ""),
                    "speaker_id": str(chunk.get("speaker_id") or ""),
                    "original_segment_text": str(chunk.get("original_segment_text") or ""),
                }
                for chunk in batch
            ]
            started = perf_counter()
            embeddings = self.embedder.embed_many(documents)
            self._record_stat("embed_chunks", (perf_counter() - started) * 1000.0, rows=len(documents))
            started = perf_counter()
            self.raw_collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            self._record_stat("upsert_chunks", (perf_counter() - started) * 1000.0, rows=len(ids))
            self._enforce_json_backend_guard()
        self._mark_synced(
            collection_name=getattr(self.raw_collection, "name", self.config.vector_collection_name),
            items=pending,
            item_id_key="chunk_id",
        )

    def upsert_objects(self, objects: list[dict[str, Any]]) -> None:
        if not objects:
            return
        pending = self._filter_pending_items(
            collection_name=getattr(self.object_collection, "name", self.config.object_collection_name),
            items=objects,
            item_id_key="object_id",
            text_key="object_text",
        )
        if not pending:
            return

        max_batch_size = self._max_batch_size()
        for start_index in range(0, len(pending), max_batch_size):
            batch = pending[start_index : start_index + max_batch_size]
            ids = [str(obj["object_id"]) for obj in batch]
            documents = [str(obj["object_text"]) for obj in batch]
            metadatas = [
                {
                    "object_type": obj["object_type"],
                    "source_chunk_id": str(obj.get("source_chunk_id") or ""),
                    "source_node_id": str(obj.get("source_node_id") or ""),
                    "scope": str(obj.get("scope") or "global"),
                    "workspace": str(obj.get("workspace") or ""),
                    "project": str(obj.get("project") or ""),
                    "session_id": str(obj.get("session_id") or ""),
                    "status": str(obj.get("status") or "active"),
                    "entity": str(obj.get("entity") or ""),
                    "attribute": str(obj.get("attribute") or ""),
                    "canonical_key": str(obj.get("canonical_key") or ""),
                    "temporal_marker": str(obj.get("temporal_marker") or ""),
                    "source_type": str(obj.get("source_type") or ""),
                    "source_ref": str(obj.get("source_ref") or ""),
                }
                for obj in batch
            ]
            started = perf_counter()
            embeddings = self.embedder.embed_many(documents)
            self._record_stat("embed_objects", (perf_counter() - started) * 1000.0, rows=len(documents))
            started = perf_counter()
            self.object_collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            self._record_stat("upsert_objects", (perf_counter() - started) * 1000.0, rows=len(ids))
            self._enforce_json_backend_guard()
        self._mark_synced(
            collection_name=getattr(self.object_collection, "name", self.config.object_collection_name),
            items=pending,
            item_id_key="object_id",
        )

    def upsert_representations(self, representations: list[dict[str, Any]]) -> None:
        if not representations:
            return
        pending = self._filter_pending_items(
            collection_name=getattr(self.proxy_collection, "name", self.config.proxy_collection_name),
            items=representations,
            item_id_key="representation_id",
            text_key="text",
        )
        if not pending:
            return

        max_batch_size = self._max_batch_size()
        for start_index in range(0, len(pending), max_batch_size):
            batch = pending[start_index : start_index + max_batch_size]
            ids = [str(item["representation_id"]) for item in batch]
            documents = [str(item.get("text") or "") for item in batch]
            metadatas = [
                {
                    "parent_id": str(item.get("parent_id") or ""),
                    "parent_type": str(item.get("parent_type") or "chunk"),
                    "proxy_kind": str(item.get("proxy_kind") or "summary"),
                    "scope": str(item.get("scope") or "global"),
                    "workspace": str(item.get("workspace") or ""),
                    "project": str(item.get("project") or ""),
                    "session_id": str(item.get("session_id") or ""),
                    "time_bucket": str(item.get("time_bucket") or ""),
                    "entity_tags": str(item.get("entity_tags") or ""),
                    "task_type_tag": str(item.get("task_type_tag") or ""),
                }
                for item in batch
            ]
            started = perf_counter()
            embeddings = self.embedder.embed_many(documents)
            self._record_stat("embed_representations", (perf_counter() - started) * 1000.0, rows=len(documents))
            started = perf_counter()
            self.proxy_collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            self._record_stat("upsert_representations", (perf_counter() - started) * 1000.0, rows=len(ids))
            self._enforce_json_backend_guard()
        self._mark_synced(
            collection_name=getattr(self.proxy_collection, "name", self.config.proxy_collection_name),
            items=pending,
            item_id_key="representation_id",
        )

    def search(self, query: str, top_k: int = 8, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        query_embedding = self.embed_query(query)
        return self.search_by_embedding(query_embedding, top_k=top_k, where=where)

    def search_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        started = perf_counter()
        result = self._query_with_retry(
            collection_name="raw",
            query_embedding=query_embedding,
            n_results=max(1, top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        self._increment_counter("dense_search_count", 1)
        self._record_stat("search_chunks", (perf_counter() - started) * 1000.0, rows=len(result.get("ids", [[]])[0]))
        return self._format_query_rows(
            id_key="chunk_id",
            ids=list(result.get("ids", [[]])[0]),
            docs=list(result.get("documents", [[]])[0]),
            metas=list(result.get("metadatas", [[]])[0]),
            dists=list(result.get("distances", [[]])[0]),
        )

    def search_many(self, queries: list[str], top_k: int = 8, where: dict[str, Any] | None = None) -> list[list[dict[str, Any]]]:
        clean_queries = [query for query in queries if query.strip()]
        if not clean_queries:
            return []
        started = perf_counter()
        query_embeddings = self.embedder.embed_many(clean_queries)
        self._record_stat("embed_query", (perf_counter() - started) * 1000.0, rows=len(clean_queries))
        try:
            query_batch_size = max(1, int(os.getenv("SPHERE_VECTOR_QUERY_BATCH_SIZE", "64") or "64"))
        except ValueError:
            query_batch_size = 64
        started = perf_counter()
        result_ids: list[list[Any]] = []
        result_docs: list[list[Any]] = []
        result_metas: list[list[Any]] = []
        result_dists: list[list[Any]] = []
        for start_index in range(0, len(query_embeddings), query_batch_size):
            batch_embeddings = query_embeddings[start_index : start_index + query_batch_size]
            result = self._query_many_with_retry(
                collection_name="raw",
                query_embeddings=batch_embeddings,
                n_results=max(1, top_k),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            result_ids.extend(list(result.get("ids", [])))
            result_docs.extend(list(result.get("documents", [])))
            result_metas.extend(list(result.get("metadatas", [])))
            result_dists.extend(list(result.get("distances", [])))
        self._increment_counter("dense_search_count", len(clean_queries))
        if len(clean_queries) > 1:
            self._increment_counter("query_embedding_reuse_count", len(clean_queries) - 1)
        if query_batch_size > 1:
            self._increment_counter("batched_query_call_count", (len(clean_queries) + query_batch_size - 1) // query_batch_size)
        self._record_stat("search_chunks", (perf_counter() - started) * 1000.0, rows=sum(len(ids) for ids in result_ids))
        all_items: list[list[dict[str, Any]]] = []
        for ids, docs, metas, dists in zip(result_ids, result_docs, result_metas, result_dists):
            all_items.append(
                self._format_query_rows(
                    id_key="chunk_id",
                    ids=list(ids),
                    docs=list(docs),
                    metas=list(metas),
                    dists=list(dists),
                )
            )
        return all_items

    def search_objects(self, query: str, top_k: int = 8, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        query_embedding = self.embed_query(query)
        return self.search_objects_by_embedding(query_embedding, top_k=top_k, where=where)

    def search_objects_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        started = perf_counter()
        result = self._query_with_retry(
            collection_name="object",
            query_embedding=query_embedding,
            n_results=max(1, top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        self._increment_counter("object_search_count", 1)
        self._record_stat("search_objects", (perf_counter() - started) * 1000.0, rows=len(result.get("ids", [[]])[0]))
        return self._format_query_rows(
            id_key="object_id",
            ids=list(result.get("ids", [[]])[0]),
            docs=list(result.get("documents", [[]])[0]),
            metas=list(result.get("metadatas", [[]])[0]),
            dists=list(result.get("distances", [[]])[0]),
        )

    def search_proxies(
        self,
        query: str,
        top_k: int = 8,
        where: dict[str, Any] | None = None,
        proxy_kinds: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        query_embedding = self.embed_query(query)
        return self.search_proxies_by_embedding(query_embedding, top_k=top_k, where=where, proxy_kinds=proxy_kinds)

    def search_proxies_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
        proxy_kinds: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        allowed_kinds = set(proxy_kinds or [])
        effective_where = self._merge_where_filters(
            where,
            {"proxy_kind": {"$in": sorted(allowed_kinds)}} if allowed_kinds else None,
        )
        started = perf_counter()
        result = self._query_with_retry(
            collection_name="proxy",
            query_embedding=query_embedding,
            n_results=max(1, top_k),
            where=effective_where,
            include=["documents", "metadatas", "distances"],
        )
        raw_hits = len(result.get("ids", [[]])[0])
        self._increment_counter("proxy_search_count", 1)
        if allowed_kinds:
            self._increment_counter("prefilter_applied", 1)
            self._increment_counter("raw_vector_hits", raw_hits)
        self._record_stat("search_proxies", (perf_counter() - started) * 1000.0, rows=len(result.get("ids", [[]])[0]))
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        items: list[dict[str, Any]] = []
        for representation_id, doc, meta, dist in zip(ids, docs, metas, dists):
            if allowed_kinds and str((meta or {}).get("proxy_kind") or "") not in allowed_kinds:
                self._increment_counter("filtered_out_count", 1)
                continue
            distance = float(dist or 0.0)
            similarity = self._distance_to_similarity(distance)
            items.append(
                {
                    "representation_id": representation_id,
                    "document": doc,
                    "metadata": meta,
                    "distance": distance,
                    "similarity": round(similarity, 4),
                }
            )
        if allowed_kinds:
            self._increment_counter("postfilter_hits", len(items))
        return items

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.raw_collection.delete(ids=chunk_ids)
            if self.storage is not None:
                self.storage.delete_vector_sync_state(
                    getattr(self.raw_collection, "name", self.config.vector_collection_name),
                    chunk_ids,
                )

    def delete_objects(self, object_ids: list[str]) -> None:
        if object_ids:
            self.object_collection.delete(ids=object_ids)
            if self.storage is not None:
                self.storage.delete_vector_sync_state(
                    getattr(self.object_collection, "name", self.config.object_collection_name),
                    object_ids,
                )

    def delete_representations(self, representation_ids: list[str]) -> None:
        if representation_ids:
            self.proxy_collection.delete(ids=representation_ids)
            if self.storage is not None:
                self.storage.delete_vector_sync_state(
                    getattr(self.proxy_collection, "name", self.config.proxy_collection_name),
                    representation_ids,
                )

    def count(self) -> int:
        return int(self.raw_collection.count())

    def info(self) -> dict[str, Any]:
        return {
            "raw_collection": getattr(self.raw_collection, "name", self.config.vector_collection_name),
            "raw_count": self.count(),
            "object_collection": getattr(self.object_collection, "name", self.config.object_collection_name),
            "object_count": int(self.object_collection.count()),
            "proxy_collection": getattr(self.proxy_collection, "name", self.config.proxy_collection_name),
            "proxy_count": int(self.proxy_collection.count()),
            "embedding_provider": self.embedder.info.provider,
            "embedding_model": self.embedder.info.model_name,
            "embedding_dim": int(self.embedder.info.embedding_dim),
            "embedding_preprocess_version": EMBEDDING_PREPROCESS_VERSION,
            "normalize_embeddings": True,
            "fallback_in_use": self.embedder.info.fallback_in_use,
            "embedding_cache": self.embedder.cache_stats(),
            "vector_backend": self._vector_backend_resolved,
            "vector_backend_requested": self.config.vector_backend,
            "vector_fallback_in_use": self._fallback_in_use,
            "vector_count": int(self.raw_collection.count()) + int(self.object_collection.count()) + int(self.proxy_collection.count()),
            "json_scan_warning": self._json_scan_warning,
            "primary_embedding_load_error": getattr(self.embedder, "primary_load_error", None),
        }

    def close(self) -> None:
        self.embedder.close()
        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            close_client()

    def _filter_pending_items(
        self,
        collection_name: str,
        items: list[dict[str, Any]],
        item_id_key: str,
        text_key: str,
    ) -> list[dict[str, Any]]:
        states = (
            self.storage.fetch_vector_sync_state(collection_name, [str(item[item_id_key]) for item in items])
            if self.storage is not None
            else {}
        )
        pending: list[dict[str, Any]] = []
        skipped = 0
        for item in items:
            item_id = str(item[item_id_key])
            content_hash = str(item.get("content_hash") or stable_content_hash(str(item.get(text_key) or "")))
            item["content_hash"] = content_hash
            if states.get(item_id) == content_hash:
                skipped += 1
                continue
            pending.append(item)
        self._increment_counter("vector_dedup_hit_count", skipped)
        self._increment_counter("vector_dedup_miss_count", len(pending))
        if skipped:
            self._record_stat(f"skip_{item_id_key}_upserts", 0.0, rows=skipped)
        return pending

    def _mark_synced(self, collection_name: str, items: list[dict[str, Any]], item_id_key: str) -> None:
        if self.storage is None or not items:
            return
        self.storage.upsert_vector_sync_state(
            collection_name,
            [
                (str(item[item_id_key]), str(item.get("content_hash") or ""))
                for item in items
                if item.get(item_id_key) and item.get("content_hash")
            ],
        )

    def _reconcile_sync_state(self) -> None:
        if self.storage is None:
            return
        raw_name = getattr(self.raw_collection, "name", self.config.vector_collection_name)
        object_name = getattr(self.object_collection, "name", self.config.object_collection_name)
        proxy_name = getattr(self.proxy_collection, "name", self.config.proxy_collection_name)
        if self.raw_collection.count() == 0 and self.storage.count_vector_sync_state(raw_name) > 0:
            self.storage.clear_vector_sync_state(raw_name)
        if self.object_collection.count() == 0 and self.storage.count_vector_sync_state(object_name) > 0:
            self.storage.clear_vector_sync_state(object_name)
        if self.proxy_collection.count() == 0 and self.storage.count_vector_sync_state(proxy_name) > 0:
            self.storage.clear_vector_sync_state(proxy_name)
