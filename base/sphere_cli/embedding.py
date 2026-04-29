from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Protocol, Sequence, cast

import numpy as np

from .embedding_cache import PersistentEmbeddingCache
from .utils import normalize_text_for_hash, stable_content_hash, tokenize


class _SentenceTransformerModel(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = ...,
        normalize_embeddings: bool = ...,
    ) -> Any: ...


_ST_MODEL_CACHE: dict[str, _SentenceTransformerModel] = {}
_COMPUTE_MS_HINTS: dict[str, float] = {}


class LocalHashEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"local-hash-{dim}"
        self.is_available = True

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec.tolist()
        for tok in tokens:
            idx = abs(hash(tok)) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32).tolist()

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, fail_fast: bool = False) -> None:
        self.model_name = model_name
        self.name = model_name
        self._model: _SentenceTransformerModel | None = None
        self.is_available = False
        self.load_error: str | None = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = _ST_MODEL_CACHE.get(model_name)
            if model is None:
                model = cast(_SentenceTransformerModel, SentenceTransformer(model_name))
                _ST_MODEL_CACHE[model_name] = model
            self._model = model
            self.is_available = True
        except Exception as exc:
            self._model = None
            self.is_available = False
            self.load_error = f"{exc.__class__.__name__}: {exc}"
            if fail_fast:
                raise RuntimeError(
                    f"Embedding model {model_name!r} is unavailable and SPHERE_EMBEDDING_FAIL_FAST=1 is set. "
                    "Install/cache the model or unset fail-fast for development fallback."
                ) from exc

    def embed(self, text: str) -> list[float]:
        if not self.is_available or self._model is None:
            raise RuntimeError("SentenceTransformer model is not available")
        encoded = self._model.encode([text], normalize_embeddings=True)
        return [float(value) for value in encoded[0].tolist()]

    def embed_many(self, texts: Iterable[str], batch_size: int = 64) -> list[list[float]]:
        rows = list(texts)
        if not rows:
            return []
        if not self.is_available or self._model is None:
            raise RuntimeError("SentenceTransformer model is not available")
        encoded = self._model.encode(rows, batch_size=batch_size, normalize_embeddings=True)
        return [[float(value) for value in row] for row in encoded.tolist()]


@dataclass
class EmbedderInfo:
    provider: str
    model_name: str
    fallback_in_use: bool


class EmbeddingProvider:
    def __init__(
        self,
        preferred_model_name: str,
        fallback_dim: int = 384,
        cache_size: int = 10000,
        cache_path: str | Path | None = None,
        fail_fast: bool = False,
    ) -> None:
        self.primary = SentenceTransformerEmbedder(preferred_model_name, fail_fast=fail_fast)
        self.fallback = LocalHashEmbedder(fallback_dim)
        self.active = self.primary if self.primary.is_available else self.fallback
        self.info = EmbedderInfo(
            provider="sentence_transformer" if self.primary.is_available else "local_hash",
            model_name=self.active.name,
            fallback_in_use=not self.primary.is_available,
        )
        self.primary_load_error = self.primary.load_error
        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size
        self._persistent_cache = PersistentEmbeddingCache(cache_path) if cache_path else None
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_hit_ms_saved = 0.0
        self._actual_embedding_compute_ms = 0.0
        self._avg_compute_ms_per_miss = float(_COMPUTE_MS_HINTS.get(self._hint_key(), 0.0))

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        rows = list(texts)
        if not rows:
            return []

        results: list[list[float] | None] = [None] * len(rows)
        unique_requests: dict[str, dict[str, Any]] = {}
        for index, text in enumerate(rows):
            normalized = normalize_text_for_hash(text)
            content_hash = stable_content_hash(normalized)
            cache_key = self._cache_key(content_hash)
            request = unique_requests.setdefault(
                cache_key,
                {
                    "indexes": [],
                    "normalized_text": normalized,
                    "content_hash": content_hash,
                },
            )
            request["indexes"].append(index)

        missing_cache_keys: list[str] = []
        for cache_key, request in unique_requests.items():
            cached = self._cache.get(cache_key)
            if cached is None:
                missing_cache_keys.append(cache_key)
                continue
            for index in request["indexes"]:
                results[index] = cached

        if self._persistent_cache is not None and missing_cache_keys:
            persisted = self._persistent_cache.get_many(missing_cache_keys)
            for cache_key, embedding in persisted.items():
                request = unique_requests[cache_key]
                self._remember(cache_key, embedding)
                for index in request["indexes"]:
                    results[index] = embedding
            missing_cache_keys = [cache_key for cache_key in missing_cache_keys if cache_key not in persisted]

        compute_ms = 0.0
        computed_embeddings: dict[str, list[float]] = {}
        if missing_cache_keys:
            started = perf_counter()
            fresh_embeddings = self.active.embed_many(
                [unique_requests[cache_key]["normalized_text"] for cache_key in missing_cache_keys]
            )
            compute_ms = (perf_counter() - started) * 1000.0
            self._actual_embedding_compute_ms += compute_ms
            per_miss_ms = compute_ms / len(missing_cache_keys)
            if self._avg_compute_ms_per_miss <= 0.0:
                self._avg_compute_ms_per_miss = per_miss_ms
            else:
                self._avg_compute_ms_per_miss = (self._avg_compute_ms_per_miss * 0.7) + (per_miss_ms * 0.3)
            _COMPUTE_MS_HINTS[self._hint_key()] = self._avg_compute_ms_per_miss
            for cache_key, embedding in zip(missing_cache_keys, fresh_embeddings):
                computed_embeddings[cache_key] = embedding
                self._remember(cache_key, embedding)
                for index in unique_requests[cache_key]["indexes"]:
                    results[index] = embedding
            if self._persistent_cache is not None:
                self._persistent_cache.put_many(
                    [
                        (
                            cache_key,
                            str(unique_requests[cache_key]["content_hash"]),
                            str(unique_requests[cache_key]["normalized_text"]),
                            embedding,
                            self.info.provider,
                            self.info.model_name,
                        )
                        for cache_key, embedding in computed_embeddings.items()
                    ]
                )

        unresolved = [index for index, value in enumerate(results) if value is None]
        if unresolved:
            raise RuntimeError(f"Missing embeddings for {len(unresolved)} request(s)")

        miss_count = len(missing_cache_keys)
        hit_count = len(rows) - miss_count
        self._cache_hits += hit_count
        self._cache_misses += miss_count
        estimated_per_hit_ms = (compute_ms / miss_count) if miss_count > 0 else self._avg_compute_ms_per_miss
        self._cache_hit_ms_saved += hit_count * max(estimated_per_hit_ms, 0.0)
        return results  # type: ignore[return-value]

    def cache_stats(self) -> dict[str, int | float]:
        stats = self.snapshot_stats(reset=False)
        return {
            "entries": int(stats["entries"]),
            "persistent_entries": self._persistent_cache.count() if self._persistent_cache is not None else 0,
            "hits": int(stats["embedding_cache_hit_count"]),
            "misses": int(stats["embedding_cache_miss_count"]),
            "hit_ms_saved": float(stats["embedding_cache_hit_ms_saved"]),
            "actual_embedding_compute_ms": float(stats["actual_embedding_compute_ms"]),
        }

    def close(self) -> None:
        if self._persistent_cache is not None:
            self._persistent_cache.close()

    def snapshot_stats(self, reset: bool = False) -> dict[str, int | float]:
        snapshot = {
            "entries": len(self._cache),
            "embedding_cache_hit_count": self._cache_hits,
            "embedding_cache_miss_count": self._cache_misses,
            "embedding_cache_hit_ms_saved": round(self._cache_hit_ms_saved, 2),
            "actual_embedding_compute_ms": round(self._actual_embedding_compute_ms, 2),
        }
        if reset:
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_hit_ms_saved = 0.0
            self._actual_embedding_compute_ms = 0.0
        return snapshot

    def _cache_key(self, content_hash: str) -> str:
        return f"{self.info.provider}:{self.info.model_name}:{content_hash}"

    def _remember(self, cache_key: str, embedding: list[float]) -> None:
        if cache_key in self._cache or len(self._cache) < self._cache_size:
            self._cache[cache_key] = embedding

    def _hint_key(self) -> str:
        return f"{self.info.provider}:{self.info.model_name}"
