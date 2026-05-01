#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "base"
BENCHMARKS = BASE / "benchmarks"
for path in (str(BASE), str(BENCHMARKS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_support import build_index_metadata, rank_benchmark_sources  # noqa: E402
from sphere_cli.config import AppConfig  # noqa: E402


class SmokeVectorStore:
    def __init__(self, source_records: dict[str, dict[str, Any]]) -> None:
        self.source_records = source_records

    def search(self, query: str, top_k: int = 8, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows = []
        query_terms = {term.lower() for term in query.split() if len(term) > 2}
        for source_id, record in self.source_records.items():
            text = str(record.get("normalized_text") or record.get("text") or "").lower()
            overlap = sum(1 for term in query_terms if term in text)
            if overlap <= 0:
                continue
            rows.append(
                {
                    "chunk_id": f"chunk-{source_id}",
                    "document": record.get("text") or "",
                    "metadata": {
                        "source_segment_id": source_id,
                        "source_doc_id": record.get("source_doc_id"),
                        "benchmark_name": record.get("benchmark_name"),
                    },
                    "similarity": min(0.99, 0.45 + overlap * 0.12),
                }
            )
        rows.sort(key=lambda item: (-float(item.get("similarity") or 0.0), str(item.get("chunk_id") or "")))
        return rows[: max(1, top_k)]

    def search_objects(self, query: str, top_k: int = 8, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return []

    def info(self) -> dict[str, Any]:
        return {
            "embedding_provider": "sentence_transformer",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "normalize_embeddings": True,
            "fallback_in_use": False,
            "embedding_preprocess_version": "normalize_text_for_hash_v2",
            "vector_backend": "smoke",
            "vector_fallback_in_use": False,
            "vector_count": len(self.source_records),
            "json_scan_warning": "",
        }


class SmokeStorage:
    def search_chunks_fts(self, query: str, limit: int) -> list[dict[str, Any]]:
        return []

    def search_objects_fts(self, query: str, limit: int, object_types: list[str] | None = None) -> list[dict[str, Any]]:
        return []

    def fetch_objects_by_ids(self, object_ids: list[str]) -> list[dict[str, Any]]:
        return []

    def get_retrieval_cache(self, query_fingerprint: str, memory_version: int) -> dict[str, Any] | None:
        return None

    def put_retrieval_cache(self, **kwargs: Any) -> None:
        return None


def _sample_corpus() -> list[dict[str, Any]]:
    return [
        {"source_segment_id": "seg-1", "source_doc_id": "session-a", "session_id": "session-a", "text": "Melanie prefers strong coffee in the workshop.", "timestamp": "2024-03-02"},
        {"source_segment_id": "seg-2", "source_doc_id": "session-a", "session_id": "session-a", "text": "Later Melanie adjusted the grinder setting.", "timestamp": "2024-03-03"},
        {"source_segment_id": "seg-3", "source_doc_id": "session-b", "session_id": "session-b", "text": "Nora likes tea and garden planning.", "timestamp": "2024-04-04"},
        {"source_segment_id": "seg-4", "source_doc_id": "session-b", "session_id": "session-b", "text": "CloneMem admission needs parent session expansion for segment recall.", "timestamp": "2024-04-05"},
    ]


def _latest_result_comparison() -> dict[str, Any]:
    result_root = ROOT.parent / "BenchmarkResult"
    if not result_root.exists():
        return {"available": False, "reason": "BenchmarkResult directory not found"}
    candidates = sorted([path for path in result_root.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True)
    result_summaries = []
    for result_dir in candidates[:12]:
        metric_files = sorted(result_dir.rglob("*metrics*.json"))
        if not metric_files:
            continue
        result_summaries.append(
            {
                "result_dir": str(result_dir),
                "metrics": [_metric_file_summary(path) for path in metric_files[:8]],
            }
        )
        if len(result_summaries) >= 2:
            break
    if result_summaries:
        comparison: dict[str, Any] = {"available": True, "latest": result_summaries[0]}
        if len(result_summaries) > 1:
            comparison["previous"] = result_summaries[1]
            comparison["delta"] = _metric_delta(result_summaries[0], result_summaries[1])
        return comparison
    return {"available": False, "reason": "no recent metrics json found"}


def _nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_metric(payload: dict[str, Any], paths: list[tuple[str, ...]]) -> float | None:
    for path in paths:
        value = _nested_get(payload, path)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _metric_file_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}
    benchmark = path.parent.name
    totals = dict(payload.get("totals_ms") or {})
    retrieval_ms = totals.get("total_retrieval_ms")
    if not isinstance(retrieval_ms, (int, float)):
        retrieval_ms = payload.get("retrieval_time_ms")
    metrics = {
        "candidate_recall@100": _first_metric(
            payload,
            [
                ("candidate_recall_summary", "candidate_recall@100"),
                ("candidate_recall@100",),
                ("metrics", "candidate_recall@100"),
            ],
        ),
        "recall@5": _first_metric(
            payload,
            [
                ("metrics", "segment", "recall_frac@5"),
                ("metrics", "message", "recall@5"),
                ("recall@5",),
            ],
        ),
        "recall@10": _first_metric(
            payload,
            [
                ("metrics", "segment", "recall_frac@10"),
                ("metrics", "message", "recall@10"),
                ("recall@10",),
            ],
        ),
        "ndcg@10": _first_metric(
            payload,
            [
                ("metrics", "segment", "ndcg_any@10"),
                ("metrics", "message", "ndcg@10"),
                ("ndcg@10",),
            ],
        ),
    }
    return {
        "benchmark": benchmark,
        "path": str(path),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "retrieval_time_ms": float(retrieval_ms) if isinstance(retrieval_ms, (int, float)) else None,
        "metrics": {key: value for key, value in metrics.items() if value is not None},
    }


def _metric_delta(latest: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    previous_by_benchmark = {str(item.get("benchmark") or ""): item for item in list(previous.get("metrics") or [])}
    deltas: dict[str, Any] = {}
    for item in list(latest.get("metrics") or []):
        benchmark = str(item.get("benchmark") or "")
        prev = previous_by_benchmark.get(benchmark)
        if not benchmark or not prev:
            continue
        row: dict[str, Any] = {}
        for key in ("retrieval_time_ms", "elapsed_seconds"):
            current_value = item.get(key)
            previous_value = prev.get(key)
            if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                row[f"{key}_delta"] = round(float(current_value) - float(previous_value), 4)
        latest_metrics = dict(item.get("metrics") or {})
        previous_metrics = dict(prev.get("metrics") or {})
        metric_deltas = {}
        for key, current_value in latest_metrics.items():
            previous_value = previous_metrics.get(key)
            if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                metric_deltas[f"{key}_delta"] = round(float(current_value) - float(previous_value), 6)
        if metric_deltas:
            row["metrics"] = metric_deltas
        if row:
            deltas[benchmark] = row
    return deltas


def main() -> int:
    corpus = _sample_corpus()
    config = AppConfig(base_dir=ROOT)
    config.route_aware_gating_enabled = True
    config.retrieval_early_exit_enabled = True
    config.retrieval_min_seed_candidates = 2
    config.candidate_recall_eval_k = 2
    metadata = build_index_metadata(
        corpus_items=corpus,
        chunks=[
            {
                "chunk_id": f"chunk-{item['source_segment_id']}",
                "text": item["text"],
                "source_segment_id": item["source_segment_id"],
                "source_doc_id": item["source_doc_id"],
                "benchmark_name": "knowme",
            }
            for item in corpus
        ],
        benchmark_name="knowme",
        adapter_version="smoke",
        runtime_fingerprint={
            "embedding_provider": "sentence_transformer",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "normalize_embeddings": True,
            "embedding_preprocess_version": "normalize_text_for_hash_v2",
            "fingerprint_hash": "smoke",
        },
        raw_counts={"memory_count": len(corpus), "question_count": 1},
    )
    vector_store = SmokeVectorStore(metadata["source_records_by_id"])
    storage = SmokeStorage()
    start = perf_counter()
    with redirect_stdout(StringIO()):
        result = rank_benchmark_sources(
            query="What coffee does Melanie prefer in the workshop?",
            benchmark_name="knowme",
            vector_store=vector_store,
            storage=storage,
            index_metadata=metadata,
            route_context={},
            config=config,
            pool_limit=20,
        )
    timings = dict(result.get("timings_ms") or {})
    payload = {
        "total_ms": round((perf_counter() - start) * 1000.0, 2),
        "dense_ms": timings.get("dense_ms", 0.0),
        "lexical_ms": timings.get("lexical_ms", 0.0),
        "entity_ms": timings.get("entity_ms", 0.0),
        "temporal_ms": timings.get("temporal_ms", 0.0),
        "exact_phrase_ms": timings.get("exact_phrase_ms", 0.0),
        "profile_ms": timings.get("profile_ms", 0.0),
        "decomposition_channel_ms": timings.get("decomposition_channel_ms", 0.0),
        "session_bundle_ms": timings.get("session_bundle_ms", 0.0),
        "temporal_neighbor_ms": timings.get("temporal_neighbor_ms", 0.0),
        "parent_session_ms": timings.get("parent_session_ms", 0.0),
        "fusion_ms": timings.get("fusion_ms", 0.0),
        "rerank_ms": timings.get("rerank_ms", 0.0),
        "inhibition_ms": timings.get("inhibition_ms", 0.0),
        "skipped_channels": list(result.get("gated_channels") or []) + list((result.get("early_exit") or {}).get("skipped_expensive_channels") or []),
        "early_exit_triggered": bool((result.get("early_exit") or {}).get("early_exit_triggered")),
        "candidate_counts_by_channel": dict(result.get("candidate_source_stats") or {}),
        "side_index_audit": dict(result.get("side_index_audit") or {}),
        "side_index_audit_summary": dict(result.get("side_index_audit_summary") or {}),
        "latest_artifact_comparison": _latest_result_comparison(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
