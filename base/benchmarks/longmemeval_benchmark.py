from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.activation_engine import ActivationEngine
from sphere_cli.config import AppConfig
from sphere_cli.evidence_pipeline import EvidencePipeline
from sphere_cli.memory_writer import MemoryWriter
from sphere_cli.models import MemoryNode
from sphere_cli.path_router import PathRouter
from sphere_cli.reranker import RetrievalReranker
from sphere_cli.storage import Storage
from sphere_cli.utils import deterministic_angle, stable_content_hash
from sphere_cli.vector_store import VectorStore
from benchmark_support import (
    BENCHMARK_CHUNKER_VERSION,
    annotate_chunks_for_benchmark,
    assert_benchmark_vector_guard,
    best_gold_rank,
    build_candidate_recall_summary,
    build_index_metadata,
    build_integrity_report,
    build_oracle_retrieval_report,
    build_per_channel_contribution_report,
    build_performance_cache_report,
    build_query_diagnostic,
    build_raw_counts,
    build_query_failure,
    build_result_metadata,
    build_runtime_fingerprint,
    build_topk_debug_record,
    candidate_rows,
    configure_benchmark_determinism,
    force_benchmark_config,
    ndcg_at,
    rank_benchmark_sources,
    recall_at,
    report_root,
    write_json,
    write_jsonl,
)
from token_economy_support import add_token_economy_args, record_token_economy_for_metrics

KS = [1, 3, 5, 10, 30, 50]
LONGMEMEVAL_ADAPTER_VERSION = "2026-04-25"
BENCHMARK_WORKSPACE_CACHE_VERSION = "longmemeval_v9"
FOCUSED_QUERY_STOPWORDS = {
    "again",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "after",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "for",
    "with",
    "from",
    "into",
    "about",
    "back",
    "can",
    "confirm",
    "could",
    "earlier",
    "if",
    "last",
    "looking",
    "our",
    "planning",
    "plan",
    "previous",
    "provided",
    "remind",
    "reminder",
    "suggest",
    "suggested",
    "time",
    "their",
    "they",
    "them",
    "upcoming",
    "wanted",
    "wondering",
    "would",
    "you",
    "kind",
    "type",
    "types",
    "kinds",
}
FOCUSED_QUERY_DOMAIN_RE = re.compile(
    r"\b(?:destination|trip|travel|vacation|holiday|city|country|visit|game|games|gaming)\b",
    re.IGNORECASE,
)
FOCUSED_QUERY_PRIORITY_RE = re.compile(
    r"^(?:19|20)\d{2}$|(?:type|kind|style|genre|category|change|changed|next|after|factor|reason|because|why)",
    re.IGNORECASE,
)
FOCUSED_QUERY_ANCHOR_TOKEN_RE = re.compile(
    r"^(?:hotel|hostel|phone|number|tourism|board|chess|move|vegan|eatery|restaurant|locations?)$",
    re.IGNORECASE,
)
FOCUSED_QUERY_CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
FOCUSED_QUERY_CHESS_TOKEN_RE = re.compile(r"\b(?:\d+|[KQRBN]?[a-h][1-8][+#]?)\b")
FOCUSED_QUERY_PHRASE_HINTS = (
    "phone number",
    "tourism board",
    "multiple locations",
)


def path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


WORKSPACE_CACHE_ROOT = path_from_env(
    "SPHERE_LONGMEMEVAL_CACHE_ROOT",
    ROOT / "benchmarks" / ".cache" / "longmemeval_workspaces",
)
SHARED_EMBEDDING_CACHE_DIR = path_from_env(
    "SPHERE_LONGMEMEVAL_EMBED_CACHE_ROOT",
    ROOT / "benchmarks" / ".cache" / "embedding_cache",
)


@dataclass
class BenchmarkWorkspace:
    signature: str
    base_dir: Path
    config: AppConfig
    storage: Storage
    vector_store: VectorStore
    writer: MemoryWriter
    activation: ActivationEngine
    router: PathRouter
    reranker: RetrievalReranker
    pipeline: EvidencePipeline
    vector_info: dict[str, Any]
    ingest_profile: dict[str, Any]
    index_metadata: dict[str, Any]
    build_elapsed_ms: float
    manifest_path: Path

    def close(self) -> None:
        self.vector_store.close()
        self.storage.close_persistent()


def dcg(relevances: list[float], k: int) -> float:
    score = 0.0
    for index, rel in enumerate(relevances[:k]):
        score += rel / math.log2(index + 2)
    return score


def ndcg(ranked_ids: list[str], correct_ids: set[str], k: int) -> float:
    relevances = [1.0 if item_id in correct_ids else 0.0 for item_id in ranked_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg(relevances, k) / idcg


def evaluate_retrieval(ranked_ids: list[str], correct_ids: set[str], k: int) -> tuple[float, float, float]:
    top_ids = set(ranked_ids[:k])
    recall_any = float(any(item_id in top_ids for item_id in correct_ids))
    recall_all = float(all(item_id in top_ids for item_id in correct_ids))
    return recall_any, recall_all, ndcg(ranked_ids, correct_ids, k)


def session_id_from_corpus_id(corpus_id: str) -> str:
    if "_turn_" in corpus_id:
        return corpus_id.rsplit("_turn_", 1)[0]
    return corpus_id


def remap_rank_rows(rows: list[dict[str, Any]], id_mapper: Any) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        source_id = id_mapper(str(row.get("source_id") or row.get("source_segment_id") or ""))
        if not source_id or source_id in seen:
            continue
        remapped.append({**row, "source_id": source_id, "source_segment_id": source_id})
        seen.add(source_id)
    return remapped


def build_corpus(entry: dict[str, Any], granularity: str) -> list[dict[str, Any]]:
    corpus_items: list[dict[str, Any]] = []
    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]

    for session, session_id, date in zip(sessions, session_ids, dates):
        if granularity == "session":
            user_turns = [turn["content"].strip() for turn in session if turn["role"] == "user" and turn["content"].strip()]
            if not user_turns:
                continue
            corpus_items.append(
                {
                    "corpus_id": session_id,
                    "session_id": session_id,
                    "timestamp": date,
                    "text": "\n".join(user_turns),
                }
            )
            continue

        turn_number = 0
        for turn in session:
            if turn["role"] != "user":
                continue
            content = turn["content"].strip()
            if not content:
                continue
            corpus_items.append(
                {
                    "corpus_id": f"{session_id}_turn_{turn_number}",
                    "session_id": session_id,
                    "timestamp": date,
                    "text": content,
                }
            )
            turn_number += 1

    return corpus_items


def make_summary(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def normalize_question_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return " ".join(lowered.split())


def dedupe_preserving_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def extract_focus_anchor_tokens(question: str, normalized: str) -> list[str]:
    anchors: list[str] = []

    for match in FOCUSED_QUERY_CAPITALIZED_PHRASE_RE.finditer(question or ""):
        for token in normalize_question_text(match.group(0)).split():
            if len(token) >= 3 and token not in FOCUSED_QUERY_STOPWORDS:
                anchors.append(token)

    for phrase in FOCUSED_QUERY_PHRASE_HINTS:
        if phrase in normalized:
            anchors.extend(token for token in phrase.split() if token not in FOCUSED_QUERY_STOPWORDS)

    if "chess" in normalized:
        for match in FOCUSED_QUERY_CHESS_TOKEN_RE.finditer(question or ""):
            token = normalize_question_text(match.group(0))
            if token and len(token) >= 2:
                anchors.append(token)

    return dedupe_preserving_order(anchors)


def build_focused_query(question: str) -> str | None:
    normalized = normalize_question_text(question)
    if not normalized or FOCUSED_QUERY_DOMAIN_RE.search(normalized) is None:
        return None
    tokens = [token for token in normalized.split() if len(token) >= 3 and token not in FOCUSED_QUERY_STOPWORDS]
    if not tokens:
        return None
    anchor_tokens = extract_focus_anchor_tokens(question, normalized)
    anchor_token_set = set(anchor_tokens)
    prioritized: list[str] = list(anchor_tokens)
    fallback: list[str] = []
    for token in tokens:
        if token in anchor_token_set:
            continue
        if FOCUSED_QUERY_PRIORITY_RE.search(token) or FOCUSED_QUERY_ANCHOR_TOKEN_RE.search(token):
            prioritized.append(token)
        else:
            fallback.append(token)
    focused_tokens = dedupe_preserving_order(prioritized + fallback)
    if len(focused_tokens) < 2:
        return None
    focused_query = " ".join(focused_tokens[:6])
    if normalize_question_text(focused_query) == normalized:
        return None
    return focused_query


def build_route_context(entry: dict[str, Any]) -> dict[str, Any]:
    route_context: dict[str, Any] = {
        "benchmark": "longmemeval",
        "question_type": entry["question_type"],
    }
    focused_query = build_focused_query(entry["question"])
    if focused_query:
        route_context["focused_query"] = focused_query
    return route_context


def build_services(
    base_dir: Path,
    shared_cache_dir: Path | None = None,
) -> tuple[AppConfig, Storage, VectorStore, MemoryWriter, ActivationEngine, PathRouter, RetrievalReranker]:
    config = force_benchmark_config(AppConfig.from_env(base_dir=base_dir))
    if shared_cache_dir is not None:
        config.shared_cache_dir = shared_cache_dir
        config.cache_dir = shared_cache_dir
        config.ingest_state_path = config.cache_dir / "ingest_state.json"
        config.embedding_cache_path = config.cache_dir / "embedding_cache.sqlite3"
    storage = Storage(config)
    storage.init_db()
    storage.open_persistent()
    vector_store = VectorStore(config, storage=storage)
    writer = MemoryWriter(storage, config)
    activation = ActivationEngine(storage, vector_store)
    router = PathRouter()
    reranker = RetrievalReranker(config)
    return config, storage, vector_store, writer, activation, router, reranker


def runtime_stats_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    def diff_bucket(after_bucket: dict[str, Any], before_bucket: dict[str, Any]) -> dict[str, Any]:
        after_ops = after_bucket.get("ops", {}) if after_bucket else {}
        before_ops = before_bucket.get("ops", {}) if before_bucket else {}
        keys = set(after_ops) | set(before_ops)
        ops: dict[str, Any] = {}
        for key in keys:
            a = after_ops.get(key, {})
            b = before_ops.get(key, {})
            total_ms = round(float(a.get("total_ms", 0.0)) - float(b.get("total_ms", 0.0)), 2)
            calls = int(a.get("calls", 0)) - int(b.get("calls", 0))
            rows = int(a.get("rows", 0)) - int(b.get("rows", 0))
            if calls > 0 or rows > 0 or abs(total_ms) > 0.01:
                ops[key] = {"total_ms": total_ms, "calls": calls, "rows": rows}
        return {
            "total_ms": round(float(after_bucket.get("total_ms", 0.0)) - float(before_bucket.get("total_ms", 0.0)), 2),
            "calls": int(after_bucket.get("calls", 0)) - int(before_bucket.get("calls", 0)),
            "rows": int(after_bucket.get("rows", 0)) - int(before_bucket.get("rows", 0)),
            "ops": ops,
        }

    def diff_numeric_map(after_map: dict[str, Any], before_map: dict[str, Any]) -> dict[str, float]:
        keys = set(after_map) | set(before_map)
        delta: dict[str, float] = {}
        for key in keys:
            value = float(after_map.get(key, 0.0)) - float(before_map.get(key, 0.0))
            if abs(value) > 0.0001:
                delta[key] = round(value, 2)
        return delta

    return {
        "storage": diff_bucket(after.get("storage", {}), before.get("storage", {})),
        "vector": diff_bucket(after.get("vector", {}), before.get("vector", {})),
        "vector_counters": diff_numeric_map(after.get("vector", {}).get("counters", {}), before.get("vector", {}).get("counters", {})),
        "embedding_cache": diff_numeric_map(after.get("vector", {}).get("embedding_cache", {}), before.get("vector", {}).get("embedding_cache", {})),
    }


def flatten_numeric_metrics(prefix: str, value: Any, bucket: dict[str, list[float]]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        bucket[prefix].append(float(value))
        return
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_numeric_metrics(next_prefix, child, bucket)


def summarize_numeric_metrics(bucket: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: round(sum(values) / len(values), 2)
        for key, values in bucket.items()
        if values
    }


def materialize_ingest_profile_for_question(ingest_profile: dict[str, Any], ingest_reuse: dict[str, Any]) -> dict[str, Any]:
    if not ingest_reuse.get("workspace_reused"):
        return ingest_profile
    counts = dict(ingest_profile.get("counts") or {})
    zero_timings = {key: 0.0 for key in dict(ingest_profile.get("timings_ms") or {})}
    return {
        "counts": counts,
        "timings_ms": zero_timings,
        "dedup": {
            "node_hit_count": int(counts.get("nodes", 0)),
            "node_miss_count": 0,
            "chunk_hit_count": int(counts.get("chunks", 0)),
            "chunk_miss_count": 0,
            "object_hit_count": int(counts.get("objects", 0)),
            "object_miss_count": 0,
        },
        "backend": {
            "storage": {"total_ms": 0.0, "calls": 0, "rows": 0, "ops": {}},
            "vector": {"total_ms": 0.0, "calls": 0, "rows": 0, "ops": {}},
            "vector_counters": {
                "vector_dedup_hit_count": 0.0,
                "vector_dedup_miss_count": 0.0,
            },
            "embedding_cache": {
                "embedding_cache_hit_count": 0.0,
                "embedding_cache_miss_count": 0.0,
                "embedding_cache_hit_ms_saved": 0.0,
                "actual_embedding_compute_ms": 0.0,
            },
        },
    }


def summarize_question_profile(
    question_profile: dict[str, Any],
    retrieval_ms: float,
    completion_ms: float,
    question_total_ms: float,
    ingest_ms: float,
    ingest_reuse: dict[str, Any],
) -> dict[str, Any]:
    backend_sections = []
    ingest_backend = dict(question_profile.get("ingest", {}).get("backend") or {})
    if ingest_backend:
        backend_sections.append(ingest_backend)
    pipeline_profile = dict(question_profile.get("pipeline") or {})
    for key in ("retrieval", "completion", "cognitive"):
        backend = dict(pipeline_profile.get(key, {}).get("backend") or {})
        if backend:
            backend_sections.append(backend)

    storage_ms = round(sum(float(section.get("storage", {}).get("total_ms", 0.0)) for section in backend_sections), 2)
    vector_ms = round(sum(float(section.get("vector", {}).get("total_ms", 0.0)) for section in backend_sections), 2)
    embedding_cache_hit_count = int(sum(float(section.get("embedding_cache", {}).get("embedding_cache_hit_count", 0.0)) for section in backend_sections))
    embedding_cache_miss_count = int(sum(float(section.get("embedding_cache", {}).get("embedding_cache_miss_count", 0.0)) for section in backend_sections))
    embedding_cache_hit_ms_saved = round(sum(float(section.get("embedding_cache", {}).get("embedding_cache_hit_ms_saved", 0.0)) for section in backend_sections), 2)
    actual_embedding_compute_ms = round(sum(float(section.get("embedding_cache", {}).get("actual_embedding_compute_ms", 0.0)) for section in backend_sections), 2)
    vector_dedup_hit_count = int(sum(float(section.get("vector_counters", {}).get("vector_dedup_hit_count", 0.0)) for section in backend_sections))
    vector_dedup_miss_count = int(sum(float(section.get("vector_counters", {}).get("vector_dedup_miss_count", 0.0)) for section in backend_sections))

    ingest_counts = dict(question_profile.get("ingest", {}).get("counts") or {})
    ingest_dedup = dict(question_profile.get("ingest", {}).get("dedup") or {})
    retrieval_counts = dict(pipeline_profile.get("retrieval", {}).get("candidate_counts") or {})
    completion_counts = dict(pipeline_profile.get("completion", {}).get("candidate_counts") or {})
    retrieval_ranking = dict(pipeline_profile.get("retrieval", {}).get("ranking") or {})
    retrieval_selection = dict(pipeline_profile.get("retrieval", {}).get("selection") or {})
    retrieval_fallback = dict(pipeline_profile.get("retrieval", {}).get("storage_fallback") or {})

    return {
        "question_total_ms": round(question_total_ms, 2),
        "ingest_ms": round(ingest_ms, 2),
        "retrieval_ms": round(retrieval_ms, 2),
        "completion_ms": round(completion_ms, 2),
        "total_storage_ms": storage_ms,
        "total_vector_ms": vector_ms,
        "embedding_cache_hit_count": embedding_cache_hit_count,
        "embedding_cache_miss_count": embedding_cache_miss_count,
        "embedding_cache_hit_ms_saved": embedding_cache_hit_ms_saved,
        "actual_embedding_compute_ms": actual_embedding_compute_ms,
        "vector_dedup_hit_count": vector_dedup_hit_count,
        "vector_dedup_miss_count": vector_dedup_miss_count,
        "dedup_hit_count": int(sum(float(value) for key, value in ingest_dedup.items() if key.endswith("hit_count"))),
        "dedup_miss_count": int(sum(float(value) for key, value in ingest_dedup.items() if key.endswith("miss_count"))),
        "cache_reuse_saved_ms": round(float(ingest_reuse.get("cache_reuse_saved_ms", 0.0)), 2),
        "ingest_lookup_ms": round(float(ingest_reuse.get("ingest_lookup_ms", 0.0)), 2),
        "workspace_reused": bool(ingest_reuse.get("workspace_reused", False)),
        "workspace_reused_from_disk": bool(ingest_reuse.get("workspace_reused_from_disk", False)),
        "object_lookup_ms": round(float(question_profile.get("pipeline", {}).get("completion", {}).get("timings_ms", {}).get("object_lookup_ms", 0.0)), 2),
        "object_support_join_ms": round(float(question_profile.get("pipeline", {}).get("completion", {}).get("timings_ms", {}).get("object_support_join_ms", 0.0)), 2),
        "extracted_preference_object_count": int(ingest_counts.get("extracted_preference_object_count", 0)),
        "extracted_temporal_object_count": int(ingest_counts.get("extracted_temporal_object_count", 0)),
        "extracted_state_update_object_count": int(ingest_counts.get("extracted_state_update_object_count", 0)),
        "candidate_counts": {
            "dense": int(retrieval_counts.get("dense_chunks", 0)),
            "sparse": int(retrieval_counts.get("sparse_chunks", 0)),
            "object": int(retrieval_counts.get("dense_objects", 0)) + int(retrieval_counts.get("sparse_objects", 0)),
            "merged": int(retrieval_counts.get("merged_chunks", 0)),
            "rerank_pool_requested": int(retrieval_counts.get("rerank_pool_requested", 0)),
            "rerank_pool": int(retrieval_counts.get("rerank_pool", 0)),
            "final": int(retrieval_counts.get("final_evidence", 0)),
            "supporting_context": int(completion_counts.get("supporting_context", 0)),
            "evidence_objects": int(completion_counts.get("evidence_objects", 0)),
        },
        "retrieval_debug": {
            "temporal_prior_positive_count": round(float(retrieval_ranking.get("temporal_prior_positive_count", 0.0)), 2),
            "top_margin_1_2": round(float(retrieval_ranking.get("top_margin_1_2", 0.0)), 4),
            "storage_node_fallback_candidates": round(float(retrieval_fallback.get("node_fallback_candidates", 0.0)), 2),
            "storage_neighbor_fallback_candidates": round(float(retrieval_fallback.get("neighbor_fallback_candidates", 0.0)), 2),
            "diversity_overflow": round(float(retrieval_selection.get("diversity_overflow", 0.0)), 2),
        },
    }


def classify_bottleneck(stage_name: str) -> str:
    lowered = stage_name.lower()
    if any(token in lowered for token in ("service_init", "json_write", "load_data", "ingest", "insert_", "upsert_")):
        return "implementation/setup"
    if any(token in lowered for token in ("sqlite", "storage.", "vector.", "candidate_merge", "fetch_", "hydrate_")):
        return "implementation/runtime"
    if any(token in lowered for token in ("cross_encoder", "dense_vector", "dense_object", "sparse_", "completion", "supporting_context", "rank_evidence")):
        return "algorithm/runtime"
    return "mixed"


def ingest_corpus(
    corpus_items: list[dict[str, Any]],
    storage: Storage,
    vector_store: VectorStore,
    writer: MemoryWriter,
    shell: int,
    sector: str,
    zone: str,
    question_type: str,
    benchmark_name: str,
    adapter_version: str,
    runtime_fingerprint: dict[str, Any],
    raw_counts: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    ordered_corpus_ids: list[str] = []
    corpus_by_node_id: dict[str, dict[str, Any]] = {}
    nodes: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    neighbors: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    chunk_prepare_ms = 0.0
    neighbor_build_ms = 0.0
    object_extract_ms = 0.0

    for index, item in enumerate(corpus_items):
        theta, phi = deterministic_angle(item["corpus_id"])
        node = MemoryNode(
            id=f"bench_{index:04d}",
            shell=shell,
            sector=sector,
            zone=zone,
            cell=item["corpus_id"],
            molecular_type="dialog_session",
            summary=make_summary(item["text"]),
            content_hash=stable_content_hash(item["text"]),
            content_ref=item["corpus_id"],
            source_ref=str(item.get("source_segment_id") or item["corpus_id"]),
            raw_content=item["text"],
            theta=theta,
            phi=phi,
            importance=0.45,
            creative_score=0.1,
            stability_score=0.5,
            stage="long_term",
            tags=f"benchmark longmemeval {question_type}",
            created_at=item["timestamp"],
            last_accessed_at=item["timestamp"],
        )
        nodes.append(node.to_dict())
        stage_start = perf_counter()
        node_chunks = writer.prepare_chunks(node, source_kind="text", source_path=item["corpus_id"])
        chunk_prepare_ms += (perf_counter() - stage_start) * 1000.0
        stage_start = perf_counter()
        node_neighbors = writer.build_chunk_neighbors(node_chunks)
        neighbor_build_ms += (perf_counter() - stage_start) * 1000.0
        stage_start = perf_counter()
        node_objects = writer.extract_objects(node, node_chunks)
        object_extract_ms += (perf_counter() - stage_start) * 1000.0
        annotate_chunks_for_benchmark(
            node_chunks,
            corpus_item=item,
            benchmark_name=benchmark_name,
            adapter_version=adapter_version,
        )
        chunks.extend(node_chunks)
        neighbors.extend(node_neighbors)
        objects.extend(node_objects)

        ordered_corpus_ids.append(item["corpus_id"])
        corpus_by_node_id[node.id] = item

    runtime_before = {
        "storage": storage.snapshot_stats(reset=False),
        "vector": vector_store.snapshot_stats(reset=False),
    }
    storage.insert_nodes(nodes)
    storage.insert_chunks(chunks)
    storage.insert_chunk_neighbors(neighbors)
    storage.insert_objects(objects)
    vector_store.upsert_chunks(chunks)
    vector_store.upsert_objects(objects)
    runtime_after = {
        "storage": storage.snapshot_stats(reset=False),
        "vector": vector_store.snapshot_stats(reset=False),
    }
    grain_distribution: dict[str, int] = {}
    object_type_distribution: dict[str, int] = {}
    for chunk in chunks:
        grain = str(chunk.get("grain") or "unknown")
        grain_distribution[grain] = grain_distribution.get(grain, 0) + 1
    for obj in objects:
        object_type = str(obj.get("object_type") or "unknown")
        object_type_distribution[object_type] = object_type_distribution.get(object_type, 0) + 1
    node_hashes = [str(node.get("content_hash") or "") for node in nodes if node.get("content_hash")]
    chunk_hashes = [str(chunk.get("content_hash") or "") for chunk in chunks if chunk.get("content_hash")]
    object_hashes = [str(obj.get("content_hash") or "") for obj in objects if obj.get("content_hash")]
    ingest_profile = {
        "counts": {
            "nodes": len(nodes),
            "chunks": len(chunks),
            "neighbors": len(neighbors),
            "objects": len(objects),
            "grain_distribution": grain_distribution,
            "object_type_distribution": object_type_distribution,
            "extracted_preference_object_count": object_type_distribution.get("preference", 0),
            "extracted_temporal_object_count": object_type_distribution.get("temporal_reference", 0),
            "extracted_state_update_object_count": object_type_distribution.get("state_update", 0),
        },
        "timings_ms": {
            "prepare_chunks_ms": round(chunk_prepare_ms, 2),
            "build_neighbors_ms": round(neighbor_build_ms, 2),
            "extract_objects_ms": round(object_extract_ms, 2),
        },
        "dedup": {
            "node_hit_count": len(node_hashes) - len(set(node_hashes)),
            "node_miss_count": len(set(node_hashes)),
            "chunk_hit_count": len(chunk_hashes) - len(set(chunk_hashes)),
            "chunk_miss_count": len(set(chunk_hashes)),
            "object_hit_count": len(object_hashes) - len(set(object_hashes)),
            "object_miss_count": len(set(object_hashes)),
        },
        "backend": runtime_stats_delta(runtime_after, runtime_before),
    }
    index_metadata = build_index_metadata(
        corpus_items=corpus_items,
        chunks=chunks,
        benchmark_name=benchmark_name,
        adapter_version=adapter_version,
        runtime_fingerprint=runtime_fingerprint,
        raw_counts=raw_counts or {},
    )

    return ordered_corpus_ids, corpus_by_node_id, ingest_profile, index_metadata


def build_workspace_signature(
    corpus_items: list[dict[str, Any]],
    granularity: str,
    shell: int,
    sector: str,
    zone: str,
    benchmark_name: str,
    adapter_version: str,
    runtime_fingerprint: dict[str, Any],
    config: AppConfig | None = None,
) -> str:
    payload = {
        "cache_version": BENCHMARK_WORKSPACE_CACHE_VERSION,
        "benchmark_name": benchmark_name,
        "benchmark_adapter_version": adapter_version,
        "granularity": granularity,
        "shell": shell,
        "sector": sector,
        "zone": zone,
        "chunker_version": BENCHMARK_CHUNKER_VERSION,
        "chunk_size": int(config.chunk_size if config else 0),
        "chunk_overlap": int(config.chunk_overlap if config else 0),
        "local_window_span": int(config.local_window_span if config else 0),
        "embed_local_grain": bool(config.embed_local_grain if config else False),
        "embedding_model": str(config.embedding_model_name if config else ""),
        "runtime_fingerprint_hash": str(runtime_fingerprint.get("fingerprint_hash") or ""),
        "creative_mode": str(config.creative_mode_name if config else "off"),
        "creative_beam_width": int(config.creative_beam_width if config else 0),
        "creative_max_hops": int(config.creative_max_hops if config else 0),
        "creative_neighbors_per_hop": int(config.creative_neighbors_per_hop if config else 0),
        "creative_max_output_paths": int(config.creative_max_output_paths if config else 0),
        "items": [
            {
                "corpus_id": str(item["corpus_id"]),
                "timestamp": str(item["timestamp"]),
                "text_hash": stable_content_hash(str(item["text"])),
            }
            for item in corpus_items
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class BenchmarkWorkspaceManager:
    def __init__(self, cache_root: Path, shared_cache_dir: Path) -> None:
        self.cache_root = cache_root
        self.shared_cache_dir = shared_cache_dir
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.shared_cache_dir.mkdir(parents=True, exist_ok=True)
        self.keep_workspace_open = os.environ.get("SPHERE_BENCHMARK_KEEP_WORKSPACE_OPEN", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        try:
            self.max_open_workspaces = max(1, int(os.environ.get("SPHERE_BENCHMARK_MAX_OPEN_WORKSPACES", "1")))
        except ValueError:
            self.max_open_workspaces = 1
        self._workspaces: OrderedDict[str, BenchmarkWorkspace] = OrderedDict()

    def _reuse_diagnostics(
        self,
        *,
        workspace: BenchmarkWorkspace,
        reused_from_disk: bool,
        reused_in_memory: bool,
        ingest_lookup_ms: float,
    ) -> dict[str, Any]:
        return {
            "workspace_reused": True,
            "workspace_reused_from_disk": bool(reused_from_disk),
            "workspace_reused_in_memory": bool(reused_in_memory),
            "workspace_retention_enabled": bool(self.keep_workspace_open),
            "workspace_retention_max_open": int(self.max_open_workspaces),
            "cache_reuse_saved_ms": round(workspace.build_elapsed_ms, 2),
            "ingest_lookup_ms": round(ingest_lookup_ms, 2),
        }

    def _evict_surplus(self, keep_signature: str | None = None, collect_garbage: bool = True) -> None:
        if not self.keep_workspace_open:
            return
        evicted = False
        while len(self._workspaces) > self.max_open_workspaces:
            signature, workspace = self._workspaces.popitem(last=False)
            if keep_signature is not None and signature == keep_signature:
                self._workspaces[signature] = workspace
                break
            workspace.close()
            evicted = True
        if evicted and collect_garbage:
            gc.collect()

    def _acquire_build_lock(self, signature: str) -> Any:
        lock_path = self.cache_root / f"{signature}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    @staticmethod
    def _release_build_lock(handle: Any) -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def acquire(
        self,
        corpus_items: list[dict[str, Any]],
        granularity: str,
        question_type: str,
        shell: int,
        sector: str,
        zone: str,
        benchmark_name: str,
        adapter_version: str,
        raw_counts: dict[str, Any] | None = None,
    ) -> tuple[BenchmarkWorkspace, dict[str, Any]]:
        probe_config = force_benchmark_config(
            AppConfig(base_dir=self.cache_root / "_probe", shared_cache_dir=self.shared_cache_dir)
        )
        probe_runtime_fingerprint = build_runtime_fingerprint(
            config=probe_config,
            benchmark_name=benchmark_name,
            adapter_version=adapter_version,
            granularity=granularity,
        )
        signature = build_workspace_signature(
            corpus_items=corpus_items,
            granularity=granularity,
            shell=shell,
            sector=sector,
            zone=zone,
            benchmark_name=benchmark_name,
            adapter_version=adapter_version,
            runtime_fingerprint=probe_runtime_fingerprint,
            config=probe_config,
        )
        if signature in self._workspaces:
            workspace = self._workspaces[signature]
            self._workspaces.move_to_end(signature)
            return workspace, self._reuse_diagnostics(
                workspace=workspace,
                reused_from_disk=False,
                reused_in_memory=True,
                ingest_lookup_ms=0.0,
            )

        build_lock = self._acquire_build_lock(signature)
        try:
            if signature in self._workspaces:
                workspace = self._workspaces[signature]
                self._workspaces.move_to_end(signature)
                return workspace, self._reuse_diagnostics(
                    workspace=workspace,
                    reused_from_disk=False,
                    reused_in_memory=True,
                    ingest_lookup_ms=0.0,
                )

            workspace_dir = self.cache_root / signature
            manifest_path = workspace_dir / "workspace_manifest.json"
            manifest: dict[str, Any] | None = None
            lookup_start = perf_counter()
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    manifest = None

            manifest_valid = bool(
                manifest
                and manifest.get("signature") == signature
                and manifest.get("cache_version") == BENCHMARK_WORKSPACE_CACHE_VERSION
            )

            if not manifest_valid and workspace_dir.exists():
                shutil.rmtree(workspace_dir, ignore_errors=True)

            config, storage, vector_store, writer, activation, router, reranker = build_services(
                workspace_dir,
                shared_cache_dir=self.shared_cache_dir,
            )
            runtime_fingerprint = build_runtime_fingerprint(
                config=config,
                benchmark_name=benchmark_name,
                adapter_version=adapter_version,
                granularity=granularity,
                vector_info=vector_store.info(),
            )
            pipeline = EvidencePipeline(storage, vector_store, activation, router, config=config)
            manifest_fingerprint = dict((manifest or {}).get("index_metadata", {}).get("fingerprint") or {})
            if manifest_valid and manifest_fingerprint:
                stable_fingerprint_match = all(
                    manifest_fingerprint.get(key) == runtime_fingerprint.get(key)
                    for key in (
                        "embedding_provider",
                        "embedding_model",
                        "embedding_dim",
                        "normalize_embeddings",
                        "embedding_preprocess_version",
                        "fallback_in_use",
                        "vector_backend",
                        "vector_fallback_in_use",
                        "benchmark_adapter_version",
                    )
                ) and json.dumps(
                    manifest_fingerprint.get("chunker") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ) == json.dumps(
                    runtime_fingerprint.get("chunker") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if not stable_fingerprint_match:
                    manifest_valid = False
                    vector_store.close()
                    storage.close_persistent()
                    shutil.rmtree(workspace_dir, ignore_errors=True)
                    config, storage, vector_store, writer, activation, router, reranker = build_services(
                        workspace_dir,
                        shared_cache_dir=self.shared_cache_dir,
                    )
                    runtime_fingerprint = build_runtime_fingerprint(
                        config=config,
                        benchmark_name=benchmark_name,
                        adapter_version=adapter_version,
                        granularity=granularity,
                        vector_info=vector_store.info(),
                    )
                    pipeline = EvidencePipeline(storage, vector_store, activation, router, config=config)
            if manifest_valid and manifest is not None:
                expected_chunk_count = int((manifest.get("ingest_profile") or {}).get("counts", {}).get("chunks", 0))
                if expected_chunk_count > 0 and vector_store.count() == 0:
                    manifest_valid = False
                    vector_store.close()
                    storage.close_persistent()
                    shutil.rmtree(workspace_dir, ignore_errors=True)
                    config, storage, vector_store, writer, activation, router, reranker = build_services(
                        workspace_dir,
                        shared_cache_dir=self.shared_cache_dir,
                    )
                    runtime_fingerprint = build_runtime_fingerprint(
                        config=config,
                        benchmark_name=benchmark_name,
                        adapter_version=adapter_version,
                        granularity=granularity,
                        vector_info=vector_store.info(),
                    )
                    pipeline = EvidencePipeline(storage, vector_store, activation, router, config=config)

            if manifest_valid and manifest is not None:
                loaded_index_metadata = dict(manifest.get("index_metadata") or {})
                loaded_index_metadata["workspace_dir"] = str(workspace_dir)
                assert_benchmark_vector_guard(
                    vector_info=vector_store.info(),
                    runtime_fingerprint=runtime_fingerprint,
                    index_fingerprint=dict(loaded_index_metadata.get("fingerprint") or {}),
                )
                workspace = BenchmarkWorkspace(
                    signature=signature,
                    base_dir=workspace_dir,
                    config=config,
                    storage=storage,
                    vector_store=vector_store,
                    writer=writer,
                    activation=activation,
                    router=router,
                    reranker=reranker,
                    pipeline=pipeline,
                    vector_info=vector_store.info(),
                    ingest_profile=dict(manifest.get("ingest_profile") or {}),
                    index_metadata=loaded_index_metadata,
                    build_elapsed_ms=float(manifest.get("build_elapsed_ms") or 0.0),
                    manifest_path=manifest_path,
                )
                self._workspaces[signature] = workspace
                self._evict_surplus(keep_signature=signature)
                return workspace, self._reuse_diagnostics(
                    workspace=workspace,
                    reused_from_disk=True,
                    reused_in_memory=False,
                    ingest_lookup_ms=(perf_counter() - lookup_start) * 1000.0,
                )

            build_start = perf_counter()
            _, _, ingest_profile, index_metadata = ingest_corpus(
                corpus_items,
                storage=storage,
                vector_store=vector_store,
                writer=writer,
                shell=shell,
                sector=sector,
                zone=zone,
                question_type=question_type,
                benchmark_name=benchmark_name,
                adapter_version=adapter_version,
                runtime_fingerprint=runtime_fingerprint,
                raw_counts=raw_counts,
            )
            index_metadata["workspace_dir"] = str(workspace_dir)
            assert_benchmark_vector_guard(
                vector_info=vector_store.info(),
                runtime_fingerprint=runtime_fingerprint,
                index_fingerprint=dict(index_metadata.get("fingerprint") or {}),
            )
            build_elapsed_ms = round((perf_counter() - build_start) * 1000.0, 2)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_payload = {
                "signature": signature,
                "cache_version": BENCHMARK_WORKSPACE_CACHE_VERSION,
                "build_elapsed_ms": build_elapsed_ms,
                "ingest_profile": ingest_profile,
                "index_metadata": index_metadata,
            }
            tmp_manifest_path = manifest_path.with_suffix(".json.tmp")
            tmp_manifest_path.write_text(
                json.dumps(manifest_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_manifest_path.replace(manifest_path)
            workspace = BenchmarkWorkspace(
                signature=signature,
                base_dir=workspace_dir,
                config=config,
                storage=storage,
                vector_store=vector_store,
                writer=writer,
                activation=activation,
                router=router,
                reranker=reranker,
                pipeline=pipeline,
                vector_info=vector_store.info(),
                ingest_profile=ingest_profile,
                index_metadata=index_metadata,
                build_elapsed_ms=build_elapsed_ms,
                manifest_path=manifest_path,
            )
            self._workspaces[signature] = workspace
            self._evict_surplus(keep_signature=signature)
            return workspace, {
                "workspace_reused": False,
                "workspace_reused_from_disk": False,
                "workspace_reused_in_memory": False,
                "workspace_retention_enabled": bool(self.keep_workspace_open),
                "workspace_retention_max_open": int(self.max_open_workspaces),
                "cache_reuse_saved_ms": 0.0,
                "ingest_lookup_ms": round((perf_counter() - lookup_start) * 1000.0, 2),
            }
        finally:
            self._release_build_lock(build_lock)

    def close_all(self) -> None:
        for workspace in self._workspaces.values():
            workspace.close()
        self._workspaces.clear()

    def release(self, signature: str, collect_garbage: bool = True, force_close: bool = False) -> None:
        if self.keep_workspace_open and not force_close:
            workspace = self._workspaces.get(signature)
            if workspace is not None:
                self._workspaces.move_to_end(signature)
                self._evict_surplus(keep_signature=signature, collect_garbage=collect_garbage)
            return
        workspace = self._workspaces.pop(signature, None)
        if workspace is None:
            return
        workspace.close()
        del workspace
        if collect_garbage:
            gc.collect()


def rank_vector(
    query: str,
    vector_store: VectorStore,
    ordered_corpus_ids: list[str],
    corpus_by_node_id: dict[str, dict[str, Any]],
    top_k: int,
    chunk_pool: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, float]]:
    total_start = perf_counter()
    raw_hits = vector_store.search(query, top_k=chunk_pool)
    best_by_corpus_id: dict[str, dict[str, Any]] = {}
    for hit in raw_hits:
        metadata = hit.get("metadata") or {}
        node_id = str(metadata.get("node_id") or "")
        if node_id not in corpus_by_node_id:
            continue
        corpus_item = corpus_by_node_id[node_id]
        corpus_id = corpus_item["corpus_id"]
        current = best_by_corpus_id.get(corpus_id)
        if current is None or hit["similarity"] > current["score"]:
            best_by_corpus_id[corpus_id] = {
                "corpus_id": corpus_id,
                "score": float(hit["similarity"]),
                "text": corpus_item["text"],
                "timestamp": corpus_item["timestamp"],
            }

    ranked = sorted(best_by_corpus_id.values(), key=lambda item: item["score"], reverse=True)
    ranked_ids = [item["corpus_id"] for item in ranked]
    for corpus_id in ordered_corpus_ids:
        if corpus_id not in best_by_corpus_id:
            ranked_ids.append(corpus_id)
    return (
        ranked_ids[: max(top_k, len(ordered_corpus_ids))],
        ranked[:top_k],
        {"vector_total_ms": round((perf_counter() - total_start) * 1000.0, 2)},
    )


def rank_bm25(
    query: str,
    storage: Storage,
    ordered_corpus_ids: list[str],
    corpus_by_node_id: dict[str, dict[str, Any]],
    top_k: int,
    chunk_pool: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, float]]:
    total_start = perf_counter()
    raw_hits = storage.search_chunks_fts(query, limit=max(top_k, chunk_pool))
    best_by_corpus_id: dict[str, dict[str, Any]] = {}
    for hit in raw_hits:
        node_id = str(hit.get("node_id") or "")
        corpus_item = corpus_by_node_id.get(node_id)
        if not corpus_item:
            continue
        corpus_id = str(corpus_item["corpus_id"])
        raw_score = float(hit.get("bm25_score") or 0.0)
        current = best_by_corpus_id.get(corpus_id)
        if current is None or raw_score < float(current["raw_bm25_score"]):
            best_by_corpus_id[corpus_id] = {
                "corpus_id": corpus_id,
                "raw_bm25_score": raw_score,
                "score": -raw_score,
                "text": corpus_item["text"],
                "timestamp": corpus_item["timestamp"],
            }

    ranked = sorted(best_by_corpus_id.values(), key=lambda item: (float(item["raw_bm25_score"]), item["corpus_id"]))
    ranked_ids = [item["corpus_id"] for item in ranked]
    for corpus_id in ordered_corpus_ids:
        if corpus_id not in best_by_corpus_id:
            ranked_ids.append(corpus_id)
    return (
        ranked_ids[: max(top_k, len(ordered_corpus_ids))],
        ranked[:top_k],
        {
            "bm25_total_ms": round((perf_counter() - total_start) * 1000.0, 2),
            "lexical_ms": round((perf_counter() - total_start) * 1000.0, 2),
        },
    )


def rank_hybrid(
    query: str,
    activation: ActivationEngine,
    router: PathRouter,
    reranker: RetrievalReranker,
    ordered_corpus_ids: list[str],
    corpus_by_node_id: dict[str, dict[str, Any]],
    task_type: str,
    top_k: int,
    rerank_mode: str,
    apply_rerank: bool,
) -> tuple[list[str], list[dict[str, Any]], dict[str, float]]:
    total_start = perf_counter()
    route = router.resolve(task_type)
    candidate_count = max(top_k * 3, 24)
    stage_start = perf_counter()
    candidates = activation.main_activation(
        query,
        route.preferred_shells,
        route.preferred_sectors,
        top_k=candidate_count,
    )
    activation_ms = round((perf_counter() - stage_start) * 1000.0, 2)
    stage_start = perf_counter()
    ranked_nodes = (
        reranker.rerank(query, candidates, top_k=top_k, mode=rerank_mode)
        if apply_rerank
        else candidates[:top_k]
    )
    rerank_ms = round((perf_counter() - stage_start) * 1000.0, 2)

    ranked_ids: list[str] = []
    ranked_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in ranked_nodes:
        corpus_item = corpus_by_node_id.get(node["id"])
        if not corpus_item:
            continue
        corpus_id = corpus_item["corpus_id"]
        if corpus_id in seen:
            continue
        score = float(node.get("rerank_score", node.get("fusion_score", node.get("vector_score", 0.0))))
        ranked_ids.append(corpus_id)
        ranked_items.append(
            {
                "corpus_id": corpus_id,
                "score": score,
                "text": corpus_item["text"],
                "timestamp": corpus_item["timestamp"],
            }
        )
        seen.add(corpus_id)

    for corpus_id in ordered_corpus_ids:
        if corpus_id not in seen:
            ranked_ids.append(corpus_id)

    return (
        ranked_ids[: max(top_k, len(ordered_corpus_ids))],
        ranked_items,
        {
            "activation_ms": activation_ms,
            "rerank_ms": rerank_ms if apply_rerank else 0.0,
            "hybrid_total_ms": round((perf_counter() - total_start) * 1000.0, 2),
        },
    )


def rank_evidence(
    query: str,
    pipeline: EvidencePipeline,
    ordered_corpus_ids: list[str],
    corpus_by_node_id: dict[str, dict[str, Any]],
    task_type: str,
    top_k: int,
    object_top_k: int = 4,
    support_top_k: int = 4,
    cognitive_top_k: int = 0,
    route_context: dict[str, Any] | None = None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    result = pipeline.retrieve_evidence(
        query,
        task_type=task_type,
        evidence_top_k=top_k,
        route_context=route_context,
    )
    completion = None
    cognitive = None
    if object_top_k > 0 or support_top_k > 0:
        completion = pipeline.complete_with_objects(
            query=query,
            evidence=result,
            support_top_k=support_top_k,
            object_top_k=object_top_k,
        )
    if completion is not None and cognitive_top_k > 0:
        cognitive = pipeline.augment_cognitively(
            query=query,
            task_type=task_type,
            completion=completion,
            cognitive_top_k=cognitive_top_k,
        )
    ranked_ids: list[str] = []
    ranked_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in result.candidates:
        corpus_item = corpus_by_node_id.get(str(chunk.get("node_id") or ""))
        if not corpus_item:
            continue
        corpus_id = corpus_item["corpus_id"]
        if corpus_id in seen:
            continue
        ranked_ids.append(corpus_id)
        ranked_items.append(
            {
                "corpus_id": corpus_id,
                "score": float(chunk.get("evidence_score", 0.0)),
                "text": corpus_item["text"],
                "timestamp": corpus_item["timestamp"],
            }
        )
        seen.add(corpus_id)
    for corpus_id in ordered_corpus_ids:
        if corpus_id not in seen:
            ranked_ids.append(corpus_id)
    stage_timings = dict(result.timings_ms)
    diagnostics = {"retrieval": {**result.diagnostics, "timings_ms": dict(result.timings_ms)}}
    if completion is not None:
        for key, value in completion.timings_ms.items():
            stage_timings[f"completion_{key}"] = value
        diagnostics["completion"] = {**completion.diagnostics, "timings_ms": dict(completion.timings_ms)}
    if cognitive is not None:
        for key, value in cognitive.timings_ms.items():
            stage_timings[f"cognitive_{key}"] = value
        diagnostics["cognitive"] = {**cognitive.diagnostics, "timings_ms": dict(cognitive.timings_ms)}
    else:
        diagnostics["cognitive"] = {
            "candidate_counts": {
                "evidence_nodes": len(completion.evidence_nodes if completion is not None else result.evidence_nodes),
                "relevant_experience": 0,
                "creative_reflections": 0,
                "alternative_paths": 0,
            },
            "decisions": {
                "executed": False,
                "reason": "cognitive_top_k_zero" if cognitive_top_k <= 0 else "completion_not_available",
            },
            "prism": {},
            "timings_ms": {"total_ms": 0.0, "prism_total_ms": 0.0},
        }
    return ranked_ids[: max(top_k, len(ordered_corpus_ids))], ranked_items, stage_timings, diagnostics


def run_benchmark(
    data_file: Path,
    mode: str,
    granularity: str,
    top_k: int,
    limit: int,
    rerank_mode: str,
    task_type: str,
    shell: int,
    sector: str,
    zone: str,
    chunk_pool: int,
    out_file: Path | None,
    use_cross_encoder: bool = False,
    object_top_k: int = 4,
    support_top_k: int = 4,
    cognitive_top_k: int = 0,
) -> dict[str, Any]:
    determinism = configure_benchmark_determinism()
    load_start = perf_counter()
    with data_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    load_data_ms = round((perf_counter() - load_start) * 1000.0, 2)

    if limit > 0:
        data = data[:limit]

    metrics_session = {f"recall_any@{k}": [] for k in KS}
    metrics_session.update({f"recall_all@{k}": [] for k in KS})
    metrics_session.update({f"ndcg_any@{k}": [] for k in KS})

    metrics_turn = {f"recall_any@{k}": [] for k in KS}
    metrics_turn.update({f"recall_all@{k}": [] for k in KS})
    metrics_turn.update({f"ndcg_any@{k}": [] for k in KS})

    per_type = defaultdict(lambda: defaultdict(list))
    timing_metrics = defaultdict(list)
    profile_metrics: dict[str, list[float]] = defaultdict(list)
    results_log: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    topk_debug_rows: list[dict[str, Any]] = []
    vector_info: dict[str, Any] | None = None
    index_metadata: dict[str, Any] | None = None
    runtime_fingerprint: dict[str, Any] | None = None
    runtime_config: dict[str, Any] | None = None
    start_time = datetime.now()
    totals_ms = {
        "total_ingest_ms": 0.0,
        "total_retrieval_ms": 0.0,
        "total_completion_ms": 0.0,
        "total_storage_ms": 0.0,
        "total_vector_ms": 0.0,
        "total_cache_reuse_saved_ms": 0.0,
        "total_embedding_cache_hit_ms_saved": 0.0,
    }
    workspace_manager = BenchmarkWorkspaceManager(WORKSPACE_CACHE_ROOT, SHARED_EMBEDDING_CACHE_DIR)
    all_gold_segment_ids: set[str] = set()
    all_gold_document_ids: set[str] = set()
    aggregate_raw_counts: dict[str, int] = defaultdict(int)
    aggregate_chunk_metadata: dict[str, dict[str, Any]] = {}
    aggregate_indexed_segment_ids: set[str] = set()
    aggregate_indexed_doc_ids: set[str] = set()

    cross_encoder_model = None
    if use_cross_encoder:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("Cross-encoder reranking enabled")
        except Exception as exc:
            print(f"Cross-encoder not available: {exc}")

    for index, entry in enumerate(data, start=1):
        question_start = perf_counter()
        corpus_items = build_corpus(entry, granularity=granularity)
        if not corpus_items:
            continue

        ordered_corpus_ids = [str(item["corpus_id"]) for item in corpus_items]
        corpus_by_node_id = {f"bench_{item_index:04d}": item for item_index, item in enumerate(corpus_items)}
        answer_session_ids = set(entry["answer_session_ids"])
        raw_counts = build_raw_counts(
            corpus_items,
            question_count=1,
            session_count=len({str(item.get("session_id") or item["corpus_id"]) for item in corpus_items}),
        )
        question_profile: dict[str, Any] = {}
        retry_count = 0
        current_workspace_manager = workspace_manager
        while True:
            stage_start = perf_counter()
            workspace, ingest_reuse = current_workspace_manager.acquire(
                corpus_items=corpus_items,
                granularity=granularity,
                question_type=entry["question_type"],
                shell=shell,
                sector=sector,
                zone=zone,
                benchmark_name="longmemeval",
                adapter_version=LONGMEMEVAL_ADAPTER_VERSION,
                raw_counts=raw_counts,
            )
            acquire_total_ms = round((perf_counter() - stage_start) * 1000.0, 2)
            ingest_profile = workspace.ingest_profile
            if ingest_reuse.get("workspace_reused"):
                ingest_ms = round(float(ingest_reuse.get("ingest_lookup_ms", 0.0)), 2)
                service_init_ms = round(max(0.0, acquire_total_ms - ingest_ms), 2)
            else:
                ingest_ms = round(float(workspace.build_elapsed_ms), 2)
                service_init_ms = round(max(0.0, acquire_total_ms - ingest_ms), 2)
            actual_ingest_profile = materialize_ingest_profile_for_question(workspace.ingest_profile, ingest_reuse)
            if vector_info is None:
                vector_info = workspace.vector_info
            if index_metadata is None:
                index_metadata = workspace.index_metadata
            if runtime_fingerprint is None:
                runtime_fingerprint = build_runtime_fingerprint(
                    config=workspace.config,
                    benchmark_name="longmemeval",
                    adapter_version=LONGMEMEVAL_ADAPTER_VERSION,
                    granularity=granularity,
                    vector_info=workspace.vector_info,
                )
            for key, value in dict((workspace.index_metadata or {}).get("raw_counts") or {}).items():
                if isinstance(value, (int, float)):
                    aggregate_raw_counts[key] += int(value)
            aggregate_indexed_segment_ids.update(set((workspace.index_metadata or {}).get("indexed_segment_ids") or []))
            aggregate_indexed_doc_ids.update(set((workspace.index_metadata or {}).get("indexed_doc_ids") or []))
            for chunk_id, meta in dict((workspace.index_metadata or {}).get("chunk_metadata_by_id") or {}).items():
                aggregate_chunk_metadata[f"{workspace.signature}:{chunk_id}"] = dict(meta)
            if runtime_config is None:
                runtime_config = {
                    "embedding_model": workspace.config.embedding_model_name,
                    "embed_local_grain": bool(workspace.config.embed_local_grain),
                    "rerank_mode_default": workspace.config.rerank_mode_default,
                    "cross_encoder_model": workspace.config.cross_encoder_model_name,
                    "cross_encoder_requested": bool(use_cross_encoder),
                    "cross_encoder_loaded": bool(cross_encoder_model is not None),
                    "creative_mode": workspace.config.creative_mode_name,
                    "enable_benchmark_route_tuning": bool(workspace.config.enable_benchmark_route_tuning),
                    "creative_beam_width": int(workspace.config.creative_beam_width),
                    "creative_max_hops": int(workspace.config.creative_max_hops),
                    "creative_neighbors_per_hop": int(workspace.config.creative_neighbors_per_hop),
                    "creative_max_output_paths": int(workspace.config.creative_max_output_paths),
                }

            workspace.pipeline.cross_encoder = cross_encoder_model
            try:
                if mode == "vector":
                    ranked_ids, ranked_items, stage_timing_ms = rank_vector(
                        entry["question"],
                        vector_store=workspace.vector_store,
                        ordered_corpus_ids=ordered_corpus_ids,
                        corpus_by_node_id=corpus_by_node_id,
                        top_k=top_k,
                        chunk_pool=chunk_pool,
                    )
                    question_profile = {"ingest": actual_ingest_profile, "ingest_cached": workspace.ingest_profile, "reuse": ingest_reuse}
                elif mode == "bm25":
                    ranked_ids, ranked_items, stage_timing_ms = rank_bm25(
                        entry["question"],
                        storage=workspace.storage,
                        ordered_corpus_ids=ordered_corpus_ids,
                        corpus_by_node_id=corpus_by_node_id,
                        top_k=top_k,
                        chunk_pool=chunk_pool,
                    )
                    question_profile = {"ingest": actual_ingest_profile, "ingest_cached": workspace.ingest_profile, "reuse": ingest_reuse}
                elif mode == "evidence":
                    evidence_stage_start = perf_counter()
                    route_context = build_route_context(entry)
                    hybrid_trace = rank_benchmark_sources(
                        query=entry["question"],
                        benchmark_name="longmemeval",
                        vector_store=workspace.vector_store,
                        storage=workspace.storage,
                        index_metadata=workspace.index_metadata,
                        config=workspace.config,
                        route_context=route_context,
                        pool_limit=max(200, top_k * 4),
                    )
                    final_source_rows = candidate_rows(hybrid_trace["final_candidates"], limit=max(200, top_k * 4))
                    ranked_ids = [str(row.get("source_id") or "") for row in final_source_rows]
                    ranked_items = [
                        {
                            "corpus_id": str(row.get("source_id") or ""),
                            "score": float(row.get("post_inhibition_score") or row.get("rerank_score") or 0.0),
                            "text": str(row.get("preview") or ""),
                            "timestamp": str(row.get("timestamp") or ""),
                        }
                        for row in final_source_rows[:top_k]
                    ]
                    stage_timing_ms = {
                        "total_ms": round((perf_counter() - evidence_stage_start) * 1000.0, 2),
                        **dict(hybrid_trace.get("timings_ms") or {}),
                    }
                    evidence_profile = {
                        "retrieval": {
                            "candidate_counts": {
                                **hybrid_trace["candidate_source_stats"],
                                "final_evidence": len(final_source_rows),
                            },
                            "ranking": {
                                "broad_top_candidates": candidate_rows(hybrid_trace["broad_candidates"], limit=100),
                                "reranked_top_candidates": candidate_rows(hybrid_trace["reranked_candidates"], limit=100),
                                "final_top_candidates": final_source_rows[:100],
                            },
                            "selection": {
                                "fusion_audit": dict(hybrid_trace.get("fusion_audit") or {}),
                                "inhibition_audit": dict(hybrid_trace.get("inhibition_audit") or {}),
                            },
                            "query_features": dict(hybrid_trace.get("query_features") or {}),
                            "query_decomposition": dict(hybrid_trace.get("query_decomposition") or {}),
                            "channel_stats": dict(hybrid_trace.get("channel_stats") or {}),
                            "timings_ms": dict(stage_timing_ms),
                        },
                        "completion": {"candidate_counts": {}, "timings_ms": {"total_ms": 0.0}},
                        "cognitive": {
                            "candidate_counts": {
                                "evidence_nodes": len(final_source_rows),
                                "relevant_experience": 0,
                                "creative_reflections": 0,
                                "alternative_paths": 0,
                            },
                            "decisions": {"executed": False, "reason": "benchmark_mode_creative_disabled"},
                            "timings_ms": {"total_ms": 0.0, "prism_total_ms": 0.0},
                        },
                    }
                    question_profile = {
                        "ingest": actual_ingest_profile,
                        "ingest_cached": workspace.ingest_profile,
                        "pipeline": evidence_profile,
                        "reuse": ingest_reuse,
                    }
                elif mode == "activation":
                    ranked_ids, ranked_items, stage_timing_ms = rank_hybrid(
                        entry["question"],
                        activation=workspace.activation,
                        router=workspace.router,
                        reranker=workspace.reranker,
                        ordered_corpus_ids=ordered_corpus_ids,
                        corpus_by_node_id=corpus_by_node_id,
                        task_type=task_type,
                        top_k=top_k,
                        rerank_mode=rerank_mode,
                        apply_rerank=False,
                    )
                    question_profile = {"ingest": actual_ingest_profile, "ingest_cached": workspace.ingest_profile, "reuse": ingest_reuse}
                else:
                    ranked_ids, ranked_items, stage_timing_ms = rank_hybrid(
                        entry["question"],
                        activation=workspace.activation,
                        router=workspace.router,
                        reranker=workspace.reranker,
                        ordered_corpus_ids=ordered_corpus_ids,
                        corpus_by_node_id=corpus_by_node_id,
                        task_type=task_type,
                        top_k=top_k,
                        rerank_mode=rerank_mode,
                        apply_rerank=True,
                    )
                    question_profile = {"ingest": actual_ingest_profile, "ingest_cached": workspace.ingest_profile, "reuse": ingest_reuse}
                if retry_count:
                    question_profile["retry_count"] = retry_count
                break
            except Exception as exc:
                current_workspace_manager.release(workspace.signature, force_close=True)
                lowered = str(exc).lower()
                retryable = "nothing found on disk" in lowered or "creating hnsw segment reader" in lowered
                if not retryable or retry_count >= 2:
                    raise
                sleep(0.25 * (retry_count + 1))
                retry_count += 1
                retry_cache_root = WORKSPACE_CACHE_ROOT / "_question_retries" / f"q{index:04d}_try{retry_count}"
                current_workspace_manager = BenchmarkWorkspaceManager(retry_cache_root, SHARED_EMBEDDING_CACHE_DIR)
                print(f"[retry {retry_count}] rebuilding workspace for {entry['question_id'][:24]} after vector store error")

        question_total_ms = round((perf_counter() - question_start) * 1000.0, 2)
        retrieval_ms = float(stage_timing_ms.get("total_ms", stage_timing_ms.get("vector_total_ms", stage_timing_ms.get("hybrid_total_ms", 0.0))))
        completion_ms = float(stage_timing_ms.get("completion_total_ms", 0.0)) + float(stage_timing_ms.get("cognitive_total_ms", 0.0))
        question_summary = summarize_question_profile(
            question_profile=question_profile,
            retrieval_ms=retrieval_ms,
            completion_ms=completion_ms,
            question_total_ms=question_total_ms,
            ingest_ms=ingest_ms,
            ingest_reuse=ingest_reuse,
        )
        question_profile["summary"] = question_summary
        stage_timing_ms = {
            "service_init_ms": service_init_ms,
            "ingest_ms": ingest_ms,
            "cache_reuse_saved_ms": float(ingest_reuse.get("cache_reuse_saved_ms", 0.0)),
            "retrieval_ms": retrieval_ms,
            "completion_ms": completion_ms,
            **stage_timing_ms,
            "question_total_ms": question_total_ms,
        }
        ingest_profile = question_profile.get("ingest", {})
        for key, value in ingest_profile.get("timings_ms", {}).items():
            stage_timing_ms[f"ingest_{key}"] = float(value)
        ingest_backend = ingest_profile.get("backend", {})
        stage_timing_ms["ingest_storage_ms"] = float(ingest_backend.get("storage", {}).get("total_ms", 0.0))
        stage_timing_ms["ingest_vector_ms"] = float(ingest_backend.get("vector", {}).get("total_ms", 0.0))
        stage_timing_ms["embedding_cache_hit_ms_saved"] = float(question_summary.get("embedding_cache_hit_ms_saved", 0.0))
        for key, value in stage_timing_ms.items():
            timing_metrics[key].append(float(value))
        flatten_numeric_metrics("profile", {key: value for key, value in question_profile.items() if key != "ingest_cached"}, profile_metrics)
        totals_ms["total_ingest_ms"] += float(ingest_ms)
        totals_ms["total_retrieval_ms"] += float(retrieval_ms)
        totals_ms["total_completion_ms"] += float(completion_ms)
        totals_ms["total_storage_ms"] += float(question_summary.get("total_storage_ms", 0.0))
        totals_ms["total_vector_ms"] += float(question_summary.get("total_vector_ms", 0.0))
        totals_ms["total_cache_reuse_saved_ms"] += float(question_summary.get("cache_reuse_saved_ms", 0.0))
        totals_ms["total_embedding_cache_hit_ms_saved"] += float(question_summary.get("embedding_cache_hit_ms_saved", 0.0))

        all_gold_segment_ids.update(answer_session_ids)
        all_gold_document_ids.update(answer_session_ids)
        session_level_ids = [session_id_from_corpus_id(corpus_id) for corpus_id in ranked_ids]
        turn_correct = {corpus_id for corpus_id in ranked_ids if session_id_from_corpus_id(corpus_id) in answer_session_ids}
        broad_rows = []
        reranked_rows = []
        final_rows = []
        if mode == "evidence":
            ranking_payload = dict((question_profile.get("pipeline") or {}).get("retrieval", {}).get("ranking") or {})
            broad_rows = list(ranking_payload.get("broad_top_candidates") or [])
            reranked_rows = list(ranking_payload.get("reranked_top_candidates") or [])
            final_rows = list(ranking_payload.get("final_top_candidates") or [])
            if granularity != "session":
                broad_rows = remap_rank_rows(broad_rows, session_id_from_corpus_id)
                reranked_rows = remap_rank_rows(reranked_rows, session_id_from_corpus_id)
                final_rows = remap_rank_rows(final_rows, session_id_from_corpus_id)

        entry_metrics = {"session": {}, "turn": {}}
        for k in KS:
            session_ra, session_rl, session_nd = evaluate_retrieval(session_level_ids, answer_session_ids, k)
            metrics_session[f"recall_any@{k}"].append(session_ra)
            metrics_session[f"recall_all@{k}"].append(session_rl)
            metrics_session[f"ndcg_any@{k}"].append(session_nd)
            entry_metrics["session"][f"recall_any@{k}"] = session_ra
            entry_metrics["session"][f"ndcg_any@{k}"] = session_nd

            turn_ra, turn_rl, turn_nd = evaluate_retrieval(ranked_ids, turn_correct, k)
            metrics_turn[f"recall_any@{k}"].append(turn_ra)
            metrics_turn[f"recall_all@{k}"].append(turn_rl)
            metrics_turn[f"ndcg_any@{k}"].append(turn_nd)
            entry_metrics["turn"][f"recall_any@{k}"] = turn_ra
            entry_metrics["turn"][f"ndcg_any@{k}"] = turn_nd

        qtype = entry["question_type"]
        per_type[qtype]["recall_any@5"].append(entry_metrics["session"]["recall_any@5"])
        per_type[qtype]["recall_any@10"].append(entry_metrics["session"]["recall_any@10"])
        per_type[qtype]["ndcg_any@10"].append(entry_metrics["session"]["ndcg_any@10"])
        if mode == "evidence":
            query_diag = build_query_diagnostic(
                benchmark_name="longmemeval",
                query_id=entry["question_id"],
                query_text=entry["question"],
                answer_text=entry["answer"],
                gold_segment_ids=answer_session_ids,
                gold_evidence_ids=answer_session_ids,
                broad_rows=broad_rows,
                reranked_rows=reranked_rows,
                final_rows=final_rows,
                trace=hybrid_trace,
                index_metadata=workspace.index_metadata,
            )
            failure_row = build_query_failure(
                benchmark_name="longmemeval",
                query_id=entry["question_id"],
                query_text=entry["question"],
                answer_text=entry["answer"],
                gold_segment_ids=answer_session_ids,
                gold_evidence_ids=answer_session_ids,
                broad_rows=broad_rows,
                reranked_rows=reranked_rows,
                final_rows=final_rows,
                index_metadata=workspace.index_metadata,
                trace=hybrid_trace,
            )
            if failure_row is not None:
                query_diag["failure_type"] = str(failure_row.get("failure_type") or "unknown")
                failure_rows.append(failure_row)
            topk_debug_rows.append(
                build_topk_debug_record(
                    benchmark_name="longmemeval",
                    query_id=entry["question_id"],
                    query_text=entry["question"],
                    answer_text=entry["answer"],
                    gold_segment_ids=answer_session_ids,
                    failure_type=str(query_diag.get("failure_type") or "ok"),
                    broad_rows=broad_rows,
                    reranked_rows=reranked_rows,
                    final_rows=final_rows,
                    trace=hybrid_trace,
                )
            )
            candidate_diagnostics.append(query_diag)
            oracle_report_for_query = build_oracle_retrieval_report(
                benchmark_name="longmemeval",
                oracle_items=[
                    {
                        "query_id": entry["question_id"],
                        "sample_id": entry["question_id"],
                        "question_type": entry["question_type"],
                        "gold_segment_ids": answer_session_ids,
                        "route_context": build_route_context(entry),
                    }
                ],
                vector_store=workspace.vector_store,
                storage=workspace.storage,
                index_metadata=workspace.index_metadata,
                config=workspace.config,
                pool_limit=50,
            )
            oracle_rows.extend(oracle_report_for_query["rows"])

        results_log.append(
            {
                "question_id": entry["question_id"],
                "question_type": qtype,
                "question": entry["question"],
                "answer": entry["answer"],
                "answer_session_ids": sorted(answer_session_ids),
                "metrics": entry_metrics,
                "stage_timing_ms": stage_timing_ms,
                "profiling": question_profile,
                "ranked_items": ranked_items,
                "candidate_recall": candidate_diagnostics[-1] if mode == "evidence" and candidate_diagnostics else None,
            }
        )

        print(
            f"[{index:4}/{len(data)}] {entry['question_id'][:24]:24} "
            f"R@5={entry_metrics['session']['recall_any@5']:.0f} "
            f"R@10={entry_metrics['session']['recall_any@10']:.0f} "
            f"Ingest={ingest_ms:.0f}ms ReuseSaved={float(ingest_reuse.get('cache_reuse_saved_ms', 0.0)):.0f}ms"
        )
        current_workspace_manager.release(workspace.signature)

    workspace_manager.close_all()

    elapsed_seconds = (datetime.now() - start_time).total_seconds()
    summary_metrics = {
        "session": {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in metrics_session.items()
        },
        "turn": {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in metrics_turn.items()
        },
    }
    per_type_summary = {
        qtype: {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in metric_map.items()
        }
        for qtype, metric_map in per_type.items()
    }
    timing_summary = {
        key: round(sum(values) / len(values), 2)
        for key, values in timing_metrics.items()
        if values
    }
    profile_summary = summarize_numeric_metrics(profile_metrics)
    question_total_sum = sum(timing_metrics.get("question_total_ms", []))
    bottlenecks: list[dict[str, Any]] = []
    for key, values in timing_metrics.items():
        if not values or key == "question_total_ms" or key.endswith("_saved_ms"):
            continue
        total_ms = float(sum(values))
        bottlenecks.append(
            {
                "stage": key,
                "avg_ms": round(total_ms / len(values), 2),
                "total_ms": round(total_ms, 2),
                "share_pct": round((total_ms / question_total_sum) * 100.0, 2) if question_total_sum else 0.0,
                "classification": classify_bottleneck(key),
            }
        )
    bottlenecks.sort(key=lambda item: item["total_ms"], reverse=True)

    total_elapsed_ms = round(elapsed_seconds * 1000.0, 2)
    total_io_ms = round(load_data_ms, 2)
    payload = {
        "data_file": str(data_file),
        "mode": mode,
        "rerank_mode": rerank_mode if mode == "hybrid" else None,
        "requested_rerank_mode": rerank_mode,
        "rerank_mode_active": mode == "hybrid",
        "granularity": granularity,
        "question_count": len(results_log),
        "top_k": top_k,
        "task_type": task_type,
        "shell": shell,
        "sector": sector,
        "zone": zone,
        "chunk_pool": chunk_pool,
        "object_top_k": object_top_k,
        "support_top_k": support_top_k,
        "cognitive_top_k": cognitive_top_k,
        "cross_encoder_requested": bool(use_cross_encoder),
        "cross_encoder_loaded": bool(cross_encoder_model is not None),
        "elapsed_seconds": elapsed_seconds,
        "benchmark_io_ms": {
            "load_data_ms": load_data_ms,
        },
        "totals_ms": {
            "total_elapsed_ms": total_elapsed_ms,
            "total_ingest_ms": round(totals_ms["total_ingest_ms"], 2),
            "total_retrieval_ms": round(totals_ms["total_retrieval_ms"], 2),
            "total_completion_ms": round(totals_ms["total_completion_ms"], 2),
            "total_storage_ms": round(totals_ms["total_storage_ms"], 2),
            "total_vector_ms": round(totals_ms["total_vector_ms"], 2),
            "total_io_ms": total_io_ms,
        },
        "reuse_summary": {
            "total_cache_reuse_saved_ms": round(totals_ms["total_cache_reuse_saved_ms"], 2),
            "total_embedding_cache_hit_ms_saved": round(totals_ms["total_embedding_cache_hit_ms_saved"], 2),
        },
        "vector_info": vector_info,
        "runtime_config": runtime_config or {},
        "metrics": summary_metrics,
        "per_type": per_type_summary,
        "stage_timing_ms": timing_summary,
        "profiling_summary": profile_summary,
        "bottlenecks": bottlenecks[:12],
        "results": results_log,
    }
    aggregate_index_metadata = {
        "benchmark_name": "longmemeval",
        "fingerprint": dict(index_metadata.get("fingerprint") if index_metadata else runtime_fingerprint or {}),
        "index_built_at": str(index_metadata.get("index_built_at") if index_metadata else ""),
        "raw_counts": dict(aggregate_raw_counts),
        "index_doc_count": len(aggregate_indexed_doc_ids),
        "chunk_count": len(aggregate_chunk_metadata),
        "unique_segment_count": len(aggregate_indexed_segment_ids),
        "indexed_doc_ids": sorted(aggregate_indexed_doc_ids),
        "indexed_segment_ids": sorted(aggregate_indexed_segment_ids),
        "chunk_metadata_by_id": aggregate_chunk_metadata,
    }
    if vector_info is None:
        vector_info = {}
    if index_metadata is None:
        index_metadata = aggregate_index_metadata
    if runtime_fingerprint is None:
        runtime_fingerprint = {}
    payload.update(
        build_result_metadata(
            project_root=ROOT.parent,
            benchmark_name="longmemeval",
            question_count=len(results_log),
            vector_info=vector_info,
            index_metadata=aggregate_index_metadata,
            runtime_fingerprint=runtime_fingerprint,
            determinism=determinism,
        )
    )
    if mode == "evidence":
        reports_dir = report_root(out_file, "longmemeval")
        integrity_path = reports_dir / "integrity" / "longmemeval_integrity_report.json"
        candidate_path = reports_dir / "diagnostics" / "longmemeval_candidate_recall.json"
        oracle_path = reports_dir / "diagnostics" / "longmemeval_oracle_retrieval.json"
        channel_path = reports_dir / "diagnostics" / "longmemeval_channel_contribution.json"
        performance_path = reports_dir / "diagnostics" / "longmemeval_performance_cache.json"
        failure_path = reports_dir / "failures" / "longmemeval_failures.jsonl"
        topk_debug_path = reports_dir / "debug" / "longmemeval_topk_debug.jsonl"
        integrity_report = build_integrity_report(
            benchmark_name="longmemeval",
            raw_counts=dict(aggregate_index_metadata.get("raw_counts") or {}),
            index_metadata=aggregate_index_metadata,
            gold_segment_ids=all_gold_segment_ids,
            gold_document_ids=all_gold_document_ids,
        )
        candidate_report = build_candidate_recall_summary(
            benchmark_name="longmemeval",
            rows=candidate_diagnostics,
        )
        oracle_report = {
            "benchmark_name": "longmemeval",
            "oracle_query_count": len(oracle_rows),
            "oracle_recall@1": round(sum(1.0 for row in oracle_rows if row.get("top1_hit")) / max(1, len(oracle_rows)), 4),
            "oracle_recall@5": round(sum(1.0 for row in oracle_rows if row.get("top5_hit")) / max(1, len(oracle_rows)), 4),
            "oracle_recall@10": round(sum(1.0 for row in oracle_rows if row.get("top10_hit")) / max(1, len(oracle_rows)), 4),
            "rows": oracle_rows,
        }
        channel_report = build_per_channel_contribution_report(
            benchmark_name="longmemeval",
            rows=candidate_diagnostics,
        )
        performance_report = build_performance_cache_report(
            benchmark_name="longmemeval",
            timing_summary=timing_summary,
            reuse_summary=dict(payload.get("reuse_summary") or {}),
            runtime_config=runtime_config,
        )
        write_json(integrity_path, integrity_report)
        write_json(candidate_path, candidate_report)
        write_json(oracle_path, oracle_report)
        write_json(channel_path, channel_report)
        write_json(performance_path, performance_report)
        write_jsonl(failure_path, failure_rows)
        write_jsonl(topk_debug_path, topk_debug_rows)
        payload["reports"] = {
            "integrity": str(integrity_path),
            "candidate_recall": str(candidate_path),
            "oracle_retrieval": str(oracle_path),
            "channel_contribution": str(channel_path),
            "performance_cache": str(performance_path),
            "failures": str(failure_path),
            "topk_debug": str(topk_debug_path),
        }
        payload["integrity_report"] = integrity_report
        payload["candidate_recall_report"] = {
            key: value
            for key, value in candidate_report.items()
            if key != "failures"
        }
        payload["oracle_retrieval_report"] = {
            key: value for key, value in oracle_report.items() if key != "rows"
        }
        payload["channel_contribution_report"] = channel_report
        payload["performance_cache_report"] = performance_report
        payload["failure_summary"] = dict(candidate_report.get("failure_type_distribution") or {})

    print("\nSummary")
    print(f"  Questions:   {payload['question_count']}")
    print(f"  Mode:        {mode}")
    print(f"  Granularity: {granularity}")
    print(f"  Recall@5:    {summary_metrics['session']['recall_any@5']:.4f}")
    print(f"  Recall@10:   {summary_metrics['session']['recall_any@10']:.4f}")
    print(f"  NDCG@10:     {summary_metrics['session']['ndcg_any@10']:.4f}")
    print(f"  Time:        {elapsed_seconds:.1f}s")
    if timing_summary:
        for key in sorted(timing_summary):
            print(f"  {key}: {timing_summary[key]:.2f}ms")
    if bottlenecks:
        print("  Top Bottlenecks:")
        for item in bottlenecks[:5]:
            print(f"    - {item['stage']}: {item['avg_ms']:.2f}ms avg ({item['share_pct']:.2f}%, {item['classification']})")
    if vector_info:
        print(f"  Embedding:   {vector_info['embedding_model']} ({vector_info['embedding_provider']})")

    if out_file is not None:
        payload["benchmark_io_ms"]["json_write_ms"] = 0.0
        serialize_start = perf_counter()
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        payload["benchmark_io_ms"]["json_serialize_ms"] = round((perf_counter() - serialize_start) * 1000.0, 2)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        write_start = perf_counter()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(serialized, encoding="utf-8")
        payload["benchmark_io_ms"]["json_write_ms"] = round((perf_counter() - write_start) * 1000.0, 2)
        payload["totals_ms"]["total_io_ms"] = round(
            load_data_ms
            + float(payload["benchmark_io_ms"].get("json_serialize_ms", 0.0))
            + float(payload["benchmark_io_ms"].get("json_write_ms", 0.0)),
            2,
        )
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved:       {out_file}")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Sphere Memory CLI with LongMemEval.")
    parser.add_argument("data_file", type=Path, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--mode", choices=["vector", "bm25", "evidence", "activation", "hybrid"], default="evidence")
    parser.add_argument("--granularity", choices=["session", "turn"], default="session")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N questions")
    parser.add_argument("--rerank-mode", choices=["rule", "hybrid", "cross_encoder"], default="rule")
    parser.add_argument("--task-type", default="qa")
    parser.add_argument("--shell", type=int, default=2)
    parser.add_argument("--sector", default="knowledge")
    parser.add_argument("--zone", default="longmemeval")
    parser.add_argument("--chunk-pool", type=int, default=400)
    parser.add_argument("--cross-encoder", action="store_true", help="Enable cross-encoder reranking in evidence pipeline")
    parser.add_argument("--object-top-k", type=int, default=4, help="Number of structured evidence objects to keep for profiling/completion.")
    parser.add_argument("--support-top-k", type=int, default=4, help="Number of supporting context chunks to expand for profiling/completion.")
    parser.add_argument("--cognitive-top-k", type=int, default=0, help="Optional cognitive expansion budget for profiling.")
    parser.add_argument("--out", type=Path, default=None)
    add_token_economy_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_file = args.out
    if out_file is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rerank_tag = f"_{args.rerank_mode}" if args.mode == "hybrid" else ""
        limit_tag = f"_limit{args.limit}" if args.limit else ""
        out_file = ROOT / "benchmarks" / f"results_longmemeval_{args.mode}{rerank_tag}_{args.granularity}{limit_tag}_{stamp}.json"

    run_benchmark(
        data_file=args.data_file,
        mode=args.mode,
        granularity=args.granularity,
        top_k=args.top_k,
        limit=args.limit,
        rerank_mode=args.rerank_mode,
        task_type=args.task_type,
        shell=args.shell,
        sector=args.sector,
        zone=args.zone,
        chunk_pool=args.chunk_pool,
        out_file=out_file,
        use_cross_encoder=args.cross_encoder,
        object_top_k=args.object_top_k,
        support_top_k=args.support_top_k,
        cognitive_top_k=args.cognitive_top_k,
    )
    if args.record_token_economy:
        record_token_economy_for_metrics(
            metrics_path=out_file,
            output_dir=args.token_economy_output,
            tokenizer_model=args.tokenizer_model,
            baseline_types=args.token_economy_baseline_types,
            modes=args.token_economy_modes,
            context_token_budget=args.context_token_budget,
            recent_k=args.recent_k,
            low_saving_threshold=args.low_saving_threshold,
            quality_drop_threshold=args.quality_drop_threshold,
            evidence_bloat_threshold=args.evidence_bloat_threshold,
            metadata_bloat_threshold=args.metadata_bloat_threshold,
        )


if __name__ == "__main__":
    main()
