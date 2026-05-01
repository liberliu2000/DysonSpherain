from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from longmemeval_benchmark import (
    KS,
    BenchmarkWorkspaceManager,
    LONGMEMEVAL_ADAPTER_VERSION,
    SHARED_EMBEDDING_CACHE_DIR as DEFAULT_SHARED_EMBEDDING_CACHE_DIR,
    classify_bottleneck,
    evaluate_retrieval,
    flatten_numeric_metrics,
    materialize_ingest_profile_for_question,
    rank_bm25,
    rank_evidence,
    rank_hybrid,
    rank_vector,
    summarize_numeric_metrics,
    summarize_question_profile,
)
from benchmark_support import (
    best_gold_rank,
    build_candidate_recall_summary,
    build_clonemem_failure_taxonomy,
    build_integrity_report,
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
    ndcg_at,
    rank_benchmark_sources,
    recall_at,
    report_root,
    _focused_query_variants,
    _query_features,
    _retrieval_policy,
    write_json,
    write_jsonl,
)
from shard_utils import filter_sharded_items, load_allowlist, validate_shard_args
from token_economy_support import add_token_economy_args, record_token_economy_for_metrics


QUESTION_TYPE_TO_RUNTIME = {
    "single_point_factual": "qa",
    "comparison": "temporal_reasoning",
    "trajectory": "trace",
    "pattern": "trace",
    "causal": "trace",
    "counterfactual": "trace",
    "inference": "qa",
    "inferential": "qa",
    "unanswerable": "qa",
}


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


WORKSPACE_CACHE_ROOT = path_from_env(
    "SPHERE_CLONEMEM_CACHE_ROOT",
    ROOT / "benchmarks" / ".cache" / "clonemem_workspaces",
)
SHARED_EMBEDDING_CACHE_DIR = path_from_env(
    "SPHERE_CLONEMEM_EMBED_CACHE_ROOT",
    DEFAULT_SHARED_EMBEDDING_CACHE_DIR,
)
CLONEMEM_ADAPTER_VERSION = f"{LONGMEMEVAL_ADAPTER_VERSION}-clonemem"


def default_data_root() -> Path:
    candidates = [
        ROOT / "data" / "benchmarks" / "clonemem",
        ROOT.parent.parent / "tmp_benchmark_sources" / "CloneMemBench" / "data" / "releases",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def init_metric_bucket() -> dict[str, list[float]]:
    bucket: dict[str, list[float]] = {}
    for metric in ("recall_frac", "recall_any", "recall_all", "ndcg_any"):
        for k in KS:
            bucket[f"{metric}@{k}"] = []
    return bucket


def summarize_metric_bucket(bucket: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in bucket.items()
    }


def recall_fraction(ranked_ids: list[str], correct_ids: set[str], k: int) -> float:
    if not correct_ids:
        return 1.0
    return len(set(ranked_ids[:k]) & correct_ids) / len(correct_ids)


def build_parent_to_segment_trace_rows(
    *,
    query_id: str,
    query_text: str,
    question_type: str,
    gold_segment_ids: set[str],
    broad_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    trace: dict[str, Any],
) -> list[dict[str, Any]]:
    parent_audit = dict(trace.get("parent_audit") or {})
    selected = list(parent_audit.get("selected_child_anchors") or [])
    if not selected:
        return []
    broad_rank = {str(row.get("source_id") or ""): row.get("rank") for row in broad_rows}
    final_rank = {str(row.get("source_id") or ""): row.get("rank") for row in final_rows}
    parent_rank = {
        str(row.get("parent_id") or ""): rank
        for rank, row in enumerate(list(parent_audit.get("parent_candidates") or []), start=1)
    }
    query_features = dict(trace.get("query_features") or {})
    rows: list[dict[str, Any]] = []
    for item in selected:
        source_id = str(item.get("source_id") or "")
        parent_id = str(item.get("parent_id") or "")
        rows.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "question_type": question_type,
                "parent_id": parent_id,
                "selected_segment_id": source_id,
                "is_gold_segment": source_id in gold_segment_ids,
                "anchor_terms": list(item.get("matched_anchor_terms") or []),
                "query_anchor_terms": list(query_features.get("anchor_terms") or [])[:40],
                "score_components": {
                    "anchor_priority": item.get("anchor_priority"),
                    "direct_match": item.get("direct_match"),
                },
                "why_selected": "parent_session_anchor_preselection_diagnostic_only",
                "parent_rank": parent_rank.get(parent_id),
                "segment_rank_before": broad_rank.get(source_id),
                "segment_rank_after": final_rank.get(source_id),
                "order_index": item.get("order_index"),
            }
        )
    return rows


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _compact_result_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "sample_name": row.get("sample_name"),
        "question_id": row.get("question_id"),
        "native_question_id": row.get("native_question_id"),
        "question_type": row.get("question_type"),
        "context_len": row.get("context_len"),
        "language": row.get("language"),
        "correct_choice_id": row.get("correct_choice_id"),
        "evidence_ids": row.get("evidence_ids"),
        "metrics": row.get("metrics"),
        "stage_timing_ms": row.get("stage_timing_ms"),
        "retrieved_trace_ids": row.get("retrieved_trace_ids"),
    }
    candidate = row.get("candidate_recall")
    if isinstance(candidate, dict):
        compact["candidate_recall"] = {
            key: candidate.get(key)
            for key in (
                "candidate_recall@10",
                "candidate_recall@50",
                "candidate_recall@100",
                "candidate_recall@200",
                "candidate_ndcg@10",
                "final_recall@10",
                "final_ndcg@10",
                "failure_type",
                "dense_hit@100",
                "fused_hit@100",
            )
            if key in candidate
        }
    return compact


def _prefetch_dense_hits_for_questions(
    *,
    questions: list[dict[str, Any]],
    vector_store: Any,
    config: Any,
    pool_limit: int,
    bundle: dict[str, Any],
    sample_name: str,
) -> tuple[dict[str, list[list[dict[str, Any]]]], dict[str, Any]]:
    if not _env_enabled("SPHERE_CLONEMEM_CROSS_QUESTION_DENSE_BATCH", True):
        return {}, {"enabled": False, "reason": "disabled"}
    policy = _retrieval_policy(config, "clonemem", pool_limit)
    if not bool(policy.get("dense_semantic", True)) or not hasattr(vector_store, "search_many"):
        return {}, {"enabled": False, "reason": "dense_disabled_or_unsupported"}
    started = perf_counter()
    query_refs: list[tuple[str, int]] = []
    query_texts: list[str] = []
    variant_counts: dict[str, int] = {}
    feature_ms = 0.0
    for entry in questions:
        qid = str(entry.get("question_id") or "")
        route_context = {
            "benchmark": "clonemem",
            "question_type": entry.get("question_type"),
            "person_name": bundle["person_name"],
            "person_id": bundle["person_id"],
            "sample_name": sample_name,
            "question_time": entry.get("question_time"),
            "language": bundle["language"],
        }
        feature_start = perf_counter()
        query_features = _query_features(str(entry.get("question") or ""), route_context=route_context)
        feature_ms += (perf_counter() - feature_start) * 1000.0
        variants = _focused_query_variants(str(entry.get("question") or ""), query_features)
        variant_counts[qid] = len(variants)
        for variant_index, variant in enumerate(variants):
            query_refs.append((qid, variant_index))
            query_texts.append(variant)
    if not query_texts:
        return {}, {"enabled": True, "query_count": 0, "variant_count": 0}
    search_start = perf_counter()
    hit_lists = vector_store.search_many(query_texts, top_k=max(1, int(policy["dense_top_k"])))
    search_ms = (perf_counter() - search_start) * 1000.0
    by_question: dict[str, list[list[dict[str, Any]]]] = {
        qid: [[] for _ in range(count)]
        for qid, count in variant_counts.items()
    }
    for (qid, variant_index), hits in zip(query_refs, hit_lists):
        by_question[qid][variant_index] = hits
    return by_question, {
        "enabled": True,
        "question_count": len(questions),
        "variant_count": len(query_texts),
        "dense_top_k": int(policy["dense_top_k"]),
        "feature_ms": round(feature_ms, 2),
        "search_ms": round(search_ms, 2),
        "total_ms": round((perf_counter() - started) * 1000.0, 2),
    }


def detect_language(sample_path: Path) -> str:
    match = re.search(r"_benchmark_(en|zh)$", sample_path.stem)
    return match.group(1) if match else "unknown"


def detect_sample_name(sample_path: Path) -> str:
    language = detect_language(sample_path)
    suffix = f"_benchmark_{language}"
    if sample_path.stem.endswith(suffix):
        return sample_path.stem[: -len(suffix)]
    return sample_path.stem


def build_trace_text(item: dict[str, Any]) -> str:
    parts = [
        f"Medium: {str(item.get('medium') or '').strip()}",
        f"Event Date: {str(item.get('event_date') or '').strip()}",
        "",
        str(item.get("content") or "").strip(),
    ]
    return "\n".join(part for part in parts if part is not None).strip()


def normalize_trace_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item is not None and str(item).strip()}
    return {str(value)}


def extract_evidence_ids(qa_item: dict[str, Any]) -> set[str]:
    evidence_ids = set(normalize_trace_ids(qa_item.get("digital_trace_ids")))
    for evidence_item in qa_item.get("evidence", []) or []:
        evidence_ids.update(normalize_trace_ids(evidence_item.get("digital_trace_ids")))
    return evidence_ids


def discover_sample_paths(data_root: Path, context_len: str, language: str) -> list[Path]:
    paths: list[Path] = []
    for context_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if context_len != "all" and context_dir.name != context_len:
            continue
        pattern = "*_benchmark_*.json" if language == "all" else f"*_benchmark_{language}.json"
        paths.extend(sorted(context_dir.glob(pattern)))
    return paths


def load_sample_bundle(sample_path: Path) -> dict[str, Any]:
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    language = detect_language(sample_path)
    sample_name = detect_sample_name(sample_path)
    context_len = sample_path.parent.name
    context_items = [
        {
            "corpus_id": str(item["id"]),
            "source_segment_id": str(item["id"]),
            "source_doc_id": sample_name,
            "sample_id": sample_name,
            "conversation_id": sample_name,
            "speaker_id": str(data.get("person_name") or ""),
            "timestamp": str(item.get("event_date") or ""),
            "text": build_trace_text(item),
        }
        for item in data.get("context", [])
        if build_trace_text(item)
    ]
    questions: list[dict[str, Any]] = []
    for qa_item in data.get("qa_items", []):
        question_type = str(qa_item.get("question_type") or "").strip() or "qa"
        questions.append(
            {
                "question_id": f"{sample_name}_{qa_item['id']}",
                "native_question_id": str(qa_item["id"]),
                "question_type": question_type,
                "task_type": QUESTION_TYPE_TO_RUNTIME.get(question_type, "qa"),
                "question": str(qa_item.get("question") or "").strip(),
                "answer": str(qa_item.get("answer") or ""),
                "dimension": str(qa_item.get("dimension") or ""),
                "correct_choice_id": str(qa_item.get("correct_choice_id") or ""),
                "evidence_ids": extract_evidence_ids(qa_item),
            }
        )
    return {
        "sample_path": sample_path,
        "sample_name": sample_name,
        "person_name": str(data.get("person_name") or ""),
        "person_id": str(data.get("person_id") or ""),
        "context_len": context_len,
        "language": language,
        "corpus_items": context_items,
        "questions": questions,
    }


def run_benchmark(
    data_root: Path,
    context_len: str,
    language: str,
    mode: str,
    top_k: int,
    limit: int,
    question_limit: int,
    rerank_mode: str,
    shell: int,
    sector: str,
    zone: str,
    chunk_pool: int,
    out_file: Path | None,
    use_cross_encoder: bool = False,
    object_top_k: int = 4,
    support_top_k: int = 4,
    cognitive_top_k: int = 0,
    shard_index: int | None = None,
    shard_count: int | None = None,
    question_id_allowlist_path: Path | None = None,
    sample_id_allowlist_path: Path | None = None,
    max_questions: int = 0,
    resume_existing: bool = False,
) -> dict[str, Any]:
    validate_shard_args(shard_index, shard_count)
    if resume_existing and out_file is not None and out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    determinism = configure_benchmark_determinism()
    sample_paths = discover_sample_paths(data_root, context_len=context_len, language=language)
    if not sample_paths:
        raise FileNotFoundError(
            f"No CloneMem sample files found under {data_root} for context_len={context_len} language={language}"
        )
    question_id_allowlist = load_allowlist(question_id_allowlist_path)
    sample_id_allowlist = load_allowlist(sample_id_allowlist_path)
    if sample_id_allowlist:
        sample_paths = [
            path
            for path in sample_paths
            if detect_sample_name(path) in sample_id_allowlist
        ]
    if limit > 0:
        sample_paths = sample_paths[:limit]

    load_start = perf_counter()
    bundles = [load_sample_bundle(path) for path in sample_paths]
    load_data_ms = round((perf_counter() - load_start) * 1000.0, 2)

    metrics_segment = init_metric_bucket()
    timing_metrics: dict[str, list[float]] = defaultdict(list)
    profile_metrics: dict[str, list[float]] = defaultdict(list)
    per_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_context_len: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_language: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_sample: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    results_log: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    topk_debug_rows: list[dict[str, Any]] = []
    parent_to_segment_trace_rows: list[dict[str, Any]] = []
    vector_info: dict[str, Any] | None = None
    index_metadata: dict[str, Any] | None = None
    runtime_fingerprint: dict[str, Any] | None = None
    runtime_config: dict[str, Any] | None = None
    route_policy_config: dict[str, Any] | None = None
    total_questions = 0
    start_time = datetime.now()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
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
    aggregated_signatures: set[str] = set()
    all_gold_segment_ids: set[str] = set()
    all_gold_document_ids: set[str] = set()
    aggregate_raw_counts: dict[str, int] = defaultdict(int)
    aggregate_chunk_metadata: dict[str, dict[str, Any]] = {}
    aggregate_indexed_segment_ids: set[str] = set()
    aggregate_indexed_doc_ids: set[str] = set()
    oracle_rows: list[dict[str, Any]] = []
    oracle_retrieval_mode = str(os.getenv("SPHERE_ORACLE_RETRIEVAL_MODE") or "direct_index").strip().lower()
    shard_meta = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shard_question_count": 0,
        "total_available_question_count": 0,
        "shard_assignment_method": "unsharded",
        "question_id_allowlist_count": len(question_id_allowlist),
        "sample_id_allowlist_count": len(sample_id_allowlist),
        "max_questions": int(max_questions or 0),
    }
    remaining_questions = int(max_questions or 0)

    cross_encoder_model = None
    if use_cross_encoder:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("Cross-encoder reranking enabled")
        except Exception as exc:
            print(f"Cross-encoder not available: {exc}")

    dense_prefetch_by_signature: dict[str, dict[str, list[list[dict[str, Any]]]]] = {}
    dense_prefetch_profile_by_signature: dict[str, dict[str, Any]] = {}

    for bundle in bundles:
        if int(max_questions or 0) > 0 and int(shard_meta["shard_question_count"]) >= int(max_questions):
            break
        sample_name = str(bundle["sample_name"])
        corpus_items = list(bundle["corpus_items"])
        questions = list(bundle["questions"])
        if question_limit > 0:
            questions = questions[:question_limit]
        current_limit = remaining_questions if remaining_questions > 0 else 0
        filtered_questions, bundle_shard_meta = filter_sharded_items(
            questions,
            benchmark_name="clonemem",
            shard_index=shard_index,
            shard_count=shard_count,
            question_id_getter=lambda item: str(item.get("question_id") or ""),
            sample_id_getter=lambda item: sample_name,
            question_text_getter=lambda item: str(item.get("question") or ""),
            question_id_allowlist=question_id_allowlist,
            sample_id_allowlist=sample_id_allowlist,
            max_questions=current_limit,
        )
        shard_meta["total_available_question_count"] += int(bundle_shard_meta["total_available_question_count"])
        shard_meta["shard_question_count"] += int(bundle_shard_meta["shard_question_count"])
        if shard_meta["shard_assignment_method"] == "unsharded":
            shard_meta["shard_assignment_method"] = str(bundle_shard_meta["shard_assignment_method"])
        questions = filtered_questions
        if int(max_questions or 0) > 0:
            remaining_questions = max(0, remaining_questions - len(questions))
        ordered_corpus_ids = [str(item["corpus_id"]) for item in corpus_items]
        corpus_by_node_id = {
            f"bench_{item_index:04d}": item
            for item_index, item in enumerate(corpus_items)
        }
        raw_counts = build_raw_counts(corpus_items, question_count=len(questions), session_count=1)
        print(
            f"[sample {sample_name}] "
            f"context_len={bundle['context_len']} language={bundle['language']} "
            f"segments={len(corpus_items)} questions={len(questions)}"
        )

        for question_index, entry in enumerate(questions, start=1):
            total_questions += 1
            question_profile: dict[str, Any] = {}
            retry_count = 0
            current_workspace_manager = workspace_manager

            while True:
                question_start = perf_counter()
                stage_start = perf_counter()
                workspace, ingest_reuse = current_workspace_manager.acquire(
                    corpus_items=corpus_items,
                    granularity="segment",
                    question_type="clonemem",
                    shell=shell,
                    sector=sector,
                    zone=f"{zone}_{sample_name}",
                    benchmark_name="clonemem",
                    adapter_version=CLONEMEM_ADAPTER_VERSION,
                    raw_counts=raw_counts,
                )
                acquire_total_ms = round((perf_counter() - stage_start) * 1000.0, 2)
                if ingest_reuse.get("workspace_reused"):
                    ingest_ms = round(float(ingest_reuse.get("ingest_lookup_ms", 0.0)), 2)
                    service_init_ms = round(max(0.0, acquire_total_ms - ingest_ms), 2)
                else:
                    ingest_ms = round(float(workspace.build_elapsed_ms), 2)
                    service_init_ms = round(max(0.0, acquire_total_ms - ingest_ms), 2)

                actual_ingest_profile = materialize_ingest_profile_for_question(
                    workspace.ingest_profile,
                    ingest_reuse,
                )
                if vector_info is None:
                    vector_info = workspace.vector_info
                if index_metadata is None:
                    index_metadata = workspace.index_metadata
                if runtime_fingerprint is None:
                    runtime_fingerprint = build_runtime_fingerprint(
                        config=workspace.config,
                        benchmark_name="clonemem",
                        adapter_version=CLONEMEM_ADAPTER_VERSION,
                        granularity="segment",
                        vector_info=workspace.vector_info,
                    )
                if workspace.signature not in aggregated_signatures:
                    aggregated_signatures.add(workspace.signature)
                    for key, value in dict((workspace.index_metadata or {}).get("raw_counts") or {}).items():
                        if isinstance(value, (int, float)):
                            aggregate_raw_counts[key] += int(value)
                    aggregate_indexed_segment_ids.update(
                        set((workspace.index_metadata or {}).get("indexed_segment_ids") or [])
                    )
                    aggregate_indexed_doc_ids.update(
                        set((workspace.index_metadata or {}).get("indexed_doc_ids") or [])
                    )
                    for chunk_id, meta in dict((workspace.index_metadata or {}).get("chunk_metadata_by_id") or {}).items():
                        aggregate_chunk_metadata[f"{workspace.signature}:{chunk_id}"] = dict(meta)
                    source_records = dict((workspace.index_metadata or {}).get("source_records_by_id") or {})
                    oracle_gold_ids = {
                        evidence_id
                        for question in questions
                        for evidence_id in set(question.get("evidence_ids") or [])
                    }
                    for gold_id in sorted(oracle_gold_ids):
                        source_record = dict(source_records.get(gold_id) or {})
                        query_text = str(source_record.get("text") or source_record.get("original_segment_text") or "").strip()
                        if not query_text:
                            oracle_rows.append(
                                {
                                    "sample_name": sample_name,
                                    "gold_segment_id": gold_id,
                                    "found_in_index": False,
                                    "self_rank": None,
                                    "top1_hit": False,
                                    "top5_hit": False,
                                    "top10_hit": False,
                                }
                            )
                            continue
                        if oracle_retrieval_mode in {"self_retrieval", "retrieval", "full"}:
                            oracle_trace = rank_benchmark_sources(
                                query=query_text,
                                benchmark_name="clonemem",
                                vector_store=workspace.vector_store,
                                storage=workspace.storage,
                                index_metadata=workspace.index_metadata,
                                config=workspace.config,
                                route_context={
                                    "benchmark": "clonemem",
                                    "question_type": "oracle",
                                    "person_name": bundle["person_name"],
                                    "person_id": bundle["person_id"],
                                    "sample_name": sample_name,
                                    "language": bundle["language"],
                                },
                                pool_limit=50,
                            )
                            oracle_final_rows = candidate_rows(oracle_trace["final_candidates"], limit=50)
                            self_rank = best_gold_rank(oracle_final_rows, {gold_id})
                        else:
                            self_rank = 1
                        oracle_rows.append(
                            {
                                "sample_name": sample_name,
                                "gold_segment_id": gold_id,
                                "found_in_index": True,
                                "self_rank": self_rank,
                                "top1_hit": bool(self_rank is not None and self_rank <= 1),
                                "top5_hit": bool(self_rank is not None and self_rank <= 5),
                                "top10_hit": bool(self_rank is not None and self_rank <= 10),
                                "oracle_retrieval_mode": oracle_retrieval_mode,
                            }
                        )
                if runtime_config is None:
                    route_policy_config = _retrieval_policy(
                        workspace.config,
                        "clonemem",
                        max(300, top_k * 6),
                    )
                    runtime_config = {
                        "benchmark_name": "clonemem",
                        "embedding_model": workspace.config.embedding_model_name,
                        "embed_local_grain": bool(workspace.config.embed_local_grain),
                        "rerank_mode_default": workspace.config.rerank_mode_default,
                        "cross_encoder_model": workspace.config.cross_encoder_model_name,
                        "cross_encoder_requested": bool(use_cross_encoder),
                        "cross_encoder_loaded": bool(cross_encoder_model is not None),
                        "creative_mode": workspace.config.creative_mode_name,
                        "enable_benchmark_route_tuning": bool(workspace.config.enable_benchmark_route_tuning),
                        "route_scope": "benchmark_only",
                        "route_policy_hash": stable_hash(route_policy_config),
                        "clonemem_route_policy": {
                            "promotion": "phase5_lexical_anchor_gate_protected_top3",
                            "clonemem_lexical_anchor_gate_enabled": bool(
                                route_policy_config.get("clonemem_lexical_anchor_gate_enabled", False)
                            ),
                            "clonemem_lexical_anchor_gate_protected_top_k": int(
                                route_policy_config.get("clonemem_lexical_anchor_gate_protected_top_k") or 0
                            ),
                            "clonemem_lexical_anchor_gate_factor": float(
                                route_policy_config.get("clonemem_lexical_anchor_gate_factor") or 0.0
                            ),
                            "clonemem_lexical_anchor_gate_min_support": int(
                                route_policy_config.get("clonemem_lexical_anchor_gate_min_support") or 0
                            ),
                            "clonemem_lexical_anchor_gate_min_anchor_score": float(
                                route_policy_config.get("clonemem_lexical_anchor_gate_min_anchor_score") or 0.0
                            ),
                        },
                    }

                dense_prefetch_profile: dict[str, Any] = {}
                precomputed_dense_hit_lists: list[list[dict[str, Any]]] | None = None
                if mode == "evidence":
                    if workspace.signature not in dense_prefetch_by_signature:
                        prefetch_hits, prefetch_profile = _prefetch_dense_hits_for_questions(
                            questions=questions,
                            vector_store=workspace.vector_store,
                            config=workspace.config,
                            pool_limit=max(300, top_k * 6),
                            bundle=bundle,
                            sample_name=sample_name,
                        )
                        dense_prefetch_by_signature[workspace.signature] = prefetch_hits
                        dense_prefetch_profile_by_signature[workspace.signature] = prefetch_profile
                    dense_prefetch_profile = dense_prefetch_profile_by_signature.get(workspace.signature, {})
                    precomputed_dense_hit_lists = dense_prefetch_by_signature.get(workspace.signature, {}).get(
                        str(entry.get("question_id") or "")
                    )

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
                        question_profile = {
                            "ingest": actual_ingest_profile,
                            "ingest_cached": workspace.ingest_profile,
                            "reuse": ingest_reuse,
                        }
                    elif mode == "bm25":
                        ranked_ids, ranked_items, stage_timing_ms = rank_bm25(
                            entry["question"],
                            storage=workspace.storage,
                            ordered_corpus_ids=ordered_corpus_ids,
                            corpus_by_node_id=corpus_by_node_id,
                            top_k=top_k,
                            chunk_pool=chunk_pool,
                        )
                        question_profile = {
                            "ingest": actual_ingest_profile,
                            "ingest_cached": workspace.ingest_profile,
                            "reuse": ingest_reuse,
                        }
                    elif mode == "evidence":
                        evidence_stage_start = perf_counter()
                        hybrid_trace = rank_benchmark_sources(
                            query=entry["question"],
                            benchmark_name="clonemem",
                            vector_store=workspace.vector_store,
                            storage=workspace.storage,
                            index_metadata=workspace.index_metadata,
                            config=workspace.config,
                            route_context={
                                "benchmark": "clonemem",
                                "question_type": entry["question_type"],
                                "person_name": bundle["person_name"],
                                "person_id": bundle["person_id"],
                                "sample_name": sample_name,
                                "question_time": entry.get("question_time"),
                                "language": bundle["language"],
                            },
                            pool_limit=max(300, top_k * 6),
                            precomputed_dense_hit_lists=precomputed_dense_hit_lists,
                        )
                        final_source_rows = candidate_rows(
                            hybrid_trace["final_candidates"],
                            limit=max(300, top_k * 6),
                        )
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
                                "dense_prefetch": dict(dense_prefetch_profile),
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
                            task_type=entry["task_type"],
                            top_k=top_k,
                            rerank_mode=rerank_mode,
                            apply_rerank=False,
                        )
                        question_profile = {
                            "ingest": actual_ingest_profile,
                            "ingest_cached": workspace.ingest_profile,
                            "reuse": ingest_reuse,
                        }
                    else:
                        ranked_ids, ranked_items, stage_timing_ms = rank_hybrid(
                            entry["question"],
                            activation=workspace.activation,
                            router=workspace.router,
                            reranker=workspace.reranker,
                            ordered_corpus_ids=ordered_corpus_ids,
                            corpus_by_node_id=corpus_by_node_id,
                            task_type=entry["task_type"],
                            top_k=top_k,
                            rerank_mode=rerank_mode,
                            apply_rerank=True,
                        )
                        question_profile = {
                            "ingest": actual_ingest_profile,
                            "ingest_cached": workspace.ingest_profile,
                            "reuse": ingest_reuse,
                        }
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
                    retry_cache_root = (
                        WORKSPACE_CACHE_ROOT
                        / "_question_retries"
                        / f"{sample_name}_q{question_index:04d}_try{retry_count}"
                    )
                    current_workspace_manager = BenchmarkWorkspaceManager(
                        retry_cache_root,
                        SHARED_EMBEDDING_CACHE_DIR,
                    )

            question_total_ms = round((perf_counter() - question_start) * 1000.0, 2)
            retrieval_ms = float(
                stage_timing_ms.get(
                    "total_ms",
                    stage_timing_ms.get("vector_total_ms", stage_timing_ms.get("hybrid_total_ms", 0.0)),
                )
            )
            completion_ms = float(stage_timing_ms.get("completion_total_ms", 0.0)) + float(
                stage_timing_ms.get("cognitive_total_ms", 0.0)
            )
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
            stage_timing_ms["ingest_storage_ms"] = float(
                ingest_backend.get("storage", {}).get("total_ms", 0.0)
            )
            stage_timing_ms["ingest_vector_ms"] = float(
                ingest_backend.get("vector", {}).get("total_ms", 0.0)
            )
            stage_timing_ms["embedding_cache_hit_ms_saved"] = float(
                question_summary.get("embedding_cache_hit_ms_saved", 0.0)
            )
            for key, value in stage_timing_ms.items():
                timing_metrics[key].append(float(value))
            flatten_numeric_metrics(
                "profile",
                {key: value for key, value in question_profile.items() if key != "ingest_cached"},
                profile_metrics,
            )
            totals_ms["total_ingest_ms"] += float(ingest_ms)
            totals_ms["total_retrieval_ms"] += float(retrieval_ms)
            totals_ms["total_completion_ms"] += float(completion_ms)
            totals_ms["total_storage_ms"] += float(question_summary.get("total_storage_ms", 0.0))
            totals_ms["total_vector_ms"] += float(question_summary.get("total_vector_ms", 0.0))
            totals_ms["total_cache_reuse_saved_ms"] += float(
                question_summary.get("cache_reuse_saved_ms", 0.0)
            )
            totals_ms["total_embedding_cache_hit_ms_saved"] += float(
                question_summary.get("embedding_cache_hit_ms_saved", 0.0)
            )

            entry_metrics: dict[str, float] = {}
            for k in KS:
                frac = recall_fraction(ranked_ids, entry["evidence_ids"], k)
                recall_any, recall_all, ndcg_any = evaluate_retrieval(ranked_ids, entry["evidence_ids"], k)
                metrics_segment[f"recall_frac@{k}"].append(frac)
                metrics_segment[f"recall_any@{k}"].append(recall_any)
                metrics_segment[f"recall_all@{k}"].append(recall_all)
                metrics_segment[f"ndcg_any@{k}"].append(ndcg_any)
                entry_metrics[f"recall_frac@{k}"] = frac
                entry_metrics[f"recall_any@{k}"] = recall_any
                entry_metrics[f"ndcg_any@{k}"] = ndcg_any

            question_type = entry["question_type"]
            bundle_context_len = str(bundle["context_len"])
            bundle_language = str(bundle["language"])
            per_type[question_type]["recall_frac@10"].append(entry_metrics["recall_frac@10"])
            per_type[question_type]["ndcg_any@10"].append(entry_metrics["ndcg_any@10"])
            per_context_len[bundle_context_len]["recall_frac@10"].append(entry_metrics["recall_frac@10"])
            per_context_len[bundle_context_len]["ndcg_any@10"].append(entry_metrics["ndcg_any@10"])
            per_language[bundle_language]["recall_frac@10"].append(entry_metrics["recall_frac@10"])
            per_language[bundle_language]["ndcg_any@10"].append(entry_metrics["ndcg_any@10"])
            per_sample[sample_name]["recall_frac@10"].append(entry_metrics["recall_frac@10"])
            per_sample[sample_name]["ndcg_any@10"].append(entry_metrics["ndcg_any@10"])
            all_gold_segment_ids.update(set(entry["evidence_ids"]))
            all_gold_document_ids.add(sample_name)

            if mode == "evidence":
                ranking_payload = dict((question_profile.get("pipeline") or {}).get("retrieval", {}).get("ranking") or {})
                broad_rows = list(ranking_payload.get("broad_top_candidates") or [])
                reranked_rows = list(ranking_payload.get("reranked_top_candidates") or [])
                final_rows = list(ranking_payload.get("final_top_candidates") or [])
                query_diag = build_query_diagnostic(
                    benchmark_name="clonemem",
                    query_id=entry["question_id"],
                    query_text=entry["question"],
                    answer_text=entry["answer"],
                    gold_segment_ids=set(entry["evidence_ids"]),
                    gold_evidence_ids=set(entry["evidence_ids"]),
                    broad_rows=broad_rows,
                    reranked_rows=reranked_rows,
                    final_rows=final_rows,
                    trace=hybrid_trace,
                    index_metadata=workspace.index_metadata,
                )
                failure_row = build_query_failure(
                    benchmark_name="clonemem",
                    query_id=entry["question_id"],
                    query_text=entry["question"],
                    answer_text=entry["answer"],
                    gold_segment_ids=set(entry["evidence_ids"]),
                    gold_evidence_ids=set(entry["evidence_ids"]),
                    broad_rows=broad_rows,
                    reranked_rows=reranked_rows,
                    final_rows=final_rows,
                    index_metadata=workspace.index_metadata,
                    trace=hybrid_trace,
                )
                if failure_row is not None:
                    query_diag["failure_type"] = str(failure_row.get("failure_type") or "unknown")
                    failure_rows.append(failure_row)
                query_diag["recall_frac@10"] = entry_metrics["recall_frac@10"]
                query_diag["recall_any@10"] = entry_metrics["recall_any@10"]
                query_diag["ndcg_any@10"] = entry_metrics["ndcg_any@10"]
                query_diag["whether_gold_in_candidate_100"] = recall_at(broad_rows, set(entry["evidence_ids"]), 100) > 0.0
                topk_debug_rows.append(
                    build_topk_debug_record(
                        benchmark_name="clonemem",
                        query_id=entry["question_id"],
                        query_text=entry["question"],
                        answer_text=entry["answer"],
                        gold_segment_ids=set(entry["evidence_ids"]),
                        failure_type=str(query_diag.get("failure_type") or "ok"),
                        broad_rows=broad_rows,
                        reranked_rows=reranked_rows,
                        final_rows=final_rows,
                        trace=hybrid_trace,
                    )
                )
                parent_to_segment_trace_rows.extend(
                    build_parent_to_segment_trace_rows(
                        query_id=entry["question_id"],
                        query_text=entry["question"],
                        question_type=question_type,
                        gold_segment_ids=set(entry["evidence_ids"]),
                        broad_rows=broad_rows,
                        final_rows=final_rows,
                        trace=hybrid_trace,
                    )
                )
                candidate_diagnostics.append(query_diag)

            results_log.append(
                {
                    "sample_name": sample_name,
                    "sample_path": str(bundle["sample_path"]),
                    "person_name": bundle["person_name"],
                    "person_id": bundle["person_id"],
                    "context_len": bundle["context_len"],
                    "language": bundle["language"],
                    "question_id": entry["question_id"],
                    "native_question_id": entry["native_question_id"],
                    "question_type": question_type,
                    "task_type": entry["task_type"],
                    "dimension": entry["dimension"],
                    "correct_choice_id": entry["correct_choice_id"],
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "evidence_ids": sorted(entry["evidence_ids"]),
                    "metrics": entry_metrics,
                    "stage_timing_ms": stage_timing_ms,
                    "profiling": question_profile,
                    "retrieved_trace_ids": ranked_ids[:top_k],
                    "ranked_items": ranked_items,
                    "candidate_recall": candidate_diagnostics[-1] if mode == "evidence" and candidate_diagnostics else None,
                }
            )

            print(
                f"  [q {question_index:04d}/{len(questions):04d}] "
                f"{question_type[:24]:24} "
                f"R@10={entry_metrics['recall_frac@10']:.2f} "
                f"Ingest={ingest_ms:.0f}ms ReuseSaved={float(ingest_reuse.get('cache_reuse_saved_ms', 0.0)):.0f}ms"
            )
            current_workspace_manager.release(workspace.signature)
        gc.collect()

    workspace_manager.close_all()
    if gc_was_enabled:
        gc.enable()

    elapsed_seconds = (datetime.now() - start_time).total_seconds()
    segment_summary = summarize_metric_bucket(metrics_segment)
    per_type_summary = {
        question_type: {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in metric_map.items()
        }
        for question_type, metric_map in per_type.items()
    }
    per_context_len_summary = {
        name: {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in metric_map.items()
        }
        for name, metric_map in per_context_len.items()
    }
    per_language_summary = {
        name: {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in metric_map.items()
        }
        for name, metric_map in per_language.items()
    }
    per_sample_summary = {
        name: {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in metric_map.items()
        }
        for name, metric_map in per_sample.items()
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
                "share_pct": round((total_ms / question_total_sum) * 100.0, 2)
                if question_total_sum
                else 0.0,
                "classification": classify_bottleneck(key),
            }
        )
    bottlenecks.sort(key=lambda item: item["total_ms"], reverse=True)

    total_elapsed_ms = round(elapsed_seconds * 1000.0, 2)
    compact_metrics_enabled = _env_enabled("SPHERE_CLONEMEM_COMPACT_METRICS", True)
    payload_results = (
        [_compact_result_row(row) for row in results_log]
        if compact_metrics_enabled
        else results_log
    )
    payload = {
        "data_root": str(data_root),
        "context_len": context_len,
        "language": language,
        "mode": mode,
        "rerank_mode": rerank_mode if mode == "hybrid" else None,
        "requested_rerank_mode": rerank_mode,
        "rerank_mode_active": mode == "hybrid",
        "sample_count": len(bundles),
        "sample_names": [str(bundle["sample_name"]) for bundle in bundles],
        "question_count": total_questions,
        "question_limit": question_limit,
        **shard_meta,
        "top_k": top_k,
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
            "total_io_ms": round(load_data_ms, 2),
        },
        "reuse_summary": {
            "total_cache_reuse_saved_ms": round(totals_ms["total_cache_reuse_saved_ms"], 2),
            "total_embedding_cache_hit_ms_saved": round(
                totals_ms["total_embedding_cache_hit_ms_saved"],
                2,
            ),
        },
        "question_types": sorted(per_type_summary),
        "vector_info": vector_info,
        "runtime_config": runtime_config or {},
        "route_policy_config": route_policy_config or {},
        "config_hash": stable_hash(runtime_config or {}),
        "route_policy_hash": stable_hash(route_policy_config or {}),
        "metrics": {
            "segment": segment_summary,
        },
        "per_type": per_type_summary,
        "per_context_len": per_context_len_summary,
        "per_language": per_language_summary,
        "per_sample": per_sample_summary,
        "stage_timing_ms": timing_summary,
        "profiling_summary": profile_summary,
        "bottlenecks": bottlenecks[:12],
        "results": payload_results,
        "artifact_policy": {
            "compact_metrics_enabled": bool(compact_metrics_enabled),
            "full_result_rows_in_metrics": not bool(compact_metrics_enabled),
            "full_result_rows_available": False,
        },
    }
    aggregate_index_metadata = {
        "benchmark_name": "clonemem",
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
    oracle_report = {
        "benchmark_name": "clonemem",
        "oracle_query_count": len(oracle_rows),
        "oracle_recall@1": round(sum(1.0 for row in oracle_rows if row.get("top1_hit")) / max(1, len(oracle_rows)), 4),
        "oracle_recall@5": round(sum(1.0 for row in oracle_rows if row.get("top5_hit")) / max(1, len(oracle_rows)), 4),
        "oracle_recall@10": round(sum(1.0 for row in oracle_rows if row.get("top10_hit")) / max(1, len(oracle_rows)), 4),
        "oracle_retrieval_mode": oracle_retrieval_mode,
        "rows": oracle_rows,
    }
    if vector_info is None:
        vector_info = {}
    if runtime_fingerprint is None:
        runtime_fingerprint = {}
    payload.update(
        build_result_metadata(
            project_root=ROOT.parent,
            benchmark_name="clonemem",
            question_count=total_questions,
            vector_info=vector_info,
            index_metadata=aggregate_index_metadata,
            runtime_fingerprint=runtime_fingerprint,
            determinism=determinism,
        )
    )
    if mode == "evidence":
        reports_dir = report_root(out_file, "clonemem")
        integrity_path = reports_dir / "integrity" / "clonemem_integrity_report.json"
        candidate_path = reports_dir / "diagnostics" / "clonemem_candidate_recall.json"
        oracle_path = reports_dir / "diagnostics" / "clonemem_oracle_retrieval.json"
        channel_path = reports_dir / "diagnostics" / "clonemem_channel_contribution.json"
        taxonomy_path = reports_dir / "diagnostics" / "clonemem_failure_taxonomy.json"
        performance_path = reports_dir / "diagnostics" / "clonemem_performance_cache.json"
        parent_trace_path = reports_dir / "diagnostics" / "parent_to_segment_selection_traces.jsonl"
        failure_path = reports_dir / "failures" / "clonemem_failures.jsonl"
        topk_debug_path = reports_dir / "debug" / "clonemem_topk_debug.jsonl"
        full_results_path = reports_dir / "debug" / "clonemem_full_results.jsonl"
        integrity_report = build_integrity_report(
            benchmark_name="clonemem",
            raw_counts=dict(aggregate_index_metadata.get("raw_counts") or {}),
            index_metadata=aggregate_index_metadata,
            gold_segment_ids=all_gold_segment_ids,
            gold_document_ids=all_gold_document_ids,
        )
        if oracle_report["oracle_recall@10"] < 1.0:
            integrity_report["p0_bugs"] = list(dict.fromkeys(list(integrity_report.get("p0_bugs") or []) + ["oracle_self_retrieval_failed"]))
        candidate_report = build_candidate_recall_summary(
            benchmark_name="clonemem",
            rows=candidate_diagnostics,
        )
        channel_report = build_per_channel_contribution_report(
            benchmark_name="clonemem",
            rows=candidate_diagnostics,
        )
        taxonomy_report = build_clonemem_failure_taxonomy(candidate_diagnostics)
        performance_report = build_performance_cache_report(
            benchmark_name="clonemem",
            timing_summary=timing_summary,
            reuse_summary=dict(payload.get("reuse_summary") or {}),
            runtime_config=runtime_config,
        )
        write_json(integrity_path, integrity_report)
        write_json(candidate_path, candidate_report)
        write_json(oracle_path, oracle_report)
        write_json(channel_path, channel_report)
        write_json(taxonomy_path, taxonomy_report)
        write_json(performance_path, performance_report)
        try:
            parent_trace_limit = int(os.getenv("SPHERE_CLONEMEM_PARENT_TRACE_ROW_LIMIT", "1000") or "1000")
        except ValueError:
            parent_trace_limit = 1000
        try:
            topk_debug_limit = int(os.getenv("SPHERE_CLONEMEM_TOPK_DEBUG_ROW_LIMIT", "200") or "200")
        except ValueError:
            topk_debug_limit = 200
        parent_trace_rows_to_write = (
            parent_to_segment_trace_rows
            if parent_trace_limit < 0
            else parent_to_segment_trace_rows[:parent_trace_limit]
        )
        topk_debug_rows_to_write = (
            topk_debug_rows
            if topk_debug_limit < 0
            else topk_debug_rows[:topk_debug_limit]
        )
        write_jsonl(parent_trace_path, parent_trace_rows_to_write)
        write_jsonl(failure_path, failure_rows)
        write_jsonl(topk_debug_path, topk_debug_rows_to_write)
        if _env_enabled("SPHERE_CLONEMEM_WRITE_FULL_RESULTS", False):
            write_jsonl(full_results_path, results_log)
            payload["artifact_policy"]["full_result_rows_available"] = True
            payload["artifact_policy"]["full_result_rows_path"] = str(full_results_path)
        payload["reports"] = {
            "integrity": str(integrity_path),
            "candidate_recall": str(candidate_path),
            "oracle_retrieval": str(oracle_path),
            "channel_contribution": str(channel_path),
            "failure_taxonomy": str(taxonomy_path),
            "performance_cache": str(performance_path),
            "parent_to_segment_selection_traces": str(parent_trace_path),
            "failures": str(failure_path),
            "topk_debug": str(topk_debug_path),
        }
        payload["integrity_report"] = integrity_report
        payload["candidate_recall_report"] = {
            key: value for key, value in candidate_report.items() if key != "queries"
        }
        payload["oracle_retrieval_report"] = {
            key: value for key, value in oracle_report.items() if key != "rows"
        }
        payload["channel_contribution_report"] = channel_report
        payload["failure_taxonomy_report"] = taxonomy_report
        payload["performance_cache_report"] = performance_report
        payload["parent_to_segment_trace_report"] = {
            "trace_count": len(parent_to_segment_trace_rows),
            "written_trace_count": len(parent_trace_rows_to_write),
            "trace_row_limit": parent_trace_limit,
            "gold_selected_count": sum(1 for row in parent_to_segment_trace_rows if row.get("is_gold_segment")),
        }
        payload["topk_debug_report"] = {
            "debug_row_count": len(topk_debug_rows),
            "written_debug_row_count": len(topk_debug_rows_to_write),
            "debug_row_limit": topk_debug_limit,
        }
        payload["failure_summary"] = dict(candidate_report.get("failure_type_distribution") or {})

    print("\nSummary")
    print(f"  Samples:         {payload['sample_count']}")
    print(f"  Questions:       {payload['question_count']}")
    print(f"  Mode:            {mode}")
    print(f"  Segment R@10:    {segment_summary['recall_frac@10']:.4f}")
    print(f"  Segment Any@10:  {segment_summary['recall_any@10']:.4f}")
    print(f"  Segment NDCG@10: {segment_summary['ndcg_any@10']:.4f}")
    print(f"  Time:            {elapsed_seconds:.1f}s")
    if timing_summary:
        for key in sorted(timing_summary):
            print(f"  {key}: {timing_summary[key]:.2f}ms")
    if bottlenecks:
        print("  Top Bottlenecks:")
        for item in bottlenecks[:5]:
            print(
                f"    - {item['stage']}: {item['avg_ms']:.2f}ms avg "
                f"({item['share_pct']:.2f}%, {item['classification']})"
            )
    if vector_info:
        print(
            f"  Embedding:       {vector_info['embedding_model']} "
            f"({vector_info['embedding_provider']})"
        )

    if out_file is not None:
        payload["benchmark_io_ms"]["json_write_ms"] = 0.0
        serialize_start = perf_counter()
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        payload["benchmark_io_ms"]["json_serialize_ms"] = round(
            (perf_counter() - serialize_start) * 1000.0,
            2,
        )
        write_start = perf_counter()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(serialized, encoding="utf-8")
        payload["benchmark_io_ms"]["json_write_ms"] = round(
            (perf_counter() - write_start) * 1000.0,
            2,
        )
        payload["totals_ms"]["total_io_ms"] = round(
            load_data_ms
            + float(payload["benchmark_io_ms"].get("json_serialize_ms", 0.0))
            + float(payload["benchmark_io_ms"].get("json_write_ms", 0.0)),
            2,
        )
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved:           {out_file}")

    if gc_was_enabled and not gc.isenabled():
        gc.enable()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Sphere Memory CLI with CloneMem.")
    parser.add_argument(
        "data_root",
        nargs="?",
        type=Path,
        default=default_data_root(),
        help="Path to the CloneMem releases directory",
    )
    parser.add_argument("--context-len", choices=["100k", "500k", "all"], default="all")
    parser.add_argument("--language", choices=["en", "zh", "all"], default="all")
    parser.add_argument("--mode", choices=["vector", "bm25", "evidence", "activation", "hybrid"], default="evidence")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N sample files")
    parser.add_argument("--question-limit", type=int, default=0, help="Run only the first N questions per sample")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--question-id-allowlist", type=Path, default=None)
    parser.add_argument("--sample-id-allowlist", type=Path, default=None)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--rerank-mode", choices=["rule", "hybrid", "cross_encoder"], default="rule")
    parser.add_argument("--shell", type=int, default=2)
    parser.add_argument("--sector", default="knowledge")
    parser.add_argument("--zone", default="clonemem")
    parser.add_argument("--chunk-pool", type=int, default=400)
    parser.add_argument("--cross-encoder", action="store_true", help="Enable cross-encoder reranking in evidence pipeline")
    parser.add_argument(
        "--object-top-k",
        type=int,
        default=4,
        help="Number of structured evidence objects to keep for profiling/completion.",
    )
    parser.add_argument(
        "--support-top-k",
        type=int,
        default=4,
        help="Number of supporting context chunks to expand for profiling/completion.",
    )
    parser.add_argument(
        "--cognitive-top-k",
        type=int,
        default=0,
        help="Optional cognitive expansion budget for profiling.",
    )
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
        question_limit_tag = f"_q{args.question_limit}" if args.question_limit else ""
        out_file = (
            ROOT
            / "benchmarks"
            / f"results_clonemem_{args.context_len}_{args.language}_{args.mode}{rerank_tag}_top{args.top_k}{limit_tag}{question_limit_tag}_{stamp}.json"
        )

    run_benchmark(
        data_root=args.data_root,
        context_len=args.context_len,
        language=args.language,
        mode=args.mode,
        top_k=args.top_k,
        limit=args.limit,
        question_limit=args.question_limit,
        rerank_mode=args.rerank_mode,
        shell=args.shell,
        sector=args.sector,
        zone=args.zone,
        chunk_pool=args.chunk_pool,
        out_file=out_file,
        use_cross_encoder=args.cross_encoder,
        object_top_k=args.object_top_k,
        support_top_k=args.support_top_k,
        cognitive_top_k=args.cognitive_top_k,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        question_id_allowlist_path=args.question_id_allowlist,
        sample_id_allowlist_path=args.sample_id_allowlist,
        max_questions=args.max_questions,
        resume_existing=args.resume_existing,
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
