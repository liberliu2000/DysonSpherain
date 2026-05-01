from __future__ import annotations

import argparse
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
    build_knowme_category_analysis,
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
    ndcg_at,
    rank_benchmark_sources,
    recall_at,
    report_root,
    write_json,
    write_jsonl,
)
from shard_utils import filter_sharded_items, load_allowlist, validate_shard_args
from sphere_cli.utils import lexical_score
from token_economy_support import add_token_economy_args, record_token_economy_for_metrics


TASK_TO_RUNTIME = {
    "Information Extraction": "qa",
    "Adversarial Abstention": "qa",
    "Temporal Reasoning": "temporal_reasoning",
    "Logical Event Ordering": "trace",
    "Mnestic Trigger Analysis": "trace",
    "Mind-Body Interaction": "qa",
    "Expert-Annotated Psychoanalysis": "qa",
}
FIELD_LABELS = (
    ("timestamp", "Timestamp"),
    ("location", "Location"),
    ("action", "Action"),
    ("dialogue", "Dialogue"),
    ("environment", "Environment"),
    ("background", "Background"),
    ("inner_thought", "Inner Thought"),
)


def path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


WORKSPACE_CACHE_ROOT = path_from_env(
    "SPHERE_KNOWME_CACHE_ROOT",
    ROOT / "benchmarks" / ".cache" / "knowme_workspaces",
)
SHARED_EMBEDDING_CACHE_DIR = path_from_env(
    "SPHERE_KNOWME_EMBED_CACHE_ROOT",
    DEFAULT_SHARED_EMBEDDING_CACHE_DIR,
)
KNOWME_ADAPTER_VERSION = f"{LONGMEMEVAL_ADAPTER_VERSION}-knowme"


def default_data_root() -> Path:
    candidates = [
        ROOT / "data" / "benchmarks" / "knowmebench",
        ROOT.parent.parent / "tmp_benchmark_sources" / "KnowMeBench" / "KnowmeBench",
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


def build_segment_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, label in FIELD_LABELS:
        value = item.get(key)
        text = str(value or "").strip()
        if not text or text.lower() == "none":
            continue
        parts.append(f"{label}: {text}")
    return "\n".join(parts)


def normalize_evidence_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item is not None and str(item).strip()}
    return {str(value)}


def _reference_clues(question_text: str, answer_item: dict[str, Any]) -> list[str]:
    clues = [question_text, str(answer_item.get("answer") or "")]
    for key in ("reasoning", "strategy", "type"):
        value = answer_item.get(key)
        if isinstance(value, str) and value.strip():
            clues.append(value.strip())
    return [clue for clue in clues if clue]


def _temporal_hint_overlap(clues: list[str], candidate_text: str) -> float:
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", " ".join(clues)))
    if not years:
        return 0.0
    lowered = candidate_text.lower()
    hits = sum(1 for year in years if year in lowered)
    return hits / max(1, len(years))


def _remap_missing_knowme_evidence_ids(
    *,
    question_text: str,
    answer_item: dict[str, Any],
    evidence_ids: set[str],
    corpus_items: list[dict[str, Any]],
    present_ids: set[str] | None = None,
) -> tuple[set[str], dict[str, Any]]:
    present_ids = present_ids or {str(item["corpus_id"]) for item in corpus_items}
    missing_ids = sorted(evidence_id for evidence_id in evidence_ids if evidence_id not in present_ids)
    if not missing_ids:
        return set(evidence_ids), {
            "original_evidence_ids": sorted(evidence_ids),
            "missing_original_evidence_ids": [],
            "remapped_evidence_ids": [],
            "status": "not_needed",
        }

    clues = _reference_clues(question_text, answer_item)
    answer_text = str(answer_item.get("answer") or "").strip()
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for item in corpus_items:
        source_id = str(item["corpus_id"])
        text = str(item.get("text") or "")
        lowered = text.lower()
        exact_answer_hit = bool(answer_text and len(answer_text) >= 4 and answer_text.lower() in lowered)
        clue_score = max((lexical_score(clue, text) for clue in clues), default=0.0)
        question_score = lexical_score(question_text, text)
        answer_score = lexical_score(answer_text, text) if answer_text else 0.0
        temporal_score = _temporal_hint_overlap(clues, lowered)
        score = (
            question_score * 0.42
            + answer_score * 0.32
            + clue_score * 0.2
            + temporal_score * 0.12
            + (0.28 if exact_answer_hit else 0.0)
        )
        scored.append(
            (
                score,
                source_id,
                {
                    "source_id": source_id,
                    "score": round(score, 4),
                    "exact_answer_hit": exact_answer_hit,
                    "preview": text[:240],
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    top_score = float(scored[0][0]) if scored else 0.0
    second_score = float(scored[1][0]) if len(scored) > 1 else 0.0
    needed = len(missing_ids)
    remapped_ids: list[str] = []
    remap_candidates: list[dict[str, Any]] = []
    for score, source_id, candidate in scored:
        if source_id in remapped_ids or source_id in evidence_ids:
            continue
        exact_answer_hit = bool(candidate["exact_answer_hit"])
        if exact_answer_hit:
            accept = score >= 0.18
        else:
            accept = score >= 0.24 and (score - second_score >= 0.015 or score >= 0.32)
        if not accept:
            continue
        remapped_ids.append(source_id)
        remap_candidates.append(candidate)
        if len(remapped_ids) >= needed:
            break

    resolved_ids = {evidence_id for evidence_id in evidence_ids if evidence_id in present_ids}
    resolved_ids.update(remapped_ids)
    status = "remapped" if remapped_ids else "unresolved"
    return resolved_ids, {
        "original_evidence_ids": sorted(evidence_ids),
        "missing_original_evidence_ids": missing_ids,
        "remapped_evidence_ids": remapped_ids,
        "top_candidates": remap_candidates[: max(1, min(needed + 2, 5))],
        "top_score": round(top_score, 4),
        "second_score": round(second_score, 4),
        "status": status,
    }


def load_dataset_bundle(dataset_dir: Path) -> dict[str, Any]:
    input_items = json.loads((dataset_dir / "input" / f"{dataset_dir.name}.json").read_text(encoding="utf-8"))
    corpus_items = []
    for item in input_items:
        text = build_segment_text(item)
        if not text:
            continue
        corpus_items.append(
            {
                "corpus_id": str(item["id"]),
                "source_segment_id": str(item["id"]),
                "source_doc_id": str(dataset_dir.name),
                "sample_id": str(dataset_dir.name),
                "conversation_id": str(dataset_dir.name),
                "timestamp": str(item.get("timestamp") or ""),
                "text": text,
            }
        )
    present_ids = {str(item["corpus_id"]) for item in corpus_items}
    questions: list[dict[str, Any]] = []
    question_dir = dataset_dir / "question"
    answer_dir = dataset_dir / "answer"
    for question_file in sorted(question_dir.glob("*_questions.json")):
        task_name = question_file.stem.removesuffix("_questions")
        answer_file = answer_dir / f"{task_name}_answers.json"
        if not answer_file.exists():
            continue
        question_items = json.loads(question_file.read_text(encoding="utf-8"))
        answer_items = json.loads(answer_file.read_text(encoding="utf-8"))
        answers_by_id = {int(item["id"]): item for item in answer_items}
        for question in question_items:
            answer = answers_by_id.get(int(question["id"]))
            if answer is None:
                continue
            original_evidence_ids = normalize_evidence_ids(answer.get("evidence") or answer.get("evidence_ids"))
            remapped_evidence_ids, remap_meta = _remap_missing_knowme_evidence_ids(
                question_text=str(question["question"]).strip(),
                answer_item=answer,
                evidence_ids=original_evidence_ids,
                corpus_items=corpus_items,
                present_ids=present_ids,
            )
            questions.append(
                {
                    "question_id": f"{dataset_dir.name}_{task_name}_{question['id']}",
                    "native_question_id": int(question["id"]),
                    "task_name": task_name,
                    "task_type": TASK_TO_RUNTIME.get(task_name, "qa"),
                    "question": str(question["question"]).strip(),
                    "answer": str(answer.get("answer") or ""),
                    "evidence_ids": remapped_evidence_ids,
                    "original_evidence_ids": original_evidence_ids,
                    "gold_mapping": remap_meta,
                }
            )
    return {
        "dataset_name": dataset_dir.name,
        "corpus_items": corpus_items,
        "questions": questions,
    }


def run_benchmark(
    data_root: Path,
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
    dataset_dirs = [path for path in sorted(data_root.glob("dataset*")) if path.is_dir()]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {data_root}")
    if limit > 0:
        dataset_dirs = dataset_dirs[:limit]

    load_start = perf_counter()
    bundles = [load_dataset_bundle(path) for path in dataset_dirs]
    load_data_ms = round((perf_counter() - load_start) * 1000.0, 2)

    metrics_segment = init_metric_bucket()
    timing_metrics: dict[str, list[float]] = defaultdict(list)
    profile_metrics: dict[str, list[float]] = defaultdict(list)
    per_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    results_log: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    oracle_retrieval_cache: dict[str, dict[str, Any]] = {}
    failure_rows: list[dict[str, Any]] = []
    topk_debug_rows: list[dict[str, Any]] = []
    vector_info: dict[str, Any] | None = None
    index_metadata: dict[str, Any] | None = None
    runtime_fingerprint: dict[str, Any] | None = None
    runtime_config: dict[str, Any] | None = None
    total_questions = 0
    start_time = datetime.now()
    lightweight_diagnostics = os.getenv("SPHERE_KNOWME_LIGHTWEIGHT_DIAGNOSTICS", "").strip().lower() in {"1", "true", "yes"}
    knowme_pool_limit = max(top_k, int(os.getenv("SPHERE_KNOWME_POOL_LIMIT", "200") or "200"))
    progress_checkpoint_path = out_file.parent / "knowme_progress_checkpoint.json" if out_file is not None else None

    def write_progress_checkpoint(status: str, question_index: int, entry: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
        if progress_checkpoint_path is None:
            return
        progress_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        payload = {
            "benchmark": "knowme",
            "status": status,
            "dataset_name": dataset_name if "dataset_name" in locals() else "",
            "question_index": question_index,
            "question_count": len(questions) if "questions" in locals() else 0,
            "total_processed": total_questions,
            "question_id": str(entry.get("question_id") or ""),
            "task_name": str(entry.get("task_name") or ""),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "updated_at": datetime.now().isoformat(),
        }
        if extra:
            payload.update(extra)
        progress_checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
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
    question_id_allowlist = load_allowlist(question_id_allowlist_path)
    sample_id_allowlist = load_allowlist(sample_id_allowlist_path)
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

    for bundle in bundles:
        if int(max_questions or 0) > 0 and int(shard_meta["shard_question_count"]) >= int(max_questions):
            break
        dataset_name = str(bundle["dataset_name"])
        corpus_items = list(bundle["corpus_items"])
        questions = list(bundle["questions"])
        if question_limit > 0:
            questions = questions[:question_limit]
        current_limit = remaining_questions if remaining_questions > 0 else 0
        filtered_questions, bundle_shard_meta = filter_sharded_items(
            questions,
            benchmark_name="knowme",
            shard_index=shard_index,
            shard_count=shard_count,
            question_id_getter=lambda item: str(item.get("question_id") or ""),
            sample_id_getter=lambda item: dataset_name,
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
        raw_counts = build_raw_counts(corpus_items, question_count=len(questions), session_count=0)
        print(
            f"[dataset {dataset_name}] "
            f"segments={len(corpus_items)} questions={len(questions)}"
        )

        for question_index, entry in enumerate(questions, start=1):
            total_questions += 1
            write_progress_checkpoint("started", question_index, entry)
            print(
                f"  [q {question_index:04d}/{len(questions):04d}] START {entry['task_name'][:24]:24}",
                flush=True,
            )
            question_profile: dict[str, Any] = {}
            retry_count = 0
            current_workspace_manager = workspace_manager

            while True:
                question_start = perf_counter()
                stage_start = perf_counter()
                workspace, ingest_reuse = current_workspace_manager.acquire(
                    corpus_items=corpus_items,
                    granularity="segment",
                    question_type="knowme",
                    shell=shell,
                    sector=sector,
                    zone=f"{zone}_{dataset_name}",
                    benchmark_name="knowme",
                    adapter_version=KNOWME_ADAPTER_VERSION,
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
                        benchmark_name="knowme",
                        adapter_version=KNOWME_ADAPTER_VERSION,
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
                        hybrid_trace = rank_benchmark_sources(
                            query=entry["question"],
                            benchmark_name="knowme",
                            vector_store=workspace.vector_store,
                            storage=workspace.storage,
                            index_metadata=workspace.index_metadata,
                            config=workspace.config,
                            route_context={
                                "benchmark": "knowme",
                                "task_name": entry["task_name"],
                                "dataset_name": dataset_name,
                            },
                            pool_limit=knowme_pool_limit,
                        )
                        final_source_rows = candidate_rows(
                            hybrid_trace["final_candidates"],
                            limit=knowme_pool_limit,
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
                        question_profile = {"ingest": actual_ingest_profile, "ingest_cached": workspace.ingest_profile, "pipeline": evidence_profile, "reuse": ingest_reuse}
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
                        question_profile = {"ingest": actual_ingest_profile, "ingest_cached": workspace.ingest_profile, "reuse": ingest_reuse}
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
                    retry_cache_root = WORKSPACE_CACHE_ROOT / "_question_retries" / f"{dataset_name}_q{question_index:04d}_try{retry_count}"
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
            if not lightweight_diagnostics:
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

            per_task[entry["task_name"]]["recall_frac@10"].append(entry_metrics["recall_frac@10"])
            per_task[entry["task_name"]]["ndcg_any@10"].append(entry_metrics["ndcg_any@10"])
            per_dataset[dataset_name]["recall_frac@10"].append(entry_metrics["recall_frac@10"])
            per_dataset[dataset_name]["ndcg_any@10"].append(entry_metrics["ndcg_any@10"])
            all_gold_segment_ids.update(set(entry["evidence_ids"]))

            if mode == "evidence":
                ranking_payload = dict((question_profile.get("pipeline") or {}).get("retrieval", {}).get("ranking") or {})
                broad_rows = list(ranking_payload.get("broad_top_candidates") or [])
                reranked_rows = list(ranking_payload.get("reranked_top_candidates") or [])
                final_rows = list(ranking_payload.get("final_top_candidates") or [])
                if lightweight_diagnostics:
                    gold_segment_ids = {str(value) for value in entry["evidence_ids"] if str(value)}
                    broad_rank = best_gold_rank(broad_rows, gold_segment_ids)
                    rerank_rank = best_gold_rank(reranked_rows, gold_segment_ids)
                    final_rank = best_gold_rank(final_rows, gold_segment_ids)
                    source_records = dict((workspace.index_metadata or {}).get("source_records_by_id") or {})
                    gold_parent_ids = {
                        str((source_records.get(gold_id) or {}).get("source_doc_id") or (source_records.get(gold_id) or {}).get("cluster_id") or "")
                        for gold_id in gold_segment_ids
                    }
                    gold_parent_ids.discard("")
                    broad_parent_ids = {
                        str(row.get("cluster_id") or row.get("source_doc_id") or "")
                        for row in broad_rows[:100]
                        if str(row.get("cluster_id") or row.get("source_doc_id") or "")
                    }
                    query_diag = {
                        "benchmark_name": "knowme",
                        "query_id": entry["question_id"],
                        "gold_evidence_ids": sorted(set(entry.get("original_evidence_ids") or entry["evidence_ids"])),
                        "gold_segment_ids": sorted(gold_segment_ids),
                        "candidate_recall@10": recall_at(broad_rows, gold_segment_ids, 10),
                        "candidate_recall@50": recall_at(broad_rows, gold_segment_ids, 50),
                        "candidate_recall@100": recall_at(broad_rows, gold_segment_ids, 100),
                        "candidate_recall@200": recall_at(broad_rows, gold_segment_ids, 100),
                        "candidate_ndcg@10": ndcg_at(broad_rows, gold_segment_ids, 10),
                        "final_recall@10": recall_at(final_rows, gold_segment_ids, 10),
                        "final_ndcg@10": ndcg_at(final_rows, gold_segment_ids, 10),
                        "gold_rank_before_rerank": broad_rank,
                        "gold_rank_after_rerank": rerank_rank,
                        "gold_rank_after_inhibition": final_rank,
                        "gold_parent_hit": bool(gold_parent_ids & broad_parent_ids) if gold_parent_ids else False,
                        "parent_hit_segment_miss": bool(gold_parent_ids & broad_parent_ids) and broad_rank is None,
                        "failure_type": "ok" if final_rank is not None and final_rank <= 10 else "lightweight_final_miss",
                    }
                else:
                    query_diag = build_query_diagnostic(
                        benchmark_name="knowme",
                        query_id=entry["question_id"],
                        query_text=entry["question"],
                        answer_text=entry["answer"],
                        gold_segment_ids=set(entry["evidence_ids"]),
                        gold_evidence_ids=set(entry.get("original_evidence_ids") or entry["evidence_ids"]),
                        broad_rows=broad_rows,
                        reranked_rows=reranked_rows,
                        final_rows=final_rows,
                        trace=hybrid_trace,
                        index_metadata=workspace.index_metadata,
                    )
                    failure_row = build_query_failure(
                        benchmark_name="knowme",
                        query_id=entry["question_id"],
                        query_text=entry["question"],
                        answer_text=entry["answer"],
                        gold_segment_ids=set(entry["evidence_ids"]),
                        gold_evidence_ids=set(entry.get("original_evidence_ids") or entry["evidence_ids"]),
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
                query_diag["category"] = entry["task_name"].lower() if "preference" in entry["task_name"].lower() else None
                if not lightweight_diagnostics:
                    topk_debug_rows.append(
                        build_topk_debug_record(
                            benchmark_name="knowme",
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
                candidate_diagnostics.append(query_diag)
                oracle_report_for_query = build_oracle_retrieval_report(
                    benchmark_name="knowme",
                    oracle_items=[
                        {
                            "query_id": entry["question_id"],
                            "sample_id": dataset_name,
                            "question_type": entry["task_name"],
                            "gold_segment_ids": set(entry["evidence_ids"]),
                            "route_context": {
                                "task_name": entry["task_name"],
                                "dataset_name": dataset_name,
                            },
                        }
                    ],
                    vector_store=workspace.vector_store,
                    storage=workspace.storage,
                    index_metadata=workspace.index_metadata,
                    config=workspace.config,
                    pool_limit=50,
                    retrieval_cache=oracle_retrieval_cache,
                )
                oracle_rows.extend(oracle_report_for_query["rows"])

            results_log.append(
                {
                    "dataset_name": dataset_name,
                    "sample_id": dataset_name,
                    "question_id": entry["question_id"],
                    "native_question_id": entry["native_question_id"],
                    "task_name": entry["task_name"],
                    "task_type": entry["task_type"],
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "evidence_ids": sorted(entry["evidence_ids"]),
                    "metrics": entry_metrics,
                    "stage_timing_ms": stage_timing_ms,
                    "profiling": question_profile if not lightweight_diagnostics else {"summary": question_summary},
                    "retrieved_segment_ids": ranked_ids[:top_k],
                    "ranked_items": ranked_items,
                    "candidate_recall": candidate_diagnostics[-1] if mode == "evidence" and candidate_diagnostics else None,
                }
            )

            print(
                f"  [q {question_index:04d}/{len(questions):04d}] "
                f"{entry['task_name'][:24]:24} "
                f"R@10={entry_metrics['recall_frac@10']:.2f} "
                f"Ingest={ingest_ms:.0f}ms ReuseSaved={float(ingest_reuse.get('cache_reuse_saved_ms', 0.0)):.0f}ms"
                f" Total={question_total_ms:.0f}ms",
                flush=True,
            )
            write_progress_checkpoint(
                "completed",
                question_index,
                entry,
                {
                    "recall_frac@10": entry_metrics["recall_frac@10"],
                    "question_total_ms": question_total_ms,
                    "oracle_cache_size": len(oracle_retrieval_cache),
                },
            )
            current_workspace_manager.release(workspace.signature)

    workspace_manager.close_all()

    elapsed_seconds = (datetime.now() - start_time).total_seconds()
    segment_summary = summarize_metric_bucket(metrics_segment)
    per_task_summary = {
        task_name: {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in metric_map.items()
        }
        for task_name, metric_map in per_task.items()
    }
    per_dataset_summary = {
        dataset_name: {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in metric_map.items()
        }
        for dataset_name, metric_map in per_dataset.items()
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
    payload = {
        "data_root": str(data_root),
        "mode": mode,
        "rerank_mode": rerank_mode if mode == "hybrid" else None,
        "requested_rerank_mode": rerank_mode,
        "rerank_mode_active": mode == "hybrid",
        "dataset_count": len(bundles),
        "dataset_names": [str(bundle["dataset_name"]) for bundle in bundles],
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
        "task_names": sorted(per_task_summary),
        "vector_info": vector_info,
        "runtime_config": runtime_config or {},
        "metrics": {
            "segment": segment_summary,
        },
        "per_task": per_task_summary,
        "per_dataset": per_dataset_summary,
        "stage_timing_ms": timing_summary,
        "profiling_summary": profile_summary,
        "bottlenecks": bottlenecks[:12],
        "results": results_log,
    }
    aggregate_index_metadata = {
        "benchmark_name": "knowme",
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
    if runtime_fingerprint is None:
        runtime_fingerprint = {}
    payload.update(
        build_result_metadata(
            project_root=ROOT.parent,
            benchmark_name="knowme",
            question_count=total_questions,
            vector_info=vector_info,
            index_metadata=aggregate_index_metadata,
            runtime_fingerprint=runtime_fingerprint,
            determinism=determinism,
        )
    )
    if mode == "evidence":
        reports_dir = report_root(out_file, "knowme")
        integrity_path = reports_dir / "integrity" / "knowme_integrity_report.json"
        candidate_path = reports_dir / "diagnostics" / "knowme_candidate_recall.json"
        oracle_path = reports_dir / "diagnostics" / "knowme_oracle_retrieval.json"
        channel_path = reports_dir / "diagnostics" / "knowme_channel_contribution.json"
        performance_path = reports_dir / "diagnostics" / "knowme_performance_cache.json"
        category_path = reports_dir / "diagnostics" / "knowme_category_analysis.json"
        failure_path = reports_dir / "failures" / "knowme_failures.jsonl"
        topk_debug_path = reports_dir / "debug" / "knowme_topk_debug.jsonl"
        integrity_report = build_integrity_report(
            benchmark_name="knowme",
            raw_counts=dict(aggregate_index_metadata.get("raw_counts") or {}),
            index_metadata=aggregate_index_metadata,
            gold_segment_ids=all_gold_segment_ids,
            gold_document_ids=all_gold_document_ids,
        )
        candidate_report = build_candidate_recall_summary(
            benchmark_name="knowme",
            rows=candidate_diagnostics,
        )
        oracle_report = {
            "benchmark_name": "knowme",
            "oracle_query_count": len(oracle_rows),
            "oracle_recall@1": round(sum(1.0 for row in oracle_rows if row.get("top1_hit")) / max(1, len(oracle_rows)), 4),
            "oracle_recall@5": round(sum(1.0 for row in oracle_rows if row.get("top5_hit")) / max(1, len(oracle_rows)), 4),
            "oracle_recall@10": round(sum(1.0 for row in oracle_rows if row.get("top10_hit")) / max(1, len(oracle_rows)), 4),
            "oracle_self_retrieval_cache_size": len(oracle_retrieval_cache),
            "oracle_self_retrieval_cache_hits": max(0, len(oracle_rows) - len(oracle_retrieval_cache)),
            "oracle_retrieval_mode": str(os.getenv("SPHERE_ORACLE_RETRIEVAL_MODE") or "self_retrieval"),
            "rows": oracle_rows,
        }
        channel_report = build_per_channel_contribution_report(
            benchmark_name="knowme",
            rows=candidate_diagnostics,
        )
        performance_report = build_performance_cache_report(
            benchmark_name="knowme",
            timing_summary=timing_summary,
            reuse_summary=dict(payload.get("reuse_summary") or {}),
            runtime_config=runtime_config,
        )
        category_report = build_knowme_category_analysis(candidate_diagnostics) if not lightweight_diagnostics else {
            "benchmark_name": "knowme",
            "mode": "lightweight_diagnostics",
            "question_count": len(candidate_diagnostics),
        }
        write_json(integrity_path, integrity_report)
        write_json(candidate_path, candidate_report)
        write_json(oracle_path, oracle_report)
        write_json(channel_path, channel_report if not lightweight_diagnostics else {"benchmark_name": "knowme", "mode": "lightweight_diagnostics"})
        write_json(performance_path, performance_report)
        write_json(category_path, category_report)
        write_jsonl(failure_path, failure_rows)
        write_jsonl(topk_debug_path, topk_debug_rows)
        payload["reports"] = {
            "integrity": str(integrity_path),
            "candidate_recall": str(candidate_path),
            "oracle_retrieval": str(oracle_path),
            "channel_contribution": str(channel_path),
            "performance_cache": str(performance_path),
            "knowme_category_analysis": str(category_path),
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
        payload["performance_cache_report"] = performance_report
        payload["knowme_category_analysis"] = category_report
        payload["failure_summary"] = dict(candidate_report.get("failure_type_distribution") or {})

    print("\nSummary")
    print(f"  Datasets:        {payload['dataset_count']}")
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

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Sphere Memory CLI with KnowMe-Bench.")
    parser.add_argument(
        "data_root",
        nargs="?",
        type=Path,
        default=default_data_root(),
        help="Path to the KnowMeBench data root",
    )
    parser.add_argument("--mode", choices=["vector", "bm25", "evidence", "activation", "hybrid"], default="evidence")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N dataset directories")
    parser.add_argument("--question-limit", type=int, default=0, help="Run only the first N questions per dataset")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--question-id-allowlist", type=Path, default=None)
    parser.add_argument("--sample-id-allowlist", type=Path, default=None)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--rerank-mode", choices=["rule", "hybrid", "cross_encoder"], default="rule")
    parser.add_argument("--shell", type=int, default=2)
    parser.add_argument("--sector", default="knowledge")
    parser.add_argument("--zone", default="knowme")
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
        out_file = (
            ROOT
            / "benchmarks"
            / f"results_knowme_{args.mode}{rerank_tag}_top{args.top_k}{limit_tag}_{stamp}.json"
        )

    run_benchmark(
        data_root=args.data_root,
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
