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
    SHARED_EMBEDDING_CACHE_DIR as DEFAULT_SHARED_EMBEDDING_CACHE_DIR,
    classify_bottleneck,
    evaluate_retrieval,
    flatten_numeric_metrics,
    materialize_ingest_profile_for_question,
    rank_evidence,
    rank_hybrid,
    rank_vector,
    summarize_numeric_metrics,
    summarize_question_profile,
)


CATEGORIES = {
    1: "single_hop",
    2: "temporal",
    3: "temporal_inference",
    4: "open_domain",
    5: "adversarial",
}
PREFERENCE_TERMS = {
    "prefer",
    "preferred",
    "prefers",
    "favorite",
    "favourite",
    "like",
    "likes",
    "love",
    "loves",
    "enjoy",
    "enjoys",
    "dislike",
    "avoids",
    "avoid",
    "hate",
    "hates",
}
TEMPORAL_TERMS = {
    "before",
    "after",
    "earlier",
    "later",
    "latest",
    "current",
    "currently",
    "when",
    "timeline",
    "ago",
    "yesterday",
    "today",
    "tomorrow",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "day",
    "days",
}
EVIDENCE_SESSION_RE = re.compile(r"^D(\d+):")


def path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


WORKSPACE_CACHE_ROOT = path_from_env(
    "SPHERE_LOCOMO_CACHE_ROOT",
    ROOT / "benchmarks" / ".cache" / "locomo_workspaces",
)
SHARED_EMBEDDING_CACHE_DIR = path_from_env(
    "SPHERE_LOCOMO_EMBED_CACHE_ROOT",
    DEFAULT_SHARED_EMBEDDING_CACHE_DIR,
)


def dialog_text(dialog: dict[str, Any]) -> str:
    speaker = str(dialog.get("speaker") or "?").strip() or "?"
    text = str(dialog.get("text") or "").strip()
    return f"{speaker}: {text}" if text else f"{speaker}:"


def load_sessions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = sample["conversation"]
    session_summaries = sample.get("session_summary") or {}
    sessions: list[dict[str, Any]] = []
    session_num = 1
    while True:
        session_key = f"session_{session_num}"
        if session_key not in conversation:
            break
        sessions.append(
            {
                "session_num": session_num,
                "session_id": session_key,
                "date": str(conversation.get(f"{session_key}_date_time") or ""),
                "dialogs": list(conversation.get(session_key) or []),
                "summary": str(session_summaries.get(f"{session_key}_summary") or ""),
            }
        )
        session_num += 1
    return sessions


def build_corpus_from_sessions(sessions: list[dict[str, Any]], granularity: str) -> list[dict[str, Any]]:
    corpus_items: list[dict[str, Any]] = []
    for session in sessions:
        session_id = str(session["session_id"])
        timestamp = str(session.get("date") or "")
        dialogs = list(session.get("dialogs") or [])
        if granularity == "session":
            texts = [dialog_text(dialog) for dialog in dialogs if str(dialog.get("text") or "").strip()]
            if not texts:
                continue
            corpus_items.append(
                {
                    "corpus_id": session_id,
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "text": "\n".join(texts),
                }
            )
            continue

        for dialog_index, dialog in enumerate(dialogs, start=1):
            text = str(dialog.get("text") or "").strip()
            if not text:
                continue
            corpus_id = str(dialog.get("dia_id") or f"D{session['session_num']}:{dialog_index}")
            corpus_items.append(
                {
                    "corpus_id": corpus_id,
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "text": dialog_text(dialog),
                }
            )
    return corpus_items


def ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def session_id_from_corpus_id(corpus_id: str) -> str:
    match = EVIDENCE_SESSION_RE.match(corpus_id)
    if match:
        return f"session_{match.group(1)}"
    return corpus_id


def evidence_to_dialog_ids(evidence: list[Any]) -> set[str]:
    return {str(item).strip() for item in evidence if str(item).strip()}


def evidence_to_session_ids(evidence: list[Any]) -> set[str]:
    session_ids: set[str] = set()
    for item in evidence:
        match = EVIDENCE_SESSION_RE.match(str(item).strip())
        if match:
            session_ids.add(f"session_{match.group(1)}")
    return session_ids


def recall_fraction(ranked_ids: list[str], correct_ids: set[str], k: int) -> float:
    if not correct_ids:
        return 1.0
    top_ids = set(ranked_ids[:k])
    return len(top_ids & correct_ids) / len(correct_ids)


def infer_task_type(question: str, category: int) -> str:
    lowered = question.lower()
    if any(term in lowered for term in PREFERENCE_TERMS):
        return "preference_lookup"
    if category in {2, 3} or any(term in lowered for term in TEMPORAL_TERMS):
        return "temporal_reasoning"
    return "qa"


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


def run_benchmark(
    data_file: Path,
    mode: str,
    granularity: str,
    top_k: int,
    limit: int,
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
) -> dict[str, Any]:
    load_start = perf_counter()
    with data_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    load_data_ms = round((perf_counter() - load_start) * 1000.0, 2)

    if limit > 0:
        data = data[:limit]

    metrics_dialog = init_metric_bucket()
    metrics_session = init_metric_bucket()
    timing_metrics: dict[str, list[float]] = defaultdict(list)
    profile_metrics: dict[str, list[float]] = defaultdict(list)
    per_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    results_log: list[dict[str, Any]] = []
    vector_info: dict[str, Any] | None = None
    runtime_config: dict[str, Any] | None = None
    total_questions = 0
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

    cross_encoder_model = None
    if use_cross_encoder:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("Cross-encoder reranking enabled")
        except Exception as exc:
            print(f"Cross-encoder not available: {exc}")

    for conv_index, sample in enumerate(data, start=1):
        sample_id = str(sample.get("sample_id") or f"conv_{conv_index:03d}")
        sessions = load_sessions(sample)
        corpus_items = build_corpus_from_sessions(sessions, granularity=granularity)
        if not corpus_items:
            continue
        qa_pairs = list(sample.get("qa") or [])
        ordered_corpus_ids = [str(item["corpus_id"]) for item in corpus_items]
        corpus_by_node_id = {
            f"bench_{item_index:04d}": item
            for item_index, item in enumerate(corpus_items)
        }
        print(
            f"[conversation {conv_index:02d}/{len(data):02d}] "
            f"{sample_id}: sessions={len(sessions)} docs={len(corpus_items)} qa={len(qa_pairs)}"
        )

        for qa_index, qa in enumerate(qa_pairs, start=1):
            total_questions += 1
            question = str(qa.get("question") or "").strip()
            answer = str(qa.get("answer") or qa.get("adversarial_answer") or "")
            category_id = int(qa.get("category") or 0)
            category_name = CATEGORIES.get(category_id, f"category_{category_id}")
            gold_dialog_ids = evidence_to_dialog_ids(list(qa.get("evidence") or []))
            gold_session_ids = evidence_to_session_ids(list(qa.get("evidence") or []))
            question_task_type = infer_task_type(question, category_id)
            question_profile: dict[str, Any] = {}
            retry_count = 0
            current_workspace_manager = workspace_manager

            while True:
                question_start = perf_counter()
                stage_start = perf_counter()
                workspace, ingest_reuse = current_workspace_manager.acquire(
                    corpus_items=corpus_items,
                    granularity=granularity,
                    question_type="locomo",
                    shell=shell,
                    sector=sector,
                    zone=zone,
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
                if runtime_config is None:
                    runtime_config = {
                        "embedding_model": workspace.config.embedding_model_name,
                        "embed_local_grain": bool(workspace.config.embed_local_grain),
                        "rerank_mode_default": workspace.config.rerank_mode_default,
                        "cross_encoder_model": workspace.config.cross_encoder_model_name,
                        "cross_encoder_requested": bool(use_cross_encoder),
                        "cross_encoder_loaded": bool(cross_encoder_model is not None),
                        "creative_mode": workspace.config.creative_mode_name,
                        "creative_beam_width": int(workspace.config.creative_beam_width),
                        "creative_max_hops": int(workspace.config.creative_max_hops),
                        "creative_neighbors_per_hop": int(workspace.config.creative_neighbors_per_hop),
                        "creative_max_output_paths": int(workspace.config.creative_max_output_paths),
                    }

                workspace.pipeline.cross_encoder = cross_encoder_model
                try:
                    if mode == "vector":
                        ranked_ids, ranked_items, stage_timing_ms = rank_vector(
                            question,
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
                    elif mode == "evidence":
                        ranked_ids, ranked_items, stage_timing_ms, evidence_profile = rank_evidence(
                            question,
                            pipeline=workspace.pipeline,
                            ordered_corpus_ids=ordered_corpus_ids,
                            corpus_by_node_id=corpus_by_node_id,
                            task_type=question_task_type,
                            top_k=top_k,
                            object_top_k=object_top_k,
                            support_top_k=support_top_k,
                            cognitive_top_k=cognitive_top_k,
                            route_context={
                                "benchmark": "locomo",
                                "question_type": category_name,
                                "granularity": granularity,
                            },
                        )
                        question_profile = {
                            "ingest": actual_ingest_profile,
                            "ingest_cached": workspace.ingest_profile,
                            "pipeline": evidence_profile,
                            "reuse": ingest_reuse,
                        }
                    elif mode == "activation":
                        ranked_ids, ranked_items, stage_timing_ms = rank_hybrid(
                            question,
                            activation=workspace.activation,
                            router=workspace.router,
                            reranker=workspace.reranker,
                            ordered_corpus_ids=ordered_corpus_ids,
                            corpus_by_node_id=corpus_by_node_id,
                            task_type=question_task_type,
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
                            question,
                            activation=workspace.activation,
                            router=workspace.router,
                            reranker=workspace.reranker,
                            ordered_corpus_ids=ordered_corpus_ids,
                            corpus_by_node_id=corpus_by_node_id,
                            task_type=question_task_type,
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
                    current_workspace_manager.release(workspace.signature)
                    lowered = str(exc).lower()
                    retryable = "nothing found on disk" in lowered or "creating hnsw segment reader" in lowered
                    if not retryable or retry_count >= 2:
                        raise
                    sleep(0.25 * (retry_count + 1))
                    retry_count += 1
                    retry_cache_root = WORKSPACE_CACHE_ROOT / "_question_retries" / f"{sample_id}_q{qa_index:03d}_try{retry_count}"
                    current_workspace_manager = BenchmarkWorkspaceManager(
                        retry_cache_root,
                        SHARED_EMBEDDING_CACHE_DIR,
                    )
                    print(
                        f"[retry {retry_count}] rebuilding workspace for "
                        f"{sample_id} q{qa_index:03d} after vector store error"
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

            session_ranked_ids = ordered_unique(
                [session_id_from_corpus_id(corpus_id) for corpus_id in ranked_ids]
            )
            entry_metrics = {"dialog": {}, "session": {}}
            for k in KS:
                dialog_frac = recall_fraction(ranked_ids, gold_dialog_ids, k)
                dialog_ra, dialog_rl, dialog_nd = evaluate_retrieval(ranked_ids, gold_dialog_ids, k)
                metrics_dialog[f"recall_frac@{k}"].append(dialog_frac)
                metrics_dialog[f"recall_any@{k}"].append(dialog_ra)
                metrics_dialog[f"recall_all@{k}"].append(dialog_rl)
                metrics_dialog[f"ndcg_any@{k}"].append(dialog_nd)
                entry_metrics["dialog"][f"recall_frac@{k}"] = dialog_frac
                entry_metrics["dialog"][f"recall_any@{k}"] = dialog_ra
                entry_metrics["dialog"][f"ndcg_any@{k}"] = dialog_nd

                session_frac = recall_fraction(session_ranked_ids, gold_session_ids, k)
                session_ra, session_rl, session_nd = evaluate_retrieval(
                    session_ranked_ids,
                    gold_session_ids,
                    k,
                )
                metrics_session[f"recall_frac@{k}"].append(session_frac)
                metrics_session[f"recall_any@{k}"].append(session_ra)
                metrics_session[f"recall_all@{k}"].append(session_rl)
                metrics_session[f"ndcg_any@{k}"].append(session_nd)
                entry_metrics["session"][f"recall_frac@{k}"] = session_frac
                entry_metrics["session"][f"recall_any@{k}"] = session_ra
                entry_metrics["session"][f"ndcg_any@{k}"] = session_nd

            per_category[category_name]["dialog_recall_frac@5"].append(
                entry_metrics["dialog"]["recall_frac@5"]
            )
            per_category[category_name]["dialog_recall_frac@10"].append(
                entry_metrics["dialog"]["recall_frac@10"]
            )
            per_category[category_name]["session_recall_frac@5"].append(
                entry_metrics["session"]["recall_frac@5"]
            )
            per_category[category_name]["session_recall_frac@10"].append(
                entry_metrics["session"]["recall_frac@10"]
            )
            per_category[category_name]["session_ndcg_any@10"].append(
                entry_metrics["session"]["ndcg_any@10"]
            )

            results_log.append(
                {
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "category": category_id,
                    "category_name": category_name,
                    "task_type": question_task_type,
                    "question": question,
                    "answer": answer,
                    "gold_dialog_ids": sorted(gold_dialog_ids),
                    "gold_session_ids": sorted(gold_session_ids),
                    "metrics": entry_metrics,
                    "stage_timing_ms": stage_timing_ms,
                    "profiling": question_profile,
                    "retrieved_dialog_ids": ranked_ids[:top_k],
                    "retrieved_session_ids": session_ranked_ids[:top_k],
                    "ranked_items": ranked_items,
                }
            )

            print(
                f"  [q {qa_index:03d}/{len(qa_pairs):03d}] "
                f"{category_name:18} "
                f"D-R@10={entry_metrics['dialog']['recall_frac@10']:.2f} "
                f"S-R@10={entry_metrics['session']['recall_frac@10']:.2f} "
                f"Ingest={ingest_ms:.0f}ms ReuseSaved={float(ingest_reuse.get('cache_reuse_saved_ms', 0.0)):.0f}ms"
            )
            current_workspace_manager.release(workspace.signature)

    workspace_manager.close_all()

    elapsed_seconds = (datetime.now() - start_time).total_seconds()
    dialog_summary = summarize_metric_bucket(metrics_dialog)
    session_summary = summarize_metric_bucket(metrics_session)
    per_category_summary = {
        category_name: {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in metric_map.items()
        }
        for category_name, metric_map in per_category.items()
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
        "data_file": str(data_file),
        "mode": mode,
        "rerank_mode": rerank_mode if mode == "hybrid" else None,
        "requested_rerank_mode": rerank_mode,
        "rerank_mode_active": mode == "hybrid",
        "granularity": granularity,
        "conversation_count": len(data),
        "question_count": total_questions,
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
        "category_names": CATEGORIES,
        "vector_info": vector_info,
        "runtime_config": runtime_config or {},
        "metrics": {
            "dialog": dialog_summary,
            "session": session_summary,
        },
        "per_category": per_category_summary,
        "stage_timing_ms": timing_summary,
        "profiling_summary": profile_summary,
        "bottlenecks": bottlenecks[:12],
        "results": results_log,
    }

    print("\nSummary")
    print(f"  Conversations:   {payload['conversation_count']}")
    print(f"  Questions:       {payload['question_count']}")
    print(f"  Mode:            {mode}")
    print(f"  Granularity:     {granularity}")
    print(f"  Dialog R@10:     {dialog_summary['recall_frac@10']:.4f}")
    print(f"  Session R@10:    {session_summary['recall_frac@10']:.4f}")
    print(f"  Session NDCG@10: {session_summary['ndcg_any@10']:.4f}")
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
    parser = argparse.ArgumentParser(description="Benchmark Sphere Memory CLI with LoCoMo.")
    parser.add_argument("data_file", type=Path, help="Path to locomo10.json")
    parser.add_argument("--mode", choices=["vector", "evidence", "activation", "hybrid"], default="evidence")
    parser.add_argument("--granularity", choices=["session", "dialog"], default="session")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N conversations")
    parser.add_argument("--rerank-mode", choices=["rule", "hybrid", "cross_encoder"], default="rule")
    parser.add_argument("--shell", type=int, default=2)
    parser.add_argument("--sector", default="knowledge")
    parser.add_argument("--zone", default="locomo")
    parser.add_argument("--chunk-pool", type=int, default=400)
    parser.add_argument("--cross-encoder", action="store_true", help="Enable cross-encoder reranking in evidence pipeline")
    parser.add_argument("--object-top-k", type=int, default=4, help="Number of structured evidence objects to keep for profiling/completion.")
    parser.add_argument("--support-top-k", type=int, default=4, help="Number of supporting context chunks to expand for profiling/completion.")
    parser.add_argument("--cognitive-top-k", type=int, default=0, help="Optional cognitive expansion budget for profiling.")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_file = args.out
    if out_file is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rerank_tag = f"_{args.rerank_mode}" if args.mode == "hybrid" else ""
        limit_tag = f"_limit{args.limit}" if args.limit else ""
        out_file = ROOT / "benchmarks" / f"results_locomo_{args.mode}{rerank_tag}_{args.granularity}_top{args.top_k}{limit_tag}_{stamp}.json"

    run_benchmark(
        data_file=args.data_file,
        mode=args.mode,
        granularity=args.granularity,
        top_k=args.top_k,
        limit=args.limit,
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
    )


if __name__ == "__main__":
    main()
