from __future__ import annotations

import argparse
import json
import os
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


def load_dataset_bundle(dataset_dir: Path) -> dict[str, Any]:
    input_items = json.loads((dataset_dir / "input" / f"{dataset_dir.name}.json").read_text(encoding="utf-8"))
    corpus_items = [
        {
            "corpus_id": str(item["id"]),
            "timestamp": str(item.get("timestamp") or ""),
            "text": build_segment_text(item),
        }
        for item in input_items
        if build_segment_text(item)
    ]
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
            questions.append(
                {
                    "question_id": f"{dataset_dir.name}_{task_name}_{question['id']}",
                    "native_question_id": int(question["id"]),
                    "task_name": task_name,
                    "task_type": TASK_TO_RUNTIME.get(task_name, "qa"),
                    "question": str(question["question"]).strip(),
                    "answer": str(answer.get("answer") or ""),
                    "evidence_ids": normalize_evidence_ids(answer.get("evidence")),
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
) -> dict[str, Any]:
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

    for bundle in bundles:
        dataset_name = str(bundle["dataset_name"])
        corpus_items = list(bundle["corpus_items"])
        questions = list(bundle["questions"])
        if question_limit > 0:
            questions = questions[:question_limit]
        ordered_corpus_ids = [str(item["corpus_id"]) for item in corpus_items]
        corpus_by_node_id = {
            f"bench_{item_index:04d}": item
            for item_index, item in enumerate(corpus_items)
        }
        print(
            f"[dataset {dataset_name}] "
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
                    question_type="knowme",
                    shell=shell,
                    sector=sector,
                    zone=f"{zone}_{dataset_name}",
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
                    elif mode == "evidence":
                        ranked_ids, ranked_items, stage_timing_ms, evidence_profile = rank_evidence(
                            entry["question"],
                            pipeline=workspace.pipeline,
                            ordered_corpus_ids=ordered_corpus_ids,
                            corpus_by_node_id=corpus_by_node_id,
                            task_type=entry["task_type"],
                            top_k=top_k,
                            object_top_k=object_top_k,
                            support_top_k=support_top_k,
                            cognitive_top_k=cognitive_top_k,
                            route_context={
                                "benchmark": "knowme",
                                "task_name": entry["task_name"],
                                "dataset_name": dataset_name,
                            },
                        )
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
                    current_workspace_manager.release(workspace.signature)
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

            results_log.append(
                {
                    "dataset_name": dataset_name,
                    "question_id": entry["question_id"],
                    "native_question_id": entry["native_question_id"],
                    "task_name": entry["task_name"],
                    "task_type": entry["task_type"],
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "evidence_ids": sorted(entry["evidence_ids"]),
                    "metrics": entry_metrics,
                    "stage_timing_ms": stage_timing_ms,
                    "profiling": question_profile,
                    "retrieved_segment_ids": ranked_ids[:top_k],
                    "ranked_items": ranked_items,
                }
            )

            print(
                f"  [q {question_index:04d}/{len(questions):04d}] "
                f"{entry['task_name'][:24]:24} "
                f"R@10={entry_metrics['recall_frac@10']:.2f} "
                f"Ingest={ingest_ms:.0f}ms ReuseSaved={float(ingest_reuse.get('cache_reuse_saved_ms', 0.0)):.0f}ms"
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
    parser.add_argument("--mode", choices=["vector", "evidence", "activation", "hybrid"], default="evidence")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N dataset directories")
    parser.add_argument("--question-limit", type=int, default=0, help="Run only the first N questions per dataset")
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
    )


if __name__ == "__main__":
    main()
