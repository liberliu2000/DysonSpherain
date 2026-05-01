#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_BASELINES = [
    "dense_only_minilm",
    "dense_only_bge_or_e5",
    "bm25",
    "dense_bm25_rrf",
    "cross_encoder_or_llm_reranker_subset",
    "oracle_candidate",
    "oracle_parent",
    "oracle_segment",
    "dysonspherain_full",
]
BLOCKED_BASELINE_REASONS = {
    "dense_only_bge_or_e5": (
        "blocked_model_unavailable: no local BGE/E5 embedding artifact exists under the formal "
        "results root; local_hash fallback is disallowed for formal baseline claims"
    ),
    "cross_encoder_or_llm_reranker_subset": (
        "blocked_model_unavailable: no artifact-backed cross-encoder or LLM reranker subset run "
        "exists under the formal results root; pending rows must not be hand-filled"
    ),
}
MAX_METRICS_BYTES = 25 * 1024 * 1024

PRIMARY_METRICS = [
    "recall_any@5",
    "recall_any@10",
    "recall@10",
    "Recall@10",
    "recall_frac@10",
    "final_recall@10",
    "ndcg@10",
    "NDCG@10",
    "ndcg_any@10",
    "final_ndcg@10",
    "candidate_recall@100",
    "oracle_candidate_recall@100",
    "oracle_parent_recall@100",
    "oracle_recall@1",
    "oracle_recall@5",
    "oracle_recall@10",
]

SKIP_DIR_NAMES = {".git", ".venv", ".venv312", "__pycache__", ".cache", "cache", "chroma", "vector_store", "workspace", "workspaces"}
EXPECTED_FULL_QUESTION_COUNTS = {
    "longmemeval": 500,
    "locomo": 1986,
    "knowme": 1010,
    "clonemem": 2374,
}


@dataclass
class BaselineRecord:
    baseline: str
    benchmark: str
    status: str
    metrics_path: str | None = None
    artifact_dir: str | None = None
    question_count: int | None = None
    sample_count: int | None = None
    elapsed_seconds: float | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    fallback_in_use: bool | None = None
    metrics: dict[str, Any] | None = None
    warnings: list[str] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _classify_baseline(path: Path, payload: dict[str, Any]) -> str:
    text = " ".join(part.lower() for part in path.parts)
    method = str(payload.get("baseline") or payload.get("method") or payload.get("run_type") or "").lower()
    mode = str(payload.get("mode") or "").lower()
    haystack = f"{text} {method}"
    if mode == "vector":
        return "dense_only_minilm"
    if mode == "bm25":
        return "bm25"
    if mode == "evidence":
        return "dysonspherain_full"
    if "vector" in haystack or "dense_only" in haystack or "dense-only" in haystack:
        if "bge" in haystack or "e5" in haystack:
            return "dense_only_bge_or_e5"
        return "dense_only_minilm"
    if "bm25" in method and ("rrf" in method or "hybrid" in method):
        return "dense_bm25_rrf"
    if method == "bm25" or method == "bm25_only" or method == "lexical_bm25":
        return "bm25"
    if "cross_encoder" in haystack or "cross-encoder" in haystack or "llm_reranker" in haystack:
        return "cross_encoder_or_llm_reranker_subset"
    if "oracle_candidate" in haystack:
        return "oracle_candidate"
    if "oracle_parent" in haystack:
        return "oracle_parent"
    if "oracle_segment" in haystack:
        return "oracle_segment"
    return "dysonspherain_full"


def _benchmark_name(path: Path, payload: dict[str, Any]) -> str:
    value = payload.get("benchmark") or payload.get("dataset")
    if value:
        return str(value).lower()
    for part in reversed(path.parts):
        lower = part.lower()
        for name in ("longmemeval", "locomo", "knowme", "clonemem", "convomem"):
            if name in lower:
                return name
    return "unknown"


def _embedding_info(payload: dict[str, Any]) -> dict[str, Any]:
    info = payload.get("embedding_info")
    if not isinstance(info, dict):
        info = payload.get("embedding") if isinstance(payload.get("embedding"), dict) else {}
    info = dict(info or {})
    if "embedding_provider" not in info and "provider" in info:
        info["embedding_provider"] = info.get("provider")
    if "embedding_model" not in info and "model" in info:
        info["embedding_model"] = info.get("model")
    return info


def _extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    return {key: value for key in PRIMARY_METRICS if (value := _find_metric(metrics, key)) is not None}


def _find_metric(payload: Any, metric: str) -> Any:
    if not isinstance(payload, dict):
        return None
    if metric in payload:
        return payload[metric]
    lower_metric = metric.lower()
    for key, value in payload.items():
        if str(key).lower() == lower_metric:
            return value
    for preferred in ("session", "segment", "turn", "dialog"):
        nested = payload.get(preferred)
        if isinstance(nested, dict):
            found = _find_metric(nested, metric)
            if found is not None:
                return found
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_metric(value, metric)
            if found is not None:
                return found
    return None


def record_from_metrics(path: Path) -> BaselineRecord:
    size = path.stat().st_size
    if size > MAX_METRICS_BYTES:
        return BaselineRecord(
            baseline=_classify_baseline(path, {}),
            benchmark=_benchmark_name(path, {}),
            status="oversized_skipped",
            metrics_path=str(path),
            artifact_dir=str(path.parent),
            warnings=[f"metrics.json is {size} bytes; skipped to avoid loading giant per-query artifact"],
        )
    payload = _load_json(path)
    embedding = _embedding_info(payload)
    benchmark = _benchmark_name(path, payload)
    baseline = _classify_baseline(path, payload)
    metrics = _extract_metrics(payload)
    fallback = embedding.get("fallback_in_use", payload.get("fallback_in_use"))
    warnings: list[str] = []
    if fallback:
        warnings.append("fallback_in_use=true; exclude from formal baseline tables")
    if not metrics:
        warnings.append("no primary metrics extracted")
    question_count = payload.get("total_question_count") or payload.get("question_count")
    expected_count = EXPECTED_FULL_QUESTION_COUNTS.get(benchmark)
    if isinstance(question_count, int) and expected_count and question_count < expected_count:
        warnings.append(f"sample_or_partial_run=true; question_count={question_count} expected_full={expected_count}")
    return BaselineRecord(
        baseline=baseline,
        benchmark=benchmark,
        status="available",
        metrics_path=str(path),
        artifact_dir=str(path.parent),
        question_count=question_count,
        sample_count=payload.get("sample_count") or payload.get("total_sample_count"),
        elapsed_seconds=payload.get("elapsed_seconds") or payload.get("wall_clock_elapsed_seconds"),
        embedding_provider=embedding.get("embedding_provider"),
        embedding_model=embedding.get("embedding_model"),
        fallback_in_use=bool(fallback) if fallback is not None else None,
        metrics=metrics,
        warnings=warnings,
    )


def iter_metrics_files(root: Path, *, max_depth: int = 3) -> list[Path]:
    root = root.resolve()
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [
                name
                for name in dirnames
                if name not in SKIP_DIR_NAMES and not name.startswith(".") and "cache" not in name.lower()
            ]
        for name in ("metrics.json", "merged_metrics.json", "compact_metrics.json"):
            if name in filenames:
                found.append(current / name)
    return sorted(found)


def _run_scope_priority(record: BaselineRecord) -> tuple[int, int, int, int, int, float]:
    path_text = str(record.metrics_path or "").lower()
    promoted_clonemem_route = "phase5_lexical_anchor_gate_protected_top3_full" in path_text
    full_like = not any(
        marker in path_text
        for marker in (
            "smoke",
            "sample",
            "medium",
            "probe",
            "alpha",
            "evidence_blend",
            "lexical_anchor_gate",
            "supplemental",
            "guard_off",
            "guard-on",
            "guard_on",
        )
    ) or promoted_clonemem_route
    route_promotion = 1 if promoted_clonemem_route else 0
    return (
        1 if record.status == "available" else 0,
        1 if record.fallback_in_use is False else 0,
        1 if full_like else 0,
        route_promotion,
        int(record.question_count or record.sample_count or 0),
        Path(record.metrics_path or "").stat().st_mtime if record.metrics_path else 0.0,
    )


def _prefer_record(previous: BaselineRecord, current: BaselineRecord) -> BaselineRecord:
    if previous.metrics_path and current.metrics_path:
        previous_name = Path(previous.metrics_path).name
        current_name = Path(current.metrics_path).name
        if previous.status != "available" and current.status == "available":
            return current
        if previous.status == "available" and current.status != "available":
            return previous
        if previous_name != "merged_metrics.json" and current_name == "merged_metrics.json" and current.status == "available":
            return current
        if previous_name == "merged_metrics.json" and current_name != "merged_metrics.json" and previous.status == "available":
            return previous
    return current if _run_scope_priority(current) >= _run_scope_priority(previous) else previous


def discover_baselines(results_root: Path) -> list[BaselineRecord]:
    records = [record_from_metrics(path) for path in iter_metrics_files(results_root)]
    latest: dict[tuple[str, str], BaselineRecord] = {}
    for record in records:
        key = (record.benchmark, record.baseline)
        previous = latest.get(key)
        if previous is None:
            latest[key] = record
            continue
        latest[key] = _prefer_record(previous, record)
    benchmarks = sorted({record.benchmark for record in latest.values() if record.benchmark != "unknown"})
    for benchmark in benchmarks:
        for baseline in REQUIRED_BASELINES:
            latest.setdefault(
                (benchmark, baseline),
                BaselineRecord(
                    baseline=baseline,
                    benchmark=benchmark,
                    status="blocked" if baseline in BLOCKED_BASELINE_REASONS else "pending",
                    warnings=[
                        BLOCKED_BASELINE_REASONS.get(baseline, "no artifact-backed metrics.json found"),
                    ],
                ),
            )
    return sorted(latest.values(), key=lambda item: (item.benchmark, item.baseline))


def write_markdown(records: list[BaselineRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline Comparison Table",
        "",
        "This table is artifact-backed. Missing baselines are marked `pending`; no metric is hand-filled.",
        "",
        "| benchmark | baseline | status | primary metrics | q | fallback | artifact | warnings |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for record in records:
        metric_text = ", ".join(f"{k}={v}" for k, v in (record.metrics or {}).items()) or ""
        warnings = "; ".join(record.warnings or [])
        lines.append(
            f"| {record.benchmark} | {record.baseline} | {record.status} | {metric_text} | "
            f"{record.question_count or ''} | {record.fallback_in_use} | {record.metrics_path or ''} | {warnings} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Artifact-backed matched-budget baseline summary.")
    parser.add_argument("--results-root", type=Path, default=Path("../BenchmarkResult"), help="Directory containing benchmark metrics.json artifacts")
    parser.add_argument("--out", type=Path, default=Path("artifacts/baselines"), help="Output artifact directory")
    parser.add_argument("--report", type=Path, default=Path("reports/baseline_comparison_table.md"), help="Markdown report path")
    args = parser.parse_args()

    records = discover_baselines(args.results_root)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dysonspherain.baselines.v1",
        "results_root": str(args.results_root),
        "required_baselines": REQUIRED_BASELINES,
        "records": [asdict(record) for record in records],
        "formal_use_warning": "Use only records with fallback_in_use=false and complete matched-budget metadata for paper tables.",
    }
    (args.out / "baseline_runs.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(records, args.report)
    print(json.dumps({"records": len(records), "out": str(args.out), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
