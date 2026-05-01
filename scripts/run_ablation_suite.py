#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_ABLATIONS = [
    "no_route_conditioned_admission",
    "no_safe_fusion",
    "no_parent_to_segment_selector",
    "no_temporal_routing",
    "no_lexical_probe",
    "no_rerank_guard",
    "no_inhibition",
    "no_benchmark_route_tuning",
    "full_admission",
]
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
]

SKIP_DIR_NAMES = {".git", ".venv", ".venv312", "__pycache__", ".cache", "cache", "chroma", "vector_store", "workspace", "workspaces"}


@dataclass
class AblationRecord:
    ablation: str
    benchmark: str
    status: str
    metrics_path: str | None = None
    artifact_dir: str | None = None
    question_count: int | None = None
    elapsed_seconds: float | None = None
    metrics: dict[str, Any] | None = None
    failure_bucket_delta: dict[str, Any] | None = None
    fallback_in_use: bool | None = None
    warnings: list[str] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _classify_ablation(path: Path, payload: dict[str, Any]) -> str:
    text = " ".join(part.lower() for part in path.parts)
    method = str(payload.get("ablation") or payload.get("method") or payload.get("run_type") or "").lower()
    haystack = f"{text} {method}"
    mapping = {
        "no_route_conditioned_admission": ("no_route", "route_off", "disable_route", "no_route_conditioned"),
        "no_safe_fusion": ("no_safe_fusion", "unsafe_fusion", "safe_fusion_off"),
        "no_parent_to_segment_selector": ("no_parent_to_segment", "parent_selector_off", "no_parent_segment"),
        "no_temporal_routing": ("no_temporal", "temporal_off"),
        "no_lexical_probe": ("no_lexical", "lexical_off", "no_bm25"),
        "no_rerank_guard": ("no_rerank_guard", "rerank_guard_off"),
        "no_inhibition": ("no_inhibition", "inhibition_off"),
        "no_benchmark_route_tuning": ("no_benchmark_route", "route_tuning_off", "benchmark_route_off"),
    }
    for ablation, needles in mapping.items():
        if any(needle in haystack for needle in needles):
            return ablation
    return "full_admission"


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


def _fallback(payload: dict[str, Any]) -> bool | None:
    embedding = payload.get("embedding_info") if isinstance(payload.get("embedding_info"), dict) else {}
    value = embedding.get("fallback_in_use", payload.get("fallback_in_use"))
    return bool(value) if value is not None else None


def record_from_metrics(path: Path) -> AblationRecord:
    size = path.stat().st_size
    if size > MAX_METRICS_BYTES:
        return AblationRecord(
            ablation=_classify_ablation(path, {}),
            benchmark=_benchmark_name(path, {}),
            status="oversized_skipped",
            metrics_path=str(path),
            artifact_dir=str(path.parent),
            warnings=[f"metrics.json is {size} bytes; skipped to avoid loading giant per-query artifact"],
        )
    payload = _load_json(path)
    classified_ablation = _classify_ablation(path, payload)
    mode = str(payload.get("mode") or "").strip().lower()
    explicit_run_type = str(payload.get("ablation") or payload.get("run_type") or "").strip().lower()
    if classified_ablation == "full_admission" and mode in {"bm25", "vector", "artifact_rrf"} and explicit_run_type not in {
        "full_admission",
        "full",
    }:
        return AblationRecord(
            ablation="__skip__",
            benchmark=_benchmark_name(path, payload),
            status="skipped_non_ablation_baseline",
            metrics_path=str(path),
            artifact_dir=str(path.parent),
            warnings=[f"mode={mode} baseline artifact is not a formal ablation"],
        )
    failure_delta = payload.get("failure_bucket_delta") if isinstance(payload.get("failure_bucket_delta"), dict) else None
    metrics = _extract_metrics(payload)
    warnings: list[str] = []
    fallback = _fallback(payload)
    if fallback:
        warnings.append("fallback_in_use=true; exclude from formal ablation tables")
    if not metrics:
        warnings.append("no primary metrics extracted")
    return AblationRecord(
        ablation=classified_ablation,
        benchmark=_benchmark_name(path, payload),
        status="available",
        metrics_path=str(path),
        artifact_dir=str(path.parent),
        question_count=payload.get("total_question_count") or payload.get("question_count"),
        elapsed_seconds=payload.get("elapsed_seconds") or payload.get("wall_clock_elapsed_seconds"),
        metrics=metrics,
        failure_bucket_delta=failure_delta,
        fallback_in_use=fallback,
        warnings=warnings,
    )


def iter_metrics_files(root: Path, *, max_depth: int = 5) -> list[Path]:
    root = root.resolve()
    found: list[Path] = []
    metric_names = {"metrics.json", "merged_metrics.json", "compact_metrics.json"}
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
        for name in sorted(metric_names & set(filenames)):
            found.append(current / name)
    return sorted(found)


def _run_scope_priority(record: AblationRecord) -> tuple[int, int, int, int, float]:
    path_text = str(record.metrics_path or "").lower()
    full_like = not any(
        marker in path_text
        for marker in (
            "smoke",
            "sample",
            "medium",
            "probe",
            "alpha",
            "evidence_blend",
            "supplemental",
            "guard_off",
            "guard-on",
            "guard_on",
        )
    )
    return (
        1 if record.status == "available" else 0,
        1 if record.fallback_in_use is False else 0,
        1 if full_like else 0,
        int(record.question_count or 0),
        Path(record.metrics_path or "").stat().st_mtime if record.metrics_path else 0.0,
    )


def _prefer_record(previous: AblationRecord, current: AblationRecord) -> AblationRecord:
    return current if _run_scope_priority(current) >= _run_scope_priority(previous) else previous


def discover_ablations(results_root: Path) -> list[AblationRecord]:
    records = [record_from_metrics(path) for path in iter_metrics_files(results_root)]
    latest: dict[tuple[str, str], AblationRecord] = {}
    benchmark_names = sorted({record.benchmark for record in records if record.benchmark != "unknown"})
    for record in records:
        if record.ablation == "__skip__":
            continue
        key = (record.benchmark, record.ablation)
        previous = latest.get(key)
        if previous is None:
            latest[key] = record
            continue
        latest[key] = _prefer_record(previous, record)
    benchmarks = sorted(set(benchmark_names) | {record.benchmark for record in latest.values() if record.benchmark != "unknown"})
    for benchmark in benchmarks:
        for ablation in REQUIRED_ABLATIONS:
            latest.setdefault(
                (benchmark, ablation),
                AblationRecord(
                    ablation=ablation,
                    benchmark=benchmark,
                    status="pending",
                    warnings=["no artifact-backed metrics.json found"],
                ),
            )
    return sorted(latest.values(), key=lambda item: (item.benchmark, item.ablation))


def write_markdown(records: list[AblationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ablation Table",
        "",
        "This table is artifact-backed. Missing ablations are marked `pending`; no metric is hand-filled.",
        "",
        "| benchmark | ablation | status | primary metrics | q | fallback | failure delta | artifact | warnings |",
        "|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for record in records:
        metric_text = ", ".join(f"{k}={v}" for k, v in (record.metrics or {}).items()) or ""
        failure_text = json.dumps(record.failure_bucket_delta or {}, ensure_ascii=False, sort_keys=True)
        warnings = "; ".join(record.warnings or [])
        lines.append(
            f"| {record.benchmark} | {record.ablation} | {record.status} | {metric_text} | "
            f"{record.question_count or ''} | {record.fallback_in_use} | `{failure_text}` | {record.metrics_path or ''} | {warnings} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_component_report(records: list[AblationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NeurIPS Component Ablation",
        "",
        "This report is generated from artifact-backed ablation records. Pending rows mean the required run has not yet produced a usable `metrics.json` artifact.",
        "",
    ]
    by_benchmark: dict[str, list[AblationRecord]] = {}
    for record in records:
        by_benchmark.setdefault(record.benchmark, []).append(record)
    for benchmark, benchmark_records in sorted(by_benchmark.items()):
        lines.extend([f"## {benchmark}", ""])
        for record in sorted(benchmark_records, key=lambda item: item.ablation):
            metric_text = ", ".join(f"{k}={v}" for k, v in (record.metrics or {}).items()) or "pending"
            warning_text = "; ".join(record.warnings or [])
            lines.append(f"- `{record.ablation}` status=`{record.status}` metrics={metric_text} warnings={warning_text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_delta_csv(records: list[AblationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bucket_names = sorted({key for record in records for key in (record.failure_bucket_delta or {}).keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "ablation", "status", "metrics_path", *bucket_names])
        for record in records:
            delta = record.failure_bucket_delta or {}
            writer.writerow([record.benchmark, record.ablation, record.status, record.metrics_path or "", *[delta.get(key, "") for key in bucket_names]])


def write_waterfall_data(records: list[AblationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "ablation", "status", "metric", "value", "artifact"])
        for record in records:
            if not record.metrics:
                writer.writerow([record.benchmark, record.ablation, record.status, "", "", record.metrics_path or ""])
                continue
            for metric, value in sorted(record.metrics.items()):
                writer.writerow([record.benchmark, record.ablation, record.status, metric, value, record.metrics_path or ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Artifact-backed ablation suite summary.")
    parser.add_argument("--results-root", type=Path, default=Path("../BenchmarkResult"), help="Directory containing benchmark metrics.json artifacts")
    parser.add_argument("--out", type=Path, default=Path("artifacts/ablations"), help="Output artifact directory")
    parser.add_argument("--report", type=Path, default=Path("reports/ablation_table.md"), help="Markdown report path")
    parser.add_argument("--component-report", type=Path, default=Path("reports/neurips_component_ablation.md"), help="Mechanism-level report path")
    parser.add_argument("--failure-delta-csv", type=Path, default=Path("reports/failure_bucket_delta_by_ablation.csv"), help="Failure bucket delta CSV path")
    parser.add_argument("--waterfall-csv", type=Path, default=Path("figures/ablation_waterfall_data.csv"), help="Ablation waterfall data CSV path")
    args = parser.parse_args()

    records = discover_ablations(args.results_root)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dysonspherain.ablations.v1",
        "results_root": str(args.results_root),
        "required_ablations": REQUIRED_ABLATIONS,
        "records": [asdict(record) for record in records],
        "formal_use_warning": "Use only records with fallback_in_use=false, matched query sets, and complete failure-bucket deltas for paper tables.",
    }
    (args.out / "ablation_runs.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(records, args.report)
    write_component_report(records, args.component_report)
    write_failure_delta_csv(records, args.failure_delta_csv)
    write_waterfall_data(records, args.waterfall_csv)
    print(
        json.dumps(
            {
                "records": len(records),
                "out": str(args.out),
                "report": str(args.report),
                "component_report": str(args.component_report),
                "failure_delta_csv": str(args.failure_delta_csv),
                "waterfall_csv": str(args.waterfall_csv),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
