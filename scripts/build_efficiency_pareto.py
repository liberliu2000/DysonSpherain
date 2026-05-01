#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


MAX_METRICS_BYTES = 25 * 1024 * 1024
SKIP_DIR_NAMES = {".git", ".venv", ".venv312", "__pycache__", ".cache", "cache", "chroma", "vector_store", "workspace", "workspaces"}
QUALITY_KEYS = ("recall@10", "Recall@10", "recall_frac@10", "ndcg@10", "NDCG@10", "ndcg_any@10", "candidate_recall@100")
TIMING_KEYS = ("retrieval_ms", "fusion_time_ms", "rerank_time_ms", "dense_ms", "sparse_ms", "embedding_ms", "total_ms")
METRICS_FILENAMES = {"metrics.json", "merged_metrics.json"}
EXPECTED_SWEEPS = ("candidate_top50", "candidate_top100", "candidate_top200", "candidate_top500", "rerank_top20", "rerank_top50", "rerank_top100")


def iter_metrics_files(root: Path, *, max_depth: int = 6) -> list[Path]:
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
        for filename in sorted(METRICS_FILENAMES & set(filenames)):
            found.append(current / filename)
    return sorted(found)


def _load(path: Path) -> dict[str, Any] | None:
    if path.stat().st_size > MAX_METRICS_BYTES:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _benchmark(path: Path, payload: dict[str, Any]) -> str:
    if payload.get("benchmark"):
        return str(payload["benchmark"]).lower()
    for part in reversed(path.parts):
        lower = part.lower()
        for name in ("longmemeval", "locomo", "knowme", "clonemem", "convomem"):
            if name in lower:
                return name
    return "unknown"


def _find_metric(payload: Any, key: str) -> Any:
    if not isinstance(payload, dict):
        return None
    if key in payload:
        return payload[key]
    lower_key = key.lower()
    for current_key, value in payload.items():
        if str(current_key).lower() == lower_key:
            return value
    for preferred in ("metrics", "session", "segment", "turn", "dialog", "stage_timing_ms", "performance_cache_report", "timing_summary", "timings"):
        nested = payload.get(preferred)
        if isinstance(nested, dict):
            found = _find_metric(nested, key)
            if found is not None:
                return found
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_metric(value, key)
            if found is not None:
                return found
    return None


def build_records(results_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_metrics_files(results_root):
        payload = _load(path)
        if payload is None:
            records.append({"artifact": str(path), "status": "oversized_or_unreadable"})
            continue
        record = {
            "artifact": str(path),
            "status": "available",
            "benchmark": _benchmark(path, payload),
            "question_count": _find_metric(payload, "question_count") or _find_metric(payload, "total_question_count"),
            "elapsed_seconds": _find_metric(payload, "elapsed_seconds") or _find_metric(payload, "wall_clock_elapsed_seconds"),
        }
        for key in QUALITY_KEYS:
            value = _find_metric(payload, key)
            if value is not None:
                record[key] = value
        for key in TIMING_KEYS:
            value = _find_metric(payload, key)
            if value is not None:
                record[key] = value
        records.append(record)
    return records


def _sweep_key(payload: dict[str, Any], path: Path) -> str | None:
    if payload.get("sweep_name"):
        return str(payload["sweep_name"])
    sweep_type = payload.get("sweep_type")
    budget = payload.get("budget")
    if sweep_type and budget is not None:
        prefix = "candidate_top" if str(sweep_type) == "candidate" else "rerank_top" if str(sweep_type) == "rerank" else str(sweep_type)
        return f"{prefix}{budget}"
    for part in path.parts:
        lower = part.lower()
        for expected in EXPECTED_SWEEPS:
            if expected in lower:
                return expected
    return None


def build_sweep_status(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sweep: dict[str, dict[str, Any]] = {}
    priority_by_sweep: dict[str, tuple[int, int, int]] = {}
    for record in records:
        if record.get("status") != "available":
            continue
        artifact_path = Path(str(record["artifact"]))
        payload = _load(artifact_path)
        if not payload:
            continue
        sweep = _sweep_key(payload, artifact_path)
        if sweep is None or sweep not in EXPECTED_SWEEPS:
            continue
        formal_eligible = bool(payload.get("formal_eligible"))
        has_sweep_metadata = 1 if payload.get("schema") == "dysonspherain.efficiency_budget_sweep.v1" else 0
        is_merged = 1 if artifact_path.name == "merged_metrics.json" else 0
        question_count = record.get("question_count")
        try:
            question_count_value = int(question_count or 0)
        except (TypeError, ValueError):
            question_count_value = 0
        priority = (1 if formal_eligible else 0, has_sweep_metadata, is_merged, question_count_value)
        current = {
            "sweep": sweep,
            "status": "available" if formal_eligible else "smoke_available",
            "formal_eligible": formal_eligible,
            "artifact": record["artifact"],
            "benchmark": record.get("benchmark"),
            "question_count": record.get("question_count"),
            "sweep_type": payload.get("sweep_type"),
            "budget": payload.get("budget"),
            "run_scope": payload.get("run_scope") or payload.get("run_type"),
            "fallback_in_use": _find_metric(payload, "fallback_in_use"),
        }
        previous_priority = priority_by_sweep.get(sweep)
        if previous_priority is None or priority > previous_priority:
            by_sweep[sweep] = current
            priority_by_sweep[sweep] = priority
    rows: list[dict[str, Any]] = []
    for sweep in EXPECTED_SWEEPS:
        rows.append(
            by_sweep.get(
                sweep,
                {
                    "sweep": sweep,
                    "status": "pending",
                    "formal_eligible": False,
                    "artifact": None,
                    "benchmark": None,
                    "question_count": None,
                },
            )
        )
    return rows


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["benchmark", "status", "question_count", "elapsed_seconds", *QUALITY_KEYS, *TIMING_KEYS, "artifact"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_report(records: list[dict[str, Any]], sweep_status: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    available = [record for record in records if record.get("status") == "available"]
    lines = [
        "# Efficiency Quality Pareto",
        "",
        "This report is artifact-backed. Candidate-budget and rerank-budget sweeps remain pending unless a matching artifact exists.",
        "",
        f"- records_scanned: `{len(records)}`",
        f"- available_records: `{len(available)}`",
        "",
        "## Budget Sweep Status",
        "",
        "| sweep | status | note |",
        "|---|---|---|",
    ]
    for row in sweep_status:
        note = row.get("artifact") or "no dedicated matched-budget artifact registered yet"
        lines.append(f"| {row['sweep']} | {row['status']} | {note} |")
    lines.extend(["", "## Available Artifacts", ""])
    for record in available[:80]:
        lines.append(
            f"- `{record.get('benchmark')}` q=`{record.get('question_count')}` elapsed=`{record.get('elapsed_seconds')}` artifact=`{record.get('artifact')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build artifact-backed efficiency-quality Pareto data.")
    parser.add_argument("--results-root", type=Path, default=Path("../BenchmarkResult"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/profiling/efficiency_quality_pareto.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/efficiency_quality_pareto.md"))
    parser.add_argument("--csv", type=Path, default=Path("figures/pareto_curve_data.csv"))
    args = parser.parse_args()

    records = build_records(args.results_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sweep_status = build_sweep_status(records)
    args.out.write_text(
        json.dumps(
            {"schema": "dysonspherain.efficiency_pareto.v1", "records": records, "sweep_status": sweep_status},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_csv(records, args.csv)
    write_report(records, sweep_status, args.report)
    print(json.dumps({"records": len(records), "out": str(args.out), "report": str(args.report), "csv": str(args.csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
