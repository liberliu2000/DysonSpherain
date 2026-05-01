#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "base"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from sphere_cli.experiment_registry import BenchmarkRun, latest_run, load_registry  # noqa: E402


BENCHMARKS = ("longmemeval", "locomo", "knowme", "clonemem")
EXPECTED_QUESTIONS = {
    "longmemeval": 500,
    "locomo": 1986,
    "knowme": 1010,
    "clonemem": 2374,
}


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_formal_runs(base_dir: Path) -> dict[str, BenchmarkRun]:
    runs = load_registry(base_dir)
    selected: dict[str, BenchmarkRun] = {}
    for benchmark in BENCHMARKS:
        candidates = [
            run
            for run in runs
            if run.project == "DysonSpherain"
            and run.dataset.lower() == benchmark
            and run.run_type == "full"
            and run.fallback_in_use is False
            and (run.question_count or 0) >= EXPECTED_QUESTIONS[benchmark]
        ]
        if candidates:
            selected[benchmark] = latest_run(candidates, project="DysonSpherain", dataset=benchmark)
    return selected


def _source_metrics_path(run: BenchmarkRun) -> Path | None:
    raw = run.metadata.get("source_metrics_path") if isinstance(run.metadata, dict) else None
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
    for name in ("merged_metrics.json", "metrics.json"):
        path = Path(run.artifact_dir) / name
        if path.exists():
            return path
    return None


def _diagnostic_paths(run: BenchmarkRun, suffix: str) -> list[Path]:
    metrics = _source_metrics_path(run)
    if metrics is None:
        return []
    payload = _load(metrics)
    paths: list[Path] = []
    for source in payload.get("source_files") or []:
        source_path = Path(str(source))
        candidate = source_path.parent / "reports" / "diagnostics" / suffix
        if candidate.exists():
            paths.append(candidate)
    direct = metrics.parent / "reports" / "diagnostics" / suffix
    if direct.exists():
        paths.append(direct)
    return sorted(dict.fromkeys(paths))


def _candidate_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _load(path)
        rows.extend(row for row in payload.get("queries") or [] if isinstance(row, dict))
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def _oracle_candidate(run: BenchmarkRun, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    value = _mean_key(rows, "candidate_recall@100")
    if value is None:
        value = _safe_float(run.metrics.get("candidate_recall@100"))
    if value is None:
        return None
    return {
        "baseline": "oracle_candidate",
        "metrics": {
            "candidate_recall@100": value,
            "oracle_candidate_recall@100": value,
        },
        "diagnostic": "candidate_recall@100 upper bound from candidate recall diagnostics",
    }


def _oracle_parent(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    parent_values = [1.0 if row.get("gold_parent_hit") else 0.0 for row in rows if "gold_parent_hit" in row]
    if not parent_values:
        return None
    return {
        "baseline": "oracle_parent",
        "metrics": {"oracle_parent_recall@100": mean(parent_values)},
        "diagnostic": "gold_parent_hit rate from candidate recall diagnostics",
    }


def _oracle_segment(benchmark: str, oracle_paths: list[Path]) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in oracle_paths:
        payload = _load(path)
        summaries.append(payload)
        rows.extend(row for row in payload.get("rows") or [] if isinstance(row, dict))
    if rows:
        denom = max(1, len(rows))
        metrics = {
            "oracle_recall@1": sum(1 for row in rows if row.get("top1_hit")) / denom,
            "oracle_recall@5": sum(1 for row in rows if row.get("top5_hit")) / denom,
            "oracle_recall@10": sum(1 for row in rows if row.get("top10_hit")) / denom,
        }
    elif summaries:
        weighted: dict[str, list[float]] = defaultdict(list)
        for payload in summaries:
            count = int(payload.get("oracle_query_count") or 0)
            for key in ("oracle_recall@1", "oracle_recall@5", "oracle_recall@10"):
                value = _safe_float(payload.get(key))
                if value is not None:
                    weighted[key].extend([value] * max(1, count))
        metrics = {key: mean(values) for key, values in weighted.items() if values}
    else:
        return None
    if not metrics:
        return None
    return {
        "baseline": "oracle_segment",
        "metrics": metrics,
        "diagnostic": f"oracle retrieval diagnostic rows for {benchmark}",
    }


def _write_metric(out_root: Path, benchmark: str, run: BenchmarkRun, item: dict[str, Any], sources: list[Path]) -> Path:
    baseline = str(item["baseline"])
    out_dir = out_root / f"{benchmark}_{baseline}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dysonspherain.artifact_oracle_baseline.v1",
        "benchmark": benchmark,
        "baseline": baseline,
        "run_type": "full",
        "status": "available",
        "total_question_count": run.question_count,
        "question_count": run.question_count,
        "elapsed_seconds": 0.0,
        "embedding_info": {
            "embedding_provider": run.embedding_provider,
            "embedding_model": run.embedding_model,
            "fallback_in_use": run.fallback_in_use,
        },
        "fallback_in_use": run.fallback_in_use,
        "metrics": item["metrics"],
        "diagnostic": item["diagnostic"],
        "source_run_id": run.run_id,
        "source_artifact_dir": run.artifact_dir,
        "source_files": [str(path) for path in sources],
        "formal_use_warning": "Oracle baseline is an artifact-derived diagnostic upper bound, not a deployed retrieval method.",
    }
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_oracle_baselines(*, base_dir: Path, out_root: Path) -> dict[str, Any]:
    runs = _latest_formal_runs(base_dir)
    records: list[dict[str, Any]] = []
    for benchmark, run in sorted(runs.items()):
        candidate_paths = _diagnostic_paths(run, f"{benchmark}_candidate_recall.json")
        oracle_paths = _diagnostic_paths(run, f"{benchmark}_oracle_retrieval.json")
        rows = _candidate_rows(candidate_paths)
        for item, sources in (
            (_oracle_candidate(run, rows), candidate_paths),
            (_oracle_parent(rows), candidate_paths),
            (_oracle_segment(benchmark, oracle_paths), oracle_paths),
        ):
            if item is None:
                continue
            metrics_path = _write_metric(out_root, benchmark, run, item, sources)
            records.append(
                {
                    "benchmark": benchmark,
                    "baseline": item["baseline"],
                    "metrics_path": str(metrics_path),
                    "source_count": len(sources),
                    "metrics": item["metrics"],
                }
            )
    return {"schema": "dysonspherain.oracle_baseline_export.v1", "records": records}


def write_report(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Artifact Oracle Baselines",
        "",
        "These rows are generated from existing diagnostics. Missing oracle rows remain absent rather than fabricated.",
        "",
        "| benchmark | baseline | metrics | sources | artifact |",
        "|---|---|---:|---:|---|",
    ]
    for row in summary.get("records") or []:
        metrics = ", ".join(f"{key}={value}" for key, value in (row.get("metrics") or {}).items())
        lines.append(f"| {row.get('benchmark')} | {row.get('baseline')} | {metrics} | {row.get('source_count')} | {row.get('metrics_path')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export artifact-derived oracle baseline metrics.")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-root", type=Path, default=Path("/Users/yanbo/DysonSpherain/BenchmarkResult/20260428_artifact_oracle_baselines_v1"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/baselines/oracle_baseline_exports.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/oracle_baseline_exports.md"))
    args = parser.parse_args()
    summary = build_oracle_baselines(base_dir=args.base_dir.resolve(), out_root=args.out_root)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_report(summary, args.report)
    print(json.dumps({"records": len(summary["records"]), "out_root": str(args.out_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
