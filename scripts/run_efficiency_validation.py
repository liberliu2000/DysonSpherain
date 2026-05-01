#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ALL = ROOT / "base" / "benchmarks" / "run_all_benchmarks.py"
RESULT_ROOT = ROOT.parent / "BenchmarkResult"
REDLINES_MIN = {
    "longmemeval": {"warning": 10, "fail": 10},
    "locomo": {"warning": 45, "fail": 45},
    "knowme": {"warning": 35, "fail": 35},
    "clonemem": {"warning": 75, "fail": 90},
    "all": {"warning": 150, "fail": 210},
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_files(result_dir: Path) -> list[Path]:
    return sorted(path for path in result_dir.rglob("*metrics*.json") if path.is_file())


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first(payload: dict[str, Any], paths: list[tuple[str, ...]]) -> Any:
    for path in paths:
        value = _nested(payload, *path)
        if value is not None:
            return value
    return None


def summarize_result(result_dir: Path) -> dict[str, Any]:
    benchmarks: dict[str, Any] = {}
    for path in _metric_files(result_dir):
        payload = _load(path)
        benchmark = str(payload.get("benchmark") or payload.get("benchmark_name") or path.parent.name).lower()
        timings = dict(payload.get("timings") or payload.get("stage_timing_ms") or {})
        side_audit = dict(payload.get("side_index_audit_summary") or {})
        rows = list(payload.get("results") or [])
        per_query_retrieval = [
            float((row.get("stage_timing_ms") or {}).get("retrieval_ms"))
            for row in rows
            if isinstance((row.get("stage_timing_ms") or {}).get("retrieval_ms"), (int, float))
        ]
        per_query_retrieval.sort()
        p50 = per_query_retrieval[len(per_query_retrieval) // 2] if per_query_retrieval else None
        p95 = per_query_retrieval[int(len(per_query_retrieval) * 0.95)] if per_query_retrieval else None
        benchmarks[benchmark] = {
            "path": str(path),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "wall_clock_elapsed_seconds": payload.get("wall_clock_elapsed_seconds", payload.get("elapsed_seconds")),
            "serial_equivalent_elapsed_seconds": payload.get("serial_elapsed_seconds_sum", payload.get("elapsed_seconds")),
            "speedup_estimate": payload.get("speedup_estimate", 1.0),
            "retrieval_time_ms_mean": timings.get("retrieval_ms") or _nested(payload, "stage_timing_ms", "retrieval_ms"),
            "retrieval_time_ms_p50": p50,
            "retrieval_time_ms_p95": p95,
            "fusion_time_ms": timings.get("fusion_ms") or _nested(payload, "stage_timing_ms", "fusion_ms"),
            "rerank_time_ms": timings.get("rerank_ms") or _nested(payload, "stage_timing_ms", "rerank_ms"),
            "full_scan_total_records_scored": side_audit.get("full_scan_total_records_scored"),
            "indexed_fast_path_channel_count": side_audit.get("indexed_fast_path_channel_count"),
            "legacy_fallback_channel_count": side_audit.get("legacy_fallback_channel_count"),
            "recall@5": _first(payload, [("metrics", "segment", "recall_frac@5"), ("metrics", "session", "recall_frac@5"), ("metrics", "message", "recall@5"), ("recall@5",)]),
            "recall@10": _first(payload, [("metrics", "segment", "recall_frac@10"), ("metrics", "session", "recall_frac@10"), ("metrics", "message", "recall@10"), ("recall@10",)]),
            "ndcg@10": _first(payload, [("metrics", "segment", "ndcg_any@10"), ("metrics", "session", "ndcg_any@10"), ("metrics", "message", "ndcg@10"), ("ndcg@10",)]),
            "candidate_recall@100": _first(payload, [("candidate_recall_summary", "candidate_recall@100"), ("candidate_recall_report", "candidate_recall@100")]),
            "oracle_recall@10": _first(payload, [("oracle_summary", "oracle_recall@10"), ("oracle_retrieval_report", "oracle_recall@10")]),
            "failure_taxonomy": payload.get("failure_taxonomy") or payload.get("failure_summary"),
            "quality_guardrail_status": payload.get("quality_guardrail_status", "unknown"),
        }
    manifest = result_dir / "run_manifest.json"
    if manifest.exists():
        benchmarks["_manifest"] = _load(manifest)
    return {"result_dir": str(result_dir), "benchmarks": benchmarks}


def compare_results(latest: Path, previous: Path) -> dict[str, Any]:
    latest_summary = summarize_result(latest)
    previous_summary = summarize_result(previous)
    deltas: dict[str, Any] = {}
    for benchmark, current in latest_summary["benchmarks"].items():
        if benchmark.startswith("_"):
            continue
        prev = previous_summary["benchmarks"].get(benchmark)
        if not prev:
            continue
        row: dict[str, Any] = {}
        for key in ("elapsed_seconds", "wall_clock_elapsed_seconds", "serial_equivalent_elapsed_seconds", "recall@10", "ndcg@10", "candidate_recall@100"):
            if isinstance(current.get(key), (int, float)) and isinstance(prev.get(key), (int, float)):
                row[f"{key}_delta"] = round(float(current[key]) - float(prev[key]), 6)
        deltas[benchmark] = row
    return {"latest": latest_summary, "previous": previous_summary, "deltas": deltas}


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "efficiency_validation_report.json"
    md_path = out_dir / "efficiency_validation_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Efficiency Validation Report", ""]
    lines.append(f"Mode: `{report.get('mode')}`")
    lines.append("")
    for benchmark, row in dict(report.get("summary", {}).get("benchmarks") or {}).items():
        if benchmark.startswith("_"):
            continue
        lines.append(f"## {benchmark}")
        for key in ("elapsed_seconds", "wall_clock_elapsed_seconds", "serial_equivalent_elapsed_seconds", "speedup_estimate", "recall@10", "ndcg@10", "candidate_recall@100", "quality_guardrail_status"):
            lines.append(f"- `{key}`: `{row.get(key)}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or compare benchmark efficiency validation.")
    parser.add_argument("--mode", choices=["smoke", "full", "compare"], default="compare")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "benchmark")
    parser.add_argument("--out", type=Path, default=RESULT_ROOT / "efficiency_validation")
    parser.add_argument("--benchmarks", default="longmemeval,locomo,knowme,clonemem")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--latest", type=Path, default=None)
    parser.add_argument("--previous", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    if args.mode in {"smoke", "full"}:
        command = [
            sys.executable,
            str(RUN_ALL),
            "--data-root",
            str(args.data_root),
            "--out",
            str(args.out),
            "--benchmarks",
            args.benchmarks,
            "--chunked",
            "--chunks",
            str(args.chunks),
            "--workers",
            str(args.workers),
            "--resume",
        ]
        if args.mode == "smoke":
            command.extend(["--benchmarks", args.benchmarks])
        completed = subprocess.run(command, cwd=ROOT, text=True, check=False)
        if completed.returncode != 0:
            print(f"validation command failed: {completed.returncode}", file=sys.stderr)
    if args.mode == "compare":
        candidates = sorted([path for path in RESULT_ROOT.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True) if RESULT_ROOT.exists() else []
        latest = args.latest or (candidates[0] if candidates else None)
        previous = args.previous or (candidates[1] if len(candidates) > 1 else None)
        if latest is None or previous is None:
            raise SystemExit("Need --latest and --previous or at least two BenchmarkResult directories")
        report = {"mode": "compare", **compare_results(latest, previous), "redlines_minutes": REDLINES_MIN}
    else:
        report = {"mode": args.mode, "summary": summarize_result(args.out), "redlines_minutes": REDLINES_MIN}
    report["validation_elapsed_seconds"] = round(time.monotonic() - started, 3)
    write_reports(report, args.out if args.mode != "compare" else (args.latest or RESULT_ROOT / "efficiency_compare"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
