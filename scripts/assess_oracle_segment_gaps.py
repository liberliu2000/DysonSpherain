#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "base"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from sphere_cli.experiment_registry import BenchmarkRun, latest_run, load_registry  # noqa: E402


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
    for benchmark, expected in EXPECTED_QUESTIONS.items():
        candidates = [
            run
            for run in runs
            if run.project == "DysonSpherain"
            and run.dataset.lower() == benchmark
            and run.run_type == "full"
            and run.fallback_in_use is False
            and (run.question_count or 0) >= expected
        ]
        if candidates:
            selected[benchmark] = latest_run(candidates, project="DysonSpherain", dataset=benchmark)
    return selected


def _source_metrics_path(run: BenchmarkRun) -> Path | None:
    raw = run.metadata.get("source_metrics_path") if isinstance(run.metadata, dict) else None
    if raw and Path(str(raw)).exists():
        return Path(str(raw))
    for name in ("merged_metrics.json", "metrics.json"):
        candidate = Path(run.artifact_dir) / name
        if candidate.exists():
            return candidate
    return None


def _source_files(run: BenchmarkRun) -> list[Path]:
    metrics_path = _source_metrics_path(run)
    if metrics_path is None:
        return []
    payload = _load(metrics_path)
    files = [Path(str(item)) for item in payload.get("source_files") or []]
    return [path for path in files if path.exists()] or [metrics_path]


def _oracle_paths(run: BenchmarkRun, benchmark: str) -> list[Path]:
    paths: list[Path] = []
    for source in _source_files(run):
        candidate = source.parent / "reports" / "diagnostics" / f"{benchmark}_oracle_retrieval.json"
        if candidate.exists():
            paths.append(candidate)
    direct = Path(run.artifact_dir) / "reports" / "diagnostics" / f"{benchmark}_oracle_retrieval.json"
    if direct.exists():
        paths.append(direct)
    return sorted(dict.fromkeys(paths))


def _existing_baseline_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = _load(path)
    rows = payload.get("records") if isinstance(payload.get("records"), list) else []
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        records[(str(row.get("benchmark")), str(row.get("baseline")))] = row
    return records


def build_report(base_dir: Path, baseline_path: Path) -> dict[str, Any]:
    runs = _latest_formal_runs(base_dir)
    baseline_records = _existing_baseline_records(baseline_path)
    rows: list[dict[str, Any]] = []
    for benchmark in EXPECTED_QUESTIONS:
        run = runs.get(benchmark)
        existing = baseline_records.get((benchmark, "oracle_segment"), {})
        existing_metrics = existing.get("metrics_path")
        if existing.get("status") == "available" and existing_metrics and Path(str(existing_metrics)).exists():
            rows.append(
                {
                    "benchmark": benchmark,
                    "status": "already_available",
                    "reason": "oracle_segment baseline already has artifact-backed metrics",
                    "run_id": run.run_id if run else None,
                    "artifact_dir": run.artifact_dir if run else None,
                    "source_metrics_path": str(_source_metrics_path(run) or "") if run else "",
                    "oracle_diagnostic_paths": [],
                    "existing_baseline_status": existing.get("status"),
                    "existing_metrics_path": existing_metrics,
                }
            )
            continue
        if run is None:
            rows.append(
                {
                    "benchmark": benchmark,
                    "status": "blocked",
                    "reason": "no current full non-fallback run in registry",
                    "existing_baseline_status": existing.get("status"),
                }
            )
            continue
        oracle_paths = _oracle_paths(run, benchmark)
        status = "exportable" if oracle_paths else "blocked_missing_source_diagnostics"
        reason = (
            "current formal full source contains oracle retrieval diagnostics"
            if oracle_paths
            else "current formal full source metrics do not reference oracle retrieval diagnostics; do not hand-fill oracle_segment"
        )
        rows.append(
            {
                "benchmark": benchmark,
                "status": status,
                "reason": reason,
                "run_id": run.run_id,
                "artifact_dir": run.artifact_dir,
                "source_metrics_path": str(_source_metrics_path(run) or ""),
                "oracle_diagnostic_paths": [str(path) for path in oracle_paths],
                "existing_baseline_status": existing.get("status"),
                "existing_metrics_path": existing.get("metrics_path"),
            }
        )
    return {
        "schema": "dysonspherain.oracle_segment_gap_assessment.v1",
        "rows": rows,
        "summary": {
            "already_available": sum(1 for row in rows if row["status"] == "already_available"),
            "exportable": sum(1 for row in rows if row["status"] == "exportable"),
            "blocked_missing_source_diagnostics": sum(1 for row in rows if row["status"] == "blocked_missing_source_diagnostics"),
            "blocked": sum(1 for row in rows if row["status"].startswith("blocked")),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Oracle Segment Gap Assessment",
        "",
        "This report checks whether pending `oracle_segment` baseline rows can be exported from the current formal full-run diagnostics.",
        "",
        f"- summary: `{report['summary']}`",
        "",
        "| benchmark | status | existing_baseline | run_id | source metrics | oracle diagnostics | reason |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row.get('benchmark')} | {row.get('status')} | {row.get('existing_baseline_status')} | "
            f"{row.get('run_id', '')} | {row.get('source_metrics_path', '')} | "
            f"{len(row.get('oracle_diagnostic_paths') or [])} | {row.get('reason')} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Rows marked `blocked_missing_source_diagnostics` should remain artifact-backed pending/blocked until a full non-fallback run emits oracle retrieval diagnostics. They must not be filled from unrelated exploratory runs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess oracle_segment baseline gaps against current formal full diagnostics.")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--baselines", type=Path, default=Path("artifacts/baselines/baseline_runs.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/baselines/oracle_segment_gap_assessment.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/oracle_segment_gap_assessment.md"))
    args = parser.parse_args()
    report = build_report(args.base_dir.resolve(), args.baselines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.report)
    print(json.dumps({"summary": report["summary"], "out": str(args.out), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
