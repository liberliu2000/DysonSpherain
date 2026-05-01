#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _table(rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    lines = []
    for row_index, row in enumerate(rows):
        lines.append("| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |")
        if row_index == 0:
            lines.append("| " + " | ".join("-" * widths[index] for index in range(len(row))) + " |")
    return lines


def write_neurips_summary(base_dir: Path) -> None:
    validation = _load(base_dir / "artifacts/formal_protocol_validation.json")
    gap = _load(base_dir / "artifacts/formal_evidence_gap_report.json")
    rows = [["benchmark", "status", "q", "elapsed_s", "primary metrics"]]
    for item in validation.get("full_benchmarks") or []:
        metrics = item.get("metrics") or {}
        primary = ", ".join(f"{key}={value}" for key, value in metrics.items()) or "pending"
        rows.append([item.get("benchmark", ""), item.get("status", ""), item.get("question_count", ""), item.get("elapsed_seconds", ""), primary])
    evidence_rows = [["area", "available", "pending", "total"]]
    for key in ("baselines", "ablations", "leave_one_benchmark_out", "statistics", "efficiency"):
        section = (gap or {}).get(key) if isinstance((gap or {}).get(key), dict) else {}
        if key == "efficiency":
            expected_sweeps = section.get("expected_sweeps") or []
            available = section.get("formal_sweep_available", 0)
            total = len(expected_sweeps) if isinstance(expected_sweeps, list) else int(expected_sweeps or 0)
            pending = len(section.get("sweep_pending_items") or [])
        elif key == "statistics":
            available = section.get("paired_delta_available", 0)
            total = section.get("paired_delta_reports", 0)
            pending = max(0, int(total or 0) - int(available or 0))
        else:
            available = section.get("available", 0)
            total = section.get("total", 0)
            pending = section.get("pending", max(0, int(total or 0) - int(available or 0)))
        evidence_rows.append([key, available, pending, total])
    lines = [
        "# NeurIPS Upgrade Summary",
        "",
        "This report is generated from artifacts. Pending evidence remains pending; no result is hand-filled.",
        "",
        f"- formal_protocol_status: `{validation.get('overall_status', 'unknown')}`",
        f"- combined_full_elapsed_seconds: `{validation.get('combined_elapsed_seconds', '')}`",
        "",
        "## Full Benchmark Snapshot",
        "",
        *_table(rows),
        "",
        "## Formal Evidence Coverage",
        "",
        *_table(evidence_rows),
        "",
        "## Interpretation",
        "",
        "- Retrieval-efficiency and benchmark infrastructure are implemented and tested.",
        "- Current formal protocol status is artifact-backed; blocked unavailable-model baselines remain explicitly marked rather than hand-filled.",
        "- CloneMem Phase 5 now uses a route-only protected top-3 lexical anchor gate promotion; older CloneMem full runs remain non-comparable when route policy or config hashes differ.",
        "- Blocked baselines are not completed experiments; they are unavailable-model rows documented with artifact-backed reasons.",
        "",
        "## Artifact Pointers",
        "",
        "- `artifacts/formal_protocol_validation.json`",
        "- `artifacts/formal_evidence_gap_report.json`",
        "- `artifacts/baselines/baseline_runs.json`",
        "- `artifacts/ablations/ablation_runs.json`",
        "- `artifacts/statistics/paired_delta_full_vs_dense.json`",
        "- `artifacts/profiling/efficiency_quality_pareto.json`",
        "- `reports/phase2_memory_os_cli_acceptance.md`",
        "- `reports/phase5_clonemem_route_only_promotion_decision.md`",
    ]
    (base_dir / "reports/NEURIPS_UPGRADE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_repro_package(base_dir: Path) -> None:
    lines = [
        "# Reproducibility Package",
        "",
        "## Environment",
        "",
        "- Python: `./.venv312/bin/python`",
        "- PYTHONPATH: `base`",
        "- Formal benchmark env: `SPHERE_CREATIVE_MODE=off`, `SPHERE_EMBEDDING_FAIL_FAST=1`, `SPHERE_VECTOR_BACKEND=chroma`, `SPHERE_VECTOR_FAIL_FAST_ON_FALLBACK=1`, `SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING=1`",
        "",
        "## Core Validation Commands",
        "",
        "```bash",
        "PYTHONPATH=base ./.venv312/bin/python -m unittest discover tests",
        "./.venv312/bin/python scripts/run_baselines.py --results-root /Users/yanbo/DysonSpherain/BenchmarkResult",
        "./.venv312/bin/python scripts/run_ablation_suite.py --results-root /Users/yanbo/DysonSpherain/BenchmarkResult",
        "./.venv312/bin/python scripts/build_formal_evidence_gap_report.py",
        "./.venv312/bin/python scripts/generate_paper_outputs.py",
        "./.venv312/bin/python scripts/validate_formal_protocol.py",
        "```",
        "",
        "## Full Benchmark Entry Points",
        "",
        "Use `base/benchmarks/run_benchmark_chunked.py` for chunked full runs and `base/benchmarks/run_all_benchmarks.py --chunked` for bundled runs. Use `--resume` to skip completed chunks and `--force` only when intentionally rerunning.",
        "",
        "## Expected Artifact Layout",
        "",
        "- `BenchmarkResult/<run>/<benchmark>/merged_metrics.json`",
        "- `BenchmarkResult/<run>/<benchmark>/chunk_*/metrics.json`",
        "- `BenchmarkResult/<run>/run_manifest.json`",
        "- `artifacts/*` generated summaries",
        "- `reports/*` generated markdown reports",
        "- `paper/tables/*.tex` and `paper/figures/data/*.csv`",
        "",
        "## Known Compute Requirements",
        "",
        "Current hard redlines: LongMemEval <= 10 min, LoCoMo <= 45 min, KnowMe <= 35 min, CloneMem warning > 75 min and fail > 90 min. Combined elapsed is diagnostic with warning > 4 h and fail > 6 h.",
        "",
        "## Non-Negotiable Guards",
        "",
        "- Formal benchmark results must be full, artifact-backed, and `fallback_in_use=false`.",
        "- `local_hash` is not valid for formal benchmark claims.",
        "- Pending or blocked rows must not be replaced with hand-filled values; unavailable-model baselines stay explicitly blocked until real artifacts exist.",
    ]
    (base_dir / "reports/REPRODUCIBILITY_PACKAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_memory_os_summary(base_dir: Path) -> None:
    project_state = _load(base_dir / "artifacts/project_state/dysonspherain_d6ecde6258cd.json")
    lines = [
        "# Memory OS Upgrade Summary",
        "",
        "## Implemented Surfaces",
        "",
        "- project-memory CRUD/search/archive CLI acceptance recorded in `reports/phase2_memory_os_cli_acceptance.md`",
        "- memory schema and project state artifacts under `artifacts/project_state/`",
        "- experiment registry under `base/sphere_cli/experiment_registry.py`",
        "- context compiler under `base/sphere_cli/context_compiler.py`",
        "- agent preflight/postrun and execution ledger CLI surfaces",
        "- conflict/lifecycle management under `base/sphere_cli/memory_lifecycle.py`",
        "- local-first security redaction under `base/sphere_cli/security.py`",
        "",
        "## Current Project State",
        "",
        f"- current_goal: `{project_state.get('current_goal', '')}`",
        f"- current_phase: `{project_state.get('current_phase', '')}`",
        f"- constraints: `{len(project_state.get('constraints') or [])}`",
        f"- tracked benchmark statuses: `{len(project_state.get('latest_benchmark_status') or {})}`",
        "",
        "## Known Limitations",
        "",
        "- Project state phase labels may lag the latest artifact-backed reports unless regenerated after each formal pass.",
        "- ConvoMem is not part of the current four-benchmark formal validation bundle.",
        "- Memory OS report content is artifact-backed and does not imply unresolved benchmark rows are complete.",
    ]
    (base_dir / "reports/MEMORY_OS_UPGRADE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ledger_summary(base_dir: Path) -> None:
    ledger = base_dir / "artifacts/execution_ledger/dysonspherain_d6ecde6258cd.jsonl"
    lines = [
        "# Final Execution Ledger Summary",
        "",
        f"- ledger_path: `{ledger}`",
        f"- ledger_entry_count: `{_jsonl_count(ledger)}`",
        "",
        "## Validation Commands From This Phase",
        "",
        "- `python -m unittest discover tests`",
        "- `scripts/validate_formal_protocol.py`",
        "- `scripts/generate_paper_outputs.py`",
        "- `scripts/run_leave_one_benchmark_out.py`",
        "",
        "## Status",
        "",
        "The ledger surface exists and is covered by tests. This summary reports the available local ledger artifact without inventing missing run metadata.",
    ]
    (base_dir / "reports/FINAL_EXECUTION_LEDGER_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final artifact-backed project reports.")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    base_dir = args.base_dir.resolve()
    (base_dir / "reports").mkdir(parents=True, exist_ok=True)
    write_neurips_summary(base_dir)
    write_repro_package(base_dir)
    write_memory_os_summary(base_dir)
    write_ledger_summary(base_dir)
    print(json.dumps({"status": "ok", "reports": 4}, ensure_ascii=False))


if __name__ == "__main__":
    main()
