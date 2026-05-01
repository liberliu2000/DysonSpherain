#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "base"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from sphere_cli.experiment_registry import BenchmarkRun, compare_runs, latest_run, load_registry  # noqa: E402


EXPECTED_QUESTIONS = {
    "longmemeval": 500,
    "locomo": 1986,
    "knowme": 1010,
    "clonemem": 2374,
}

REDLINES_SECONDS = {
    "longmemeval": {"fail": 10 * 60},
    "locomo": {"fail": 45 * 60},
    "knowme": {"fail": 35 * 60},
    "clonemem": {"warning": 75 * 60, "fail": 90 * 60},
}

COMBINED_REDLINE_SECONDS = {"warning": 4 * 60 * 60, "fail": 6 * 60 * 60}


@dataclass
class FormalRunStatus:
    benchmark: str
    status: str
    run_id: str | None = None
    artifact_dir: str | None = None
    question_count: int | None = None
    elapsed_seconds: float | None = None
    fallback_in_use: bool | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    compared_to_run_id: str | None = None
    comparison_status: str = "not_applicable"
    comparison_warnings: list[str] = field(default_factory=list)
    quality_deltas: dict[str, float] = field(default_factory=dict)


def _full_nonfallback_runs(runs: list[BenchmarkRun], benchmark: str) -> list[BenchmarkRun]:
    expected = EXPECTED_QUESTIONS[benchmark]
    selected = [
        run
        for run in runs
        if run.project == "DysonSpherain"
        and run.dataset.lower() == benchmark
        and run.run_type == "full"
        and run.fallback_in_use is False
        and (run.question_count or 0) >= expected
    ]
    return sorted(selected, key=lambda run: run.timestamp)


def _metric_deltas(current: BenchmarkRun, previous: BenchmarkRun | None) -> dict[str, float]:
    if previous is None:
        return {}
    deltas: dict[str, float] = {}
    for key, value in current.metrics.items():
        if key not in previous.metrics:
            continue
        try:
            deltas[key] = float(value) - float(previous.metrics[key])
        except (TypeError, ValueError):
            continue
    return deltas


def _same_metric_comparison_scope(current: BenchmarkRun, previous: BenchmarkRun) -> tuple[bool, list[str]]:
    comparison = compare_runs(previous, current)
    warnings = list(comparison.get("warnings") or [])
    hard_warnings = {
        "different_dataset",
        "different_run_type",
        "different_question_count",
        "different_embedding_provider",
        "different_embedding_model",
        "different_fallback_in_use",
        "different_config_hash",
        "different_dataset_version",
        "different_route_policy_config",
        "different_sample_count",
    }
    missing_required = []
    for label, value in (
        ("current_config_hash_missing", current.config_hash),
        ("previous_config_hash_missing", previous.config_hash),
        ("current_dataset_version_missing", current.dataset_version),
        ("previous_dataset_version_missing", previous.dataset_version),
    ):
        if value in (None, ""):
            missing_required.append(label)
    scope_warnings = warnings + missing_required
    return not any(warning in hard_warnings for warning in warnings) and not missing_required, scope_warnings


def _latest_comparable_previous(current: BenchmarkRun, candidates: list[BenchmarkRun]) -> tuple[BenchmarkRun | None, list[str]]:
    rejected_reasons: list[str] = []
    for previous in sorted((run for run in candidates if run.run_id != current.run_id), key=lambda run: run.timestamp, reverse=True):
        comparable, warnings = _same_metric_comparison_scope(current, previous)
        if comparable:
            return previous, warnings
        rejected_reasons.append(f"{previous.run_id}: {', '.join(warnings) if warnings else 'unknown_scope_mismatch'}")
    return None, rejected_reasons[:8]


def _validate_run(runs: list[BenchmarkRun], benchmark: str) -> FormalRunStatus:
    candidates = _full_nonfallback_runs(runs, benchmark)
    if not candidates:
        return FormalRunStatus(
            benchmark=benchmark,
            status="missing",
            errors=["no full non-fallback artifact with expected question count"],
        )
    current = latest_run(candidates, project="DysonSpherain", dataset=benchmark)
    previous_candidates = [run for run in candidates if run.run_id != current.run_id]
    previous, comparison_warnings = _latest_comparable_previous(current, previous_candidates) if previous_candidates else (None, [])
    warnings: list[str] = []
    errors: list[str] = []

    if current.fallback_in_use is not False:
        errors.append("fallback_in_use is not false")
    provider_text = f"{current.embedding_provider or ''} {current.embedding_model or ''}".lower()
    if "local_hash" in provider_text or "local-hash" in provider_text:
        errors.append("local_hash embedding is not allowed for formal full benchmark")
    if (current.question_count or 0) < EXPECTED_QUESTIONS[benchmark]:
        errors.append(f"question_count below expected full size {EXPECTED_QUESTIONS[benchmark]}")
    if current.command is None:
        warnings.append("exact command missing")
    if current.config_hash is None:
        warnings.append("config_hash missing")
    if current.dataset_version is None:
        warnings.append("dataset_version missing")
    if current.code_commit is None:
        warnings.append("code_commit missing")

    elapsed = float(current.elapsed_seconds or 0.0)
    redline = REDLINES_SECONDS[benchmark]
    if elapsed and elapsed > float(redline.get("fail", 0.0)):
        errors.append(f"elapsed_seconds exceeds fail redline {redline['fail']}")
    elif elapsed and redline.get("warning") and elapsed > float(redline["warning"]):
        warnings.append(f"elapsed_seconds exceeds warning redline {redline['warning']}")

    comparison_status = "matched" if previous is not None else ("non_comparable" if previous_candidates else "no_previous_run")
    deltas = _metric_deltas(current, previous)
    for key, delta in sorted(deltas.items()):
        if key in {"recall_any@10", "recall_frac@10", "final_recall@10", "ndcg_any@10", "final_ndcg@10", "candidate_recall@100"} and delta < -1e-9:
            warnings.append(f"{key} decreased vs comparable previous full non-fallback run by {delta:.6f}")

    status = "failed" if errors else ("warning" if warnings else "passed")
    return FormalRunStatus(
        benchmark=benchmark,
        status=status,
        run_id=current.run_id,
        artifact_dir=current.artifact_dir,
        question_count=current.question_count,
        elapsed_seconds=current.elapsed_seconds,
        fallback_in_use=current.fallback_in_use,
        embedding_provider=current.embedding_provider,
        embedding_model=current.embedding_model,
        metrics=current.metrics,
        warnings=warnings,
        errors=errors,
        compared_to_run_id=previous.run_id if previous else None,
        comparison_status=comparison_status,
        comparison_warnings=comparison_warnings,
        quality_deltas=deltas,
    )


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _section_status(path: Path, section: str, available_key: str = "available", total_key: str = "total") -> dict[str, Any]:
    payload = _load(path)
    data = payload.get(section) if isinstance(payload.get(section), dict) else {}
    available = int(data.get(available_key) or data.get("available_records") or 0)
    total = int(data.get(total_key) or data.get("records") or 0)
    if section == "efficiency" and int(data.get("expected_sweeps") or 0):
        available = int(data.get("formal_sweep_available") or 0)
        total = int(data.get("expected_sweeps") or 0)
    blocked = int(data.get("blocked") or 0)
    effective_total = max(0, total - blocked)
    pending = max(0, effective_total - available)
    status = "passed" if effective_total and pending == 0 else "pending"
    return {
        "status": status,
        "available": available,
        "blocked": blocked,
        "total": total,
        "effective_total": effective_total,
        "pending": pending,
    }


def build_report(*, base_dir: Path) -> dict[str, Any]:
    runs = load_registry(base_dir)
    full_runs = [_validate_run(runs, benchmark) for benchmark in EXPECTED_QUESTIONS]
    combined_elapsed = sum(float(run.elapsed_seconds or 0.0) for run in full_runs if run.status != "missing")
    combined_status = "passed"
    combined_warnings: list[str] = []
    combined_errors: list[str] = []
    if combined_elapsed > COMBINED_REDLINE_SECONDS["fail"]:
        combined_status = "failed"
        combined_errors.append(f"combined elapsed exceeds fail redline {COMBINED_REDLINE_SECONDS['fail']}")
    elif combined_elapsed > COMBINED_REDLINE_SECONDS["warning"]:
        combined_status = "warning"
        combined_warnings.append(f"combined elapsed exceeds warning redline {COMBINED_REDLINE_SECONDS['warning']}")

    gap = base_dir / "artifacts" / "formal_evidence_gap_report.json"
    evidence = {
        "baselines": _section_status(gap, "baselines"),
        "ablations": _section_status(gap, "ablations"),
        "leave_one_benchmark_out": _section_status(gap, "leave_one_benchmark_out"),
        "statistics": _section_status(gap, "statistics", available_key="paired_delta_available", total_key="paired_delta_reports"),
        "efficiency": _section_status(gap, "efficiency", available_key="available_records", total_key="records"),
    }
    required_outputs = [
        "paper/tables/main_results.tex",
        "paper/tables/ablation_with_ci.tex",
        "paper/tables/failure_taxonomy.tex",
        "paper/tables/efficiency.tex",
        "paper/figures/data/failure_bucket_delta.csv",
        "paper/figures/data/pareto_curve_data.csv",
        "paper/figures/data/ablation_waterfall_data.csv",
        "paper/appendix/diagnostic_case_studies.md",
        "paper/appendix/reproducibility_checklist.md",
    ]
    paper_outputs = [
        {"path": path, "exists": (base_dir / path).exists(), "bytes": (base_dir / path).stat().st_size if (base_dir / path).exists() else 0}
        for path in required_outputs
    ]
    has_failures = any(run.status in {"failed", "missing"} for run in full_runs) or combined_status == "failed"
    has_pending = any(item["status"] == "pending" for item in evidence.values()) or any(not item["exists"] for item in paper_outputs)
    has_warnings = any(run.status == "warning" for run in full_runs) or combined_status == "warning"
    overall = "failed" if has_failures else ("pending" if has_pending else ("warning" if has_warnings else "passed"))
    return {
        "schema": "dysonspherain.formal_protocol_validation.v1",
        "overall_status": overall,
        "full_benchmarks": [asdict(run) for run in full_runs],
        "combined_elapsed_seconds": combined_elapsed,
        "combined_redline_status": {
            "status": combined_status,
            "warnings": combined_warnings,
            "errors": combined_errors,
            "warning_seconds": COMBINED_REDLINE_SECONDS["warning"],
            "fail_seconds": COMBINED_REDLINE_SECONDS["fail"],
        },
        "evidence_sections": evidence,
        "paper_outputs": paper_outputs,
        "formal_use_warning": "Do not promote pending, fallback, local_hash, partial, or quality-regressed artifacts as formal results.",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Formal Protocol Validation",
        "",
        f"Overall status: `{report['overall_status']}`",
        "",
        "## Full Benchmarks",
        "",
        "| benchmark | status | q | elapsed_s | fallback | run_id | comparison | warnings | errors |",
        "|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in report["full_benchmarks"]:
        comparison = row.get("comparison_status") or ""
        if row.get("compared_to_run_id"):
            comparison = f"{comparison}: {row.get('compared_to_run_id')}"
        lines.append(
            f"| {row['benchmark']} | {row['status']} | {row.get('question_count') or ''} | {row.get('elapsed_seconds') or ''} | "
            f"{row.get('fallback_in_use')} | {row.get('run_id') or ''} | {comparison} | {'; '.join(row.get('warnings') or [])} | {'; '.join(row.get('errors') or [])} |"
        )
    lines.extend(
        [
            "",
            "## Benchmark Comparison Scope",
            "",
            "Quality deltas are only computed against a previous full non-fallback run with matching benchmark, run type, question scope, embedding, fallback status, config hash, dataset version, and route policy. Non-comparable previous runs are skipped instead of being reported as regressions.",
            "",
            "| benchmark | comparison_status | compared_to | non_comparable_reasons |",
            "|---|---|---|---|",
        ]
    )
    for row in report["full_benchmarks"]:
        lines.append(
            f"| {row['benchmark']} | {row.get('comparison_status') or ''} | {row.get('compared_to_run_id') or ''} | "
            f"{'; '.join(row.get('comparison_warnings') or [])} |"
        )
    lines.extend(["", "## Evidence Sections", "", "| section | status | available | blocked | total | pending |", "|---|---|---:|---:|---:|---:|"])
    for name, section in report["evidence_sections"].items():
        lines.append(
            f"| {name} | {section['status']} | {section['available']} | {section.get('blocked', 0)} | "
            f"{section['total']} | {section['pending']} |"
        )
    lines.extend(["", "## Paper Outputs", "", "| path | exists | bytes |", "|---|---:|---:|"])
    for item in report["paper_outputs"]:
        lines.append(f"| {item['path']} | {item['exists']} | {item['bytes']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate formal full benchmark protocol readiness from artifacts.")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("artifacts/formal_protocol_validation.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/formal_protocol_validation.md"))
    args = parser.parse_args()
    report = build_report(base_dir=args.base_dir.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.report)
    print(json.dumps({"status": report["overall_status"], "out": str(args.out), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
