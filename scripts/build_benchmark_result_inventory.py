#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


FORMAL_MARKERS = (
    "matched_baseline_full",
    "formal_ablation_full",
    "phase14_lobo_full",
    "phase11_knowme_full",
    "artifact_oracle_baselines",
    "artifact_rrf_baselines",
    "bm25_full",
)
DIAGNOSTIC_MARKERS = (
    "phase5",
    "phase11",
    "lexical_anchor_gate",
    "evidence_blend",
    "parent_anchor",
    "parent_tail",
    "dense_anchor",
    "guard",
    "probe",
    "sample",
    "medium",
    "smoke",
)
ARCHIVE_LOW_PRIORITY_MARKERS = (
    "smoke",
    "sample",
    "probe",
    "alpha",
    "guard_off",
    "supplemental",
    "parent_tail_medium",
)
METRICS_NAMES = {"metrics.json", "merged_metrics.json", "compact_metrics.json"}
DIAGNOSTIC_NAMES = {
    "candidate_recall.json",
    "failure_taxonomy.json",
    "oracle_retrieval.json",
    "performance_cache.json",
}


@dataclass
class ResultDirectory:
    name: str
    path: str
    size_bytes: int
    size_human: str
    category: str
    keep_priority: str
    reason: str
    metrics_files: int = 0
    diagnostic_files: int = 0
    failure_files: int = 0
    benchmarks: list[str] = field(default_factory=list)
    supports_formal_protocol: bool = False
    supports_paper_tables: bool = False
    archive_candidate: bool = False


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}T"


def _dir_size(path: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in {".git", ".venv", ".venv312"}]
        for filename in filenames:
            try:
                total += (Path(dirpath) / filename).stat().st_size
            except OSError:
                continue
    return total


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size > 25 * 1024 * 1024:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _referenced_roots(paths: list[Path], result_root: Path) -> set[str]:
    roots: set[str] = set()
    for path in paths:
        payload = _load_json(path)
        stack: list[Any] = [payload]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
            elif isinstance(current, str) and str(result_root) in current:
                try:
                    rel = Path(current).resolve().relative_to(result_root.resolve())
                except Exception:
                    continue
                if rel.parts:
                    roots.add(rel.parts[0])
    return roots


def _scan_counts(path: Path) -> tuple[int, int, int, list[str]]:
    metrics = 0
    diagnostics = 0
    failures = 0
    benchmarks: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in {".git", ".venv", ".venv312"}]
        lower_parts = " ".join(part.lower() for part in Path(dirpath).parts)
        for benchmark in ("longmemeval", "locomo", "knowme", "clonemem", "convomem"):
            if benchmark in lower_parts:
                benchmarks.add(benchmark)
        for filename in filenames:
            lower = filename.lower()
            if lower in METRICS_NAMES:
                metrics += 1
            if any(lower.endswith(name) for name in DIAGNOSTIC_NAMES):
                diagnostics += 1
            if "failures" in lower or "failure" in lower:
                failures += 1
    return metrics, diagnostics, failures, sorted(benchmarks)


def _classify(name: str, formal_refs: set[str], paper_refs: set[str], metrics: int, diagnostics: int) -> tuple[str, str, str, bool]:
    lower = name.lower()
    if name == "20260429_phase5_lexical_anchor_gate_protected_top3_full_v1":
        reason = "current CloneMem route-only promoted full artifact referenced by formal protocol validation"
        return "formal_evidence", "keep", reason, False
    if name == "20260428_clonemem_lexical_anchor_gate_full_v1":
        reason = "rejected unprotected lexical-gate full run; retain as diagnostic/provenance, not as current main full evidence"
        return "diagnostic_or_failed_experiment", "review", reason, False
    if name == "20260429_phase5_lexical_anchor_gate_protected_top3_medium_100k_en_v1":
        reason = "matched medium validation for promoted gate; useful provenance but lower priority than the full promoted artifact"
        return "archive_low_priority", "archive_candidate", reason, True
    formal = name in formal_refs or any(marker in lower for marker in FORMAL_MARKERS)
    paper = name in paper_refs and formal
    low_priority = any(marker in lower for marker in ARCHIVE_LOW_PRIORITY_MARKERS) and not formal
    diagnostic = diagnostics > 0 or any(marker in lower for marker in DIAGNOSTIC_MARKERS)
    if low_priority:
        reason = "exploratory smoke/sample/probe artifact; preserve until archived, but not needed as a primary formal evidence root"
        return "archive_low_priority", "archive_candidate", reason, True
    if formal or paper:
        reason = "referenced by formal/paper artifacts or matches formal evidence naming"
        return "formal_evidence", "keep", reason, False
    if diagnostic:
        reason = "diagnostic or failed-experiment evidence; keep until claims are finalized"
        return "diagnostic_or_failed_experiment", "review", reason, low_priority
    if metrics:
        reason = "has metrics but is not referenced by current formal reports"
        return "archive_low_priority", "archive_candidate", reason, True
    return "archive_low_priority", "archive_candidate", "no metrics or diagnostics detected at inventory depth", True


def build_inventory(result_root: Path, base_dir: Path) -> dict[str, Any]:
    formal_refs = _referenced_roots(
        [
            base_dir / "artifacts" / "formal_protocol_validation.json",
            base_dir / "artifacts" / "formal_evidence_gap_report.json",
            base_dir / "artifacts" / "baselines" / "baseline_runs.json",
            base_dir / "artifacts" / "ablations" / "ablation_runs.json",
            base_dir / "artifacts" / "lobo" / "lobo_protocol.json",
        ],
        result_root,
    )
    paper_refs = _referenced_roots(
        [
            base_dir / "reports" / "NEURIPS_UPGRADE_SUMMARY.md",
            base_dir / "reports" / "REPRODUCIBILITY_PACKAGE.md",
            base_dir / "reports" / "formal_protocol_validation.md",
            base_dir / "reports" / "formal_evidence_gap_report.md",
        ],
        result_root,
    )
    rows: list[ResultDirectory] = []
    for child in sorted(result_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        metrics, diagnostics, failures, benchmarks = _scan_counts(child)
        category, priority, reason, archive_candidate = _classify(child.name, formal_refs, paper_refs, metrics, diagnostics)
        size = _dir_size(child)
        rows.append(
            ResultDirectory(
                name=child.name,
                path=str(child),
                size_bytes=size,
                size_human=_human_size(size),
                category=category,
                keep_priority=priority,
                reason=reason,
                metrics_files=metrics,
                diagnostic_files=diagnostics,
                failure_files=failures,
                benchmarks=benchmarks,
                supports_formal_protocol=child.name in formal_refs,
                supports_paper_tables=child.name in paper_refs,
                archive_candidate=archive_candidate,
            )
        )
    summary: dict[str, Any] = {
        "schema": "dysonspherain.benchmark_result_inventory.v1",
        "result_root": str(result_root),
        "total_size_bytes": sum(row.size_bytes for row in rows),
        "total_size_human": _human_size(sum(row.size_bytes for row in rows)),
        "counts_by_category": {},
        "size_by_category": {},
        "formal_referenced_roots": sorted(formal_refs),
        "paper_referenced_roots": sorted(paper_refs),
        "records": [asdict(row) for row in rows],
    }
    for row in rows:
        summary["counts_by_category"][row.category] = summary["counts_by_category"].get(row.category, 0) + 1
        summary["size_by_category"][row.category] = summary["size_by_category"].get(row.category, 0) + row.size_bytes
    summary["size_by_category_human"] = {key: _human_size(value) for key, value in summary["size_by_category"].items()}
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = summary["records"]
    lines = [
        "# BenchmarkResult Inventory",
        "",
        "This inventory classifies result directories for evidence management. It does not mark any directory for deletion.",
        "",
        f"- result_root: `{summary['result_root']}`",
        f"- total_size: `{summary['total_size_human']}`",
        f"- categories: `{summary['counts_by_category']}`",
        f"- category_sizes: `{summary['size_by_category_human']}`",
        "",
        "## Formal Evidence",
        "",
        "| directory | size | benchmarks | metrics | diagnostics | reason |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in records:
        if row["category"] != "formal_evidence":
            continue
        lines.append(
            f"| `{row['name']}` | {row['size_human']} | {', '.join(row['benchmarks'])} | "
            f"{row['metrics_files']} | {row['diagnostic_files']} | {row['reason']} |"
        )
    lines.extend(["", "## Diagnostic Or Failed Experiments", "", "| directory | size | benchmarks | metrics | diagnostics | archive_candidate | reason |", "|---|---:|---|---:|---:|---:|---|"])
    for row in records:
        if row["category"] != "diagnostic_or_failed_experiment":
            continue
        lines.append(
            f"| `{row['name']}` | {row['size_human']} | {', '.join(row['benchmarks'])} | "
            f"{row['metrics_files']} | {row['diagnostic_files']} | {row['archive_candidate']} | {row['reason']} |"
        )
    lines.extend(["", "## Archive Low Priority", "", "| directory | size | benchmarks | metrics | diagnostics | reason |", "|---|---:|---|---:|---:|---|"])
    for row in records:
        if row["category"] != "archive_low_priority":
            continue
        lines.append(
            f"| `{row['name']}` | {row['size_human']} | {', '.join(row['benchmarks'])} | "
            f"{row['metrics_files']} | {row['diagnostic_files']} | {row['reason']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a classified inventory of BenchmarkResult directories.")
    parser.add_argument("--result-root", type=Path, default=Path("../BenchmarkResult"))
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmark_result_inventory.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/benchmark_result_inventory.md"))
    args = parser.parse_args()
    summary = build_inventory(args.result_root.resolve(), args.base_dir.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(summary, args.report)
    print(json.dumps({"records": len(summary["records"]), "out": str(args.out), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
