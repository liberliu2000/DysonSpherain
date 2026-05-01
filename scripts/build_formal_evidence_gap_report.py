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


def _records(path: Path) -> list[dict[str, Any]]:
    payload = _load(path)
    records = payload.get("records") or payload.get("rows")
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _summarize(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    available = [record for record in records if record.get("status") == "available"]
    blocked = [record for record in records if record.get("status") == "blocked"]
    pending = [record for record in records if record.get("status") not in {"available", "blocked"}]
    return {
        "total": len(records),
        "available": len(available),
        "blocked": len(blocked),
        "pending": len(pending),
        "pending_items": [
            {
                "benchmark": record.get("benchmark") or record.get("held_out"),
                key: record.get(key),
                "status": record.get("status"),
                "warnings": record.get("warnings"),
            }
            for record in pending
        ],
        "blocked_items": [
            {
                "benchmark": record.get("benchmark") or record.get("held_out"),
                key: record.get(key),
                "status": record.get("status"),
                "warnings": record.get("warnings"),
            }
            for record in blocked
        ],
    }


def build_report(
    *,
    baselines_path: Path,
    ablations_path: Path,
    lobo_path: Path,
    paired_delta_path: Path,
    pareto_path: Path,
) -> dict[str, Any]:
    baseline_records = _records(baselines_path)
    ablation_records = _records(ablations_path)
    lobo_rows = _records(lobo_path)
    paired = _load(paired_delta_path)
    pareto = _load(pareto_path)
    paired_reports = paired.get("reports") if isinstance(paired.get("reports"), list) else []
    pareto_records = pareto.get("records") if isinstance(pareto.get("records"), list) else []
    sweep_rows = pareto.get("sweep_status") if isinstance(pareto.get("sweep_status"), list) else []
    formal_sweeps = [row for row in sweep_rows if isinstance(row, dict) and row.get("status") == "available" and row.get("formal_eligible") is True]
    return {
        "schema": "dysonspherain.formal_evidence_gap.v1",
        "baselines": _summarize(baseline_records, "baseline"),
        "ablations": _summarize(ablation_records, "ablation"),
        "leave_one_benchmark_out": {
            "total": len(lobo_rows),
            "available": sum(1 for row in lobo_rows if row.get("status") == "available"),
            "pending": sum(1 for row in lobo_rows if row.get("status") != "available"),
            "pending_items": [row for row in lobo_rows if row.get("status") != "available"],
        },
        "statistics": {
            "paired_delta_reports": len(paired_reports),
            "paired_delta_available": sum(1 for row in paired_reports if row.get("status") == "available"),
        },
        "efficiency": {
            "records": len(pareto_records),
            "available_records": sum(1 for row in pareto_records if row.get("status") == "available"),
            "expected_sweeps": len(sweep_rows),
            "formal_sweep_available": len(formal_sweeps),
            "sweep_pending_items": [row for row in sweep_rows if isinstance(row, dict) and row.get("status") != "available"],
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baselines = report["baselines"]
    ablations = report["ablations"]
    lobo = report["leave_one_benchmark_out"]
    stats = report["statistics"]
    efficiency = report["efficiency"]
    efficiency_total = efficiency.get("expected_sweeps") or efficiency["records"]
    efficiency_available = efficiency.get("formal_sweep_available") if efficiency.get("expected_sweeps") else efficiency["available_records"]
    efficiency_pending = max(0, int(efficiency_total or 0) - int(efficiency_available or 0))
    lines = [
        "# Formal Evidence Gap Report",
        "",
        "This report is generated from artifact registries and marks missing evidence as pending rather than fabricating results.",
        "",
        "## Summary",
        "",
        "| area | available | pending | total |",
        "|---|---:|---:|---:|",
        f"| baselines | {baselines['available']} | {baselines['pending']} | {baselines['total']} |",
        f"| ablations | {ablations['available']} | {ablations['pending']} | {ablations['total']} |",
        f"| leave-one-benchmark-out | {lobo['available']} | {lobo['pending']} | {lobo['total']} |",
        f"| paired delta statistics | {stats['paired_delta_available']} | {stats['paired_delta_reports'] - stats['paired_delta_available']} | {stats['paired_delta_reports']} |",
        f"| efficiency sweeps | {efficiency_available} | {efficiency_pending} | {efficiency_total} |",
        "",
        "## Pending Baselines",
        "",
    ]
    if baselines.get("blocked"):
        lines.extend(["", f"Blocked baselines: `{baselines['blocked']}`. Blocked rows carry artifact-backed reasons and are excluded from long-term pending counts.", ""])
    for item in baselines["pending_items"]:
        lines.append(f"- `{item.get('benchmark')}` `{item.get('baseline')}` status=`{item.get('status')}`")
    lines.extend(["", "## Blocked Baselines", ""])
    for item in baselines.get("blocked_items") or []:
        warnings = "; ".join(str(warning) for warning in (item.get("warnings") or []))
        lines.append(f"- `{item.get('benchmark')}` `{item.get('baseline')}` status=`{item.get('status')}` reason={warnings}")
    lines.extend(["", "## Pending Ablations", ""])
    for item in ablations["pending_items"]:
        lines.append(f"- `{item.get('benchmark')}` `{item.get('ablation')}` status=`{item.get('status')}`")
    lines.extend(["", "## Pending Leave-One-Benchmark-Out Rows", ""])
    for item in lobo["pending_items"]:
        lines.append(f"- held_out=`{item.get('held_out')}` train=`{', '.join(item.get('train_benchmarks') or [])}` status=`{item.get('status')}`")
    lines.extend(["", "## Pending Efficiency Sweeps", ""])
    for item in efficiency.get("sweep_pending_items") or []:
        lines.append(
            f"- `{item.get('sweep')}` status=`{item.get('status')}` formal_eligible=`{item.get('formal_eligible')}` artifact=`{item.get('artifact')}`"
        )
    lines.extend(
        [
            "",
            "## Raw Efficiency Artifact Scan",
            "",
            f"- available_records: `{efficiency['available_records']}`",
            f"- total_records: `{efficiency['records']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build formal evidence gap summary from generated artifacts.")
    parser.add_argument("--baselines", type=Path, default=Path("artifacts/baselines/baseline_runs.json"))
    parser.add_argument("--ablations", type=Path, default=Path("artifacts/ablations/ablation_runs.json"))
    parser.add_argument("--lobo", type=Path, default=Path("artifacts/lobo/lobo_protocol.json"))
    parser.add_argument("--paired-delta", type=Path, default=Path("artifacts/statistics/paired_delta_full_vs_dense.json"))
    parser.add_argument("--pareto", type=Path, default=Path("artifacts/profiling/efficiency_quality_pareto.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/formal_evidence_gap_report.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/formal_evidence_gap_report.md"))
    args = parser.parse_args()

    report = build_report(
        baselines_path=args.baselines,
        ablations_path=args.ablations,
        lobo_path=args.lobo,
        paired_delta_path=args.paired_delta,
        pareto_path=args.pareto,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.report)
    print(json.dumps({"out": str(args.out), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
