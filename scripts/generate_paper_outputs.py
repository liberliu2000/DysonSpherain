#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import json
from pathlib import Path


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clonemem_route_note() -> dict:
    validation = _load(Path("artifacts/formal_protocol_validation.json"))
    for item in validation.get("full_benchmarks") or []:
        if str(item.get("benchmark") or "").lower() != "clonemem":
            continue
        artifact_dir = Path(str(item.get("artifact_dir") or ""))
        metrics = _load(artifact_dir / "merged_metrics.json")
        return {
            "run_id": item.get("run_id", ""),
            "artifact_dir": str(artifact_dir),
            "comparison_status": item.get("comparison_status", ""),
            "config_hash": metrics.get("config_hash", ""),
            "route_policy_hash": metrics.get("route_policy_hash", ""),
        }
    return {}


def write_main_table() -> None:
    baseline = _load(Path("artifacts/baselines/baseline_runs.json"))
    records = baseline.get("records", [])
    clonemem_note = _clonemem_route_note()
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    lines = [
        "% Artifact-backed baseline table. Pending rows indicate missing formal runs.",
        "% CloneMem dysonspherain_full uses the formal route-only promoted run when present.",
        (
            "% CloneMem route-policy hash: "
            f"{clonemem_note.get('route_policy_hash', '')}; comparison: {clonemem_note.get('comparison_status', '')}."
        ),
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Benchmark & Method & Status & Metrics \\\\",
        "\\midrule",
    ]
    for record in records[:80]:
        metrics = record.get("metrics") or {}
        metric_text = ", ".join(f"{k}={v}" for k, v in metrics.items()) or "pending"
        lines.append(
            f"{record.get('benchmark', '')} & {record.get('baseline', '')} & {record.get('status', '')} & {metric_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    Path("paper/tables/main_results.tex").write_text("\n".join(lines), encoding="utf-8")


def write_ablation_table() -> None:
    ablations = _load(Path("artifacts/ablations/ablation_runs.json"))
    records = ablations.get("records", [])
    lines = [
        "% Artifact-backed ablation table. Pending rows indicate missing formal runs.",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Benchmark & Ablation & Status & Metrics \\\\",
        "\\midrule",
    ]
    for record in records[:80]:
        metrics = record.get("metrics") or {}
        metric_text = ", ".join(f"{k}={v}" for k, v in metrics.items()) or "pending"
        lines.append(
            f"{record.get('benchmark', '')} & {record.get('ablation', '')} & {record.get('status', '')} & {metric_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    Path("paper/tables/ablation_with_ci.tex").write_text("\n".join(lines), encoding="utf-8")


def write_failure_taxonomy_table() -> None:
    source = Path("reports/failure_bucket_delta_by_ablation.csv")
    target = Path("paper/figures/data/failure_bucket_delta.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        target.write_text("benchmark,ablation,status\n", encoding="utf-8")

    rows = []
    if source.exists():
        with source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    lines = [
        "% Artifact-backed failure taxonomy deltas. Pending rows indicate missing formal runs.",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Benchmark & Ablation & Status & Artifact \\\\",
        "\\midrule",
    ]
    for row in rows[:80]:
        lines.append(
            f"{row.get('benchmark', '')} & {row.get('ablation', '')} & {row.get('status', '')} & {Path(row.get('metrics_path', '')).name if row.get('metrics_path') else ''} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    Path("paper/tables/failure_taxonomy.tex").write_text("\n".join(lines), encoding="utf-8")


def write_efficiency_table() -> None:
    pareto = _load(Path("artifacts/profiling/efficiency_quality_pareto.json"))
    records = [row for row in pareto.get("records", []) if isinstance(row, dict)]
    lines = [
        "% Artifact-backed efficiency table. Oversized or unreadable artifacts are retained as status rows.",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Benchmark & Status & Questions & Total ms & Artifact \\\\",
        "\\midrule",
    ]
    for record in records[:80]:
        lines.append(
            f"{record.get('benchmark', '')} & {record.get('status', '')} & {record.get('question_count', '')} & "
            f"{record.get('total_ms', '')} & {Path(record.get('artifact', '')).name if record.get('artifact') else ''} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    Path("paper/tables/efficiency.tex").write_text("\n".join(lines), encoding="utf-8")


def write_case_studies() -> None:
    diagnostics = Path("reports/diagnostics")
    cases = [
        ("LoCoMo ordering repair case", diagnostics / "locomo_ordering_failures.jsonl"),
        ("CloneMem parent-to-segment repair case", diagnostics / "clonemem_parent_hit_segment_miss_examples.jsonl"),
        ("KnowMe profile/entity admission case", diagnostics / "knowme_segment_admission_failures.jsonl"),
        ("Remaining failure case", diagnostics / "clonemem_reranker_dropped_gold_examples.jsonl"),
    ]
    Path("paper/appendix").mkdir(parents=True, exist_ok=True)
    lines = [
        "# Diagnostic Case Studies",
        "",
        "Cases are referenced from diagnostic JSONL artifacts. No unsupported examples are invented.",
        "",
    ]
    for title, path in cases:
        lines.extend([f"## {title}", ""])
        if path.exists():
            first = ""
            with path.open(encoding="utf-8") as handle:
                first = handle.readline().strip()
            lines.append(f"- source: `{path}`")
            lines.append(f"- first_record: `{first[:500]}`")
        else:
            lines.append("- pending: diagnostic source artifact is missing")
        lines.append("")
    Path("paper/appendix/diagnostic_case_studies.md").write_text("\n".join(lines), encoding="utf-8")


def write_repro_checklist() -> None:
    lines = [
        "# Reproducibility Checklist",
        "",
        "- Formal runs must use `SPHERE_EMBEDDING_FAIL_FAST=1`.",
        "- Formal runs must use `SPHERE_VECTOR_BACKEND=chroma` and `fallback_in_use=false`.",
        "- Tables must be regenerated from artifacts, not hand-filled.",
        "- Smoke, partial, fallback, and full runs must remain distinguishable.",
        "- Missing baselines/ablations stay pending until artifacts exist.",
    ]
    Path("paper/appendix/reproducibility_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_route_policy_appendix() -> None:
    note = _clonemem_route_note()
    lines = [
        "# Route Policy Comparability Note",
        "",
        "Paper-facing comparisons follow the formal protocol comparability rule: quality deltas are computed only between runs with matching benchmark, run type, question scope, embedding, fallback status, config hash, dataset version, and route policy.",
        "",
        "## CloneMem Route-Only Promotion",
        "",
        f"- run_id: `{note.get('run_id', '')}`",
        f"- artifact_dir: `{note.get('artifact_dir', '')}`",
        f"- config_hash: `{note.get('config_hash', '')}`",
        f"- route_policy_hash: `{note.get('route_policy_hash', '')}`",
        f"- formal_comparison_status: `{note.get('comparison_status', '')}`",
        "",
        "The CloneMem protected top-3 lexical anchor gate is promoted only through the CloneMem benchmark route policy. Older CloneMem full runs are not treated as matched formal comparisons when their config hash, dataset version, or route policy differs.",
        "",
    ]
    Path("paper/appendix").mkdir(parents=True, exist_ok=True)
    Path("paper/appendix/route_policy_comparability.md").write_text("\n".join(lines), encoding="utf-8")


def write_figure_data() -> None:
    target = Path("paper/figures/data/system_overview_nodes.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("CLI", "Research Evaluation Harness"),
        ("CLI", "Memory OS Interface"),
        ("Research Evaluation Harness", "Experiment Registry"),
        ("Memory OS Interface", "Runtime Context Compiler"),
        ("Experiment Registry", "Retrieval Pipeline and Diagnostics"),
        ("Runtime Context Compiler", "Retrieval Pipeline and Diagnostics"),
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target"])
        writer.writerows(rows)
    for source, name in (
        (Path("figures/pareto_curve_data.csv"), "pareto_curve_data.csv"),
        (Path("figures/ablation_waterfall_data.csv"), "ablation_waterfall_data.csv"),
    ):
        destination = target.parent / name
        if source.exists():
            shutil.copyfile(source, destination)
        else:
            destination.write_text("status\npending\n", encoding="utf-8")


def main() -> None:
    write_main_table()
    write_ablation_table()
    write_failure_taxonomy_table()
    write_efficiency_table()
    write_case_studies()
    write_repro_checklist()
    write_route_policy_appendix()
    write_figure_data()
    print(json.dumps({"status": "ok", "paper_outputs": 10}, ensure_ascii=False))


if __name__ == "__main__":
    main()
