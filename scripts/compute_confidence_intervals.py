#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


MAX_METRICS_BYTES = 25 * 1024 * 1024
DEFAULT_METRICS = [
    "recall@5",
    "recall@10",
    "Recall@5",
    "Recall@10",
    "recall_any@5",
    "recall_any@10",
    "recall_frac@10",
    "final_recall@10",
    "ndcg@10",
    "NDCG@10",
    "ndcg_any@10",
    "final_ndcg@10",
    "candidate_recall@100",
]


@dataclass
class MetricCI:
    metric: str
    n: int
    mean: float
    ci_low: float
    ci_high: float


@dataclass
class StatisticsReport:
    status: str
    metrics_path: str
    sample_unit: str
    resamples: int
    random_seed: int
    metric_cis: list[MetricCI]
    warnings: list[str]


@dataclass
class PairedDeltaCI:
    metric: str
    n: int
    mean_delta: float
    ci_low: float
    ci_high: float
    wins: int
    ties: int
    losses: int


@dataclass
class PairedStatisticsReport:
    status: str
    a_metrics_path: str
    b_metrics_path: str
    sample_unit: str
    resamples: int
    random_seed: int
    pair_key: str
    metric_deltas: list[PairedDeltaCI]
    warnings: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_METRICS_BYTES:
        return {"_oversized": path.stat().st_size}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_error": "top-level JSON is not an object"}


def _row_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    keys = ("per_question", "per_question_metrics", "question_results", "question_rows", "results", "rows")
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        for key in keys:
            value = diagnostics.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _find_metric_value(row, metric)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _find_metric_value(payload: Any, metric: str) -> Any:
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
            found = _find_metric_value(nested, metric)
            if found is not None:
                return found
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_metric_value(value, metric)
            if found is not None:
                return found
    return None


def bootstrap_ci(values: list[float], *, resamples: int, seed: int) -> tuple[float, float, float]:
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(resamples):
        estimates.append(mean(values[rng.randrange(n)] for _ in range(n)))
    estimates.sort()
    low_idx = max(0, int(0.025 * len(estimates)) - 1)
    high_idx = min(len(estimates) - 1, int(0.975 * len(estimates)))
    return mean(values), estimates[low_idx], estimates[high_idx]


def _bootstrap_delta_ci(deltas: list[float], *, resamples: int, seed: int) -> tuple[float, float, float]:
    if not deltas:
        raise ValueError("no deltas")
    rng = random.Random(seed)
    n = len(deltas)
    estimates = []
    for _ in range(resamples):
        estimates.append(mean(deltas[rng.randrange(n)] for _ in range(n)))
    estimates.sort()
    low_idx = max(0, int(0.025 * len(estimates)) - 1)
    high_idx = min(len(estimates) - 1, int(0.975 * len(estimates)))
    return mean(deltas), estimates[low_idx], estimates[high_idx]


def _row_pair_key(row: dict[str, Any], index: int, *, pair_key: str = "auto") -> str:
    if pair_key != "auto":
        value = row.get(pair_key)
        if value is not None:
            return str(value)
    sample_id = row.get("sample_id")
    question_id = row.get("question_id") or row.get("query_id")
    if sample_id is not None and question_id is not None:
        return f"{sample_id}::{question_id}"
    for key in ("question_id", "query_id", "sample_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return f"__index__:{index}"


def _rows_by_pair_key(rows: list[dict[str, Any]], *, pair_key: str = "auto") -> dict[str, dict[str, Any]]:
    counts: dict[str, int] = {}
    keyed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        base_key = _row_pair_key(row, index, pair_key=pair_key)
        occurrence = counts.get(base_key, 0)
        counts[base_key] = occurrence + 1
        key = base_key if occurrence == 0 else f"{base_key}::__occurrence__:{occurrence}"
        keyed[key] = row
    return keyed


def compute_paired_delta_report(
    a_metrics_path: Path,
    b_metrics_path: Path,
    *,
    metrics: list[str] | None = None,
    resamples: int = 1000,
    seed: int = 13,
    sample_unit: str = "question",
    pair_key: str = "auto",
) -> PairedStatisticsReport:
    warnings: list[str] = []
    a_payload = _load_json(a_metrics_path)
    b_payload = _load_json(b_metrics_path)
    for label, payload in (("a", a_payload), ("b", b_payload)):
        if "_oversized" in payload:
            warnings.append(f"{label} metrics.json is {payload['_oversized']} bytes; use compact artifacts")
        if "_error" in payload:
            warnings.append(f"{label} metrics error: {payload['_error']}")
    if warnings:
        return PairedStatisticsReport(
            status="skipped",
            a_metrics_path=str(a_metrics_path),
            b_metrics_path=str(b_metrics_path),
            sample_unit=sample_unit,
            resamples=resamples,
            random_seed=seed,
            pair_key=pair_key,
            metric_deltas=[],
            warnings=warnings,
        )
    a_rows = _row_candidates(a_payload)
    b_rows = _row_candidates(b_payload)
    if not a_rows or not b_rows:
        return PairedStatisticsReport(
            status="pending",
            a_metrics_path=str(a_metrics_path),
            b_metrics_path=str(b_metrics_path),
            sample_unit=sample_unit,
            resamples=resamples,
            random_seed=seed,
            pair_key=pair_key,
            metric_deltas=[],
            warnings=["both artifacts need per-question rows for paired delta CI"],
        )
    a_by_key = _rows_by_pair_key(a_rows, pair_key=pair_key)
    b_by_key = _rows_by_pair_key(b_rows, pair_key=pair_key)
    shared_keys = sorted(set(a_by_key) & set(b_by_key))
    if not shared_keys:
        return PairedStatisticsReport(
            status="pending",
            a_metrics_path=str(a_metrics_path),
            b_metrics_path=str(b_metrics_path),
            sample_unit=sample_unit,
            resamples=resamples,
            random_seed=seed,
            pair_key=pair_key,
            metric_deltas=[],
            warnings=["no shared row identifiers found for paired comparison"],
        )
    metric_deltas: list[PairedDeltaCI] = []
    for index, metric in enumerate(metrics or DEFAULT_METRICS):
        deltas: list[float] = []
        wins = ties = losses = 0
        for key in shared_keys:
            a_value = _find_metric_value(a_by_key[key], metric)
            b_value = _find_metric_value(b_by_key[key], metric)
            if isinstance(a_value, bool):
                a_value = 1.0 if a_value else 0.0
            if isinstance(b_value, bool):
                b_value = 1.0 if b_value else 0.0
            if not isinstance(a_value, (int, float)) or not isinstance(b_value, (int, float)):
                continue
            delta = float(b_value) - float(a_value)
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
        if not deltas:
            continue
        mean_delta, low, high = _bootstrap_delta_ci(deltas, resamples=resamples, seed=seed + index)
        metric_deltas.append(
            PairedDeltaCI(
                metric=metric,
                n=len(deltas),
                mean_delta=mean_delta,
                ci_low=low,
                ci_high=high,
                wins=wins,
                ties=ties,
                losses=losses,
            )
        )
    if not metric_deltas:
        warnings.append("shared rows exist but no requested metric values were found in both artifacts")
    return PairedStatisticsReport(
        status="available" if metric_deltas else "pending",
        a_metrics_path=str(a_metrics_path),
        b_metrics_path=str(b_metrics_path),
        sample_unit=sample_unit,
        resamples=resamples,
        random_seed=seed,
        pair_key=pair_key,
        metric_deltas=metric_deltas,
        warnings=warnings,
    )


def compute_report(
    metrics_path: Path,
    *,
    metrics: list[str] | None = None,
    resamples: int = 1000,
    seed: int = 13,
    sample_unit: str = "question",
) -> StatisticsReport:
    payload = _load_json(metrics_path)
    warnings: list[str] = []
    if "_oversized" in payload:
        return StatisticsReport(
            status="skipped",
            metrics_path=str(metrics_path),
            sample_unit=sample_unit,
            resamples=resamples,
            random_seed=seed,
            metric_cis=[],
            warnings=[f"metrics.json is {payload['_oversized']} bytes; use a compact per-question artifact"],
        )
    if "_error" in payload:
        return StatisticsReport(
            status="error",
            metrics_path=str(metrics_path),
            sample_unit=sample_unit,
            resamples=resamples,
            random_seed=seed,
            metric_cis=[],
            warnings=[str(payload["_error"])],
        )
    rows = _row_candidates(payload)
    if not rows:
        return StatisticsReport(
            status="pending",
            metrics_path=str(metrics_path),
            sample_unit=sample_unit,
            resamples=resamples,
            random_seed=seed,
            metric_cis=[],
            warnings=["no per-question rows found; cannot compute bootstrap CI"],
        )
    metric_cis: list[MetricCI] = []
    for index, metric in enumerate(metrics or DEFAULT_METRICS):
        values = _metric_values(rows, metric)
        if not values:
            continue
        m, low, high = bootstrap_ci(values, resamples=resamples, seed=seed + index)
        metric_cis.append(MetricCI(metric=metric, n=len(values), mean=m, ci_low=low, ci_high=high))
    if not metric_cis:
        warnings.append("per-question rows exist but no requested metric values were found")
    return StatisticsReport(
        status="available" if metric_cis else "pending",
        metrics_path=str(metrics_path),
        sample_unit=sample_unit,
        resamples=resamples,
        random_seed=seed,
        metric_cis=metric_cis,
        warnings=warnings,
    )


def write_markdown(reports: list[StatisticsReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Statistical Significance Report",
        "",
        "Bootstrap confidence intervals are computed only from artifact-backed per-question rows.",
        "",
        "| metrics_path | status | sample_unit | metric | n | mean | ci_low | ci_high | warnings |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for report in reports:
        if not report.metric_cis:
            lines.append(f"| {report.metrics_path} | {report.status} | {report.sample_unit} |  |  |  |  |  | {'; '.join(report.warnings)} |")
            continue
        for ci in report.metric_cis:
            lines.append(
                f"| {report.metrics_path} | {report.status} | {report.sample_unit} | {ci.metric} | {ci.n} | "
                f"{ci.mean:.6f} | {ci.ci_low:.6f} | {ci.ci_high:.6f} | {'; '.join(report.warnings)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paired_delta_markdown(reports: list[PairedStatisticsReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paired Delta Confidence Intervals",
        "",
        "Deltas are computed as B minus A over shared per-question rows.",
        "",
        "| a_metrics_path | b_metrics_path | status | metric | n | mean_delta | ci_low | ci_high | wins | ties | losses | warnings |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for report in reports:
        if not report.metric_deltas:
            lines.append(
                f"| {report.a_metrics_path} | {report.b_metrics_path} | {report.status} |  |  |  |  |  |  |  |  | "
                f"{'; '.join(report.warnings)} |"
            )
            continue
        for delta in report.metric_deltas:
            lines.append(
                f"| {report.a_metrics_path} | {report.b_metrics_path} | {report.status} | {delta.metric} | {delta.n} | "
                f"{delta.mean_delta:.6f} | {delta.ci_low:.6f} | {delta.ci_high:.6f} | "
                f"{delta.wins} | {delta.ties} | {delta.losses} | {'; '.join(report.warnings)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_run_label(metrics_path: str) -> str:
    stem = Path(metrics_path).stem
    if stem.startswith("compact_"):
        stem = stem[len("compact_") :]
    return stem.replace("_", "\\_")


def write_tex(reports: list[StatisticsReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated from artifact-backed bootstrap outputs. Pending rows are omitted.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Run & Metric & Mean & CI Low & CI High \\\\",
        "\\midrule",
    ]
    for report in reports:
        for ci in report.metric_cis:
            run_name = tex_run_label(report.metrics_path)
            metric = ci.metric.replace("_", "\\_")
            lines.append(f"{run_name} & {metric} & {ci.mean:.4f} & {ci.ci_low:.4f} & {ci.ci_high:.4f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute bootstrap CIs from per-question metric artifacts.")
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics.json files")
    parser.add_argument("--out", type=Path, default=Path("artifacts/statistics/bootstrap_ci.json"), help="Output JSON path")
    parser.add_argument("--report", type=Path, default=Path("reports/statistical_significance_report.md"), help="Markdown report path")
    parser.add_argument("--tex", type=Path, default=Path("paper/tables/main_results_with_ci.tex"), help="LaTeX table path")
    parser.add_argument("--metric", action="append", dest="metrics_filter", help="Metric to include; may repeat")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample-unit", default="question")
    parser.add_argument("--paired-a", type=Path, action="append", help="Baseline compact metrics for paired delta CI; may repeat")
    parser.add_argument("--paired-b", type=Path, action="append", help="Comparison compact metrics for paired delta CI; may repeat")
    parser.add_argument("--paired-out", type=Path, default=Path("artifacts/statistics/paired_delta_ci.json"))
    parser.add_argument("--paired-report", type=Path, default=Path("reports/paired_delta_confidence_report.md"))
    parser.add_argument("--pair-key", default="auto")
    args = parser.parse_args()

    reports = [
        compute_report(path, metrics=args.metrics_filter, resamples=args.resamples, seed=args.seed, sample_unit=args.sample_unit)
        for path in args.metrics
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "dysonspherain.bootstrap_ci.v1", "reports": [asdict(report) for report in reports]}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(reports, args.report)
    write_tex(reports, args.tex)
    paired_reports: list[PairedStatisticsReport] = []
    paired_a = args.paired_a or []
    paired_b = args.paired_b or []
    if paired_a or paired_b:
        if len(paired_a) != len(paired_b):
            parser.error("--paired-a and --paired-b must be supplied the same number of times")
        paired_reports = [
            compute_paired_delta_report(
                a_path,
                b_path,
                metrics=args.metrics_filter,
                resamples=args.resamples,
                seed=args.seed,
                sample_unit=args.sample_unit,
                pair_key=args.pair_key,
            )
            for a_path, b_path in zip(paired_a, paired_b)
        ]
        args.paired_out.parent.mkdir(parents=True, exist_ok=True)
        args.paired_out.write_text(
            json.dumps(
                {"schema": "dysonspherain.paired_delta_ci.v1", "reports": [asdict(report) for report in paired_reports]},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        write_paired_delta_markdown(paired_reports, args.paired_report)
    print(
        json.dumps(
            {
                "reports": len(reports),
                "paired_reports": len(paired_reports),
                "out": str(args.out),
                "report": str(args.report),
                "tex": str(args.tex),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
