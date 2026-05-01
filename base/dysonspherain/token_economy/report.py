from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .metrics import TokenEconomySample, detect_token_economy_failures, summarize_samples


MODE_COMPARISON_FIELDS = [
    "mode",
    "baseline_type",
    "context_token_budget",
    "sample_count",
    "avg_raw_history_tokens",
    "avg_final_prompt_tokens",
    "avg_saved_tokens_abs",
    "avg_saved_tokens_ratio",
    "median_saved_tokens_ratio",
    "p10_saved_tokens_ratio",
    "p90_saved_tokens_ratio",
    "avg_latency_seconds",
]

TRADEOFF_FIELDS = [
    "mode",
    "baseline_type",
    "context_token_budget",
    "sample_id",
    "final_prompt_tokens",
    "saved_tokens_ratio",
    "recall_at_5",
    "recall_at_10",
    "ndcg_at_10",
    "gold_rank",
    "latency_seconds",
]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _group_key(sample: TokenEconomySample) -> tuple[str, str, int | None]:
    return (sample.mode, sample.baseline_type, sample.context_token_budget)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def _mode_rows(samples: list[TokenEconomySample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted({_group_key(item) for item in samples}):
        group = [item for item in samples if _group_key(item) == key]
        ratios = [item.saved_tokens_ratio for item in group]
        rows.append(
            {
                "mode": key[0],
                "baseline_type": key[1],
                "context_token_budget": key[2],
                "sample_count": len(group),
                "avg_raw_history_tokens": _mean([float(item.raw_history_tokens) for item in group]),
                "avg_final_prompt_tokens": _mean([float(item.final_prompt_tokens) for item in group]),
                "avg_saved_tokens_abs": _mean([float(item.saved_tokens_abs) for item in group]),
                "avg_saved_tokens_ratio": _mean(ratios),
                "median_saved_tokens_ratio": _percentile(ratios, 0.5),
                "p10_saved_tokens_ratio": _percentile(ratios, 0.1),
                "p90_saved_tokens_ratio": _percentile(ratios, 0.9),
                "avg_latency_seconds": _mean([item.latency_seconds for item in group]),
            }
        )
    return rows


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _tradeoff_rows(samples: list[TokenEconomySample]) -> list[dict[str, Any]]:
    rows = []
    for item in samples:
        rows.append(
            {
                "mode": item.mode,
                "baseline_type": item.baseline_type,
                "context_token_budget": item.context_token_budget,
                "sample_id": item.sample_id,
                "final_prompt_tokens": item.final_prompt_tokens,
                "saved_tokens_ratio": item.saved_tokens_ratio,
                "recall_at_5": item.retrieval_quality.recall_at_5,
                "recall_at_10": item.retrieval_quality.recall_at_10,
                "ndcg_at_10": item.retrieval_quality.ndcg_at_10,
                "gold_rank": item.retrieval_quality.gold_rank,
                "latency_seconds": item.latency_seconds,
            }
        )
    return rows


def _summary_markdown(summary: dict[str, Any], failure_cases: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Token Economy Evaluation Report",
        "",
        "## Executive Summary",
        "",
        f"- Samples: {summary.get('sample_count', 0)}",
        f"- Tokenizer: {summary.get('tokenizer_name', '')}",
        f"- Fallback tokenizer used count: {summary.get('fallback_tokenizer_used_count', 0)}",
        f"- Mean saved token ratio: {summary.get('mean_saved_tokens_ratio', 0):.4f}",
        f"- Token regression count: {summary.get('token_regression_count', 0)}",
        "",
        "## Experimental Setup",
        "",
        f"- Modes: {', '.join(summary.get('modes') or []) or 'none'}",
        f"- Baseline types: {', '.join(summary.get('baseline_types') or []) or 'none'}",
        f"- Context budgets: {summary.get('context_token_budgets') or []}",
        "",
        "## Token Savings by Mode",
        "",
    ]
    for mode, payload in (summary.get("by_mode") or {}).items():
        lines.append(f"- `{mode}`: avg_saved_tokens_ratio={payload.get('avg_saved_tokens_ratio', 0):.4f}, avg_final_prompt_tokens={payload.get('avg_final_prompt_tokens', 0):.1f}")
    lines.extend(
        [
            "",
            "## Token Breakdown",
            "",
            "- See `per_sample.jsonl` for system, instruction, query, evidence, metadata, and memory header token fields.",
            "",
            "## Budget Policy Results",
            "",
            "- See `mode_comparison.csv` for budget-level aggregates.",
            "",
            "## Quality / Token Trade-off",
            "",
            "- Token saving and retrieval quality are reported separately in `token_quality_tradeoff.csv`.",
            "",
            "## Failure Cases",
            "",
        ]
    )
    for key, rows in failure_cases.items():
        lines.append(f"- {key}: {len(rows)}")
    lines.extend(["", "## Recommendations", ""])
    if summary.get("fallback_tokenizer_used_count", 0):
        lines.append("- Install or enable tiktoken-compatible tokenization before treating token counts as official.")
    if failure_cases.get("metadata_bloat"):
        lines.append("- Compress memory headers, metadata, and repeated instructions.")
    if failure_cases.get("evidence_bloat"):
        lines.append("- Improve evidence packing before increasing token budgets.")
    if failure_cases.get("token_regression"):
        lines.append("- Inspect token regression cases before enabling automatic injection broadly.")
    if not any(failure_cases.values()):
        lines.append("- No token economy failure cases were detected at the configured thresholds.")
    lines.extend(["", "## Limitations", "", "- Baseline and quality fields are best-effort when source samples do not include oracle evidence or benchmark metrics."])
    return "\n".join(lines) + "\n"


def write_report(
    samples: list[TokenEconomySample],
    output_dir: Path,
    *,
    filename_prefix: str = "",
    low_saving_threshold: float = 0.2,
    evidence_bloat_threshold: float = 0.85,
    metadata_bloat_threshold: float = 0.25,
    quality_drop_threshold: float = 0.05,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_cases = detect_token_economy_failures(
        samples,
        low_saving_threshold=low_saving_threshold,
        evidence_bloat_threshold=evidence_bloat_threshold,
        metadata_bloat_threshold=metadata_bloat_threshold,
        quality_drop_threshold=quality_drop_threshold,
    )
    summary = summarize_samples(samples)
    summary["failure_case_counts"] = {key: len(value) for key, value in failure_cases.items()}
    prefix = filename_prefix
    per_sample_name = f"{prefix}per_sample.jsonl"
    (output_dir / per_sample_name).write_text(
        "\n".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) for item in samples) + ("\n" if samples else ""),
        encoding="utf-8",
    )
    if not prefix:
        # Backward-compatible alias for the first implementation pass.
        (output_dir / "samples.jsonl").write_text((output_dir / per_sample_name).read_text(encoding="utf-8"), encoding="utf-8")
    _write_json(output_dir / f"{prefix}summary.json", summary)
    _write_json(output_dir / f"{prefix}failure_cases.json", failure_cases)
    _write_csv(output_dir / f"{prefix}mode_comparison.csv", MODE_COMPARISON_FIELDS, _mode_rows(samples))
    _write_csv(output_dir / f"{prefix}token_quality_tradeoff.csv", TRADEOFF_FIELDS, _tradeoff_rows(samples))
    (output_dir / f"{prefix}summary.md").write_text(_summary_markdown(summary, failure_cases), encoding="utf-8")
    return summary
