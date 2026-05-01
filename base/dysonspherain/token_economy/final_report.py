from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _best_mode(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "unavailable"
    def score(row: dict[str, str]) -> float:
        try:
            return float(row.get("avg_saved_tokens_ratio") or 0.0)
        except ValueError:
            return 0.0
    best = max(rows, key=score)
    return f"{best.get('mode')} / {best.get('baseline_type')} / budget={best.get('context_token_budget')} ({score(best):.4f})"


def write_final_report(token_economy_dir: Path, output: Path | None = None, *, benchmark_rerun_status: str = "") -> Path:
    output = output or token_economy_dir / "final_report.md"
    summary = _load_summary(token_economy_dir / "summary.json")
    mode_rows = _read_csv_rows(token_economy_dir / "mode_comparison.csv")
    failure_cases = json.loads((token_economy_dir / "failure_cases.json").read_text(encoding="utf-8"))
    tradeoff_rows = _read_csv_rows(token_economy_dir / "token_quality_tradeoff.csv")
    quality_available = any(row.get("recall_at_10") or row.get("ndcg_at_10") or row.get("gold_rank") for row in tradeoff_rows)
    oracle_rows = [row for row in mode_rows if row.get("baseline_type") == "oracle_minimal"]
    lines = [
        "# DysonSpherain Token Economy Final Report",
        "",
        "## Artifact Source",
        "",
        f"- Token economy artifact directory: `{token_economy_dir}`",
        f"- Benchmark rerun status: {benchmark_rerun_status or 'not recorded'}",
        "",
        "## Executive Summary",
        "",
        f"- Samples: {summary.get('sample_count', 0)}",
        f"- Modes: {', '.join(summary.get('modes') or [])}",
        f"- Baselines: {', '.join(summary.get('baseline_types') or [])}",
        f"- Context budgets: {summary.get('context_token_budgets') or []}",
        f"- Tokenizer: {summary.get('tokenizer_name')}",
        f"- Fallback tokenizer used count: {summary.get('fallback_tokenizer_used_count', 0)}",
        f"- Mean saved token ratio: {summary.get('mean_saved_tokens_ratio', 0):.4f}",
        f"- Token regression count: {summary.get('token_regression_count', 0)}",
        f"- Best average saving mode: {_best_mode(mode_rows)}",
        "",
        "## Quality Trade-off",
        "",
        f"- Retrieval quality fields available: {'yes' if quality_available else 'no'}",
        "- Token saving and retrieval quality are reported separately in `token_quality_tradeoff.csv`.",
        "",
        "## Oracle Minimal Baseline",
        "",
        f"- Oracle baseline rows available: {len(oracle_rows)}",
        "- `oracle_minimal` is a theoretical lower bound; token regression against oracle does not imply the retrieval context is worse than full history.",
        "",
        "## Failure Cases",
        "",
    ]
    for key, rows in failure_cases.items():
        lines.append(f"- {key}: {len(rows)}")
    lines.extend(["", "## Prompt Part Waste", ""])
    counts = summary.get("failure_case_counts") or {}
    if counts.get("metadata_bloat"):
        lines.append("- Metadata/instruction/header overhead is a detected bloat source; compress static headers before broad deployment.")
    if counts.get("evidence_bloat"):
        lines.append("- Evidence dominates some final prompts; improve evidence packing before raising budgets.")
    if not counts.get("metadata_bloat") and not counts.get("evidence_bloat"):
        lines.append("- No evidence or metadata bloat was detected at configured thresholds.")
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "- Treat conservative mode as the default only when quality metrics remain stable for the same benchmark surface.",
            "- Use exploratory mode for diagnostic or creative expansion, not as a default token-saving policy.",
            "- Keep reporting token savings and retrieval quality as separate metrics.",
            "- Re-run on official full benchmark artifacts after any retrieval pipeline change.",
            "",
            "## Limitations",
            "",
        ]
    )
    if summary.get("fallback_tokenizer_used_count", 0):
        lines.append("- Some samples used fallback tokenization; token counts are not official.")
    else:
        lines.append("- Token counts used the configured tokenizer without fallback.")
    lines.append("- Some benchmark artifacts may be historical or capped; check each source artifact before making leaderboard claims.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
