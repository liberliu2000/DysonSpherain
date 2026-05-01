#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT.parent / "BenchmarkResult"
DEFAULT_DIAGNOSTICS_DIR = ROOT / "reports" / "diagnostics"
DEFAULT_REPORT_PATH = ROOT / "reports" / "phase4_diagnostic_consolidation.md"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_candidate_recall_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.name.endswith("_candidate_recall.json"):
            files.append(root)
        elif root.exists():
            files.extend(root.rglob("*_candidate_recall.json"))
    return sorted({path.resolve() for path in files})


def _rank_lost(before_rank: Any, after_rank: Any, limit: int) -> bool:
    if not isinstance(before_rank, int) or before_rank > limit:
        return False
    if after_rank is None:
        return True
    if isinstance(after_rank, int):
        return after_rank > limit
    return False


def _diagnostic_row(path: Path, payload: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": str(path),
        "benchmark_name": query.get("benchmark_name") or payload.get("benchmark_name"),
        "query_id": query.get("query_id") or query.get("question_id"),
        "query_text": query.get("query_text"),
        "gold_evidence_ids": query.get("gold_evidence_ids") or [],
        "gold_segment_ids": query.get("gold_segment_ids") or [],
        "dense_gold_rank": query.get("dense_gold_rank"),
        "fused_gold_rank": query.get("fused_gold_rank"),
        "gold_rank_before_rerank": query.get("gold_rank_before_rerank"),
        "gold_rank_after_rerank": query.get("gold_rank_after_rerank"),
        "gold_rank_after_inhibition": query.get("gold_rank_after_inhibition"),
        "candidate_recall@100": query.get("candidate_recall@100"),
        "final_recall@10": query.get("final_recall@10"),
        "dense_hit@100": bool(query.get("dense_hit@100")),
        "fused_hit@100": bool(query.get("fused_hit@100")),
        "failure_type": query.get("failure_type"),
        "channels_that_hit_gold": query.get("channels_that_hit_gold") or [],
        "channel_stats": query.get("channel_stats") or {},
    }


def consolidate_candidate_recall_files(paths: Iterable[Path]) -> dict[str, Any]:
    fusion_violations: list[dict[str, Any]] = []
    reranker_drops: list[dict[str, Any]] = []
    benchmark_failures: dict[str, dict[str, list[dict[str, Any]]]] = {
        "clonemem": {
            "parent_hit_segment_miss": [],
            "lexical_miss": [],
            "temporal_miss": [],
            "reranker_dropped_gold": [],
        },
        "locomo": {"ordering_failures": []},
        "knowme": {"segment_admission_failures": []},
    }
    scanned_files = 0
    scanned_queries = 0

    for path in sorted(paths):
        payload = _load_json(path)
        queries = payload.get("queries") or []
        if not isinstance(queries, list):
            continue
        scanned_files += 1
        for query in queries:
            if not isinstance(query, dict):
                continue
            scanned_queries += 1
            row = _diagnostic_row(path, payload, query)
            benchmark_name = str(row.get("benchmark_name") or "").lower()
            failure_type = str(query.get("failure_type") or "ok")
            if bool(query.get("dense_hit@100")) and not bool(query.get("fused_hit@100")):
                row["violation_type"] = "dense_hit_lost_before_fusion_top100"
                fusion_violations.append(row)
            if (
                query.get("failure_type") == "reranker_dropped_gold"
                or _rank_lost(query.get("gold_rank_before_rerank"), query.get("gold_rank_after_rerank"), 100)
            ):
                row["drop_type"] = "reranker_dropped_gold"
                reranker_drops.append(row)
            if benchmark_name == "clonemem" and failure_type in benchmark_failures["clonemem"]:
                benchmark_failures["clonemem"][failure_type].append(row)
            if benchmark_name == "locomo" and (
                failure_type in {"reranker_dropped_gold", "inhibition_suppressed_gold", "local_candidate_crowding"}
                or _rank_lost(query.get("gold_rank_before_rerank"), query.get("gold_rank_after_rerank"), 10)
            ):
                benchmark_failures["locomo"]["ordering_failures"].append(row)
            if benchmark_name == "knowme" and (
                failure_type
                in {
                    "query_gold_mapping_empty",
                    "parent_hit_segment_miss",
                    "profile_fact_missing",
                    "entity_miss",
                    "temporal_miss",
                    "gold_missing_from_candidate_pool",
                }
                or bool(query.get("parent_hit_segment_miss"))
            ):
                benchmark_failures["knowme"]["segment_admission_failures"].append(row)

    return {
        "scanned_files": scanned_files,
        "scanned_queries": scanned_queries,
        "fusion_dense_preservation_violations": fusion_violations,
        "reranker_dropped_gold_examples": reranker_drops,
        "benchmark_failures": benchmark_failures,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _counter_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))


def _channel_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for channel in row.get("channels_that_hit_gold") or []:
            counter[str(channel)] += 1
    return dict(counter.most_common())


def write_outputs(summary: dict[str, Any], diagnostics_dir: Path, report_path: Path) -> dict[str, str]:
    fusion_path = diagnostics_dir / "fusion_dense_preservation_violations.jsonl"
    reranker_path = diagnostics_dir / "reranker_dropped_gold_examples.jsonl"
    fusion_rows = list(summary.get("fusion_dense_preservation_violations") or [])
    reranker_rows = list(summary.get("reranker_dropped_gold_examples") or [])
    benchmark_failures = dict(summary.get("benchmark_failures") or {})
    _write_jsonl(fusion_path, fusion_rows)
    _write_jsonl(reranker_path, reranker_rows)
    clonemem_failures = dict(benchmark_failures.get("clonemem") or {})
    locomo_failures = dict(benchmark_failures.get("locomo") or {})
    knowme_failures = dict(benchmark_failures.get("knowme") or {})
    clonemem_parent_path = diagnostics_dir / "clonemem_parent_hit_segment_miss_examples.jsonl"
    clonemem_lexical_path = diagnostics_dir / "clonemem_lexical_miss_examples.jsonl"
    clonemem_temporal_path = diagnostics_dir / "clonemem_temporal_miss_examples.jsonl"
    clonemem_reranker_path = diagnostics_dir / "clonemem_reranker_dropped_gold_examples.jsonl"
    locomo_ordering_path = diagnostics_dir / "locomo_ordering_failures.jsonl"
    knowme_segment_path = diagnostics_dir / "knowme_segment_admission_failures.jsonl"
    _write_jsonl(clonemem_parent_path, list(clonemem_failures.get("parent_hit_segment_miss") or []))
    _write_jsonl(clonemem_lexical_path, list(clonemem_failures.get("lexical_miss") or []))
    _write_jsonl(clonemem_temporal_path, list(clonemem_failures.get("temporal_miss") or []))
    _write_jsonl(clonemem_reranker_path, list(clonemem_failures.get("reranker_dropped_gold") or []))
    _write_jsonl(locomo_ordering_path, list(locomo_failures.get("ordering_failures") or []))
    _write_jsonl(knowme_segment_path, list(knowme_failures.get("segment_admission_failures") or []))

    lines = [
        "# Phase 4 Diagnostic Consolidation",
        "",
        "This report is generated from existing `*_candidate_recall.json` artifacts.",
        "It does not change retrieval, fusion, reranking, or candidate admission behavior.",
        "",
        "## Summary",
        "",
        f"- scanned candidate recall files: {summary.get('scanned_files', 0)}",
        f"- scanned query rows: {summary.get('scanned_queries', 0)}",
        f"- dense preservation violations: {len(fusion_rows)}",
        f"- reranker dropped gold examples: {len(reranker_rows)}",
        "",
        "## Breakdown",
        "",
        "### Dense Preservation Violations",
        "",
        f"- by benchmark: `{json.dumps(_counter_by(fusion_rows, 'benchmark_name'), sort_keys=True)}`",
        f"- by failure type: `{json.dumps(_counter_by(fusion_rows, 'failure_type'), sort_keys=True)}`",
        f"- channels that hit gold: `{json.dumps(_channel_counter(fusion_rows), sort_keys=True)}`",
        "",
        "### Reranker Dropped Gold Examples",
        "",
        f"- by benchmark: `{json.dumps(_counter_by(reranker_rows, 'benchmark_name'), sort_keys=True)}`",
        f"- by failure type: `{json.dumps(_counter_by(reranker_rows, 'failure_type'), sort_keys=True)}`",
        f"- channels that hit gold: `{json.dumps(_channel_counter(reranker_rows), sort_keys=True)}`",
        "",
        "## Phase 5 Benchmark-Specific Diagnostics",
        "",
        "### CloneMem",
        "",
        f"- parent_hit_segment_miss: {len(clonemem_failures.get('parent_hit_segment_miss') or [])}",
        f"- lexical_miss: {len(clonemem_failures.get('lexical_miss') or [])}",
        f"- temporal_miss: {len(clonemem_failures.get('temporal_miss') or [])}",
        f"- reranker_dropped_gold: {len(clonemem_failures.get('reranker_dropped_gold') or [])}",
        "",
        "### LoCoMo",
        "",
        f"- ordering_failures: {len(locomo_failures.get('ordering_failures') or [])}",
        "",
        "### KnowMe",
        "",
        f"- segment_admission_failures: {len(knowme_failures.get('segment_admission_failures') or [])}",
        "",
        "## Outputs",
        "",
        f"- `{fusion_path}`",
        f"- `{reranker_path}`",
        f"- `{clonemem_parent_path}`",
        f"- `{clonemem_lexical_path}`",
        f"- `{clonemem_temporal_path}`",
        f"- `{clonemem_reranker_path}`",
        f"- `{locomo_ordering_path}`",
        f"- `{knowme_segment_path}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "fusion_dense_preservation_violations": str(fusion_path),
        "reranker_dropped_gold_examples": str(reranker_path),
        "clonemem_parent_hit_segment_miss_examples": str(clonemem_parent_path),
        "clonemem_lexical_miss_examples": str(clonemem_lexical_path),
        "clonemem_temporal_miss_examples": str(clonemem_temporal_path),
        "clonemem_reranker_dropped_gold_examples": str(clonemem_reranker_path),
        "locomo_ordering_failures": str(locomo_ordering_path),
        "knowme_segment_admission_failures": str(knowme_segment_path),
        "report": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate Phase 4 candidate-recall diagnostics.")
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help="BenchmarkResult root, run directory, or candidate_recall.json file. May be repeated.",
    )
    parser.add_argument("--diagnostics-dir", type=Path, default=DEFAULT_DIAGNOSTICS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = args.root or [DEFAULT_RESULT_ROOT]
    paths = find_candidate_recall_files(roots)
    summary = consolidate_candidate_recall_files(paths)
    outputs = write_outputs(summary, args.diagnostics_dir, args.report)
    print(json.dumps({"summary": {k: v for k, v in summary.items() if isinstance(v, int)}, "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
