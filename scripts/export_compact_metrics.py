#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


JQ_FILTER = r"""
{
  schema: "dysonspherain.compact_metrics.v1",
  benchmark: (.benchmark // .benchmark_name // null),
  mode: (.mode // null),
  baseline: (.baseline // .method // null),
  question_count: (.question_count // .total_question_count // null),
  sample_count: (.sample_count // .total_sample_count // null),
  elapsed_seconds: (.elapsed_seconds // .wall_clock_elapsed_seconds // null),
  top_k: (.top_k // null),
  fallback_in_use: (.fallback_in_use // .embedding_info.fallback_in_use // .vector_info.fallback_in_use // null),
  embedding_provider: (.embedding_provider // .embedding_info.embedding_provider // .vector_info.embedding_provider // null),
  embedding_model: (.embedding_model // .embedding_info.embedding_model // .vector_info.embedding_model // null),
  metrics: (.metrics // {}),
  candidate_recall_report: (.candidate_recall_report // {}),
  failure_summary: (.failure_summary // {}),
  results: [
    (.results // .question_results // .question_rows // [])[] |
    {
      question_id,
      sample_id,
      question_type,
      task_type,
      metrics,
      candidate_recall: (
        .candidate_recall // {} |
        {
          "candidate_recall@10": .["candidate_recall@10"],
          "candidate_recall@50": .["candidate_recall@50"],
          "candidate_recall@100": .["candidate_recall@100"],
          "candidate_recall@200": .["candidate_recall@200"],
          "candidate_ndcg@10": .["candidate_ndcg@10"],
          "final_recall@10": .["final_recall@10"],
          "final_ndcg@10": .["final_ndcg@10"],
          "dense_hit@100": .["dense_hit@100"],
          "fused_hit@100": .["fused_hit@100"],
          "failure_type": .failure_type,
          "failure_bucket": .failure_bucket,
          "best_gold_rank": .best_gold_rank,
          "dense_gold_rank": .dense_gold_rank
        }
      ),
      stage_timing_ms,
      failure_type,
      failure_bucket,
      gold_ids,
      answer_session_ids,
      answer_segment_ids
    }
  ]
}
"""


def compact_with_jq(path: Path) -> dict[str, Any]:
    jq = shutil.which("jq")
    if not jq:
        raise RuntimeError("jq is required for streaming compact export of large metrics files")
    with tempfile.NamedTemporaryFile("w+b", suffix=".json") as tmp:
        subprocess.run([jq, JQ_FILTER, str(path)], stdout=tmp, stderr=subprocess.PIPE, check=True)
        tmp.flush()
        tmp.seek(0)
        payload = json.loads(tmp.read().decode("utf-8"))
    payload["source_metrics_path"] = str(path)
    return payload


def merge_payloads(payloads: list[dict[str, Any]], *, source_label: str | None = None) -> dict[str, Any]:
    if not payloads:
        raise ValueError("no payloads")
    first = payloads[0]
    results: list[dict[str, Any]] = []
    elapsed_sum = 0.0
    question_count_sum = 0
    for payload in payloads:
        rows = payload.get("results") if isinstance(payload.get("results"), list) else []
        results.extend(rows)
        elapsed = payload.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            elapsed_sum += float(elapsed)
        q_count = payload.get("question_count")
        if isinstance(q_count, int):
            question_count_sum += q_count
    merged = {
        "schema": "dysonspherain.compact_metrics.v1",
        "source_label": source_label,
        "source_metrics_paths": [payload.get("source_metrics_path") for payload in payloads],
        "benchmark": first.get("benchmark"),
        "mode": first.get("mode"),
        "baseline": first.get("baseline"),
        "question_count": question_count_sum or len(results) or first.get("question_count"),
        "sample_count": first.get("sample_count"),
        "elapsed_seconds": elapsed_sum or first.get("elapsed_seconds"),
        "fallback_in_use": any(bool(payload.get("fallback_in_use")) for payload in payloads),
        "embedding_provider": first.get("embedding_provider"),
        "embedding_model": first.get("embedding_model"),
        "metrics": first.get("metrics") or {},
        "candidate_recall_report": first.get("candidate_recall_report") or {},
        "failure_summary": first.get("failure_summary") or {},
        "results": results,
        "compact_result_count": len(results),
    }
    return merged


def write_report(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Compact Metrics Export",
        "",
        f"- benchmark: `{payload.get('benchmark')}`",
        f"- source_label: `{payload.get('source_label')}`",
        f"- source_files: `{len(payload.get('source_metrics_paths') or [])}`",
        f"- compact_result_count: `{payload.get('compact_result_count')}`",
        f"- question_count: `{payload.get('question_count')}`",
        f"- fallback_in_use: `{payload.get('fallback_in_use')}`",
        f"- embedding_provider: `{payload.get('embedding_provider')}`",
        f"- embedding_model: `{payload.get('embedding_model')}`",
        "",
        "The export excludes large ranking payloads and keeps only per-question metrics, candidate recall, timing, and gold-id fields needed for statistical analysis.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact per-question metrics from large benchmark metrics.json files.")
    parser.add_argument("metrics", nargs="+", type=Path, help="Input metrics.json files")
    parser.add_argument("--out", type=Path, required=True, help="Output compact JSON")
    parser.add_argument("--report", type=Path, help="Optional markdown report")
    parser.add_argument("--source-label", help="Source label for merged chunks")
    args = parser.parse_args()

    payloads = [compact_with_jq(path) for path in args.metrics]
    merged = merge_payloads(payloads, source_label=args.source_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.report:
        write_report(merged, args.report)
    print(json.dumps({"out": str(args.out), "results": merged["compact_result_count"], "fallback_in_use": merged["fallback_in_use"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
