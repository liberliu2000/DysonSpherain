from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

KS = [1, 3, 5, 10, 30, 50]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def audit_field(payloads: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [payload.get(field) for payload in payloads]
    normalized = {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values}
    return {
        "consistent": len(normalized) == 1,
        "values": values if len(values) <= 3 else values[:3],
    }


def build_summary(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_session = defaultdict(list)
    metrics_turn = defaultdict(list)
    per_type = defaultdict(lambda: defaultdict(list))

    for item in results:
        qtype = str(item["question_type"])
        session_metrics = item["metrics"]["session"]
        turn_metrics = item["metrics"]["turn"]
        for k in KS:
            session_ra = float(session_metrics.get(f"recall_any@{k}", 0.0))
            session_nd = float(session_metrics.get(f"ndcg_any@{k}", 0.0))
            turn_ra = float(turn_metrics.get(f"recall_any@{k}", 0.0))
            turn_nd = float(turn_metrics.get(f"ndcg_any@{k}", 0.0))

            metrics_session[f"recall_any@{k}"].append(session_ra)
            metrics_session[f"ndcg_any@{k}"].append(session_nd)
            metrics_turn[f"recall_any@{k}"].append(turn_ra)
            metrics_turn[f"ndcg_any@{k}"].append(turn_nd)

            if f"recall_all@{k}" in session_metrics:
                metrics_session[f"recall_all@{k}"].append(float(session_metrics[f"recall_all@{k}"]))
            if f"recall_all@{k}" in turn_metrics:
                metrics_turn[f"recall_all@{k}"].append(float(turn_metrics[f"recall_all@{k}"]))

        per_type[qtype]["recall_any@5"].append(float(session_metrics.get("recall_any@5", 0.0)))
        per_type[qtype]["recall_any@10"].append(float(session_metrics.get("recall_any@10", 0.0)))
        per_type[qtype]["ndcg_any@10"].append(float(session_metrics.get("ndcg_any@10", 0.0)))

    metrics = {
        "session": {key: mean(values) for key, values in metrics_session.items()},
        "turn": {key: mean(values) for key, values in metrics_turn.items()},
    }
    per_type_summary = {
        qtype: {key: mean(values) for key, values in metric_map.items()}
        for qtype, metric_map in per_type.items()
    }
    return metrics, per_type_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge chunked LongMemEval benchmark outputs.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON files to merge")
    parser.add_argument("--out", type=Path, required=True, help="Merged JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = []
    for path in args.inputs:
        payloads.append(json.loads(path.read_text(encoding="utf-8")))

    payloads.sort(key=lambda item: str(item.get("data_file", "")))
    first = payloads[0]

    results: list[dict[str, Any]] = []
    for payload in payloads:
        results.extend(payload.get("results", []))

    metrics, per_type = build_summary(results)
    stage_timing_bucket: dict[str, list[float]] = defaultdict(list)
    for payload in payloads:
        for key, value in dict(payload.get("stage_timing_ms") or {}).items():
            try:
                stage_timing_bucket[key].append(float(value))
            except (TypeError, ValueError):
                continue
    stage_timing_summary = {
        key: mean(values)
        for key, values in stage_timing_bucket.items()
        if values
    }
    config_audit_fields = [
        "mode",
        "granularity",
        "top_k",
        "task_type",
        "shell",
        "sector",
        "zone",
        "chunk_pool",
        "object_top_k",
        "support_top_k",
        "cognitive_top_k",
        "requested_rerank_mode",
        "rerank_mode",
        "rerank_mode_active",
        "cross_encoder_requested",
        "cross_encoder_loaded",
        "runtime_config",
    ]

    merged = {
        "data_file": "merged:" + ",".join(str(path) for path in args.inputs),
        "mode": first.get("mode"),
        "rerank_mode": first.get("rerank_mode"),
        "requested_rerank_mode": first.get("requested_rerank_mode"),
        "rerank_mode_active": first.get("rerank_mode_active"),
        "granularity": first.get("granularity"),
        "question_count": len(results),
        "top_k": first.get("top_k"),
        "task_type": first.get("task_type"),
        "shell": first.get("shell"),
        "sector": first.get("sector"),
        "zone": first.get("zone"),
        "chunk_pool": first.get("chunk_pool"),
        "object_top_k": first.get("object_top_k"),
        "support_top_k": first.get("support_top_k"),
        "cognitive_top_k": first.get("cognitive_top_k"),
        "cross_encoder_requested": first.get("cross_encoder_requested"),
        "cross_encoder_loaded": first.get("cross_encoder_loaded"),
        "elapsed_seconds": sum(float(payload.get("elapsed_seconds", 0.0)) for payload in payloads),
        "vector_info": first.get("vector_info"),
        "runtime_config": first.get("runtime_config", {}),
        "metrics": metrics,
        "per_type": per_type,
        "stage_timing_ms": stage_timing_summary,
        "config_audit": {field: audit_field(payloads, field) for field in config_audit_fields},
        "results": results,
        "source_files": [str(path) for path in args.inputs],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Merged Summary")
    print(f"  Files:       {len(args.inputs)}")
    print(f"  Questions:   {merged['question_count']}")
    print(f"  Mode:        {merged['mode']}")
    print(f"  Granularity: {merged['granularity']}")
    print(f"  Recall@5:    {metrics['session'].get('recall_any@5', 0.0):.4f}")
    print(f"  Recall@10:   {metrics['session'].get('recall_any@10', 0.0):.4f}")
    print(f"  NDCG@10:     {metrics['session'].get('ndcg_any@10', 0.0):.4f}")
    print(f"  Saved:       {args.out}")


if __name__ == "__main__":
    main()
