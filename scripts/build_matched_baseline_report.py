#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PAIRS = {
    "longmemeval": {
        "dense": "../BenchmarkResult/20260428_matched_baseline_full_v1/longmemeval_vector/longmemeval/merged_metrics.json",
        "full": "../BenchmarkResult/20260427_other_full_rerun_v1/longmemeval/longmemeval/merged_metrics.json",
    },
    "locomo": {
        "dense": "../BenchmarkResult/20260428_matched_baseline_full_v1/locomo_vector/locomo/merged_metrics.json",
        "full": "../BenchmarkResult/20260427_other_full_rerun_v1/locomo/locomo/merged_metrics.json",
    },
    "knowme": {
        "dense": "../BenchmarkResult/20260428_matched_baseline_full_v1/knowme_vector/knowme/merged_metrics.json",
        "full": "../BenchmarkResult/20260427_other_full_rerun_v1/knowme/knowme/merged_metrics.json",
    },
    "clonemem": {
        "dense": "../BenchmarkResult/20260428_matched_baseline_full_v1/clonemem_vector/clonemem/merged_metrics.json",
        "full": "../BenchmarkResult/20260427_clonemem_full_sample_sharded_v1/clonemem/clonemem/merged_metrics.json",
    },
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_metric(payload: Any, metric: str) -> Any:
    if not isinstance(payload, dict):
        return None
    if metric in payload:
        return payload[metric]
    lower = metric.lower()
    for key, value in payload.items():
        if str(key).lower() == lower:
            return value
    for preferred in ("session", "segment", "turn", "dialog"):
        value = payload.get(preferred)
        if isinstance(value, dict):
            found = find_metric(value, metric)
            if found is not None:
                return found
    for value in payload.values():
        if isinstance(value, dict):
            found = find_metric(value, metric)
            if found is not None:
                return found
    return None


def run_record(dataset: str, method: str, path: Path) -> dict[str, Any]:
    payload = load(path)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    embedding = payload.get("embedding_info") if isinstance(payload.get("embedding_info"), dict) else {}
    return {
        "dataset": dataset,
        "method": method,
        "artifact": str(path),
        "question_count": payload.get("total_question_count") or payload.get("question_count"),
        "fallback_in_use": embedding.get("fallback_in_use", payload.get("fallback_in_use")),
        "recall_any@5": find_metric(metrics, "recall_any@5"),
        "recall_any@10": find_metric(metrics, "recall_any@10"),
        "recall_frac@10": find_metric(metrics, "recall_frac@10"),
        "ndcg_any@10": find_metric(metrics, "ndcg_any@10"),
        "candidate_recall@100": find_metric(metrics, "candidate_recall@100"),
        "final_recall@10": find_metric(metrics, "final_recall@10"),
        "final_ndcg@10": find_metric(metrics, "final_ndcg@10"),
        "retrieval_ms": find_metric(payload.get("timings") or {}, "retrieval_ms"),
    }


def build_rows(pairs: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, methods in pairs.items():
        for method, raw_path in methods.items():
            path = Path(raw_path)
            if path.exists():
                rows.append(run_record(dataset, method, path))
            else:
                rows.append({"dataset": dataset, "method": method, "status": "missing", "artifact": str(path)})
    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Matched Baseline Full v1",
        "",
        "This table is generated from artifact `merged_metrics.json` files. Missing fields are left blank.",
        "",
        "| dataset | method | q | fallback | R@5 | R@10 | recall_frac@10 | NDCG@10 | candidate_recall@100 | final_R@10 | final_NDCG@10 | retrieval_ms | artifact |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset')} | {row.get('method')} | {row.get('question_count', '')} | {row.get('fallback_in_use', '')} | "
            f"{row.get('recall_any@5', '')} | {row.get('recall_any@10', '')} | {row.get('recall_frac@10', '')} | "
            f"{row.get('ndcg_any@10', '')} | {row.get('candidate_recall@100', '')} | {row.get('final_recall@10', '')} | "
            f"{row.get('final_ndcg@10', '')} | {row.get('retrieval_ms', '')} | {row.get('artifact')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build artifact-backed matched baseline report.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/baselines/matched_baseline_full_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/matched_baseline_full_v1_report.md"))
    args = parser.parse_args()
    rows = build_rows(DEFAULT_PAIRS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "dysonspherain.matched_baseline.v1", "rows": rows}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(rows, args.report)
    print(json.dumps({"rows": len(rows), "out": str(args.out), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
