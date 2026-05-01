from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_hash(value: Any) -> str | None:
    if value in (None, "", {}, []):
        return None
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_nonempty(payloads: list[dict[str, Any]], key: str) -> Any:
    for payload in payloads:
        value = payload.get(key)
        if value not in (None, "", {}, []):
            return value
    return None


def _question_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for row in list(payload.get("results") or payload.get("queries") or []):
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _collect_metric_values(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        buckets[f"{key}.{sub_key}"].append(float(sub_value))
            elif isinstance(value, (int, float)):
                buckets[str(key)].append(float(value))
        candidate = row.get("candidate_recall")
        if isinstance(candidate, dict):
            for key, value in candidate.items():
                if isinstance(value, (int, float)) and (
                    str(key).startswith("candidate_recall@")
                    or str(key).startswith("candidate_ndcg@")
                    or str(key).startswith("final_recall@")
                    or str(key).startswith("final_ndcg@")
                ):
                    buckets[str(key)].append(float(value))
    return buckets


def _weighted_payload_metric(payloads: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    numerator = 0.0
    denominator = 0
    for payload in payloads:
        node: Any = payload
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if not isinstance(node, (int, float)):
            continue
        count = int(payload.get("question_count") or len(payload.get("results") or []) or 0)
        numerator += float(node) * count
        denominator += count
    if denominator <= 0:
        return None
    return numerator / denominator


def _nested_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = _collect_metric_values(rows)
    nested: dict[str, Any] = {}
    for key, values in buckets.items():
        if "." in key:
            prefix, sub_key = key.split(".", 1)
            nested.setdefault(prefix, {})[sub_key] = _mean(values)
        else:
            nested[key] = _mean(values)
    return nested


def _fallback_metrics(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    candidate_paths = [
        ("metrics", "segment", "recall_frac@5"),
        ("metrics", "segment", "recall_frac@10"),
        ("metrics", "segment", "ndcg_any@10"),
        ("metrics", "session", "recall_frac@5"),
        ("metrics", "session", "recall_frac@10"),
        ("metrics", "session", "ndcg_any@10"),
        ("metrics", "dialog", "recall_frac@5"),
        ("metrics", "dialog", "recall_frac@10"),
        ("metrics", "dialog", "ndcg_any@10"),
        ("candidate_recall_report", "candidate_recall@100"),
        ("candidate_recall_report", "candidate_ndcg@10"),
        ("candidate_recall_report", "final_recall@10"),
        ("candidate_recall_report", "final_ndcg@10"),
    ]
    for path in candidate_paths:
        value = _weighted_payload_metric(payloads, path)
        if value is None:
            continue
        node = merged
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return merged


def _failure_taxonomy(rows: list[dict[str, Any]], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    for row in rows:
        candidate = row.get("candidate_recall")
        failure = None
        if isinstance(candidate, dict):
            failure = candidate.get("failure_type")
        failure = failure or row.get("failure_type")
        if failure:
            counter[str(failure)] += 1
    if not counter:
        for payload in payloads:
            for key, value in dict(payload.get("failure_summary") or {}).items():
                try:
                    counter[str(key)] += int(value)
                except (TypeError, ValueError):
                    continue
    total = sum(counter.values())
    return {
        "counts": dict(counter),
        "ratios": {key: value / total for key, value in counter.items()} if total else {},
    }


def _candidate_recall_summary(rows: list[dict[str, Any]], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = _collect_metric_values(rows)
    summary = {
        key: _mean(values)
        for key, values in buckets.items()
        if key.startswith("candidate_recall@")
        or key.startswith("candidate_ndcg@")
        or key.startswith("final_recall@")
        or key.startswith("final_ndcg@")
    }
    if summary:
        return summary
    fallback = _fallback_metrics(payloads)
    return dict(fallback.get("candidate_recall_report") or {})


def _oracle_summary(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        report = payload.get("oracle_retrieval_report") or {}
        rows.extend(report.get("rows") or payload.get("oracle_rows") or [])
    if rows:
        denom = max(1, len(rows))
        return {
            "oracle_query_count": len(rows),
            "oracle_recall@1": sum(1 for row in rows if row.get("top1_hit")) / denom,
            "oracle_recall@5": sum(1 for row in rows if row.get("top5_hit")) / denom,
            "oracle_recall@10": sum(1 for row in rows if row.get("top10_hit")) / denom,
        }
    numerators: dict[str, float] = defaultdict(float)
    denominator = 0
    for payload in payloads:
        report = dict(payload.get("oracle_retrieval_report") or {})
        count = int(report.get("oracle_query_count") or 0)
        denominator += count
        for key, value in report.items():
            if key.startswith("oracle_recall@") and isinstance(value, (int, float)):
                numerators[key] += float(value) * count
    if denominator <= 0:
        return {}
    return {"oracle_query_count": denominator, **{key: value / denominator for key, value in numerators.items()}}


def _timings(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for payload in payloads:
        count = max(1, int(payload.get("question_count") or len(payload.get("results") or []) or 1))
        for key, value in dict(payload.get("stage_timing_ms") or {}).items():
            if isinstance(value, (int, float)):
                buckets[key].extend([float(value)] * count)
    return {key: round(_mean(values), 4) for key, values in buckets.items()}


def _quality_status(metrics: dict[str, Any], candidate: dict[str, Any], failed_shards: list[str]) -> str:
    if failed_shards:
        return "failed_incomplete_shards"
    if candidate and _as_float(candidate.get("candidate_recall@100"), 1.0) < 0.95:
        return "warning_candidate_recall_below_0_95"
    return "passed"


def merge_payloads(paths: list[Path], run_manifest: Path | None = None) -> dict[str, Any]:
    payloads = [_load(path) for path in paths]
    rows = _question_rows(payloads)
    benchmark = str(payloads[0].get("benchmark_name") or payloads[0].get("benchmark") or "").lower()
    if not benchmark:
        benchmark = str(payloads[0].get("zone") or "benchmark").lower()
    metrics = _nested_metrics_from_rows(rows) if rows else _fallback_metrics(payloads)
    candidate = _candidate_recall_summary(rows, payloads)
    failures = _failure_taxonomy(rows, payloads)
    manifest_payload = _load(run_manifest) if run_manifest and run_manifest.exists() else {}
    chunks = list(manifest_payload.get("chunks") or [])
    successful = [str(chunk.get("chunk")) for chunk in chunks if str(chunk.get("status")) in {"completed", "skipped_existing"}]
    failed = [str(chunk.get("chunk")) for chunk in chunks if str(chunk.get("status")) not in {"completed", "skipped_existing"}]
    serial_elapsed = sum(_as_float(payload.get("elapsed_seconds")) for payload in payloads)
    wall = _as_float(manifest_payload.get("wall_clock_elapsed_seconds"), serial_elapsed)
    mode = next((str(payload.get("mode")) for payload in payloads if payload.get("mode")), None)
    ablation = next((str(payload.get("ablation")) for payload in payloads if payload.get("ablation")), None)
    run_type = next((str(payload.get("run_type")) for payload in payloads if payload.get("run_type")), None)
    runtime_config = _first_nonempty(payloads, "runtime_config") or {}
    route_policy_config = _first_nonempty(payloads, "route_policy_config") or {}
    dataset_version = _first_nonempty(payloads, "dataset_version")
    merged = {
        "benchmark": benchmark,
        "mode": mode,
        "ablation": ablation,
        "run_type": run_type or ablation,
        "shard_count": len(payloads),
        "successful_shards": len(successful) if chunks else len(payloads),
        "failed_shards": failed,
        "total_question_count": sum(int(payload.get("question_count") or len(payload.get("results") or []) or 0) for payload in payloads),
        "metrics": metrics,
        "candidate_recall_summary": candidate,
        "oracle_summary": _oracle_summary(payloads),
        "failure_taxonomy": failures,
        "timings": _timings(payloads),
        "serial_elapsed_seconds_sum": serial_elapsed,
        "wall_clock_elapsed_seconds": wall,
        "speedup_estimate": serial_elapsed / wall if wall > 0 else 0.0,
        "vector_info": payloads[0].get("vector_info") or {},
        "runtime_config": runtime_config,
        "route_policy_config": route_policy_config,
        "config_hash": _first_nonempty(payloads, "config_hash") or _stable_hash(runtime_config),
        "route_policy_hash": _first_nonempty(payloads, "route_policy_hash") or _stable_hash(route_policy_config),
        "dataset_version": dataset_version,
        "embedding_info": {
            "provider": (payloads[0].get("vector_info") or {}).get("embedding_provider"),
            "model": (payloads[0].get("vector_info") or {}).get("embedding_model"),
            "fallback_in_use": (payloads[0].get("vector_info") or {}).get("fallback_in_use"),
        },
        "quality_guardrail_status": _quality_status(metrics, candidate, failed),
        "source_files": [str(path) for path in paths],
    }
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge chunked benchmark shard outputs.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_payloads(args.inputs, args.run_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: merged[k] for k in ("benchmark", "total_question_count", "speedup_estimate", "quality_guardrail_status")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
