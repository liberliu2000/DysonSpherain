#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any


CHANNELS = (
    "dense_semantic",
    "lexical_sparse",
    "entity_aware",
    "exact_phrase",
    "temporal_anchor",
    "temporal_neighbor",
    "parent_session",
    "query_decomposition",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _best_gold_rank(rows: list[dict[str, Any]], gold_ids: set[str]) -> int | None:
    for row in rows:
        source_id = str(row.get("source_id") or row.get("source_segment_id") or "")
        if source_id in gold_ids:
            rank = _safe_int(row.get("rank"))
            return rank if rank is not None else rows.index(row) + 1
    return None


def _top_gold_rows(rows: list[dict[str, Any]], gold_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("source_id") or row.get("source_segment_id") or "") in gold_ids]


def _read_topk_debug(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    by_query: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_id = str(row.get("query_id") or "")
            if query_id:
                by_query[query_id] = row
    return by_query


def _bucket_query(query: dict[str, Any], debug: dict[str, Any] | None) -> list[str]:
    buckets: list[str] = []
    channel_stats = dict(query.get("channel_stats") or {})
    channels_hit = {
        channel
        for channel, payload in channel_stats.items()
        if bool(dict(payload or {}).get("gold_hit"))
    }
    dense_rank = _safe_int(query.get("dense_gold_rank") or channel_stats.get("dense_semantic", {}).get("gold_rank"))
    fused_rank = _safe_int(query.get("fused_gold_rank") or query.get("gold_rank_before_rerank"))
    rerank_rank = _safe_int(query.get("gold_rank_after_rerank"))
    final_rank = _safe_int(query.get("gold_rank_after_inhibition"))
    candidate_recall_100 = _safe_float(query.get("candidate_recall@100"))
    final_recall_10 = _safe_float(query.get("final_recall@10") or query.get("recall_frac@10"))

    if dense_rank is not None and dense_rank <= 200 and (fused_rank is None or fused_rank > 100):
        buckets.append("dense_hit_not_broad100")
    if channels_hit and (fused_rank is None or fused_rank > 100):
        buckets.append("channel_hit_not_broad100")
    if bool(query.get("parent_hit_segment_miss")):
        buckets.append("parent_hit_segment_miss")
    if fused_rank is not None and fused_rank <= 10 and (rerank_rank is None or rerank_rank > 10):
        buckets.append("reranker_dropped_gold")
    if rerank_rank is not None and rerank_rank <= 10 and (final_rank is None or final_rank > 10):
        buckets.append("inhibition_suppressed_gold")
    if candidate_recall_100 <= 0.0:
        buckets.append("no_gold_in_candidate100")
    elif final_recall_10 <= 0.0:
        buckets.append("candidate100_hit_final10_miss")
    if debug:
        gold_ids = {str(item) for item in debug.get("gold_segment_ids") or [] if str(item)}
        before_rows = list(debug.get("topk_before_rerank") or [])
        after_rows = list(debug.get("topk_after_rerank") or [])
        final_rows = list(debug.get("topk_after_inhibition") or [])
        before_rank = _best_gold_rank(before_rows, gold_ids)
        after_rank = _best_gold_rank(after_rows, gold_ids)
        final_debug_rank = _best_gold_rank(final_rows, gold_ids)
        if before_rank is not None and before_rank <= 20 and (after_rank is None or after_rank > 20):
            buckets.append("top20_rerank_drop")
        if final_debug_rank is None and before_rank is not None:
            buckets.append("debug_final_dropped_seen_gold")
    return list(dict.fromkeys(buckets)) or ["ok_or_unbucketed"]


def analyze_many(
    candidate_paths: list[Path],
    *,
    topk_debug_paths: list[Path] | None = None,
    max_examples: int = 8,
) -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    debug_by_query: dict[str, dict[str, Any]] = {}
    for candidate_path in candidate_paths:
        payload = _load_json(candidate_path)
        queries.extend(item for item in list(payload.get("queries") or []) if isinstance(item, dict))
    for topk_debug_path in topk_debug_paths or []:
        debug_by_query.update(_read_topk_debug(topk_debug_path))
    bucket_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    channel_hit_counts: Counter[str] = Counter()
    channel_hit_not_broad_counts: Counter[str] = Counter()
    parent_anchor_counts: list[int] = []
    parent_gold_selected = 0
    parent_trace_count = 0
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    numeric: dict[str, list[float]] = defaultdict(list)

    for query in queries:
        query_id = str(query.get("query_id") or "")
        debug = debug_by_query.get(query_id)
        buckets = _bucket_query(query, debug)
        bucket_counts.update(buckets)
        failure_counts[str(query.get("failure_type") or "unknown")] += 1
        channel_stats = dict(query.get("channel_stats") or {})
        fused_rank = _safe_int(query.get("fused_gold_rank") or query.get("gold_rank_before_rerank"))
        for channel in CHANNELS:
            channel_payload = dict(channel_stats.get(channel) or {})
            if bool(channel_payload.get("gold_hit")):
                channel_hit_counts[channel] += 1
                if fused_rank is None or fused_rank > 100:
                    channel_hit_not_broad_counts[channel] += 1
        parent_audit = dict(query.get("parent_audit") or {})
        selected = list(parent_audit.get("selected_child_anchors") or [])
        parent_anchor_counts.append(len(selected))
        parent_trace_count += len(selected)
        gold_ids = {str(item) for item in query.get("gold_segment_ids") or [] if str(item)}
        parent_gold_selected += sum(1 for row in selected if str(row.get("source_id") or "") in gold_ids)
        for metric in ("candidate_recall@100", "candidate_ndcg@10", "final_recall@10", "final_ndcg@10"):
            numeric[metric].append(_safe_float(query.get(metric)))
        for bucket in buckets:
            if len(examples[bucket]) >= max_examples:
                continue
            example = {
                "query_id": query_id,
                "failure_type": str(query.get("failure_type") or ""),
                "candidate_recall@100": _safe_float(query.get("candidate_recall@100")),
                "final_recall@10": _safe_float(query.get("final_recall@10") or query.get("recall_frac@10")),
                "dense_gold_rank": query.get("dense_gold_rank"),
                "fused_gold_rank": query.get("fused_gold_rank") or query.get("gold_rank_before_rerank"),
                "gold_rank_after_rerank": query.get("gold_rank_after_rerank"),
                "gold_rank_after_inhibition": query.get("gold_rank_after_inhibition"),
                "channels_that_hit_gold": list(query.get("channels_that_hit_gold") or []),
                "parent_anchor_selected_count": len(selected),
                "query_text": str(query.get("query_text") or "")[:300],
            }
            if debug:
                gold_debug_rows = _top_gold_rows(list(debug.get("topk_before_rerank") or []), gold_ids)
                example["gold_rows_before_rerank"] = [
                    {
                        "rank": row.get("rank"),
                        "source_id": row.get("source_id"),
                        "support_count": row.get("support_count"),
                        "dense_score": row.get("dense_score"),
                        "bm25_score": row.get("bm25_score"),
                        "exact_phrase_score": row.get("exact_phrase_score"),
                        "parent_score": row.get("parent_score"),
                        "decomposition_score": row.get("decomposition_score"),
                        "local_window_score": row.get("local_window_score"),
                        "rerank_score": row.get("rerank_score"),
                    }
                    for row in gold_debug_rows[:5]
                ]
            examples[bucket].append(example)

    q_count = len(queries)
    summary = {
        "candidate_path": str(candidate_paths[0]) if len(candidate_paths) == 1 else "",
        "candidate_paths": [str(path) for path in candidate_paths],
        "topk_debug_path": str(topk_debug_paths[0]) if topk_debug_paths and len(topk_debug_paths) == 1 else "",
        "topk_debug_paths": [str(path) for path in topk_debug_paths or []],
        "question_count": q_count,
        "metrics": {
            key: round(mean(values), 6) if values else 0.0
            for key, values in sorted(numeric.items())
        },
        "bucket_counts": dict(bucket_counts),
        "bucket_rates": {
            key: round(value / max(1, q_count), 6)
            for key, value in sorted(bucket_counts.items())
        },
        "failure_type_counts": dict(failure_counts),
        "channel_gold_hit_counts": dict(channel_hit_counts),
        "channel_hit_not_broad100_counts": dict(channel_hit_not_broad_counts),
        "channel_hit_not_broad100_rates_given_hit": {
            channel: round(channel_hit_not_broad_counts[channel] / max(1, channel_hit_counts[channel]), 6)
            for channel in CHANNELS
            if channel_hit_counts[channel]
        },
        "parent_to_segment": {
            "avg_selected_child_anchors": round(mean(parent_anchor_counts), 4) if parent_anchor_counts else 0.0,
            "selected_child_anchor_trace_count": parent_trace_count,
            "gold_selected_child_anchor_count": parent_gold_selected,
            "gold_selected_child_anchor_rate_per_query": round(parent_gold_selected / max(1, q_count), 6),
        },
        "examples": dict(examples),
    }
    return summary


def analyze(candidate_path: Path, *, topk_debug_path: Path | None = None, max_examples: int = 8) -> dict[str, Any]:
    return analyze_many(
        [candidate_path],
        topk_debug_paths=[topk_debug_path] if topk_debug_path else None,
        max_examples=max_examples,
    )


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# CloneMem Admission Failure Analysis",
        "",
        f"Candidate diagnostics: `{summary.get('candidate_path') or str(len(summary.get('candidate_paths') or [])) + ' files'}`",
        f"Top-k debug: `{summary.get('topk_debug_path') or (str(len(summary.get('topk_debug_paths') or [])) + ' files' if summary.get('topk_debug_paths') else 'not provided')}`",
        f"Questions: {summary['question_count']}",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary.get("metrics", {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Failure Buckets", "", "| bucket | count | rate |", "|---|---:|---:|"])
    rates = dict(summary.get("bucket_rates") or {})
    for key, value in sorted(dict(summary.get("bucket_counts") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value} | {rates.get(key, 0.0)} |")
    lines.extend(["", "## Channel Hit But Not Broad@100", "", "| channel | hits | misses after hit | rate |", "|---|---:|---:|---:|"])
    hits = dict(summary.get("channel_gold_hit_counts") or {})
    misses = dict(summary.get("channel_hit_not_broad100_counts") or {})
    miss_rates = dict(summary.get("channel_hit_not_broad100_rates_given_hit") or {})
    for channel in CHANNELS:
        if channel not in hits:
            continue
        lines.append(f"| {channel} | {hits.get(channel, 0)} | {misses.get(channel, 0)} | {miss_rates.get(channel, 0.0)} |")
    parent = dict(summary.get("parent_to_segment") or {})
    lines.extend([
        "",
        "## Parent-To-Segment",
        "",
        f"- avg selected child anchors: {parent.get('avg_selected_child_anchors', 0.0)}",
        f"- selected child anchor traces: {parent.get('selected_child_anchor_trace_count', 0)}",
        f"- gold selected child anchors: {parent.get('gold_selected_child_anchor_count', 0)}",
        f"- gold selected child anchor rate/query: {parent.get('gold_selected_child_anchor_rate_per_query', 0.0)}",
        "",
        "## Example Query IDs",
        "",
    ])
    for bucket, rows in sorted(dict(summary.get("examples") or {}).items()):
        lines.append(f"### {bucket}")
        for row in rows[:5]:
            lines.append(
                f"- `{row.get('query_id')}` failure={row.get('failure_type')} "
                f"cand100={row.get('candidate_recall@100')} final10={row.get('final_recall@10')} "
                f"dense={row.get('dense_gold_rank')} fused={row.get('fused_gold_rank')} "
                f"rerank={row.get('gold_rank_after_rerank')}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CloneMem candidate admission failure buckets.")
    parser.add_argument("--candidate", required=True, action="append", type=Path)
    parser.add_argument("--topk-debug", action="append", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()
    summary = analyze_many(args.candidate, topk_debug_paths=args.topk_debug, max_examples=max(1, args.max_examples))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, args.report)
    print(json.dumps({"questions": summary["question_count"], "out": str(args.out), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
