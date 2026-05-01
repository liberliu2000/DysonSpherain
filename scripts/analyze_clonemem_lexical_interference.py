#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = ("candidate_recall@100", "final_recall@10", "final_ndcg@10", "recall_any@10", "ndcg_any@10")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


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


def _query_key(row: dict[str, Any]) -> str:
    for key in ("query_id", "question_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    sample_id = str(row.get("sample_id") or row.get("source_doc_id") or "").strip()
    text = str(row.get("query_text") or "").strip()
    return f"{sample_id}::{text[:200]}"


def _load_candidate_queries(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_query: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = _load_json(path)
        for row in list(payload.get("queries") or []):
            if isinstance(row, dict):
                by_query[_query_key(row)] = row
    return by_query


def _load_topk_debug(paths: list[Path] | None) -> dict[str, dict[str, Any]]:
    by_query: dict[str, dict[str, Any]] = {}
    for path in paths or []:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    by_query[_query_key(row)] = row
    return by_query


def _rank_of_gold(rows: list[dict[str, Any]], gold_ids: set[str]) -> int | None:
    for index, row in enumerate(rows, start=1):
        source_id = str(row.get("source_id") or row.get("source_segment_id") or "")
        if source_id in gold_ids:
            return _safe_int(row.get("rank")) or index
    return None


def _source_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("source_id") or row.get("source_segment_id") or "") for row in rows if str(row.get("source_id") or row.get("source_segment_id") or "")}


def _parent_anchor_ids(row: dict[str, Any]) -> set[str]:
    parent_audit = dict(row.get("parent_audit") or {})
    anchors = list(parent_audit.get("selected_child_anchors") or [])
    return {str(item.get("source_id") or "") for item in anchors if isinstance(item, dict) and str(item.get("source_id") or "")}


def _parent_anchor_terms(row: dict[str, Any]) -> set[str]:
    parent_audit = dict(row.get("parent_audit") or {})
    terms: set[str] = set()
    for item in list(parent_audit.get("selected_child_anchors") or []):
        if not isinstance(item, dict):
            continue
        terms.update(str(term or "").strip().lower() for term in list(item.get("matched_anchor_terms") or []) if str(term or "").strip())
    return terms


def _channel_count(row: dict[str, Any], channel: str) -> int:
    stats = dict(row.get("channel_stats") or {})
    payload = dict(stats.get(channel) or {})
    return int(_safe_float(payload.get("candidate_count")))


def _channel_gold_hit(row: dict[str, Any], channel: str) -> bool:
    stats = dict(row.get("channel_stats") or {})
    payload = dict(stats.get(channel) or {})
    return bool(payload.get("gold_hit"))


def _classify_delta(default_row: dict[str, Any], variant_row: dict[str, Any], default_debug: dict[str, Any] | None, variant_debug: dict[str, Any] | None) -> list[str]:
    labels: list[str] = []
    final_delta = _safe_float(variant_row.get("final_recall@10") or variant_row.get("recall_frac@10")) - _safe_float(
        default_row.get("final_recall@10") or default_row.get("recall_frac@10")
    )
    ndcg_delta = _safe_float(variant_row.get("final_ndcg@10") or variant_row.get("ndcg_any@10")) - _safe_float(
        default_row.get("final_ndcg@10") or default_row.get("ndcg_any@10")
    )
    candidate_delta = _safe_float(variant_row.get("candidate_recall@100")) - _safe_float(default_row.get("candidate_recall@100"))
    if final_delta > 1e-9 or ndcg_delta > 1e-9:
        labels.append("variant_improved")
    if final_delta < -1e-9 or ndcg_delta < -1e-9:
        labels.append("variant_regressed")
    if abs(candidate_delta) <= 1e-9 and (final_delta or ndcg_delta):
        labels.append("ranking_changed_without_candidate_recall_change")
    if _channel_count(default_row, "lexical_sparse") > 0 and _channel_count(variant_row, "lexical_sparse") == 0:
        labels.append("lexical_removed")
    if _parent_anchor_ids(default_row) != _parent_anchor_ids(variant_row):
        labels.append("parent_anchor_changed")
    if _parent_anchor_terms(default_row) != _parent_anchor_terms(variant_row):
        labels.append("parent_anchor_terms_changed")
    if bool(default_row.get("parent_hit_segment_miss")) != bool(variant_row.get("parent_hit_segment_miss")):
        labels.append("parent_hit_segment_miss_changed")
    if str(default_row.get("failure_type") or "") != str(variant_row.get("failure_type") or ""):
        labels.append("failure_type_changed")
    gold_ids = {str(item) for item in default_row.get("gold_segment_ids") or variant_row.get("gold_segment_ids") or [] if str(item)}
    if default_debug and variant_debug:
        default_before = list(default_debug.get("topk_before_rerank") or [])
        variant_before = list(variant_debug.get("topk_before_rerank") or [])
        default_rank = _rank_of_gold(default_before, gold_ids)
        variant_rank = _rank_of_gold(variant_before, gold_ids)
        if default_rank != variant_rank:
            labels.append("pre_rerank_gold_rank_changed")
        default_top = _source_ids(default_before[:20])
        variant_top = _source_ids(variant_before[:20])
        if default_top - variant_top:
            labels.append("default_only_top20_candidates")
        if variant_top - default_top:
            labels.append("variant_only_top20_candidates")
    return labels or ["unchanged_or_unclassified"]


def analyze(
    default_candidate_paths: list[Path],
    variant_candidate_paths: list[Path],
    *,
    default_topk_paths: list[Path] | None = None,
    variant_topk_paths: list[Path] | None = None,
    max_examples: int = 12,
) -> dict[str, Any]:
    default_queries = _load_candidate_queries(default_candidate_paths)
    variant_queries = _load_candidate_queries(variant_candidate_paths)
    default_debug = _load_topk_debug(default_topk_paths)
    variant_debug = _load_topk_debug(variant_topk_paths)
    common_keys = sorted(set(default_queries) & set(variant_queries))
    default_only = sorted(set(default_queries) - set(variant_queries))
    variant_only = sorted(set(variant_queries) - set(default_queries))

    label_counts: Counter[str] = Counter()
    failure_transitions: Counter[str] = Counter()
    metric_deltas: dict[str, list[float]] = defaultdict(list)
    channel_count_deltas: dict[str, list[float]] = defaultdict(list)
    channel_hit_transitions: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for key in common_keys:
        default_row = default_queries[key]
        variant_row = variant_queries[key]
        d_debug = default_debug.get(key)
        v_debug = variant_debug.get(key)
        labels = _classify_delta(default_row, variant_row, d_debug, v_debug)
        label_counts.update(labels)
        failure_transitions[f"{default_row.get('failure_type') or 'unknown'} -> {variant_row.get('failure_type') or 'unknown'}"] += 1
        for metric in METRIC_KEYS:
            metric_deltas[metric].append(_safe_float(variant_row.get(metric)) - _safe_float(default_row.get(metric)))
        for channel in ("lexical_sparse", "parent_session", "query_decomposition", "dense_semantic", "entity_aware", "temporal_anchor"):
            channel_count_deltas[channel].append(_channel_count(variant_row, channel) - _channel_count(default_row, channel))
            if _channel_gold_hit(default_row, channel) != _channel_gold_hit(variant_row, channel):
                channel_hit_transitions[f"{channel}:{_channel_gold_hit(default_row, channel)}->{_channel_gold_hit(variant_row, channel)}"] += 1
        for label in labels:
            if len(examples[label]) >= max_examples:
                continue
            gold_ids = {str(item) for item in default_row.get("gold_segment_ids") or variant_row.get("gold_segment_ids") or [] if str(item)}
            example = {
                "query_id": key,
                "query_text": str(default_row.get("query_text") or variant_row.get("query_text") or "")[:300],
                "default_failure": default_row.get("failure_type"),
                "variant_failure": variant_row.get("failure_type"),
                "default_candidate_recall@100": _safe_float(default_row.get("candidate_recall@100")),
                "variant_candidate_recall@100": _safe_float(variant_row.get("candidate_recall@100")),
                "default_final_recall@10": _safe_float(default_row.get("final_recall@10") or default_row.get("recall_frac@10")),
                "variant_final_recall@10": _safe_float(variant_row.get("final_recall@10") or variant_row.get("recall_frac@10")),
                "default_final_ndcg@10": _safe_float(default_row.get("final_ndcg@10") or default_row.get("ndcg_any@10")),
                "variant_final_ndcg@10": _safe_float(variant_row.get("final_ndcg@10") or variant_row.get("ndcg_any@10")),
                "default_parent_anchor_count": len(_parent_anchor_ids(default_row)),
                "variant_parent_anchor_count": len(_parent_anchor_ids(variant_row)),
                "default_parent_anchor_terms": sorted(_parent_anchor_terms(default_row))[:24],
                "variant_parent_anchor_terms": sorted(_parent_anchor_terms(variant_row))[:24],
                "default_channels_that_hit_gold": list(default_row.get("channels_that_hit_gold") or []),
                "variant_channels_that_hit_gold": list(variant_row.get("channels_that_hit_gold") or []),
            }
            if d_debug and v_debug:
                example["default_gold_rank_before_rerank"] = _rank_of_gold(list(d_debug.get("topk_before_rerank") or []), gold_ids)
                example["variant_gold_rank_before_rerank"] = _rank_of_gold(list(v_debug.get("topk_before_rerank") or []), gold_ids)
                example["default_top20_minus_variant_top20_count"] = len(
                    _source_ids(list(d_debug.get("topk_before_rerank") or [])[:20])
                    - _source_ids(list(v_debug.get("topk_before_rerank") or [])[:20])
                )
                example["variant_top20_minus_default_top20_count"] = len(
                    _source_ids(list(v_debug.get("topk_before_rerank") or [])[:20])
                    - _source_ids(list(d_debug.get("topk_before_rerank") or [])[:20])
                )
            examples[label].append(example)

    return {
        "schema": "dysonspherain.clonemem_lexical_interference.v1",
        "default_candidate_paths": [str(path) for path in default_candidate_paths],
        "variant_candidate_paths": [str(path) for path in variant_candidate_paths],
        "default_topk_paths": [str(path) for path in default_topk_paths or []],
        "variant_topk_paths": [str(path) for path in variant_topk_paths or []],
        "question_count": len(common_keys),
        "default_only_question_count": len(default_only),
        "variant_only_question_count": len(variant_only),
        "label_counts": dict(label_counts),
        "label_rates": {key: round(value / max(1, len(common_keys)), 6) for key, value in sorted(label_counts.items())},
        "failure_transitions": dict(failure_transitions),
        "mean_metric_deltas_variant_minus_default": {
            key: round(mean(values), 6) if values else 0.0 for key, values in sorted(metric_deltas.items())
        },
        "mean_channel_candidate_count_deltas_variant_minus_default": {
            key: round(mean(values), 6) if values else 0.0 for key, values in sorted(channel_count_deltas.items())
        },
        "channel_gold_hit_transitions": dict(channel_hit_transitions),
        "examples": dict(examples),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# CloneMem Lexical Interference Analysis",
        "",
        f"Questions compared: {summary['question_count']}",
        f"Default-only questions: {summary['default_only_question_count']}",
        f"Variant-only questions: {summary['variant_only_question_count']}",
        "",
        "## Mean Deltas",
        "",
        "All deltas are `variant - default`, where the intended variant is usually `no_lexical_probe`.",
        "",
        "| metric | delta |",
        "|---|---:|",
    ]
    for key, value in dict(summary.get("mean_metric_deltas_variant_minus_default") or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Labels", "", "| label | count | rate |", "|---|---:|---:|"])
    rates = dict(summary.get("label_rates") or {})
    for key, value in sorted(dict(summary.get("label_counts") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value} | {rates.get(key, 0.0)} |")
    lines.extend(["", "## Channel Candidate Count Deltas", "", "| channel | mean delta |", "|---|---:|"])
    for key, value in dict(summary.get("mean_channel_candidate_count_deltas_variant_minus_default") or {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Channel Gold-Hit Transitions", "", "| transition | count |", "|---|---:|"])
    for key, value in sorted(dict(summary.get("channel_gold_hit_transitions") or {}).items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Failure Transitions", "", "| transition | count |", "|---|---:|"])
    for key, value in sorted(dict(summary.get("failure_transitions") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Examples", ""])
    for label, rows in sorted(dict(summary.get("examples") or {}).items()):
        lines.append(f"### {label}")
        for row in rows[:5]:
            lines.append(
                f"- `{row.get('query_id')}` cand100 {row.get('default_candidate_recall@100')} -> {row.get('variant_candidate_recall@100')}; "
                f"final10 {row.get('default_final_recall@10')} -> {row.get('variant_final_recall@10')}; "
                f"ndcg10 {row.get('default_final_ndcg@10')} -> {row.get('variant_final_ndcg@10')}; "
                f"failure {row.get('default_failure')} -> {row.get('variant_failure')}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CloneMem default vs lexical-disabled diagnostics.")
    parser.add_argument("--default-candidate", action="append", required=True, type=Path)
    parser.add_argument("--variant-candidate", action="append", required=True, type=Path)
    parser.add_argument("--default-topk", action="append", type=Path)
    parser.add_argument("--variant-topk", action="append", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()
    summary = analyze(
        args.default_candidate,
        args.variant_candidate,
        default_topk_paths=args.default_topk,
        variant_topk_paths=args.variant_topk,
        max_examples=max(1, args.max_examples),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(summary, args.report)
    print(json.dumps({"questions": summary["question_count"], "out": str(args.out), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
