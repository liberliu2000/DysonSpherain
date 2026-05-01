from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _first_metric_group(metrics: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    if any("@" in str(key) for key in metrics):
        return metrics
    for key in ("session", "segment", "message", "dialog", "turn"):
        if isinstance(metrics.get(key), dict):
            return metrics[key]
    for value in metrics.values():
        if isinstance(value, dict):
            return value
    return {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _nested_metric(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload.get(key)
    for value in payload.values():
        if isinstance(value, dict):
            found = _nested_metric(value, *keys)
            if found is not None:
                return found
    return None


def _candidate_quality(row: dict[str, Any]) -> dict[str, Any]:
    candidate = row.get("candidate_recall") if isinstance(row.get("candidate_recall"), dict) else {}
    candidate_report = row.get("candidate_recall_report") if isinstance(row.get("candidate_recall_report"), dict) else {}
    metrics = _first_metric_group(row.get("metrics") if isinstance(row.get("metrics"), dict) else {})
    return {
        "recall_at_5": _first_present(metrics.get("recall_frac@5"), metrics.get("recall_any@5"), row.get("recall_at_5"), _nested_metric(row, "recall_frac@5", "recall_any@5")),
        "recall_at_10": _first_present(
            metrics.get("recall_frac@10"),
            metrics.get("recall_any@10"),
            candidate.get("final_recall@10"),
            candidate_report.get("final_recall@10"),
            row.get("recall_at_10"),
            _nested_metric(row, "recall_frac@10", "recall_any@10", "final_recall@10"),
        ),
        "ndcg_at_10": _first_present(
            metrics.get("ndcg_any@10"),
            candidate.get("final_ndcg@10"),
            candidate.get("candidate_ndcg@10"),
            candidate_report.get("final_ndcg@10"),
            candidate_report.get("candidate_ndcg@10"),
            row.get("ndcg_at_10"),
            _nested_metric(row, "ndcg_any@10", "final_ndcg@10", "candidate_ndcg@10"),
        ),
        "gold_rank": _first_present(
            candidate.get("gold_rank_after_inhibition"),
            candidate.get("gold_rank_after_rerank"),
            candidate.get("gold_rank_before_rerank"),
            row.get("gold_rank"),
            _nested_metric(row, "gold_rank_after_inhibition", "gold_rank_after_rerank", "gold_rank_before_rerank", "gold_rank"),
        ),
        "candidate_recall_at_100": _first_present(
            candidate.get("candidate_recall@100"),
            candidate_report.get("candidate_recall@100"),
            row.get("candidate_recall_at_100"),
            _nested_metric(row, "candidate_recall@100"),
        ),
    }


def _ranked_context(row: dict[str, Any], limit: int = 10) -> str:
    ranked = row.get("ranked_items") if isinstance(row.get("ranked_items"), list) else []
    parts: list[str] = []
    for index, item in enumerate(ranked[:limit], start=1):
        text = str(item.get("text") or item.get("preview") or "") if isinstance(item, dict) else str(item)
        corpus_id = str(item.get("corpus_id") or item.get("source_id") or "") if isinstance(item, dict) else ""
        score = item.get("score") if isinstance(item, dict) else None
        parts.append(f"[{index}] id={corpus_id} score={score} {text}")
    return "\n".join(parts)


def _oracle_context(row: dict[str, Any], oracle_rows_by_query: dict[str, list[dict[str, Any]]]) -> str:
    query_id = str(row.get("question_id") or row.get("query_id") or "")
    oracle_rows = oracle_rows_by_query.get(query_id, [])
    gold_ids = row.get("evidence_ids") or row.get("answer_session_ids") or row.get("gold_session_ids") or []
    candidate = row.get("candidate_recall") if isinstance(row.get("candidate_recall"), dict) else {}
    gold_ids = candidate.get("gold_evidence_ids") or candidate.get("gold_segment_ids") or gold_ids
    if not oracle_rows and not gold_ids:
        return ""
    return json.dumps(
        {
            "gold_evidence_ids": gold_ids,
            "oracle_rows": oracle_rows[:5],
            "answer": row.get("answer"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _load_oracle_rows(benchmark_dir: Path, benchmark: str) -> dict[str, list[dict[str, Any]]]:
    path = benchmark_dir / "reports" / "diagnostics" / f"{benchmark}_oracle_retrieval.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    except Exception:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("query_id") or ""), []).append(row)
    return grouped


def payloads_from_benchmark_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    benchmark = str(payload.get("benchmark_name") or metrics_path.parent.name)
    oracle_rows = _load_oracle_rows(metrics_path.parent, benchmark)
    rows = payload.get("results") if isinstance(payload.get("results"), list) else []
    if not rows:
        quality = _candidate_quality(payload)
        return [
            {
                "sample_id": benchmark,
                "query": f"Token economy diagnostic for {benchmark}",
                "history": json.dumps(payload, ensure_ascii=False, sort_keys=True)[:16000],
                "retrieved_context": json.dumps(payload.get("metrics") or payload, ensure_ascii=False, sort_keys=True),
                "metadata": json.dumps({"metrics_path": str(metrics_path), "benchmark": benchmark}, ensure_ascii=False),
                "retrieval_quality": quality,
                "candidate_count": int((payload.get("candidate_recall_report") or {}).get("candidate_count") or 0) if isinstance(payload.get("candidate_recall_report"), dict) else 0,
                "final_context_item_count": 1,
            }
        ]
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        query = str(row.get("question") or row.get("query") or row.get("query_text") or "")
        ranked_context = _ranked_context(row)
        candidate = row.get("candidate_recall") if isinstance(row.get("candidate_recall"), dict) else {}
        metadata = {
            "benchmark": benchmark,
            "metrics_path": str(metrics_path),
            "question_id": row.get("question_id") or row.get("query_id"),
            "candidate_recall@100": candidate.get("candidate_recall@100"),
            "final_ndcg@10": candidate.get("final_ndcg@10"),
            "gold_rank_after_inhibition": candidate.get("gold_rank_after_inhibition"),
            "gold_evidence_count": len(candidate.get("gold_evidence_ids") or candidate.get("gold_segment_ids") or []),
        }
        result.append(
            {
                "sample_id": str(row.get("question_id") or row.get("query_id") or row.get("sample_id") or f"{benchmark}_{index}"),
                "query": query,
                "history": json.dumps(row, ensure_ascii=False, sort_keys=True)[:20000],
                "retrieved_context": ranked_context or json.dumps(candidate or row.get("metrics") or {}, ensure_ascii=False, sort_keys=True),
                "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                "oracle_context": _oracle_context(row, oracle_rows),
                "gold_evidence": _oracle_context(row, oracle_rows),
                "retrieval_quality": _candidate_quality(row),
                "retrieved_evidence_count": len(row.get("ranked_items") or []),
                "candidate_count": sum(int(ch.get("candidate_count") or 0) for ch in (candidate.get("channel_stats") or {}).values() if isinstance(ch, dict))
                or int(candidate.get("candidate_count") or (row.get("candidate_count") or 0)),
                "final_context_item_count": min(10, len(row.get("ranked_items") or [])) if isinstance(row.get("ranked_items"), list) else 1,
            }
        )
    return result


def payloads_from_benchmark_artifact_root(root: Path) -> list[dict[str, Any]]:
    metrics_paths = sorted(root.glob("*/metrics.json"))
    if not metrics_paths:
        metrics_paths = sorted(root.glob("*/*/metrics.json"))
    if root.name.endswith(".json") and root.name == "metrics.json":
        metrics_paths = [root]
    payloads: list[dict[str, Any]] = []
    for path in metrics_paths:
        payloads.extend(payloads_from_benchmark_metrics(path))
    return payloads
