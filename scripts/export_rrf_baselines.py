#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT_RESULT_DIR = Path("/Users/yanbo/DysonSpherain/BenchmarkResult")
DEFAULT_OUT_ROOT = ROOT_RESULT_DIR / "20260428_artifact_rrf_baselines_v1"
KS = (1, 3, 5, 10, 30, 50)

DEFAULT_SOURCES = {
    "longmemeval": {
        "dense": ROOT_RESULT_DIR / "20260428_matched_baseline_full_v1/longmemeval_vector/longmemeval/merged_metrics.json",
        "bm25": ROOT_RESULT_DIR / "20260428_longmemeval_bm25_full_v1/merged_metrics.json",
    },
    "locomo": {
        "dense": ROOT_RESULT_DIR / "20260428_matched_baseline_full_v1/locomo_vector/locomo/merged_metrics.json",
        "bm25": ROOT_RESULT_DIR / "20260428_locomo_bm25_full_v1/locomo/merged_metrics.json",
    },
    "knowme": {
        "dense": ROOT_RESULT_DIR / "20260428_matched_baseline_full_v1/knowme_vector/knowme/merged_metrics.json",
        "bm25": ROOT_RESULT_DIR / "20260428_knowme_bm25_full_v1/knowme/merged_metrics.json",
    },
    "clonemem": {
        "dense": ROOT_RESULT_DIR / "20260428_matched_baseline_full_v1/clonemem_vector/clonemem/merged_metrics.json",
        "bm25": ROOT_RESULT_DIR / "20260428_clonemem_bm25_full_v1/clonemem/merged_metrics.json",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results") or payload.get("queries") or payload.get("rows") or []
    return [row for row in rows if isinstance(row, dict)]


def _chunk_metrics_for(path: Path) -> list[Path]:
    root = path.parent if path.is_file() else path
    if path.name in {"merged_metrics.json", "compact_metrics.json"}:
        root = path.parent
    candidates = sorted(root.glob("chunk_*/metrics.json"))
    if candidates:
        return candidates
    return [path] if path.is_file() else sorted(path.glob("**/metrics.json"))


def collect_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path) if path.is_file() and path.stat().st_size < 128 * 1024 * 1024 else {}
    rows = _rows_from_payload(payload)
    if rows:
        return rows
    collected: list[dict[str, Any]] = []
    for metrics_path in _chunk_metrics_for(path):
        if metrics_path == path:
            continue
        collected.extend(_rows_from_payload(load_json(metrics_path)))
    return collected


def row_key(row: dict[str, Any]) -> str:
    question_id = row.get("question_id") or row.get("native_question_id")
    if question_id:
        parts = [
            str(question_id),
            str(row.get("sample_id") or row.get("sample_name") or row.get("person_id") or ""),
            str(row.get("native_question_id") or ""),
            str(row.get("question") or ""),
        ]
        return "qid:" + "|".join(parts)
    parts = [
        str(row.get("sample_id") or row.get("sample_name") or ""),
        str(row.get("qa_index") or row.get("original_qa_index") or ""),
        str(row.get("question") or ""),
    ]
    return "composite:" + "|".join(parts)


def index_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for row in rows:
        key = row_key(row)
        if key in indexed:
            duplicates.add(key)
            continue
        indexed[key] = row
    for key in duplicates:
        indexed.pop(key, None)
    return indexed


def ranked_corpus_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in row.get("ranked_items") or []:
        if not isinstance(item, dict):
            continue
        corpus_id = str(item.get("corpus_id") or item.get("source_id") or item.get("source_segment_id") or "")
        if corpus_id and corpus_id not in seen:
            ids.append(corpus_id)
            seen.add(corpus_id)
    if ids:
        return ids
    for key in ("retrieved_dialog_ids", "retrieved_session_ids", "retrieved_segment_ids", "retrieved_trace_ids"):
        for corpus_id in row.get(key) or []:
            corpus_id = str(corpus_id)
            if corpus_id and corpus_id not in seen:
                ids.append(corpus_id)
                seen.add(corpus_id)
    return ids


def rrf_fuse(rankings: list[list[str]], *, rrf_k: int = 60, limit: int = 100) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    serial = 0
    for ranking in rankings:
        for rank, corpus_id in enumerate(ranking, start=1):
            if corpus_id not in first_seen:
                first_seen[corpus_id] = serial
                serial += 1
            scores[corpus_id] += 1.0 / (rrf_k + rank)
    return [
        corpus_id
        for corpus_id, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], first_seen.get(item[0], 10**9), item[0]),
        )[:limit]
    ]


def dcg(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))


def ndcg(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    relevances = [1.0 if item_id in gold_ids else 0.0 for item_id in ranked_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    denom = dcg(ideal, k)
    return dcg(relevances, k) / denom if denom else 0.0


def evaluate(ranked_ids: list[str], gold_ids: set[str], k: int) -> dict[str, float]:
    if not gold_ids:
        return {"recall_frac": 0.0, "recall_any": 0.0, "recall_all": 0.0, "ndcg_any": 0.0}
    top_ids = set(ranked_ids[:k])
    hit_count = len(top_ids & gold_ids)
    return {
        "recall_frac": hit_count / len(gold_ids),
        "recall_any": float(hit_count > 0),
        "recall_all": float(gold_ids.issubset(top_ids)),
        "ndcg_any": ndcg(ranked_ids, gold_ids, k),
    }


def session_id_from_corpus_id(corpus_id: str) -> str:
    dialog_match = re.match(r"^D(\d+):", corpus_id)
    if dialog_match:
        return f"session_{int(dialog_match.group(1))}"
    if "_turn_" in corpus_id:
        return corpus_id.rsplit("_turn_", 1)[0]
    return corpus_id


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics") or {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    buckets[f"{key}.{sub_key}"].append(float(sub_value))
            elif isinstance(value, (int, float)):
                buckets[str(key)].append(float(value))
    out: dict[str, Any] = {}
    for key, values in sorted(buckets.items()):
        if "." in key:
            prefix, sub_key = key.split(".", 1)
            out.setdefault(prefix, {})[sub_key] = sum(values) / len(values)
        else:
            out[key] = sum(values) / len(values)
    return out


def _flat_metrics(ranked_ids: list[str], gold_ids: set[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in KS:
        values = evaluate(ranked_ids, gold_ids, k)
        for name, value in values.items():
            metrics[f"{name}@{k}"] = value
    return metrics


def _longmemeval_metrics(ranked_ids: list[str], row: dict[str, Any]) -> dict[str, dict[str, float]]:
    gold = {str(item) for item in row.get("answer_session_ids") or []}
    session_ranked = []
    seen: set[str] = set()
    for corpus_id in ranked_ids:
        session_id = session_id_from_corpus_id(corpus_id)
        if session_id not in seen:
            session_ranked.append(session_id)
            seen.add(session_id)
    session_metrics: dict[str, float] = {}
    turn_metrics: dict[str, float] = {}
    for k in KS:
        session_values = evaluate(session_ranked, gold, k)
        turn_values = evaluate(ranked_ids, gold, k)
        session_metrics[f"recall_any@{k}"] = session_values["recall_any"]
        session_metrics[f"ndcg_any@{k}"] = session_values["ndcg_any"]
        turn_metrics[f"recall_any@{k}"] = turn_values["recall_any"]
        turn_metrics[f"ndcg_any@{k}"] = turn_values["ndcg_any"]
    return {"session": session_metrics, "turn": turn_metrics}


def _locomo_metrics(ranked_ids: list[str], row: dict[str, Any]) -> dict[str, dict[str, float]]:
    dialog_gold = {str(item) for item in row.get("gold_dialog_ids") or []}
    session_gold = {str(item) for item in row.get("gold_session_ids") or []}
    session_ranked = []
    seen_sessions: set[str] = set()
    for corpus_id in ranked_ids:
        session_id = session_id_from_corpus_id(corpus_id)
        if session_id not in seen_sessions:
            session_ranked.append(session_id)
            seen_sessions.add(session_id)
    dialog_metrics: dict[str, float] = {}
    session_metrics: dict[str, float] = {}
    for k in KS:
        dialog_values = evaluate(ranked_ids, dialog_gold, k)
        session_values = evaluate(session_ranked, session_gold, k)
        for name, value in dialog_values.items():
            dialog_metrics[f"{name}@{k}"] = value
        for name, value in session_values.items():
            session_metrics[f"{name}@{k}"] = value
    return {"dialog": dialog_metrics, "session": session_metrics}


def recompute_row(benchmark: str, dense_row: dict[str, Any], bm25_row: dict[str, Any], *, rrf_k: int, limit: int) -> dict[str, Any]:
    ranked_ids = rrf_fuse([ranked_corpus_ids(dense_row), ranked_corpus_ids(bm25_row)], rrf_k=rrf_k, limit=limit)
    base = dict(dense_row)
    if benchmark == "longmemeval":
        metrics = _longmemeval_metrics(ranked_ids, dense_row)
    elif benchmark == "locomo":
        metrics = _locomo_metrics(ranked_ids, dense_row)
        base["retrieved_dialog_ids"] = ranked_ids[:limit]
        base["retrieved_session_ids"] = [session_id_from_corpus_id(item) for item in ranked_ids[:limit]]
    else:
        gold = {str(item) for item in dense_row.get("evidence_ids") or []}
        metrics = _flat_metrics(ranked_ids, gold)
        if benchmark == "knowme":
            base["retrieved_segment_ids"] = ranked_ids[:limit]
        if benchmark == "clonemem":
            base["retrieved_trace_ids"] = ranked_ids[:limit]
    base["metrics"] = metrics
    base["ranked_items"] = [{"corpus_id": corpus_id, "rrf_rank": index + 1} for index, corpus_id in enumerate(ranked_ids[:limit])]
    base["candidate_recall"] = None
    return base


def export_benchmark(
    benchmark: str,
    *,
    dense_path: Path,
    bm25_path: Path,
    out_root: Path,
    rrf_k: int = 60,
    limit: int = 100,
) -> dict[str, Any]:
    dense_rows = index_rows(collect_rows(dense_path))
    bm25_rows = index_rows(collect_rows(bm25_path))
    common_keys = sorted(set(dense_rows) & set(bm25_rows))
    if not common_keys:
        raise RuntimeError(f"No aligned rows for {benchmark}: dense={dense_path} bm25={bm25_path}")
    fused_rows = [
        recompute_row(benchmark, dense_rows[key], bm25_rows[key], rrf_k=rrf_k, limit=limit)
        for key in common_keys
    ]
    payload = {
        "schema": "dysonspherain.artifact_rrf_baseline.v1",
        "benchmark": benchmark,
        "baseline": "dense_bm25_rrf",
        "method": "dense_bm25_rrf",
        "mode": "artifact_rrf",
        "run_type": "full",
        "total_question_count": len(fused_rows),
        "question_count": len(fused_rows),
        "rrf_k": rrf_k,
        "rank_limit": limit,
        "metrics": aggregate(fused_rows),
        "results": fused_rows,
        "source_files": {"dense": str(dense_path), "bm25": str(bm25_path)},
        "fallback_in_use": False,
        "embedding_info": {"fallback_in_use": False},
        "formal_use_warning": "Artifact-level RRF baseline recomputed from full dense/vector and BM25 artifacts; no benchmark gold IDs are injected into ranking.",
    }
    out_dir = out_root / f"{benchmark}_dense_bm25_rrf"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    compact = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "results",
        }
    }
    compact["schema"] = "dysonspherain.artifact_rrf_baseline.compact.v1"
    compact["detail_metrics_path"] = str(out_dir / "metrics.json")
    (out_dir / "compact_metrics.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "benchmark": benchmark,
        "question_count": len(fused_rows),
        "metrics_path": str(out_dir / "metrics.json"),
        "compact_metrics_path": str(out_dir / "compact_metrics.json"),
        "metrics": payload["metrics"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dense+BM25 artifact-level RRF baselines.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rank-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for benchmark, paths in DEFAULT_SOURCES.items():
        records.append(
            export_benchmark(
                benchmark,
                dense_path=paths["dense"],
                bm25_path=paths["bm25"],
                out_root=args.out_root,
                rrf_k=args.rrf_k,
                limit=args.rank_limit,
            )
        )
    summary = {
        "schema": "dysonspherain.artifact_rrf_export_summary.v1",
        "records": records,
        "out_root": str(args.out_root),
    }
    (args.out_root / "rrf_baseline_exports.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"records": len(records), "out_root": str(args.out_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
