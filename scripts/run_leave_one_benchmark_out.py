#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BENCHMARKS = ["longmemeval", "locomo", "knowme", "clonemem"]
PRIMARY_METRICS = [
    "recall_any@5",
    "recall_any@10",
    "recall@10",
    "Recall@10",
    "recall_frac@10",
    "final_recall@10",
    "ndcg@10",
    "NDCG@10",
    "ndcg_any@10",
    "final_ndcg@10",
    "candidate_recall@100",
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_metric(payload: Any, metric: str) -> Any:
    if not isinstance(payload, dict):
        return None
    if metric in payload:
        return payload[metric]
    lower_metric = metric.lower()
    for key, value in payload.items():
        if str(key).lower() == lower_metric:
            return value
    for preferred in ("session", "segment", "turn", "dialog"):
        nested = payload.get(preferred)
        if isinstance(nested, dict):
            found = _find_metric(nested, metric)
            if found is not None:
                return found
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_metric(value, metric)
            if found is not None:
                return found
    return None


def _primary_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    return {key: value for key in PRIMARY_METRICS if (value := _find_metric(metrics, key)) is not None}


def _result_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    metrics = _primary_metrics(payload)
    candidate = payload.get("candidate_recall_summary") if isinstance(payload.get("candidate_recall_summary"), dict) else {}
    failures = payload.get("failure_taxonomy") if isinstance(payload.get("failure_taxonomy"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    embedding = payload.get("embedding_info") if isinstance(payload.get("embedding_info"), dict) else {}
    vector = payload.get("vector_info") if isinstance(payload.get("vector_info"), dict) else {}
    return {
        "source_metrics_path": str(path),
        "benchmark": str(payload.get("benchmark") or payload.get("dataset") or ""),
        "question_count": payload.get("total_question_count") or payload.get("question_count"),
        "sample_count": payload.get("sample_count") or payload.get("total_sample_count"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "wall_clock_elapsed_seconds": payload.get("wall_clock_elapsed_seconds"),
        "serial_elapsed_seconds_sum": payload.get("serial_elapsed_seconds_sum"),
        "quality_guardrail_status": payload.get("quality_guardrail_status"),
        "metrics": metrics,
        "candidate_recall_summary": candidate,
        "failure_taxonomy": failures,
        "timings": timings,
        "embedding_info": embedding,
        "vector_info": vector,
        "fallback_in_use": bool(
            payload.get("fallback_in_use")
            or embedding.get("fallback_in_use")
            or vector.get("fallback_in_use")
        ),
    }


def _parse_result_map(values: list[str] | None) -> dict[str, Path]:
    result_map: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--heldout-result must be BENCHMARK=PATH, got: {value}")
        benchmark, raw_path = value.split("=", 1)
        benchmark = benchmark.strip().lower()
        if benchmark not in BENCHMARKS:
            raise SystemExit(f"unknown held-out benchmark: {benchmark}")
        result_map[benchmark] = Path(raw_path).expanduser()
    return result_map


def _discover_lobo_result(results_root: Path, held_out: str) -> Path | None:
    if not results_root.exists():
        return None
    candidates: list[Path] = []
    for name in ("merged_metrics.json", "metrics.json", "compact_metrics.json"):
        candidates.extend(results_root.glob(f"**/*lobo*{held_out}*/**/{name}"))
        candidates.extend(results_root.glob(f"**/*leave*{held_out}*/**/{name}"))
        candidates.extend(results_root.glob(f"**/*heldout*{held_out}*/**/{name}"))
    candidates = [path for path in candidates if path.is_file() and path.stat().st_size < 25 * 1024 * 1024]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _policy_payload(held_out: str, train: list[str]) -> dict[str, Any]:
    payload = {
        "schema": "dysonspherain.route_policy_lobo.v1",
        "status": "policy_config_ready",
        "train_benchmarks": train,
        "held_out": held_out,
        "selection_rule": "Use route-aware gating defaults selected without held-out benchmark metrics.",
        "environment": {
            "SPHERE_ROUTE_AWARE_GATING_ENABLED": "true",
            "SPHERE_ROUTE_AWARE_GATING_AGGRESSIVENESS": "safe",
            "SPHERE_RETRIEVAL_EARLY_EXIT_ENABLED": "true",
            "SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING": "1",
        },
        "constraints": [
            "no held-out benchmark tuning",
            "preserve dense preserve and safe fusion",
            "do not hardcode gold ids or benchmark answers",
            "do not use local_hash fallback for formal validation",
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["config_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload


def build_protocol(*, results_root: Path | None = None, heldout_results: dict[str, Path] | None = None) -> list[dict[str, object]]:
    rows = []
    heldout_results = heldout_results or {}
    for held_out in BENCHMARKS:
        train = [name for name in BENCHMARKS if name != held_out]
        policy_id = f"route_policy_train_{'_'.join(train)}"
        result_artifact = Path(f"artifacts/lobo/{held_out}_heldout_metrics.json")
        source = heldout_results.get(held_out)
        if source is None and results_root is not None:
            source = _discover_lobo_result(results_root, held_out)
        result_summary: dict[str, Any] | None = None
        status = "pending"
        note = "No held-out full benchmark artifact has been produced for this protocol row yet."
        if source is not None and source.exists():
            result_summary = _result_summary(source)
            result_artifact.parent.mkdir(parents=True, exist_ok=True)
            result_artifact.write_text(
                json.dumps(
                    {
                        "schema": "dysonspherain.lobo_heldout_result.v1",
                        "held_out": held_out,
                        "train_benchmarks": train,
                        **result_summary,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            status = "available"
            note = "Artifact-backed held-out result summary is available."
        rows.append(
            {
                "held_out": held_out,
                "train_benchmarks": train,
                "route_policy_artifact": f"artifacts/lobo/{policy_id}.json",
                "result_artifact": str(result_artifact),
                "source_metrics_path": str(source) if source else None,
                "status": status,
                "note": note,
                "metrics": dict(result_summary.get("metrics") or {}) if result_summary else None,
                "candidate_recall_summary": dict(result_summary.get("candidate_recall_summary") or {}) if result_summary else None,
                "quality_guardrail_status": result_summary.get("quality_guardrail_status") if result_summary else None,
                "elapsed_seconds": result_summary.get("elapsed_seconds") if result_summary else None,
                "wall_clock_elapsed_seconds": result_summary.get("wall_clock_elapsed_seconds") if result_summary else None,
                "fallback_in_use": result_summary.get("fallback_in_use") if result_summary else None,
            }
        )
    return rows


def write_policy_stubs(rows: list[dict[str, object]]) -> None:
    for row in rows:
        path = Path(str(row["route_policy_artifact"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _policy_payload(str(row["held_out"]), list(row["train_benchmarks"]))  # type: ignore[arg-type]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_report(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Leave-One-Benchmark-Out Report",
        "",
        "This report defines the held-out generalization protocol. Result rows remain `pending` until full held-out artifacts exist.",
        "",
        "| held_out | train_benchmarks | policy_artifact | result_artifact | status | metrics | latency | notes |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        metric_text = ", ".join(f"{key}={value}" for key, value in metrics.items()) or ""
        latency = row.get("wall_clock_elapsed_seconds") or row.get("elapsed_seconds") or ""
        lines.append(
            f"| {row['held_out']} | {', '.join(row['train_benchmarks'])} | {row['route_policy_artifact']} | "
            f"{row['result_artifact']} | {row['status']} | {metric_text} | {latency} | {row.get('note') or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leave-one-benchmark-out generalization protocol artifacts.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/lobo/lobo_protocol.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/leave_one_benchmark_out_report.md"))
    parser.add_argument("--results-root", type=Path, default=None, help="Optional directory to discover explicitly named LOBO held-out runs.")
    parser.add_argument(
        "--heldout-result",
        action="append",
        help="Explicit artifact binding in BENCHMARK=PATH form. The path must be a real held-out run; tuned full runs are not auto-promoted.",
    )
    args = parser.parse_args()

    rows = build_protocol(results_root=args.results_root, heldout_results=_parse_result_map(args.heldout_result))
    write_policy_stubs(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "dysonspherain.lobo_protocol.v1", "rows": rows}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_report(rows, args.report)
    print(json.dumps({"rows": len(rows), "out": str(args.out), "report": str(args.report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
