from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from dysonspherain.product import (
    apply_maintenance_suggestion,
    benchmark_compare,
    benchmark_record,
    configure_embedding_backend,
    configure_product_vector_backend,
    create_context_pack,
    doctor,
    init_product_store,
    maintenance_suggestions,
    privacy_policy,
    product_embedding_backends,
    product_vector_backends,
    rebuild_product_embeddings,
    rebuild_product_vector_index,
    remember,
    register_alias,
    retrieve,
    resolve_alias,
    runtime_event,
)


def run_smoke(base_dir: Path) -> dict[str, object]:
    project = "test_project"
    init = init_product_store(base_dir, project_id=project)
    created = remember(
        base_dir,
        project_id=project,
        evidence_type="decision",
        text="Use temporal expansion for ambiguous benchmark queries.",
        tags=["benchmark", "temporal"],
    )
    duplicate = remember(
        base_dir,
        project_id=project,
        evidence_type="decision",
        text="Use temporal expansion for ambiguous benchmark queries.",
        tags=["benchmark", "temporal"],
    )
    retrieved = retrieve(
        base_dir,
        project_id=project,
        query="temporal expansion ambiguous benchmark",
        show_audit=True,
        context_pack=True,
        max_tokens=2000,
    )
    wake = create_context_pack(base_dir, project_id=project, query="temporal expansion ambiguous benchmark", max_tokens=2000)
    before = runtime_event(base_dir, project_id=project, event_type="before_task", payload={"task": "debug retrieval regression"}, max_tokens=1200)
    error = runtime_event(
        base_dir,
        project_id=project,
        event_type="on_error",
        payload={"error": 'Traceback (most recent call last):\n  File "x.py", line 3, in f\nValueError: bad'},
        max_tokens=1200,
    )
    after = runtime_event(base_dir, project_id=project, event_type="after_task", payload={"summary": "Stored product smoke summary"}, max_tokens=1200)

    run_a = base_dir / "run_a" / "metrics.json"
    run_b = base_dir / "run_b" / "metrics.json"
    run_a.parent.mkdir(parents=True, exist_ok=True)
    run_b.parent.mkdir(parents=True, exist_ok=True)
    run_a.write_text(json.dumps({"benchmark": "unit", "metrics": {"recall": 0.7, "candidate_recall@100": 0.9, "latency_ms": 100}}), encoding="utf-8")
    run_b.write_text(json.dumps({"benchmark": "unit", "metrics": {"recall": 0.8, "candidate_recall@100": 0.95, "latency_ms": 110}}), encoding="utf-8")
    bench = benchmark_record(base_dir, project_id=project, artifact=run_b, benchmark="unit")
    compare = benchmark_compare(base_dir, project_id=project, current=run_b, baseline=run_a)
    register_alias(base_dir, project_id=project, alias="temporal expansion policy", canonical=created["capsule_id"])
    alias = resolve_alias(base_dir, project_id=project, value="temporal expansion policy")
    maintenance = maintenance_suggestions(base_dir, project_id=project)
    duplicate_suggestion = next((item for item in maintenance.get("suggestions", []) if item.get("type") == "duplicate_merge"), None)
    applied_maintenance = (
        apply_maintenance_suggestion(base_dir, project_id=project, suggestion_id=duplicate_suggestion["suggestion_id"], canonical_id=created["capsule_id"])
        if duplicate_suggestion
        else {"status": "missing"}
    )
    embedding_rebuild = rebuild_product_embeddings(base_dir, project_id=project)
    backends = product_embedding_backends(base_dir)
    backend_config = configure_embedding_backend(base_dir, backend="local_hash_embedding")
    vector_backends = product_vector_backends(base_dir)
    vector_config = configure_product_vector_backend(base_dir, backend="sqlite_inline")
    vector_rebuild = rebuild_product_vector_index(base_dir, project_id=project)
    health = doctor(base_dir, project_id=project)
    privacy = privacy_policy(base_dir)
    trace = retrieved.get("retrieval_trace") or {}
    dense_probe = (trace.get("probe_results") or {}).get("dense_probe") or {}

    checks = {
        "raw_trace_stored": bool(created.get("raw_id")),
        "capsule_created": bool(created.get("capsule_id")),
        "dense_probe_available": int(dense_probe.get("count") or 0) > 0 and not trace.get("unavailable_probes"),
        "retrieval_returns_capsule": created["capsule_id"] in [item["capsule_id"] for item in retrieved.get("candidates", [])],
        "retrieval_trace_saved": bool((retrieved.get("retrieval_trace") or {}).get("trace_id")),
        "wake_context_pack": bool(wake.get("context_pack_id")),
        "runtime_events": all(item.get("status") == "ok" for item in (before, error, after)),
        "benchmark_registered": bool(bench.get("run_id")),
        "regression_report": compare.get("status") == "ok",
        "alias_resolution": alias.get("canonical") == created["capsule_id"],
        "maintenance_suggestions": any(item.get("type") == "duplicate_merge" and duplicate["capsule_id"] in item.get("capsule_ids", []) for item in maintenance.get("suggestions", [])),
        "maintenance_apply": applied_maintenance.get("status") == "applied",
        "embedding_rebuild": embedding_rebuild.get("status") == "ok" and embedding_rebuild.get("capsules_seen", 0) >= 1,
        "embedding_backend_registry": bool(backends.get("backends", {}).get("local_hash_embedding", {}).get("available")) and backend_config.get("status") == "ok",
        "vector_backend_registry": bool(vector_backends.get("backends", {}).get("sqlite_inline", {}).get("available")) and vector_config.get("status") == "ok" and vector_rebuild.get("status") == "ok",
        "health_report": health.get("status") in {"ok", "warning"},
        "privacy_local_first": bool(privacy.get("local_only")),
        "encryption_status_reported": privacy.get("encryption_at_rest", {}).get("status") == "not_configured",
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "project": project,
        "base_dir": str(base_dir),
        "checks": checks,
        "artifacts": {
            "db": init["db_path"],
            "context_pack_id": wake.get("context_pack_id"),
            "benchmark_dashboard": str(base_dir / ".memory" / "artifacts" / "benchmark_lab"),
        },
        "known_limitations": [
            "large product stores should configure the optional Chroma ANN vector index instead of SQLite inline scan",
            "SQLCipher encryption requires optional dependencies and an operator-provided key environment variable",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/product_acceptance_smoke.json"))
    args = parser.parse_args()
    if args.base_dir is None:
        with tempfile.TemporaryDirectory(prefix="dyson_product_acceptance_") as tmp:
            payload = run_smoke(Path(tmp))
    else:
        args.base_dir.mkdir(parents=True, exist_ok=True)
        payload = run_smoke(args.base_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
