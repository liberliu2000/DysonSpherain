from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import sleep
from typing import Any

from .events import stable_hash
from .ledger import append_event_payload, replay_events, write_projection
from .situation_graph import build_and_save_graph


@dataclass(frozen=True)
class MaintenanceJob:
    job_id: str
    trigger: str
    job_type: str
    event_ids: list[str]
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaintenanceResult:
    job_id: str
    status: str
    actions: list[str]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRIGGER_JOBS = {
    "session_ended": ["refresh_projection", "prepare_next_session_context"],
    "large_file_changed": ["refresh_index", "refresh_projection"],
    "benchmark_finished": ["extract_metric_deltas", "refresh_projection"],
    "metric_regression_detected": ["detect_contradictions", "generate_recovery_summary", "refresh_projection"],
    "artifact_updated": ["refresh_projection", "refresh_index"],
    "user_preference_declared": ["extract_constraints", "refresh_projection"],
    "decision_changed": ["extract_decisions", "detect_contradictions", "refresh_projection"],
    "contradiction_detected": ["detect_contradictions", "generate_recovery_summary"],
    "repeated_failure_detected": ["extract_failures", "generate_recovery_summary"],
    "index_staleness_detected": ["refresh_index"],
    "cache_miss_spike_detected": ["refresh_index"],
}


def schedule_memory_maintenance(trigger: str, event_ids: list[str]) -> list[MaintenanceJob]:
    job_types = TRIGGER_JOBS.get(trigger, ["refresh_projection"])
    return [
        MaintenanceJob(
            job_id=f"job_{stable_hash([trigger, job_type, sorted(event_ids)])[:18]}",
            trigger=trigger,
            job_type=job_type,
            event_ids=list(event_ids),
            metadata={"idempotency_key": stable_hash([trigger, job_type, sorted(event_ids)])},
        )
        for job_type in job_types
    ]


def _append_job_log(base_dir: Path, payload: dict[str, Any]) -> None:
    path = base_dir / "data" / "maintenance" / "jobs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _queue_path(base_dir: Path) -> Path:
    return base_dir / "data" / "maintenance" / "queue.jsonl"


def _append_queue_record(base_dir: Path, payload: dict[str, Any]) -> None:
    path = _queue_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def enqueue_maintenance_jobs(base_dir: Path, trigger: str, event_ids: list[str], *, project: str = "DysonSpherain") -> list[MaintenanceJob]:
    jobs = schedule_memory_maintenance(trigger, event_ids)
    for job in jobs:
        _append_queue_record(base_dir, {**job.to_dict(), "status": "pending", "project": project})
    return jobs


def load_pending_jobs(base_dir: Path, *, project: str = "DysonSpherain") -> list[MaintenanceJob]:
    path = _queue_path(base_dir)
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("project") or project) != project:
            continue
        latest[str(row.get("job_id") or "")] = row
    jobs: list[MaintenanceJob] = []
    for row in latest.values():
        if row.get("status") != "pending":
            continue
        jobs.append(
            MaintenanceJob(
                job_id=str(row["job_id"]),
                trigger=str(row.get("trigger") or "manual"),
                job_type=str(row.get("job_type") or "refresh_projection"),
                event_ids=list(row.get("event_ids") or []),
                status="pending",
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return sorted(jobs, key=lambda job: job.job_id)


def run_scheduler_once(base_dir: Path, *, project: str = "DysonSpherain", limit: int = 10) -> dict[str, Any]:
    pending = load_pending_jobs(base_dir, project=project)[: max(1, int(limit))]
    results: list[dict[str, Any]] = []
    for job in pending:
        _append_queue_record(base_dir, {**job.to_dict(), "status": "running", "project": project})
        result = run_maintenance_job(base_dir, job, project=project)
        _append_queue_record(base_dir, {**job.to_dict(), "status": "finished" if result.status == "ok" else "error", "project": project, "result": result.to_dict()})
        results.append(result.to_dict())
    return {"status": "ok", "pending_before": len(pending), "ran": len(results), "results": results}


def run_scheduler_daemon(base_dir: Path, *, project: str = "DysonSpherain", interval_seconds: float = 5.0, limit: int = 10, max_loops: int | None = None) -> dict[str, Any]:
    loops = 0
    total_ran = 0
    last_result: dict[str, Any] = {"status": "ok", "ran": 0}
    while max_loops is None or loops < max_loops:
        last_result = run_scheduler_once(base_dir, project=project, limit=limit)
        total_ran += int(last_result.get("ran") or 0)
        loops += 1
        if max_loops is not None and loops >= max_loops:
            break
        sleep(max(0.1, float(interval_seconds)))
    return {"status": "ok", "loops": loops, "total_ran": total_ran, "last_result": last_result}


def _event_summary(base_dir: Path, project: str, event_types: set[str]) -> list[dict[str, Any]]:
    return [event.to_dict() for event in replay_events(base_dir, project=project) if event.event_type in event_types]


def _detect_contradictions(base_dir: Path, project: str) -> list[dict[str, Any]]:
    events = replay_events(base_dir, project=project)
    current_by_key: dict[str, str] = {}
    contradictions: list[dict[str, Any]] = []
    for event in events:
        key = str(event.payload.get("key") or event.payload.get("title") or event.event_type)
        value = str(event.payload.get("value") or event.payload.get("content") or event.payload.get("summary") or "")
        if event.event_type in {"decision_made", "constraint_added", "constraint_changed"} and key in current_by_key and current_by_key[key] != value:
            contradictions.append({"key": key, "previous": current_by_key[key], "current": value, "event_id": event.event_id})
        if value:
            current_by_key[key] = value
    return contradictions


def _event_text_for_index(event: Any) -> str:
    payload = event.payload
    return "\n".join(
        str(value)
        for value in (
            payload.get("title"),
            payload.get("summary"),
            payload.get("content"),
            payload.get("message"),
            payload.get("path"),
            event.event_type,
        )
        if value
    )


def index_ledger_events(base_dir: Path, event_ids: list[str], *, project: str = "DysonSpherain") -> dict[str, Any]:
    events = replay_events(base_dir, project=project)
    selected = [event for event in events if not event_ids or event.event_id in set(event_ids)]
    chunks: list[dict[str, Any]] = []
    for index, event in enumerate(selected):
        text = _event_text_for_index(event)
        if not text.strip():
            continue
        chunks.append(
            {
                "chunk_id": f"ledger_{event.event_id}",
                "text": text,
                "node_id": event.event_id,
                "shell": 0,
                "sector": "memory_runtime",
                "zone": "ledger",
                "cell": event.event_type,
                "chunk_index": index,
                "grain": "event",
                "scope": "project",
                "workspace": str(base_dir),
                "project": project,
                "session_id": event.session_id or "",
                "source_path": "",
                "source_type": "memory_event",
                "source_ref": event.event_id,
                "summary": text[:240],
                "created_at": event.timestamp,
                "timestamp": event.timestamp,
            }
        )
    try:
        from sphere_cli.runtime import UnifiedMemoryRuntime

        runtime = UnifiedMemoryRuntime.from_base_dir(base_dir, config_overrides={"project_name": project})
        before = runtime.services.vector_store.info()
        runtime.services.vector_store.upsert_chunks(chunks)
        after = runtime.services.vector_store.info()
        status = "ok"
        error = ""
    except Exception as exc:
        before = {}
        after = {}
        status = "error"
        error = str(exc)
    report = {
        "project": project,
        "status": status,
        "event_ids": event_ids,
        "indexed_event_count": len(chunks) if status == "ok" else 0,
        "attempted_event_count": len(chunks),
        "backend_before": before,
        "backend_after": after,
        "error": error,
        "fallback_in_use": bool(after.get("fallback_in_use") or after.get("vector_fallback_in_use")) if after else False,
    }
    write_projection(base_dir, "ledger_vector_index_report.json", report)
    return report


def run_maintenance_job(base_dir: Path, job: MaintenanceJob, *, project: str = "DysonSpherain") -> MaintenanceResult:
    actions: list[str] = []
    try:
        _append_job_log(base_dir, {"status": "started", **job.to_dict()})
        append_event_payload(
            base_dir,
            event_type="maintenance_job_scheduled",
            payload=job.to_dict(),
            source="memory_runtime_scheduler",
            actor="system",
            project=project,
        )
        if job.job_type in {"refresh_projection", "extract_decisions", "extract_constraints", "extract_failures", "extract_metric_deltas", "detect_contradictions"}:
            build_and_save_graph(base_dir, project=project)
            actions.append("projection_refreshed")
        if job.job_type == "extract_decisions":
            write_projection(base_dir, "extracted_decisions.json", {"project": project, "items": _event_summary(base_dir, project, {"decision_made"})})
            actions.append("decisions_extracted")
        if job.job_type == "extract_constraints":
            write_projection(base_dir, "extracted_constraints.json", {"project": project, "items": _event_summary(base_dir, project, {"constraint_added", "constraint_changed", "preference_declared"})})
            actions.append("constraints_extracted")
        if job.job_type == "extract_failures":
            write_projection(base_dir, "extracted_failures.json", {"project": project, "items": _event_summary(base_dir, project, {"failure_observed", "regression_detected"})})
            actions.append("failures_extracted")
        if job.job_type == "extract_metric_deltas":
            write_projection(base_dir, "metric_deltas.json", {"project": project, "items": _event_summary(base_dir, project, {"metric_changed", "benchmark_finished", "regression_detected"})})
            actions.append("metric_deltas_extracted")
        if job.job_type == "detect_contradictions":
            contradictions = _detect_contradictions(base_dir, project)
            write_projection(base_dir, "contradictions.json", {"project": project, "count": len(contradictions), "items": contradictions})
            actions.append("contradictions_detected")
        if job.job_type == "refresh_index":
            index_report = index_ledger_events(base_dir, job.event_ids, project=project)
            if index_report["status"] != "ok":
                raise RuntimeError(f"ledger_vector_index_failed:{index_report['error']}")
            index_path = base_dir / "data" / "indexes" / "index_freshness.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps({"project": project, "job_id": job.job_id, "event_ids": job.event_ids, "status": "fresh", "indexed_event_count": index_report["indexed_event_count"], "fallback_in_use": index_report["fallback_in_use"]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            append_event_payload(
                base_dir,
                event_type="index_refreshed",
                payload={"job_id": job.job_id, "event_ids": job.event_ids},
                source="memory_runtime_scheduler",
                actor="system",
                project=project,
            )
            actions.append("index_refresh_recorded")
        if job.job_type == "generate_recovery_summary":
            failures = _event_summary(base_dir, project, {"failure_observed", "regression_detected"})
            recoveries = _event_summary(base_dir, project, {"recovery_attempted", "patch_applied"})
            write_projection(base_dir, "recovery_summary.json", {"project": project, "failures": failures[-10:], "recoveries": recoveries[-10:]})
            actions.append("generate_recovery_summary_recorded")
        if job.job_type == "prepare_next_session_context":
            graph = build_and_save_graph(base_dir, project=project)
            write_projection(base_dir, "next_session_context.json", {"project": project, "tasks": [node.__dict__ for node in graph.nodes if node.node_type == "Task"][-5:]})
            actions.append("prepare_next_session_context_recorded")
        if job.job_type == "compact_low_value_events":
            events = replay_events(base_dir, project=project)
            write_projection(base_dir, "safe_compaction_plan.json", {"project": project, "raw_events_preserved": True, "low_value_event_count": sum(1 for event in events if event.event_type in {"retrieval_performed", "context_packet_compiled"})})
            actions.append(f"{job.job_type}_recorded")
        append_event_payload(
            base_dir,
            event_type="maintenance_job_finished",
            payload={"job_id": job.job_id, "job_type": job.job_type, "actions": actions},
            source="memory_runtime_scheduler",
            actor="system",
            project=project,
        )
        result = MaintenanceResult(job.job_id, "ok", actions)
        _append_job_log(base_dir, {"status": "finished", **job.to_dict(), "actions": actions})
        return result
    except Exception as exc:
        _append_job_log(base_dir, {"status": "error", **job.to_dict(), "actions": actions, "error": str(exc)})
        return MaintenanceResult(job.job_id, "error", actions, str(exc))
