from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any


EVENT_TYPES = {
    "user_instruction_received",
    "assistant_response_generated",
    "agent_action_observed",
    "tool_call_observed",
    "file_changed",
    "benchmark_started",
    "benchmark_finished",
    "metric_changed",
    "regression_detected",
    "hypothesis_created",
    "decision_made",
    "patch_applied",
    "artifact_created",
    "artifact_updated",
    "constraint_added",
    "constraint_changed",
    "preference_declared",
    "failure_observed",
    "recovery_attempted",
    "retrieval_performed",
    "context_packet_compiled",
    "memory_compacted",
    "index_refreshed",
    "maintenance_job_scheduled",
    "maintenance_job_finished",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    event_type: str
    timestamp: str
    source: str
    actor: str
    project: str
    session_id: str | None
    parent_event_id: str | None
    payload: dict[str, Any]
    content_hash: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_event(
    *,
    event_type: str,
    payload: dict[str, Any],
    source: str = "manual",
    actor: str = "user",
    project: str = "DysonSpherain",
    session_id: str | None = None,
    parent_event_id: str | None = None,
    timestamp: str | None = None,
    provenance: dict[str, Any] | None = None,
    event_id: str | None = None,
) -> MemoryEvent:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown_event_type:{event_type}")
    ts = timestamp or now_iso()
    normalized = {
        "event_type": event_type,
        "timestamp": ts,
        "source": source,
        "actor": actor,
        "project": project,
        "session_id": session_id or "",
        "parent_event_id": parent_event_id or "",
        "payload": payload,
    }
    content_hash = stable_hash(normalized)
    return MemoryEvent(
        event_id=event_id or f"evt_{content_hash[:20]}",
        event_type=event_type,
        timestamp=ts,
        source=source,
        actor=actor,
        project=project,
        session_id=session_id,
        parent_event_id=parent_event_id,
        payload=payload,
        content_hash=content_hash,
        provenance=provenance or {},
    )
