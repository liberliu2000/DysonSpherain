from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dysonspherain.memory_runtime.adapters import ClaudeCodeAdapter
from dysonspherain.memory_runtime.events import build_event
from dysonspherain.memory_runtime.ledger import append_event
from dysonspherain.memory_runtime.scheduler import enqueue_maintenance_jobs


def append_hook_event(cwd: str, hook_event: str, payload: dict[str, Any], *, event_type: str | None = None) -> dict[str, Any]:
    root = Path(cwd).resolve()
    project = str(payload.get("project") or "DysonSpherain")
    session_id = str(payload.get("session_id") or "") or None
    adapter = ClaudeCodeAdapter()
    if hook_event == "UserPromptSubmit":
        events = adapter.capture_input({**payload, "project": project, "session_id": session_id})
    elif hook_event == "PostToolUse":
        events = adapter.capture_tool_use({**payload, "project": project, "session_id": session_id, "hook_event": hook_event})
    else:
        summary = str(payload.get("summary") or payload.get("reason") or payload.get("stop_reason") or payload.get("compact_summary") or hook_event)
        events = [
            build_event(
                event_type=event_type or "agent_action_observed",
                payload={
                    "title": hook_event,
                    "summary": summary[:2000],
                    "hook_event": hook_event,
                    "raw_payload_preview": json.dumps(payload, ensure_ascii=False, sort_keys=True)[:4000],
                },
                source="claude_code",
                actor="agent",
                project=project,
                session_id=session_id,
                provenance={"adapter": "claude_code", "hook_event": hook_event},
            )
        ]
    results = [append_event(root, event).to_dict() for event in events]
    event_ids = [str(result.get("event_id")) for result in results if not result.get("duplicate")]
    queued = [job.to_dict() for job in enqueue_maintenance_jobs(root, "index_staleness_detected", event_ids, project=project)] if event_ids else []
    return {"ledger_status": "ok", "ledger_results": results, "queued_index_jobs": queued}
