from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dysonspherain.writeback.deduper import benchmark_results_hash, canonical_content, classify_duplicate, files_changed_hash
from dysonspherain.writeback.sanitizer import sanitize_payload
from sphere_cli.project_state import write_agent_run_summary


@dataclass
class WriteMemoryRequest:
    cwd: str
    session_id: str
    task_goal: str
    summary: str
    files_changed: list[str]
    commands_run: list[str]
    tests_run: list[str]
    benchmark_results: list[str]
    failures: list[str]
    next_actions: list[str]
    source: str = "manual"
    project: str = "DysonSpherain"


@dataclass
class WriteMemoryResult:
    status: str
    memory_id: str | None
    dedupe: dict[str, Any]
    sanitizer: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_memory(request: WriteMemoryRequest) -> WriteMemoryResult:
    base_dir = Path(request.cwd or ".").resolve()
    raw_payload = asdict(request)
    sanitized = sanitize_payload(raw_payload)
    content = canonical_content(sanitized.payload)
    dedupe = classify_duplicate(base_dir, request.project, content)
    if dedupe.is_duplicate:
        return WriteMemoryResult(status="duplicate", memory_id=dedupe.duplicate_of, dedupe=dedupe.to_dict(), sanitizer=sanitized.to_dict())

    metadata = {
        "session_id": sanitized.payload.get("session_id"),
        "task_goal": sanitized.payload.get("task_goal"),
        "files_changed": sanitized.payload.get("files_changed") or [],
        "commands_run": sanitized.payload.get("commands_run") or [],
        "tests_run": sanitized.payload.get("tests_run") or [],
        "benchmark_results": sanitized.payload.get("benchmark_results") or [],
        "failures": sanitized.payload.get("failures") or [],
        "next_actions": sanitized.payload.get("next_actions") or [],
        "benchmark_results_hash": benchmark_results_hash(sanitized.payload),
        "files_changed_hash": files_changed_hash(sanitized.payload),
        **dedupe.to_dict(),
    }
    record = write_agent_run_summary(
        base_dir,
        request.project,
        json.dumps(sanitized.payload, ensure_ascii=False, sort_keys=True),
        str(sanitized.payload.get("source") or request.source),
        metadata=metadata,
    )
    return WriteMemoryResult(status="ok", memory_id=str(record.get("memory_id")), dedupe=dedupe.to_dict(), sanitizer=sanitized.to_dict())
