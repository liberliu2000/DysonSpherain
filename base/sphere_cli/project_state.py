from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment_registry import BenchmarkRun, latest_run, load_registry
from .security import redact_payload, redact_secrets
from .utils import stable_content_hash


@dataclass
class ProjectState:
    project: str
    current_goal: str | None = None
    current_phase: str | None = None
    latest_benchmark_status: dict[str, Any] = field(default_factory=dict)
    known_blockers: list[str] = field(default_factory=list)
    active_tasks: list[str] = field(default_factory=list)
    recent_decisions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    do_not_do: list[str] = field(default_factory=list)
    relevant_artifacts: list[str] = field(default_factory=list)
    updated_at: str = ""
    source_memory_ids: list[str] = field(default_factory=list)


DEFAULT_CONSTRAINTS = [
    "Do not trade Recall/NDCG/candidate_recall for speed.",
    "Do not globally lower top-k or final candidate pool size to create speedups.",
    "Do not remove dense preserve, safe fusion, candidate recall audit, or oracle report.",
    "Official benchmark validation must not use local_hash fallback.",
]

DEFAULT_DO_NOT_DO = [
    "Do not hardcode benchmark gold ids, questions, answers, or metrics.",
    "Do not use benchmark_name as a cheat path to recall gold candidates.",
    "Do not disable KnowMe or CloneMem necessary candidate generation globally.",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_dir(base_dir: Path) -> Path:
    return base_dir / "artifacts" / "project_state"


def state_path(base_dir: Path, project: str) -> Path:
    slug = stable_content_hash(project)[:12]
    return state_dir(base_dir) / f"{project.lower()}_{slug}.json"


def memory_log_path(base_dir: Path, project: str) -> Path:
    slug = stable_content_hash(project)[:12]
    return state_dir(base_dir) / f"{project.lower()}_{slug}_memories.jsonl"


def _load_memory_events(base_dir: Path, project: str) -> list[dict[str, Any]]:
    path = memory_log_path(base_dir, project)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def list_memories(
    base_dir: Path,
    project: str,
    *,
    include_archived: bool = False,
    memory_type: str | None = None,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for event in _load_memory_events(base_dir, project):
        event_type = str(event.get("event") or "create")
        memory_id = str(event.get("memory_id") or "")
        if not memory_id:
            continue
        if event_type == "update":
            record = records.setdefault(memory_id, {"memory_id": memory_id, "project": project, "status": "current"})
            patch = event.get("patch") if isinstance(event.get("patch"), dict) else {}
            record.update(patch)
            record["updated_at"] = event.get("updated_at") or event.get("created_at") or _now()
            continue
        if event_type == "archive":
            record = records.setdefault(memory_id, {"memory_id": memory_id, "project": project})
            record["status"] = "archived"
            record["updated_at"] = event.get("updated_at") or event.get("created_at") or _now()
            continue
        record = dict(event)
        record.setdefault("status", "current")
        record.setdefault("title", str(record.get("content") or "")[:80])
        records[memory_id] = record
    result = list(records.values())
    if not include_archived:
        result = [record for record in result if record.get("status") != "archived"]
    if memory_type:
        result = [record for record in result if record.get("memory_type") == memory_type]
    result.sort(key=lambda record: str(record.get("updated_at") or record.get("created_at") or ""), reverse=True)
    return result


def get_memory(base_dir: Path, project: str, memory_id: str) -> dict[str, Any] | None:
    for record in list_memories(base_dir, project, include_archived=True):
        if record.get("memory_id") == memory_id:
            return record
    return None


def search_memories(
    base_dir: Path,
    project: str,
    query: str,
    *,
    include_archived: bool = False,
    memory_type: str | None = None,
) -> list[dict[str, Any]]:
    terms = [term.lower() for term in query.split() if term.strip()]
    records = list_memories(base_dir, project, include_archived=include_archived, memory_type=memory_type)
    if not terms:
        return records
    matches: list[dict[str, Any]] = []
    for record in records:
        haystack = " ".join(
            str(record.get(key) or "")
            for key in ("memory_id", "memory_type", "title", "summary", "content", "source", "status")
        ).lower()
        metadata = record.get("metadata")
        if metadata:
            haystack += " " + json.dumps(metadata, ensure_ascii=False).lower()
        if all(term in haystack for term in terms):
            matches.append(record)
    return matches


def update_memory(base_dir: Path, project: str, memory_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    if get_memory(base_dir, project, memory_id) is None:
        raise KeyError(memory_id)
    safe_patch = redact_payload(dict(patch))
    for key in ("content", "source", "summary", "title"):
        if key in safe_patch and safe_patch[key] is not None:
            safe_patch[key] = redact_secrets(str(safe_patch[key]))
    row = {
        "event": "update",
        "memory_id": memory_id,
        "project": project,
        "patch": safe_patch,
        "updated_at": _now(),
    }
    path = memory_log_path(base_dir, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return get_memory(base_dir, project, memory_id) or row


def archive_memory(base_dir: Path, project: str, memory_id: str) -> dict[str, Any]:
    if get_memory(base_dir, project, memory_id) is None:
        raise KeyError(memory_id)
    row = {
        "event": "archive",
        "memory_id": memory_id,
        "project": project,
        "updated_at": _now(),
    }
    path = memory_log_path(base_dir, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return get_memory(base_dir, project, memory_id) or row


def load_project_state(base_dir: Path, project: str) -> ProjectState:
    path = state_path(base_dir, project)
    if not path.exists():
        return ProjectState(
            project=project,
            current_goal=None,
            current_phase=None,
            constraints=list(DEFAULT_CONSTRAINTS),
            do_not_do=list(DEFAULT_DO_NOT_DO),
            updated_at=_now(),
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ProjectState(**payload)


def save_project_state(base_dir: Path, state: ProjectState) -> Path:
    state.updated_at = _now()
    path = state_path(base_dir, state.project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run_status(run: BenchmarkRun | None) -> dict[str, Any]:
    if run is None:
        return {"status": "missing"}
    return {
        "run_id": run.run_id,
        "run_type": run.run_type,
        "artifact_dir": run.artifact_dir,
        "question_count": run.question_count,
        "sample_count": run.sample_count,
        "elapsed_seconds": run.elapsed_seconds,
        "fallback_in_use": run.fallback_in_use,
        "metrics": dict(run.metrics),
        "comparability_warnings": list(run.comparability_warnings),
    }


def update_project_state_from_registry(base_dir: Path, project: str) -> ProjectState:
    state = load_project_state(base_dir, project)
    runs = load_registry(base_dir)
    benchmark_status = {
        dataset: _run_status(latest_run(runs, project=project, dataset=dataset))
        for dataset in ("longmemeval", "locomo", "knowme", "clonemem", "convomem")
    }
    state.latest_benchmark_status = benchmark_status
    state.current_phase = state.current_phase or "Phase 5/6"
    state.current_goal = state.current_goal or "Quality-preserving DysonSpherain retrieval and benchmark efficiency improvement"
    for artifact in (
        "reports/phase4_diagnostic_consolidation.md",
        "reports/phase5_clonemem_parent_anchor_local_window_report.md",
        "CHANGELOG.md",
    ):
        if artifact not in state.relevant_artifacts:
            state.relevant_artifacts.append(artifact)
    save_project_state(base_dir, state)
    return state


def render_project_state_markdown(state: ProjectState) -> str:
    lines = [
        f"# Project State: {state.project}",
        "",
        f"- updated_at: `{state.updated_at}`",
        f"- current_phase: `{state.current_phase}`",
        f"- current_goal: `{state.current_goal}`",
        "",
        "## Latest Benchmark Status",
        "",
    ]
    for benchmark, payload in sorted(state.latest_benchmark_status.items()):
        lines.append(f"- `{benchmark}`: `{payload.get('run_type', payload.get('status'))}` q=`{payload.get('question_count')}` fallback=`{payload.get('fallback_in_use')}`")
    lines.extend(["", "## Constraints", ""])
    lines.extend(f"- {item}" for item in state.constraints)
    lines.extend(["", "## Do Not Do", ""])
    lines.extend(f"- {item}" for item in state.do_not_do)
    lines.extend(["", "## Active Tasks", ""])
    lines.extend(f"- {item}" for item in state.active_tasks)
    lines.extend(["", "## Recent Decisions", ""])
    lines.extend(f"- {item}" for item in state.recent_decisions)
    lines.extend(["", "## Relevant Artifacts", ""])
    lines.extend(f"- `{item}`" for item in state.relevant_artifacts)
    return "\n".join(lines) + "\n"


def write_memory(
    base_dir: Path,
    *,
    memory_type: str,
    project: str,
    content: str,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_content = redact_secrets(content)
    safe_source = redact_secrets(source)
    safe_metadata = redact_payload(dict(metadata or {}))
    payload = {
        "memory_id": stable_content_hash(json.dumps([memory_type, project, safe_content, safe_source, safe_metadata, _now()], ensure_ascii=False))[:16],
        "memory_type": memory_type,
        "project": project,
        "title": str((metadata or {}).get("title") or safe_content[:80]),
        "content": safe_content,
        "source": safe_source,
        "status": str((metadata or {}).get("status") or "current"),
        "metadata": safe_metadata,
        "created_at": _now(),
    }
    path = memory_log_path(base_dir, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    state = load_project_state(base_dir, project)
    if payload["memory_id"] not in state.source_memory_ids:
        state.source_memory_ids.append(payload["memory_id"])
    if memory_type == "decision":
        state.recent_decisions.append(content)
    elif memory_type == "constraint":
        if content not in state.constraints:
            state.constraints.append(content)
    elif memory_type in {"task", "experiment"}:
        state.active_tasks.append(content)
    save_project_state(base_dir, state)
    return payload


def write_fact(base_dir: Path, project: str, content: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return write_memory(base_dir, memory_type="fact", project=project, content=content, source=source, metadata=metadata)


def write_decision(base_dir: Path, project: str, content: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return write_memory(base_dir, memory_type="decision", project=project, content=content, source=source, metadata=metadata)


def write_experiment(base_dir: Path, project: str, content: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return write_memory(base_dir, memory_type="experiment", project=project, content=content, source=source, metadata=metadata)


def write_failure(base_dir: Path, project: str, content: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return write_memory(base_dir, memory_type="failure", project=project, content=content, source=source, metadata=metadata)


def write_task(base_dir: Path, project: str, content: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return write_memory(base_dir, memory_type="task", project=project, content=content, source=source, metadata=metadata)


def write_constraint(base_dir: Path, project: str, content: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return write_memory(base_dir, memory_type="constraint", project=project, content=content, source=source, metadata=metadata)


def write_conversation_summary(base_dir: Path, project: str, content: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return write_memory(base_dir, memory_type="conversation_summary", project=project, content=content, source=source, metadata=metadata)


def write_agent_run_summary(base_dir: Path, project: str, content: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return write_memory(base_dir, memory_type="agent_run_summary", project=project, content=content, source=source, metadata=metadata)
