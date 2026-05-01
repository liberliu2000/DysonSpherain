from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .project_state import load_project_state, render_project_state_markdown
from .security import redact_payload, redact_secrets
from .utils import stable_content_hash


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class ExecutionStep:
    step_id: str
    description: str
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    notes: str | None = None
    artifacts: list[str] = field(default_factory=list)


@dataclass
class ExecutionRun:
    run_id: str
    project: str
    task: str
    status: str
    started_at: str
    updated_at: str
    completed_at: str | None = None
    steps: list[ExecutionStep] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    benchmarks_run: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    next_action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ledger_dir(base_dir: Path) -> Path:
    return base_dir / "artifacts" / "execution_ledger"


def ledger_path(base_dir: Path, project: str) -> Path:
    slug = stable_content_hash(project)[:12]
    return ledger_dir(base_dir) / f"{project.lower()}_{slug}.jsonl"


def _new_run_id(project: str, task: str, timestamp: str) -> str:
    return "run_" + stable_content_hash(json.dumps([project, task, timestamp], ensure_ascii=False))[:16]


def _step_from_payload(payload: dict[str, Any]) -> ExecutionStep:
    return ExecutionStep(
        step_id=str(payload.get("step_id") or ""),
        description=str(payload.get("description") or ""),
        status=str(payload.get("status") or "pending"),
        started_at=payload.get("started_at"),
        completed_at=payload.get("completed_at"),
        notes=payload.get("notes"),
        artifacts=[str(item) for item in payload.get("artifacts", [])],
    )


def run_from_payload(payload: dict[str, Any]) -> ExecutionRun:
    steps = [_step_from_payload(item) for item in payload.get("steps", []) if isinstance(item, dict)]
    return ExecutionRun(
        run_id=str(payload.get("run_id") or ""),
        project=str(payload.get("project") or "default"),
        task=str(payload.get("task") or ""),
        status=str(payload.get("status") or "unknown"),
        started_at=str(payload.get("started_at") or ""),
        updated_at=str(payload.get("updated_at") or ""),
        completed_at=payload.get("completed_at"),
        steps=steps,
        changed_files=[str(item) for item in payload.get("changed_files", [])],
        artifacts=[str(item) for item in payload.get("artifacts", [])],
        tests_run=[str(item) for item in payload.get("tests_run", [])],
        benchmarks_run=[str(item) for item in payload.get("benchmarks_run", [])],
        errors=[str(item) for item in payload.get("errors", [])],
        next_action=payload.get("next_action"),
        metadata=dict(payload.get("metadata") or {}),
    )


def load_execution_runs(base_dir: Path, project: str) -> list[ExecutionRun]:
    path = ledger_path(base_dir, project)
    if not path.exists():
        return []
    runs: dict[str, ExecutionRun] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            run = run_from_payload(payload)
            if run.run_id:
                runs[run.run_id] = run
    return sorted(runs.values(), key=lambda item: item.updated_at or item.started_at)


def get_execution_run(base_dir: Path, project: str, run_id: str) -> ExecutionRun | None:
    for run in load_execution_runs(base_dir, project):
        if run.run_id == run_id:
            return run
    return None


def append_execution_run(base_dir: Path, run: ExecutionRun) -> Path:
    run.updated_at = _now()
    run.task = redact_secrets(run.task)
    run.changed_files = [redact_secrets(item) for item in run.changed_files]
    run.artifacts = [redact_secrets(item) for item in run.artifacts]
    run.tests_run = [redact_secrets(item) for item in run.tests_run]
    run.benchmarks_run = [redact_secrets(item) for item in run.benchmarks_run]
    run.errors = [redact_secrets(item) for item in run.errors]
    run.next_action = redact_secrets(run.next_action) if run.next_action else None
    run.metadata = redact_payload(run.metadata)
    path = ledger_path(base_dir, run.project)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(run), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def create_execution_run(
    base_dir: Path,
    *,
    project: str,
    task: str,
    status: str = "running",
    steps: list[ExecutionStep] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecutionRun:
    timestamp = _now()
    run = ExecutionRun(
        run_id=_new_run_id(project, task, timestamp),
        project=project,
        task=task,
        status=status,
        started_at=timestamp,
        updated_at=timestamp,
        completed_at=timestamp if status in TERMINAL_STATUSES else None,
        steps=list(steps or []),
        metadata=dict(metadata or {}),
    )
    append_execution_run(base_dir, run)
    return run


def update_execution_run(
    base_dir: Path,
    *,
    project: str,
    run_id: str,
    status: str | None = None,
    task: str | None = None,
    changed_files: list[str] | None = None,
    artifacts: list[str] | None = None,
    tests_run: list[str] | None = None,
    benchmarks_run: list[str] | None = None,
    errors: list[str] | None = None,
    next_action: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecutionRun:
    existing = get_execution_run(base_dir, project, run_id)
    if existing is None:
        timestamp = _now()
        existing = ExecutionRun(
            run_id=run_id,
            project=project,
            task=task or run_id,
            status="running",
            started_at=timestamp,
            updated_at=timestamp,
        )
    if task is not None:
        existing.task = task
    if status is not None:
        existing.status = status
        if status in TERMINAL_STATUSES:
            existing.completed_at = _now()
        elif existing.completed_at is not None:
            existing.completed_at = None
    for attr, values in (
        ("changed_files", changed_files),
        ("artifacts", artifacts),
        ("tests_run", tests_run),
        ("benchmarks_run", benchmarks_run),
        ("errors", errors),
    ):
        if values is None:
            continue
        current = list(getattr(existing, attr))
        for value in values:
            if value not in current:
                current.append(value)
        setattr(existing, attr, current)
    if next_action is not None:
        existing.next_action = next_action
    if metadata:
        existing.metadata.update(metadata)
    append_execution_run(base_dir, existing)
    return existing


def record_postrun(
    base_dir: Path,
    *,
    project: str,
    summary: str,
    source: str,
    run_id: str | None = None,
    task: str | None = None,
    status: str = "completed",
    artifacts: list[str] | None = None,
    tests_run: list[str] | None = None,
    benchmarks_run: list[str] | None = None,
    changed_files: list[str] | None = None,
    errors: list[str] | None = None,
    next_action: str | None = None,
) -> ExecutionRun:
    metadata = {"source": source, "summary": summary[:4000]}
    if run_id:
        return update_execution_run(
            base_dir,
            project=project,
            run_id=run_id,
            status=status,
            task=task,
            changed_files=changed_files,
            artifacts=artifacts,
            tests_run=tests_run,
            benchmarks_run=benchmarks_run,
            errors=errors,
            next_action=next_action,
            metadata=metadata,
        )
    inferred_task = task or (summary.splitlines()[0][:160] if summary.strip() else "Agent postrun")
    run = create_execution_run(
        base_dir,
        project=project,
        task=inferred_task,
        status=status,
        metadata=metadata,
    )
    run.artifacts = list(artifacts or [])
    run.tests_run = list(tests_run or [])
    run.benchmarks_run = list(benchmarks_run or [])
    run.changed_files = list(changed_files or [])
    run.errors = list(errors or [])
    run.next_action = next_action
    append_execution_run(base_dir, run)
    return run


def render_ledger_list(runs: list[ExecutionRun]) -> str:
    lines = ["# Execution Ledger", ""]
    if not runs:
        lines.append("No execution runs recorded.")
        return "\n".join(lines) + "\n"
    for run in reversed(runs):
        terminal = " terminal" if run.status in TERMINAL_STATUSES else ""
        lines.append(f"- `{run.run_id}` status=`{run.status}`{terminal} updated=`{run.updated_at}` task={run.task}")
    return "\n".join(lines) + "\n"


def render_resume_packet(base_dir: Path, project: str, run_id: str) -> str:
    run = get_execution_run(base_dir, project, run_id)
    if run is None:
        raise KeyError(run_id)
    state = load_project_state(base_dir, project)
    unfinished_steps = [step for step in run.steps if step.status not in TERMINAL_STATUSES]
    lines = [
        "# Agent Resume Packet",
        "",
        "## Run",
        "",
        f"- run_id: `{run.run_id}`",
        f"- project: `{run.project}`",
        f"- status: `{run.status}`",
        f"- task: {run.task}",
        f"- started_at: `{run.started_at}`",
        f"- updated_at: `{run.updated_at}`",
        f"- completed_at: `{run.completed_at}`",
        "",
        "## Safety Status",
        "",
    ]
    if run.status in TERMINAL_STATUSES:
        lines.append("- This run is terminal; do not treat it as an in-progress success unless status is `completed`.")
    else:
        lines.append("- This run is non-terminal; resume from partial artifacts and unresolved steps.")
    lines.extend(["", "## Next Action", "", run.next_action or "Continue from the latest valid unfinished step.", ""])
    lines.extend(["## Unfinished Steps", ""])
    if unfinished_steps:
        for step in unfinished_steps:
            lines.append(f"- `{step.step_id}` status=`{step.status}` {step.description}")
    else:
        lines.append("- None recorded.")
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- `{item}`" for item in run.artifacts) if run.artifacts else lines.append("- None recorded.")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {item}" for item in run.errors) if run.errors else lines.append("- None recorded.")
    lines.extend(["", "## Project State Snapshot", "", render_project_state_markdown(state)])
    return "\n".join(lines)
