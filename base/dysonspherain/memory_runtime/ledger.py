from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from .events import MemoryEvent, build_event


@dataclass(frozen=True)
class AppendResult:
    status: str
    event_id: str
    content_hash: str
    path: str
    duplicate: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def ledger_dir(base_dir: Path) -> Path:
    return base_dir / "data" / "ledger"


def projections_dir(base_dir: Path) -> Path:
    return base_dir / "data" / "projections"


def ledger_path(base_dir: Path, timestamp: str) -> Path:
    month = timestamp[:7].replace("-", "")
    return ledger_dir(base_dir) / f"events_{month}.jsonl"


def _event_from_dict(row: dict[str, object]) -> MemoryEvent:
    return MemoryEvent(
        event_id=str(row.get("event_id") or ""),
        event_type=str(row.get("event_type") or ""),
        timestamp=str(row.get("timestamp") or ""),
        source=str(row.get("source") or ""),
        actor=str(row.get("actor") or ""),
        project=str(row.get("project") or ""),
        session_id=str(row.get("session_id") or "") or None,
        parent_event_id=str(row.get("parent_event_id") or "") or None,
        payload=dict(row.get("payload") or {}),
        content_hash=str(row.get("content_hash") or ""),
        provenance=dict(row.get("provenance") or {}),
    )


def replay_events(base_dir: Path, *, project: str | None = None, from_path: Path | None = None) -> list[MemoryEvent]:
    paths = [from_path] if from_path else sorted(ledger_dir(base_dir).glob("events_*.jsonl"))
    events: list[MemoryEvent] = []
    for path in paths:
        if path is None or not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            event = _event_from_dict(row)
            if project is None or event.project == project:
                events.append(event)
    return sorted(events, key=lambda event: (event.timestamp, event.event_id))


def content_hashes(base_dir: Path) -> set[str]:
    return {event.content_hash for event in replay_events(base_dir)}


def append_event(base_dir: Path, event: MemoryEvent) -> AppendResult:
    path = ledger_path(base_dir, event.timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    if event.content_hash in content_hashes(base_dir):
        return AppendResult("duplicate", event.event_id, event.content_hash, str(path), duplicate=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return AppendResult("ok", event.event_id, event.content_hash, str(path), duplicate=False)


def append_event_payload(
    base_dir: Path,
    *,
    event_type: str,
    payload: dict[str, object],
    source: str = "manual",
    actor: str = "user",
    project: str = "DysonSpherain",
    session_id: str | None = None,
    parent_event_id: str | None = None,
    timestamp: str | None = None,
    provenance: dict[str, object] | None = None,
) -> AppendResult:
    event = build_event(
        event_type=event_type,
        payload=dict(payload),
        source=source,
        actor=actor,
        project=project,
        session_id=session_id,
        parent_event_id=parent_event_id,
        timestamp=timestamp,
        provenance=provenance,
    )
    return append_event(base_dir, event)


def write_projection(base_dir: Path, name: str, payload: dict[str, object]) -> Path:
    path = projections_dir(base_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def replay_from_iterable(events: Iterable[MemoryEvent]) -> list[MemoryEvent]:
    return sorted(list(events), key=lambda event: (event.timestamp, event.event_id))

