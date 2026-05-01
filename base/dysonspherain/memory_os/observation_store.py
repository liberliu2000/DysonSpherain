from __future__ import annotations

import fnmatch
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dysonspherain.utils.token_counter import TokenCounter
from sphere_cli.project_state import archive_memory, get_memory, list_memories, search_memories
from sphere_cli.security import redact_payload, redact_secrets
from sphere_cli.utils import stable_content_hash


PRIVATE_RE = re.compile(r"(?is)<private>.*?</private>")


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    project: str
    kind: str
    title: str
    content: str
    source: str
    session_id: str
    created_at: str
    updated_at: str
    token_count: int
    citation: str
    metadata: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def observation_db_path(base_dir: Path) -> Path:
    return base_dir / "artifacts" / "memory_os" / "observations.sqlite3"


def _connect(base_dir: Path) -> sqlite3.Connection:
    path = observation_db_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
          observation_id TEXT PRIMARY KEY,
          project TEXT NOT NULL,
          kind TEXT NOT NULL,
          title TEXT NOT NULL,
          content TEXT NOT NULL,
          source TEXT NOT NULL,
          session_id TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          token_count INTEGER NOT NULL DEFAULT 0,
          citation TEXT NOT NULL,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          archived INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_project_time ON observations(project, updated_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(project, session_id)")
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts
        USING fts5(observation_id UNINDEXED, project UNINDEXED, title, content, source, metadata)
        """
    )
    conn.commit()


def _load_ignore_patterns(base_dir: Path) -> list[str]:
    path = base_dir / ".dysonignore"
    if not path.exists():
        return []
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            patterns.append(item)
    return patterns


def _ignored_by_policy(base_dir: Path, values: list[str]) -> bool:
    patterns = _load_ignore_patterns(base_dir)
    if not patterns:
        return False
    for value in values:
        text = str(value or "")
        for pattern in patterns:
            if fnmatch.fnmatch(text, pattern) or pattern in text:
                return True
    return False


def sanitize_observation_payload(base_dir: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if "<private>" in raw.lower():
        payload = json.loads(PRIVATE_RE.sub("[redacted-private]", raw))
    title = redact_secrets(str(payload.get("title") or ""))
    content = redact_secrets(str(payload.get("content") or payload.get("summary") or ""))
    source = redact_secrets(str(payload.get("source") or "manual"))
    metadata = redact_payload(dict(payload.get("metadata") or {}))
    files = [str(item) for item in metadata.get("files_changed") or payload.get("files_changed") or []]
    if _ignored_by_policy(base_dir, [title, content, source, *files]):
        return None
    return {**payload, "title": title, "content": content, "source": source, "metadata": metadata}


def _stable_observation_id(project: str, kind: str, title: str, content: str, source: str, session_id: str) -> str:
    digest = stable_content_hash(json.dumps([project, kind, title, content, source, session_id], ensure_ascii=False, sort_keys=True))[:16]
    return f"obs_{digest}"


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    metadata = json.loads(row["metadata_json"] or "{}")
    return ObservationRecord(
        observation_id=str(row["observation_id"]),
        project=str(row["project"]),
        kind=str(row["kind"]),
        title=str(row["title"]),
        content=str(row["content"]),
        source=str(row["source"]),
        session_id=str(row["session_id"] or ""),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        token_count=int(row["token_count"] or 0),
        citation=str(row["citation"]),
        metadata=metadata if isinstance(metadata, dict) else {},
    ).__dict__


def write_observation(
    base_dir: Path,
    *,
    project: str,
    kind: str,
    title: str,
    content: str,
    source: str,
    session_id: str = "",
    metadata: dict[str, Any] | None = None,
    observation_id: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    payload = sanitize_observation_payload(
        base_dir,
        {
            "project": project,
            "kind": kind,
            "title": title,
            "content": content,
            "source": source,
            "session_id": session_id,
            "metadata": metadata or {},
        },
    )
    if payload is None:
        return {"status": "skipped", "reason": "dysonignore"}
    project = str(payload.get("project") or project)
    kind = str(payload.get("kind") or kind)
    title = str(payload.get("title") or content[:80])
    content = str(payload.get("content") or "")
    source = str(payload.get("source") or source)
    session_id = str(payload.get("session_id") or "")
    metadata = dict(payload.get("metadata") or {})
    observation_id = observation_id or _stable_observation_id(project, kind, title, content, source, session_id)
    now = updated_at or created_at or _now()
    created = created_at or now
    citation = f"dyson://observation/{observation_id}"
    token_count = TokenCounter().count("\n".join([title, content])).tokens
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    with _connect(base_dir) as conn:
        conn.execute(
            """
            INSERT INTO observations (
              observation_id, project, kind, title, content, source, session_id,
              created_at, updated_at, token_count, citation, metadata_json, archived
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(observation_id) DO UPDATE SET
              title=excluded.title,
              content=excluded.content,
              source=excluded.source,
              session_id=excluded.session_id,
              updated_at=excluded.updated_at,
              token_count=excluded.token_count,
              citation=excluded.citation,
              metadata_json=excluded.metadata_json,
              archived=0
            """,
            (observation_id, project, kind, title, content, source, session_id, created, now, token_count, citation, metadata_json),
        )
        conn.execute("DELETE FROM observations_fts WHERE observation_id = ?", (observation_id,))
        conn.execute(
            "INSERT INTO observations_fts(observation_id, project, title, content, source, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (observation_id, project, title, content, source, metadata_json),
        )
        conn.commit()
    return {"status": "ok", **get_observation(base_dir, project, observation_id)}


def write_token_economy_event(
    base_dir: Path,
    *,
    project: str,
    session_id: str,
    prompt: str,
    decision: str,
    injected_tokens: int,
    baseline_context_tokens: int,
    estimated_saved_tokens: int,
    budget_usage_ratio: float,
    source: str = "claude_code_user_prompt_submit",
    metadata: dict[str, Any] | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    baseline = max(0, int(baseline_context_tokens or 0))
    saved = max(0, int(estimated_saved_tokens or 0))
    saving_ratio = (saved / baseline) if baseline else 0.0
    payload = {
        "prompt_preview": str(prompt or "")[:240],
        "decision": decision,
        "injected_tokens": int(injected_tokens or 0),
        "baseline_context_tokens": baseline,
        "estimated_saved_tokens": saved,
        "saving_ratio": saving_ratio,
        "budget_usage_ratio": float(budget_usage_ratio or 0.0),
        **dict(metadata or {}),
    }
    return write_observation(
        base_dir,
        project=project,
        kind="token_economy_event",
        title=f"Token economy: {decision}",
        content=(
            f"decision={decision}; injected_tokens={int(injected_tokens or 0)}; "
            f"baseline_context_tokens={baseline}; estimated_saved_tokens={saved}; "
            f"saving_ratio={saving_ratio:.4f}"
        ),
        source=source,
        session_id=session_id,
        metadata=payload,
        created_at=created_at,
        updated_at=updated_at,
    )


def sync_project_memories(base_dir: Path, project: str) -> int:
    count = 0
    for record in list_memories(base_dir, project, include_archived=True):
        memory_id = str(record.get("memory_id") or "")
        if not memory_id:
            continue
        status = str(record.get("status") or "current")
        write_observation(
            base_dir,
            project=project,
            kind=str(record.get("memory_type") or "memory"),
            title=str(record.get("title") or memory_id),
            content=str(record.get("summary") or record.get("content") or ""),
            source=str(record.get("source") or "project_memory"),
            session_id=str((record.get("metadata") or {}).get("session_id") or ""),
            metadata={**dict(record.get("metadata") or {}), "memory_id": memory_id, "project_memory_status": status},
            observation_id=f"obs_mem_{memory_id}",
        )
        if status == "archived":
            delete_observation(base_dir, project, f"obs_mem_{memory_id}", hard=False)
        count += 1
    return count


def search_observations(
    base_dir: Path,
    *,
    project: str,
    query: str = "",
    limit: int = 10,
    include_archived: bool = False,
    kind: str | None = None,
) -> dict[str, Any]:
    sync_project_memories(base_dir, project)
    limit = max(1, min(int(limit or 10), 100))
    where = ["o.project = ?", "o.archived = ?" if not include_archived else "1 = 1"]
    params: list[Any] = [project]
    if not include_archived:
        params.append(0)
    if kind:
        where.append("o.kind = ?")
        params.append(kind)
    rows: list[sqlite3.Row]
    with _connect(base_dir) as conn:
        if query.strip():
            fts_query = " ".join(term.replace('"', "") for term in query.split() if term.strip()) or query
            sql = f"""
                SELECT o.*, bm25(observations_fts) AS rank
                FROM observations_fts
                JOIN observations o USING(observation_id)
                WHERE observations_fts MATCH ? AND {' AND '.join(where)}
                ORDER BY rank ASC, o.updated_at DESC
                LIMIT ?
            """
            try:
                rows = conn.execute(sql, [fts_query, *params, limit]).fetchall()
            except sqlite3.OperationalError:
                memories = search_memories(base_dir, project, query, include_archived=include_archived, memory_type=kind)
                rows = []
                ids = [f"obs_mem_{item.get('memory_id')}" for item in memories[:limit]]
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    rows = conn.execute(f"SELECT * FROM observations WHERE observation_id IN ({placeholders})", ids).fetchall()
        else:
            sql = f"SELECT * FROM observations o WHERE {' AND '.join(where)} ORDER BY o.updated_at DESC LIMIT ?"
            rows = conn.execute(sql, [*params, limit]).fetchall()
    items = []
    for idx, row in enumerate(rows):
        record = _row_to_record(row)
        items.append(
            {
                "observation_id": record["observation_id"],
                "title": record["title"],
                "kind": record["kind"],
                "source": record["source"],
                "updated_at": record["updated_at"],
                "token_cost": record["token_count"],
                "citation": record["citation"],
                "score": 1.0 / (idx + 1),
                "snippet": record["content"][:240],
                "metadata": record["metadata"],
            }
        )
    return {"status": "ok", "project": project, "query": query, "count": len(items), "observations": items}


def get_observation(base_dir: Path, project: str, observation_id: str) -> dict[str, Any]:
    with _connect(base_dir) as conn:
        row = conn.execute("SELECT * FROM observations WHERE project = ? AND observation_id = ?", (project, observation_id)).fetchone()
    if row is None and observation_id.startswith("obs_mem_"):
        memory_id = observation_id.removeprefix("obs_mem_")
        memory = get_memory(base_dir, project, memory_id)
        if memory:
            sync_project_memories(base_dir, project)
            return get_observation(base_dir, project, observation_id)
    if row is None:
        raise KeyError(observation_id)
    return _row_to_record(row)


def get_observations(base_dir: Path, *, project: str, observation_ids: list[str]) -> dict[str, Any]:
    items = []
    for observation_id in observation_ids:
        try:
            items.append(get_observation(base_dir, project, observation_id))
        except KeyError:
            items.append({"observation_id": observation_id, "status": "missing"})
    return {"status": "ok", "project": project, "observations": items}


def timeline(base_dir: Path, *, project: str, observation_id: str | None = None, session_id: str | None = None, limit: int = 20) -> dict[str, Any]:
    sync_project_memories(base_dir, project)
    limit = max(1, min(int(limit or 20), 100))
    params: list[Any] = [project]
    where = ["project = ?", "archived = 0"]
    anchor: dict[str, Any] | None = None
    if observation_id:
        anchor = get_observation(base_dir, project, observation_id)
        session_id = session_id or str(anchor.get("session_id") or "")
    if session_id:
        where.append("session_id = ?")
        params.append(session_id)
    with _connect(base_dir) as conn:
        rows = conn.execute(f"SELECT * FROM observations WHERE {' AND '.join(where)} ORDER BY updated_at DESC LIMIT ?", [*params, limit]).fetchall()
    return {
        "status": "ok",
        "project": project,
        "anchor": anchor,
        "events": [
            {
                "observation_id": item["observation_id"],
                "kind": item["kind"],
                "title": item["title"],
                "updated_at": item["updated_at"],
                "citation": item["citation"],
                "token_cost": item["token_count"],
            }
            for item in (_row_to_record(row) for row in rows)
        ],
    }


def export_observations(base_dir: Path, project: str, output: Path) -> Path:
    sync_project_memories(base_dir, project)
    data = search_observations(base_dir, project=project, limit=100, include_archived=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def delete_observation(base_dir: Path, project: str, observation_id: str, *, hard: bool = False) -> dict[str, Any]:
    with _connect(base_dir) as conn:
        if hard:
            conn.execute("DELETE FROM observations WHERE project = ? AND observation_id = ?", (project, observation_id))
            conn.execute("DELETE FROM observations_fts WHERE observation_id = ?", (observation_id,))
        else:
            conn.execute("UPDATE observations SET archived = 1, updated_at = ? WHERE project = ? AND observation_id = ?", (_now(), project, observation_id))
        conn.commit()
    if observation_id.startswith("obs_mem_"):
        memory_id = observation_id.removeprefix("obs_mem_")
        try:
            archive_memory(base_dir, project, memory_id)
        except KeyError:
            pass
    return {"status": "deleted", "observation_id": observation_id, "hard": hard}


def apply_retention(base_dir: Path, project: str, *, keep_last: int = 200) -> dict[str, Any]:
    sync_project_memories(base_dir, project)
    keep_last = max(1, int(keep_last or 200))
    with _connect(base_dir) as conn:
        rows = conn.execute(
            "SELECT observation_id FROM observations WHERE project = ? AND archived = 0 ORDER BY updated_at DESC LIMIT -1 OFFSET ?",
            (project, keep_last),
        ).fetchall()
        ids = [str(row["observation_id"]) for row in rows]
        for observation_id in ids:
            conn.execute("UPDATE observations SET archived = 1, updated_at = ? WHERE project = ? AND observation_id = ?", (_now(), project, observation_id))
        conn.commit()
    return {"status": "ok", "archived_count": len(ids), "keep_last": keep_last}


def token_economy_summary(base_dir: Path, *, project: str, limit: int = 100) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    with _connect(base_dir) as conn:
        rows = conn.execute(
            """
            SELECT * FROM observations
            WHERE project = ? AND kind = 'token_economy_event' AND archived = 0
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (project, max(1, min(int(limit or 100), 500))),
        ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        record = _row_to_record(row)
        meta = dict(record.get("metadata") or {})
        baseline = int(meta.get("baseline_context_tokens") or 0)
        saved = int(meta.get("estimated_saved_tokens") or 0)
        injected = int(meta.get("injected_tokens") or 0)
        ratio = (saved / baseline) if baseline else 0.0
        events.append(
            {
                "observation_id": record["observation_id"],
                "session_id": record["session_id"],
                "updated_at": record["updated_at"],
                "decision": meta.get("decision") or record["title"],
                "prompt_preview": meta.get("prompt_preview") or "",
                "baseline_context_tokens": baseline,
                "injected_tokens": injected,
                "estimated_saved_tokens": saved,
                "saving_ratio": ratio,
                "budget_usage_ratio": float(meta.get("budget_usage_ratio") or 0.0),
                "citation": record["citation"],
            }
        )

    def window(days: int) -> dict[str, Any]:
        cutoff = now - timedelta(days=days)
        selected = [event for event in events if _parse_time(event["updated_at"]) >= cutoff]
        baseline_total = sum(int(event["baseline_context_tokens"]) for event in selected)
        saved_total = sum(int(event["estimated_saved_tokens"]) for event in selected)
        injected_total = sum(int(event["injected_tokens"]) for event in selected)
        return {
            "days": days,
            "event_count": len(selected),
            "baseline_context_tokens": baseline_total,
            "injected_tokens": injected_total,
            "estimated_saved_tokens": saved_total,
            "saving_ratio": (saved_total / baseline_total) if baseline_total else 0.0,
        }

    return {
        "status": "ok",
        "project": project,
        "windows": {
            "24h": window(1),
            "7d": window(7),
            "30d": window(30),
        },
        "events": events,
    }


def _unique_values(values: list[Any], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            iterable = value
        else:
            iterable = [value]
        for item in iterable:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
            if len(result) >= limit:
                return result
    return result


def render_resume_markdown(payload: dict[str, Any]) -> str:
    if payload.get("status") != "ok":
        return "No previous DysonSpherain session context is available.\n"
    lines = [
        "# DysonSpherain Resume Context",
        "",
        f"- session_id: `{payload.get('session_id') or ''}`",
        f"- last_updated_at: `{payload.get('last_updated_at') or ''}`",
        f"- event_count: `{payload.get('event_count') or 0}`",
    ]
    token = payload.get("token_economy") or {}
    if token:
        lines.extend(
            [
                f"- estimated_saved_tokens: `{token.get('estimated_saved_tokens', 0)}`",
                f"- token_saving_ratio: `{float(token.get('saving_ratio') or 0.0):.4f}`",
            ]
        )
    for section, title in (
        ("current_goal", "Current Goal"),
        ("last_summary", "Last Summary"),
        ("next_actions", "Next Actions"),
        ("failures", "Failures / Blockers"),
        ("files_changed", "Relevant Files"),
        ("tests_run", "Tests Run"),
        ("commands_run", "Commands Run"),
    ):
        value = payload.get(section)
        if not value:
            continue
        lines.extend(["", f"## {title}", ""])
        if isinstance(value, list):
            lines.extend(f"- {item}" for item in value)
        else:
            lines.append(str(value))
    events = payload.get("timeline") or []
    if events:
        lines.extend(["", "## Recent Timeline", ""])
        for event in events[:8]:
            lines.append(f"- `{event.get('updated_at')}` {event.get('kind')}: {event.get('title')} ({event.get('citation')})")
    return "\n".join(lines) + "\n"


def resume_context(
    base_dir: Path,
    *,
    project: str,
    session_id: str | None = None,
    lookback_hours: int = 24,
    limit: int = 12,
    token_budget: int = 1200,
) -> dict[str, Any]:
    sync_project_memories(base_dir, project)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(lookback_hours or 24)))
    with _connect(base_dir) as conn:
        if not session_id:
            row = conn.execute(
                """
                SELECT session_id FROM observations
                WHERE project = ? AND archived = 0 AND session_id != ''
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (project,),
            ).fetchone()
            session_id = str(row["session_id"]) if row else ""
        rows = conn.execute(
            """
            SELECT * FROM observations
            WHERE project = ? AND archived = 0 AND session_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (project, session_id or "", max(1, min(int(limit or 12), 50))),
        ).fetchall()
    records = [_row_to_record(row) for row in rows]
    records = [record for record in records if _parse_time(record["updated_at"]) >= cutoff] or records
    if not records:
        return {
            "status": "empty",
            "project": project,
            "session_id": session_id or "",
            "rendered_context": "No previous DysonSpherain session context is available.\n",
            "token_estimate": {"estimated_tokens": 0, "budget": token_budget, "over_budget": False},
        }
    metadata = [dict(record.get("metadata") or {}) for record in records]
    summaries = [record["content"] for record in records if record.get("kind") in {"agent_run_summary", "stop_summary", "conversation_summary"} and record.get("content")]
    token_events = [meta for record, meta in zip(records, metadata) if record.get("kind") == "token_economy_event"]
    baseline_total = sum(int(meta.get("baseline_context_tokens") or 0) for meta in token_events)
    saved_total = sum(int(meta.get("estimated_saved_tokens") or 0) for meta in token_events)
    payload = {
        "status": "ok",
        "project": project,
        "session_id": session_id or "",
        "last_updated_at": max(str(record.get("updated_at") or "") for record in records),
        "event_count": len(records),
        "current_goal": next((str(meta.get("task_goal")) for meta in metadata if meta.get("task_goal")), ""),
        "last_summary": summaries[0] if summaries else records[0].get("content", ""),
        "next_actions": _unique_values([meta.get("next_actions") for meta in metadata]),
        "failures": _unique_values([meta.get("failures") for meta in metadata]),
        "files_changed": _unique_values([meta.get("files_changed") for meta in metadata]),
        "tests_run": _unique_values([meta.get("tests_run") for meta in metadata]),
        "commands_run": _unique_values([meta.get("commands_run") for meta in metadata]),
        "token_economy": {
            "baseline_context_tokens": baseline_total,
            "estimated_saved_tokens": saved_total,
            "saving_ratio": (saved_total / baseline_total) if baseline_total else 0.0,
        },
        "timeline": [
            {
                "observation_id": record["observation_id"],
                "kind": record["kind"],
                "title": record["title"],
                "updated_at": record["updated_at"],
                "citation": record["citation"],
            }
            for record in records
        ],
    }
    rendered = render_resume_markdown(payload)
    count = TokenCounter().count(rendered)
    if count.tokens > token_budget:
        rendered = rendered[: max(240, int(token_budget) * 4)].rstrip() + "\n...[truncated]\n"
        count = TokenCounter().count(rendered)
    payload["rendered_context"] = rendered
    payload["token_estimate"] = {"estimated_tokens": count.tokens, "budget": token_budget, "over_budget": count.tokens > token_budget}
    return payload
