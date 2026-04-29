from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_SCOPE = "global"
DEFAULT_SCOPE_ORDER = ("project", "session", "global")
SUPPORTED_SCOPE_KINDS = {"global", "project", "session", "scratch"}
SUPPORTED_MEMORY_OBJECT_TYPES = {
    "artifact",
    "constraint",
    "decision",
    "entity",
    "event",
    "fact",
    "goal",
    "open_loop",
    "pattern",
    "persona",
    "personal_context",
    "preference",
    "project",
    "relation",
    "solution_card",
    "state",
    "state_update",
    "temporal_reference",
}
MODE_PROFILES: dict[str, dict[str, int]] = {
    "fast": {
        "retrieval_topk_coarse": 18,
        "retrieval_topk_fine": 6,
        "segment_rerank_topk_default": 8,
        "confusing_cluster_topk_default": 14,
    },
    "balanced": {},
    "deep": {
        "retrieval_topk_coarse": 32,
        "retrieval_topk_fine": 10,
        "segment_rerank_topk_default": 18,
        "confusing_cluster_topk_default": 28,
    },
}


def normalize_name(value: str | None) -> str | None:
    cleaned = " ".join(str(value or "").strip().split())
    return cleaned or None


def normalize_mode(value: str | None) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in MODE_PROFILES:
        return lowered
    return "balanced"


def parse_scope(scope: str | None) -> tuple[str, str | None]:
    raw = str(scope or "").strip()
    if not raw or raw.lower() == "global":
        return "global", None
    if ":" in raw:
        kind, name = raw.split(":", 1)
        kind = kind.strip().lower()
        name = normalize_name(name)
        if kind in SUPPORTED_SCOPE_KINDS and name:
            return kind, name
    lowered = raw.lower()
    if lowered in SUPPORTED_SCOPE_KINDS:
        return lowered, None
    return "global", None


def compose_scope(
    *,
    scope: str | None = None,
    project: str | None = None,
    session_id: str | None = None,
) -> str:
    if str(scope or "").strip().lower() == "global":
        return DEFAULT_SCOPE
    explicit_kind, explicit_name = parse_scope(scope)
    if explicit_kind in {"project", "session", "scratch"} and explicit_name:
        return f"{explicit_kind}:{explicit_name}"
    normalized_project = normalize_name(project)
    if normalized_project:
        return f"project:{normalized_project}"
    normalized_session = normalize_name(session_id)
    if normalized_session:
        return f"session:{normalized_session}"
    return DEFAULT_SCOPE


def normalize_scope_order(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_SCOPE_ORDER
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",")]
    else:
        parts = [str(part).strip().lower() for part in value]
    cleaned: list[str] = []
    for part in parts:
        if part not in SUPPORTED_SCOPE_KINDS:
            continue
        if part not in cleaned:
            cleaned.append(part)
    if "global" not in cleaned:
        cleaned.append("global")
    return tuple(cleaned or list(DEFAULT_SCOPE_ORDER))


@dataclass
class WorkspaceContext:
    workspace: str | None = None
    project: str | None = None
    session_id: str | None = None
    scope: str = DEFAULT_SCOPE
    scope_order: tuple[str, ...] = DEFAULT_SCOPE_ORDER
    mode: str = "balanced"

    @classmethod
    def from_values(
        cls,
        *,
        workspace: str | None = None,
        project: str | None = None,
        session_id: str | None = None,
        scope: str | None = None,
        scope_order: str | list[str] | tuple[str, ...] | None = None,
        mode: str | None = None,
    ) -> "WorkspaceContext":
        normalized_project = normalize_name(project)
        normalized_session = normalize_name(session_id)
        return cls(
            workspace=normalize_name(workspace),
            project=normalized_project,
            session_id=normalized_session,
            scope=compose_scope(scope=scope, project=normalized_project, session_id=normalized_session),
            scope_order=normalize_scope_order(scope_order),
            mode=normalize_mode(mode),
        )

    def to_route_context(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scope": self.scope,
            "scope_order": list(self.scope_order),
            "mode": self.mode,
        }
        if self.workspace:
            payload["workspace"] = self.workspace
        if self.project:
            payload["project"] = self.project
        if self.session_id:
            payload["session_id"] = self.session_id
        return payload

    def candidate_scope_rank(
        self,
        *,
        scope: str | None,
        project: str | None,
        session_id: str | None,
        workspace: str | None,
    ) -> tuple[int, str]:
        if not any([self.workspace, self.project, self.session_id]) and self.scope == DEFAULT_SCOPE:
            return -1, "inactive"
        candidate_workspace = normalize_name(workspace)
        if self.workspace and candidate_workspace and candidate_workspace != self.workspace:
            return -2, "workspace_mismatch"
        normalized_project = normalize_name(project)
        normalized_session = normalize_name(session_id)
        scope_kind, _ = parse_scope(scope)
        for index, kind in enumerate(self.scope_order):
            if kind == "session" and self.session_id and normalized_session == self.session_id:
                return index, "session_match"
            if kind == "project" and self.project and normalized_project == self.project:
                return index, "project_match"
            if kind == "global" and (scope_kind == "global" or (not normalized_project and not normalized_session)):
                return index, "global_match"
        return -2, "scope_mismatch"

    def candidate_scope_bonus(
        self,
        *,
        scope: str | None,
        project: str | None,
        session_id: str | None,
        workspace: str | None,
        weight: float,
    ) -> tuple[float, str]:
        rank, label = self.candidate_scope_rank(
            scope=scope,
            project=project,
            session_id=session_id,
            workspace=workspace,
        )
        if rank < 0:
            return 0.0, label
        denominator = max(1, len(self.scope_order))
        bonus = max(0.0, (denominator - rank) / denominator) * max(0.0, float(weight or 0.0))
        return round(bonus, 4), label


def apply_mode_profile(config: Any) -> None:
    mode = normalize_mode(getattr(config, "mode", None))
    for key, value in MODE_PROFILES.get(mode, {}).items():
        current = getattr(config, key, None)
        if not isinstance(current, int):
            continue
        if mode == "fast":
            setattr(config, key, min(current, value))
        else:
            setattr(config, key, max(current, value))
