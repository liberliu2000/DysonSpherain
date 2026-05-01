from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .events import MemoryEvent, build_event
from .evidence_vm import RecallIntent, infer_recall_intent


class AgentAdapter(Protocol):
    name: str

    def capture_input(self, payload: dict[str, Any]) -> list[MemoryEvent]:
        ...

    def capture_output(self, payload: dict[str, Any]) -> list[MemoryEvent]:
        ...

    def capture_tool_use(self, payload: dict[str, Any]) -> list[MemoryEvent]:
        ...

    def build_context_request(self, payload: dict[str, Any]) -> RecallIntent:
        ...


@dataclass(frozen=True)
class GenericCliAdapter:
    name: str = "generic_cli"

    def capture_input(self, payload: dict[str, Any]) -> list[MemoryEvent]:
        prompt = str(payload.get("prompt") or payload.get("content") or "")
        if not prompt:
            return []
        return [
            build_event(
                event_type="user_instruction_received",
                payload={"content": prompt, "title": payload.get("title") or prompt[:80]},
                source=self.name,
                actor="user",
                project=str(payload.get("project") or "DysonSpherain"),
                session_id=str(payload.get("session_id") or "") or None,
                provenance={"adapter": self.name},
            )
        ]

    def capture_output(self, payload: dict[str, Any]) -> list[MemoryEvent]:
        text = str(payload.get("response") or payload.get("content") or "")
        if not text:
            return []
        return [
            build_event(
                event_type="assistant_response_generated",
                payload={"content": text, "summary": text[:240]},
                source=self.name,
                actor="assistant",
                project=str(payload.get("project") or "DysonSpherain"),
                session_id=str(payload.get("session_id") or "") or None,
                provenance={"adapter": self.name},
            )
        ]

    def capture_tool_use(self, payload: dict[str, Any]) -> list[MemoryEvent]:
        return [
            build_event(
                event_type="tool_call_observed",
                payload=dict(payload),
                source=self.name,
                actor="agent",
                project=str(payload.get("project") or "DysonSpherain"),
                session_id=str(payload.get("session_id") or "") or None,
                provenance={"adapter": self.name},
            )
        ]

    def build_context_request(self, payload: dict[str, Any]) -> RecallIntent:
        return infer_recall_intent(str(payload.get("prompt") or payload.get("query") or ""))


@dataclass(frozen=True)
class CodexAdapter(GenericCliAdapter):
    name: str = "codex"


@dataclass(frozen=True)
class ClaudeCodeAdapter(GenericCliAdapter):
    name: str = "claude_code"

    def capture_tool_use(self, payload: dict[str, Any]) -> list[MemoryEvent]:
        tool_name = str(payload.get("tool_name") or payload.get("name") or "unknown_tool")
        return [
            build_event(
                event_type="tool_call_observed",
                payload={"tool_name": tool_name, "input": payload.get("input") or payload.get("arguments") or {}, "result": payload.get("result") or ""},
                source=self.name,
                actor="agent",
                project=str(payload.get("project") or "DysonSpherain"),
                session_id=str(payload.get("session_id") or "") or None,
                provenance={"adapter": self.name, "hook_event": payload.get("hook_event") or "PostToolUse"},
            )
        ]


@dataclass(frozen=True)
class ManualImportAdapter(GenericCliAdapter):
    name: str = "manual_import"

    def capture_input(self, payload: dict[str, Any]) -> list[MemoryEvent]:
        event_type = str(payload.get("event_type") or "user_instruction_received")
        content = str(payload.get("content") or payload.get("summary") or "")
        if not content and not payload:
            return []
        return [
            build_event(
                event_type=event_type,
                payload=dict(payload.get("payload") or {"content": content, "title": payload.get("title") or content[:80]}),
                source=self.name,
                actor=str(payload.get("actor") or "user"),
                project=str(payload.get("project") or "DysonSpherain"),
                session_id=str(payload.get("session_id") or "") or None,
                provenance={"adapter": self.name, "import_path": payload.get("path") or ""},
            )
        ]


def adapter_for_source(source: str) -> AgentAdapter:
    normalized = source.lower().replace("-", "_")
    if normalized == "codex":
        return CodexAdapter()
    if normalized in {"claude", "claude_code"}:
        return ClaudeCodeAdapter()
    if normalized in {"manual", "manual_import"}:
        return ManualImportAdapter()
    return GenericCliAdapter(name=normalized or "generic_cli")
