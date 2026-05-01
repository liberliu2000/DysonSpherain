from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dysonspherain.memory_os.observation_store import write_observation
from .runtime_ledger import append_hook_event


def _valuable(payload: dict) -> bool:
    tool_name = str(payload.get("tool_name") or payload.get("name") or "")
    if tool_name in {"Read", "LS", "Glob"}:
        return False
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return any(marker in text.lower() for marker in ("pytest", "error", "failed", "benchmark", "apply_patch", "write", "edit", "metrics"))


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("should_write") is False or not _valuable(payload):
        print("{}")
        return
    cwd = str(payload.get("cwd") or os.getcwd())
    tool_name = str(payload.get("tool_name") or payload.get("name") or "tool")
    ledger_result = append_hook_event(cwd, "PostToolUse", payload)
    result = write_observation(
        Path(cwd).resolve(),
        project=str(payload.get("project") or "DysonSpherain"),
        kind="tool_event",
        title=f"Tool event: {tool_name}",
        content=json.dumps(payload.get("result") or payload.get("output") or payload, ensure_ascii=False, sort_keys=True)[:8000],
        source="claude_code_post_tool_use",
        session_id=str(payload.get("session_id") or ""),
        metadata={"tool_name": tool_name, "files_changed": payload.get("files_changed") or []},
    )
    result["dysonLedger"] = ledger_result
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
