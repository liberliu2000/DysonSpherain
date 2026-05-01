from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dysonspherain.memory_os.observation_store import write_observation
from .runtime_ledger import append_hook_event


def main() -> None:
    payload = json.load(sys.stdin)
    summary = str(payload.get("summary") or payload.get("reason") or payload.get("stop_reason") or "")
    if payload.get("should_write") is False or len(summary.strip()) < 20:
        print("{}")
        return
    cwd = str(payload.get("cwd") or os.getcwd())
    ledger_result = append_hook_event(cwd, "Stop", payload, event_type="assistant_response_generated")
    result = write_observation(
        Path(cwd).resolve(),
        project=str(payload.get("project") or "DysonSpherain"),
        kind="stop_summary",
        title=str(payload.get("title") or "Agent stop summary"),
        content=summary,
        source="claude_code_stop",
        session_id=str(payload.get("session_id") or ""),
        metadata={"next_actions": payload.get("next_actions") or []},
    )
    result["dysonLedger"] = ledger_result
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
