from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dysonspherain.memory_os.observation_store import resume_context
from dysonspherain.memory_os.project_state import ProjectStateRequest, get_project_state
from .runtime_ledger import append_hook_event


def main() -> None:
    payload = json.load(sys.stdin)
    cwd = str(payload.get("cwd") or os.getcwd())
    append_hook_event(cwd, "SessionStart", payload, event_type="agent_action_observed")
    state = get_project_state(ProjectStateRequest(cwd=cwd, token_budget=int(payload.get("token_budget") or 800)))
    resume = resume_context(
        Path(cwd).resolve(),
        project=str(payload.get("project") or "DysonSpherain"),
        session_id=str(payload.get("session_id")) if payload.get("session_id") else None,
        lookback_hours=int(payload.get("lookback_hours") or 24),
        token_budget=int(payload.get("resume_token_budget") or 1200),
    )
    text = json.dumps(state.get("project_state") or {}, ensure_ascii=False, sort_keys=True)
    resume_text = str(resume.get("rendered_context") or "")
    if (not text.strip() or text == "{}") and resume.get("status") != "ok":
        print("{}")
        return
    context = "DysonSpherain project state:\n" + text
    if resume.get("status") == "ok":
        context += "\n\n" + resume_text
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context,
                }
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
