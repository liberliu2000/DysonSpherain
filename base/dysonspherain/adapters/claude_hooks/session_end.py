from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dysonspherain.memory_os.write_service import WriteMemoryRequest, write_memory
from dysonspherain.writeback.session_summarizer import merge_hook_payload
from .runtime_ledger import append_hook_event


def _log_error(cwd: str, payload: dict | None, exc: Exception) -> Path:
    root = Path(cwd or os.getcwd()).resolve()
    log_path = root / "artifacts" / "claude_hooks" / "session_end_errors.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "error": str(exc),
        "session_id": (payload or {}).get("session_id") if isinstance(payload, dict) else None,
        "transcript_path": (payload or {}).get("transcript_path") if isinstance(payload, dict) else None,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return log_path


def main() -> None:
    raw_payload = None
    try:
        raw_payload = json.load(sys.stdin)
        payload = merge_hook_payload(raw_payload)
        has_explicit_value = any(payload.get(key) not in (None, "", [], {}) for key in ("summary", "files_changed", "commands_run", "tests_run", "benchmark_results", "failures", "next_actions"))
        if payload.get("should_write") is False and not has_explicit_value:
            print(json.dumps({"status": "skipped", "reason": "no_long_term_value"}, ensure_ascii=False))
            return
        ledger_result = append_hook_event(str(payload.get("cwd") or os.getcwd()), "SessionEnd", payload, event_type="assistant_response_generated")
        result = write_memory(
            WriteMemoryRequest(
                cwd=str(payload.get("cwd") or os.getcwd()),
                session_id=str(payload.get("session_id") or ""),
                task_goal=str(payload.get("task_goal") or payload.get("reason") or "Claude Code session ended"),
                summary=str(payload.get("summary") or ""),
                files_changed=list(payload.get("files_changed") or []),
                commands_run=list(payload.get("commands_run") or []),
                tests_run=list(payload.get("tests_run") or []),
                benchmark_results=list(payload.get("benchmark_results") or []),
                failures=list(payload.get("failures") or []),
                next_actions=list(payload.get("next_actions") or []),
                source="claude_code",
            )
        )
        output = result.to_dict()
        output["dysonLedger"] = ledger_result
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        cwd = str((raw_payload or {}).get("cwd") or os.getcwd()) if isinstance(raw_payload, dict) else os.getcwd()
        log_path = _log_error(cwd, raw_payload, exc)
        print(json.dumps({"status": "error", "error": str(exc), "log_path": str(log_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
