from __future__ import annotations

import json
import os
import sys

from dysonspherain.memory_os.write_service import WriteMemoryRequest, write_memory
from .runtime_ledger import append_hook_event


def main() -> None:
    payload = json.load(sys.stdin)
    summary = str(payload.get("summary") or payload.get("compact_summary") or "")
    if len(summary.strip()) < 80:
        print("{}")
        return
    cwd = str(payload.get("cwd") or os.getcwd())
    ledger_result = append_hook_event(cwd, "PostCompact", payload, event_type="memory_compacted")
    result = write_memory(
        WriteMemoryRequest(
            cwd=cwd,
            session_id=str(payload.get("session_id") or ""),
            task_goal="Claude Code post-compact summary",
            summary=summary,
            files_changed=[],
            commands_run=[],
            tests_run=[],
            benchmark_results=[],
            failures=[],
            next_actions=[],
            source="claude_code_post_compact",
        )
    )
    output = result.to_dict()
    output["dysonLedger"] = ledger_result
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
