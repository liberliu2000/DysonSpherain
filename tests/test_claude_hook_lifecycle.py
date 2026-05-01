from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.memory_runtime.ledger import replay_events


class ClaudeHookLifecycleTests(unittest.TestCase):
    def _run_hook(self, module: str, payload: dict) -> dict:
        proc = subprocess.run(
            [sys.executable, "-m", module],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            cwd=str(ROOT.parent),
            env={"PYTHONPATH": str(ROOT)},
            check=True,
        )
        return json.loads(proc.stdout or "{}")

    def test_session_start_returns_project_state_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_hook("dysonspherain.adapters.claude_hooks.session_start", {"cwd": tmp, "token_budget": 400})
            self.assertIn("hookSpecificOutput", result)
            self.assertIn("DysonSpherain project state", result["hookSpecificOutput"]["additionalContext"])
            self.assertIn("recommended_focus", result["hookSpecificOutput"]["additionalContext"])
            self.assertTrue(any(event.provenance.get("hook_event") == "SessionStart" for event in replay_events(Path(tmp))))

    def test_session_start_includes_resume_context_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._run_hook(
                "dysonspherain.adapters.claude_hooks.stop",
                {"cwd": tmp, "session_id": "previous", "summary": "Continue with resume context tests and MCP schema updates."},
            )
            result = self._run_hook("dysonspherain.adapters.claude_hooks.session_start", {"cwd": tmp, "token_budget": 400})
            context = result["hookSpecificOutput"]["additionalContext"]
            self.assertIn("DysonSpherain Resume Context", context)
            self.assertIn("Continue with resume context", context)

    def test_post_tool_use_writes_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_hook(
                "dysonspherain.adapters.claude_hooks.post_tool_use",
                {"cwd": tmp, "session_id": "s1", "tool_name": "Bash", "output": "pytest passed"},
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["kind"], "tool_event")
            self.assertTrue(any(event.provenance.get("hook_event") == "PostToolUse" for event in replay_events(Path(tmp))))

    def test_stop_writes_summary_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_hook(
                "dysonspherain.adapters.claude_hooks.stop",
                {"cwd": tmp, "session_id": "s1", "summary": "Finished a useful token economy implementation checkpoint."},
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["kind"], "stop_summary")
            self.assertTrue(any(event.provenance.get("hook_event") == "Stop" for event in replay_events(Path(tmp))))


if __name__ == "__main__":
    unittest.main()
