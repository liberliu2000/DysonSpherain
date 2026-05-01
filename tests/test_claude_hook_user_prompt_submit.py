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

from sphere_cli.project_state import write_memory
from dysonspherain.memory_os.observation_store import write_observation
from dysonspherain.memory_os.observation_store import token_economy_summary
from dysonspherain.memory_runtime.ledger import replay_events


class ClaudeHookUserPromptSubmitTests(unittest.TestCase):
    def test_short_prompt_returns_empty_object(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "dysonspherain.adapters.claude_hooks.user_prompt_submit"],
            input=json.dumps({"prompt": "hi"}),
            text=True,
            capture_output=True,
            env={"PYTHONPATH": str(ROOT)},
            check=True,
        )
        self.assertEqual(json.loads(proc.stdout), {})

    def test_relevant_prompt_returns_additional_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_memory(Path(tmp), memory_type="decision", project="DysonSpherain", content="Benchmark regression uses artifact-backed diagnosis.", source="test")
            proc = subprocess.run(
                [sys.executable, "-m", "dysonspherain.adapters.claude_hooks.user_prompt_submit"],
                input=json.dumps({"prompt": "Please debug the benchmark regression using prior project memory.", "cwd": tmp}),
                text=True,
                capture_output=True,
                env={"PYTHONPATH": str(ROOT)},
                check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertIn("additionalContext", payload["hookSpecificOutput"])
            self.assertIn("estimated_saved_tokens", payload["hookSpecificOutput"]["additionalContext"])
            summary = token_economy_summary(Path(tmp), project="DysonSpherain")
            self.assertGreaterEqual(len(summary["events"]), 1)
            self.assertTrue(any(event.event_type == "user_instruction_received" for event in replay_events(Path(tmp))))

    def test_short_continuation_prompt_uses_resume_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_observation(
                Path(tmp),
                project="DysonSpherain",
                kind="agent_run_summary",
                title="Previous task summary",
                content="Last session repaired resume context and planned memory intent integration.",
                source="test",
                session_id="s-continue",
                metadata={"task_goal": "resume continuation", "next_actions": ["Implement dyson_memory_intent"]},
            )
            proc = subprocess.run(
                [sys.executable, "-m", "dysonspherain.adapters.claude_hooks.user_prompt_submit"],
                input=json.dumps({"prompt": "继续", "cwd": tmp, "session_id": "s-new"}),
                text=True,
                capture_output=True,
                env={"PYTHONPATH": str(ROOT)},
                check=True,
            )
            payload = json.loads(proc.stdout)
            context = payload["hookSpecificOutput"]["additionalContext"]
            self.assertIn("DysonSpherain Resume Context", context)
            self.assertIn("memory intent integration", context)
            self.assertIn("intent_reason=cross_session_continuation", context)


if __name__ == "__main__":
    unittest.main()
