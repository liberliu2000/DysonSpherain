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

from sphere_cli.project_state import list_memories


class ClaudeHookPostCompactTests(unittest.TestCase):
    def test_short_summary_returns_empty_object(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "dysonspherain.adapters.claude_hooks.post_compact"],
            input=json.dumps({"summary": "too short"}),
            text=True,
            capture_output=True,
            env={"PYTHONPATH": str(ROOT)},
            check=True,
        )
        self.assertEqual(json.loads(proc.stdout), {})

    def test_long_summary_writes_and_dedupes_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = (
                "PostCompact preserved the benchmark regression goal, files changed, "
                "tests run, remaining token economy tasks, and next actions for the agent."
            )
            payload = {"cwd": tmp, "session_id": "s1", "compact_summary": summary}
            first = subprocess.run(
                [sys.executable, "-m", "dysonspherain.adapters.claude_hooks.post_compact"],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                env={"PYTHONPATH": str(ROOT)},
                check=True,
            )
            second = subprocess.run(
                [sys.executable, "-m", "dysonspherain.adapters.claude_hooks.post_compact"],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                env={"PYTHONPATH": str(ROOT)},
                check=True,
            )
            first_payload = json.loads(first.stdout)
            second_payload = json.loads(second.stdout)
            self.assertEqual(first_payload["status"], "ok")
            self.assertEqual(second_payload["status"], "duplicate")
            records = list_memories(Path(tmp), "DysonSpherain", include_archived=True)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source"], "claude_code_post_compact")


if __name__ == "__main__":
    unittest.main()
