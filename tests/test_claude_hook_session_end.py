from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"


class ClaudeHookSessionEndTests(unittest.TestCase):
    def test_session_end_logs_write_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "project_state").write_text("not a directory", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "dysonspherain.adapters.claude_hooks.session_end"],
                input=json.dumps({"cwd": str(root), "session_id": "s1", "summary": "x" * 120}),
                text=True,
                capture_output=True,
                env={"PYTHONPATH": str(ROOT)},
                check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["status"], "error")
            self.assertIn("log_path", payload)
            self.assertTrue(Path(payload["log_path"]).exists())


if __name__ == "__main__":
    unittest.main()
