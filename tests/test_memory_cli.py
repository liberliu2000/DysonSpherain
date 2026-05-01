from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.cli import app


class MemoryCliTests(unittest.TestCase):
    def test_memory_remember_search_inspect_update_archive(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous_cwd = Path.cwd()
            os.chdir(root)
            self.addCleanup(os.chdir, previous_cwd)

            result = runner.invoke(
                app,
                [
                    "memory",
                    "remember",
                    "--project",
                    "DysonSpherain",
                    "--type",
                    "decision",
                    "--content",
                    "Use artifact-backed pending rows only.",
                    "--title",
                    "formal evidence rule",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            memory_id = json.loads(result.output)["memory_id"]

            result = runner.invoke(app, ["memory", "search", "pending", "--project", "DysonSpherain", "--json"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result.output)[0]["memory_id"], memory_id)

            result = runner.invoke(app, ["memory", "inspect", memory_id, "--project", "DysonSpherain"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result.output)["title"], "formal evidence rule")

            result = runner.invoke(
                app,
                ["memory", "update", memory_id, "--project", "DysonSpherain", "--status", "superseded"],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result.output)["status"], "superseded")

            result = runner.invoke(app, ["memory", "archive", memory_id, "--project", "DysonSpherain"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn(memory_id, result.output)


if __name__ == "__main__":
    unittest.main()
