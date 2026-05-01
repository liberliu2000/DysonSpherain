from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.execution_ledger import (
    create_execution_run,
    get_execution_run,
    load_execution_runs,
    record_postrun,
    render_resume_packet,
    update_execution_run,
)


class ExecutionLedgerTests(unittest.TestCase):
    def test_create_update_and_load_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = create_execution_run(root, project="DysonSpherain", task="Phase 9 ledger", status="running")

            updated = update_execution_run(
                root,
                project="DysonSpherain",
                run_id=run.run_id,
                status="completed",
                artifacts=["reports/phase9.md"],
                tests_run=["python -m unittest tests/test_execution_ledger.py"],
                next_action="Proceed to Phase 10.",
            )
            runs = load_execution_runs(root, "DysonSpherain")

            self.assertEqual(len(runs), 1)
            self.assertEqual(updated.status, "completed")
            self.assertIsNotNone(updated.completed_at)
            self.assertEqual(runs[0].artifacts, ["reports/phase9.md"])
            self.assertEqual(get_execution_run(root, "DysonSpherain", run.run_id).next_action, "Proceed to Phase 10.")

    def test_postrun_creates_terminal_run_and_resume_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            run = record_postrun(
                root,
                project="DysonSpherain",
                summary="Implemented Phase 9 ledger.",
                source="unit-test.log",
                status="completed",
                artifacts=["artifacts/execution_ledger/demo.jsonl"],
            )
            packet = render_resume_packet(root, "DysonSpherain", run.run_id)

            self.assertIn("# Agent Resume Packet", packet)
            self.assertIn(run.run_id, packet)
            self.assertIn("terminal", packet)
            self.assertIn("artifacts/execution_ledger/demo.jsonl", packet)

    def test_missing_run_update_is_not_marked_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            run = update_execution_run(
                root,
                project="DysonSpherain",
                run_id="run_external",
                task="Recovered interrupted run",
                next_action="Inspect partial artifacts.",
            )

            self.assertEqual(run.status, "running")
            self.assertIsNone(run.completed_at)


if __name__ == "__main__":
    unittest.main()
