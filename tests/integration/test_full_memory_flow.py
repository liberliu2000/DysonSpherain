from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dysonspherain.memory_runtime.events import build_event
from dysonspherain.memory_runtime.ledger import append_event, replay_events
from dysonspherain.memory_runtime.runtime import graph_state, recall_runtime


ROOT = Path(__file__).resolve().parents[2] / "base"


class FullMemoryFlowTests(unittest.TestCase):
    def test_full_memory_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            events = [
                build_event(event_type="user_instruction_received", payload={"content": "Continue runtime upgrade"}, source="test", actor="user", timestamp="2026-04-30T00:00:00+00:00"),
                build_event(event_type="agent_action_observed", payload={"summary": "Implemented ledger"}, source="test", actor="assistant", timestamp="2026-04-30T00:01:00+00:00"),
                build_event(event_type="file_changed", payload={"path": "base/dysonspherain/memory_runtime/ledger.py"}, source="test", actor="assistant", timestamp="2026-04-30T00:02:00+00:00"),
                build_event(event_type="benchmark_finished", payload={"summary": "runtime tests passed", "metric": "pytest"}, source="test", actor="system", timestamp="2026-04-30T00:03:00+00:00"),
            ]
            for event in events:
                append_event(base, event)
            graph = graph_state(base)
            recall = recall_runtime(base, "继续 runtime upgrade", budget=1200, trace=True)
            replayed = [event.to_dict() for event in replay_events(base)]
            self.assertEqual(graph["source_event_count"], 4)
            self.assertEqual(recall["status"], "ok")
            self.assertIn("packet", recall)
            self.assertGreaterEqual(len(replayed), 4)

    def test_cli_recall_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            event_file = base / "event.json"
            event_file.write_text(json.dumps({"event_type": "user_instruction_received", "payload": {"content": "Continue task"}, "timestamp": "2026-04-30T00:00:00+00:00"}), encoding="utf-8")
            env = {"PYTHONPATH": str(ROOT)}
            subprocess.run([sys.executable, "-m", "sphere_cli.cli", "memory", "append", "--file", str(event_file)], cwd=base, env=env, check=True, capture_output=True, text=True)
            proc = subprocess.run([sys.executable, "-m", "sphere_cli.cli", "memory", "recall", "继续任务", "--budget", "500"], cwd=base, env=env, check=True, capture_output=True, text=True)
            self.assertIn("DysonSpherain Context Packet", proc.stdout)


if __name__ == "__main__":
    unittest.main()

