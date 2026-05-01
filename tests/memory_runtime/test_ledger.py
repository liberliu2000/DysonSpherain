from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dysonspherain.memory_runtime.events import build_event
from dysonspherain.memory_runtime.ledger import append_event, replay_events


class LedgerTests(unittest.TestCase):
    def test_event_ledger_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            event = build_event(
                event_type="user_instruction_received",
                payload={"content": "continue the DysonSpherain runtime upgrade"},
                source="test",
                actor="user",
                project="DysonSpherain",
                timestamp="2026-04-30T00:00:00+00:00",
            )
            first = append_event(base, event)
            second = append_event(base, event)
            self.assertEqual(first.status, "ok")
            self.assertEqual(second.status, "duplicate")
            self.assertEqual(len(replay_events(base, project="DysonSpherain")), 1)


if __name__ == "__main__":
    unittest.main()

