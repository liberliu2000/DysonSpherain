from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dysonspherain.memory_runtime.config import load_runtime_config, save_runtime_config
from dysonspherain.memory_runtime.ledger import replay_events


class RuntimeConfigTests(unittest.TestCase):
    def test_runtime_config_save_writes_ledger_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result = save_runtime_config(base, {"context_budget": 900, "ui_animation_intensity": "low"})
            config = load_runtime_config(base)
            events = replay_events(base, project="DysonSpherain")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(config.context_budget, 900)
            self.assertEqual(config.ui_animation_intensity, "low")
            self.assertTrue(any(event.event_type == "constraint_changed" for event in events))

    def test_runtime_config_saves_operator_and_backend_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result = save_runtime_config(
                base,
                {
                    "embedding_backend": "test_dense",
                    "lexical_backend": "test_fts",
                    "projection_backend": "json_projection",
                    "cache_policy": "no_cache",
                    "enabled_operators": ["recent_event_scan", "artifact_lookup"],
                    "scheduler_triggers": ["session_ended"],
                },
            )
            config = result["config"]
            self.assertEqual(config["embedding_backend"], "test_dense")
            self.assertEqual(config["enabled_operators"], ["recent_event_scan", "artifact_lookup"])
            self.assertIn("scheduler_triggers", config)


if __name__ == "__main__":
    unittest.main()
