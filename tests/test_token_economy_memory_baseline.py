from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.evaluation.token_economy import _history_for_baseline
from sphere_cli.project_state import write_memory


class TokenEconomyMemoryBaselineTests(unittest.TestCase):
    def test_memory_db_builds_full_history_and_naive_recent_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_memory(root, memory_type="decision", project="DysonSpherain", content="older benchmark decision", source="test")
            write_memory(root, memory_type="failure", project="DysonSpherain", content="newer CloneMem regression note", source="test")
            full, full_extra = _history_for_baseline({}, "full_history", 1, memory_base_dir=root, project="DysonSpherain")
            recent, recent_extra = _history_for_baseline({}, "naive_recent", 1, memory_base_dir=root, project="DysonSpherain")
        self.assertIn("older benchmark decision", full)
        self.assertIn("newer CloneMem regression note", full)
        self.assertIn("newer CloneMem regression note", recent)
        self.assertNotIn("older benchmark decision", recent)
        self.assertEqual(full_extra["baseline_source"], "memory_store")
        self.assertEqual(recent_extra["baseline_memory_count"], 1)


if __name__ == "__main__":
    unittest.main()
