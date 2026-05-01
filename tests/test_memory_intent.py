from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.memory_os.memory_intent import classify_memory_intent


class MemoryIntentTests(unittest.TestCase):
    def test_continuation_prompt_prefers_resume_and_search(self) -> None:
        intent = classify_memory_intent("继续", cwd="/tmp/project", task_type="coding").to_dict()
        self.assertTrue(intent["should_call_memory"])
        self.assertEqual(intent["reason"], "cross_session_continuation")
        self.assertEqual(intent["recommended_tools"], ["dyson_resume_context", "dyson_search_memory"])
        self.assertEqual(intent["token_budget"], 1200)

    def test_trivial_short_prompt_skips_memory(self) -> None:
        intent = classify_memory_intent("hi").to_dict()
        self.assertFalse(intent["should_call_memory"])
        self.assertEqual(intent["reason"], "low_value_short_prompt")
        self.assertEqual(intent["recommended_tools"], [])

    def test_benchmark_prompt_prefers_project_recall_pipeline(self) -> None:
        intent = classify_memory_intent("Check CloneMem candidate_recall regression").to_dict()
        self.assertTrue(intent["should_call_memory"])
        self.assertEqual(intent["reason"], "benchmark_or_regression")
        self.assertIn("dyson_project_state", intent["recommended_tools"])
        self.assertIn("dyson_recall", intent["recommended_tools"])


if __name__ == "__main__":
    unittest.main()

