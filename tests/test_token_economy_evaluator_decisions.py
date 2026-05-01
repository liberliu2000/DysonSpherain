from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.token_economy.evaluator import evaluate
from dysonspherain.utils.token_counter import TokenCounter


class TokenEconomyEvaluatorDecisionTests(unittest.TestCase):
    def test_repeated_context_is_skipped_or_summarized(self) -> None:
        context = "base/sphere_cli/cli.py fixed regression evidence\n" * 50
        result = evaluate(query="fix regression", candidate_context=context, existing_context=context, task_type="debug")
        self.assertIn(result.decision, {"skip", "inject_summary_only"})
        self.assertGreater(result.duplication_score, 0.8)

    def test_relevant_low_duplicate_context_is_injected(self) -> None:
        result = evaluate(
            query="fix token economy evaluator in base/dysonspherain/token_economy/evaluator.py",
            candidate_context="base/dysonspherain/token_economy/evaluator.py contains duplication_score and quality guard logic.",
            existing_context="unrelated note",
            task_type="coding",
        )
        self.assertEqual(result.decision, "inject")

    def test_secret_like_context_returns_file_refs_only(self) -> None:
        result = evaluate(
            query="debug config",
            candidate_context="Use base/config.py. password=supersecret123456 should not be injected as prose.",
            task_type="debug",
        )
        self.assertEqual(result.decision, "return_file_refs_only")
        self.assertEqual(result.risk, "high")

    def test_over_budget_context_summarizes(self) -> None:
        result = evaluate(query="debug benchmark regression", candidate_context="regression evidence " * 1000, token_budget=20, task_type="benchmark")
        self.assertEqual(result.decision, "inject_summary_only")

    def test_fallback_tokenizer_is_visible_in_reason(self) -> None:
        result = evaluate(
            query="debug benchmark regression",
            candidate_context="benchmark regression evidence",
            counter=TokenCounter(strategy="char_heuristic"),
            task_type="benchmark",
        )
        self.assertTrue(result.fallback_tokenizer_used)
        self.assertIn("Fallback tokenizer used", result.reason)


if __name__ == "__main__":
    unittest.main()
