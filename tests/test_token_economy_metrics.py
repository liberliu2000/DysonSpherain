from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.token_economy.metrics import RetrievalQuality, TokenEconomySample, detect_token_economy_failures


class TokenEconomyMetricsTests(unittest.TestCase):
    def test_metric_ratios_avoid_divide_by_zero(self) -> None:
        sample = TokenEconomySample(
            sample_id="s1",
            query="q",
            mode="conservative",
            baseline_type="full_history",
            context_token_budget=100,
            raw_history_tokens=0,
            raw_history_chars=0,
            retrieved_context_tokens=10,
            retrieved_context_chars=40,
            final_prompt_tokens=10,
            final_prompt_chars=40,
        ).finalize()
        self.assertEqual(sample.saved_tokens_ratio, 0.0)
        self.assertEqual(sample.compression_ratio, 0.0)

    def test_saved_token_ratios(self) -> None:
        sample = TokenEconomySample(
            sample_id="s1",
            query="q",
            mode="conservative",
            baseline_type="full_history",
            context_token_budget=100,
            raw_history_tokens=100,
            raw_history_chars=400,
            retrieved_context_tokens=20,
            retrieved_context_chars=80,
            final_prompt_tokens=25,
            final_prompt_chars=100,
        ).finalize()
        self.assertEqual(sample.saved_tokens_abs, 75)
        self.assertEqual(sample.saved_tokens_ratio, 0.75)
        self.assertEqual(sample.context_reduction_ratio, 0.75)

    def test_paired_quality_drop_is_detected(self) -> None:
        strong = TokenEconomySample(
            sample_id="s1",
            query="q",
            mode="off",
            baseline_type="full_history",
            context_token_budget=100,
            raw_history_tokens=100,
            raw_history_chars=400,
            retrieved_context_tokens=100,
            retrieved_context_chars=400,
            final_prompt_tokens=100,
            final_prompt_chars=400,
            retrieval_quality=RetrievalQuality(recall_at_10=0.95),
        ).finalize()
        compressed = TokenEconomySample(
            sample_id="s1",
            query="q",
            mode="conservative",
            baseline_type="full_history",
            context_token_budget=100,
            raw_history_tokens=100,
            raw_history_chars=400,
            retrieved_context_tokens=20,
            retrieved_context_chars=80,
            final_prompt_tokens=30,
            final_prompt_chars=120,
            retrieval_quality=RetrievalQuality(recall_at_10=0.70),
        ).finalize()
        failures = detect_token_economy_failures([strong, compressed], quality_drop_threshold=0.05)
        self.assertEqual(len(failures["paired_quality_drop"]), 1)
        self.assertEqual(failures["paired_quality_drop"][0]["mode"], "conservative")


if __name__ == "__main__":
    unittest.main()
