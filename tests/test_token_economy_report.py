from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.token_economy.metrics import TokenEconomySample, detect_token_economy_failures
from dysonspherain.token_economy.report import write_report


class TokenEconomyReportTests(unittest.TestCase):
    def test_report_writes_required_artifacts_and_failures(self) -> None:
        sample = TokenEconomySample(
            sample_id="s1",
            query="q",
            mode="conservative",
            baseline_type="full_history",
            context_token_budget=100,
            raw_history_tokens=50,
            raw_history_chars=200,
            retrieved_context_tokens=100,
            retrieved_context_chars=400,
            final_prompt_tokens=100,
            final_prompt_chars=400,
            evidence_tokens=90,
            metadata_tokens=5,
            instruction_tokens=20,
            memory_header_tokens=10,
        ).finalize()
        failures = detect_token_economy_failures([sample])
        self.assertEqual(len(failures["token_regression"]), 1)
        self.assertEqual(len(failures["evidence_bloat"]), 1)
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_report([sample], Path(tmp))
            for name in ("per_sample.jsonl", "summary.json", "summary.md", "mode_comparison.csv", "token_quality_tradeoff.csv", "failure_cases.json"):
                self.assertTrue((Path(tmp) / name).exists(), name)
            payload = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["sample_count"], summary["sample_count"])

    def test_report_writes_prefixed_benchmark_artifacts(self) -> None:
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
            final_prompt_tokens=30,
            final_prompt_chars=120,
        ).finalize()
        with tempfile.TemporaryDirectory() as tmp:
            write_report([sample], Path(tmp), filename_prefix="token_economy_")
            for name in (
                "token_economy_per_sample.jsonl",
                "token_economy_summary.json",
                "token_economy_summary.md",
                "token_economy_mode_comparison.csv",
                "token_economy_token_quality_tradeoff.csv",
                "token_economy_failure_cases.json",
            ):
                self.assertTrue((Path(tmp) / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
