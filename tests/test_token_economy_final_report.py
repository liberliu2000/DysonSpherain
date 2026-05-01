from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.token_economy.metrics import TokenEconomySample
from dysonspherain.token_economy.report import write_report
from dysonspherain.token_economy.final_report import write_final_report


class TokenEconomyFinalReportTests(unittest.TestCase):
    def test_final_report_uses_artifacts(self) -> None:
        sample = TokenEconomySample(
            sample_id="s1",
            query="q",
            mode="conservative",
            baseline_type="full_history",
            context_token_budget=1000,
            raw_history_tokens=1000,
            raw_history_chars=4000,
            retrieved_context_tokens=100,
            retrieved_context_chars=400,
            final_prompt_tokens=150,
            final_prompt_chars=600,
        ).finalize()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_report([sample], root)
            report = write_final_report(root, benchmark_rerun_status="unit test")
            text = report.read_text(encoding="utf-8")
            self.assertIn("DysonSpherain Token Economy Final Report", text)
            self.assertIn("unit test", text)


if __name__ == "__main__":
    unittest.main()
