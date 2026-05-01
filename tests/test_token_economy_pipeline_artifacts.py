from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.evaluation.token_economy import main


class TokenEconomyPipelineArtifactsTests(unittest.TestCase):
    def test_smoke_outputs_standard_artifacts_without_benchmark_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "token_economy"
            main(
                [
                    "--smoke",
                    "--output",
                    str(output),
                    "--modes",
                    "off,conservative,minimal",
                    "--baseline-types",
                    "full_history,naive_recent,manual_summary",
                    "--context-token-budget",
                    "800,1200",
                ]
            )
            for name in (
                "manifest.json",
                "per_sample.jsonl",
                "summary.json",
                "summary.md",
                "mode_comparison.csv",
                "token_quality_tradeoff.csv",
                "failure_cases.json",
                "tokenizer_calibration.json",
                "ledger_summary.json",
                "final_report.md",
            ):
                self.assertTrue((output / name).exists(), name)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("llm_prompt_token_economy", summary)
            self.assertIn("local_compute_economy", summary)
            self.assertIn("decision_distribution", summary)


if __name__ == "__main__":
    unittest.main()
