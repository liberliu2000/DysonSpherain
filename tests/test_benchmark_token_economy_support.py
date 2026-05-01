from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
BENCHMARKS = ROOT / "benchmarks"
for path in (ROOT, BENCHMARKS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from token_economy_support import add_token_economy_args, record_token_economy_for_metrics


class BenchmarkTokenEconomySupportTests(unittest.TestCase):
    def test_shared_args_include_record_flag(self) -> None:
        parser = argparse.ArgumentParser()
        add_token_economy_args(parser)
        args = parser.parse_args(["--record-token-economy", "--token-economy-modes", "off", "--context-token-budget", "200"])
        self.assertTrue(args.record_token_economy)
        self.assertEqual(args.token_economy_modes, "off")
        self.assertEqual(args.context_token_budget, "200")
        self.assertEqual(args.low_saving_threshold, 0.2)
        self.assertEqual(args.quality_drop_threshold, 0.05)
        self.assertEqual(args.evidence_bloat_threshold, 0.85)
        self.assertEqual(args.metadata_bloat_threshold, 0.25)

    def test_metrics_recording_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_path = root / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "benchmark_name": "longmemeval",
                        "results": [
                            {
                                "question_id": "q1",
                                "question": "What changed?",
                                "metrics": {"session": {"recall_any@10": 1.0, "ndcg_any@10": 0.9}},
                                "candidate_recall": {"candidate_recall@100": 1.0, "gold_rank_after_rerank": 1},
                                "ranked_items": [{"corpus_id": "s1", "text": "evidence", "score": 1.0}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = record_token_economy_for_metrics(
                metrics_path=metrics_path,
                modes="off",
                baseline_types="full_history",
                context_token_budget="200",
                low_saving_threshold=0.99,
                quality_drop_threshold=0.01,
                evidence_bloat_threshold=0.5,
                metadata_bloat_threshold=0.1,
            )
            self.assertTrue((root / "token_economy_per_sample.jsonl").exists())
            self.assertTrue((root / "token_economy_summary.json").exists())
            self.assertTrue((root / "token_economy_summary.md").exists())
            self.assertTrue((root / "token_economy_mode_comparison.csv").exists())
            self.assertTrue((root / "token_economy_token_quality_tradeoff.csv").exists())
            self.assertTrue((root / "token_economy_failure_cases.json").exists())
            self.assertTrue((root / "token_economy_manifest.json").exists())
            ledger_path = root / "token_economy_ledger_events.jsonl"
            self.assertTrue(ledger_path.exists())
            ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(ledger_rows), 1)
            self.assertEqual(ledger_rows[0]["adapter"], "benchmark")
            self.assertEqual(ledger_rows[0]["ledger_version"], "token-economy-ledger-v1")
            self.assertEqual(result["summary"]["sample_count"], 1)
            self.assertEqual(result["thresholds"]["low_saving_threshold"], 0.99)


if __name__ == "__main__":
    unittest.main()
