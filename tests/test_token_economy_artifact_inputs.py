from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.token_economy.artifact_inputs import payloads_from_benchmark_metrics


class TokenEconomyArtifactInputsTests(unittest.TestCase):
    def test_metrics_parser_extracts_quality_for_core_benchmarks(self) -> None:
        rows = [
            {
                "question_id": "longmemeval_1",
                "question": "q1",
                "metrics": {"session": {"recall_any@5": 0.5, "recall_any@10": 1.0, "ndcg_any@10": 0.9}},
                "candidate_recall": {"candidate_recall@100": 1.0, "gold_rank_after_rerank": 2},
            },
            {
                "question_id": "locomo_1",
                "question": "q2",
                "metrics": {"session": {"recall_frac@10": 0.8, "ndcg_any@10": 0.7}},
                "candidate_recall": {"candidate_recall@100": 0.9, "gold_rank_after_inhibition": 4},
            },
            {
                "question_id": "knowme_1",
                "question": "q3",
                "metrics": {"segment": {"recall_frac@10": 0.6, "ndcg_any@10": 0.5}},
                "candidate_recall": {"candidate_recall@100": 0.7, "final_ndcg@10": 0.55, "gold_rank_before_rerank": 6},
            },
            {
                "question_id": "clonemem_1",
                "question": "q4",
                "metrics": {"segment": {"recall_frac@5": 0.2, "recall_frac@10": 0.3, "ndcg_any@10": 0.25}},
                "candidate_recall": {"candidate_recall@100": 0.4, "final_recall@10": 0.3, "gold_rank_after_inhibition": 8},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps({"benchmark_name": "mixed", "results": rows}), encoding="utf-8")
            payloads = payloads_from_benchmark_metrics(path)
        self.assertEqual(len(payloads), 4)
        for payload in payloads:
            quality = payload["retrieval_quality"]
            self.assertIsNotNone(quality["recall_at_10"])
            self.assertIsNotNone(quality["ndcg_at_10"])
            self.assertIsNotNone(quality["gold_rank"])
            self.assertIsNotNone(quality["candidate_recall_at_100"])


if __name__ == "__main__":
    unittest.main()
