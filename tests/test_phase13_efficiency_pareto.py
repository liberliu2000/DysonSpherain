from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_efficiency_pareto import build_records, build_sweep_status, iter_metrics_files


class Phase13EfficiencyParetoTests(unittest.TestCase):
    def test_iter_metrics_files_includes_nested_merged_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = root / "run" / "locomo" / "locomo" / "merged_metrics.json"
            merged.parent.mkdir(parents=True)
            merged.write_text(json.dumps({"benchmark": "locomo"}), encoding="utf-8")

            files = iter_metrics_files(root)

            self.assertIn(merged.resolve(), [path.resolve() for path in files])

    def test_build_records_extracts_nested_quality_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "run" / "clonemem" / "clonemem" / "merged_metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(
                json.dumps(
                    {
                        "benchmark": "clonemem",
                        "total_question_count": 2374,
                        "wall_clock_elapsed_seconds": 42.0,
                        "metrics": {
                            "segment": {"recall_frac@10": 0.12, "ndcg_any@10": 0.2},
                            "candidate_recall@100": 0.5,
                        },
                        "timings": {"retrieval_ms": 12.5},
                    }
                ),
                encoding="utf-8",
            )

            records = build_records(root)

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["benchmark"], "clonemem")
            self.assertEqual(record["question_count"], 2374)
            self.assertEqual(record["elapsed_seconds"], 42.0)
            self.assertEqual(record["recall_frac@10"], 0.12)
            self.assertEqual(record["candidate_recall@100"], 0.5)
            self.assertEqual(record["retrieval_ms"], 12.5)

    def test_build_sweep_status_recognizes_dedicated_budget_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "candidate_top50" / "longmemeval" / "metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(
                json.dumps(
                    {
                        "schema": "dysonspherain.efficiency_budget_sweep.v1",
                        "benchmark": "longmemeval",
                        "sweep_type": "candidate",
                        "budget": 50,
                        "sweep_name": "candidate_top50",
                        "run_scope": "smoke",
                        "formal_eligible": False,
                        "fallback_in_use": False,
                        "question_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            records = build_records(root)
            status = build_sweep_status(records)
            candidate50 = next(row for row in status if row["sweep"] == "candidate_top50")
            rerank100 = next(row for row in status if row["sweep"] == "rerank_top100")

            self.assertEqual(candidate50["status"], "smoke_available")
            self.assertFalse(candidate50["formal_eligible"])
            self.assertEqual(rerank100["status"], "pending")


if __name__ == "__main__":
    unittest.main()
