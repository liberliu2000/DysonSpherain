from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_formal_evidence_gap_report import build_report, write_markdown
from run_leave_one_benchmark_out import build_protocol, write_policy_stubs


class FormalEvidenceGapReportTests(unittest.TestCase):
    def test_gap_report_counts_available_and_pending_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baselines = root / "baselines.json"
            ablations = root / "ablations.json"
            lobo = root / "lobo.json"
            paired = root / "paired.json"
            pareto = root / "pareto.json"
            baselines.write_text(
                json.dumps(
                    {
                        "records": [
                            {"benchmark": "locomo", "baseline": "dense_only_minilm", "status": "available"},
                            {"benchmark": "locomo", "baseline": "bm25", "status": "pending"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            ablations.write_text(json.dumps({"records": [{"benchmark": "locomo", "ablation": "no_safe_fusion", "status": "pending"}]}), encoding="utf-8")
            lobo.write_text(json.dumps({"rows": [{"held_out": "locomo", "train_benchmarks": ["knowme"], "status": "pending"}]}), encoding="utf-8")
            paired.write_text(json.dumps({"reports": [{"status": "available"}]}), encoding="utf-8")
            pareto.write_text(
                json.dumps(
                    {
                        "records": [{"status": "available"}, {"status": "oversized_or_unreadable"}],
                        "sweep_status": [
                            {"sweep": "candidate_top50", "status": "available", "formal_eligible": True},
                            {"sweep": "rerank_top20", "status": "smoke_available", "formal_eligible": False},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(
                baselines_path=baselines,
                ablations_path=ablations,
                lobo_path=lobo,
                paired_delta_path=paired,
                pareto_path=pareto,
            )

            self.assertEqual(report["baselines"]["available"], 1)
            self.assertEqual(report["baselines"]["pending"], 1)
            self.assertEqual(report["ablations"]["pending"], 1)
            self.assertEqual(report["leave_one_benchmark_out"]["pending"], 1)
            self.assertEqual(report["statistics"]["paired_delta_available"], 1)
            self.assertEqual(report["efficiency"]["available_records"], 1)
            self.assertEqual(report["efficiency"]["formal_sweep_available"], 1)
            self.assertEqual(report["efficiency"]["expected_sweeps"], 2)

    def test_write_markdown_lists_pending_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gap.md"
            report = {
                "baselines": {"available": 0, "pending": 1, "total": 1, "pending_items": [{"benchmark": "locomo", "baseline": "bm25", "status": "pending"}]},
                "ablations": {"available": 0, "pending": 0, "total": 0, "pending_items": []},
                "leave_one_benchmark_out": {"available": 0, "pending": 0, "total": 0, "pending_items": []},
                "statistics": {"paired_delta_reports": 0, "paired_delta_available": 0},
                "efficiency": {"records": 0, "available_records": 0},
            }

            write_markdown(report, path)

            text = path.read_text(encoding="utf-8")
            self.assertIn("Formal Evidence Gap Report", text)
            self.assertIn("`locomo` `bm25`", text)

    def test_lobo_protocol_binds_explicit_heldout_artifact_without_fabricating_others(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_path = root / "locomo_heldout" / "merged_metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "benchmark": "locomo",
                        "total_question_count": 2,
                        "wall_clock_elapsed_seconds": 12.5,
                        "quality_guardrail_status": "passed",
                        "metrics": {"recall_any@10": 0.9, "ndcg_any@10": 0.8},
                        "candidate_recall_summary": {"candidate_recall@100": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                rows = build_protocol(heldout_results={"locomo": metrics_path})
                write_policy_stubs(rows)
            finally:
                os.chdir(previous)

            locomo = next(row for row in rows if row["held_out"] == "locomo")
            clonemem = next(row for row in rows if row["held_out"] == "clonemem")
            self.assertEqual(locomo["status"], "available")
            self.assertEqual(locomo["metrics"]["recall_any@10"], 0.9)
            self.assertEqual(clonemem["status"], "pending")
            self.assertTrue((root / "artifacts" / "lobo" / "locomo_heldout_metrics.json").exists())
            policy = json.loads((root / "artifacts" / "lobo" / "route_policy_train_longmemeval_knowme_clonemem.json").read_text(encoding="utf-8"))
            self.assertEqual(policy["held_out"], "locomo")
            self.assertIn("config_hash", policy)


if __name__ == "__main__":
    unittest.main()
