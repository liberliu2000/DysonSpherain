from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from compute_confidence_intervals import (
    MetricCI,
    StatisticsReport,
    bootstrap_ci,
    compute_paired_delta_report,
    compute_report,
    tex_run_label,
    write_paired_delta_markdown,
    write_tex,
)


class ConfidenceIntervalTests(unittest.TestCase):
    def test_bootstrap_ci_is_deterministic(self) -> None:
        first = bootstrap_ci([0.0, 1.0, 1.0, 0.0], resamples=100, seed=7)
        second = bootstrap_ci([0.0, 1.0, 1.0, 0.0], resamples=100, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.5)

    def test_compute_report_from_per_question_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "question_results": [
                            {"recall_frac@10": 1.0, "ndcg@10": 0.5},
                            {"recall_frac@10": 0.0, "ndcg@10": 0.0},
                            {"recall_frac@10": 1.0, "ndcg@10": 1.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = compute_report(path, metrics=["recall_frac@10"], resamples=100, seed=3)

            self.assertEqual(report.status, "available")
            self.assertEqual(report.metric_cis[0].metric, "recall_frac@10")
            self.assertEqual(report.metric_cis[0].n, 3)

    def test_compute_report_without_rows_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps({"metrics": {"recall_frac@10": 0.5}}), encoding="utf-8")

            report = compute_report(path)

            self.assertEqual(report.status, "pending")
            self.assertIn("no per-question rows", report.warnings[0])

    def test_tex_run_label_uses_compact_artifact_stem(self) -> None:
        label = tex_run_label("artifacts/statistics/compact_longmemeval_full.json")

        self.assertEqual(label, "longmemeval\\_full")
        self.assertNotEqual(label, "statistics")

    def test_write_tex_does_not_use_parent_directory_as_run_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "statistics" / "compact_locomo_full.json"
            report = StatisticsReport(
                status="available",
                metrics_path=str(metrics_path),
                sample_unit="question",
                resamples=10,
                random_seed=3,
                metric_cis=[
                    MetricCI(
                        metric="recall_frac@10",
                        n=2,
                        mean=0.5,
                        ci_low=0.25,
                        ci_high=0.75,
                    )
                ],
                warnings=[],
            )
            out = Path(tmp) / "table.tex"

            write_tex([report], out)

            tex = out.read_text(encoding="utf-8")
            self.assertIn("locomo\\_full & recall\\_frac@10", tex)
            self.assertNotIn("statistics & recall\\_frac@10", tex)

    def test_paired_delta_report_uses_shared_question_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a_path = Path(tmp) / "a.json"
            b_path = Path(tmp) / "b.json"
            a_path.write_text(
                json.dumps(
                    {
                        "question_results": [
                            {"question_id": "q1", "recall_any@10": 0.0},
                            {"question_id": "q2", "recall_any@10": 1.0},
                            {"question_id": "q3", "recall_any@10": 1.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            b_path.write_text(
                json.dumps(
                    {
                        "question_results": [
                            {"question_id": "q3", "recall_any@10": 1.0},
                            {"question_id": "q1", "recall_any@10": 1.0},
                            {"question_id": "q2", "recall_any@10": 0.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = compute_paired_delta_report(a_path, b_path, metrics=["recall_any@10"], resamples=100, seed=5)

            self.assertEqual(report.status, "available")
            delta = report.metric_deltas[0]
            self.assertEqual(delta.n, 3)
            self.assertAlmostEqual(delta.mean_delta, 0.0)
            self.assertEqual(delta.wins, 1)
            self.assertEqual(delta.ties, 1)
            self.assertEqual(delta.losses, 1)

    def test_paired_delta_auto_key_combines_sample_and_question_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a_path = Path(tmp) / "a.json"
            b_path = Path(tmp) / "b.json"
            a_path.write_text(
                json.dumps(
                    {
                        "question_results": [
                            {"sample_id": "s1", "question_id": "q1", "recall_any@10": 0.0},
                            {"sample_id": "s2", "question_id": "q1", "recall_any@10": 1.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            b_path.write_text(
                json.dumps(
                    {
                        "question_results": [
                            {"sample_id": "s2", "question_id": "q1", "recall_any@10": 1.0},
                            {"sample_id": "s1", "question_id": "q1", "recall_any@10": 1.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = compute_paired_delta_report(a_path, b_path, metrics=["recall_any@10"], resamples=20, seed=5)

            delta = report.metric_deltas[0]
            self.assertEqual(delta.n, 2)
            self.assertEqual(delta.wins, 1)
            self.assertEqual(delta.ties, 1)

    def test_paired_delta_preserves_duplicate_row_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a_path = Path(tmp) / "a.json"
            b_path = Path(tmp) / "b.json"
            a_path.write_text(
                json.dumps(
                    {
                        "question_results": [
                            {"question_id": "q1", "recall_any@10": 0.0},
                            {"question_id": "q1", "recall_any@10": 1.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            b_path.write_text(
                json.dumps(
                    {
                        "question_results": [
                            {"question_id": "q1", "recall_any@10": 1.0},
                            {"question_id": "q1", "recall_any@10": 1.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = compute_paired_delta_report(a_path, b_path, metrics=["recall_any@10"], resamples=20, seed=5)

            delta = report.metric_deltas[0]
            self.assertEqual(delta.n, 2)
            self.assertEqual(delta.wins, 1)
            self.assertEqual(delta.ties, 1)

    def test_write_paired_delta_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a_path = Path(tmp) / "a.json"
            b_path = Path(tmp) / "b.json"
            a_path.write_text(json.dumps({"question_results": [{"question_id": "q1", "ndcg@10": 0.2}]}), encoding="utf-8")
            b_path.write_text(json.dumps({"question_results": [{"question_id": "q1", "ndcg@10": 0.5}]}), encoding="utf-8")
            report = compute_paired_delta_report(a_path, b_path, metrics=["ndcg@10"], resamples=20, seed=1)
            out = Path(tmp) / "paired.md"

            write_paired_delta_markdown([report], out)

            text = out.read_text(encoding="utf-8")
            self.assertIn("B minus A", text)
            self.assertIn("ndcg@10", text)
            self.assertIn("0.300000", text)


if __name__ == "__main__":
    unittest.main()
