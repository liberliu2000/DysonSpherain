from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.experiment_registry import (
    compare_runs,
    ingest_artifacts,
    latest_run,
    load_registry,
    regression_explanation,
    write_compare_report,
    write_regression_report,
)


def write_metrics(path: Path, **overrides: object) -> None:
    payload = {
        "benchmark": "clonemem",
        "total_question_count": 2374,
        "wall_clock_elapsed_seconds": 100.0,
        "embedding_info": {
            "embedding_provider": "sentence_transformer",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "fallback_in_use": False,
        },
        "metrics": {
            "recall_frac@10": 0.2,
            "recall_any@10": 0.3,
            "ndcg_any@10": 0.4,
            "candidate_recall@100": 0.5,
        },
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ExperimentRegistryTests(unittest.TestCase):
    def test_ingest_loads_metrics_from_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_path = root / "runs" / "clonemem" / "merged_metrics.json"
            write_metrics(metrics_path)

            ingested = ingest_artifacts(root / "runs", base_dir=root, project="DysonSpherain")
            runs = load_registry(root)

            self.assertEqual(len(ingested), 1)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].dataset, "clonemem")
            self.assertEqual(runs[0].metrics["recall_frac@10"], 0.2)
            self.assertEqual(runs[0].question_count, 2374)
            self.assertFalse(runs[0].fallback_in_use)
            self.assertTrue((root / "reports" / "artifact_registry_summary.md").exists())

    def test_latest_prefers_full_nonfallback_over_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "run_a_smoke" / "clonemem" / "metrics.json", total_question_count=20)
            write_metrics(root / "run_b_full" / "clonemem" / "merged_metrics.json", total_question_count=2374)
            ingest_artifacts(root, base_dir=root, project="DysonSpherain")

            latest = latest_run(load_registry(root), project="DysonSpherain", dataset="CloneMem")

            self.assertIsNotNone(latest)
            self.assertEqual(latest.run_type, "full")
            self.assertEqual(latest.question_count, 2374)

    def test_compare_warns_on_fallback_and_run_type_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "full" / "clonemem" / "metrics.json", total_question_count=2374)
            write_metrics(
                root / "smoke" / "clonemem" / "metrics.json",
                total_question_count=20,
                embedding_info={
                    "embedding_provider": "local_hash",
                    "embedding_model": "local_hash",
                    "fallback_in_use": True,
                },
            )
            ingest_artifacts(root, base_dir=root, project="DysonSpherain")
            runs = sorted(load_registry(root), key=lambda run: run.run_type)
            comparison = compare_runs(runs[0], runs[1])

            self.assertIn("different_run_type", comparison["warnings"])
            self.assertIn("different_fallback_in_use", comparison["warnings"])

    def test_regression_explanation_uses_metric_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "old" / "clonemem" / "metrics.json", metrics={"recall_frac@10": 0.3})
            write_metrics(root / "new" / "clonemem" / "metrics.json", metrics={"recall_frac@10": 0.2})
            ingest_artifacts(root, base_dir=root, project="DysonSpherain")

            explanation = regression_explanation(load_registry(root), project="DysonSpherain", dataset="clonemem")

            self.assertEqual(explanation["status"], "regression_detected")
            self.assertEqual(explanation["regressions"][0]["metric"], "recall_frac@10")

    def test_reports_are_generated_from_stored_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "old" / "clonemem" / "metrics.json", metrics={"recall_frac@10": 0.3})
            write_metrics(root / "new" / "clonemem" / "metrics.json", metrics={"recall_frac@10": 0.2})
            ingest_artifacts(root, base_dir=root, project="DysonSpherain")
            runs = load_registry(root)
            comparison = compare_runs(runs[0], runs[1])
            explanation = regression_explanation(runs, project="DysonSpherain", dataset="clonemem")

            compare_path = write_compare_report(root, comparison)
            regression_path = write_regression_report(root, explanation)

            self.assertIn("Benchmark Run Compare", compare_path.read_text(encoding="utf-8"))
            self.assertIn("Regression Explanation", regression_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
