from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / "base"
BENCHMARKS_DIR = ROOT / "benchmarks"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from draft_updater import (
    load_benchmark_artifacts,
    render_multichannel_report,
    render_updated_draft,
    write_updated_outputs,
)


class DraftUpdaterTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _populate_result_root(self, root: Path) -> None:
        metric_templates = {
            "longmemeval": {
                "question_count": 500,
                "fallback_in_use": False,
                "metrics": {"session": {"recall_any@5": 0.96, "recall_any@10": 0.97, "ndcg_any@10": 0.91}},
            },
            "locomo": {
                "question_count": 1986,
                "fallback_in_use": False,
                "metrics": {"session": {"recall_frac@10": 0.9, "recall_any@10": 0.92, "ndcg_any@10": 0.76}},
            },
            "knowme": {
                "question_count": 1010,
                "fallback_in_use": False,
                "metrics": {"segment": {"recall_frac@10": 0.56, "recall_any@10": 0.58, "ndcg_any@10": 0.41}},
            },
            "clonemem": {
                "question_count": 2374,
                "fallback_in_use": False,
                "metrics": {"segment": {"recall_frac@10": 0.14, "recall_any@10": 0.35, "ndcg_any@10": 0.19}},
            },
        }
        for benchmark, metrics in metric_templates.items():
            benchmark_root = root / benchmark
            self._write_json(benchmark_root / "metrics.json", metrics)
            self._write_json(
                benchmark_root / "reports" / "integrity" / f"{benchmark}_integrity_report.json",
                {"benchmark_name": benchmark, "p0_bugs": []},
            )
            self._write_json(
                benchmark_root / "reports" / "diagnostics" / f"{benchmark}_candidate_recall.json",
                {
                    "benchmark_name": benchmark,
                    "candidate_recall@10": 0.8,
                    "candidate_recall@100": 0.9,
                    "final_recall@10": 0.7,
                },
            )
            self._write_json(
                benchmark_root / "reports" / "diagnostics" / f"{benchmark}_channel_contribution.json",
                {
                    "benchmark_name": benchmark,
                    "channels": {
                        "dense_semantic": {"gold_hit_rate": 0.7, "avg_gold_rank_when_hit": 5.0},
                        "lexical_sparse": {"gold_hit_rate": 0.6, "avg_gold_rank_when_hit": 7.0},
                    },
                },
            )
            self._write_json(
                benchmark_root / "reports" / "diagnostics" / f"{benchmark}_performance_cache.json",
                {
                    "benchmark_name": benchmark,
                    "timing_summary": {"retrieval_ms": 10.0},
                    "reuse_summary": {"total_cache_reuse_saved_ms": 100.0},
                },
            )
        self._write_json(
            root / "clonemem" / "reports" / "diagnostics" / "clonemem_oracle_retrieval.json",
            {"oracle_recall@10": 1.0},
        )
        self._write_json(
            root / "clonemem" / "reports" / "diagnostics" / "clonemem_failure_taxonomy.json",
            {"failure_type_distribution": {"dense_semantic_miss": 10}},
        )
        self._write_json(
            root / "knowme" / "reports" / "diagnostics" / "knowme_category_analysis.json",
            {"categories": {"preference query": {"question_count": 10}}},
        )

    def _populate_baseline_root_metrics_only(self, root: Path) -> None:
        metric_templates = {
            "longmemeval": {
                "question_count": 500,
                "fallback_in_use": False,
                "metrics": {"session": {"recall_any@5": 0.97, "recall_any@10": 0.98, "ndcg_any@10": 0.92}},
            },
            "locomo": {
                "question_count": 1986,
                "fallback_in_use": False,
                "metrics": {"session": {"recall_frac@10": 0.91, "recall_any@10": 0.93, "ndcg_any@10": 0.77}},
            },
            "knowme": {
                "question_count": 1010,
                "fallback_in_use": False,
                "metrics": {"segment": {"recall_frac@10": 0.57, "recall_any@10": 0.59, "ndcg_any@10": 0.42}},
            },
            "clonemem": {
                "question_count": 2374,
                "fallback_in_use": False,
                "metrics": {"segment": {"recall_frac@10": 0.15, "recall_any@10": 0.36, "ndcg_any@10": 0.20}},
            },
        }
        for benchmark, metrics in metric_templates.items():
            self._write_json(root / benchmark / "metrics.json", metrics)

    def test_missing_artifacts_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "result_root"
            root.mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                load_benchmark_artifacts(root)

    def test_updated_draft_reads_latest_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_root = Path(tmp) / "results"
            self._populate_result_root(result_root)
            draft = render_updated_draft(
                draft_source="# Old draft\nthree benchmark package\n",
                result_root=result_root,
                baseline_root=None,
            )
            self.assertIn("0.5600", draft)
            self.assertIn("0.1400", draft)
            self.assertIn("four-benchmark", draft.lower())
            self.assertIn("CloneMem", draft)

    def test_write_outputs_emits_draft_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_root = Path(tmp) / "results"
            self._populate_result_root(result_root)
            draft_path = Path(tmp) / "draft.md"
            draft_path.write_text("# draft\n", encoding="utf-8")
            draft_output = Path(tmp) / "draft_updated_multichannel.md"
            report_output = Path(tmp) / "multichannel_report.md"
            write_updated_outputs(
                draft_path=draft_path,
                draft_output=draft_output,
                report_output=report_output,
                result_root=result_root,
                baseline_root=None,
            )
            self.assertTrue(draft_output.exists())
            self.assertTrue(report_output.exists())
            self.assertIn("KnowMe", draft_output.read_text(encoding="utf-8"))
            self.assertIn("Before / After", report_output.read_text(encoding="utf-8"))

    def test_baseline_root_can_be_metrics_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_root = Path(tmp) / "results"
            baseline_root = Path(tmp) / "baseline"
            self._populate_result_root(result_root)
            self._populate_baseline_root_metrics_only(baseline_root)
            draft = render_updated_draft(
                draft_source="# Old draft\n",
                result_root=result_root,
                baseline_root=baseline_root,
            )
            self.assertIn("regressed", draft)
            self.assertIn("n/a", render_multichannel_report(result_root=result_root, baseline_root=baseline_root))


if __name__ == "__main__":
    unittest.main()
