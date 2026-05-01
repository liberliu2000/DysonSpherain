from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.consolidate_phase4_diagnostics import (  # noqa: E402
    consolidate_candidate_recall_files,
    find_candidate_recall_files,
    write_outputs,
)


class Phase4DiagnosticConsolidationTests(unittest.TestCase):
    def _write_candidate_report(self, root: Path) -> Path:
        path = root / "run" / "reports" / "diagnostics" / "clonemem_candidate_recall.json"
        path.parent.mkdir(parents=True)
        payload = {
            "benchmark_name": "clonemem",
            "queries": [
                {
                    "query_id": "q_dense_lost",
                    "query_text": "Which section mentioned metric_alpha?",
                    "gold_evidence_ids": ["e1"],
                    "gold_segment_ids": ["s1"],
                    "dense_hit@100": True,
                    "fused_hit@100": False,
                    "dense_gold_rank": 4,
                    "fused_gold_rank": None,
                    "failure_type": "gold_missing_from_candidate_pool",
                    "channel_stats": {"dense_semantic": {"gold_hit": True}},
                },
                {
                    "query_id": "q_rerank_drop",
                    "query_text": "Where was the setting changed?",
                    "gold_evidence_ids": ["e2"],
                    "gold_segment_ids": ["s2"],
                    "dense_hit@100": True,
                    "fused_hit@100": True,
                    "gold_rank_before_rerank": 8,
                    "gold_rank_after_rerank": 140,
                    "failure_type": "reranker_dropped_gold",
                },
                {
                    "query_id": "q_lexical",
                    "query_text": "Which exact metric changed?",
                    "dense_hit@100": True,
                    "fused_hit@100": True,
                    "gold_rank_before_rerank": 42,
                    "gold_rank_after_rerank": None,
                    "failure_type": "lexical_miss",
                },
                {
                    "query_id": "q_ok",
                    "query_text": "Who owns the task?",
                    "dense_hit@100": True,
                    "fused_hit@100": True,
                    "gold_rank_before_rerank": 1,
                    "gold_rank_after_rerank": 1,
                    "failure_type": "ok",
                },
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_finds_candidate_recall_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_candidate_report(root)

            files = find_candidate_recall_files([root])

            self.assertEqual(files, [path.resolve()])

    def test_consolidates_dense_preservation_and_reranker_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_candidate_report(Path(tmp))

            summary = consolidate_candidate_recall_files([path])

            self.assertEqual(summary["scanned_files"], 1)
            self.assertEqual(summary["scanned_queries"], 4)
            self.assertEqual(len(summary["fusion_dense_preservation_violations"]), 1)
            self.assertEqual(summary["fusion_dense_preservation_violations"][0]["query_id"], "q_dense_lost")
            self.assertEqual(len(summary["reranker_dropped_gold_examples"]), 2)
            self.assertEqual(summary["reranker_dropped_gold_examples"][0]["query_id"], "q_rerank_drop")
            clonemem = summary["benchmark_failures"]["clonemem"]
            self.assertEqual(len(clonemem["reranker_dropped_gold"]), 1)
            self.assertEqual(len(clonemem["lexical_miss"]), 1)

    def test_writes_jsonl_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_candidate_report(root)
            summary = consolidate_candidate_recall_files([path])

            outputs = write_outputs(summary, root / "diagnostics", root / "phase4.md")

            fusion_lines = Path(outputs["fusion_dense_preservation_violations"]).read_text(encoding="utf-8").splitlines()
            reranker_lines = Path(outputs["reranker_dropped_gold_examples"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(fusion_lines), 1)
            self.assertEqual(len(reranker_lines), 2)
            clonemem_reranker = Path(outputs["clonemem_reranker_dropped_gold_examples"]).read_text(
                encoding="utf-8"
            ).splitlines()
            clonemem_lexical = Path(outputs["clonemem_lexical_miss_examples"]).read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(clonemem_reranker), 1)
            self.assertEqual(len(clonemem_lexical), 1)
            report = Path(outputs["report"]).read_text(encoding="utf-8")
            self.assertIn("dense preservation violations: 1", report)
            self.assertIn("by benchmark", report)
            self.assertIn("clonemem", report)
            self.assertIn("Phase 5 Benchmark-Specific Diagnostics", report)


if __name__ == "__main__":
    unittest.main()
