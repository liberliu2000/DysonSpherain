from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "base"
BENCHMARKS = BASE / "benchmarks"
for path in (BASE, BENCHMARKS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clonemem_benchmark import build_parent_to_segment_trace_rows


class ParentToSegmentSelectorTraceTests(unittest.TestCase):
    def test_trace_rows_are_diagnostic_only_and_ranked(self) -> None:
        trace = {
            "query_features": {"anchor_terms": ["mortgage", "shenzhen"]},
            "parent_audit": {
                "parent_candidates": [{"parent_id": "parent-a", "score": 0.9}],
                "selected_child_anchors": [
                    {
                        "parent_id": "parent-a",
                        "source_id": "seg-2",
                        "order_index": 12,
                        "anchor_priority": 0.87,
                        "direct_match": 0.72,
                        "matched_anchor_terms": ["mortgage"],
                    }
                ],
            },
        }

        rows = build_parent_to_segment_trace_rows(
            query_id="q1",
            query_text="How much is the mortgage in Shenzhen?",
            question_type="single_point_factual",
            gold_segment_ids={"seg-2"},
            broad_rows=[{"source_id": "seg-2", "rank": 17}],
            final_rows=[{"source_id": "seg-2", "rank": 12}],
            trace=trace,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["selected_segment_id"], "seg-2")
        self.assertTrue(rows[0]["is_gold_segment"])
        self.assertEqual(rows[0]["parent_rank"], 1)
        self.assertEqual(rows[0]["segment_rank_before"], 17)
        self.assertEqual(rows[0]["segment_rank_after"], 12)
        self.assertEqual(rows[0]["why_selected"], "parent_session_anchor_preselection_diagnostic_only")


if __name__ == "__main__":
    unittest.main()
