from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
BENCHMARKS = ROOT / "benchmarks"
for path in (ROOT, BENCHMARKS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_support import _query_features


class QueryAnchorExtractionTests(unittest.TestCase):
    def test_extracts_code_metric_and_temporal_anchors(self) -> None:
        features = _query_features(
            "What changed in base/benchmarks/run_all_benchmarks.py after April 2026 for Recall@10?"
        )

        self.assertIn("base/benchmarks/run_all_benchmarks.py", features["code_like_terms"])
        self.assertIn("recall@10", features["metric_like_terms"])
        self.assertIn("april", features["specific_temporal_terms"])
        self.assertIn("2026", features["specific_temporal_terms"])
        self.assertIn("run_all_benchmarks", features["anchor_terms"])

    def test_extracts_exact_phrase_entity_preference_and_cjk_anchor(self) -> None:
        features = _query_features(
            '昨天 Alice said she prefers oat milk; what was the exact phrase "stable routing"?'
        )

        self.assertIn("stable routing", features["phrases"])
        self.assertIn("alice", features["entities"])
        self.assertIn("prefers", features["anchor_terms"])
        self.assertIn("oat", features["anchor_terms"])
        self.assertIn("昨天", features["cjk_terms"])


if __name__ == "__main__":
    unittest.main()
