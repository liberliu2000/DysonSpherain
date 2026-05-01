from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from analyze_clonemem_lexical_interference import analyze


def write_candidate(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"queries": rows}), encoding="utf-8")


class CloneMemLexicalInterferenceTests(unittest.TestCase):
    def test_analyze_detects_ranking_change_without_candidate_recall_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_path = root / "default.json"
            variant_path = root / "variant.json"
            base_row = {
                "query_id": "q1",
                "query_text": "question",
                "candidate_recall@100": 1.0,
                "final_recall@10": 0.0,
                "final_ndcg@10": 0.0,
                "failure_type": "candidate100_hit_final10_miss",
                "channel_stats": {"lexical_sparse": {"candidate_count": 10, "gold_hit": False}},
                "parent_audit": {"selected_child_anchors": [{"source_id": "a", "matched_anchor_terms": ["weak"]}]},
                "gold_segment_ids": ["gold"],
            }
            variant_row = {
                **base_row,
                "final_recall@10": 1.0,
                "final_ndcg@10": 0.5,
                "failure_type": "ok",
                "channel_stats": {"lexical_sparse": {"candidate_count": 0, "gold_hit": False}},
                "parent_audit": {"selected_child_anchors": [{"source_id": "gold", "matched_anchor_terms": ["strong"]}]},
            }
            write_candidate(default_path, [base_row])
            write_candidate(variant_path, [variant_row])

            summary = analyze([default_path], [variant_path])

            self.assertEqual(summary["question_count"], 1)
            self.assertEqual(summary["label_counts"]["ranking_changed_without_candidate_recall_change"], 1)
            self.assertEqual(summary["label_counts"]["parent_anchor_changed"], 1)
            self.assertEqual(summary["label_counts"]["lexical_removed"], 1)


if __name__ == "__main__":
    unittest.main()
