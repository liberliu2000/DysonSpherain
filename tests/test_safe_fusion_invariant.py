from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
BENCHMARKS = ROOT / "benchmarks"
for path in (ROOT, BENCHMARKS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_support import _apply_clonemem_evidence_consensus_admission, _fuse_candidate_sources, merge_candidate_sources


def _candidate(source_id: str, *, dense: float = 0.0, lexical: float = 0.0, parent: str = "doc") -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_segment_id": source_id,
        "source_doc_id": parent,
        "sample_id": parent,
        "conversation_id": parent,
        "session_id": parent,
        "turn_id": "",
        "speaker_id": "",
        "timestamp": "",
        "text": source_id,
        "normalized_text": source_id,
        "token_list": [source_id],
        "entity_terms": [],
        "temporal_terms": [],
        "specific_temporal_terms": [],
        "order_index": 0,
        "source_retrievers": ["dense"] if dense else ["bm25"],
        "source_chunk_ids": [],
        "best_chunk_id": "",
        "dense_score": dense,
        "bm25_score": lexical,
        "entity_score": 0.0,
        "temporal_score": 0.0,
        "profile_score": 0.0,
        "session_score": 0.0,
        "exact_phrase_score": 0.0,
        "speaker_score": 0.0,
        "temporal_neighbor_score": 0.0,
        "parent_score": 0.0,
        "decomposition_score": 0.0,
        "local_window_score": 0.0,
        "fusion_score": 0.0,
        "broad_score": 0.0,
        "rerank_score": 0.0,
        "post_inhibition_score": 0.0,
        "inhibition_penalty": 0.0,
    }


class SafeFusionInvariantTests(unittest.TestCase):
    def test_safe_fusion_retains_dense_anchor_candidates_under_parent_cap_pressure(self) -> None:
        dense = {
            "dense-1": _candidate("dense-1", dense=0.95, parent="dense-parent"),
            "dense-2": _candidate("dense-2", dense=0.9, parent="dense-parent"),
        }
        lexical = {
            f"lex-{idx}": _candidate(f"lex-{idx}", lexical=1.0 - idx * 0.01, parent="lex-parent")
            for idx in range(10)
        }
        merged = merge_candidate_sources(dense, lexical)

        final_rows, audit, _ = _fuse_candidate_sources(
            benchmark_name="clonemem",
            merged=merged,
            channel_maps={
                "dense_semantic": dense,
                "lexical_sparse": lexical,
            },
            policy={
                "fusion_method": "rrf",
                "rrf_k": 60,
                "safe_fusion_enabled": True,
                "dense_preserve_enabled": True,
                "dense_anchor_top_k": 2,
                "dense_anchor_min_keep": 2,
                "destructive_filter_guard_enabled": True,
                "duplicate_collapse_enabled": True,
                "near_duplicate_collapse_enabled": True,
                "duplicate_collapse_safe_mode": True,
                "parent_cap_after_gold_agnostic_anchor": True,
                "max_candidates_per_parent": 1,
                "final_candidate_pool_size": 4,
                "min_parent_diversity": 1,
                "channel_gating_enabled": True,
            },
        )

        final_ids = {str(row["source_id"]) for row in final_rows}
        self.assertTrue({"dense-1", "dense-2"}.issubset(final_ids))
        self.assertEqual(audit["protected_dense_candidate_count"], 2)
        self.assertGreaterEqual(audit["protected_dense_retained_count"], 2)
        self.assertFalse(audit["parent_cap_applied"])

    def test_clonemem_channel_tail_rescue_promotes_bounded_tail_candidates(self) -> None:
        lexical = {
            f"lex-{idx:03d}": _candidate(f"lex-{idx:03d}", lexical=1.0 - idx * 0.001, parent=f"doc-{idx}")
            for idx in range(130)
        }
        merged = merge_candidate_sources(lexical)

        final_rows, audit, _ = _fuse_candidate_sources(
            benchmark_name="clonemem",
            merged=merged,
            channel_maps={"lexical_sparse": lexical},
            policy={
                "fusion_method": "rrf",
                "rrf_k": 60,
                "safe_fusion_enabled": True,
                "dense_preserve_enabled": False,
                "dense_anchor_top_k": 100,
                "dense_anchor_min_keep": 80,
                "destructive_filter_guard_enabled": True,
                "duplicate_collapse_enabled": True,
                "near_duplicate_collapse_enabled": True,
                "duplicate_collapse_safe_mode": True,
                "parent_cap_after_gold_agnostic_anchor": True,
                "max_candidates_per_parent": 20,
                "final_candidate_pool_size": 130,
                "min_parent_diversity": 1,
                "channel_gating_enabled": True,
                "clonemem_channel_tail_rescue_enabled": True,
                "clonemem_channel_tail_rescue_max_rank": 120,
                "clonemem_channel_tail_rescue_per_channel": 2,
                "clonemem_channel_tail_rescue_target_rank": 90,
            },
        )

        top100_ids = [str(row["source_id"]) for row in final_rows[:100]]
        self.assertIn("lex-100", top100_ids)
        self.assertIn("lex-101", top100_ids)
        self.assertEqual(audit["clonemem_channel_tail_rescue_count"], 2)
        self.assertTrue(any(row.get("clonemem_channel_tail_rescue_reason") for row in final_rows[:100]))

    def test_clonemem_evidence_consensus_admission_requires_multiple_channels(self) -> None:
        consensus = _candidate("consensus", parent="consensus-doc")
        consensus["entity_score"] = 0.5
        consensus["decomposition_score"] = 0.5
        single_channel = _candidate("single-channel", parent="single-doc")
        single_channel["entity_score"] = 0.6
        rows = [
            _candidate(f"ranked-{idx:03d}", lexical=1.0 - idx * 0.001, parent=f"ranked-doc-{idx}")
            for idx in range(130)
        ]

        final_rows, audit = _apply_clonemem_evidence_consensus_admission(
            rows,
            channel_rankings={
                "entity_aware": [single_channel, consensus],
                "query_decomposition": [consensus],
            },
            policy={
                "clonemem_evidence_consensus_admission_enabled": True,
                "clonemem_evidence_consensus_admission_max_candidates": 1,
                "clonemem_evidence_consensus_admission_min_channels": 2,
                "clonemem_evidence_consensus_admission_target_rank": 90,
            },
        )

        top100_ids = [str(row["source_id"]) for row in final_rows[:100]]
        self.assertIn("consensus", top100_ids)
        self.assertNotIn("single-channel", top100_ids)
        self.assertEqual(len(audit), 1)
        self.assertEqual(
            audit[0]["channels"],
            {"entity_aware": 2, "query_decomposition": 1},
        )

    def test_clonemem_evidence_rank_preservation_is_default_off_and_guarded(self) -> None:
        candidate = _candidate("evidence-rich", dense=0.44, lexical=0.4, parent="doc")
        candidate["broad_rank"] = 12
        candidate["broad_score"] = 0.82
        candidate["support_count"] = 5
        candidate["decomposition_score"] = 0.4

        from benchmark_support import _rank_preserved_rerank_score

        base_score = _rank_preserved_rerank_score(
            "clonemem",
            dict(candidate),
            0.42,
            policy={"clonemem_evidence_rank_preservation_enabled": False},
        )
        guarded = _rank_preserved_rerank_score(
            "clonemem",
            candidate,
            0.42,
            policy={
                "clonemem_evidence_rank_preservation_enabled": True,
                "clonemem_evidence_rank_preservation_max_rank": 20,
                "clonemem_evidence_rank_preservation_min_support": 5,
                "clonemem_evidence_rank_preservation_min_broad_score": 0.65,
                "clonemem_evidence_rank_preservation_floor": 0.68,
                "clonemem_evidence_rank_preservation_protected_top_k": 3,
            },
        )

        self.assertEqual(base_score, 0.42)
        self.assertGreater(guarded, base_score)
        self.assertTrue(candidate["clonemem_evidence_rank_preservation_applied"])

        protected_candidate = dict(candidate)
        protected_candidate["broad_rank"] = 2
        protected = _rank_preserved_rerank_score(
            "clonemem",
            protected_candidate,
            0.42,
            policy={
                "clonemem_evidence_rank_preservation_enabled": True,
                "clonemem_evidence_rank_preservation_max_rank": 20,
                "clonemem_evidence_rank_preservation_min_support": 5,
                "clonemem_evidence_rank_preservation_min_broad_score": 0.65,
                "clonemem_evidence_rank_preservation_floor": 0.68,
                "clonemem_evidence_rank_preservation_protected_top_k": 3,
            },
        )
        self.assertEqual(protected, 0.42)


if __name__ == "__main__":
    unittest.main()
