from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
BENCHMARKS = ROOT / "benchmarks"
for path in (ROOT, BENCHMARKS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_support import _clonemem_lexical_anchor_gate, _rank_preserved_rerank_score, _rerank_score


class RerankerGuardTests(unittest.TestCase):
    def test_locomo_broad_rank_floor_protects_top_broad_candidate(self) -> None:
        candidate = {
            "source_id": "seg-1",
            "broad_rank": 3,
            "dense_score": 0.2,
            "bm25_score": 0.1,
            "rerank_score": 0.0,
        }

        guarded = _rank_preserved_rerank_score("locomo", candidate, 0.01)

        self.assertGreaterEqual(guarded, 0.43)
        self.assertGreater(guarded, 0.01)

    def test_rank_floor_does_not_apply_to_longmemeval_guard_benchmark(self) -> None:
        candidate = {
            "source_id": "seg-1",
            "broad_rank": 3,
            "dense_score": 0.2,
            "bm25_score": 0.1,
            "rerank_score": 0.0,
        }

        self.assertEqual(_rank_preserved_rerank_score("longmemeval", candidate, 0.01), 0.01)

    def test_rank_floor_can_be_disabled(self) -> None:
        candidate = {
            "source_id": "seg-1",
            "broad_rank": 1,
            "support_count": 5,
        }

        self.assertEqual(_rank_preserved_rerank_score("locomo", candidate, 0.2, enabled=False), 0.2)

    def test_clonemem_dense_anchor_guard_is_narrow_and_configurable(self) -> None:
        candidate = {
            "source_id": "seg-1",
            "broad_rank": 4,
            "support_count": 3,
            "dense_score": 0.61,
            "broad_score": 0.8,
            "decomposition_score": 0.4,
        }

        guarded = _rank_preserved_rerank_score(
            "clonemem",
            candidate,
            0.35,
            policy={"clonemem_dense_anchor_rerank_guard_enabled": True},
        )

        self.assertGreater(guarded, 0.6)
        self.assertTrue(candidate["clonemem_dense_anchor_guard_applied"])
        self.assertEqual(
            _rank_preserved_rerank_score(
                "clonemem",
                dict(candidate),
                0.35,
                policy={"clonemem_dense_anchor_rerank_guard_enabled": False},
            ),
            0.35,
        )

    def test_clonemem_dense_anchor_guard_requires_dense_anchor(self) -> None:
        candidate = {
            "source_id": "seg-1",
            "broad_rank": 2,
            "support_count": 8,
            "dense_score": 0.2,
            "parent_score": 0.9,
            "exact_phrase_score": 0.9,
        }

        self.assertEqual(_rank_preserved_rerank_score("clonemem", candidate, 0.35), 0.35)

    def test_clonemem_evidence_blend_rerank_is_default_off_and_configurable(self) -> None:
        candidate = {
            "source_id": "seg-1",
            "broad_rank": 8,
            "dense_score": 0.55,
            "semantic_score": 0.55,
            "task_score": 0.1,
            "broad_score": 0.65,
            "bm25_score": 0.2,
            "exact_phrase_score": 0.3,
            "parent_score": 0.4,
            "decomposition_score": 0.2,
            "support_count": 3,
        }

        default_score = _rerank_score("clonemem", candidate)
        blended_score = _rerank_score(
            "clonemem",
            candidate,
            policy={"clonemem_evidence_blend_rerank_enabled": True, "clonemem_evidence_blend_rerank_alpha": 0.35},
        )

        self.assertNotEqual(default_score, blended_score)
        self.assertEqual(_rerank_score("longmemeval", candidate, policy={"clonemem_evidence_blend_rerank_enabled": True}), _rerank_score("longmemeval", candidate))

    def test_clonemem_evidence_blend_protects_early_broad_ranks(self) -> None:
        candidate = {
            "source_id": "seg-1",
            "broad_rank": 3,
            "dense_score": 0.55,
            "semantic_score": 0.55,
            "task_score": 0.1,
            "broad_score": 0.65,
            "bm25_score": 0.2,
            "exact_phrase_score": 0.3,
            "parent_score": 0.4,
            "decomposition_score": 0.2,
            "support_count": 3,
        }

        self.assertEqual(
            _rerank_score(
                "clonemem",
                candidate,
                policy={"clonemem_evidence_blend_rerank_enabled": True, "clonemem_evidence_blend_rerank_alpha": 0.35},
            ),
            _rerank_score("clonemem", candidate),
        )

    def test_clonemem_lexical_anchor_gate_is_default_off_and_narrow(self) -> None:
        weak_lexical = {
            "source_id": "seg-weak",
            "bm25_score": 0.9,
            "dense_score": 0.05,
            "exact_phrase_score": 0.0,
            "entity_score": 0.0,
            "temporal_score": 0.0,
            "parent_score": 0.0,
            "decomposition_score": 0.0,
            "support_count": 1,
        }
        self.assertEqual(_clonemem_lexical_anchor_gate(weak_lexical, policy={}), (1.0, ""))

        factor, reason = _clonemem_lexical_anchor_gate(
            weak_lexical,
            policy={"clonemem_lexical_anchor_gate_enabled": True},
        )
        self.assertEqual(factor, 0.35)
        self.assertEqual(reason, "weak_lexical_anchor_support")
        self.assertEqual(
            _clonemem_lexical_anchor_gate(
                weak_lexical,
                policy={
                    "clonemem_lexical_anchor_gate_enabled": True,
                    "clonemem_lexical_anchor_gate_protected_top_k": 2,
                },
                lexical_rank=2,
            ),
            (1.0, ""),
        )

        anchored = dict(weak_lexical, dense_score=0.41)
        self.assertEqual(
            _clonemem_lexical_anchor_gate(anchored, policy={"clonemem_lexical_anchor_gate_enabled": True}),
            (1.0, ""),
        )

    def test_clonemem_lexical_anchor_gate_reduces_only_bm25_component(self) -> None:
        candidate = {
            "source_id": "seg-weak",
            "broad_rank": 9,
            "dense_score": 0.05,
            "semantic_score": 0.05,
            "task_score": 0.0,
            "broad_score": 0.35,
            "bm25_score": 0.9,
            "support_count": 1,
        }
        default_score = _rerank_score("clonemem", candidate)
        gated_score = _rerank_score("clonemem", dict(candidate, clonemem_lexical_anchor_gate_factor=0.35))
        self.assertLess(gated_score, default_score)
        self.assertEqual(_rerank_score("knowme", dict(candidate, clonemem_lexical_anchor_gate_factor=0.35)), _rerank_score("knowme", candidate))


if __name__ == "__main__":
    unittest.main()
