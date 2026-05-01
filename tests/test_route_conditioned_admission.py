from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
BENCHMARKS = ROOT / "benchmarks"
for path in (ROOT, BENCHMARKS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_support import (
    _build_query_decomposition,
    _build_retrieval_side_index,
    _is_parent_anchor_term,
    _entity_source_candidates,
    _exact_phrase_source_candidates,
    _parent_session_source_candidates,
    _query_features,
    merge_candidate_sources,
)


def _record(source_id: str, text: str, *, order_index: int = 0, parent_id: str = "doc-1") -> dict[str, object]:
    features = _query_features(text)
    return {
        "source_id": source_id,
        "source_segment_id": source_id,
        "source_doc_id": parent_id,
        "sample_id": parent_id,
        "conversation_id": parent_id,
        "session_id": parent_id,
        "speaker_id": "",
        "timestamp": "",
        "benchmark_name": "clonemem",
        "text": text,
        "normalized_text": features["normalized_query"],
        "token_list": list(features["token_list"]),
        "entity_terms": list(features["entities"]),
        "temporal_terms": list(features["temporal_terms"]),
        "specific_temporal_terms": list(features["specific_temporal_terms"]),
        "text_hash": f"hash-{source_id}",
        "order_index": order_index,
    }


class RouteConditionedAdmissionTests(unittest.TestCase):
    def test_entity_and_exact_routes_admit_non_dense_candidates(self) -> None:
        source_records = {
            "dense-seed": _record("dense-seed", "A generic memory about coffee.", order_index=0),
            "entity-hit": _record("entity-hit", "Alice prefers oat milk for stable routing reviews.", order_index=1),
            "phrase-hit": _record("phrase-hit", 'The exact phrase was "stable routing".', order_index=2),
        }
        side_index = _build_retrieval_side_index(
            benchmark_name="knowme",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "route-admission"},
        )
        query = 'What did Alice say about the exact phrase "stable routing"?'
        features = _query_features(query)
        dense = {
            "dense-seed": {
                **source_records["dense-seed"],
                "source_retrievers": ["dense"],
                "source_chunk_ids": [],
                "best_chunk_id": "",
                "dense_score": 0.92,
            }
        }

        entity = _entity_source_candidates(
            benchmark_name="knowme",
            query_features=features,
            source_records=source_records,
            side_index=side_index,
            seed_candidates=dense,
            limit=10,
        )
        exact = _exact_phrase_source_candidates(
            query=query,
            benchmark_name="knowme",
            query_features=features,
            source_records=source_records,
            side_index=side_index,
            limit=10,
        )

        self.assertIn("entity-hit", entity)
        self.assertIn("phrase-hit", exact)
        self.assertNotIn("entity-hit", dense)
        self.assertNotIn("phrase-hit", dense)

    def test_parent_route_admits_child_segment_from_seed_parent(self) -> None:
        source_records = {
            "seg-1": _record("seg-1", "Alice discussed a generic route issue.", order_index=0),
            "seg-2": _record("seg-2", "Alice fixed stable routing after the Recall@10 audit.", order_index=1),
        }
        side_index = _build_retrieval_side_index(
            benchmark_name="clonemem",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "parent-admission"},
        )
        query = "What did Alice fix after the Recall@10 audit?"
        features = _query_features(query)
        seed = merge_candidate_sources(
            {
                "seg-1": {
                    **source_records["seg-1"],
                    "source_retrievers": ["dense"],
                    "source_chunk_ids": [],
                    "best_chunk_id": "",
                    "dense_score": 0.9,
                }
            }
        )

        parent_candidates, audit = _parent_session_source_candidates(
            benchmark_name="clonemem",
            query=query,
            query_features=features,
            decomposition=_build_query_decomposition(query, query_features=features),
            seed_candidates=seed,
            source_records=source_records,
            side_index=side_index,
            parent_top_k=3,
            parent_expand_segments=3,
            parent_window_radius=2,
        )

        self.assertIn("seg-2", parent_candidates)
        self.assertGreater(float(parent_candidates["seg-2"]["parent_score"]), 0.0)
        self.assertGreaterEqual(int(audit["selected_parent_count"]), 1)

    def test_clonemem_parent_timestamp_sibling_expansion_is_guarded(self) -> None:
        source_records = {
            "anchor": _record("anchor", "Alice deleted the family spreadsheet after dinner.", order_index=10),
            "distractor": _record("distractor", "Alice reviewed the spreadsheet deletion checklist.", order_index=12),
            "sibling": _record("sibling", "Mei felt relieved and agreed to walk together.", order_index=80),
        }
        source_records["anchor"]["timestamp"] = "2022-09-05T20:15:00"
        source_records["distractor"]["timestamp"] = "2022-10-01T09:00:00"
        source_records["sibling"]["timestamp"] = "2022-09-05T20:15:00"
        side_index = _build_retrieval_side_index(
            benchmark_name="clonemem",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "timestamp-sibling"},
        )
        query = "What changed after Alice deleted the family spreadsheet?"
        features = _query_features(query)
        seed = merge_candidate_sources(
            {
                "anchor": {
                    **source_records["anchor"],
                    "source_retrievers": ["dense"],
                    "source_chunk_ids": [],
                    "best_chunk_id": "",
                    "dense_score": 0.9,
                }
            }
        )

        parent_candidates, audit = _parent_session_source_candidates(
            benchmark_name="clonemem",
            query=query,
            query_features=features,
            decomposition=_build_query_decomposition(query, query_features=features),
            seed_candidates=seed,
            source_records=source_records,
            side_index=side_index,
            parent_top_k=3,
            parent_expand_segments=1,
            parent_window_radius=1,
            parent_timestamp_sibling_expansion_enabled=True,
            parent_timestamp_sibling_expansion_cap=1,
        )

        self.assertIn("sibling", parent_candidates)
        self.assertEqual(audit["timestamp_sibling_selected_count"], 1)

    def test_clonemem_strict_parent_anchor_noise_filter_removes_generic_terms(self) -> None:
        from benchmark_support import CLONEMEM_PARENT_ANCHOR_EXTRA_NOISE_TERMS

        self.assertTrue(_is_parent_anchor_term("workshop"))
        self.assertTrue(_is_parent_anchor_term("legacy"))
        self.assertFalse(
            _is_parent_anchor_term(
                "like",
                extra_noise_terms=CLONEMEM_PARENT_ANCHOR_EXTRA_NOISE_TERMS,
            )
        )
        self.assertFalse(
            _is_parent_anchor_term(
                "always",
                extra_noise_terms=CLONEMEM_PARENT_ANCHOR_EXTRA_NOISE_TERMS,
            )
        )


if __name__ == "__main__":
    unittest.main()
