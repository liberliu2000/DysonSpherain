from __future__ import annotations

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

from benchmark_support import (
    _apply_route_aware_channel_gating,
    _apply_duplicate_collapse,
    _build_query_decomposition,
    _build_retrieval_side_index,
    _entity_source_candidates,
    _exact_phrase_source_candidates,
    _fuse_candidate_sources,
    _lexical_source_candidates,
    _load_query_decomposition,
    _parent_session_source_candidates,
    _parent_anchor_terms,
    _profile_side_index_candidates,
    _query_decomposition_source_candidates,
    _query_features,
    _rank_preserved_rerank_score,
    _rerank_score,
    _select_parent_anchor_rows,
    _session_bundle_source_candidates,
    _should_early_exit_retrieval,
    _temporal_source_candidates,
    _temporal_neighbor_source_candidates,
    apply_inhibition_audit,
    build_candidate_recall_summary,
    build_clonemem_failure_taxonomy,
    build_knowme_category_analysis,
    merge_candidate_sources,
    rank_benchmark_sources,
)
from locomo_benchmark import build_corpus_from_sessions
from sphere_cli.config import AppConfig
from sphere_cli.storage import Storage
from sphere_cli.utils import tokenize


class FakeVectorStore:
    def __init__(self, hits_by_query: dict[str, list[dict[str, object]]] | None = None) -> None:
        self.hits_by_query = hits_by_query or {}

    def search(self, query: str, top_k: int = 8, where: dict[str, object] | None = None) -> list[dict[str, object]]:
        return list(self.hits_by_query.get(query, []))[:top_k]

    def search_objects(self, query: str, top_k: int = 8, where: dict[str, object] | None = None) -> list[dict[str, object]]:
        return []

    def info(self) -> dict[str, object]:
        return {
            "embedding_provider": "sentence_transformer",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "normalize_embeddings": True,
            "fallback_in_use": False,
            "embedding_preprocess_version": "normalize_text_for_hash_v2",
        }


class FakeStorage:
    def __init__(self, fts_rows: list[dict[str, object]] | None = None) -> None:
        self.fts_rows = list(fts_rows or [])
        self.cache: dict[tuple[str, int], dict[str, object]] = {}

    def search_chunks_fts(self, query: str, limit: int) -> list[dict[str, object]]:
        return list(self.fts_rows)[:limit]

    def search_objects_fts(self, query: str, limit: int, object_types: list[str] | None = None) -> list[dict[str, object]]:
        return []

    def fetch_objects_by_ids(self, object_ids: list[str]) -> list[dict[str, object]]:
        return []

    def get_retrieval_cache(self, query_fingerprint: str, memory_version: int) -> dict[str, object] | None:
        return self.cache.get((query_fingerprint, memory_version))

    def put_retrieval_cache(
        self,
        *,
        query_fingerprint: str,
        normalized_query: str,
        task_type: str,
        route_type: str,
        memory_version: int,
        payload: dict[str, object],
        created_at: str,
    ) -> None:
        self.cache[(query_fingerprint, memory_version)] = {"payload": dict(payload)}


class MultiChannelRetrievalTests(unittest.TestCase):
    @staticmethod
    def _record(
        source_id: str,
        text: str,
        *,
        source_doc_id: str = "doc-1",
        order_index: int = 0,
        speaker_id: str = "",
    ) -> dict[str, object]:
        query_features = _query_features(text)
        return {
            "source_id": source_id,
            "source_segment_id": source_id,
            "source_doc_id": source_doc_id,
            "sample_id": source_doc_id,
            "conversation_id": source_doc_id,
            "session_id": source_doc_id,
            "speaker_id": speaker_id,
            "timestamp": "",
            "benchmark_name": "clonemem",
            "text": text,
            "normalized_text": query_features["normalized_query"],
            "token_list": list(query_features["token_list"]),
            "entity_terms": list(query_features["entities"]),
            "temporal_terms": list(query_features["temporal_terms"]),
            "specific_temporal_terms": list(query_features["specific_temporal_terms"]),
            "text_hash": "hash-" + source_id,
            "order_index": order_index,
        }

    def test_query_decomposition_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(base_dir=Path(tmp))
            config.ensure_dirs()
            storage = Storage(config)
            storage.init_db()
            source_records = {
                "seg-1": self._record("seg-1", "Melanie prefers strong coffee in the workshop."),
            }
            side_index = _build_retrieval_side_index(
                benchmark_name="knowme",
                source_records=source_records,
                index_fingerprint={"fingerprint_hash": "fp-1"},
            )
            query_features = _query_features("What coffee does Melanie prefer?")
            first, meta_first = _load_query_decomposition(
                query="What coffee does Melanie prefer?",
                benchmark_name="knowme",
                storage=storage,
                side_index=side_index,
                query_features=query_features,
            )
            second, meta_second = _load_query_decomposition(
                query="What coffee does Melanie prefer?",
                benchmark_name="knowme",
                storage=storage,
                side_index=side_index,
                query_features=query_features,
            )
            self.assertFalse(meta_first["cache_hit"])
            self.assertTrue(meta_second["cache_hit"])
            self.assertEqual(first["evidence_type"], "preference")
            self.assertEqual(first, second)

    def test_locomo_session_corpus_uses_dialog_text_only(self) -> None:
        corpus = build_corpus_from_sessions(
            [
                {
                    "session_num": 1,
                    "session_id": "session_1",
                    "date": "1 January 2024",
                    "summary": "Nate discussed his favorite video game.",
                    "dialogs": [{"speaker": "Nate", "text": "I played a long match."}],
                }
            ],
            "session",
            sample_id="conv-test",
        )

        self.assertNotIn("Session Summary:", corpus[0]["text"])
        self.assertIn("Nate: I played a long match.", corpus[0]["text"])

    def test_parent_session_channel_expands_parent_children(self) -> None:
        source_records = {
            "seg-1": self._record("seg-1", "Melanie discussed the coffee grinder settings.", order_index=0),
            "seg-2": self._record("seg-2", "Melanie prefers strong coffee with oat milk.", order_index=1),
            "seg-3": self._record("seg-3", "Later she returned to the same recipe.", order_index=2),
        }
        side_index = _build_retrieval_side_index(
            benchmark_name="clonemem",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "fp-2"},
        )
        seed = merge_candidate_sources(
            {
                "seg-1": {
                    **source_records["seg-1"],
                    "source_retrievers": ["dense"],
                    "source_chunk_ids": [],
                    "best_chunk_id": "",
                    "dense_score": 0.92,
                }
            }
        )
        candidates, audit = _parent_session_source_candidates(
            benchmark_name="clonemem",
            query="What coffee does Melanie prefer?",
            query_features=_query_features("What coffee does Melanie prefer?"),
            decomposition=_build_query_decomposition(
                "What coffee does Melanie prefer?",
                query_features=_query_features("What coffee does Melanie prefer?"),
            ),
            seed_candidates=seed,
            source_records=source_records,
            side_index=side_index,
            parent_top_k=5,
            parent_expand_segments=3,
            parent_window_radius=2,
        )
        self.assertIn("seg-2", candidates)
        self.assertGreater(float(candidates["seg-2"]["parent_score"]), 0.0)
        self.assertGreaterEqual(int(audit["selected_parent_count"]), 1)
        self.assertGreaterEqual(int(audit["parent_window_radius"]), 2)

    def test_clonemem_parent_anchor_preselection_reports_child_anchors(self) -> None:
        source_records = {
            "seg-1": self._record("seg-1", "I talked about a generic trip with Melanie.", order_index=0),
            "seg-2": self._record("seg-2", "Melanie learned recall@10 debugging after the 2024 metric review.", order_index=1),
            "seg-3": self._record("seg-3", "A different parent note about lunch.", order_index=2),
        }
        side_index = _build_retrieval_side_index(
            benchmark_name="clonemem",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "fp-parent-anchor"},
        )
        query = "What did Melanie learn after the 2024 recall@10 metric review?"
        query_features = _query_features(query)
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
        candidates, audit = _parent_session_source_candidates(
            benchmark_name="clonemem",
            query=query,
            query_features=query_features,
            decomposition=_build_query_decomposition(query, query_features=query_features),
            seed_candidates=seed,
            source_records=source_records,
            side_index=side_index,
            parent_top_k=3,
            parent_expand_segments=3,
            parent_window_radius=1,
        )

        self.assertIn("seg-2", candidates)
        self.assertGreater(float(candidates["seg-2"]["local_window_score"]), 0.0)
        self.assertTrue(audit["selected_child_anchors"])
        self.assertIn("recall@10", query_features["metric_like_terms"])

    def test_route_aware_gating_is_deterministic_and_diagnostic(self) -> None:
        query = 'What exact coffee did Melanie prefer in the workshop?'
        query_features = _query_features(query)
        policy = {
            "dense_semantic": True,
            "lexical_sparse": True,
            "entity_aware": True,
            "temporal_anchor": True,
            "exact_phrase": True,
            "profile_side_index": True,
            "session_bundle": True,
            "temporal_neighbor": True,
            "parent_session": True,
            "query_decomposition": True,
            "route_aware_gating_enabled": True,
            "route_aware_gating_aggressiveness": "safe",
            "retrieval_min_seed_candidates": 80,
        }
        first = _apply_route_aware_channel_gating(policy, query_features, "knowme", {}, seed_state={"seed_candidate_count": 160})
        second = _apply_route_aware_channel_gating(policy, query_features, "knowme", {}, seed_state={"seed_candidate_count": 160})
        self.assertEqual(first["retrieval_policy_after_gating"], second["retrieval_policy_after_gating"])
        self.assertIn("retrieval_policy_before_gating", first)
        self.assertIn("gated_channels", first)

    def test_indexed_candidate_generation_contains_legacy_gold(self) -> None:
        source_records = {
            "seg-1": self._record("seg-1", "Melanie prefers strong coffee in March.", source_doc_id="session-a", order_index=0),
            "seg-2": self._record("seg-2", "Nora talked about tea in April.", source_doc_id="session-a", order_index=1),
            "seg-3": self._record("seg-3", "Melanie returned to the same grinder setting.", source_doc_id="session-a", order_index=2),
        }
        side_index = _build_retrieval_side_index(
            benchmark_name="knowme",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "fp-indexed-fast-path"},
        )
        query = "What coffee did Melanie prefer in March?"
        qf = _query_features(query)
        legacy_entity = _entity_source_candidates(query_features=qf, source_records=source_records, limit=5)
        indexed_entity = _entity_source_candidates(query_features=qf, source_records=source_records, side_index=side_index, limit=5)
        self.assertIn("seg-1", indexed_entity)
        self.assertEqual("seg-1" in legacy_entity, "seg-1" in indexed_entity)

        legacy_temporal = _temporal_source_candidates(benchmark_name="knowme", query_features=qf, source_records=source_records, limit=5)
        indexed_temporal = _temporal_source_candidates(benchmark_name="knowme", query_features=qf, source_records=source_records, side_index=side_index, limit=5)
        self.assertIn("seg-1", indexed_temporal)
        self.assertEqual("seg-1" in legacy_temporal, "seg-1" in indexed_temporal)

        legacy_exact = _exact_phrase_source_candidates(query=query, query_features=qf, source_records=source_records, limit=5)
        indexed_exact = _exact_phrase_source_candidates(query=query, query_features=qf, source_records=source_records, side_index=side_index, limit=5)
        self.assertIn("seg-1", indexed_exact)
        self.assertEqual("seg-1" in legacy_exact, "seg-1" in indexed_exact)

        seed = merge_candidate_sources({"seg-1": {**source_records["seg-1"], "source_retrievers": ["dense"], "dense_score": 0.9}})
        indexed_session = _session_bundle_source_candidates(seed_candidates=seed, source_records=source_records, side_index=side_index, limit=5, query_features=qf)
        self.assertIn("seg-1", indexed_session)
        parent_candidates, _ = _parent_session_source_candidates(
            benchmark_name="knowme",
            query=query,
            query_features=qf,
            decomposition=_build_query_decomposition(query, query_features=qf),
            seed_candidates=seed,
            source_records=source_records,
            side_index=side_index,
            parent_top_k=3,
            parent_expand_segments=3,
            parent_window_radius=2,
        )
        self.assertTrue(parent_candidates)

    def test_indexed_fast_path_does_not_default_to_full_scan_on_miss(self) -> None:
        source_records = {
            "seg-1": self._record("seg-1", "Melanie prefers strong coffee in March.", source_doc_id="session-a", order_index=0),
        }
        query = "What coffee did Melanie prefer?"
        qf = _query_features(query)
        incomplete_index: dict[str, object] = {}
        stale_index = {
            "token_to_source_ids": {"unrelated": ["seg-1"]},
            "entity_to_source_ids": {"nora": ["seg-1"]},
            "phrase_token_to_source_ids": {"unrelated": ["seg-1"]},
            "source_id_to_order_index": {"seg-1": 0},
        }

        self.assertIn("seg-1", _entity_source_candidates(query_features=qf, source_records=source_records, side_index=incomplete_index, limit=5))
        self.assertNotIn("seg-1", _entity_source_candidates(query_features=qf, source_records=source_records, side_index=stale_index, limit=5))
        self.assertNotIn(
            "seg-1",
            _exact_phrase_source_candidates(
                query=query,
                query_features=qf,
                source_records=source_records,
                side_index=stale_index,
                limit=5,
            ),
        )
        self.assertFalse(
            _query_decomposition_source_candidates(
                decomposition=_build_query_decomposition(query, query_features=qf),
                query=query,
                source_records=source_records,
                side_index=stale_index,
                limit=5,
            )
        )

    def test_early_exit_safety_rules(self) -> None:
        policy = {
            "retrieval_early_exit_enabled": True,
            "retrieval_min_seed_candidates": 2,
            "candidate_recall_eval_k": 2,
            "retrieval_confidence_margin": 0.12,
            "session_bundle": True,
            "temporal_neighbor": True,
            "parent_session": True,
            "query_decomposition": True,
        }
        seed = {
            "a": {"source_id": "a", "dense_score": 0.95},
            "b": {"source_id": "b", "dense_score": 0.5},
            "c": {"source_id": "c", "bm25_score": 0.4},
        }
        simple = _should_early_exit_retrieval(
            policy=policy,
            benchmark_name="locomo",
            query_features=_query_features("What exact value was recall@10?"),
            seed_candidates=seed,
            route_context={},
            elapsed_ms=500.0,
        )
        self.assertTrue(simple["triggered"])
        self.assertNotIn("dense_semantic", simple["skipped_channels"])
        temporal = _should_early_exit_retrieval(
            policy=policy,
            benchmark_name="knowme",
            query_features=_query_features("What did Melanie prefer after March?"),
            seed_candidates=seed,
            route_context={},
            elapsed_ms=500.0,
        )
        self.assertFalse(temporal["triggered"])
        clone = _should_early_exit_retrieval(
            policy=policy,
            benchmark_name="clonemem",
            query_features=_query_features("What exact coffee was preferred?"),
            seed_candidates=seed,
            route_context={},
            elapsed_ms=500.0,
        )
        self.assertFalse(clone["triggered"])

    def test_locomo_broad_rank_floor_preserves_broad_top10(self) -> None:
        candidate = {
            "broad_rank": 9,
            "support_count": 5,
            "session_score": 0.0,
        }

        self.assertGreater(_rank_preserved_rerank_score("locomo", candidate, 0.41), 0.43)
        self.assertEqual(_rank_preserved_rerank_score("longmemeval", candidate, 0.41), 0.41)

    def test_parent_session_channel_adds_local_window_candidates(self) -> None:
        query = "How did deleting the Family Health Management Spreadsheet affect things with Meifang?"
        query_features = _query_features(query)
        source_records = {
            "seg-1": self._record("seg-1", "He deleted the Family Health Management Spreadsheet after another tense argument.", order_index=10),
            "seg-2": self._record("seg-2", "He sat quietly with Meifang in the hallway after that fight.", order_index=11),
            "seg-3": self._record("seg-3", "Meifang said living is not about being afraid every day.", order_index=12),
            "seg-4": self._record("seg-4", "The spreadsheet tracked blood pressure and medication reminders.", order_index=40),
        }
        side_index = _build_retrieval_side_index(
            benchmark_name="clonemem",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "fp-2b"},
        )
        seed = merge_candidate_sources(
            {
                "seg-1": {
                    **source_records["seg-1"],
                    "source_retrievers": ["dense"],
                    "source_chunk_ids": [],
                    "best_chunk_id": "",
                    "dense_score": 0.94,
                }
            }
        )
        candidates, _ = _parent_session_source_candidates(
            benchmark_name="clonemem",
            query=query,
            query_features=query_features,
            decomposition=_build_query_decomposition(query, query_features=query_features),
            seed_candidates=seed,
            source_records=source_records,
            side_index=side_index,
            parent_top_k=4,
            parent_expand_segments=3,
            parent_window_radius=1,
        )
        self.assertIn("seg-3", candidates)
        self.assertGreater(float(candidates["seg-3"]["parent_score"]), 0.0)

    def test_parent_anchor_selection_prefers_new_anchor_coverage(self) -> None:
        selected = _select_parent_anchor_rows(
            [
                {
                    "source_id": "seg-near-1",
                    "anchor_priority": 0.74,
                    "matched_term_weight": 1.2,
                    "phrase_score": 0.0,
                    "direct_match": 0.48,
                    "lexical": 0.31,
                    "matched_anchor_terms": ["fingerprint"],
                },
                {
                    "source_id": "seg-near-2",
                    "anchor_priority": 0.72,
                    "matched_term_weight": 1.1,
                    "phrase_score": 0.0,
                    "direct_match": 0.47,
                    "lexical": 0.3,
                    "matched_anchor_terms": ["fingerprint"],
                },
                {
                    "source_id": "seg-far-gold",
                    "anchor_priority": 0.69,
                    "matched_term_weight": 3.8,
                    "phrase_score": 0.7,
                    "direct_match": 0.55,
                    "lexical": 0.52,
                    "matched_anchor_terms": ["foobar.py", "recall@10"],
                },
            ],
            anchor_cap=2,
        )
        self.assertEqual({row["source_id"] for row in selected[:2]}, {"seg-near-1", "seg-far-gold"})
        self.assertNotIn("seg-near-2", [row["source_id"] for row in selected[:2]])

    def test_parent_anchor_terms_filter_low_information_pronouns(self) -> None:
        filtered = _parent_anchor_terms(
            ["i", "ve", "your", "over", "you", "through", "old", "tools", "now", "retirement", "legacy", "march"]
        )

        self.assertNotIn("i", filtered)
        self.assertNotIn("you", filtered)
        self.assertNotIn("ve", filtered)
        self.assertIn("tools", filtered)
        self.assertIn("retirement", filtered)
        self.assertIn("legacy", filtered)
        self.assertIn("march", filtered)

    def test_parent_supplemental_anchor_expansion_is_opt_in(self) -> None:
        query = "Which foobar.py patch restored Recall@10 after the fingerprint mismatch?"
        query_features = _query_features(query)
        source_records = {
            "seg-1": self._record("seg-1", "The team discussed a cache fingerprint mismatch.", order_index=10),
            "seg-2": self._record(
                "seg-2",
                "The fingerprint mismatch was restored after a patch and the cache mismatch was reproduced.",
                order_index=11,
            ),
            "seg-3": self._record("seg-3", "A foobar.py patch restored Recall@10 after the fingerprint mismatch.", order_index=40),
        }
        side_index = _build_retrieval_side_index(
            benchmark_name="clonemem",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "fp-supplemental-anchor"},
        )
        seed = merge_candidate_sources(
            {
                "seg-1": {
                    **source_records["seg-1"],
                    "source_retrievers": ["dense"],
                    "source_chunk_ids": [],
                    "best_chunk_id": "",
                    "dense_score": 0.95,
                }
            }
        )
        kwargs = dict(
            benchmark_name="clonemem",
            query=query,
            query_features=query_features,
            decomposition=_build_query_decomposition(query, query_features=query_features),
            seed_candidates=seed,
            source_records=source_records,
            side_index=side_index,
            parent_top_k=3,
            parent_expand_segments=1,
            parent_window_radius=1,
        )

        default_candidates, default_audit = _parent_session_source_candidates(**kwargs)
        expanded_candidates, expanded_audit = _parent_session_source_candidates(
            **kwargs,
            parent_supplemental_anchor_expansion_enabled=True,
            parent_supplemental_anchor_expansion_cap=1,
        )

        self.assertEqual(len(default_candidates), 1)
        self.assertGreater(len(expanded_candidates), len(default_candidates))
        self.assertIn("seg-2", expanded_candidates)
        self.assertEqual(int(default_audit["supplemental_anchor_selected_count"]), 0)
        self.assertEqual(int(expanded_audit["supplemental_anchor_selected_count"]), 1)

    def test_parent_session_channel_keeps_distant_anchor_segment_for_knowme(self) -> None:
        query = "Which change in fooBar.py restored Recall@10 after the fingerprint mismatch?"
        query_features = _query_features(query)
        source_records = {
            "seg-1": self._record("seg-1", "The team began triaging a regression after the cache fingerprint mismatch.", order_index=10),
            "seg-2": self._record("seg-2", "They reviewed candidate recall drift in nearby logs.", order_index=11),
            "seg-3": self._record("seg-3", "The adapter discussion stayed focused on the mismatch and local noise.", order_index=12),
            "seg-4": self._record("seg-4", "A later patch in fooBar.py restored Recall@10 after the fingerprint mismatch.", order_index=24),
        }
        side_index = _build_retrieval_side_index(
            benchmark_name="clonemem",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "fp-2c"},
        )
        seed = merge_candidate_sources(
            {
                "seg-1": {
                    **source_records["seg-1"],
                    "source_retrievers": ["dense"],
                    "source_chunk_ids": [],
                    "best_chunk_id": "",
                    "dense_score": 0.95,
                }
            }
        )
        candidates, audit = _parent_session_source_candidates(
            benchmark_name="knowme",
            query=query,
            query_features=query_features,
            decomposition=_build_query_decomposition(query, query_features=query_features),
            seed_candidates=seed,
            source_records=source_records,
            side_index=side_index,
            parent_top_k=4,
            parent_expand_segments=2,
            parent_window_radius=1,
        )
        self.assertIn("seg-4", candidates)
        self.assertGreaterEqual(int(audit["parent_anchor_selected_count"]), 1)
        self.assertGreaterEqual(int(audit["parent_anchor_term_coverage_count"]), 2)

    def test_temporal_neighbor_channel_expands_adjacent_segments(self) -> None:
        source_records = {
            "seg-1": self._record("seg-1", "March: Melanie sorted through old tools.", order_index=0),
            "seg-2": self._record("seg-2", "April: retirement and legacy started to weigh on her mind.", order_index=1),
            "seg-3": self._record("seg-3", "May: she left chess early after losing focus.", order_index=2),
        }
        side_index = _build_retrieval_side_index(
            benchmark_name="clonemem",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "fp-3"},
        )
        seed = {
            "seg-2": {
                **source_records["seg-2"],
                "source_retrievers": ["dense"],
                "source_chunk_ids": [],
                "best_chunk_id": "",
                "dense_score": 0.88,
            }
        }
        candidates, audit = _temporal_neighbor_source_candidates(
            seed_candidates=seed,
            source_records=source_records,
            side_index=side_index,
            query_features=_query_features("In May did retirement and legacy weigh on her mind?"),
            max_neighbors_per_seed=2,
            limit=10,
        )
        self.assertIn("seg-1", candidates)
        self.assertIn("seg-3", candidates)
        self.assertGreaterEqual(audit["expanded_candidate_count"], 2)

    def test_query_decomposition_and_profile_side_index_help_profile_queries(self) -> None:
        source_records = {
            "seg-good": self._record("seg-good", "Melanie prefers strong coffee with oat milk every morning."),
            "seg-wrong": self._record("seg-wrong", "Melanie debugged a router issue after lunch."),
        }
        side_index = _build_retrieval_side_index(
            benchmark_name="knowme",
            source_records=source_records,
            index_fingerprint={"fingerprint_hash": "fp-4"},
        )
        query = "What kind of coffee does Melanie prefer?"
        query_features = _query_features(query)
        decomposition = _build_query_decomposition(query, query_features=query_features)
        decomposition_candidates = _query_decomposition_source_candidates(
            decomposition=decomposition,
            query=query,
            source_records=source_records,
            side_index=side_index,
            limit=5,
        )
        profile_candidates = _profile_side_index_candidates(
            decomposition=decomposition,
            query=query,
            side_index=side_index,
            source_records=source_records,
            limit=5,
        )
        self.assertIn("seg-good", decomposition_candidates)
        self.assertIn("seg-good", profile_candidates)
        self.assertGreater(
            float(profile_candidates["seg-good"]["profile_score"]),
            float(profile_candidates.get("seg-wrong", {}).get("profile_score", 0.0)),
        )

    def test_lexical_channel_boosts_compound_and_metric_terms(self) -> None:
        source_records = {
            "seg-good": self._record("seg-good", "The fix in fooBar.py restored Recall@10 after the cache fingerprint change."),
            "seg-bad": self._record("seg-bad", "The adapter changed a setting after a generic regression."),
        }
        storage = FakeStorage(
            [
                {"chunk_id": "chunk-good", "text": source_records["seg-good"]["text"], "bm25_score": 1.0},
                {"chunk_id": "chunk-bad", "text": source_records["seg-bad"]["text"], "bm25_score": 1.0},
            ]
        )
        candidates = _lexical_source_candidates(
            query="Which change in fooBar.py restored Recall@10?",
            benchmark_name="clonemem",
            query_features=_query_features("Which change in fooBar.py restored Recall@10?"),
            storage=storage,
            chunk_metadata_by_id={
                "chunk-good": {"chunk_id": "chunk-good", "source_segment_id": "seg-good"},
                "chunk-bad": {"chunk_id": "chunk-bad", "source_segment_id": "seg-bad"},
            },
            source_records=source_records,
            limit=5,
        )
        self.assertGreater(float(candidates["seg-good"]["bm25_score"]), float(candidates["seg-bad"]["bm25_score"]))

    def test_fusion_promotes_parent_diversity(self) -> None:
        def candidate(source_id: str, parent: str, score_key: str, score: float) -> dict[str, object]:
            row = {
                "source_id": source_id,
                "source_segment_id": source_id,
                "source_doc_id": parent,
                "session_id": parent,
                "conversation_id": parent,
                "sample_id": parent,
                "benchmark_name": "clonemem",
                "text": source_id,
                "normalized_text": source_id,
                "text_hash": source_id,
                "token_list": [source_id],
                "entity_terms": [],
                "temporal_terms": [],
                "specific_temporal_terms": [],
                "order_index": 0,
                "source_retrievers": [],
                "source_chunk_ids": [],
                "best_chunk_id": "",
                "dense_score": 0.0,
                "bm25_score": 0.0,
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
                "broad_score": score,
            }
            row[score_key] = score
            row["source_retrievers"] = [score_key]
            return row

        dense = {
            "a1": candidate("a1", "parent-a", "dense_score", 0.95),
            "a2": candidate("a2", "parent-a", "dense_score", 0.9),
            "b1": candidate("b1", "parent-b", "dense_score", 0.7),
        }
        lexical = {
            "a2": candidate("a2", "parent-a", "bm25_score", 0.93),
            "b1": candidate("b1", "parent-b", "bm25_score", 0.89),
        }
        merged = merge_candidate_sources(dense, lexical)
        fused, audit, _ = _fuse_candidate_sources(
            benchmark_name="clonemem",
            merged=merged,
            channel_maps={
                "dense_semantic": dense,
                "lexical_sparse": lexical,
            },
            policy={
                "fusion_method": "rrf",
                "rrf_k": 60,
                "duplicate_collapse_enabled": True,
                "near_duplicate_collapse_enabled": True,
                "min_parent_diversity": 2,
                "max_candidates_per_parent": 1,
                "final_candidate_pool_size": 3,
            },
        )
        top_parents = [row["source_doc_id"] for row in fused[:2]]
        self.assertEqual(set(top_parents), {"parent-a", "parent-b"})
        self.assertTrue(audit["parent_diversity_adjusted"])

    def test_clonemem_lexical_anchor_gate_audits_weak_lexical_candidates(self) -> None:
        weak = {
            "source_id": "weak-lexical",
            "source_segment_id": "weak-lexical",
            "source_doc_id": "parent-a",
            "session_id": "parent-a",
            "conversation_id": "parent-a",
            "sample_id": "parent-a",
            "benchmark_name": "clonemem",
            "text": "weak lexical distractor",
            "normalized_text": "weak lexical distractor",
            "text_hash": "weak-lexical",
            "token_list": ["weak", "lexical", "distractor"],
            "entity_terms": [],
            "temporal_terms": [],
            "specific_temporal_terms": [],
            "order_index": 0,
            "source_retrievers": ["bm25"],
            "source_chunk_ids": [],
            "best_chunk_id": "",
            "dense_score": 0.0,
            "bm25_score": 1.0,
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
            "support_count": 1,
            "broad_score": 0.12,
        }
        anchored = dict(
            weak,
            source_id="anchored",
            source_segment_id="anchored",
            dense_score=0.5,
            broad_score=0.5,
            text_hash="anchored",
        )
        dense = {"anchored": anchored}
        lexical = {"weak-lexical": weak, "anchored": dict(anchored, bm25_score=0.8)}
        merged = merge_candidate_sources(dense, lexical)
        fused, audit, _ = _fuse_candidate_sources(
            benchmark_name="clonemem",
            merged=merged,
            channel_maps={"dense_semantic": dense, "lexical_sparse": lexical},
            policy={
                "fusion_method": "rrf",
                "rrf_k": 60,
                "duplicate_collapse_enabled": True,
                "near_duplicate_collapse_enabled": True,
                "final_candidate_pool_size": 5,
                "clonemem_lexical_anchor_gate_enabled": True,
            },
        )
        weak_row = next(row for row in fused if row["source_id"] == "weak-lexical")
        anchored_row = next(row for row in fused if row["source_id"] == "anchored")
        self.assertEqual(weak_row["clonemem_lexical_anchor_gate_factor"], 0.35)
        self.assertEqual(anchored_row["clonemem_lexical_anchor_gate_factor"], 1.0)
        self.assertEqual(audit["clonemem_lexical_anchor_gate_applied_count"], 1)

    def test_safe_fusion_preserves_dense_anchor_coverage(self) -> None:
        def candidate(source_id: str, parent: str, *, dense: float = 0.0, bm25: float = 0.0) -> dict[str, object]:
            row = {
                "source_id": source_id,
                "source_segment_id": source_id,
                "source_doc_id": parent,
                "session_id": parent,
                "conversation_id": parent,
                "sample_id": parent,
                "benchmark_name": "knowme",
                "text": source_id,
                "normalized_text": source_id,
                "text_hash": f"hash-{source_id}",
                "token_list": [source_id],
                "entity_terms": [],
                "temporal_terms": [],
                "specific_temporal_terms": [],
                "order_index": 0,
                "source_retrievers": [],
                "source_chunk_ids": [],
                "best_chunk_id": "",
                "dense_score": dense,
                "bm25_score": bm25,
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
                "broad_score": max(dense, bm25),
            }
            if dense:
                row["source_retrievers"].append("dense")
            if bm25:
                row["source_retrievers"].append("bm25")
            return row

        dense = {
            "a1": candidate("a1", "parent-a", dense=0.99),
            "a2": candidate("a2", "parent-b", dense=0.97),
            "a3": candidate("a3", "parent-c", dense=0.95),
        }
        lexical = {
            "b1": candidate("b1", "parent-x", bm25=1.0),
            "b2": candidate("b2", "parent-y", bm25=0.98),
        }
        merged = merge_candidate_sources(dense, lexical)
        fused, audit, _ = _fuse_candidate_sources(
            benchmark_name="knowme",
            merged=merged,
            channel_maps={"dense_semantic": dense, "lexical_sparse": lexical},
            policy={
                "fusion_method": "rrf",
                "rrf_k": 60,
                "safe_fusion_enabled": True,
                "dense_preserve_enabled": True,
                "dense_anchor_top_k": 3,
                "dense_anchor_min_keep": 3,
                "dense_gold_agnostic_rank_floor_enabled": True,
                "channel_gating_enabled": True,
                "destructive_filter_guard_enabled": True,
                "duplicate_collapse_enabled": True,
                "near_duplicate_collapse_enabled": True,
                "duplicate_collapse_safe_mode": True,
                "parent_cap_after_gold_agnostic_anchor": True,
                "max_candidates_per_parent": 1,
                "final_candidate_pool_size": 3,
            },
        )
        self.assertEqual([row["source_id"] for row in fused], ["a1", "a2", "a3"])
        self.assertEqual(audit["protected_dense_retained_count"], 3)

    def test_duplicate_collapse_safe_mode_keeps_distinct_segment_ids(self) -> None:
        rows = [
            {
                "source_id": "seg-1",
                "source_segment_id": "seg-1",
                "source_doc_id": "parent-a",
                "text_hash": "same",
            },
            {
                "source_id": "seg-2",
                "source_segment_id": "seg-2",
                "source_doc_id": "parent-a",
                "text_hash": "same",
            },
        ]
        kept, audit = _apply_duplicate_collapse(
            rows,
            duplicate_collapse_enabled=True,
            near_duplicate_collapse_enabled=True,
            safe_mode=True,
        )
        self.assertEqual([row["source_id"] for row in kept], ["seg-1", "seg-2"])
        self.assertEqual(audit["near_duplicate_collapse_count"], 0)

    def test_parent_cap_audit_lists_skipped_candidates(self) -> None:
        dense = {
            f"a{i}": {
                **self._record(f"a{i}", f"record {i}", source_doc_id="parent-a"),
                "source_retrievers": ["dense"],
                "source_chunk_ids": [],
                "best_chunk_id": "",
                "dense_score": 1.0 - i * 0.01,
                "bm25_score": 0.0,
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
                "broad_score": 1.0 - i * 0.01,
            }
            for i in range(2)
        }
        merged = merge_candidate_sources(dense)
        fused, audit, _ = _fuse_candidate_sources(
            benchmark_name="clonemem",
            merged=merged,
            channel_maps={"dense_semantic": dense},
            policy={
                "fusion_method": "rrf",
                "rrf_k": 60,
                "safe_fusion_enabled": False,
                "duplicate_collapse_enabled": True,
                "near_duplicate_collapse_enabled": True,
                "max_candidates_per_parent": 1,
                "final_candidate_pool_size": 2,
            },
        )
        self.assertEqual(len(fused), 2)
        self.assertTrue(audit["parent_cap_applied"])
        self.assertEqual(audit["parent_cap_skipped_candidates"][0]["source_id"], "a1")

    def test_parent_cap_is_deferred_in_safe_mode(self) -> None:
        dense = {
            f"a{i}": {
                **self._record(f"a{i}", f"record {i}", source_doc_id="parent-a"),
                "source_retrievers": ["dense"],
                "source_chunk_ids": [],
                "best_chunk_id": "",
                "dense_score": 1.0 - i * 0.01,
                "bm25_score": 0.0,
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
                "broad_score": 1.0 - i * 0.01,
            }
            for i in range(3)
        }
        merged = merge_candidate_sources(dense)
        fused, audit, _ = _fuse_candidate_sources(
            benchmark_name="clonemem",
            merged=merged,
            channel_maps={"dense_semantic": dense},
            policy={
                "fusion_method": "rrf",
                "rrf_k": 60,
                "safe_fusion_enabled": True,
                "dense_preserve_enabled": True,
                "dense_anchor_top_k": 3,
                "dense_anchor_min_keep": 3,
                "destructive_filter_guard_enabled": True,
                "duplicate_collapse_enabled": True,
                "near_duplicate_collapse_enabled": True,
                "duplicate_collapse_safe_mode": True,
                "parent_cap_after_gold_agnostic_anchor": True,
                "max_candidates_per_parent": 1,
                "final_candidate_pool_size": 3,
            },
        )
        self.assertEqual(len(fused), 3)
        self.assertTrue(all(row["source_doc_id"] == "parent-a" for row in fused))
        self.assertFalse(audit["parent_cap_applied"])
        self.assertGreaterEqual(audit["potential_parent_cap_count"], 2)

    def test_inhibition_safe_mode_reorders_without_dropping_candidates(self) -> None:
        reranked = [
            {
                **self._record("seg-1", "alpha beta gamma", source_doc_id="parent-a"),
                "rerank_score": 0.8,
                "broad_rank": 1,
                "support_count": 3,
            },
            {
                **self._record("seg-2", "alpha beta delta", source_doc_id="parent-a"),
                "rerank_score": 0.79,
                "broad_rank": 2,
                "support_count": 2,
            },
            {
                **self._record("seg-3", "unrelated topic", source_doc_id="parent-b"),
                "rerank_score": 0.5,
                "broad_rank": 3,
                "support_count": 1,
            },
        ]
        adjusted, audit = apply_inhibition_audit("clonemem", reranked, safe_mode=True)
        self.assertEqual({row["source_id"] for row in adjusted}, {"seg-1", "seg-2", "seg-3"})
        self.assertEqual(len(adjusted), len(reranked))
        self.assertTrue(all(float(row["inhibition_penalty"]) <= 0.08 for row in adjusted))
        self.assertTrue(audit["safe_mode"])

    def test_candidate_recall_summary_tracks_dense_vs_fused_audit(self) -> None:
        summary = build_candidate_recall_summary(
            benchmark_name="clonemem",
            rows=[
                {
                    "query_id": "q1",
                    "query_text": "query",
                    "candidate_recall@10": 0.0,
                    "candidate_recall@50": 1.0,
                    "candidate_recall@100": 1.0,
                    "candidate_recall@200": 1.0,
                    "candidate_ndcg@10": 0.0,
                    "final_recall@10": 0.0,
                    "final_ndcg@10": 0.0,
                    "dense_hit@100": True,
                    "fused_hit@100": False,
                    "dense_gold_rank": 9,
                    "fused_gold_rank": None,
                    "gold_removed_by_duplicate_collapse": True,
                    "gold_removed_by_near_duplicate_collapse": False,
                    "gold_removed_by_parent_cap": True,
                    "gold_downranked_by_inhibition": True,
                    "failure_type": "gold_missing_from_candidate_pool",
                    "channel_stats": {"dense_semantic": {"gold_hit": True, "gold_rank": 9, "candidate_count": 100}},
                }
            ],
        )
        self.assertEqual(summary["dense_hit@100"], 1.0)
        self.assertEqual(summary["fused_hit@100"], 0.0)
        self.assertEqual(summary["gold_removed_by_duplicate_collapse_count"], 1)
        self.assertEqual(summary["gold_removed_by_parent_cap_count"], 1)
        self.assertEqual(summary["gold_downranked_by_inhibition_count"], 1)

    def test_query_feature_entity_filter_removes_question_words(self) -> None:
        features = _query_features("What did Alice change in fooBar.py?")
        self.assertIn("alice", features["entities"])
        self.assertNotIn("what", features["entities"])

    def test_locomo_rerank_preserves_high_broad_non_session_candidate(self) -> None:
        anchored = {
            **self._record("session-gold", "Maria talked about trips to Spain and England.", source_doc_id="session-gold"),
            "benchmark_name": "locomo",
            "dense_score": 0.33,
            "bm25_score": 0.14,
            "entity_score": 0.39,
            "temporal_score": 0.0,
            "session_score": 0.0,
            "parent_score": 0.36,
            "decomposition_score": 0.31,
            "exact_phrase_score": 0.4,
            "semantic_score": 0.33,
            "task_score": 0.18,
            "support_count": 7,
            "broad_score": 0.71,
            "broad_rank": 9,
        }
        noisy_bundle = {
            **self._record("session-noisy", "Maria mentioned travel in passing.", source_doc_id="session-noisy"),
            "benchmark_name": "locomo",
            "dense_score": 0.32,
            "bm25_score": 0.16,
            "entity_score": 0.39,
            "temporal_score": 0.0,
            "session_score": 0.29,
            "parent_score": 0.36,
            "decomposition_score": 0.31,
            "exact_phrase_score": 0.4,
            "semantic_score": 0.328,
            "task_score": 0.26,
            "support_count": 8,
            "broad_score": 0.8,
            "broad_rank": 11,
        }
        self.assertGreater(_rerank_score("locomo", anchored), _rerank_score("locomo", noisy_bundle))

    def test_tokenize_expands_snake_camel_path_and_version_tokens(self) -> None:
        tokens = tokenize("fooBar baz_qux config-v2 /tmp/MyFile.py Recall@10")
        for expected in (
            "foobar",
            "foo",
            "bar",
            "baz_qux",
            "baz",
            "qux",
            "config-v2",
            "config",
            "v2",
            "tmp",
            "myfile",
            "my",
            "file",
            "py",
            "recall",
            "10",
        ):
            self.assertIn(expected, tokens)
        for expected in (
            "tmp/myfile.py",
            "recall@10",
            "config-v2",
        ):
            self.assertIn(expected, tokens)

    def test_rank_benchmark_sources_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(base_dir=Path(tmp))
            config.final_candidate_pool_size = 10
            config.max_candidates_per_parent = 3
            config.min_parent_diversity = 2
            source_records = {
                "seg-1": self._record("seg-1", "Melanie prefers strong coffee with oat milk.", source_doc_id="parent-a", order_index=0, speaker_id="melanie"),
                "seg-2": self._record("seg-2", "Melanie keeps a coffee grinder in the workshop.", source_doc_id="parent-a", order_index=1, speaker_id="melanie"),
                "seg-3": self._record("seg-3", "The benchmark adapter changed the cache fingerprint.", source_doc_id="parent-b", order_index=0, speaker_id="maintainer"),
            }
            index_metadata = {
                "workspace_dir": tmp,
                "fingerprint": {"fingerprint_hash": "fp-5"},
                "source_records_by_id": source_records,
                "chunk_metadata_by_id": {
                    "chunk-1": {"chunk_id": "chunk-1", "source_segment_id": "seg-1", "source_doc_id": "parent-a"},
                    "chunk-2": {"chunk_id": "chunk-2", "source_segment_id": "seg-2", "source_doc_id": "parent-a"},
                    "chunk-3": {"chunk_id": "chunk-3", "source_segment_id": "seg-3", "source_doc_id": "parent-b"},
                },
            }
            vector_store = FakeVectorStore(
                {
                    "What coffee does Melanie prefer?": [
                        {"chunk_id": "chunk-1", "similarity": 0.97},
                        {"chunk_id": "chunk-2", "similarity": 0.71},
                    ]
                }
            )
            storage = FakeStorage(
                [
                    {"chunk_id": "chunk-1", "text": source_records["seg-1"]["text"]},
                    {"chunk_id": "chunk-2", "text": source_records["seg-2"]["text"]},
                ]
            )
            first = rank_benchmark_sources(
                query="What coffee does Melanie prefer?",
                benchmark_name="knowme",
                vector_store=vector_store,
                storage=storage,
                index_metadata=index_metadata,
                config=config,
                pool_limit=10,
            )
            second = rank_benchmark_sources(
                query="What coffee does Melanie prefer?",
                benchmark_name="knowme",
                vector_store=vector_store,
                storage=storage,
                index_metadata=index_metadata,
                config=config,
                pool_limit=10,
            )
            first_ids = [row["source_id"] for row in first["final_candidates"][:5]]
            second_ids = [row["source_id"] for row in second["final_candidates"][:5]]
            self.assertEqual(first_ids, second_ids)
            self.assertEqual(first["channel_stats"], second["channel_stats"])

    def test_rank_early_exit_skips_decomposition_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(base_dir=Path(tmp))
            config.retrieval_min_seed_candidates = 2
            config.candidate_recall_eval_k = 2
            config.retrieval_confidence_margin = 0.12
            config.retrieval_early_exit_enabled = True
            config.final_candidate_pool_size = 10
            source_records = {
                "seg-1": self._record("seg-1", "Melanie prefers strong coffee with oat milk.", source_doc_id="parent-a", order_index=0),
                "seg-2": self._record("seg-2", "Melanie keeps a coffee grinder in the workshop.", source_doc_id="parent-a", order_index=1),
                "seg-3": self._record("seg-3", "Nora prefers tea in the garden.", source_doc_id="parent-b", order_index=0),
            }
            for idx in range(4, 12):
                source_records[f"seg-{idx}"] = self._record(
                    f"seg-{idx}",
                    f"Coffee support candidate {idx} for deterministic early exit.",
                    source_doc_id=f"parent-{idx}",
                    order_index=idx,
                )
            index_metadata = {
                "workspace_dir": tmp,
                "fingerprint": {"fingerprint_hash": "fp-early-exit-rank"},
                "source_records_by_id": source_records,
                "chunk_metadata_by_id": {
                    f"chunk-{idx}": {
                        "chunk_id": f"chunk-{idx}",
                        "source_segment_id": f"seg-{idx}",
                        "source_doc_id": str(source_records[f"seg-{idx}"]["source_doc_id"]),
                    }
                    for idx in range(1, 12)
                },
            }
            result = rank_benchmark_sources(
                query="What exact coffee does Melanie prefer?",
                benchmark_name="locomo",
                vector_store=FakeVectorStore(
                    {
                        "What exact coffee does Melanie prefer?": [{"chunk_id": "chunk-1", "similarity": 0.98}]
                        + [{"chunk_id": f"chunk-{idx}", "similarity": 0.5 - idx * 0.01} for idx in range(2, 12)]
                    }
                ),
                storage=FakeStorage([{"chunk_id": "chunk-1", "text": source_records["seg-1"]["text"]}]),
                index_metadata=index_metadata,
                config=config,
                pool_limit=10,
            )
            self.assertTrue(result["early_exit"]["early_exit_triggered"])
            self.assertIn("query_decomposition", result["early_exit"]["skipped_expensive_channels"])
            self.assertEqual(result["candidate_source_stats"]["query_decomposition"], 0)
            self.assertEqual(result["candidate_source_stats"]["session_bundle"], 0)
            self.assertEqual(result["candidate_source_stats"]["parent_session"], 0)

    def test_side_index_audit_reports_fast_path_and_posting_miss_without_full_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(base_dir=Path(tmp))
            config.retrieval_early_exit_enabled = False
            source_records = {
                "seg-1": self._record("seg-1", "Melanie prefers strong coffee.", source_doc_id="parent-a", order_index=0),
                "seg-2": self._record("seg-2", "Nora likes tea.", source_doc_id="parent-b", order_index=1),
            }
            index_metadata = {
                "workspace_dir": tmp,
                "fingerprint": {"fingerprint_hash": "fp-audit"},
                "source_records_by_id": source_records,
                "chunk_metadata_by_id": {
                    "chunk-1": {"chunk_id": "chunk-1", "source_segment_id": "seg-1", "source_doc_id": "parent-a"},
                    "chunk-2": {"chunk_id": "chunk-2", "source_segment_id": "seg-2", "source_doc_id": "parent-b"},
                },
            }
            result = rank_benchmark_sources(
                query="What does Melanie prefer?",
                benchmark_name="knowme",
                vector_store=FakeVectorStore({"What does Melanie prefer?": [{"chunk_id": "chunk-1", "similarity": 0.9}]}),
                storage=FakeStorage([]),
                index_metadata=index_metadata,
                config=config,
                pool_limit=10,
            )
            audit = result["side_index_audit"]
            self.assertTrue(audit["entity"]["indexed_fast_path_available"])
            self.assertTrue(audit["entity"]["indexed_fast_path_used"])
            self.assertEqual(result["side_index_audit_summary"]["full_scan_total_records_scored"], 0)

            result_miss = rank_benchmark_sources(
                query="zzzz unmatched token",
                benchmark_name="knowme",
                vector_store=FakeVectorStore({"zzzz unmatched token": []}),
                storage=FakeStorage([]),
                index_metadata=index_metadata,
                config=config,
                pool_limit=10,
            )
            self.assertTrue(result_miss["side_index_audit"]["exact_phrase"]["indexed_fast_path_available"])
            self.assertFalse(result_miss["side_index_audit"]["exact_phrase"]["legacy_fallback_used"])

    def test_failure_and_category_reports_are_generated(self) -> None:
        clonemem_report = build_clonemem_failure_taxonomy(
            [
                {
                    "query_id": "q1",
                    "query_text": "query",
                    "failure_type": "gold_missing_from_candidate_pool",
                    "channel_stats": {
                        "dense_semantic": {"gold_hit": False},
                        "lexical_sparse": {"gold_hit": False},
                        "entity_aware": {"gold_hit": True},
                    },
                    "candidate_recall@100": 0.0,
                    "gold_rank_before_rerank": None,
                    "gold_rank_after_rerank": None,
                    "gold_rank_after_inhibition": None,
                    "top_parent_distribution": {"parent-a": 10},
                    "fusion_audit": {"local_crowding_count": 1},
                }
            ]
        )
        knowme_report = build_knowme_category_analysis(
            [
                {
                    "query_id": "k1",
                    "query_text": "Where does Melanie live now?",
                    "candidate_recall@100": 1.0,
                    "recall_frac@10": 1.0,
                    "recall_any@10": 1.0,
                    "ndcg_any@10": 1.0,
                    "failure_type": "ok",
                    "channel_stats": {"entity_aware": {"gold_hit": True, "gold_rank": 1}},
                }
            ]
        )
        self.assertIn("dense_semantic_miss", clonemem_report["failure_type_distribution"])
        self.assertIn("location query", knowme_report["categories"])


if __name__ == "__main__":
    unittest.main()
