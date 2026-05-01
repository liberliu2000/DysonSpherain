from __future__ import annotations

import tempfile
import unittest
import warnings
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "base"
BENCHMARKS_DIR = ROOT / "benchmarks"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from benchmark_support import (
    _entity_source_candidates,
    _exact_phrase_source_candidates,
    _query_features,
    _session_bundle_source_candidates,
    _local_window_source_candidates,
    assemble_broad_candidate_pool,
    assert_benchmark_vector_guard,
    build_candidate_recall_summary,
    build_integrity_report,
    build_query_diagnostic,
    build_query_failure,
    build_raw_counts,
    build_topk_debug_record,
    force_benchmark_config,
    merge_candidate_sources,
    _retrieval_policy,
)
from knowme_benchmark import load_dataset_bundle as load_knowme_dataset_bundle
from longmemeval_benchmark import BenchmarkWorkspaceManager
from sphere_cli.config import AppConfig
from sphere_cli.embedding import EmbeddingProvider
from sphere_cli.models import MemoryNode
from sphere_cli.runtime import UnifiedMemoryRuntime
from sphere_cli.storage import Storage
from sphere_cli.vector_store import VectorStore, _build_chroma_client
from merge_benchmark_results import merge_payloads
from run_benchmark_chunked import (
    KNOWME_OFFICIAL_FORMAL_ENV,
    benchmark_command,
    benchmark_cache_roots,
    benchmark_profile_env,
    detect_clonemem_sample_id,
    sample_allowlist_is_empty,
    split_round_robin,
    validate_metrics,
)
from shard_utils import belongs_to_shard, filter_sharded_items


class GuardrailTests(unittest.TestCase):
    @staticmethod
    def _candidate(source_id: str, **scores: float) -> dict[str, object]:
        base = {
            "source_id": source_id,
            "source_segment_id": source_id,
            "source_doc_id": "doc-1",
            "benchmark_name": "clonemem",
            "sample_id": "sample-1",
            "conversation_id": "sample-1",
            "session_id": "",
            "turn_id": "",
            "speaker_id": "",
            "timestamp": "",
            "text": source_id,
            "normalized_text": source_id,
            "token_list": [],
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
            "local_window_score": 0.0,
            "broad_score": 0.0,
            "rerank_score": 0.0,
            "post_inhibition_score": 0.0,
            "inhibition_penalty": 0.0,
        }
        base.update(scores)
        return base

    def test_benchmark_ablation_overrides_forced_config_without_affecting_default(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            default = force_benchmark_config(AppConfig())
        self.assertTrue(default.safe_fusion_enabled)
        self.assertTrue(default.enable_benchmark_route_tuning)

        with patch.dict("os.environ", {"SPHERE_BENCHMARK_ABLATION": "no_safe_fusion"}, clear=False):
            no_safe = force_benchmark_config(AppConfig())
        self.assertFalse(no_safe.safe_fusion_enabled)
        self.assertFalse(no_safe.destructive_filter_guard_enabled)

        with patch.dict("os.environ", {"SPHERE_BENCHMARK_ABLATION": "no_route_conditioned_admission"}, clear=False):
            no_route = force_benchmark_config(AppConfig())
        self.assertFalse(no_route.enable_benchmark_route_tuning)
        self.assertFalse(no_route.route_aware_gating_enabled)

    def test_clonemem_route_only_promotes_protected_lexical_gate(self) -> None:
        default = AppConfig()
        self.assertFalse(default.clonemem_lexical_anchor_gate_enabled)
        self.assertEqual(default.clonemem_lexical_anchor_gate_protected_top_k, 0)

        clonemem_policy = _retrieval_policy(default, "clonemem", 300)
        self.assertTrue(clonemem_policy["clonemem_lexical_anchor_gate_enabled"])
        self.assertEqual(clonemem_policy["clonemem_lexical_anchor_gate_protected_top_k"], 3)

        knowme_policy = _retrieval_policy(default, "knowme", 300)
        self.assertFalse(knowme_policy["clonemem_lexical_anchor_gate_enabled"])
        self.assertEqual(knowme_policy["clonemem_lexical_anchor_gate_protected_top_k"], 0)

        route_off = AppConfig(enable_benchmark_route_tuning=False)
        route_off_policy = _retrieval_policy(route_off, "clonemem", 300)
        self.assertFalse(route_off_policy["clonemem_lexical_anchor_gate_enabled"])
        self.assertEqual(route_off_policy["clonemem_lexical_anchor_gate_protected_top_k"], 0)

    def test_benchmark_workspace_manager_retains_open_workspace_with_lru_escape_hatch(self) -> None:
        class FakeWorkspace:
            def __init__(self, signature: str) -> None:
                self.signature = signature
                self.build_elapsed_ms = 10.0
                self.close_count = 0

            def close(self) -> None:
                self.close_count += 1

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                "os.environ",
                {
                    "SPHERE_BENCHMARK_KEEP_WORKSPACE_OPEN": "true",
                    "SPHERE_BENCHMARK_MAX_OPEN_WORKSPACES": "1",
                },
                clear=False,
            ):
                manager = BenchmarkWorkspaceManager(Path(tmp) / "cache", Path(tmp) / "shared")
                first = FakeWorkspace("sig-a")
                second = FakeWorkspace("sig-b")
                manager._workspaces["sig-a"] = first  # type: ignore[assignment]
                manager.release("sig-a")
                self.assertIn("sig-a", manager._workspaces)
                self.assertEqual(first.close_count, 0)

                manager._workspaces["sig-b"] = second  # type: ignore[assignment]
                manager._evict_surplus(keep_signature="sig-b")
                self.assertNotIn("sig-a", manager._workspaces)
                self.assertEqual(first.close_count, 1)
                self.assertIn("sig-b", manager._workspaces)

                manager.release("sig-b", force_close=True)
                self.assertNotIn("sig-b", manager._workspaces)
                self.assertEqual(second.close_count, 1)

            with patch.dict("os.environ", {"SPHERE_BENCHMARK_KEEP_WORKSPACE_OPEN": "false"}, clear=False):
                manager = BenchmarkWorkspaceManager(Path(tmp) / "cache2", Path(tmp) / "shared2")
                third = FakeWorkspace("sig-c")
                manager._workspaces["sig-c"] = third  # type: ignore[assignment]
                manager.release("sig-c")
                self.assertNotIn("sig-c", manager._workspaces)
                self.assertEqual(third.close_count, 1)

    def test_benchmark_guard_blocks_local_hash_and_fingerprint_mismatch(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_benchmark_vector_guard(
                vector_info={
                    "embedding_provider": "local_hash",
                    "embedding_model": "local-hash-384",
                    "fallback_in_use": True,
                },
                runtime_fingerprint={
                    "embedding_provider": "local_hash",
                    "embedding_model": "local-hash-384",
                    "embedding_dim": 384,
                    "normalize_embeddings": True,
                },
                index_fingerprint=None,
            )
        with self.assertRaises(RuntimeError):
            assert_benchmark_vector_guard(
                vector_info={
                    "embedding_provider": "sentence_transformer",
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "fallback_in_use": False,
                },
                runtime_fingerprint={
                    "embedding_provider": "sentence_transformer",
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_dim": 384,
                    "normalize_embeddings": True,
                },
                index_fingerprint={
                    "embedding_provider": "sentence_transformer",
                    "embedding_model": "different-model",
                    "embedding_dim": 384,
                    "normalize_embeddings": True,
                },
            )

    def test_embedding_fail_fast_blocks_silent_fallback(self) -> None:
        with self.assertRaises(RuntimeError):
            EmbeddingProvider(
                "sentence-transformers/model-that-should-not-exist-for-guard-test",
                fail_fast=True,
            )

    def test_embedding_fallback_requires_explicit_allow_env(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            with self.assertRaises(RuntimeError):
                EmbeddingProvider(
                    "sentence-transformers/model-that-should-not-exist-for-guard-test",
                    fail_fast=False,
                )
        with patch.dict("os.environ", {"ALLOW_EMBEDDING_FALLBACK": "1"}, clear=False):
            provider = EmbeddingProvider(
                "sentence-transformers/model-that-should-not-exist-for-guard-test",
                fallback_dim=384,
                fail_fast=False,
            )
            self.assertTrue(provider.info.fallback_in_use)
            provider.close()

    def test_embedding_cache_key_is_bound_to_model_dim_and_preprocess(self) -> None:
        with patch.dict("os.environ", {"ALLOW_EMBEDDING_FALLBACK": "1"}, clear=False):
            provider_384 = EmbeddingProvider(
                "sentence-transformers/model-that-should-not-exist-for-guard-test",
                fallback_dim=384,
                fail_fast=False,
            )
            provider_512 = EmbeddingProvider(
                "sentence-transformers/model-that-should-not-exist-for-guard-test",
                fallback_dim=512,
                fail_fast=False,
            )
            key_384 = provider_384._cache_key("hash-1")
            key_512 = provider_512._cache_key("hash-1")
            self.assertNotEqual(key_384, key_512)
            self.assertIn("normalize_text_for_hash_v2", key_384)
            provider_384.close()
            provider_512.close()

    def test_json_vector_backend_and_answer_generator_work_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"ALLOW_EMBEDDING_FALLBACK": "1"}, clear=False):
                runtime = UnifiedMemoryRuntime.from_base_dir(
                    Path(tmp),
                    config_overrides={
                        "vector_backend": "json",
                        "embedding_fail_fast": False,
                        "enable_benchmark_route_tuning": False,
                        "enable_lightweight_edge_writeback": True,
                    },
                )
                node = MemoryNode(
                    shell=1,
                    sector="project",
                    zone="smoke",
                    cell="temporal",
                    molecular_type="decision",
                    summary="Use temporal edge retrieval to fix wrong-time memory drift.",
                    raw_content="Use temporal edge retrieval to fix wrong-time memory drift.",
                    verification_status="verified",
                )
                write = runtime.writeback_memory(node)
                self.assertGreaterEqual(write["chunk_count"], 1)
                answer = runtime.answer("What fixes wrong-time memory drift?", evidence_top_k=4)
                self.assertFalse(answer["abstained"])
                self.assertIn("temporal edge retrieval", answer["answer"].lower())
                self.assertTrue(answer["citations"])
                info = runtime.services.vector_store.info()
                self.assertEqual(info["vector_backend"], "json")
                self.assertIn("O(N) scan", info["json_scan_warning"])

    def test_retrieve_evidence_reports_variant_dedup_and_rank_pass_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"ALLOW_EMBEDDING_FALLBACK": "1"}, clear=False):
                runtime = UnifiedMemoryRuntime.from_base_dir(
                    Path(tmp),
                    config_overrides={
                        "vector_backend": "json",
                        "embedding_fail_fast": False,
                        "enable_retrieval_cache": False,
                        "enable_object_shortcut": False,
                        "enable_benchmark_route_tuning": False,
                    },
                )
                runtime.writeback_memory(
                    MemoryNode(
                        shell=1,
                        sector="project",
                        zone="retrieval",
                        cell="optimization",
                        molecular_type="fact",
                        summary="Melanie prefers strong coffee in the workshop.",
                        raw_content="Melanie prefers strong coffee in the workshop.",
                    )
                )
                evidence = runtime.retrieve_evidence(
                    "What coffee does Melanie prefer?",
                    top_k=3,
                    route_context={"focused_query": "What coffee does Melanie prefer?"},
                )
                diagnostics = evidence.diagnostics
                self.assertGreaterEqual(int(diagnostics.get("query_variant_count_before_dedup") or 0), int(diagnostics.get("query_variant_count_after_dedup") or 0))
                self.assertGreaterEqual(int(diagnostics.get("rank_pass_count") or 0), 1)
                self.assertIn("feature_cache_hits", diagnostics)
                self.assertIn("feature_cache_misses", diagnostics)

    def test_object_shortcut_hit_skips_dense_sparse_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"ALLOW_EMBEDDING_FALLBACK": "1"}, clear=False):
                runtime = UnifiedMemoryRuntime.from_base_dir(
                    Path(tmp),
                    config_overrides={
                        "vector_backend": "json",
                        "embedding_fail_fast": False,
                        "enable_retrieval_cache": False,
                        "enable_object_shortcut": True,
                        "enable_benchmark_route_tuning": False,
                    },
                )

                def fail_search(*args, **kwargs):
                    raise AssertionError("dense search should be skipped on sufficient object shortcut")

                runtime.evidence.vector_store.search = fail_search
                runtime.evidence._sparse_chunk_hits = fail_search
                runtime.evidence._object_shortcut = lambda *args, **kwargs: {
                    "reason": "shortcut_sufficient",
                    "shortcut_hit": True,
                    "snapshot_type": "preference",
                    "objects": [
                        {
                            "object_id": "obj-1",
                            "object_type": "preference",
                            "object_text": "Melanie prefers strong coffee.",
                            "object_score": 0.92,
                        }
                    ],
                    "candidates": [
                        {
                            "chunk_id": "shortcut-chunk-1",
                            "node_id": "node-1",
                            "text": "Melanie prefers strong coffee.",
                            "metadata": {"source_node_id": "node-1", "object_type": "preference"},
                            "similarity": 0.94,
                            "rrf_score": 1.0,
                            "object_support_score": 0.92,
                        }
                    ],
                }
                evidence = runtime.retrieve_evidence("What does Melanie prefer?", top_k=3)
                self.assertEqual(evidence.timings_ms["dense_vector_ms"], 0.0)
                self.assertEqual(evidence.timings_ms["sparse_fts_ms"], 0.0)
                self.assertEqual(evidence.diagnostics["rank_pass_count"], 1)
                self.assertEqual(evidence.diagnostics["decisions"]["object_support"], "skipped_shortcut_sufficient")

    def test_vector_backend_chroma_mode_fails_fast_without_chromadb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("importlib.import_module", side_effect=ModuleNotFoundError("chromadb")):
                with self.assertRaises(RuntimeError):
                    _build_chroma_client(str(Path(tmp)), backend="chroma")

    def test_json_vector_backend_guard_warns_or_fails_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"ALLOW_EMBEDDING_FALLBACK": "1"}, clear=False):
                config = AppConfig(
                    base_dir=Path(tmp),
                    vector_backend="json",
                    json_vector_max_items=1,
                    vector_fail_fast_on_fallback=False,
                    embedding_fail_fast=False,
                )
                store = VectorStore(config)
                store.raw_collection._client._state[store.raw_collection.name] = {
                    "a": {"document": "a", "metadata": {}, "embedding": [0.0]},
                    "b": {"document": "b", "metadata": {}, "embedding": [1.0]},
                }
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    store._enforce_json_backend_guard()
                self.assertTrue(any(issubclass(item.category, RuntimeWarning) for item in caught))
                self.assertIn("O(N) scan", store.info()["json_scan_warning"])
                store.close()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"ALLOW_EMBEDDING_FALLBACK": "1"}, clear=False):
                config = AppConfig(
                    base_dir=Path(tmp),
                    vector_backend="json",
                    json_vector_max_items=1,
                    vector_fail_fast_on_fallback=True,
                    embedding_fail_fast=False,
                )
                store = VectorStore(config)
                store.raw_collection._client._state[store.raw_collection.name] = {
                    "a": {"document": "a", "metadata": {}, "embedding": [0.0]},
                    "b": {"document": "b", "metadata": {}, "embedding": [1.0]},
                }
                with self.assertRaises(RuntimeError):
                    store._enforce_json_backend_guard()
                store.close()

    def test_vector_search_by_embedding_matches_query_interfaces_and_prefilters_proxy_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"ALLOW_EMBEDDING_FALLBACK": "1"}, clear=False):
                config = AppConfig(
                    base_dir=Path(tmp),
                    vector_backend="json",
                    vector_fail_fast_on_fallback=False,
                    embedding_fail_fast=False,
                )
                store = VectorStore(config)
                query = "Melanie prefers strong coffee"
                raw_name = store.raw_collection.name
                object_name = store.object_collection.name
                proxy_name = store.proxy_collection.name
                store.raw_collection._client._state[raw_name] = {
                    "chunk-1": {
                        "document": query,
                        "metadata": {"scope": "global"},
                        "embedding": store.embedder.embed(query),
                    }
                }
                store.object_collection._client._state[object_name] = {
                    "object-1": {
                        "document": query,
                        "metadata": {"scope": "global"},
                        "embedding": store.embedder.embed(query),
                    }
                }
                store.proxy_collection._client._state[proxy_name] = {
                    "proxy-1": {
                        "document": query,
                        "metadata": {"proxy_kind": "summary", "scope": "global"},
                        "embedding": store.embedder.embed(query),
                    },
                    "proxy-2": {
                        "document": "unrelated preference",
                        "metadata": {"proxy_kind": "reflection", "scope": "global"},
                        "embedding": store.embedder.embed("unrelated preference"),
                    },
                }

                embedding = store.embed_query(query)
                self.assertEqual(
                    [row["chunk_id"] for row in store.search(query, top_k=1)],
                    [row["chunk_id"] for row in store.search_by_embedding(embedding, top_k=1)],
                )
                self.assertEqual(
                    [row["object_id"] for row in store.search_objects(query, top_k=1)],
                    [row["object_id"] for row in store.search_objects_by_embedding(embedding, top_k=1)],
                )
                proxy_rows = store.search_proxies_by_embedding(embedding, top_k=2, proxy_kinds=["summary"])
                self.assertEqual([row["representation_id"] for row in proxy_rows], ["proxy-1"])
                counters = store.snapshot_stats()["counters"]
                self.assertGreaterEqual(int(counters.get("prefilter_applied", 0)), 1)
                self.assertGreaterEqual(int(counters.get("proxy_search_count", 0)), 1)
                store.close()

    def test_fetch_preferred_chunks_for_nodes_matches_legacy_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(base_dir=Path(tmp))
            config.ensure_dirs()
            storage = Storage(config)
            storage.init_db()
            storage.insert_nodes(
                [
                    {
                        "id": "node-a",
                        "shell": 1,
                        "sector": "knowledge",
                        "zone": "z",
                        "cell": "c",
                        "molecular_type": "fact",
                        "summary": "Node A",
                        "raw_content": "Node A raw",
                        "scope": "global",
                        "workspace": "ws",
                        "project": "proj",
                        "session_id": "sess",
                    },
                    {
                        "id": "node-b",
                        "shell": 1,
                        "sector": "knowledge",
                        "zone": "z",
                        "cell": "c",
                        "molecular_type": "fact",
                        "summary": "Node B",
                        "raw_content": "Node B raw",
                    },
                ]
            )
            storage.insert_chunks(
                [
                    {"chunk_id": "a-macro", "node_id": "node-a", "chunk_index": 0, "grain": "macro", "text": "macro"},
                    {"chunk_id": "a-micro", "node_id": "node-a", "chunk_index": 1, "grain": "micro", "text": "micro"},
                    {"chunk_id": "b-macro", "node_id": "node-b", "chunk_index": 0, "grain": "macro", "text": "only macro"},
                ]
            )
            batch = storage.fetch_preferred_chunks_for_nodes(["node-a", "node-b", "missing"])
            legacy_a = storage.fetch_chunks_for_node("node-a")
            preferred_a = ([chunk for chunk in legacy_a if str(chunk.get("grain") or "") != "macro"] or legacy_a)[0]

            self.assertEqual(batch["node-a"]["chunk_id"], preferred_a["chunk_id"])
            self.assertEqual(batch["node-b"]["chunk_id"], "b-macro")
            self.assertEqual(batch["node-a"]["node_workspace"], "ws")
            self.assertNotIn("missing", batch)

    def test_lightweight_edges_are_written_after_second_related_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"ALLOW_EMBEDDING_FALLBACK": "1"}, clear=False):
                runtime = UnifiedMemoryRuntime.from_base_dir(
                    Path(tmp),
                    config_overrides={
                        "vector_backend": "json",
                        "embedding_fail_fast": False,
                        "enable_lightweight_edge_writeback": True,
                    },
                )
                for text in (
                    "Use temporal edge retrieval for temporal anchoring drift.",
                    "Use competition-aware inhibition for local candidate crowding.",
                ):
                    runtime.writeback_memory(
                        MemoryNode(
                            shell=1,
                            sector="project",
                            zone="reranking",
                            cell="decision",
                            molecular_type="decision",
                            summary=text,
                            raw_content=text,
                        )
                    )
                self.assertGreaterEqual(runtime.services.storage.count_edges(), 1)

    def test_integrity_report_marks_missing_gold_and_chunk_mapping_bugs(self) -> None:
        report = build_integrity_report(
            benchmark_name="clonemem",
            raw_counts={
                "memory_count": 2,
                "question_count": 3,
                "raw_document_count": 2,
                "raw_session_count": 1,
                "raw_segment_count": 2,
                "empty_text_count": 1,
                "timestamp_field_count": 2,
                "timestamp_parseable_count": 1,
                "timestamp_parse_rate": 0.5,
            },
            index_metadata={
                "index_doc_count": 1,
                "chunk_count": 2,
                "unique_segment_count": 1,
                "indexed_doc_ids": ["sample-1"],
                "indexed_segment_ids": ["seg-1"],
                "chunk_metadata_by_id": {
                    "chunk-1": {"source_segment_id": "seg-1", "source_doc_id": "sample-1", "benchmark_name": "clonemem"},
                    "chunk-2": {"source_segment_id": "", "source_doc_id": "sample-1", "benchmark_name": "wrong-benchmark"},
                },
                "fingerprint": {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
            },
            gold_segment_ids={"seg-1", "seg-2"},
            gold_document_ids={"sample-1"},
        )
        self.assertIn("gold_segment_not_in_index", report["p0_bugs"])
        self.assertIn("chunk_missing_source_segment_id", report["p0_bugs"])
        self.assertIn("chunk_benchmark_name_mismatch", report["p0_bugs"])
        self.assertEqual(report["memory_count"], 2)
        self.assertEqual(report["question_count"], 3)
        self.assertEqual(report["empty_text_count"], 1)
        self.assertEqual(report["timestamp_parse_rate"], 0.5)

    def test_failure_summary_tracks_representative_failures(self) -> None:
        failure = build_query_failure(
            benchmark_name="locomo",
            query_id="q1",
            query_text="What happened after the first session?",
            answer_text="answer",
            gold_segment_ids={"session_9"},
            gold_evidence_ids={"session_9"},
            broad_rows=[
                {"source_id": "session_9", "source_retrievers": ["dense", "bm25"]},
                {"source_id": "session_4", "source_retrievers": ["dense"]},
            ],
            reranked_rows=[
                {"source_id": "session_4", "source_retrievers": ["dense"]},
            ],
            final_rows=[
                {"source_id": "session_4", "source_retrievers": ["dense"]},
            ],
            index_metadata={
                "indexed_segment_ids": ["session_9"],
                "chunk_metadata_by_id": {
                    "chunk-1": {"source_segment_id": "session_9", "source_doc_id": "session_9"},
                },
            },
        )
        self.assertIsNotNone(failure)
        self.assertEqual(failure["failure_type"], "reranker_dropped_gold")

        summary = build_candidate_recall_summary(
            benchmark_name="locomo",
            rows=[
                {
                    "query_id": "q1",
                    "query_text": "query",
                    "candidate_recall@10": 1.0,
                    "candidate_recall@50": 1.0,
                    "candidate_recall@100": 1.0,
                    "candidate_recall@200": 1.0,
                    "candidate_ndcg@10": 1.0,
                    "final_recall@10": 0.0,
                    "final_ndcg@10": 0.0,
                    "gold_rank_before_rerank": 1,
                    "gold_rank_after_rerank": None,
                    "failure_type": "reranker_dropped_gold",
                },
                {
                    "query_id": "q2",
                    "query_text": "query2",
                    "candidate_recall@10": 0.0,
                    "candidate_recall@50": 0.0,
                    "candidate_recall@100": 0.0,
                    "candidate_recall@200": 0.0,
                    "candidate_ndcg@10": 0.0,
                    "final_recall@10": 0.0,
                    "final_ndcg@10": 0.0,
                    "gold_rank_before_rerank": None,
                    "gold_rank_after_rerank": None,
                    "failure_type": "gold_missing_from_index",
                },
            ],
        )
        self.assertEqual(summary["reranker_dropped_gold_count"], 1)
        self.assertEqual(summary["gold_missing_from_index_count"], 1)
        self.assertIn("reranker_dropped_gold", summary["representative_failures"])

    def test_build_raw_counts_and_topk_debug_preserve_ingest_and_rank_details(self) -> None:
        raw_counts = build_raw_counts(
            [
                {"corpus_id": "seg-1", "source_segment_id": "seg-1", "source_doc_id": "doc-1", "text": "alpha", "timestamp": "2026-04-01"},
                {"corpus_id": "seg-2", "source_segment_id": "seg-2", "source_doc_id": "doc-1", "text": "", "timestamp": "bad-timestamp"},
            ],
            question_count=5,
            session_count=1,
        )
        self.assertEqual(raw_counts["memory_count"], 2)
        self.assertEqual(raw_counts["question_count"], 5)
        self.assertEqual(raw_counts["empty_text_count"], 1)
        self.assertEqual(raw_counts["timestamp_parseable_count"], 1)

        broad_rows = [
            {"source_id": "seg-gold", "benchmark_name": "locomo", "dense_score": 0.9, "bm25_score": 0.8, "semantic_score": 0.9, "task_score": 0.6, "cluster_id": "session-1"},
            {"source_id": "seg-wrong", "benchmark_name": "locomo", "dense_score": 0.7, "bm25_score": 0.4, "semantic_score": 0.7, "task_score": 0.3, "cluster_id": "session-2"},
        ]
        reranked_rows = [
            {"source_id": "seg-wrong", "benchmark_name": "locomo", "dense_score": 0.7, "semantic_score": 0.7, "task_score": 0.3, "cluster_id": "session-2"},
            {"source_id": "seg-gold", "benchmark_name": "locomo", "dense_score": 0.9, "semantic_score": 0.9, "task_score": 0.6, "cluster_id": "session-1"},
        ]
        final_rows = list(reranked_rows)
        diag = build_query_diagnostic(
            benchmark_name="locomo",
            query_id="q-debug",
            query_text="query",
            answer_text="answer",
            gold_segment_ids={"seg-gold"},
            gold_evidence_ids={"seg-gold"},
            broad_rows=broad_rows,
            reranked_rows=reranked_rows,
            final_rows=final_rows,
        )
        self.assertEqual(diag["gold_rank_before_rerank"], 1)
        self.assertEqual(diag["gold_rank_after_rerank"], 2)
        debug_row = build_topk_debug_record(
            benchmark_name="locomo",
            query_id="q-debug",
            query_text="query",
            answer_text="answer",
            gold_segment_ids={"seg-gold"},
            failure_type="reranker_dropped_gold",
            broad_rows=broad_rows,
            reranked_rows=reranked_rows,
            final_rows=final_rows,
        )
        self.assertEqual(debug_row["failure_type"], "reranker_dropped_gold")
        self.assertTrue(debug_row["topk_before_rerank"][0]["is_gold"])
        self.assertIn("semantic_score", debug_row["topk_before_rerank"][0])
        self.assertIn("task_score", debug_row["topk_before_rerank"][0])
        self.assertIn("cluster_id", debug_row["topk_before_rerank"][0])

    def test_empty_gold_mapping_is_reported_separately(self) -> None:
        failure = build_query_failure(
            benchmark_name="knowme",
            query_id="q-empty",
            query_text="What do I prioritize most?",
            answer_text="Fairness",
            gold_segment_ids=set(),
            gold_evidence_ids=set(),
            broad_rows=[{"source_id": "seg-1", "source_retrievers": ["profile_fact"]}],
            reranked_rows=[{"source_id": "seg-1", "source_retrievers": ["profile_fact"]}],
            final_rows=[{"source_id": "seg-1", "source_retrievers": ["profile_fact"]}],
            index_metadata={
                "indexed_segment_ids": ["seg-1"],
                "chunk_metadata_by_id": {
                    "chunk-1": {"source_segment_id": "seg-1", "source_doc_id": "doc-1"},
                },
            },
        )
        self.assertIsNotNone(failure)
        self.assertEqual(failure["failure_type"], "query_gold_mapping_empty")

    def test_broad_candidate_pool_interleaves_sources(self) -> None:
        dense = {
            "d1": self._candidate("d1", dense_score=0.95, broad_score=0.95),
            "d2": self._candidate("d2", dense_score=0.91, broad_score=0.91),
        }
        lexical = {
            "l1": self._candidate("l1", bm25_score=0.88, broad_score=0.88),
            "l2": self._candidate("l2", bm25_score=0.84, broad_score=0.84),
        }
        entity = {
            "e1": self._candidate("e1", entity_score=0.8, broad_score=0.8),
        }
        merged = merge_candidate_sources(dense, lexical, entity)
        broad = assemble_broad_candidate_pool(
            benchmark_name="clonemem",
            merged=merged,
            dense=dense,
            lexical=lexical,
            entity=entity,
            temporal={},
            profile={},
            session={},
            exact={},
            local_window={},
            limit=5,
        )
        self.assertEqual([row["source_id"] for row in broad[:3]], ["e1", "d1", "l1"])

    def test_local_window_expansion_scales_with_seed_strength(self) -> None:
        source_records = {
            "seg-1": self._candidate("seg-1", source_doc_id="doc-1", order_index=1),
            "seg-2": self._candidate("seg-2", source_doc_id="doc-1", order_index=2),
            "seg-3": self._candidate("seg-3", source_doc_id="doc-1", order_index=3),
        }
        weak_seed = {
            "seg-2": self._candidate("seg-2", source_doc_id="doc-1", order_index=2, dense_score=0.15),
        }
        expanded = _local_window_source_candidates(
            seed_candidates=weak_seed,
            source_records=source_records,
            limit=10,
        )
        self.assertLess(float(expanded["seg-1"]["local_window_score"]), 0.2)
        self.assertLess(float(expanded["seg-3"]["local_window_score"]), 0.2)

    def test_session_bundle_scores_are_moderate_for_session_level_sources(self) -> None:
        source_records = {
            "session-1": self._candidate(
                "session-1",
                source_doc_id="session-1",
                session_id="session-1",
                order_index=0,
            ),
            "session-2": self._candidate(
                "session-2",
                source_doc_id="session-2",
                session_id="session-2",
                order_index=1,
            ),
        }
        seeds = {
            "session-1": self._candidate(
                "session-1",
                source_doc_id="session-1",
                session_id="session-1",
                order_index=0,
                dense_score=0.55,
                entity_score=0.42,
                source_chunk_ids=["chunk-1", "chunk-2", "chunk-3"],
            ),
            "session-2": self._candidate(
                "session-2",
                source_doc_id="session-2",
                session_id="session-2",
                order_index=1,
                dense_score=0.24,
                entity_score=0.2,
                source_chunk_ids=["chunk-4"],
            ),
        }
        expanded = _session_bundle_source_candidates(
            seed_candidates=seeds,
            source_records=source_records,
            limit=10,
        )
        self.assertLess(float(expanded["session-1"]["session_score"]), 0.4)
        self.assertLess(float(expanded["session-2"]["session_score"]), 0.4)
        self.assertGreater(float(expanded["session-1"]["session_score"]), float(expanded["session-2"]["session_score"]))

    def test_entity_overlap_downweights_person_only_hits_without_anchor_support(self) -> None:
        query_features = _query_features("What activity did Melanie used to do with her dad?")
        source_records = {
            "seg-good": self._candidate(
                "seg-good",
                text="Melanie used to go fishing with her dad every summer.",
                normalized_text="melanie used to go fishing with her dad every summer.",
                token_list=["melanie", "used", "fishing", "dad", "summer"],
                entity_terms=["melanie"],
            ),
            "seg-wrong": self._candidate(
                "seg-wrong",
                text="Melanie mentioned buying groceries after work.",
                normalized_text="melanie mentioned buying groceries after work.",
                token_list=["melanie", "buying", "groceries", "work"],
                entity_terms=["melanie"],
            ),
        }
        candidates = _entity_source_candidates(
            query_features=query_features,
            source_records=source_records,
            limit=5,
        )
        self.assertIn("seg-good", candidates)
        self.assertIn("seg-wrong", candidates)
        self.assertGreater(
            float(candidates["seg-good"]["entity_score"]),
            float(candidates["seg-wrong"]["entity_score"]),
        )

    def test_exact_phrase_candidates_help_knowme_style_anchor_queries(self) -> None:
        query = 'The title of the third poem in "Comfort for That Wound" is "Comfort for That Wound".'
        query_features = _query_features(query)
        source_records = {
            "seg-good": self._candidate(
                "seg-good",
                benchmark_name="knowme",
                text="She introduced herself and said her debut poetry collection was titled Comfort Me for That Wound.",
                normalized_text="she introduced herself and said her debut poetry collection was titled comfort me for that wound.",
                token_list=["debut", "poetry", "collection", "comfort", "wound"],
            ),
            "seg-wrong": self._candidate(
                "seg-wrong",
                benchmark_name="knowme",
                text="Years later she talked about raising three children in another apartment.",
                normalized_text="years later she talked about raising three children in another apartment.",
                token_list=["years", "later", "raising", "children", "apartment"],
            ),
        }
        exact = _exact_phrase_source_candidates(
            query=query,
            query_features=query_features,
            source_records=source_records,
            limit=5,
        )
        self.assertIn("seg-good", exact)
        self.assertGreater(
            float(exact["seg-good"]["exact_phrase_score"]),
            float(exact.get("seg-wrong", {}).get("exact_phrase_score", 0.0)),
        )

    def test_cjk_query_support_improves_clone_phrase_and_entity_matching(self) -> None:
        query = "桂林，这两个月来你是不是对退休和传承越想越重了？"
        source_records = {
            "seg-good": self._candidate(
                "seg-good",
                text="陈桂林最近总在想退休和传承的问题，觉得时间越来越紧。",
                normalized_text="陈桂林最近总在想退休和传承的问题，觉得时间越来越紧。",
            ),
            "seg-wrong": self._candidate(
                "seg-wrong",
                text="他只是在准备一次普通的体检，没有提到这些焦虑。",
                normalized_text="他只是在准备一次普通的体检，没有提到这些焦虑。",
            ),
        }
        query_features = _query_features(
            query,
            route_context={"person_name": "陈桂林", "question_time": "2022-06-03T10:30:00"},
        )
        exact = _exact_phrase_source_candidates(
            query=query,
            query_features=query_features,
            source_records=source_records,
            limit=5,
        )
        entity = _entity_source_candidates(
            query_features=query_features,
            source_records=source_records,
            limit=5,
        )
        self.assertIn("seg-good", exact)
        self.assertIn("seg-good", entity)
        self.assertGreater(
            float(exact["seg-good"]["exact_phrase_score"]),
            float(exact.get("seg-wrong", {}).get("exact_phrase_score", 0.0)),
        )
        self.assertGreater(
            float(entity["seg-good"]["entity_score"]),
            float(entity.get("seg-wrong", {}).get("entity_score", 0.0)),
        )

    def test_knowme_loader_remaps_missing_evidence_ids_from_textual_clues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset1"
            (dataset_dir / "input").mkdir(parents=True)
            (dataset_dir / "question").mkdir()
            (dataset_dir / "answer").mkdir()
            (dataset_dir / "input" / "dataset1.json").write_text(
                (
                    '[{"id": 2031, "timestamp": "1977-12-03 16:20:00", "location": "kitchen", '
                    '"action": "He took a bag of green cod from the fridge and set it on the counter.", '
                    '"dialogue": null, "environment": null, "background": null, "inner_thought": null}]'
                ),
                encoding="utf-8",
            )
            (dataset_dir / "question" / "Information Extraction_questions.json").write_text(
                '[{"id": 39, "question": "On July 17, 1976, what kind of fish did the narrator catch the most?"}]',
                encoding="utf-8",
            )
            (dataset_dir / "answer" / "Information Extraction_answers.json").write_text(
                '[{"id": 39, "answer": "Green cod", "evidence": 1764}]',
                encoding="utf-8",
            )
            bundle = load_knowme_dataset_bundle(dataset_dir)
            self.assertEqual(bundle["questions"][0]["evidence_ids"], {"2031"})
            self.assertEqual(bundle["questions"][0]["gold_mapping"]["status"], "remapped")

    def test_shard_assignment_is_deterministic_and_complete(self) -> None:
        items = [
            {"question_id": f"q-{idx}", "sample_id": "sample-a", "question": f"Question {idx}"}
            for idx in range(40)
        ]
        assignments: dict[str, int] = {}
        merged_ids: list[str] = []
        for shard_index in range(5):
            selected, meta = filter_sharded_items(
                items,
                benchmark_name="knowme",
                shard_index=shard_index,
                shard_count=5,
                question_id_getter=lambda item: item["question_id"],
                sample_id_getter=lambda item: item["sample_id"],
                question_text_getter=lambda item: item["question"],
            )
            self.assertEqual(meta["shard_assignment_method"], "question_id_hash")
            for item in selected:
                qid = item["question_id"]
                self.assertNotIn(qid, assignments)
                assignments[qid] = shard_index
                merged_ids.append(qid)
                again, _ = belongs_to_shard(
                    benchmark_name="knowme",
                    question_id=qid,
                    sample_id=item["sample_id"],
                    question_text=item["question"],
                    shard_index=shard_index,
                    shard_count=5,
                )
                self.assertTrue(again)
        self.assertEqual(set(merged_ids), {item["question_id"] for item in items})

    def test_clonemem_sample_shard_helpers_are_deterministic_and_complete(self) -> None:
        sample_ids = [f"sample-{idx:02d}" for idx in range(17)]
        first = split_round_robin(sample_ids, 5)
        second = split_round_robin(list(reversed(sample_ids)), 5)
        self.assertEqual(first, second)
        flattened = [sample_id for bucket in first for sample_id in bucket]
        self.assertEqual(set(flattened), set(sample_ids))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(detect_clonemem_sample_id(Path("alice_benchmark_en.json")), "alice")
        self.assertEqual(detect_clonemem_sample_id(Path("bob_benchmark_zh.json")), "bob")

    def test_clonemem_sample_shard_command_uses_allowlist_without_question_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunk_dir = Path(tmp) / "chunk_00"
            chunk_dir.mkdir()
            allowlist = chunk_dir / "sample_id_allowlist.txt"
            allowlist.write_text("sample-a\n", encoding="utf-8")
            args = SimpleNamespace(
                benchmark="clonemem",
                python_exe=sys.executable,
                data_root=Path(tmp),
                mode="evidence",
                top_k=50,
                rerank_mode="rule",
                granularity="session",
                context_len="all",
                language="all",
                chunks=8,
                max_questions=0,
                resume=True,
                shard_strategy="sample",
            )
            command = benchmark_command(args, 0, chunk_dir / "metrics.json", chunk_dir)
        self.assertIn("--sample-id-allowlist", command)
        self.assertIn(str(allowlist), command)
        self.assertNotIn("--shard-index", command)
        self.assertNotIn("--shard-count", command)

    def test_knowme_official_formal_profile_pins_capped_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = SimpleNamespace(benchmark="knowme", knowme_profile="official_formal", out=out)
            workspace_root, embed_root = benchmark_cache_roots(args)
        profile_env = benchmark_profile_env(args)
        self.assertEqual(profile_env, KNOWME_OFFICIAL_FORMAL_ENV)
        self.assertEqual(profile_env["SPHERE_ORACLE_RETRIEVAL_MODE"], "direct_index")
        self.assertEqual(profile_env["SPHERE_KNOWME_POOL_LIMIT"], "100")
        self.assertEqual(profile_env["SPHERE_PARENT_TOP_K"], "10")
        self.assertEqual(workspace_root, out / "cache_workspace")
        self.assertEqual(embed_root, out / "cache_embedding")

        default_args = SimpleNamespace(benchmark="knowme", knowme_profile="default", out=Path("/tmp/unused"))
        self.assertEqual(benchmark_profile_env(default_args), {})

    def test_empty_clonemem_sample_allowlist_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            allowlist = Path(tmp) / "sample_id_allowlist.txt"
            allowlist.write_text("\n", encoding="utf-8")
            nonempty = Path(tmp) / "nonempty.txt"
            nonempty.write_text("sample-a\n", encoding="utf-8")

            self.assertTrue(sample_allowlist_is_empty(allowlist))
            self.assertFalse(sample_allowlist_is_empty(nonempty))
            self.assertFalse(sample_allowlist_is_empty(None))

    def test_merge_benchmark_results_reaggregates_question_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "a.json"
            shard_b = root / "b.json"
            shard_a.write_text(
                json.dumps(
                    {
                        "benchmark_name": "knowme",
                        "question_count": 2,
                        "elapsed_seconds": 10,
                        "results": [
                            {"question_id": "q1", "metrics": {"recall_frac@10": 1.0, "ndcg_any@10": 1.0}, "candidate_recall": {"candidate_recall@100": 1.0, "failure_type": "ok"}},
                            {"question_id": "q2", "metrics": {"recall_frac@10": 0.0, "ndcg_any@10": 0.0}, "candidate_recall": {"candidate_recall@100": 1.0, "failure_type": "miss"}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            shard_b.write_text(
                json.dumps(
                    {
                        "benchmark_name": "knowme",
                        "question_count": 1,
                        "elapsed_seconds": 5,
                        "results": [
                            {"question_id": "q3", "metrics": {"recall_frac@10": 1.0, "ndcg_any@10": 0.5}, "candidate_recall": {"candidate_recall@100": 0.0, "failure_type": "miss"}}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            merged = merge_payloads([shard_a, shard_b])
            self.assertEqual(merged["total_question_count"], 3)
            self.assertAlmostEqual(merged["metrics"]["recall_frac@10"], 2 / 3)
            self.assertAlmostEqual(merged["candidate_recall_summary"]["candidate_recall@100"], 2 / 3)
            self.assertEqual(merged["failure_taxonomy"]["counts"]["miss"], 2)

    def test_runtime_parallel_channels_default_off_and_budget_diagnostics(self) -> None:
        config = AppConfig(base_dir=Path.cwd())
        self.assertFalse(config.runtime_parallel_channels_enabled)
        self.assertEqual(config.runtime_retrieval_latency_budget_ms, 0)

    def test_chunked_runner_resume_validation_accepts_existing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics.json"
            self.assertFalse(validate_metrics(metrics))
            metrics.write_text(json.dumps({"question_count": 1, "results": [{"question_id": "q1"}]}), encoding="utf-8")
            self.assertTrue(validate_metrics(metrics))


if __name__ == "__main__":
    unittest.main()
