from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sphere_cli.embedding import EmbeddingProvider
from sphere_cli.models import MemoryNode
from sphere_cli.runtime import UnifiedMemoryRuntime


class GuardrailTests(unittest.TestCase):
    def test_embedding_fail_fast_blocks_silent_fallback(self) -> None:
        with self.assertRaises(RuntimeError):
            EmbeddingProvider(
                "sentence-transformers/model-that-should-not-exist-for-guard-test",
                fail_fast=True,
            )

    def test_json_vector_backend_and_answer_generator_work_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
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

    def test_lightweight_edges_are_written_after_second_related_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
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


if __name__ == "__main__":
    unittest.main()
