from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "base"
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
for candidate in (BASE_DIR, BENCHMARKS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import convomem_benchmark


class ConvoMemBenchmarkStreamingTests(unittest.TestCase):
    def test_iter_cases_streams_across_small_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_file = Path(tmp) / "batched_000.json"
            json_file.write_text(
                json.dumps(
                    [
                        {"id": "case-1", "message": "brace { inside } text"},
                        {"id": "case-2", "message": 'escaped "quote" and comma, text'},
                    ]
                ),
                encoding="utf-8",
            )

            original_chunk_size = convomem_benchmark.JSON_STREAM_CHUNK_SIZE
            convomem_benchmark.JSON_STREAM_CHUNK_SIZE = 8
            try:
                rows = list(convomem_benchmark.iter_cases(json_file))
            finally:
                convomem_benchmark.JSON_STREAM_CHUNK_SIZE = original_chunk_size

        self.assertEqual([row[0]["id"] for row in rows], ["case-1", "case-2"])
        self.assertGreaterEqual(rows[0][1], 0.0)
        self.assertEqual(rows[1][1], 0.0)

    def test_iter_cases_limit_stops_before_later_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "convomem_cases"
            data_dir.mkdir()
            (data_dir / "batched_000.json").write_text(
                json.dumps([{"id": "case-1"}, {"id": "case-2"}]),
                encoding="utf-8",
            )
            (data_dir / "batched_001.json").write_text("{", encoding="utf-8")

            rows = list(convomem_benchmark.iter_cases(data_dir, limit=1))

        self.assertEqual(len(rows), 1)
        case, load_ms = rows[0]
        self.assertEqual(case["id"], "case-1")
        self.assertTrue(str(case["_source_file"]).endswith("batched_000.json"))
        self.assertGreaterEqual(load_ms, 0.0)

    def test_write_streamed_payload_rehydrates_results_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "convomem_metrics.json"
            spool = convomem_benchmark.create_result_spool(out_file)
            spool.append({"case_id": "case-1", "question_index": 1, "metrics": {"recall_frac@10": 1.0}})
            spool.append({"case_id": "case-2", "question_index": 2, "metrics": {"recall_frac@10": 0.5}})
            spool.close()

            payload = {
                "case_count": 2,
                "question_count": 2,
                "result_count": 2,
                "benchmark_io_ms": {"load_data_ms": 12.5},
            }

            serialize_ms, write_ms = convomem_benchmark.write_streamed_payload(
                out_file,
                payload,
                spool.path,
            )
            written = json.loads(out_file.read_text(encoding="utf-8"))

            try:
                spool.path.unlink()
            except FileNotFoundError:
                pass

        self.assertGreaterEqual(serialize_ms, 0.0)
        self.assertGreaterEqual(write_ms, 0.0)
        self.assertEqual(written["case_count"], 2)
        self.assertEqual(written["result_count"], 2)
        self.assertEqual(len(written["results"]), 2)
        self.assertEqual(written["results"][0]["case_id"], "case-1")
        self.assertEqual(written["results"][1]["question_index"], 2)

    def test_run_benchmark_reuses_workspace_within_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "batched_000.json"
            data_path.write_text(
                json.dumps(
                    [
                        {
                            "contextSize": 2,
                            "conversations": [
                                {
                                    "id": "conv-1",
                                    "messages": [
                                        {"speaker": "user", "text": "alpha"},
                                        {"speaker": "assistant", "text": "beta"},
                                    ],
                                }
                            ],
                            "evidenceItems": [
                                {"question": "q1", "answer": "a1", "category": "user"},
                                {"question": "q2", "answer": "a2", "category": "user"},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            ingest_profile = {
                "counts": {"chunks": 2, "nodes": 2, "objects": 0},
                "timings_ms": {"prepare_chunks_ms": 1.0, "extract_objects_ms": 1.0},
                "dedup": {},
                "backend": {
                    "storage": {"total_ms": 0.0, "calls": 0, "rows": 0, "ops": {}},
                    "vector": {"total_ms": 0.0, "calls": 0, "rows": 0, "ops": {}},
                    "vector_counters": {},
                    "embedding_cache": {},
                },
            }

            class FakeWorkspaceManager:
                acquire_calls = 0
                release_calls = 0

                def __init__(self, *_args, **_kwargs) -> None:
                    self._workspace = SimpleNamespace(
                        signature="fake-signature",
                        build_elapsed_ms=12.0,
                        ingest_profile=ingest_profile,
                        vector_info={"embedding_model": "fake", "embedding_provider": "fake"},
                        config=SimpleNamespace(
                            embedding_model_name="fake",
                            embed_local_grain=False,
                            rerank_mode_default="rule",
                            cross_encoder_model_name=None,
                            creative_mode_name="off",
                            creative_beam_width=0,
                            creative_max_hops=0,
                            creative_neighbors_per_hop=0,
                            creative_max_output_paths=0,
                        ),
                        pipeline=SimpleNamespace(cross_encoder=None),
                        activation=object(),
                        router=object(),
                        reranker=object(),
                        vector_store=object(),
                    )

                def acquire(self, **_kwargs):
                    FakeWorkspaceManager.acquire_calls += 1
                    return self._workspace, {
                        "workspace_reused": False,
                        "workspace_reused_from_disk": False,
                        "cache_reuse_saved_ms": 0.0,
                        "ingest_lookup_ms": 0.0,
                    }

                def release(self, _signature: str, collect_garbage: bool = True) -> None:
                    FakeWorkspaceManager.release_calls += 1

                def close_all(self) -> None:
                    return None

            def fake_rank_evidence(*_args, **_kwargs):
                return (
                    ["conv-1:m0001"],
                    [{"corpus_id": "conv-1:m0001"}],
                    {"total_ms": 1.0, "completion_total_ms": 0.0, "cognitive_total_ms": 0.0},
                    {"retrieval": {"candidate_counts": {}}, "completion": {"candidate_counts": {}}},
                )

            with patch.object(convomem_benchmark, "BenchmarkWorkspaceManager", FakeWorkspaceManager), patch.object(
                convomem_benchmark,
                "rank_evidence",
                side_effect=fake_rank_evidence,
            ):
                payload = convomem_benchmark.run_benchmark(
                    data_path=data_path,
                    mode="evidence",
                    top_k=10,
                    limit=1,
                    rerank_mode="rule",
                    shell=2,
                    sector="knowledge",
                    zone="convomem",
                    chunk_pool=10,
                    out_file=None,
                )

        self.assertEqual(payload["question_count"], 2)
        self.assertEqual(FakeWorkspaceManager.acquire_calls, 1)
        self.assertEqual(FakeWorkspaceManager.release_calls, 1)
        self.assertEqual(
            payload["results"][1]["stage_timing_ms"]["service_init_ms"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
