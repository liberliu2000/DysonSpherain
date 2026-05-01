from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "base") not in sys.path:
    sys.path.insert(0, str(ROOT / "base"))
if str(ROOT / "base" / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "base" / "benchmarks"))

import benchmark_support


class OracleRetrievalReportTests(unittest.TestCase):
    def test_builds_self_retrieval_rows_from_index_records(self) -> None:
        original_rank = benchmark_support.rank_benchmark_sources

        def fake_rank_benchmark_sources(**kwargs):
            self.assertEqual(kwargs["query"], "gold text")
            return {
                "final_candidates": [
                    {"source_id": "other", "text": "other"},
                    {"source_id": "gold-1", "text": "gold text"},
                ]
            }

        benchmark_support.rank_benchmark_sources = fake_rank_benchmark_sources
        try:
            report = benchmark_support.build_oracle_retrieval_report(
                benchmark_name="knowme",
                oracle_items=[
                    {
                        "query_id": "q1",
                        "sample_id": "dataset001",
                        "question_type": "Information Extraction",
                        "gold_segment_ids": {"gold-1"},
                    }
                ],
                vector_store=object(),
                storage=object(),
                index_metadata={
                    "source_records_by_id": {
                        "gold-1": {"text": "gold text"},
                    }
                },
                config=object(),
                pool_limit=10,
            )
        finally:
            benchmark_support.rank_benchmark_sources = original_rank

        self.assertEqual(report["oracle_query_count"], 1)
        self.assertEqual(report["oracle_recall@1"], 0.0)
        self.assertEqual(report["oracle_recall@5"], 1.0)
        self.assertEqual(report["oracle_recall@10"], 1.0)
        self.assertEqual(report["rows"][0]["self_rank"], 2)

    def test_reuses_self_retrieval_cache_for_duplicate_gold_segments(self) -> None:
        original_rank = benchmark_support.rank_benchmark_sources
        calls = []

        def fake_rank_benchmark_sources(**kwargs):
            calls.append(kwargs["query"])
            return {
                "final_candidates": [
                    {"source_id": "gold-1", "text": "gold text"},
                ]
            }

        benchmark_support.rank_benchmark_sources = fake_rank_benchmark_sources
        try:
            cache = {}
            report = benchmark_support.build_oracle_retrieval_report(
                benchmark_name="knowme",
                oracle_items=[
                    {"query_id": "q1", "sample_id": "dataset001", "question_type": "x", "gold_segment_ids": {"gold-1"}},
                    {"query_id": "q2", "sample_id": "dataset001", "question_type": "x", "gold_segment_ids": {"gold-1"}},
                ],
                vector_store=object(),
                storage=object(),
                index_metadata={"source_records_by_id": {"gold-1": {"text": "gold text"}}},
                config=object(),
                pool_limit=10,
                retrieval_cache=cache,
            )
        finally:
            benchmark_support.rank_benchmark_sources = original_rank

        self.assertEqual(calls, ["gold text"])
        self.assertEqual(report["oracle_query_count"], 2)
        self.assertEqual(report["oracle_self_retrieval_count"], 1)
        self.assertEqual(report["oracle_self_retrieval_cache_hits"], 1)
        self.assertEqual(report["oracle_self_retrieval_cache_size"], 1)

    def test_direct_index_mode_avoids_self_retrieval(self) -> None:
        original_rank = benchmark_support.rank_benchmark_sources

        def fake_rank_benchmark_sources(**kwargs):
            raise AssertionError("direct_index oracle mode should not call rank_benchmark_sources")

        benchmark_support.rank_benchmark_sources = fake_rank_benchmark_sources
        try:
            report = benchmark_support.build_oracle_retrieval_report(
                benchmark_name="knowme",
                oracle_items=[
                    {"query_id": "q1", "sample_id": "dataset001", "question_type": "x", "gold_segment_ids": {"gold-1"}},
                ],
                vector_store=object(),
                storage=object(),
                index_metadata={"source_records_by_id": {"gold-1": {"text": "gold text"}}},
                config=object(),
                pool_limit=10,
                retrieval_mode="direct_index",
            )
        finally:
            benchmark_support.rank_benchmark_sources = original_rank

        self.assertEqual(report["oracle_query_count"], 1)
        self.assertEqual(report["oracle_retrieval_mode"], "direct_index")
        self.assertEqual(report["oracle_direct_index_count"], 1)
        self.assertEqual(report["oracle_self_retrieval_count"], 0)
        self.assertEqual(report["oracle_recall@1"], 1.0)
        self.assertEqual(report["rows"][0]["self_rank"], 1)


if __name__ == "__main__":
    unittest.main()
