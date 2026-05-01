from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "base" / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "base" / "benchmarks"))

from run_ablation_suite import discover_ablations
from run_baselines import _classify_baseline, discover_baselines
from longmemeval_benchmark import rank_bm25
from merge_benchmark_results import merge_payloads


def write_metrics(path: Path, *, fallback: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "clonemem",
        "total_question_count": 10,
        "elapsed_seconds": 2.0,
        "embedding_info": {
            "embedding_provider": "sentence_transformer",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "fallback_in_use": fallback,
        },
        "metrics": {"recall_frac@10": 0.2, "candidate_recall@100": 0.5},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class Phase10ArtifactSummaryTests(unittest.TestCase):
    def test_rank_bm25_uses_fts_hits_and_deterministic_tail(self) -> None:
        class FakeStorage:
            def search_chunks_fts(self, query: str, limit: int) -> list[dict]:
                self.query = query
                self.limit = limit
                return [
                    {"node_id": "n2", "bm25_score": -2.0},
                    {"node_id": "n1", "bm25_score": -1.0},
                ]

        storage = FakeStorage()
        ranked_ids, ranked_items, timings = rank_bm25(
            "favorite restaurant",
            storage=storage,  # type: ignore[arg-type]
            ordered_corpus_ids=["c1", "c2", "c3"],
            corpus_by_node_id={
                "n1": {"corpus_id": "c1", "text": "one", "timestamp": "t1"},
                "n2": {"corpus_id": "c2", "text": "two", "timestamp": "t2"},
            },
            top_k=2,
            chunk_pool=10,
        )

        self.assertEqual(ranked_ids, ["c2", "c1", "c3"])
        self.assertEqual([item["corpus_id"] for item in ranked_items], ["c2", "c1"])
        self.assertEqual(storage.limit, 10)
        self.assertIn("bm25_total_ms", timings)

    def test_baseline_discovery_marks_available_and_pending_without_fabricating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "run_001" / "clonemem_dense_only" / "metrics.json")

            records = discover_baselines(root)
            dense = next(item for item in records if item.benchmark == "clonemem" and item.baseline == "dense_only_minilm")
            pending = next(item for item in records if item.benchmark == "clonemem" and item.baseline == "bm25")

            self.assertEqual(dense.status, "available")
            self.assertEqual(dense.metrics["recall_frac@10"], 0.2)
            self.assertEqual(pending.status, "pending")
            self.assertIsNone(pending.metrics)

    def test_ablation_discovery_marks_fallback_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "run_001" / "clonemem_no_safe_fusion" / "metrics.json", fallback=True)

            records = discover_ablations(root)
            no_safe = next(item for item in records if item.benchmark == "clonemem" and item.ablation == "no_safe_fusion")

            self.assertEqual(no_safe.status, "available")
            self.assertTrue(no_safe.fallback_in_use)
            self.assertIn("fallback_in_use=true", "; ".join(no_safe.warnings or []))

    def test_ablation_discovery_does_not_treat_bm25_baseline_as_full_admission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bm25_path = root / "run_bm25" / "clonemem" / "merged_metrics.json"
            write_metrics(bm25_path)
            payload = json.loads(bm25_path.read_text(encoding="utf-8"))
            payload["mode"] = "bm25"
            bm25_path.write_text(json.dumps(payload), encoding="utf-8")

            records = discover_ablations(root)
            full = next(item for item in records if item.benchmark == "clonemem" and item.ablation == "full_admission")

            self.assertEqual(full.status, "pending")
            self.assertIsNone(full.metrics_path)

    def test_ablation_discovery_does_not_treat_artifact_rrf_baseline_as_full_admission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rrf_path = root / "run_rrf" / "clonemem_dense_bm25_rrf" / "compact_metrics.json"
            write_metrics(rrf_path)
            payload = json.loads(rrf_path.read_text(encoding="utf-8"))
            payload["baseline"] = "dense_bm25_rrf"
            payload["method"] = "dense_bm25_rrf"
            payload["mode"] = "artifact_rrf"
            rrf_path.write_text(json.dumps(payload), encoding="utf-8")

            records = discover_ablations(root)
            full = next(item for item in records if item.benchmark == "clonemem" and item.ablation == "full_admission")

            self.assertEqual(full.status, "pending")
            self.assertIsNone(full.metrics_path)

    def test_baseline_discovery_prefers_full_over_newer_experimental_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_path = root / "20260428_full" / "clonemem_evidence" / "clonemem" / "merged_metrics.json"
            sample_path = root / "20260428_clonemem_evidence_blend_alpha025_sample_v1" / "clonemem" / "merged_metrics.json"
            write_metrics(full_path)
            write_metrics(sample_path)
            sample_payload = json.loads(sample_path.read_text(encoding="utf-8"))
            sample_payload["total_question_count"] = 20
            sample_path.write_text(json.dumps(sample_payload), encoding="utf-8")

            records = discover_baselines(root)
            full = next(item for item in records if item.benchmark == "clonemem" and item.baseline == "dysonspherain_full")

            self.assertEqual(Path(full.metrics_path or "").resolve(), full_path.resolve())

    def test_baseline_discovery_uses_compact_metrics_next_to_oversized_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_path = root / "run_bm25" / "clonemem" / "metrics.json"
            compact_path = root / "run_bm25" / "clonemem" / "compact_metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text("x" * (26 * 1024 * 1024), encoding="utf-8")
            compact_path.write_text(
                json.dumps(
                    {
                        "schema": "dysonspherain.compact_metrics.v1",
                        "benchmark": "clonemem",
                        "question_count": 200,
                        "mode": "bm25",
                        "metrics": {"segment": {"recall_frac@10": 0.12}},
                        "fallback_in_use": False,
                    }
                ),
                encoding="utf-8",
            )

            records = discover_baselines(root)
            bm25 = next(item for item in records if item.benchmark == "clonemem" and item.baseline == "bm25")

            self.assertEqual(bm25.status, "available")
            self.assertEqual(Path(bm25.metrics_path or "").resolve(), compact_path.resolve())
            self.assertEqual(bm25.metrics["recall_frac@10"], 0.12)
            self.assertIn("sample_or_partial_run=true", "; ".join(bm25.warnings or []))

    def test_baseline_classifier_does_not_treat_path_only_bm25_as_bm25(self) -> None:
        path = Path("/tmp/20260427_bm25_nocopy_sampling_v1/knowme/compact_metrics.json")

        baseline = _classify_baseline(path, {"benchmark": "knowme", "mode": "evidence"})

        self.assertEqual(baseline, "dysonspherain_full")

    def test_merge_preserves_shard_mode_for_baseline_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "chunk_00" / "metrics.json"
            shard.parent.mkdir(parents=True)
            shard.write_text(
                json.dumps(
                    {
                        "zone": "locomo",
                        "mode": "bm25",
                        "question_count": 1,
                        "metrics": {"session": {"recall_any@10": 1.0}},
                        "results": [{"metrics": {"session": {"recall_any@10": 1.0}}}],
                    }
                ),
                encoding="utf-8",
            )

            merged = merge_payloads([shard])

            self.assertEqual(merged["mode"], "bm25")
            self.assertEqual(_classify_baseline(Path("locomo/merged_metrics.json"), merged), "bm25")

    def test_baseline_discovery_prefers_available_compact_over_oversized_merged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged_path = root / "run_bm25" / "longmemeval" / "merged_metrics.json"
            compact_path = root / "run_bm25" / "longmemeval" / "compact_metrics.json"
            merged_path.parent.mkdir(parents=True)
            merged_path.write_text("x" * (26 * 1024 * 1024), encoding="utf-8")
            compact_path.write_text(
                json.dumps(
                    {
                        "schema": "dysonspherain.compact_metrics.v1",
                        "benchmark": "longmemeval",
                        "question_count": 500,
                        "mode": "bm25",
                        "metrics": {"session": {"recall_any@10": 0.96}},
                        "fallback_in_use": False,
                    }
                ),
                encoding="utf-8",
            )

            records = discover_baselines(root)
            bm25 = next(item for item in records if item.benchmark == "longmemeval" and item.baseline == "bm25")

            self.assertEqual(bm25.status, "available")
            self.assertEqual(Path(bm25.metrics_path or "").resolve(), compact_path.resolve())
            self.assertEqual(bm25.metrics["recall_any@10"], 0.96)

    def test_ablation_discovery_extracts_nested_metrics_and_prefers_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_path = root / "20260428_full" / "clonemem_full_admission" / "metrics.json"
            sample_path = root / "20260428_sample" / "clonemem_full_admission_sample" / "metrics.json"
            write_metrics(full_path)
            full_payload = json.loads(full_path.read_text(encoding="utf-8"))
            full_payload["metrics"] = {"segment": {"recall_frac@10": 0.33, "ndcg_any@10": 0.44}}
            full_payload["total_question_count"] = 100
            full_path.write_text(json.dumps(full_payload), encoding="utf-8")
            write_metrics(sample_path)
            sample_payload = json.loads(sample_path.read_text(encoding="utf-8"))
            sample_payload["metrics"] = {"segment": {"recall_frac@10": 0.99, "ndcg_any@10": 0.99}}
            sample_payload["total_question_count"] = 10
            sample_path.write_text(json.dumps(sample_payload), encoding="utf-8")

            records = discover_ablations(root)
            full = next(item for item in records if item.benchmark == "clonemem" and item.ablation == "full_admission")

            self.assertEqual(Path(full.metrics_path or "").resolve(), full_path.resolve())
            self.assertEqual(full.metrics["recall_frac@10"], 0.33)
            self.assertEqual(full.metrics["ndcg_any@10"], 0.44)


if __name__ == "__main__":
    unittest.main()
