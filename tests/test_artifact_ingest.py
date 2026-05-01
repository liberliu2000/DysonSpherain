from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.experiment_registry import discover_metric_files, ingest_artifacts, load_registry


class ArtifactIngestTests(unittest.TestCase):
    def test_directory_discovery_skips_large_legacy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small = root / "new" / "merged_metrics.json"
            large = root / "old" / "metrics.json"
            small.parent.mkdir(parents=True)
            large.parent.mkdir(parents=True)
            small.write_text(json.dumps({"benchmark": "locomo", "total_question_count": 1986}), encoding="utf-8")
            large.write_text("x" * 128, encoding="utf-8")

            files = discover_metric_files(root, max_discovered_metrics_bytes=16)

            self.assertIn(small, files)
            self.assertNotIn(large, files)

    def test_direct_large_metrics_file_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics.json"
            metrics.write_text(
                json.dumps(
                    {
                        "benchmark": "knowme",
                        "question_count": 1010,
                        "embedding_provider": "sentence_transformer",
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "metrics": {"recall_frac@10": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            files = discover_metric_files(metrics, max_discovered_metrics_bytes=1)

            self.assertEqual(files, [metrics])

    def test_ingest_records_sidecar_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            metrics = run / "metrics.json"
            diagnostics = run / "reports" / "diagnostics"
            diagnostics.mkdir(parents=True)
            metrics.write_text(
                json.dumps(
                    {
                        "benchmark": "clonemem",
                        "question_count": 2374,
                        "embedding_provider": "sentence_transformer",
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "metrics": {"candidate_recall@100": 0.3},
                    }
                ),
                encoding="utf-8",
            )
            (diagnostics / "clonemem_oracle_retrieval.json").write_text(
                json.dumps({"oracle_recall@10": 1.0, "rows": [{"large": "omitted"}]}),
                encoding="utf-8",
            )
            (diagnostics / "failure_taxonomy.json").write_text(
                json.dumps({"failure_type_distribution": {"parent_hit_segment_miss": 2}}),
                encoding="utf-8",
            )

            ingest_artifacts(run, base_dir=root, project="DysonSpherain")
            stored = load_registry(root)[0]

            self.assertEqual(stored.metadata["oracle_summary"]["oracle_recall@10"], 1.0)
            self.assertNotIn("rows", stored.metadata["oracle_summary"])
            self.assertEqual(stored.metadata["failure_taxonomy"]["parent_hit_segment_miss"], 2)
            self.assertGreaterEqual(stored.metadata["sidecar_artifact_count"], 2)

    def test_ingest_uses_run_manifest_for_sample_count_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "chunked" / "clonemem"
            merged = run / "clonemem" / "merged_metrics.json"
            manifest = run / "run_manifest.json"
            merged.parent.mkdir(parents=True)
            merged.write_text(
                json.dumps(
                    {
                        "benchmark": "clonemem",
                        "total_question_count": 2374,
                        "embedding_info": {
                            "embedding_provider": "sentence_transformer",
                            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                            "fallback_in_use": False,
                        },
                        "metrics": {"candidate_recall@100": 0.3},
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "workers": 4,
                        "shard_strategy": "sample",
                        "chunks": [
                            {"sample_ids": ["a", "b"], "command": ["python", "clonemem_benchmark.py"]},
                            {"sample_ids": ["b", "c"], "command": ["python", "clonemem_benchmark.py"]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            ingest_artifacts(merged, base_dir=root, project="DysonSpherain")
            stored = load_registry(root)[0]

            self.assertEqual(stored.sample_count, 3)
            self.assertIn("chunked_subprocess", stored.command or "")
            self.assertEqual(stored.metadata["workers"], 4)
            self.assertEqual(stored.metadata["shard_strategy"], "sample")

    def test_ingest_derives_formal_protocol_metadata_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "chunked" / "locomo_vector"
            merged = run / "locomo" / "merged_metrics.json"
            manifest = run / "run_manifest.json"
            merged.parent.mkdir(parents=True)
            merged.write_text(
                json.dumps(
                    {
                        "benchmark": "locomo",
                        "total_question_count": 1986,
                        "source_files": ["chunk_00/metrics.json", "chunk_01/metrics.json"],
                        "embedding_info": {
                            "provider": "sentence_transformer",
                            "model": "sentence-transformers/all-MiniLM-L6-v2",
                            "fallback_in_use": False,
                        },
                        "vector_info": {"vector_backend": "chroma"},
                        "metrics": {"recall_frac@10": 0.9},
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "benchmark": "locomo",
                        "data_root": "/data/locomo/locomo10.json",
                        "workers": 4,
                        "shard_strategy": "question",
                        "chunks": [
                            {"command": ["python", "locomo_benchmark.py", "--shard-index", "0"]},
                            {"command": ["python", "locomo_benchmark.py", "--shard-index", "1"]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            ingest_artifacts(merged, base_dir=root, project="DysonSpherain")
            stored = load_registry(root)[0]

            self.assertIsNotNone(stored.command)
            self.assertIsNotNone(stored.config_hash)
            self.assertIsNotNone(stored.dataset_version)
            self.assertNotEqual(stored.config_hash, stored.dataset_version)


if __name__ == "__main__":
    unittest.main()
