from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "base") not in sys.path:
    sys.path.insert(0, str(ROOT / "base"))

from export_oracle_baselines import build_oracle_baselines
from run_baselines import discover_baselines


def write_registry(root: Path, metrics: Path) -> None:
    registry = root / "artifacts" / "registry" / "benchmark_runs.jsonl"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "run_id": "clonemem-run",
                "project": "DysonSpherain",
                "dataset": "clonemem",
                "run_type": "full",
                "timestamp": "2026-04-28T00:00:00+00:00",
                "artifact_dir": str(metrics.parent),
                "metrics": {"candidate_recall@100": 0.5},
                "question_count": 2374,
                "embedding_provider": "sentence_transformer",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "fallback_in_use": False,
                "metadata": {"source_metrics_path": str(metrics)},
            }
        )
        + "\n",
        encoding="utf-8",
    )


class OracleBaselineExportTests(unittest.TestCase):
    def test_exports_only_artifact_backed_oracle_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunk = root / "run" / "chunk_00"
            diagnostics = chunk / "reports" / "diagnostics"
            diagnostics.mkdir(parents=True)
            chunk_metrics = chunk / "metrics.json"
            chunk_metrics.write_text(json.dumps({"benchmark": "clonemem"}), encoding="utf-8")
            merged = root / "run" / "merged_metrics.json"
            merged.write_text(json.dumps({"benchmark": "clonemem", "source_files": [str(chunk_metrics)]}), encoding="utf-8")
            write_registry(root, merged)
            (diagnostics / "clonemem_candidate_recall.json").write_text(
                json.dumps(
                    {
                        "queries": [
                            {"candidate_recall@100": 1.0, "gold_parent_hit": True},
                            {"candidate_recall@100": 0.0, "gold_parent_hit": False},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (diagnostics / "clonemem_oracle_retrieval.json").write_text(
                json.dumps({"rows": [{"top1_hit": True, "top5_hit": True, "top10_hit": True}]}),
                encoding="utf-8",
            )

            out = root / "oracle_out"
            summary = build_oracle_baselines(base_dir=root, out_root=out)

            self.assertEqual(len(summary["records"]), 3)
            baselines = {row["baseline"] for row in summary["records"]}
            self.assertEqual(baselines, {"oracle_candidate", "oracle_parent", "oracle_segment"})
            records = discover_baselines(out)
            oracle_candidate = next(row for row in records if row.benchmark == "clonemem" and row.baseline == "oracle_candidate")
            self.assertEqual(oracle_candidate.status, "available")
            self.assertEqual(oracle_candidate.metrics["oracle_candidate_recall@100"], 0.5)


if __name__ == "__main__":
    unittest.main()
