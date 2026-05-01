from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from export_rrf_baselines import export_benchmark, rrf_fuse
from run_baselines import _classify_baseline, discover_baselines


def write_payload(path: Path, *, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "benchmark": "knowme",
                "mode": "vector",
                "question_count": len(rows),
                "results": rows,
            }
        ),
        encoding="utf-8",
    )


class RrfBaselineExportTests(unittest.TestCase):
    def test_rrf_fuse_is_deterministic_with_tie_breaks(self) -> None:
        first = rrf_fuse([["a", "b"], ["c", "b"]], rrf_k=60, limit=3)
        second = rrf_fuse([["a", "b"], ["c", "b"]], rrf_k=60, limit=3)

        self.assertEqual(first, second)
        self.assertEqual(first[0], "b")
        self.assertEqual(first, ["b", "a", "c"])

    def test_export_recomputes_flat_metrics_from_artifact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dense = root / "dense" / "metrics.json"
            bm25 = root / "bm25" / "metrics.json"
            row = {
                "question_id": "q1",
                "question": "where is the receipt",
                "evidence_ids": ["gold"],
                "ranked_items": [{"corpus_id": "miss"}, {"corpus_id": "gold"}],
            }
            write_payload(dense, rows=[row])
            write_payload(
                bm25,
                rows=[
                    {
                        **row,
                        "ranked_items": [{"corpus_id": "gold"}, {"corpus_id": "other"}],
                    }
                ],
            )

            record = export_benchmark("knowme", dense_path=dense, bm25_path=bm25, out_root=root / "out", rrf_k=60)
            payload = json.loads(Path(record["metrics_path"]).read_text(encoding="utf-8"))

            self.assertEqual(payload["baseline"], "dense_bm25_rrf")
            self.assertEqual(payload["total_question_count"], 1)
            self.assertEqual(payload["metrics"]["recall_any@1"], 1.0)
            self.assertEqual(payload["metrics"]["recall_frac@10"], 1.0)

    def test_baseline_discovery_classifies_artifact_rrf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "run" / "knowme_dense_bm25_rrf" / "metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(
                json.dumps(
                    {
                        "benchmark": "knowme",
                        "baseline": "dense_bm25_rrf",
                        "method": "dense_bm25_rrf",
                        "mode": "artifact_rrf",
                        "total_question_count": 1010,
                        "fallback_in_use": False,
                        "metrics": {"recall_any@10": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(_classify_baseline(metrics, json.loads(metrics.read_text(encoding="utf-8"))), "dense_bm25_rrf")
            records = discover_baselines(root)
            rrf = next(record for record in records if record.benchmark == "knowme" and record.baseline == "dense_bm25_rrf")

            self.assertEqual(rrf.status, "available")
            self.assertFalse(rrf.fallback_in_use)


if __name__ == "__main__":
    unittest.main()
