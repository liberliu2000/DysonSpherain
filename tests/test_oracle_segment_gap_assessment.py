from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "base") not in sys.path:
    sys.path.insert(0, str(ROOT / "base"))

from assess_oracle_segment_gaps import build_report


class OracleSegmentGapAssessmentTests(unittest.TestCase):
    def test_existing_available_segment_baseline_is_not_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "formal" / "clonemem" / "merged_metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(json.dumps({"benchmark": "clonemem"}), encoding="utf-8")
            registry = root / "artifacts" / "registry" / "benchmark_runs.jsonl"
            registry.parent.mkdir(parents=True)
            registry.write_text(
                json.dumps(
                    {
                        "run_id": "clonemem-full",
                        "project": "DysonSpherain",
                        "dataset": "clonemem",
                        "run_type": "full",
                        "timestamp": "2026-04-29T00:00:00+00:00",
                        "artifact_dir": str(metrics.parent),
                        "metrics": {},
                        "question_count": 2374,
                        "fallback_in_use": False,
                        "metadata": {"source_metrics_path": str(metrics)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            oracle_metrics = root / "oracle" / "clonemem_oracle_segment" / "metrics.json"
            oracle_metrics.parent.mkdir(parents=True)
            oracle_metrics.write_text(json.dumps({"baseline": "oracle_segment"}), encoding="utf-8")
            baselines = root / "artifacts" / "baselines" / "baseline_runs.json"
            baselines.parent.mkdir(parents=True, exist_ok=True)
            baselines.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "benchmark": "clonemem",
                                "baseline": "oracle_segment",
                                "status": "available",
                                "metrics_path": str(oracle_metrics),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(root, baselines)
            clonemem = next(row for row in report["rows"] if row["benchmark"] == "clonemem")

            self.assertEqual(clonemem["status"], "already_available")
            self.assertEqual(report["summary"]["already_available"], 1)
            self.assertEqual(report["summary"]["blocked_missing_source_diagnostics"], 0)


if __name__ == "__main__":
    unittest.main()
