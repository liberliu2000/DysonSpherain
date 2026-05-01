from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from export_compact_metrics import merge_payloads


class CompactMetricsExportTests(unittest.TestCase):
    def test_merge_payloads_combines_rows_and_fallback(self) -> None:
        payload = merge_payloads(
            [
                {
                    "source_metrics_path": "a/metrics.json",
                    "benchmark": "clonemem",
                    "question_count": 2,
                    "elapsed_seconds": 1.5,
                    "fallback_in_use": False,
                    "embedding_provider": "sentence_transformer",
                    "embedding_model": "m",
                    "results": [{"question_id": "q1"}, {"question_id": "q2"}],
                },
                {
                    "source_metrics_path": "b/metrics.json",
                    "benchmark": "clonemem",
                    "question_count": 1,
                    "elapsed_seconds": 2.0,
                    "fallback_in_use": True,
                    "results": [{"question_id": "q3"}],
                },
            ],
            source_label="demo",
        )

        self.assertEqual(payload["question_count"], 3)
        self.assertEqual(payload["compact_result_count"], 3)
        self.assertEqual(payload["elapsed_seconds"], 3.5)
        self.assertTrue(payload["fallback_in_use"])


if __name__ == "__main__":
    unittest.main()
