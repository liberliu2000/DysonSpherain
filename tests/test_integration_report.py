from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.adapters.integration_report import write_memory_agent_integration_report


class IntegrationReportTests(unittest.TestCase):
    def test_report_records_tests_and_smoke_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_memory_agent_integration_report(root, tests_run=["pytest tests/test_x.py -q"])
            text = path.read_text(encoding="utf-8")
        self.assertIn("pytest tests/test_x.py -q", text)
        self.assertIn("Smoke / Artifact Evidence", text)
        self.assertIn("Use paired token/quality rows", text)


if __name__ == "__main__":
    unittest.main()
