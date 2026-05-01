from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from product_acceptance_smoke import run_smoke


class ProductAcceptanceSmokeTests(unittest.TestCase):
    def test_product_acceptance_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_smoke(Path(tmp))
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(all(payload["checks"].values()))


if __name__ == "__main__":
    unittest.main()
