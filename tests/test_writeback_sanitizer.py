from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.writeback.sanitizer import sanitize_payload


class WritebackSanitizerTests(unittest.TestCase):
    def test_redacts_secret_values_and_keys(self) -> None:
        result = sanitize_payload({"summary": "api_key=sk-abcdef1234567890", "token": "secret-value"})
        self.assertTrue(result.has_redaction)
        self.assertGreaterEqual(result.redaction_count, 1)
        self.assertNotIn("sk-abcdef", str(result.payload))
        self.assertNotIn("secret-value", str(result.payload))


if __name__ == "__main__":
    unittest.main()
