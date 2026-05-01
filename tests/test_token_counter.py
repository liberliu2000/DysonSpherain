from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.utils.token_counter import TokenCounter


class TokenCounterTests(unittest.TestCase):
    def test_empty_and_none_are_zero(self) -> None:
        counter = TokenCounter()
        self.assertEqual(counter.count(None).tokens, 0)
        self.assertEqual(counter.count("").tokens, 0)

    def test_count_many_reports_fallback_state(self) -> None:
        result = TokenCounter("cl100k_base").count_many(["hello", " world"])
        self.assertGreater(result.tokens, 0)
        self.assertGreater(result.chars, 0)
        self.assertTrue(result.tokenizer_name)


if __name__ == "__main__":
    unittest.main()
