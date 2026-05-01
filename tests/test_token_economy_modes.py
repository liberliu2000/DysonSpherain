from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.evaluation.token_economy import _assemble_runtime_context
from dysonspherain.evaluation.token_economy_modes import TOKEN_ECONOMY_MODE_REGISTRY, resolve_token_economy_mode


class TokenEconomyModeTests(unittest.TestCase):
    def test_registered_modes_have_explicit_runtime_adapters(self) -> None:
        self.assertIn("conservative", TOKEN_ECONOMY_MODE_REGISTRY)
        mode = resolve_token_economy_mode("conservative")
        self.assertTrue(mode.available)
        self.assertEqual(dict(mode.runtime_overrides or {}).get("mode"), "fast")

    def test_off_mode_disables_runtime_retrieval(self) -> None:
        context, extra = _assemble_runtime_context(query="q", mode="off", budget=100, memory_base_dir=None)
        self.assertEqual(context, "")
        self.assertTrue(extra["retrieval_disabled"])
        self.assertEqual(extra["mode_status"], "available")

    def test_unknown_mode_is_explicitly_unavailable(self) -> None:
        mode = resolve_token_economy_mode("unverified_route")
        self.assertFalse(mode.available)
        self.assertEqual(mode.status, "unavailable")
        context, extra = _assemble_runtime_context(query="q", mode="unverified_route", budget=100, memory_base_dir=None)
        self.assertEqual(context, "")
        self.assertEqual(extra["mode_unavailable"], "unverified_route")
        self.assertEqual(extra["unavailable_reason"], "mode_not_registered_or_unverified")


if __name__ == "__main__":
    unittest.main()
