from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.memory_os.observation_store import token_economy_summary, write_token_economy_event


class TokenEconomyLedgerTests(unittest.TestCase):
    def test_writes_standard_event_without_full_prompt_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = "debug regression " + ("secret detail " * 80)
            result = write_token_economy_event(
                root,
                project="DysonSpherain",
                session_id="s1",
                prompt=prompt,
                decision="inject",
                injected_tokens=120,
                baseline_context_tokens=1000,
                estimated_saved_tokens=880,
                budget_usage_ratio=0.1,
                adapter="claude_hook",
                task_type="debug",
                mode="conservative",
                risk="medium",
                reason="unit test",
                baseline_type="full_history",
                candidate_context_tokens=200,
                final_injected_tokens=120,
                duplicate_token_ratio=0.25,
                fallback_tokenizer_used=True,
                tokenizer_name="mixed_content_heuristic:prose",
            )
            self.assertEqual(result["kind"], "token_economy_event")
            metadata = result["metadata"]
            self.assertEqual(metadata["adapter"], "claude_hook")
            self.assertEqual(metadata["estimated_saved_tokens"], 880)
            self.assertEqual(metadata["final_injected_tokens"], 120)
            self.assertLess(len(metadata["query_preview"]), len(prompt))
            self.assertEqual(metadata["query_hash"], metadata["query_hash"][:16])
            self.assertTrue(metadata["fallback_tokenizer_used"])

    def test_summary_is_backward_compatible_and_non_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_token_economy_event(
                root,
                project="DysonSpherain",
                session_id="s1",
                prompt="q",
                decision="inject",
                injected_tokens=1200,
                baseline_context_tokens=100,
                estimated_saved_tokens=-100,
                budget_usage_ratio=2.0,
                adapter="codex_mcp",
            )
            write_token_economy_event(
                root,
                project="DysonSpherain",
                session_id="s2",
                prompt="q2",
                decision="skip",
                injected_tokens=0,
                baseline_context_tokens=200,
                estimated_saved_tokens=200,
                budget_usage_ratio=0.0,
                adapter="cli",
            )
            summary = token_economy_summary(root, project="DysonSpherain")
            self.assertEqual(summary["windows"]["24h"]["estimated_saved_tokens"], 200)
            self.assertEqual(summary["adapter_distribution"]["codex_mcp"], 1)
            self.assertEqual(summary["adapter_distribution"]["cli"], 1)
            self.assertIn("llm_prompt_token_economy", summary)
            self.assertIn("local_compute_economy", summary)


if __name__ == "__main__":
    unittest.main()
