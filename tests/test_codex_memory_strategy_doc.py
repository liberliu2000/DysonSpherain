from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "base"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from dysonspherain.adapters.mcp_server import TOOL_SCHEMAS


class CodexMemoryStrategyDocTests(unittest.TestCase):
    def test_agents_memory_policy_contains_token_economy_order(self) -> None:
        text = (BASE / "dysonspherain" / "adapters" / "codex" / "AGENTS.memory.md").read_text(encoding="utf-8")
        for needle in (
            "dyson_memory_intent",
            "dyson_project_state",
            "dyson_recall",
            "dyson_context_pack",
            "dyson_token_economy_eval",
            "inject_summary_only",
            "return_file_refs_only",
            "dyson_write_memory",
        ):
            self.assertIn(needle, text)
        self.assertIn("Never copy full raw recall results", text)

    def test_plugin_and_config_expose_token_economy_tool(self) -> None:
        plugin = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
        self.assertIn("dyson_token_economy_eval", plugin["tools"])
        config = (ROOT / ".codex" / "config.toml").read_text(encoding="utf-8")
        self.assertIn("dyson_token_economy_eval", config)
        self.assertIn("existing_context", TOOL_SCHEMAS["dyson_token_economy_eval"]["properties"])
        self.assertIn("mode", TOOL_SCHEMAS["dyson_token_economy_eval"]["properties"])


if __name__ == "__main__":
    unittest.main()
