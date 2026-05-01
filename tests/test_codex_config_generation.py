from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.adapters.codex.generate_config import doctor, install_agents_policy, install_claude_hooks, install_codex_mcp, install_plugin_manifests


class CodexConfigGenerationTests(unittest.TestCase):
    def test_installers_merge_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".codex").mkdir()
            (root / ".codex" / "config.toml").write_text("[mcp_servers.other]\ncommand = \"x\"\n", encoding="utf-8")
            self.assertTrue(install_codex_mcp(root).changed)
            self.assertFalse(install_codex_mcp(root).changed)
            self.assertTrue(install_agents_policy(root).changed)
            self.assertFalse(install_agents_policy(root).changed)
            self.assertTrue(install_claude_hooks(root).changed)
            settings = json.loads((root / ".claude" / "settings.json").read_text(encoding="utf-8"))
            self.assertIn("UserPromptSubmit", settings["hooks"])
            self.assertIn("SessionStart", settings["hooks"])
            self.assertIn("PostToolUse", settings["hooks"])
            self.assertIn("Stop", settings["hooks"])
            self.assertTrue(install_plugin_manifests(root).changed)
            self.assertFalse(install_plugin_manifests(root).changed)
            plugin = json.loads((root / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
            self.assertIn("dyson_memory_intent", plugin["tools"])
            self.assertIn("dyson_search_memory", plugin["tools"])
            self.assertIn("dyson_resume_context", plugin["tools"])
            report = doctor(root)
            self.assertTrue(report["checks"]["codex_mcp"]["ok"])
            self.assertTrue(report["checks"]["agents_policy"]["ok"])

    def test_codex_installer_updates_existing_dyson_block_with_new_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".codex").mkdir()
            (root / ".codex" / "config.toml").write_text(
                '[mcp_servers.dyson-memory]\ncommand = "python"\nenabled_tools = [\n  "dyson_recall"\n]\n',
                encoding="utf-8",
            )
            result = install_codex_mcp(root)
            self.assertTrue(result.changed)
            text = (root / ".codex" / "config.toml").read_text(encoding="utf-8")
            self.assertIn("dyson_memory_intent", text)
            self.assertIn("dyson_search_memory", text)
            self.assertIn("dyson_get_observations", text)
            self.assertIn("dyson_resume_context", text)


if __name__ == "__main__":
    unittest.main()
