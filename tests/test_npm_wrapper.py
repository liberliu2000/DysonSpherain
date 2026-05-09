from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "bin" / "dysonspherain-memory.js"


@unittest.skipIf(shutil.which("node") is None, "node is not installed")
class NpmWrapperTests(unittest.TestCase):
    def test_package_json_declares_quickstart_bin(self) -> None:
        payload = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["bin"]["dysonspherain-memory"], "bin/dysonspherain-memory.js")
        self.assertEqual(payload["bin"]["dyson-memory"], "bin/dysonspherain-memory.js")

    def test_wrapper_help_and_mcp_smoke(self) -> None:
        help_proc = subprocess.run(["node", str(WRAPPER), "--help"], text=True, capture_output=True, check=True)
        self.assertIn("npx @liberliu/dysonspherain-memory install", help_proc.stdout)
        self.assertIn("npm install -g @liberliu/dysonspherain-memory", help_proc.stdout)
        self.assertIn("npx @liberliu/dysonspherain-memory plugin install", help_proc.stdout)
        smoke = subprocess.run(["node", str(WRAPPER), "mcp-smoke", "--python", sys.executable], text=True, capture_output=True, check=True)
        self.assertIn("dyson_search_memory", smoke.stdout)

    def test_wrapper_plugin_install_and_print(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                ["node", str(WRAPPER), "plugin", "install", "--python", sys.executable, "--project", tmp],
                text=True,
                capture_output=True,
                check=True,
            )
            root = Path(tmp)
            self.assertTrue((root / ".codex-plugin" / "plugin.json").exists(), proc.stdout + proc.stderr)
            self.assertTrue((root / ".claude-plugin" / "plugin.json").exists(), proc.stdout + proc.stderr)
            printed = subprocess.run(
                ["node", str(WRAPPER), "plugin", "print", "--python", sys.executable],
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("dyson_resume_context", printed.stdout)

    def test_wrapper_install_writes_agent_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                ["node", str(WRAPPER), "install", "--python", sys.executable, "--project", tmp],
                text=True,
                capture_output=True,
                check=True,
            )
            root = Path(tmp)
            self.assertTrue((root / ".codex" / "config.toml").exists(), proc.stdout + proc.stderr)
            self.assertTrue((root / ".claude" / "settings.json").exists(), proc.stdout + proc.stderr)
            self.assertTrue((root / ".codex-plugin" / "plugin.json").exists(), proc.stdout + proc.stderr)
            self.assertIn("dyson_get_observations", (root / ".codex" / "config.toml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
