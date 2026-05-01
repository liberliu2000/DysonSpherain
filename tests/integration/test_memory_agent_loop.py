from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.adapters.mcp_server import call_tool, temporary_allowed_roots
from sphere_cli.project_state import write_memory


class MemoryAgentLoopIntegrationTests(unittest.TestCase):
    def test_memory_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_memory(root, memory_type="failure", project="DysonSpherain", content="CloneMem regression came from candidate admission failure.", source="test")
            with temporary_allowed_roots([root]):
                state = call_tool("dyson_project_state", {"cwd": tmp, "token_budget": 1200})
                recall = call_tool("dyson_recall", {"query": "continue benchmark regression repair", "cwd": tmp, "token_budget": 1600})
            self.assertEqual(state["status"], "ok")
            self.assertIn(recall["status"], {"ok", "empty"})
            self.assertLessEqual(recall["token_estimate"]["estimated_tokens"], 1600)
            write_args = {
                "cwd": tmp,
                "session_id": "s1",
                "task_goal": "repair regression",
                "summary": "Fixed candidate admission regression.",
                "files_changed": ["base/sphere_cli/evidence_pipeline.py"],
                "commands_run": ["pytest"],
                "tests_run": ["pytest"],
                "benchmark_results": ["candidate_recall@100=1.0"],
                "failures": [],
                "next_actions": [],
                "source": "codex",
            }
            with temporary_allowed_roots([root]):
                first = call_tool("dyson_write_memory", write_args)
                second = call_tool("dyson_write_memory", write_args)
                decision = call_tool("dyson_token_economy_eval", {"query": "repair regression", "candidate_context": recall.get("rendered_context") or "", "token_budget": 1600})
            self.assertEqual(first["status"], "ok")
            self.assertEqual(second["status"], "duplicate")
            self.assertIn(decision["decision"], {"inject", "skip", "inject_summary_only", "return_file_refs_only"})


if __name__ == "__main__":
    unittest.main()
