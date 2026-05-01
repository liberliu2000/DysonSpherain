from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.context_pack.builder import build_pack, build_pack_from_candidates
from dysonspherain.adapters.mcp_server import call_tool, temporary_allowed_roots
from dysonspherain.context_pack.renderers import render_context_pack
from sphere_cli.project_state import write_memory


class ContextPackTests(unittest.TestCase):
    def test_context_pack_contains_stable_sections_and_file_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_memory(
                root,
                memory_type="decision",
                project="DysonSpherain",
                content="Use artifact-backed benchmark reports for regression triage.",
                source="test",
                metadata={"files_changed": ["base/sphere_cli/evidence_pipeline.py"], "benchmark_results": ["recall@10=1.0"]},
            )
            pack, budget = build_pack(base_dir=root, project="DysonSpherain", query="benchmark regression", token_budget=800)
            rendered = render_context_pack(pack, "markdown")
            self.assertIn("Prior Decisions", rendered)
            self.assertIn("base/sphere_cli/evidence_pipeline.py", rendered)
            self.assertFalse(budget["over_budget"])

    def test_context_pack_can_pack_explicit_memory_ids_and_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = write_memory(
                root,
                memory_type="decision",
                project="DysonSpherain",
                content="Keep token economy diagnostic-only.",
                source="test",
                metadata={"files_changed": ["base/dysonspherain/evaluation/token_economy.py"]},
            )
            with temporary_allowed_roots([root]):
                result = call_tool(
                    "dyson_context_pack",
                    {
                        "cwd": str(root),
                        "project": "DysonSpherain",
                        "memory_ids": [record["memory_id"]],
                        "sections": ["prior_decisions", "relevant_files"],
                        "format": "markdown",
                        "token_budget": 800,
                    },
                )
            self.assertEqual(result["status"], "ok")
            self.assertIn("Keep token economy diagnostic-only.", result["rendered_context"])
            self.assertIn("base/dysonspherain/evaluation/token_economy.py", result["rendered_context"])
            self.assertNotIn("Core Evidence\n-", result["rendered_context"])

    def test_context_pack_compresses_supplied_candidates(self) -> None:
        candidates = [
            {
                "id": "c1",
                "text": "Candidate evidence for token economy mode adapter.",
                "path": "base/dysonspherain/evaluation/token_economy.py",
                "score": 0.91,
                "recall_at_10": 0.8,
            },
            {
                "id": "p1",
                "text": "Old prompt: repeat previous request.",
                "source": "prior_prompt",
                "score": 0.5,
            },
        ]
        pack, budget = build_pack_from_candidates(
            project="DysonSpherain",
            candidates=candidates,
            token_budget=800,
            include_prior_prompts=False,
            sections=["core_evidence", "relevant_files", "benchmark_state"],
        )
        rendered = render_context_pack(pack, "markdown")
        self.assertIn("Candidate evidence for token economy mode adapter.", rendered)
        self.assertIn("base/dysonspherain/evaluation/token_economy.py", rendered)
        self.assertIn("recall_at_10", rendered)
        self.assertNotIn("Old prompt", rendered)
        self.assertFalse(budget["over_budget"])


if __name__ == "__main__":
    unittest.main()
