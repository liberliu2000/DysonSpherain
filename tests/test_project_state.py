from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.experiment_registry import ingest_artifacts
from sphere_cli.project_state import (
    archive_memory,
    get_memory,
    list_memories,
    load_project_state,
    render_project_state_markdown,
    search_memories,
    state_path,
    update_project_state_from_registry,
    update_memory,
    write_constraint,
    write_memory,
)
from dysonspherain.memory_os.project_state import ProjectStateRequest, get_project_state


def write_metrics(path: Path) -> None:
    payload = {
        "benchmark": "clonemem",
        "total_question_count": 2374,
        "wall_clock_elapsed_seconds": 100.0,
        "embedding_info": {
            "embedding_provider": "sentence_transformer",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "fallback_in_use": False,
        },
        "metrics": {"recall_frac@10": 0.2, "candidate_recall@100": 0.5},
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ProjectStateTests(unittest.TestCase):
    def test_default_state_has_pinned_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = load_project_state(Path(tmp), "DysonSpherain")

            self.assertEqual(state.project, "DysonSpherain")
            self.assertTrue(state.constraints)
            self.assertTrue(state.do_not_do)

    def test_update_from_registry_records_latest_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "runs" / "clonemem" / "metrics.json")
            ingest_artifacts(root / "runs", base_dir=root, project="DysonSpherain")

            state = update_project_state_from_registry(root, "DysonSpherain")

            self.assertTrue(state_path(root, "DysonSpherain").exists())
            self.assertEqual(state.latest_benchmark_status["clonemem"]["question_count"], 2374)
            self.assertFalse(state.latest_benchmark_status["clonemem"]["fallback_in_use"])
            self.assertIn("reports/phase4_diagnostic_consolidation.md", state.relevant_artifacts)

    def test_write_memory_updates_state_source_ids_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            payload = write_constraint(root, "DysonSpherain", "Never use local_hash for official validation.", "test")
            write_memory(root, memory_type="decision", project="DysonSpherain", content="Keep parent-anchor change tentative.", source="test")
            state = load_project_state(root, "DysonSpherain")

            self.assertIn(payload["memory_id"], state.source_memory_ids)
            self.assertIn("Never use local_hash for official validation.", state.constraints)
            self.assertIn("Keep parent-anchor change tentative.", state.recent_decisions)

    def test_project_memory_crud_search_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            payload = write_memory(
                root,
                memory_type="decision",
                project="DysonSpherain",
                content="Use artifact-backed pending rows only.",
                source="test",
                metadata={"title": "formal evidence rule"},
            )
            memory_id = payload["memory_id"]

            self.assertEqual(get_memory(root, "DysonSpherain", memory_id)["title"], "formal evidence rule")
            self.assertEqual(search_memories(root, "DysonSpherain", "artifact pending")[0]["memory_id"], memory_id)

            updated = update_memory(root, "DysonSpherain", memory_id, {"status": "superseded", "title": "updated rule"})
            self.assertEqual(updated["status"], "superseded")
            self.assertEqual(updated["title"], "updated rule")

            archived = archive_memory(root, "DysonSpherain", memory_id)
            self.assertEqual(archived["status"], "archived")
            self.assertEqual(list_memories(root, "DysonSpherain"), [])
            self.assertEqual(list_memories(root, "DysonSpherain", include_archived=True)[0]["memory_id"], memory_id)

    def test_markdown_renderer_includes_core_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = load_project_state(Path(tmp), "DysonSpherain")

            markdown = render_project_state_markdown(state)

            self.assertIn("Project State: DysonSpherain", markdown)
            self.assertIn("Latest Benchmark Status", markdown)
            self.assertIn("Constraints", markdown)

    def test_memory_os_project_state_applies_token_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for idx in range(20):
                write_memory(root, memory_type="decision", project="DysonSpherain", content=f"Decision {idx}: long project decision with benchmark details.", source="test")
            result = get_project_state(ProjectStateRequest(cwd=str(root), token_budget=1))

            self.assertFalse(result["token_estimate"]["over_budget"])
            self.assertLessEqual(result["token_estimate"]["estimated_tokens"], 1)
            self.assertTrue(result["token_estimate"]["dropped_items"])


if __name__ == "__main__":
    unittest.main()
