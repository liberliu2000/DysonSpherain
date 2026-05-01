from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.memory_os.observation_store import (
    apply_retention,
    delete_observation,
    export_observations,
    get_observations,
    resume_context,
    search_observations,
    timeline,
    token_economy_summary,
    write_observation,
    write_token_economy_event,
)
from sphere_cli.project_state import write_memory


class ObservationStoreTests(unittest.TestCase):
    def test_search_timeline_get_and_citation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_observation(
                root,
                project="DysonSpherain",
                kind="decision",
                title="CloneMem regression",
                content="candidate_recall dropped after rerank",
                source="unit",
                session_id="s1",
            )
            second = write_observation(
                root,
                project="DysonSpherain",
                kind="tool_event",
                title="pytest run",
                content="pytest passed",
                source="unit",
                session_id="s1",
            )
            search = search_observations(root, project="DysonSpherain", query="candidate_recall", limit=5)
            self.assertEqual(search["count"], 1)
            obs_id = first["observation_id"]
            self.assertTrue(search["observations"][0]["citation"].startswith("dyson://observation/"))
            details = get_observations(root, project="DysonSpherain", observation_ids=[obs_id])
            self.assertIn("candidate_recall", details["observations"][0]["content"])
            events = timeline(root, project="DysonSpherain", session_id="s1")
            self.assertEqual({item["observation_id"] for item in events["events"]}, {first["observation_id"], second["observation_id"]})

    def test_project_memories_are_projected_into_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = write_memory(root, memory_type="decision", project="DysonSpherain", content="Keep token quality separate.", source="unit")
            result = search_observations(root, project="DysonSpherain", query="token quality")
            self.assertEqual(result["observations"][0]["observation_id"], f"obs_mem_{memory['memory_id']}")

    def test_private_and_dysonignore_are_not_stored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".dysonignore").write_text("secret-file.py\n", encoding="utf-8")
            private = write_observation(
                root,
                project="DysonSpherain",
                kind="note",
                title="private",
                content="<private>api_key=sk-secret</private> public",
                source="unit",
            )
            ignored = write_observation(
                root,
                project="DysonSpherain",
                kind="note",
                title="ignored",
                content="touch secret-file.py",
                source="unit",
            )
            self.assertEqual(private["status"], "ok")
            self.assertIn("[redacted-private]", private["content"])
            self.assertEqual(ignored["status"], "skipped")

    def test_export_delete_and_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = write_observation(root, project="DysonSpherain", kind="note", title="old", content="old note", source="unit")
            new = write_observation(root, project="DysonSpherain", kind="note", title="new", content="new note", source="unit")
            retained = apply_retention(root, "DysonSpherain", keep_last=1)
            self.assertEqual(retained["archived_count"], 1)
            export_path = export_observations(root, "DysonSpherain", root / "export.json")
            self.assertTrue(export_path.exists())
            deleted = delete_observation(root, "DysonSpherain", new["observation_id"])
            self.assertEqual(deleted["status"], "deleted")
            result = get_observations(root, project="DysonSpherain", observation_ids=[old["observation_id"], new["observation_id"]])
            self.assertEqual(len(result["observations"]), 2)

    def test_token_economy_summary_reports_window_totals_and_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_token_economy_event(
                root,
                project="DysonSpherain",
                session_id="s1",
                prompt="debug benchmark",
                decision="inject",
                injected_tokens=100,
                baseline_context_tokens=1000,
                estimated_saved_tokens=900,
                budget_usage_ratio=0.0625,
            )
            write_token_economy_event(
                root,
                project="DysonSpherain",
                session_id="s-old",
                prompt="old debug benchmark",
                decision="inject",
                injected_tokens=100,
                baseline_context_tokens=1000,
                estimated_saved_tokens=500,
                budget_usage_ratio=0.0625,
                created_at="2020-01-01T00:00:00+00:00",
                updated_at="2020-01-01T00:00:00+00:00",
            )
            summary = token_economy_summary(root, project="DysonSpherain")
            self.assertEqual(summary["windows"]["24h"]["estimated_saved_tokens"], 900)
            self.assertAlmostEqual(summary["windows"]["24h"]["saving_ratio"], 0.9)
            self.assertEqual(summary["windows"]["30d"]["estimated_saved_tokens"], 900)
            self.assertEqual(len(summary["events"]), 2)

    def test_resume_context_builds_continuation_packet_for_latest_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_observation(
                root,
                project="DysonSpherain",
                kind="agent_run_summary",
                title="Session summary",
                content="Implemented resume context and verified tests.",
                source="unit",
                session_id="s-resume",
                metadata={
                    "task_goal": "Continue MCP memory integration",
                    "files_changed": ["base/dysonspherain/memory_os/observation_store.py"],
                    "tests_run": ["pytest tests/test_observation_store.py -q"],
                    "next_actions": ["Run full pytest"],
                },
            )
            write_token_economy_event(
                root,
                project="DysonSpherain",
                session_id="s-resume",
                prompt="continue work",
                decision="inject",
                injected_tokens=120,
                baseline_context_tokens=1000,
                estimated_saved_tokens=880,
                budget_usage_ratio=0.075,
            )
            packet = resume_context(root, project="DysonSpherain")
            self.assertEqual(packet["status"], "ok")
            self.assertEqual(packet["session_id"], "s-resume")
            self.assertIn("Continue MCP memory integration", packet["rendered_context"])
            self.assertIn("estimated_saved_tokens", packet["rendered_context"])
            self.assertFalse(packet["token_estimate"]["over_budget"])


if __name__ == "__main__":
    unittest.main()
