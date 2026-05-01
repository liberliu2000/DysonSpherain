from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.memory_os.write_service import WriteMemoryRequest, write_memory


class WritebackDeduperTests(unittest.TestCase):
    def test_duplicate_agent_summary_is_not_written_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = WriteMemoryRequest(
                cwd=tmp,
                session_id="s1",
                task_goal="fix regression",
                summary="CloneMem candidate recall regression fixed by artifact-first diagnosis.",
                files_changed=["base/sphere_cli/evidence_pipeline.py"],
                commands_run=["pytest tests/test_x.py"],
                tests_run=["pytest"],
                benchmark_results=[],
                failures=[],
                next_actions=[],
                source="codex",
            )
            first = write_memory(request)
            second = write_memory(request)
            self.assertEqual(first.status, "ok")
            self.assertEqual(second.status, "duplicate")
            self.assertEqual(second.memory_id, first.memory_id)

    def test_benchmark_result_overlap_is_deduped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = WriteMemoryRequest(
                cwd=tmp,
                session_id="s1",
                task_goal="CloneMem smoke",
                summary="Ran CloneMem smoke and recorded metrics.",
                files_changed=["base/benchmarks/clonemem_benchmark.py"],
                commands_run=["python clonemem"],
                tests_run=[],
                benchmark_results=["CloneMem recall_frac@10=0.30 candidate_recall@100=0.40"],
                failures=[],
                next_actions=[],
                source="claude_code",
            )
            second = WriteMemoryRequest(
                cwd=tmp,
                session_id="s1",
                task_goal="CloneMem smoke",
                summary="PostCompact summary for the same CloneMem metric run.",
                files_changed=["base/benchmarks/clonemem_benchmark.py"],
                commands_run=[],
                tests_run=[],
                benchmark_results=["CloneMem recall_frac@10=0.30 candidate_recall@100=0.40"],
                failures=[],
                next_actions=[],
                source="claude_code_post_compact",
            )
            written = write_memory(first)
            duplicate = write_memory(second)
            self.assertEqual(written.status, "ok")
            self.assertEqual(duplicate.status, "duplicate")
            self.assertIn(duplicate.dedupe["dedupe_reason"], {"benchmark_result_dedupe", "same_task_benchmark_result", "same_file_change_summary"})

    def test_same_file_change_summary_is_deduped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = WriteMemoryRequest(
                cwd=tmp,
                session_id="s1",
                task_goal="Phase 5",
                summary="Updated session summarizer to parse commands and benchmark metrics.",
                files_changed=["base/dysonspherain/writeback/session_summarizer.py"],
                commands_run=[],
                tests_run=[],
                benchmark_results=[],
                failures=[],
                next_actions=[],
                source="claude_code",
            )
            second = WriteMemoryRequest(
                cwd=tmp,
                session_id="s2",
                task_goal="Phase 5",
                summary="Session summarizer now parses command lines and benchmark metric output.",
                files_changed=["base/dysonspherain/writeback/session_summarizer.py"],
                commands_run=[],
                tests_run=[],
                benchmark_results=[],
                failures=[],
                next_actions=[],
                source="claude_code_post_compact",
            )
            self.assertEqual(write_memory(first).status, "ok")
            duplicate = write_memory(second)
            self.assertEqual(duplicate.status, "duplicate")
            self.assertIn(duplicate.dedupe["dedupe_reason"], {"same_file_change_summary", "near_duplicate_lexical"})


if __name__ == "__main__":
    unittest.main()
