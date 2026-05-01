from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.writeback.session_summarizer import summarize_transcript


class SessionSummarizerTests(unittest.TestCase):
    def test_summarizer_extracts_commands_files_metrics_failures_and_next_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            transcript.write_text(
                "\n".join(
                    [
                        "User: fix CloneMem benchmark regression",
                        json.dumps({"tool": "exec_command", "cmd": "pytest tests/test_token_budgeter.py -q", "output": "1 passed"}),
                        json.dumps({"tool": "apply_patch", "path": "base/dysonspherain/writeback/session_summarizer.py"}),
                        "CloneMem recall_frac@10=0.30 candidate_recall@100=0.40 ndcg_any@10=0.20",
                        "ERROR: previous command failed with timeout",
                        "Next: rerun benchmark smoke-all --record-token-economy",
                    ]
                ),
                encoding="utf-8",
            )
            summary = summarize_transcript(transcript)
        self.assertTrue(summary["should_write"])
        self.assertIn("pytest tests/test_token_budgeter.py -q", summary["commands_run"])
        self.assertIn("base/dysonspherain/writeback/session_summarizer.py", summary["files_changed"])
        self.assertTrue(any("candidate_recall@100" in item for item in summary["benchmark_results"]))
        self.assertTrue(summary["failures"])
        self.assertTrue(summary["next_actions"])


if __name__ == "__main__":
    unittest.main()
