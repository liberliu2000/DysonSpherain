from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.memory_os.recall_service import RecallRequest, recall


class RecallServiceParamTests(unittest.TestCase):
    def test_runtime_recall_honors_include_flags_and_traces_channels(self) -> None:
        runtime = SimpleNamespace(
            run_query=lambda *args, **kwargs: {
                "bundle": SimpleNamespace(
                    core_evidence=[
                        {"id": "recent", "text": "Recent stable evidence", "timestamp": "2026-04-30", "score": 0.9},
                        {"id": "prompt", "text": "Old prompt: repeat task", "timestamp": "2026-04-29", "score": 0.8},
                    ],
                    supporting_context=[],
                    evidence_objects=[],
                    raw_reference_pointers=["base/sphere_cli/runtime.py"],
                    alternative_paths=[],
                ),
                "evidence": SimpleNamespace(
                    candidates=[{"id": "recent"}],
                    diagnostics={"channel_stats": {"dense": 2, "lexical": 1}},
                    query_route={"route": "debug"},
                ),
                "completion": SimpleNamespace(core_evidence=[]),
                "cognitive": SimpleNamespace(relevant_experience=[]),
            }
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "dysonspherain.memory_os.recall_service.UnifiedMemoryRuntime.from_base_dir",
            return_value=runtime,
        ):
            result = recall(
                RecallRequest(
                    query="regression",
                    cwd=tmp,
                    include_files=False,
                    include_benchmarks=False,
                    include_prior_prompts=False,
                    freshness="recent",
                )
            )
        self.assertEqual(result.status, "ok")
        rendered = result.rendered_context
        self.assertIn("Recent stable evidence", rendered)
        self.assertNotIn("Old prompt", rendered)
        self.assertNotIn("base/sphere_cli/runtime.py", rendered)
        self.assertEqual(result.context_pack["benchmark_state"], [])
        self.assertEqual(result.trace["retrieval_channels"], ["dense", "lexical"])
        self.assertEqual(result.trace["channel_stats"], {"dense": 2, "lexical": 1})
        self.assertTrue(result.trace["freshness_applied"])
        self.assertFalse(result.trace["include_files"])
        self.assertFalse(result.trace["include_benchmarks"])
        self.assertFalse(result.trace["include_prior_prompts"])


if __name__ == "__main__":
    unittest.main()
