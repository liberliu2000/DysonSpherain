from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.experiment_registry import BenchmarkRun, compare_runs, latest_run


def run(**overrides: object) -> BenchmarkRun:
    payload = {
        "run_id": "run-a",
        "project": "DysonSpherain",
        "dataset": "clonemem",
        "run_type": "full",
        "timestamp": "2026-04-27T00:00:00+00:00",
        "artifact_dir": "/tmp/run-a",
        "metrics": {"candidate_recall@100": 0.5},
        "question_count": 2374,
        "sample_count": 20,
        "elapsed_seconds": 1.0,
        "embedding_provider": "sentence_transformer",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "fallback_in_use": False,
        "metadata": {"route_policy_hash": "policy-a"},
    }
    payload.update(overrides)
    return BenchmarkRun(**payload)


class RunComparabilityTests(unittest.TestCase):
    def test_compare_warns_on_route_policy_hash_difference(self) -> None:
        a = run(run_id="a", metadata={"route_policy_hash": "policy-a"})
        b = run(run_id="b", metadata={"route_policy_hash": "policy-b"})

        comparison = compare_runs(a, b)

        self.assertIn("different_route_policy_config", comparison["warnings"])

    def test_latest_does_not_let_smoke_override_full(self) -> None:
        full = run(run_id="full", run_type="full", timestamp="2026-04-26T00:00:00+00:00")
        smoke = run(run_id="smoke", run_type="smoke", timestamp="2026-04-27T00:00:00+00:00", question_count=20)

        selected = latest_run([full, smoke], project="DysonSpherain", dataset="clonemem")

        self.assertEqual(selected.run_id, "full")

    def test_nonfallback_outranks_fallback(self) -> None:
        fallback = run(run_id="fallback", timestamp="2026-04-27T00:00:00+00:00", fallback_in_use=True)
        official = run(run_id="official", timestamp="2026-04-26T00:00:00+00:00", fallback_in_use=False)

        selected = latest_run([fallback, official], project="DysonSpherain", dataset="clonemem")

        self.assertEqual(selected.run_id, "official")

    def test_full_runs_with_same_question_count_do_not_raise_hard_sample_warning(self) -> None:
        a = run(run_id="a", sample_count=20, question_count=2374)
        b = run(run_id="b", sample_count=10, question_count=2374)

        comparison = compare_runs(a, b)

        self.assertNotIn("different_sample_count", comparison["warnings"])
        self.assertIn("different_sample_count_definition", comparison["warnings"])


if __name__ == "__main__":
    unittest.main()
