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
from sphere_cli.memory_lifecycle import (
    append_lifecycle_action,
    detect_conflicts,
    lifecycle_actions_path,
    load_conflicts,
    write_conflict_report,
)


def write_metrics(path: Path, *, question_count: int, fallback: bool, recall: float) -> None:
    payload = {
        "benchmark": "clonemem",
        "total_question_count": question_count,
        "embedding_info": {
            "embedding_provider": "local_hash" if fallback else "sentence_transformer",
            "embedding_model": "local_hash" if fallback else "sentence-transformers/all-MiniLM-L6-v2",
            "fallback_in_use": fallback,
        },
        "metrics": {"recall_frac@10": recall, "candidate_recall@100": 0.5},
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class MemoryLifecycleTests(unittest.TestCase):
    def test_detects_fallback_and_smoke_full_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "full" / "clonemem" / "metrics.json", question_count=2374, fallback=False, recall=0.3)
            write_metrics(root / "smoke" / "clonemem" / "metrics.json", question_count=20, fallback=True, recall=0.1)
            ingest_artifacts(root, base_dir=root, project="DysonSpherain")

            conflicts = detect_conflicts(root, "DysonSpherain")
            types = {conflict.conflict_type for conflict in conflicts}

            self.assertIn("fallback_conflict", types)
            self.assertIn("smoke_full_conflict", types)
            self.assertTrue(all(conflict.recommended_winner for conflict in conflicts))

    def test_conflict_report_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "full" / "clonemem" / "metrics.json", question_count=2374, fallback=False, recall=0.3)
            write_metrics(root / "smoke" / "clonemem" / "metrics.json", question_count=20, fallback=True, recall=0.1)
            ingest_artifacts(root, base_dir=root, project="DysonSpherain")
            conflicts = detect_conflicts(root, "DysonSpherain")

            write_conflict_report(root, conflicts)
            loaded = load_conflicts(root)

            self.assertEqual(len(loaded), len(conflicts))
            self.assertEqual(loaded[0].project, "DysonSpherain")

    def test_lifecycle_actions_are_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            first = append_lifecycle_action(root, "pin", {"memory_id": "m1"})
            second = append_lifecycle_action(root, "archive", {"memory_id": "m2"})

            lines = lifecycle_actions_path(root).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertNotEqual(first["action_id"], second["action_id"])


if __name__ == "__main__":
    unittest.main()
