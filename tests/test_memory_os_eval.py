from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.context_compiler import compile_context_packet, estimate_tokens
from sphere_cli.experiment_registry import ingest_artifacts
from sphere_cli.memory_lifecycle import detect_conflicts, write_conflict_report
from sphere_cli.project_state import update_project_state_from_registry, write_memory


def write_metrics(path: Path, *, fallback: bool, question_count: int, recall: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.write_text(json.dumps(payload), encoding="utf-8")


class MemoryOSEvaluationTests(unittest.TestCase):
    def test_context_packet_propagates_state_conflict_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "full" / "clonemem" / "metrics.json", fallback=False, question_count=2374, recall=0.3)
            write_metrics(root / "smoke" / "clonemem" / "metrics.json", fallback=True, question_count=20, recall=0.1)
            ingest_artifacts(root, base_dir=root, project="DysonSpherain")
            update_project_state_from_registry(root, "DysonSpherain")
            write_conflict_report(root, detect_conflicts(root, "DysonSpherain"))

            packet = compile_context_packet(
                root,
                task="What should Codex do next to debug retrieval regression?",
                project="DysonSpherain",
                mode="codex",
                max_tokens=1800,
            )

            self.assertIn("Current Project State", packet)
            self.assertIn("Conflict Warnings", packet)
            self.assertIn("fallback_conflict", packet)
            self.assertIn("Do not hardcode benchmark gold ids", packet)
            self.assertLessEqual(estimate_tokens(packet), 1800)

    def test_context_packet_redacts_memory_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_memory(root, memory_type="task", project="DysonSpherain", content="api_key=sk-abcdef1234567890", source="eval")

            packet = compile_context_packet(root, task="status", project="DysonSpherain", mode="codex")

            self.assertNotIn("sk-abcdef", packet)


if __name__ == "__main__":
    unittest.main()
