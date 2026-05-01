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
from sphere_cli.project_state import update_project_state_from_registry


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


class ContextCompilerTests(unittest.TestCase):
    def test_codex_packet_contains_required_sections_and_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_metrics(root / "full" / "clonemem" / "metrics.json", question_count=2374, fallback=False, recall=0.3)
            write_metrics(root / "smoke" / "clonemem" / "metrics.json", question_count=20, fallback=True, recall=0.1)
            ingest_artifacts(root, base_dir=root, project="DysonSpherain")
            update_project_state_from_registry(root, "DysonSpherain")
            write_conflict_report(root, detect_conflicts(root, "DysonSpherain"))

            packet = compile_context_packet(
                root,
                task="Fix current retrieval regression",
                project="DysonSpherain",
                mode="codex",
                max_tokens=2000,
            )

            self.assertIn("# Runtime Context Packet", packet)
            self.assertIn("Task Objective", packet)
            self.assertIn("Current Project State", packet)
            self.assertIn("Conflict Warnings", packet)
            self.assertIn("fallback_conflict", packet)
            self.assertLessEqual(estimate_tokens(packet), 2000)

    def test_paper_and_benchmark_modes_render_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            paper = compile_context_packet(root, task="Write paper experiment section", project="DysonSpherain", mode="paper")
            benchmark = compile_context_packet(root, task="Run benchmark diagnostics", project="DysonSpherain", mode="benchmark")

            self.assertIn("# Paper Context Packet", paper)
            self.assertIn("# Benchmark Context Packet", benchmark)


if __name__ == "__main__":
    unittest.main()
