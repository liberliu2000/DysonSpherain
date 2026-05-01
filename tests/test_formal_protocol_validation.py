from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from validate_formal_protocol import build_report


def write_registry(root: Path, rows: list[dict]) -> None:
    path = root / "artifacts" / "registry" / "benchmark_runs.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def run_row(
    dataset: str,
    *,
    run_id: str | None = None,
    timestamp: str = "2026-04-28T00:00:00+00:00",
    elapsed: float = 1.0,
    fallback: bool = False,
    provider: str = "sentence_transformer",
    recall_any: float = 1.0,
    config_hash: str | None = "cfg",
    dataset_version: str | None = "test",
) -> dict:
    expected = {"longmemeval": 500, "locomo": 1986, "knowme": 1010, "clonemem": 2374}[dataset]
    return {
        "run_id": run_id or f"{dataset}-run",
        "project": "DysonSpherain",
        "dataset": dataset,
        "run_type": "full",
        "timestamp": timestamp,
        "artifact_dir": f"/tmp/{dataset}",
        "metrics": {"recall_any@10": recall_any, "candidate_recall@100": 1.0},
        "question_count": expected,
        "elapsed_seconds": elapsed,
        "embedding_provider": provider,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2" if provider != "local_hash" else "local_hash",
        "fallback_in_use": fallback,
        "command": "python benchmark.py",
        "config_hash": config_hash,
        "dataset_version": dataset_version,
        "code_commit": "abc123",
    }


class FormalProtocolValidationTests(unittest.TestCase):
    def test_protocol_fails_local_hash_and_missing_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_registry(
                root,
                [
                    run_row("longmemeval", provider="local_hash"),
                    run_row("locomo"),
                    run_row("knowme"),
                ],
            )

            report = build_report(base_dir=root)

            self.assertEqual(report["overall_status"], "failed")
            longmem = next(row for row in report["full_benchmarks"] if row["benchmark"] == "longmemeval")
            clonemem = next(row for row in report["full_benchmarks"] if row["benchmark"] == "clonemem")
            self.assertIn("local_hash", "; ".join(longmem["errors"]))
            self.assertEqual(clonemem["status"], "missing")

    def test_protocol_marks_redline_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_registry(
                root,
                [
                    run_row("longmemeval", elapsed=601),
                    run_row("locomo"),
                    run_row("knowme"),
                    run_row("clonemem"),
                ],
            )

            report = build_report(base_dir=root)

            longmem = next(row for row in report["full_benchmarks"] if row["benchmark"] == "longmemeval")
            self.assertEqual(longmem["status"], "failed")
            self.assertIn("redline", "; ".join(longmem["errors"]))

    def test_protocol_skips_quality_delta_for_non_comparable_previous_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_registry(
                root,
                [
                    run_row(
                        "longmemeval",
                        run_id="previous",
                        timestamp="2026-04-27T00:00:00+00:00",
                        recall_any=1.0,
                        config_hash="old-cfg",
                    ),
                    run_row(
                        "longmemeval",
                        run_id="current",
                        timestamp="2026-04-28T00:00:00+00:00",
                        recall_any=0.9,
                        config_hash="new-cfg",
                    ),
                    run_row("locomo"),
                    run_row("knowme"),
                    run_row("clonemem"),
                ],
            )

            report = build_report(base_dir=root)

            longmem = next(row for row in report["full_benchmarks"] if row["benchmark"] == "longmemeval")
            self.assertEqual(longmem["comparison_status"], "non_comparable")
            self.assertIsNone(longmem["compared_to_run_id"])
            self.assertNotIn("decreased", "; ".join(longmem["warnings"]))

    def test_protocol_reports_quality_delta_for_comparable_previous_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_registry(
                root,
                [
                    run_row("longmemeval", run_id="previous", timestamp="2026-04-27T00:00:00+00:00", recall_any=1.0),
                    run_row("longmemeval", run_id="current", timestamp="2026-04-28T00:00:00+00:00", recall_any=0.9),
                    run_row("locomo"),
                    run_row("knowme"),
                    run_row("clonemem"),
                ],
            )

            report = build_report(base_dir=root)

            longmem = next(row for row in report["full_benchmarks"] if row["benchmark"] == "longmemeval")
            self.assertEqual(longmem["comparison_status"], "matched")
            self.assertEqual(longmem["compared_to_run_id"], "previous")
            self.assertIn("decreased vs comparable previous", "; ".join(longmem["warnings"]))


if __name__ == "__main__":
    unittest.main()
