from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DysonSpherain real_task_eval on a quest-local EpBench smoke dataset."
    )
    parser.add_argument(
        "--dyson-root",
        type=Path,
        default=Path("/home/liber/Projects/DysonSpherain/sphere_memory_cli_local_models/sphere_memory_cli_next"),
        help="DysonSpherain project root that contains sphere_cli and requirements.txt.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("artifacts/experiment/epbench-wave1-question-pack/real_task_eval_temporal_smoke.json"),
        help="Quest-local real_task_eval dataset path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/experiment/epbench-wave1-real-task-smoke"),
        help="Durable output root for the real_task_eval smoke report.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output root before running.",
    )
    return parser.parse_args()


def ensure_output_root(output_root: Path, clean_output: bool) -> None:
    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def render_summary(summary: dict[str, Any]) -> str:
    quality = summary.get("real_task_quality", {})
    evidence = summary.get("evidence_quality", {})
    engineering = summary.get("engineering_quality", {})
    return "\n".join(
        [
            "# EpBench Real-Task Smoke Summary",
            "",
            "## Quality",
            "",
            f"- total cases: `{quality.get('total_cases', 0)}`",
            f"- passed cases: `{quality.get('passed_cases', 0)}`",
            f"- pass rate: `{quality.get('pass_rate', 0.0)}`",
            f"- evidence recall rate: `{evidence.get('evidence_recall_rate', 0.0)}`",
            f"- temporal consistency rate: `{evidence.get('temporal_consistency_rate', 0.0)}`",
            f"- grounding safety rate: `{evidence.get('grounding_safety_rate', 0.0)}`",
            "",
            "## Engineering",
            "",
            f"- avg latency ms: `{engineering.get('avg_latency_ms', 0.0)}`",
            f"- p95 latency ms: `{engineering.get('p95_latency_ms', 0.0)}`",
            f"- avg context token delta: `{engineering.get('compression', {}).get('avg_context_token_delta', 0.0)}`",
            f"- route distribution: `{engineering.get('route_distribution', {})}`",
            "",
            "## Error counts",
            "",
            f"- `{quality.get('error_counts', {})}`",
        ]
    )


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_output_root(output_root, args.clean_output)

    dyson_root = args.dyson_root.resolve()
    dataset_path = args.dataset_path.resolve()
    report_path = output_root / "real_task_eval_report.json"
    summary_path = output_root / "summary.json"
    summary_md_path = output_root / "summary.md"
    traceback_path = output_root / "traceback.txt"

    if not dyson_root.exists():
        raise FileNotFoundError(f"Dyson root missing: {dyson_root}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path missing: {dataset_path}")

    sys.path.insert(0, str(dyson_root))

    try:
        from sphere_cli.real_task_eval import RealTaskEvaluator

        evaluator = RealTaskEvaluator(dataset_path, report_path)
        report = evaluator.run()
        summary = {
            "created_at": now_iso(),
            "success": True,
            "dyson_root": str(dyson_root),
            "dataset_path": str(dataset_path),
            "report_path": str(report_path),
            "summary": report.get("summary", {}),
        }
        write_json(summary_path, summary)
        summary_md_path.write_text(render_summary(report.get("summary", {})) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=True))
        return 0
    except Exception as exc:
        summary = {
            "created_at": now_iso(),
            "success": False,
            "dyson_root": str(dyson_root),
            "dataset_path": str(dataset_path),
            "report_path": str(report_path),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_path": str(traceback_path),
        }
        traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
