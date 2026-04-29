from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded multi-question EpBench provider smoke on cached official data."
    )
    parser.add_argument(
        "--epbench-root",
        type=Path,
        default=Path("tmp/epbench_source"),
        help="Quest-local EpBench source snapshot root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/experiment/epbench-wave1-provider-smoke"),
        help="Durable output root for the provider smoke bundle.",
    )
    parser.add_argument(
        "--question-indices",
        default="0,50,100",
        help="Comma-separated zero-based question indices to probe.",
    )
    parser.add_argument(
        "--answering-model-name",
        default="gpt-5.4",
        help="Answering model name for the bounded provider smoke.",
    )
    parser.add_argument(
        "--openai-base-url",
        default="https://right.codes/codex/v1",
        help="OpenAI-compatible base URL for the provider smoke.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Completion cap for each bounded probe call.",
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


def parse_question_indices(raw_value: str) -> list[int]:
    indices: list[int] = []
    for chunk in raw_value.split(","):
        item = chunk.strip()
        if not item:
            continue
        indices.append(int(item))
    if not indices:
        raise ValueError("question_indices must contain at least one integer")
    return indices


def render_summary_md(payload: dict[str, Any]) -> str:
    lines = [
        "# EpBench Provider Smoke Summary",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- output_root: `{payload.get('output_root')}`",
        f"- answering_model_name: `{payload['answering_model_name']}`",
        f"- openai_base_url: `{payload['openai_base_url']}`",
        f"- question_indices: `{payload['question_indices']}`",
        f"- success_count: `{payload.get('success_count')}` / `{payload['question_count']}`",
        f"- mean_success_duration_seconds: `{payload.get('mean_success_duration_seconds')}`",
        "",
        "## Question Results",
        "",
    ]
    for row in payload["questions"]:
        status = "success" if row["success"] else "failure"
        lines.append(
            f"- q{row['question_index']}: `{status}`, duration=`{row['duration_seconds']}`, "
            f"retrieval_type=`{row.get('retrieval_type')}`, get_style=`{row.get('get_style')}`, "
            f"error=`{row.get('error_type')}`"
        )
        lines.append(f"  - question: {row['question_text']}")
        if row.get("correct_answer"):
            lines.append(f"  - correct_answer: {row['correct_answer']}")
        if row.get("error_message"):
            lines.append(f"  - error_message: {row['error_message']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_output_root(output_root, args.clean_output)

    workspace_root = Path(__file__).resolve().parents[3]
    auth_probe_path = Path(__file__).with_name("run_epbench_auth_probe.py")
    question_indices = parse_question_indices(args.question_indices)

    aggregate: dict[str, Any] = {
        "created_at": now_iso(),
        "workspace_root": str(workspace_root),
        "output_root": str(output_root),
        "auth_probe_path": str(auth_probe_path.resolve()),
        "epbench_root": str(args.epbench_root.resolve()),
        "answering_model_name": args.answering_model_name,
        "openai_base_url": args.openai_base_url,
        "question_indices": question_indices,
        "question_count": len(question_indices),
        "questions": [],
    }

    success_durations: list[float] = []
    overall_start = time.time()

    try:
        for question_index in question_indices:
            question_output_root = output_root / f"question-{question_index:03d}"
            command = [
                sys.executable,
                str(auth_probe_path),
                "--epbench-root",
                str(args.epbench_root),
                "--output-root",
                str(question_output_root),
                "--answering-model-name",
                args.answering_model_name,
                "--openai-base-url",
                args.openai_base_url,
                "--question-index",
                str(question_index),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--clean-output",
            ]
            started_at = now_iso()
            run_started = time.time()
            completed = subprocess.run(
                command,
                cwd=workspace_root,
                capture_output=True,
                text=True,
            )
            wrapper_duration = round(time.time() - run_started, 3)
            summary_path = question_output_root / "summary.json"
            question_summary: dict[str, Any] = {}
            if summary_path.exists():
                question_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            question_row = {
                "question_index": question_index,
                "started_at": started_at,
                "wrapper_duration_seconds": wrapper_duration,
                "exit_code": completed.returncode,
                "stdout_path": str((question_output_root / "stdout.txt").resolve()),
                "stderr_path": str((question_output_root / "stderr.txt").resolve()),
                "summary_path": str(summary_path.resolve()),
                "success": bool(question_summary.get("success")),
                "duration_seconds": question_summary.get("duration_seconds"),
                "question_text": question_summary.get("question_text"),
                "retrieval_type": (question_summary.get("question_row") or {}).get("retrieval_type"),
                "get_style": (question_summary.get("question_row") or {}).get("get"),
                "correct_answer": (question_summary.get("question_row") or {}).get("correct_answer"),
                "error_type": question_summary.get("error_type"),
                "error_message": question_summary.get("error_message"),
            }
            (question_output_root / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (question_output_root / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if question_row["success"] and isinstance(question_row["duration_seconds"], (int, float)):
                success_durations.append(float(question_row["duration_seconds"]))
            aggregate["questions"].append(question_row)

        aggregate["total_wall_clock_seconds"] = round(time.time() - overall_start, 3)
        aggregate["success_count"] = sum(1 for row in aggregate["questions"] if row["success"])
        aggregate["failure_count"] = aggregate["question_count"] - aggregate["success_count"]
        aggregate["all_successful"] = aggregate["failure_count"] == 0
        aggregate["mean_success_duration_seconds"] = (
            round(sum(success_durations) / len(success_durations), 3) if success_durations else None
        )

        write_json(output_root / "summary.json", aggregate)
        (output_root / "summary.md").write_text(render_summary_md(aggregate), encoding="utf-8")
        print(json.dumps(aggregate, ensure_ascii=True))
        return 0
    except Exception as exc:
        failure_payload = {
            **aggregate,
            "success_count": aggregate.get("success_count", 0),
            "failure_count": aggregate.get("failure_count", 0),
            "all_successful": False,
            "mean_success_duration_seconds": None,
            "fatal_error_type": type(exc).__name__,
            "fatal_error_message": str(exc),
        }
        write_json(output_root / "summary.json", failure_payload)
        (output_root / "summary.md").write_text(render_summary_md(failure_payload), encoding="utf-8")
        print(json.dumps(failure_payload, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
