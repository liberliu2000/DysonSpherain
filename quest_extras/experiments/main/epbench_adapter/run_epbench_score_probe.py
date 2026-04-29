from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded quest-local EpBench score probe on existing provider answers."
    )
    parser.add_argument(
        "--epbench-root",
        type=Path,
        default=Path("tmp/epbench_source"),
        help="Quest-local EpBench source snapshot root.",
    )
    parser.add_argument(
        "--question-bundle-root",
        type=Path,
        default=Path("artifacts/experiment/epbench-wave2-provider-coverage-smoke"),
        help="Existing provider-answer bundle to score.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/experiment/epbench-wave3-provider-score-probe"),
        help="Durable output root for the score probe bundle.",
    )
    parser.add_argument(
        "--question-indices",
        default="0,50,56,394",
        help="Comma-separated zero-based question indices to score.",
    )
    parser.add_argument(
        "--judge-model-name",
        default="gpt-5.4",
        help="Judge model name for the quest-local score probe.",
    )
    parser.add_argument(
        "--openai-base-url",
        default="https://right.codes/codex/v1",
        help="OpenAI-compatible base URL for the quest-local judge override.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("tmp/epbench_source/.env"),
        help="Env file used to load the OpenAI-compatible API key.",
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


def parse_correct_answer(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    if isinstance(raw_value, str):
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, str):
            return [parsed]
    raise ValueError(f"unsupported correct_answer format: {raw_value!r}")


def load_question_summary(question_root: Path) -> dict[str, Any]:
    summary_path = question_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"question summary missing: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_answer_text(question_summary: dict[str, Any], question_root: Path) -> str:
    answer_path = question_summary.get("answer_path")
    if answer_path:
        path = Path(answer_path)
    else:
        path = question_root / "answer.txt"
    if not path.exists():
        raise FileNotFoundError(f"answer text missing: {path}")
    return path.read_text(encoding="utf-8").strip()


def render_summary_md(payload: dict[str, Any]) -> str:
    lines = [
        "# EpBench Quest-Local Score Probe Summary",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- output_root: `{payload['output_root']}`",
        f"- source_bundle_root: `{payload['source_bundle_root']}`",
        f"- judge_model_name: `{payload['judge_model_name']}`",
        f"- openai_base_url: `{payload['openai_base_url']}`",
        f"- question_indices: `{payload['question_indices']}`",
        f"- success_count: `{payload['success_count']}` / `{payload['question_count']}`",
        f"- mean_f1_score: `{payload.get('mean_f1_score')}`",
        "",
        "## Caveat",
        "",
        "- This bundle uses a quest-local judge override and is not an official comparable EpBench score.",
        "",
        "## Question Results",
        "",
    ]
    for row in payload["questions"]:
        status = "success" if row["success"] else "failure"
        lines.append(
            f"- q{row['question_index']}: `{status}`, retrieval_type=`{row.get('retrieval_type')}`, "
            f"get_style=`{row.get('get_style')}`, f1=`{row.get('f1_score')}`, "
            f"error=`{row.get('error_type')}`"
        )
        lines.append(f"  - question: {row.get('question_text')}")
        if row.get("error_message"):
            lines.append(f"  - error_message: {row['error_message']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    epbench_root = args.epbench_root.resolve()
    source_bundle_root = args.question_bundle_root.resolve()
    output_root = args.output_root.resolve()
    env_file = args.env_file.resolve()
    ensure_output_root(output_root, args.clean_output)

    sys.path.insert(0, str(epbench_root))
    from epbench.src.evaluation.scoring_answers import evaluate_answer
    from epbench.src.models.models_wrapper import ModelsWrapper
    from epbench.src.models.settings_wrapper import SettingsWrapper

    config = SettingsWrapper(_env_file=str(env_file))
    config.OPENAI_BASE_URL = args.openai_base_url
    judge_model = ModelsWrapper(args.judge_model_name, config)
    question_indices = parse_question_indices(args.question_indices)

    aggregate: dict[str, Any] = {
        "created_at": now_iso(),
        "epbench_root": str(epbench_root),
        "source_bundle_root": str(source_bundle_root),
        "output_root": str(output_root),
        "env_file": str(env_file),
        "judge_model_name": args.judge_model_name,
        "openai_base_url": args.openai_base_url,
        "question_indices": question_indices,
        "question_count": len(question_indices),
        "probe_kind": "quest-local-judge-override",
        "official_comparable": False,
        "questions": [],
    }

    f1_scores: list[float] = []
    overall_started = time.time()

    try:
        for question_index in question_indices:
            question_root = source_bundle_root / f"question-{question_index:03d}"
            output_question_root = output_root / f"question-{question_index:03d}"
            output_question_root.mkdir(parents=True, exist_ok=True)
            started_at = now_iso()
            run_started = time.time()

            try:
                question_summary = load_question_summary(question_root)
                if not question_summary.get("success"):
                    raise RuntimeError("source question bundle entry is not successful")

                question_row = question_summary.get("question_row") or {}
                retrieval_type = question_row.get("retrieval_type")
                get_style = question_row.get("get")
                if get_style == "chronological":
                    raise ValueError("chronological scoring is intentionally excluded from this bounded probe")
                if retrieval_type == "Full event details":
                    raise ValueError("full-event-details scoring is intentionally excluded from this bounded probe")

                answer_text = load_answer_text(question_summary, question_root)
                correct_answer = parse_correct_answer(question_row.get("correct_answer"))
                evaluation = evaluate_answer(
                    answer_text,
                    correct_answer,
                    retrieval_type,
                    judge_model,
                    None,
                    get_style,
                )
                duration_seconds = round(time.time() - run_started, 3)

                evaluation_path = output_question_root / "evaluation.json"
                judge_meta_path = output_question_root / "judge_meta.json"
                write_json(evaluation_path, evaluation)
                judge_meta = {
                    "question_index": question_index,
                    "started_at": started_at,
                    "duration_seconds": duration_seconds,
                    "judge_model_name": args.judge_model_name,
                    "openai_base_url": args.openai_base_url,
                    "retrieval_type": retrieval_type,
                    "get_style": get_style,
                    "question_text": question_summary.get("question_text"),
                    "source_summary_path": str((question_root / "summary.json").resolve()),
                    "evaluation_path": str(evaluation_path.resolve()),
                    "official_comparable": False,
                }
                write_json(judge_meta_path, judge_meta)
                row = {
                    **judge_meta,
                    "success": True,
                    "precision": evaluation.get("precision"),
                    "recall": evaluation.get("recall"),
                    "f1_score": evaluation.get("f1_score"),
                    "nb_preds": evaluation.get("nb_preds"),
                    "nb_gt": evaluation.get("nb_gt"),
                    "error_type": None,
                    "error_message": None,
                }
                if isinstance(row["f1_score"], (int, float)):
                    f1_scores.append(float(row["f1_score"]))
            except Exception as exc:
                duration_seconds = round(time.time() - run_started, 3)
                row = {
                    "question_index": question_index,
                    "started_at": started_at,
                    "duration_seconds": duration_seconds,
                    "judge_model_name": args.judge_model_name,
                    "openai_base_url": args.openai_base_url,
                    "success": False,
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "nb_preds": None,
                    "nb_gt": None,
                    "retrieval_type": None,
                    "get_style": None,
                    "question_text": None,
                    "source_summary_path": str((question_root / "summary.json").resolve()),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "official_comparable": False,
                }
                write_json(output_question_root / "judge_meta.json", row)
            aggregate["questions"].append(row)

        aggregate["total_wall_clock_seconds"] = round(time.time() - overall_started, 3)
        aggregate["success_count"] = sum(1 for row in aggregate["questions"] if row["success"])
        aggregate["failure_count"] = aggregate["question_count"] - aggregate["success_count"]
        aggregate["all_successful"] = aggregate["failure_count"] == 0
        aggregate["mean_f1_score"] = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else None
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
            "mean_f1_score": None,
            "fatal_error_type": type(exc).__name__,
            "fatal_error_message": str(exc),
        }
        write_json(output_root / "summary.json", failure_payload)
        (output_root / "summary.md").write_text(render_summary_md(failure_payload), encoding="utf-8")
        print(json.dumps(failure_payload, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
