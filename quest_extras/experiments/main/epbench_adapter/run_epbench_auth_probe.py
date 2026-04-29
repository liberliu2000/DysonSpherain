from __future__ import annotations

import argparse
import json
import pandas as pd
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded EpBench answering auth probe on cached official data."
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
        default=Path("artifacts/experiment/epbench-wave1-auth-probe"),
        help="Durable output root for auth-probe outputs.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Explicit .env path. Defaults to <epbench-root>/.env.",
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=None,
        help="Explicit data folder. Defaults to <epbench-root>/epbench/data.",
    )
    parser.add_argument(
        "--book-nb-events",
        type=int,
        default=20,
        help="Official short-book event count for the auth probe.",
    )
    parser.add_argument(
        "--answering-model-name",
        default="gpt-4o-mini-2024-07-18",
        help="Answering model name to probe with one real API call.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL override for quest-local provider probes.",
    )
    parser.add_argument(
        "--question-index",
        type=int,
        default=0,
        help="Zero-based question index to probe.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Completion cap for the single auth-probe call.",
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


def parse_nb_chapters(path: Path) -> int:
    marker = "nbchapters_"
    start = path.name.find(marker)
    if start == -1:
        raise ValueError(f"Could not parse nbchapters from {path.name}")
    start += len(marker)
    end = path.name.find("_", start)
    if end == -1:
        end = len(path.name)
    return int(path.name[start:end])


def locate_cached_book_dir(data_folder: Path, book_nb_events: int) -> Path:
    books_root = data_folder / "Udefault_Sdefault_seed0" / "books"
    if not books_root.exists():
        raise FileNotFoundError(f"Books root missing: {books_root}")
    candidates = [
        path
        for path in books_root.glob("model_claude-3-5-sonnet-20240620_itermax_10_Idefault_nbchapters_*")
        if (path / "book.json").exists() and (path / "df_qa.parquet").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No cached default EpBench book directories found under {books_root}")
    if book_nb_events <= 20:
        return min(candidates, key=parse_nb_chapters)
    return max(candidates, key=parse_nb_chapters)


def load_book_text(book_json_path: Path) -> str:
    payload = json.loads(book_json_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"Expected one-item list in {book_json_path}, got {len(payload)} items")
        return str(payload[0])
    return str(payload)


def generate_via_direct_openai_client(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    user_prompt: str,
    system_prompt: str,
    max_new_tokens: int,
) -> tuple[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if ("gpt-5" in model_name) or ("gpt-4.1" in model_name):
        request_kwargs["max_completion_tokens"] = max_new_tokens
    else:
        request_kwargs["max_tokens"] = max_new_tokens
    outputs = client.chat.completions.create(**request_kwargs)
    answer = outputs.choices[0].message.content
    reasoning = getattr(outputs.choices[0].message, "reasoning_content", None)
    return str(answer), reasoning


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_output_root(output_root, args.clean_output)

    epbench_root = args.epbench_root.resolve()
    env_file = (args.env_file or (epbench_root / ".env")).resolve()
    data_folder = (args.data_folder or (epbench_root / "epbench" / "data")).resolve()

    if not epbench_root.exists():
        raise FileNotFoundError(f"EpBench root missing: {epbench_root}")
    if not env_file.exists():
        raise FileNotFoundError(f"Env file missing: {env_file}")
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder missing: {data_folder}")

    sys.path.insert(0, str(epbench_root))

    from epbench.src.evaluation.prompts import generate_episodic_memory_prompt
    from epbench.src.models.models_wrapper import ModelsWrapper
    from epbench.src.models.settings_wrapper import SettingsWrapper

    summary: dict[str, Any] = {
        "created_at": now_iso(),
        "success": False,
        "epbench_root": str(epbench_root),
        "env_file": str(env_file),
        "data_folder": str(data_folder),
        "book_nb_events": args.book_nb_events,
        "answering_model_name": args.answering_model_name,
        "openai_base_url": args.openai_base_url,
        "question_index": args.question_index,
        "max_new_tokens": args.max_new_tokens,
    }

    try:
        cached_book_dir = locate_cached_book_dir(data_folder, args.book_nb_events)
        df_qa = pd.read_parquet(cached_book_dir / "df_qa.parquet", engine="pyarrow")
        if args.question_index < 0 or args.question_index >= len(df_qa):
            raise IndexError(
                f"question_index={args.question_index} out of range for {len(df_qa)} questions"
            )

        row = df_qa.iloc[args.question_index]
        question = row["question"]
        book_text = load_book_text(cached_book_dir / "book.json")
        prompt = generate_episodic_memory_prompt(book_text, question)
        config = SettingsWrapper(_env_file=str(env_file))

        start = time.time()
        if args.openai_base_url:
            answer, reasoning = generate_via_direct_openai_client(
                api_key=config.OPENAI_API_KEY,
                base_url=args.openai_base_url,
                model_name=args.answering_model_name,
                user_prompt=prompt,
                system_prompt="You are an expert in memory tests.",
                max_new_tokens=args.max_new_tokens,
            )
            client_mode = "direct_openai_client"
        else:
            model = ModelsWrapper(args.answering_model_name, config)
            answer, reasoning = model.generate(
                user_prompt=prompt,
                system_prompt="You are an expert in memory tests.",
                max_new_tokens=args.max_new_tokens,
                keep_reasoning=True,
            )
            client_mode = "epbench_models_wrapper"
        duration_seconds = round(time.time() - start, 3)

        (output_root / "prompt.txt").write_text(prompt, encoding="utf-8")
        (output_root / "answer.txt").write_text(str(answer), encoding="utf-8")
        if reasoning is not None:
            (output_root / "reasoning.txt").write_text(str(reasoning), encoding="utf-8")

        summary.update(
            {
                "success": True,
                "duration_seconds": duration_seconds,
                "client_mode": client_mode,
                "question_count": int(len(df_qa)),
                "cached_book_dir": str(cached_book_dir),
                "question_text": str(question),
                "question_row": {key: str(value) for key, value in row.to_dict().items()},
                "answer_path": str((output_root / "answer.txt").resolve()),
                "reasoning_path": str((output_root / "reasoning.txt").resolve())
                if reasoning is not None
                else None,
                "prompt_path": str((output_root / "prompt.txt").resolve()),
            }
        )
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=True))
        return 0
    except Exception as exc:
        summary.update(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback_path": str((output_root / "traceback.txt").resolve()),
            }
        )
        (output_root / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
