from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a quest-local EpBench question pack and real_task_eval datasets."
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
        default=Path("artifacts/experiment/epbench-wave1-question-pack"),
        help="Durable output root for exported question-pack assets.",
    )
    parser.add_argument(
        "--book-nb-events",
        type=int,
        default=20,
        help="Official short-book event count to target.",
    )
    parser.add_argument(
        "--focus-get",
        nargs="+",
        default=["latest", "chronological"],
        help="EpBench `get` values to keep in the exported temporal pack.",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=12,
        help="Case count for the bounded real_task_eval smoke dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for balanced smoke selection.",
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


def normalize_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif value is None:
        items = []
    else:
        items = [value]
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def normalize_case_row(row: pd.Series) -> dict[str, Any]:
    answers = normalize_answers(row.get("correct_answer"))
    answer_chapters = [int(item) for item in list(row.get("correct_answer_chapters") or [])]
    expected_core_terms = answers[:1] if len(answers) == 1 else []
    expected_latest_terms = answers if str(row.get("get", "")) in {"latest", "chronological"} else []
    return {
        "task_id": f"epbench_q{int(row['q_idx']):04d}_{str(row.get('get', 'all')).lower()}",
        "task_type": "qa",
        "category": "temporal",
        "query": str(row["question"]),
        "expected_core_terms": expected_core_terms,
        "expected_any_terms": answers,
        "expected_latest_terms": expected_latest_terms,
        "require_raw_evidence": True,
        "must_not_be_experience_only": True,
        "epbench_meta": {
            "q_idx": int(row["q_idx"]),
            "get": str(row.get("get", "")),
            "retrieval_type": str(row.get("retrieval_type", "")),
            "cue": str(row.get("cue", "")),
            "n_chapters_correct_answer": int(row.get("n_chapters_correct_answer", 0) or 0),
            "correct_answer_chapters": answer_chapters,
            "correct_answer_detailed": str(row.get("correct_answer_detailed", "")),
        },
    }


def select_balanced_smoke_rows(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df.copy()
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in df.iterrows():
        key = (str(row.get("get", "")), str(row.get("retrieval_type", "")))
        grouped[key].append(index)

    selected: list[int] = []
    keys = sorted(grouped)
    for key in keys:
        bucket = grouped[key][:]
        rng.shuffle(bucket)
        if bucket:
            selected.append(bucket.pop(0))
            grouped[key] = bucket
        if len(selected) >= limit:
            break

    while len(selected) < limit:
        progress = False
        for key in keys:
            bucket = grouped[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progress = True
            if len(selected) >= limit:
                break
        if not progress:
            break

    return df.loc[sorted(selected)].reset_index(drop=True)


def distribution(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = Counter(str(item) for item in df[column].tolist())
    return dict(sorted(counts.items()))


def render_summary(
    *,
    source_book_dir: Path,
    filtered_df: pd.DataFrame,
    smoke_df: pd.DataFrame,
    book_path: Path,
    full_dataset_path: Path,
    smoke_dataset_path: Path,
) -> str:
    full_dist = distribution(filtered_df, "get")
    smoke_dist = distribution(smoke_df, "get")
    full_retrieval = distribution(filtered_df, "retrieval_type")
    smoke_retrieval = distribution(smoke_df, "retrieval_type")
    return "\n".join(
        [
            "# EpBench Question Pack Export Summary",
            "",
            "## Source",
            "",
            f"- cached book dir: `{source_book_dir}`",
            f"- exported book file: `{book_path}`",
            "",
            "## Coverage",
            "",
            f"- focused question count: `{len(filtered_df)}`",
            f"- smoke question count: `{len(smoke_df)}`",
            f"- focused `get` distribution: `{full_dist}`",
            f"- smoke `get` distribution: `{smoke_dist}`",
            f"- focused retrieval distribution: `{full_retrieval}`",
            f"- smoke retrieval distribution: `{smoke_retrieval}`",
            "",
            "## Outputs",
            "",
            f"- full question pack dataset: `{full_dataset_path}`",
            f"- smoke real_task_eval dataset: `{smoke_dataset_path}`",
        ]
    )


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_output_root(output_root, args.clean_output)

    epbench_root = args.epbench_root.resolve()
    data_folder = (epbench_root / "epbench" / "data").resolve()
    source_book_dir = locate_cached_book_dir(data_folder, args.book_nb_events)
    book_text = load_book_text(source_book_dir / "book.json")
    df_qa = pd.read_parquet(source_book_dir / "df_qa.parquet", engine="pyarrow")

    focus_get = {item.strip() for item in args.focus_get if item.strip()}
    filtered_df = df_qa[df_qa["get"].astype(str).isin(focus_get)].copy().reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError(f"No EpBench questions matched focus_get={sorted(focus_get)}")

    smoke_df = select_balanced_smoke_rows(filtered_df, args.smoke_limit, args.seed)

    book_path = output_root / "epbench_book.txt"
    book_path.write_text(book_text, encoding="utf-8")

    full_cases = [normalize_case_row(row) for _, row in filtered_df.iterrows()]
    smoke_cases = [normalize_case_row(row) for _, row in smoke_df.iterrows()]

    setup = {
        "memories": [],
        "files": [
            {
                "path": "epbench_book.txt",
                "shell": 4,
                "sector": "raw",
                "zone": "epbench_book",
                "stage": "long_term",
                "tags": "epbench,official_cache,temporal",
            }
        ],
    }

    pack_manifest = {
        "created_at": now_iso(),
        "source_book_dir": str(source_book_dir),
        "book_nb_events": args.book_nb_events,
        "book_char_count": len(book_text),
        "focused_get": sorted(focus_get),
        "focused_question_count": len(filtered_df),
        "smoke_question_count": len(smoke_df),
        "focused_get_distribution": distribution(filtered_df, "get"),
        "focused_retrieval_distribution": distribution(filtered_df, "retrieval_type"),
        "smoke_get_distribution": distribution(smoke_df, "get"),
        "smoke_retrieval_distribution": distribution(smoke_df, "retrieval_type"),
        "source_question_columns": list(df_qa.columns),
    }
    write_json(output_root / "pack_manifest.json", pack_manifest)

    full_dataset = {
        "name": "epbench_temporal_question_pack_full",
        "setup": setup,
        "cases": full_cases,
    }
    smoke_dataset = {
        "name": "epbench_temporal_question_pack_smoke",
        "setup": setup,
        "cases": smoke_cases,
    }
    full_dataset_path = output_root / "real_task_eval_temporal_full.json"
    smoke_dataset_path = output_root / "real_task_eval_temporal_smoke.json"
    write_json(full_dataset_path, full_dataset)
    write_json(smoke_dataset_path, smoke_dataset)

    raw_pack = {
        "created_at": now_iso(),
        "source_book_dir": str(source_book_dir),
        "book_nb_events": args.book_nb_events,
        "rows": [
            {
                "q_idx": int(row["q_idx"]),
                "question": str(row["question"]),
                "retrieval_type": str(row.get("retrieval_type", "")),
                "get": str(row.get("get", "")),
                "cue": str(row.get("cue", "")),
                "correct_answer": normalize_answers(row.get("correct_answer")),
                "correct_answer_chapters": [int(item) for item in list(row.get("correct_answer_chapters") or [])],
                "n_chapters_correct_answer": int(row.get("n_chapters_correct_answer", 0) or 0),
            }
            for _, row in filtered_df.iterrows()
        ],
    }
    write_json(output_root / "question_pack_rows.json", raw_pack)

    summary_md = render_summary(
        source_book_dir=source_book_dir,
        filtered_df=filtered_df,
        smoke_df=smoke_df,
        book_path=book_path,
        full_dataset_path=full_dataset_path,
        smoke_dataset_path=smoke_dataset_path,
    )
    (output_root / "summary.md").write_text(summary_md + "\n", encoding="utf-8")
    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
