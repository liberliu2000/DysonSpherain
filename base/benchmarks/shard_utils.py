from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable


def stable_content_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load_allowlist(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Allowlist not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.get("ids") or payload.get("question_ids") or payload.get("sample_ids") or []
        else:
            values = payload
        return {str(item) for item in values}
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def validate_shard_args(shard_index: int | None, shard_count: int | None) -> tuple[int | None, int | None]:
    if shard_index is None and shard_count is None:
        return None, None
    if shard_index is None or shard_count is None:
        raise ValueError("--shard-index and --shard-count must be provided together")
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    return shard_index, shard_count


def shard_key(
    *,
    benchmark_name: str,
    question_id: str | None,
    sample_id: str | None,
    question_text: str | None,
) -> tuple[str, str]:
    if question_id:
        return str(question_id), "question_id_hash"
    fallback = "|".join([benchmark_name, str(sample_id or ""), str(question_text or "")])
    return fallback, "benchmark_sample_question_hash"


def belongs_to_shard(
    *,
    benchmark_name: str,
    question_id: str | None,
    sample_id: str | None,
    question_text: str | None,
    shard_index: int | None,
    shard_count: int | None,
) -> tuple[bool, str]:
    shard_index, shard_count = validate_shard_args(shard_index, shard_count)
    key, method = shard_key(
        benchmark_name=benchmark_name,
        question_id=question_id,
        sample_id=sample_id,
        question_text=question_text,
    )
    if shard_index is None or shard_count is None:
        return True, method
    return stable_content_hash(key) % shard_count == shard_index, method


def filter_sharded_items(
    items: Iterable[Any],
    *,
    benchmark_name: str,
    shard_index: int | None,
    shard_count: int | None,
    question_id_getter: Callable[[Any], str | None],
    sample_id_getter: Callable[[Any], str | None],
    question_text_getter: Callable[[Any], str | None],
    question_id_allowlist: set[str] | None = None,
    sample_id_allowlist: set[str] | None = None,
    max_questions: int = 0,
) -> tuple[list[Any], dict[str, Any]]:
    question_id_allowlist = question_id_allowlist or set()
    sample_id_allowlist = sample_id_allowlist or set()
    selected: list[Any] = []
    total = 0
    method = "unsharded"
    for item in items:
        total += 1
        question_id = question_id_getter(item)
        sample_id = sample_id_getter(item)
        question_text = question_text_getter(item)
        if question_id_allowlist and str(question_id or "") not in question_id_allowlist:
            continue
        if sample_id_allowlist and str(sample_id or "") not in sample_id_allowlist:
            continue
        belongs, item_method = belongs_to_shard(
            benchmark_name=benchmark_name,
            question_id=question_id,
            sample_id=sample_id,
            question_text=question_text,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        method = item_method if method == "unsharded" else method
        if not belongs:
            continue
        selected.append(item)
        if max_questions > 0 and len(selected) >= max_questions:
            break
    meta = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shard_question_count": len(selected),
        "total_available_question_count": total,
        "shard_assignment_method": method,
        "question_id_allowlist_count": len(question_id_allowlist),
        "sample_id_allowlist_count": len(sample_id_allowlist),
        "max_questions": int(max_questions or 0),
    }
    return selected, meta

