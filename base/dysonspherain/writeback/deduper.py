from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from sphere_cli.project_state import list_memories
from sphere_cli.utils import lexical_score


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def semantic_fingerprint(text: str) -> str:
    words = sorted(set(re.findall(r"[a-zA-Z0-9_\-/\.]+", normalized_text(text))))
    return _hash(" ".join(words[:200]))


def _safe_json(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _list_fingerprint(values: Any) -> str:
    if not isinstance(values, list):
        values = [values] if values not in (None, "", []) else []
    normalized = sorted(" ".join(str(item).lower().split()) for item in values if str(item).strip())
    return _hash(json.dumps(normalized, ensure_ascii=False))


def _payload_from_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = _safe_json(str(record.get("content") or record.get("summary") or ""))
    return payload if payload else {}


def _overlaps(a: Any, b: Any) -> bool:
    if not isinstance(a, list):
        a = [a] if a not in (None, "", []) else []
    if not isinstance(b, list):
        b = [b] if b not in (None, "", []) else []
    left = {" ".join(str(item).lower().split()) for item in a if str(item).strip()}
    right = {" ".join(str(item).lower().split()) for item in b if str(item).strip()}
    return bool(left and right and left.intersection(right))


@dataclass(frozen=True)
class DedupeResult:
    is_duplicate: bool
    duplicate_of: str | None
    dedupe_reason: str
    content_hash: str
    normalized_hash: str
    semantic_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "dedupe_reason": self.dedupe_reason,
            "content_hash": self.content_hash,
            "normalized_hash": self.normalized_hash,
            "semantic_hash": self.semantic_hash,
        }


def classify_duplicate(base_dir: Path, project: str, content: str) -> DedupeResult:
    content_hash = _hash(content)
    normalized_hash = _hash(normalized_text(content))
    semantic_hash = semantic_fingerprint(content)
    current_payload = _safe_json(content)
    current_benchmark_fp = _list_fingerprint(current_payload.get("benchmark_results"))
    current_files_fp = _list_fingerprint(current_payload.get("files_changed"))
    for record in list_memories(base_dir, project, include_archived=True):
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        existing_payload = _payload_from_record(record)
        if metadata.get("content_hash") == content_hash:
            return DedupeResult(True, str(record.get("memory_id") or ""), "exact_content_hash", content_hash, normalized_hash, semantic_hash)
        if metadata.get("normalized_hash") == normalized_hash:
            return DedupeResult(True, str(record.get("memory_id") or ""), "normalized_content_hash", content_hash, normalized_hash, semantic_hash)
        if metadata.get("semantic_hash") == semantic_hash:
            return DedupeResult(True, str(record.get("memory_id") or ""), "semantic_fingerprint", content_hash, normalized_hash, semantic_hash)
        if current_payload.get("benchmark_results") and metadata.get("benchmark_results_hash") and metadata.get("benchmark_results_hash") == current_benchmark_fp:
            return DedupeResult(True, str(record.get("memory_id") or ""), "benchmark_result_dedupe", content_hash, normalized_hash, semantic_hash)
        if current_payload.get("benchmark_results") and _overlaps(current_payload.get("benchmark_results"), metadata.get("benchmark_results") or existing_payload.get("benchmark_results")):
            if current_payload.get("task_goal") == metadata.get("task_goal") or current_payload.get("session_id") == metadata.get("session_id"):
                return DedupeResult(True, str(record.get("memory_id") or ""), "same_task_benchmark_result", content_hash, normalized_hash, semantic_hash)
        if metadata.get("files_changed_hash") and metadata.get("files_changed_hash") == current_files_fp:
            if current_payload.get("task_goal") == metadata.get("task_goal") or lexical_score(str(current_payload.get("summary") or ""), str(existing_payload.get("summary") or record.get("content") or "")) > 0.75:
                return DedupeResult(True, str(record.get("memory_id") or ""), "same_file_change_summary", content_hash, normalized_hash, semantic_hash)
        if metadata.get("task_goal") and metadata.get("task_goal") in content and metadata.get("session_id") and metadata.get("session_id") in content:
            return DedupeResult(True, str(record.get("memory_id") or ""), "same_task_window", content_hash, normalized_hash, semantic_hash)
        if current_payload.get("session_id") and current_payload.get("session_id") == metadata.get("session_id"):
            existing_summary = str(existing_payload.get("summary") or record.get("content") or "")
            if lexical_score(str(current_payload.get("summary") or ""), existing_summary) > 0.82:
                return DedupeResult(True, str(record.get("memory_id") or ""), "same_session_overlap", content_hash, normalized_hash, semantic_hash)
        existing = str(record.get("content") or record.get("summary") or "")
        if existing and lexical_score(content, existing) > 0.92:
            return DedupeResult(True, str(record.get("memory_id") or ""), "near_duplicate_lexical", content_hash, normalized_hash, semantic_hash)
    return DedupeResult(False, None, "new", content_hash, normalized_hash, semantic_hash)


def canonical_content(payload: dict[str, Any]) -> str:
    useful = {key: value for key, value in payload.items() if value not in (None, "", [], {})}
    return json.dumps(useful, ensure_ascii=False, sort_keys=True)


def benchmark_results_hash(payload: dict[str, Any]) -> str:
    return _list_fingerprint(payload.get("benchmark_results"))


def files_changed_hash(payload: dict[str, Any]) -> str:
    return _list_fingerprint(payload.get("files_changed"))
