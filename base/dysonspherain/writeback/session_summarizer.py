from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


EMPTY_SUMMARY = {
    "task_goal": "",
    "summary": "",
    "files_changed": [],
    "commands_run": [],
    "tests_run": [],
    "benchmark_results": [],
    "failures": [],
    "next_actions": [],
    "should_write": False,
}

COMMAND_KEYS = {"cmd", "command", "shell_command"}
FILE_KEYS = {"file", "path", "filename", "target_file"}
TEST_RE = re.compile(r"\b(pytest|unittest|npm test|ruff|mypy|benchmark smoke-all)\b", re.IGNORECASE)
BENCHMARK_RE = re.compile(
    r"\b(?:recall(?:_frac|_any|_all)?|ndcg(?:_any)?|candidate_recall|final_recall|gold_rank|oracle_recall)@?\d*\s*[=:]\s*[-+]?\d+(?:\.\d+)?",
    re.IGNORECASE,
)
FAILURE_RE = re.compile(r"\b(error|failed|failure|traceback|exception|timed out|timeout)\b", re.IGNORECASE)
NEXT_RE = re.compile(r"\b(next|todo|follow[- ]?up|remaining|下一步|待办)\b", re.IGNORECASE)
PATH_RE = re.compile(r"(?P<path>[\w./-]+\.(?:py|md|json|toml|yaml|yml|csv|txt))")


def _empty() -> dict[str, Any]:
    return {key: list(value) if isinstance(value, list) else value for key, value in EMPTY_SUMMARY.items()}


def _append_unique(target: list[str], value: Any, *, limit: int = 240) -> None:
    text = str(value or "").strip()
    if not text:
        return
    text = re.sub(r"\s+", " ", text)[:limit]
    if text not in target:
        target.append(text)


def _json_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _walk_json(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(_walk_json(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_walk_json(child))
    return values


def _extract_from_json(rows: list[dict[str, Any]], result: dict[str, Any]) -> None:
    for row in rows:
        for obj in _walk_json(row):
            if not isinstance(obj, dict):
                continue
            for key in COMMAND_KEYS:
                if key in obj:
                    _append_unique(result["commands_run"], obj.get(key))
            for key in FILE_KEYS:
                if key in obj and any(str(obj.get(key)).endswith(ext) for ext in (".py", ".md", ".json", ".toml", ".yaml", ".yml")):
                    _append_unique(result["files_changed"], obj.get(key))
            name = str(obj.get("name") or obj.get("tool") or obj.get("type") or "")
            if name in {"apply_patch", "edit", "write_file"}:
                for key in FILE_KEYS:
                    if obj.get(key):
                        _append_unique(result["files_changed"], obj.get(key))
            output = str(obj.get("output") or obj.get("stderr") or obj.get("stdout") or obj.get("text") or "")
            if output:
                if TEST_RE.search(output):
                    _append_unique(result["tests_run"], output)
                for match in BENCHMARK_RE.findall(output):
                    _append_unique(result["benchmark_results"], match)
                if FAILURE_RE.search(output):
                    _append_unique(result["failures"], output)
            if obj.get("summary") and not result["summary"]:
                result["summary"] = str(obj.get("summary"))[:1000].strip()
            if obj.get("task_goal") and not result["task_goal"]:
                result["task_goal"] = str(obj.get("task_goal"))[:240].strip()
            if obj.get("next_actions"):
                for item in obj.get("next_actions") if isinstance(obj.get("next_actions"), list) else [obj.get("next_actions")]:
                    _append_unique(result["next_actions"], item)


def _extract_from_lines(text: str, result: dict[str, Any]) -> None:
    useful_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        useful_lines.append(stripped)
        lowered = stripped.lower()
        if stripped.startswith("$ "):
            _append_unique(result["commands_run"], stripped[2:])
        elif any(token in lowered for token in ("cmd", "command")) and any(mark in stripped for mark in ("pytest", "python", "dyson", "git ", "ruff", "mypy")):
            _append_unique(result["commands_run"], stripped)
        if TEST_RE.search(stripped):
            _append_unique(result["tests_run"], stripped)
        for path_match in PATH_RE.finditer(stripped):
            if any(word in lowered for word in ("edit", "write", "patch", "modified", "file", "changed", "update", "create")):
                _append_unique(result["files_changed"], path_match.group("path"))
        for metric in BENCHMARK_RE.findall(stripped):
            _append_unique(result["benchmark_results"], metric)
        if any(word in lowered for word in ("benchmark", "metrics.json", "recall@", "ndcg", "candidate_recall", "gold_rank")):
            _append_unique(result["benchmark_results"], stripped)
        if FAILURE_RE.search(stripped):
            _append_unique(result["failures"], stripped)
        if NEXT_RE.search(stripped):
            _append_unique(result["next_actions"], stripped)
        if not result["task_goal"] and any(prefix in lowered for prefix in ("user:", "task:", "goal:", "目标")):
            result["task_goal"] = stripped[:240]
    if not result["summary"]:
        result["summary"] = "\n".join(useful_lines[-12:])[:1000].strip()


def summarize_transcript(path: str | Path | None, *, max_chars: int = 12000) -> dict[str, Any]:
    if not path:
        return _empty()
    transcript = Path(path)
    if not transcript.exists():
        result = _empty()
        result["failures"] = [f"transcript_missing:{transcript}"]
        return result
    text = transcript.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    result = _empty()
    _extract_from_json(_json_lines(text), result)
    _extract_from_lines(text, result)
    result["files_changed"] = result["files_changed"][:20]
    result["commands_run"] = result["commands_run"][:20]
    result["tests_run"] = result["tests_run"][:20]
    result["benchmark_results"] = result["benchmark_results"][:20]
    result["failures"] = result["failures"][:20]
    result["next_actions"] = result["next_actions"][:10]
    result["should_write"] = bool(
        result["commands_run"]
        or result["files_changed"]
        or result["tests_run"]
        or result["benchmark_results"]
        or result["failures"]
        or len(text.strip()) > 500
    )
    return result


def merge_hook_payload(payload: dict[str, Any]) -> dict[str, Any]:
    transcript_summary = summarize_transcript(payload.get("transcript_path"))
    merged = dict(transcript_summary)
    merged.update({key: value for key, value in payload.items() if value not in (None, "", [], {})})
    if isinstance(merged.get("summary"), (dict, list)):
        merged["summary"] = json.dumps(merged["summary"], ensure_ascii=False, sort_keys=True)
    return merged
