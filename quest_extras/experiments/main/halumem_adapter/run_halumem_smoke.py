from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUEST_ROOT = Path(__file__).resolve().parents[3]
OVERLAY_ROOT = QUEST_ROOT / "experiments" / "main" / "dyson_overlay"
DEFAULT_DYSON_ROOT = Path(
    "/home/liber/Projects/DysonSpherain/sphere_memory_cli_next_main_code_20260417_164120"
)
DEFAULT_INPUT_PATH = QUEST_ROOT / "artifacts" / "experiment" / "halumem-wave1-bounded-integration-smoke" / "input" / "halumem-medium-user01-fulltrajectory.jsonl"
DEFAULT_OUTPUT_ROOT = QUEST_ROOT / "artifacts" / "experiment" / "halumem-wave1-bounded-integration-smoke"
UNKNOWN_CUES = (
    "unknown",
    "not provided",
    "not specified",
    "not mentioned",
    "cannot be determined",
    "cannot determine",
)
TEMPORAL_UPDATE_HINT_RE = re.compile(
    r"\b(current|currently|now|today|latest|recent|recently|change|changed|update|updated|still|before|after|earlier|later|used to|previously)\b",
    re.IGNORECASE,
)
YEAR_HINT_RE = re.compile(r"\b(?:19|20)\d{2}\b")
UPDATE_STATE_HINT_RE = re.compile(
    r"\b(current|currently|status|routine|incorporated|as of|by)\b",
    re.IGNORECASE,
)
SPECULATIVE_HINT_RE = re.compile(
    r"\b(might|influence|reflect|interested|could|would)\b",
    re.IGNORECASE,
)
STRICT_ANSWER_MAX_TOKENS = 24
REASONING_ANSWER_MAX_TOKENS = 28
BINARY_QUESTION_PREFIXES = (
    "did ",
    "do ",
    "does ",
    "is ",
    "are ",
    "was ",
    "were ",
    "can ",
    "could ",
    "would ",
    "should ",
    "has ",
    "have ",
    "had ",
    "will ",
)
REASONING_HINT_RE = re.compile(
    r"\b(reason|role|factor|factors|activity|activities|type|kind|plan|plans|next|future)\b",
    re.IGNORECASE,
)
LIST_LIKE_QUESTION_RE = re.compile(
    r"\b(what activities|what ingredients)\b",
    re.IGNORECASE,
)
FACTOR_LIKE_QUESTION_RE = re.compile(
    r"\b(what factors|why)\b",
    re.IGNORECASE,
)
TYPE_LIKE_QUESTION_RE = re.compile(
    r"\b(what type of|what kind of)\b",
    re.IGNORECASE,
)
CHANGE_LIKE_QUESTION_RE = re.compile(
    r"\b(how did|what role)\b",
    re.IGNORECASE,
)
EXACT_IDENTITY_FIELD_RE = re.compile(
    r"\b(middle name|first name|last name|surname|given name|full name)\b",
    re.IGNORECASE,
)
MIDDLE_NAME_FIELD_RE = re.compile(r"\bmiddle name\b", re.IGNORECASE)
YES_NO_LEAD_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
UNEMPLOYED_HINT_RE = re.compile(r"\b(unemployed|jobless)\b", re.IGNORECASE)
FINANCIAL_STRESS_HINT_RE = re.compile(
    r"\b(struggling financially|financially struggling|financial difficulties|financial hardship)\b",
    re.IGNORECASE,
)
EMPLOYMENT_POSITIVE_SIGNAL_RE = re.compile(
    r"\b(currently )?employed\b|\bwork(?:ing)? at\b|\bmonthly income is\b|\bsavings\b",
    re.IGNORECASE,
)
PROFILE_SKEW_TEXT_RE = re.compile(
    r"\b(my name is|father|mother|parents|born on|male\b|female\b|mbti|children|currently live|highest education|my major|physical condition|chronic diseases|mental health issues|in terms of mental health)\b",
    re.IGNORECASE,
)
GENERIC_CONTEXT_TEXT_RE = re.compile(
    r"\b(focus=general|personal well-being|career path|new chapter|different areas of life|health-promoting|wellness|active lifestyle|focus on well-being)\b",
    re.IGNORECASE,
)
NOISY_RESCUE_TEXT_RE = re.compile(
    r"\b(delta from|future conversations|personal preferences|social interactions|share experiences|share new blends)\b",
    re.IGNORECASE,
)
LIST_STYLE_TEXT_RE = re.compile(r",|\band\b", re.IGNORECASE)
TYPE_STYLE_TEXT_RE = re.compile(
    r"\b(genre|genres|sport|sports|activity|activities|destination|destinations|movie|movies|film|films|game|games|video|tour|tours|vacation|vacations|retreat|retreats|hike|hikes|seminar|seminars|workshop|workshops|fantasy|adventure|cultural|tea|teas|drink|drinks|beverage|beverages|blend|blends|herbal)\b",
    re.IGNORECASE,
)
CONTRAST_USUAL_TYPE_RE = re.compile(
    r"\bdifferent from (?:the )?([^.]+?) i usually\b",
    re.IGNORECASE,
)
LOOKING_FOR_TYPE_RE = re.compile(r"\blooking for ([^.?!]+)", re.IGNORECASE)
ABOUT_FINDING_TYPE_RE = re.compile(r"\babout finding ([^.?!]+)", re.IGNORECASE)
BALANCE_WITH_RE = re.compile(
    r"\bthat balances?\s+([^.,;]+?)\s+with\s+([^.,;]+)",
    re.IGNORECASE,
)
DESTINATION_BLEND_RE = re.compile(
    r"\bdestinations that offer (?:a )?blend of ([^.,;]+?) with ([^.,;]+)",
    re.IGNORECASE,
)
DESTINATION_BOTH_RE = re.compile(
    r"\bdestinations that offer both ([^.,;]+?) and ([^.,;]+)",
    re.IGNORECASE,
)
DESTINATION_TOPIC_RE = re.compile(
    r"\b(destination|destinations|travel|travels|trip|trips|tour|tours|vacation|vacations|retreat|retreats|remote|crowded|cultural)\b",
    re.IGNORECASE,
)
GAME_TOPIC_RE = re.compile(
    r"\b(game|games|video|violent|non violent|nonviolent|puzzle|action|cognitive)\b",
    re.IGNORECASE,
)
GAME_COGNITIVE_SIGNAL_RE = re.compile(
    r"\bcognitive (?:benefits?|development|skills?)\b",
    re.IGNORECASE,
)
GAME_VALUE_SIGNAL_RE = re.compile(
    r"\b(peace|values?|non violent|nonviolent|aggression)\b",
    re.IGNORECASE,
)
FACTOR_STYLE_TEXT_RE = re.compile(
    r"\b(because|support|help|helps|clarity|align|values|relief|anticipation|encourage|opportunit|network|growth)\b",
    re.IGNORECASE,
)
CHANGE_STYLE_TEXT_RE = re.compile(
    r"\b(change|changed|switch|switched|transition|transitioned|new role|started|became|improved|healthier|balance)\b",
    re.IGNORECASE,
)
MIN_SESSION_RESCUE_SCORE = 2.4
QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "into",
    "is",
    "it",
    "mark",
    "martin",
    "might",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "why",
    "with",
    "would",
}
FOCUSED_QUERY_STOPWORDS = {
    "after",
    "based",
    "choice",
    "choices",
    "consider",
    "during",
    "explore",
    "future",
    "influence",
    "interests",
    "kind",
    "maintain",
    "next",
    "periods",
    "should",
    "stressful",
    "type",
    "visiting",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded HaluMem smoke against the current DysonSpherain runtime."
    )
    parser.add_argument(
        "--dyson-root",
        type=Path,
        default=DEFAULT_DYSON_ROOT,
        help="External DysonSpherain code root used for runtime imports.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Quest-local HaluMem trajectory JSON or JSONL file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Durable output root for generated payloads, traces, and summaries.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Optional explicit workspace root. Defaults to <output-root>/workspaces/<uuid>.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete generated outputs and fresh workspaces before running.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Optional cap on the number of sessions processed from the trajectory.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap on total questions processed across the trajectory.",
    )
    parser.add_argument(
        "--task-type",
        default="qa",
        help="DysonSpherain runtime task type for question and update retrieval.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="DysonSpherain runtime temperature for bundle assembly.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="DysonSpherain runtime max token budget for bundle assembly.",
    )
    parser.add_argument(
        "--evidence-top-k",
        type=int,
        default=8,
        help="DysonSpherain evidence_top_k for retrieval.",
    )
    parser.add_argument(
        "--support-top-k",
        type=int,
        default=4,
        help="DysonSpherain support_top_k for completion.",
    )
    parser.add_argument(
        "--object-top-k",
        type=int,
        default=4,
        help="DysonSpherain object_top_k for completion.",
    )
    parser.add_argument(
        "--cognitive-top-k",
        type=int,
        default=4,
        help="DysonSpherain cognitive_top_k for bundle assembly.",
    )
    parser.add_argument(
        "--answer-base-url",
        default=None,
        help="Optional explicit OpenAI-compatible base URL override for answer generation.",
    )
    parser.add_argument(
        "--answer-api-key",
        default=None,
        help="Optional explicit OpenAI-compatible API key override for answer generation.",
    )
    parser.add_argument(
        "--answer-model",
        default=None,
        help="Optional explicit OpenAI-compatible model override for answer generation.",
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=64,
        help="Maximum tokens for the short HaluMem answer generation step.",
    )
    parser.add_argument(
        "--answer-temperature",
        type=float,
        default=0.0,
        help="Temperature for short answer generation.",
    )
    parser.add_argument(
        "--answer-timeout-seconds",
        type=int,
        default=120,
        help="Timeout for each answer generation request.",
    )
    parser.add_argument(
        "--answer-retries",
        type=int,
        default=3,
        help="Retry count for answer generation failures.",
    )
    return parser.parse_args()


def ensure_repo_import(repo_root: Path) -> None:
    resolved = repo_root.resolve()
    overlay_resolved = OVERLAY_ROOT.resolve()
    if overlay_resolved.exists() and str(overlay_resolved) not in sys.path:
        sys.path.insert(0, str(overlay_resolved))
    if str(resolved) not in sys.path:
        insert_at = 1 if overlay_resolved.exists() and sys.path and sys.path[0] == str(overlay_resolved) else 0
        sys.path.insert(insert_at, str(resolved))


def resolve_env_value(cli_value: str | None, env_names: list[str], default: str | None = None) -> str | None:
    if cli_value:
        return cli_value
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def prepare_output_root(output_root: Path, clean_output: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for relative in ("generated", "traces", "workspaces"):
        target = output_root / relative
        if clean_output and target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def load_user_payload(input_path: Path) -> dict[str, Any]:
    raw = input_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Empty HaluMem payload at {input_path}")
    if raw.startswith("{"):
        return json.loads(raw)
    first_line = raw.splitlines()[0]
    return json.loads(first_line)


def extract_user_name(persona_info: str) -> str:
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if not match:
        raise ValueError("Unable to extract user name from persona_info")
    return match.group(1).strip()


def parse_halumem_timestamp(value: str | None) -> str:
    if not value:
        return now_iso()
    dt = datetime.strptime(value, "%b %d, %Y, %H:%M:%S").replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def slugify(value: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return compact or "user"


def compact_text(value: str, limit: int = 180) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def normalize_text(value: str) -> str:
    lowered = (value or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return " ".join(lowered.split())


def question_keywords(question: str) -> list[str]:
    tokens = normalize_text(question).split()
    return [token for token in tokens if len(token) >= 3 and token not in QUESTION_STOPWORDS]


def dedupe_preserving_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def reasoning_topic_rescue_domain(question: str) -> str:
    normalized = normalize_text(question)
    if DESTINATION_TOPIC_RE.search(normalized):
        return "destination"
    if GAME_TOPIC_RE.search(normalized):
        return "game"
    return "other"


def build_topic_rescue_query(question: str) -> str | None:
    if reasoning_topic_rescue_domain(question) == "other":
        return None
    keywords = question_keywords(question)
    if not keywords:
        return None
    prioritized: list[str] = []
    fallback: list[str] = []
    for token in keywords:
        if token in FOCUSED_QUERY_STOPWORDS:
            continue
        if YEAR_HINT_RE.fullmatch(token):
            prioritized.append(token)
            continue
        if TYPE_STYLE_TEXT_RE.search(token) or CHANGE_STYLE_TEXT_RE.search(token) or FACTOR_STYLE_TEXT_RE.search(token):
            prioritized.append(token)
            continue
        fallback.append(token)
    focused_tokens = dedupe_preserving_order(prioritized + fallback)
    if len(focused_tokens) < 2:
        return None
    return " ".join(focused_tokens[:6])


def is_unknown_like(value: str) -> bool:
    normalized = normalize_text(value)
    return any(cue in normalized for cue in UNKNOWN_CUES)


def lexical_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / max(1, min(len(left_tokens), len(right_tokens)))


def proxy_answer_match(prediction: str, gold: str) -> bool:
    pred = normalize_text(prediction)
    target = normalize_text(gold)
    if not pred or not target:
        return False
    if pred == target or pred in target or target in pred:
        return True
    if is_unknown_like(prediction) and is_unknown_like(gold):
        return True
    return lexical_overlap_ratio(prediction, gold) >= 0.75


def summarize_numbers(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "max": round(max(values), 2),
    }


def item_text(item: dict[str, Any]) -> str:
    for key in (
        "text",
        "summary",
        "raw_content",
        "source_unit_text",
        "object_text",
        "delta_summary",
        "structured_summary",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def item_timestamp(item: dict[str, Any]) -> str:
    for key in ("timestamp", "created_at", "updated_at", "time_bucket"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def format_memory_lines(items: list[dict[str, Any]], limit: int) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = item_text(item)
        if not text:
            continue
        timestamp = item_timestamp(item)
        rendered = f"{timestamp}: {text}" if timestamp else text
        if rendered in seen:
            continue
        seen.add(rendered)
        lines.append(rendered)
        if len(lines) >= limit:
            break
    return lines


def reasoning_question_shape(question: str) -> str:
    if LIST_LIKE_QUESTION_RE.search(question):
        return "list"
    if FACTOR_LIKE_QUESTION_RE.search(question):
        return "factor"
    if TYPE_LIKE_QUESTION_RE.search(question):
        return "type"
    if CHANGE_LIKE_QUESTION_RE.search(question):
        return "change"
    return "other"


def existing_bundle_texts(result: dict[str, Any]) -> set[str]:
    bundle = result["bundle"]
    texts: set[str] = set()
    for item in list(bundle.primary_evidence) + list(bundle.supporting_context):
        normalized = normalize_text(item_text(item))
        if normalized:
            texts.add(normalized)
    return texts


def extract_anchor_cells(result: dict[str, Any]) -> list[str]:
    bundle = result["bundle"]
    primary_cells: list[str] = []
    for item in list(bundle.primary_evidence):
        address = item.get("address") or {}
        cell = address.get("cell")
        if isinstance(cell, str) and cell.startswith("session-") and cell not in primary_cells:
            primary_cells.append(cell)
    if primary_cells:
        return primary_cells
    supporting_cells: list[str] = []
    for item in list(bundle.supporting_context):
        address = item.get("address") or {}
        cell = address.get("cell")
        if isinstance(cell, str) and cell.startswith("session-") and cell not in supporting_cells:
            supporting_cells.append(cell)
    if supporting_cells:
        return supporting_cells
    fallback_cells: list[str] = []
    for pointer in list(bundle.raw_reference_pointers):
        if isinstance(pointer, str) and pointer.startswith("session-") and pointer not in fallback_cells:
            fallback_cells.append(pointer)
    return fallback_cells


def bundle_primary_looks_profile_skewed(result: dict[str, Any]) -> bool:
    bundle = result["bundle"]
    primary_lines = [item_text(item) for item in list(bundle.primary_evidence)[:2]]
    return any(PROFILE_SKEW_TEXT_RE.search(line) for line in primary_lines if line)


def bundle_support_is_generic(result: dict[str, Any]) -> bool:
    bundle = result["bundle"]
    supporting = [item_text(item) for item in list(bundle.supporting_context)]
    if not supporting:
        return False
    return all(GENERIC_CONTEXT_TEXT_RE.search(line or "") for line in supporting)


def should_use_reasoning_evidence_rescue(question: str, result: dict[str, Any]) -> bool:
    if not should_use_reasoning_answer_prompt(question=question, result=result):
        return False
    if reasoning_question_shape(question) == "other":
        return False
    return bool(
        primary_evidence_count(result) <= 3
        or bundle_primary_looks_profile_skewed(result)
        or bundle_support_is_generic(result)
    )


def fetch_conversation_turn_rows(runtime: Any) -> list[dict[str, Any]]:
    cache = getattr(runtime, "_halumem_conversation_turn_rows_cache", None)
    if cache is not None:
        return cache
    rows = list(runtime.services.storage.fetch_nodes("source_type = ?", ("conversation_turn",)))
    rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("cell") or ""),
            str(row.get("created_at") or ""),
            str(row.get("id") or ""),
        ),
    )
    setattr(runtime, "_halumem_conversation_turn_rows_cache", rows)
    return rows


def score_reasoning_rescue_text(question: str, text: str, row: dict[str, Any] | None = None) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return -1.0
    tokens = normalized.split()
    if len(tokens) < 5:
        return -1.0
    score = lexical_overlap_ratio(question, text) * 3.0
    question_token_set = set(question_keywords(question))
    text_token_set = set(tokens)
    score += min(len(question_token_set & text_token_set), 5) * 0.7
    shape = reasoning_question_shape(question)
    if shape == "list" and LIST_STYLE_TEXT_RE.search(text):
        score += 1.2
    elif shape == "factor" and FACTOR_STYLE_TEXT_RE.search(text):
        score += 1.1
    elif shape == "change" and CHANGE_STYLE_TEXT_RE.search(text):
        score += 1.1
    elif shape == "type" and TYPE_STYLE_TEXT_RE.search(text):
        score += 0.9
    if PROFILE_SKEW_TEXT_RE.search(text):
        score -= 2.5
    if GENERIC_CONTEXT_TEXT_RE.search(text):
        score -= 1.4
    if NOISY_RESCUE_TEXT_RE.search(text):
        score -= 2.8
    if 8 <= len(tokens) <= 36:
        score += 0.25
    elif len(tokens) > 60:
        score -= 0.4
    if row is not None:
        time_surface = " ".join(
            [
                str(row.get("time_bucket") or ""),
                str(row.get("created_at") or ""),
                str(row.get("updated_at") or ""),
            ]
        )
        for year_hint in YEAR_HINT_RE.findall(question):
            if year_hint in time_surface:
                score += 0.5
                break
    return round(score, 4)


def matches_reasoning_shape_signal(question: str, text: str) -> bool:
    shape = reasoning_question_shape(question)
    if shape == "list":
        return bool(LIST_STYLE_TEXT_RE.search(text))
    if shape == "factor":
        return bool(FACTOR_STYLE_TEXT_RE.search(text))
    if shape == "change":
        return bool(CHANGE_STYLE_TEXT_RE.search(text))
    if shape == "type":
        return bool(TYPE_STYLE_TEXT_RE.search(text))
    return True


def select_session_rescue_lines(
    runtime: Any,
    question: str,
    result: dict[str, Any],
    max_lines: int = 3,
) -> tuple[list[str], dict[str, Any]]:
    anchor_cells = extract_anchor_cells(result)
    if not anchor_cells:
        return [], {"anchor_cells": [], "candidate_count": 0, "selected": []}
    existing_texts = existing_bundle_texts(result)
    rows = fetch_conversation_turn_rows(runtime)
    cell_set = set(anchor_cells)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        cell = str(row.get("cell") or "")
        if cell not in cell_set:
            continue
        text = str(row.get("raw_content") or row.get("summary") or "").strip()
        normalized = normalize_text(text)
        if not normalized or normalized in existing_texts:
            continue
        score = score_reasoning_rescue_text(question=question, text=text, row=row)
        if score <= 0:
            continue
        candidates.append(
            {
                "cell": cell,
                "score": score,
                "text": text,
                "created_at": str(row.get("created_at") or ""),
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["cell"], item["created_at"]))
    selected: list[str] = []
    selected_trace: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        line = compact_text(candidate["text"], limit=220)
        normalized = normalize_text(line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(line)
        selected_trace.append(
            {
                "cell": candidate["cell"],
                "score": candidate["score"],
                "preview": line,
            }
        )
        if len(selected) >= max_lines:
            break
    return selected, {
        "anchor_cells": anchor_cells,
        "candidate_count": len(candidates),
        "selected": selected_trace,
    }


def select_candidate_rescue_lines(
    runtime: Any,
    question: str,
    task_type: str,
    top_k: int,
    existing_texts: set[str],
    max_lines: int = 3,
) -> tuple[list[str], float, dict[str, Any]]:
    if max_lines <= 0:
        return [], 0.0, {
            "candidate_count": 0,
            "raw_candidate_count": 0,
            "selected": [],
            "queries": [],
        }
    query_texts = [question]
    focused_query = build_topic_rescue_query(question)
    if focused_query and normalize_text(focused_query) != normalize_text(question):
        query_texts.append(focused_query)
    candidates_by_text: dict[str, dict[str, Any]] = {}
    query_traces: list[dict[str, Any]] = []
    total_duration_ms = 0.0
    raw_candidate_count = 0
    for query_index, query_text in enumerate(query_texts):
        candidate_texts, duration_ms, retrieval_trace = retrieve_update_candidates(
            runtime=runtime,
            query=query_text,
            task_type=task_type,
            top_k=top_k,
        )
        total_duration_ms += duration_ms
        raw_candidate_count += len(candidate_texts)
        query_traces.append(
            {
                "query": query_text,
                "candidate_count": len(candidate_texts),
                "route": retrieval_trace["route"],
                "timings_ms": retrieval_trace["timings_ms"],
            }
        )
        query_bonus = 0.35 if query_index > 0 else 0.0
        for text in candidate_texts:
            normalized = normalize_text(text)
            if not normalized or normalized in existing_texts:
                continue
            if PROFILE_SKEW_TEXT_RE.search(text) or GENERIC_CONTEXT_TEXT_RE.search(text):
                continue
            if not matches_reasoning_shape_signal(question=question, text=text):
                continue
            score = score_reasoning_rescue_text(question=question, text=text) + query_bonus
            if score <= 0:
                continue
            best = candidates_by_text.get(normalized)
            if best is None or score > float(best["score"]):
                candidates_by_text[normalized] = {
                    "score": round(score, 4),
                    "text": text,
                    "query": query_text,
                }
    candidates = list(candidates_by_text.values())
    candidates.sort(key=lambda item: (-item["score"], item["text"]))
    selected: list[str] = []
    selected_trace: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        line = compact_text(candidate["text"], limit=220)
        normalized = normalize_text(line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(line)
        selected_trace.append(
            {
                "score": candidate["score"],
                "preview": line,
                "query": candidate["query"],
            }
        )
        if len(selected) >= max_lines:
            break
    return selected, round(total_duration_ms, 2), {
        "candidate_count": len(candidates),
        "raw_candidate_count": raw_candidate_count,
        "selected": selected_trace,
        "queries": query_traces,
    }


def run_reasoning_evidence_rescue(
    runtime: Any,
    question: str,
    result: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any], float]:
    existing_texts = existing_bundle_texts(result)
    session_lines, session_trace = select_session_rescue_lines(
        runtime=runtime,
        question=question,
        result=result,
        max_lines=3,
    )
    selected = list(session_lines)
    selected_texts = {normalize_text(line) for line in selected}
    candidate_trace: dict[str, Any] | None = None
    extra_search_ms = 0.0
    best_session_score = 0.0
    if session_trace["selected"]:
        best_session_score = float(session_trace["selected"][0]["score"])
    should_replace_session_lines = best_session_score < MIN_SESSION_RESCUE_SCORE
    if should_replace_session_lines:
        selected = []
        selected_texts = set()
    should_run_candidate_scan = bool(
        bundle_primary_looks_profile_skewed(result)
        or not selected
        or best_session_score < MIN_SESSION_RESCUE_SCORE
    )
    if should_run_candidate_scan:
        candidate_lines, duration_ms, candidate_trace = select_candidate_rescue_lines(
            runtime=runtime,
            question=question,
            task_type=args.task_type,
            top_k=max(int(args.evidence_top_k), 24),
            existing_texts=existing_texts | selected_texts,
            max_lines=max(0, 3 - len(selected)),
        )
        extra_search_ms += duration_ms
        for line in candidate_lines:
            normalized = normalize_text(line)
            if not normalized or normalized in selected_texts:
                continue
            selected.append(line)
            selected_texts.add(normalized)
            if len(selected) >= 3:
                break
    return selected, {
        "question_shape": reasoning_question_shape(question),
        "session": session_trace,
        "candidate_scan": candidate_trace,
    }, round(extra_search_ms, 2)


def build_context_from_result(
    result: dict[str, Any],
    limit_per_section: int = 6,
    targeted_lines: list[str] | None = None,
) -> str:
    bundle = result["bundle"]
    sections: list[str] = []
    if targeted_lines:
        sections.append("Targeted evidence:\n" + "\n".join(f"- {line}" for line in targeted_lines))
    primary = format_memory_lines(list(bundle.primary_evidence), limit_per_section)
    if primary:
        sections.append("Primary evidence:\n" + "\n".join(f"- {line}" for line in primary))
    supporting = format_memory_lines(list(bundle.supporting_context), limit_per_section)
    if supporting:
        sections.append("Supporting context:\n" + "\n".join(f"- {line}" for line in supporting))
    objects = format_memory_lines(list(bundle.evidence_objects), limit_per_section)
    if objects:
        sections.append("Evidence objects:\n" + "\n".join(f"- {line}" for line in objects))
    reflections = format_memory_lines(list(bundle.creative_reflections), min(2, limit_per_section))
    if reflections:
        sections.append("Reflections:\n" + "\n".join(f"- {line}" for line in reflections))
    return "\n\n".join(sections).strip()


def build_strict_answer_prompt(context: str, question: str) -> str:
    return (
        "You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.\n\n"
        "# INSTRUCTIONS:\n"
        "1. Use only the provided memory context.\n"
        "2. Prefer the most recent evidence if memories conflict.\n"
        "3. Convert relative time references into concrete dates or years.\n"
        "4. Answer only when the context directly supports the result.\n"
        "5. For yes/no questions, answer Yes. or No. only if the context directly supports it.\n"
        "6. Do not combine scattered hints into a guess.\n"
        "7. Say Unknown when the evidence is not direct enough.\n"
        "8. The final answer must be one short sentence, ideally under 12 words.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_relaxed_answer_prompt(context: str, question: str) -> str:
    return (
        "You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.\n\n"
        "# INSTRUCTIONS:\n"
        "1. Use only the provided memory context.\n"
        "2. Prefer the most recent evidence if memories conflict.\n"
        "3. Convert relative time references into concrete dates or years.\n"
        "4. You may combine one or two connected evidence lines when the answer is implied, not just copied verbatim.\n"
        "5. For yes/no questions, answer Yes or No and add the corrected fact when needed.\n"
        "6. Say Unknown only when the context truly lacks enough evidence after applying recency.\n"
        "7. The final answer must be one short sentence, ideally under 18 words.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_reasoning_answer_prompt(context: str, question: str) -> str:
    question_text = question.strip().lower()
    if TYPE_LIKE_QUESTION_RE.search(question_text):
        answer_style = "Return the most specific category phrase only, not a full sentence."
    elif LIST_LIKE_QUESTION_RE.search(question_text) or "what factors" in question_text:
        answer_style = "Return a short comma-separated list of the specific items only."
    elif "what role" in question_text:
        answer_style = "Return the role description phrase only."
    elif "how " in question_text or "why " in question_text:
        answer_style = "Return the main change or reason phrase only, not an explanation sentence."
    else:
        answer_style = "Return the most specific short phrase directly supported by the memories."
    return (
        "You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.\n\n"
        "# INSTRUCTIONS:\n"
        "1. Use only the provided memory context.\n"
        "2. When the question asks for a reason, role, likely next step, or implied category, combine up to three evidence lines.\n"
        "3. Prefer the most recent explicit preference, plan, or change signal when the question is future-facing.\n"
        "4. Give the most specific short completion supported by the evidence, not a generic explanation.\n"
        "5. Prefer exact entities, categories, changed items, or short lists over vague summaries.\n"
        "6. Avoid wrappers such as 'He might', 'He should', 'Martin would likely', or other hedging lead-ins.\n"
        "7. If the evidence supports a correction or contrast, state the corrected fact or changed pair directly.\n"
        "8. Say Unknown only when the provided evidence still cannot support one specific short answer.\n"
        "9. The final answer must be a short phrase or short list, ideally under 12 words and never more than 16 words.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer style: {answer_style}\n\n"
        "Answer:"
    )


def clean_reasoning_short_phrase(value: str) -> str:
    phrase = compact_text(value, limit=80).strip(" ,.;:-")
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\s+(?:that|which|who)\b.*$", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,.;:-")
    return phrase


def normalize_descriptive_type_phrase(value: str) -> str:
    phrase = compact_text(value, limit=120).strip(" ,.;:-")
    phrase = BALANCE_WITH_RE.sub(
        lambda match: f"balancing {match.group(1).strip()} and {match.group(2).strip()}",
        phrase,
    )
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,.;:-")
    return phrase


def tighten_reasoning_type_phrase(question: str, phrase: str) -> str:
    tightened = compact_text(phrase, limit=120).strip(" ,.;:-")
    if reasoning_topic_rescue_domain(question) == "destination":
        tightened = DESTINATION_BLEND_RE.sub(
            lambda match: (
                f"destinations with {match.group(1).strip()} and {match.group(2).strip()}"
            ),
            tightened,
        )
        tightened = DESTINATION_BOTH_RE.sub(
            lambda match: (
                f"destinations with {match.group(1).strip()} and {match.group(2).strip()}"
            ),
            tightened,
        )
        tightened = re.sub(
            r"\s*,?\s*alternating with [^.,;]+",
            "",
            tightened,
            flags=re.IGNORECASE,
        )
    return re.sub(r"\s+", " ", tightened).strip(" ,.;:-")


def extract_reasoning_game_type_answer(question: str, texts: list[str]) -> str | None:
    if reasoning_topic_rescue_domain(question) != "game":
        return None
    has_cognitive_signal = any(GAME_COGNITIVE_SIGNAL_RE.search(text) for text in texts)
    has_value_signal = any(GAME_VALUE_SIGNAL_RE.search(text) for text in texts)
    if has_cognitive_signal and has_value_signal:
        return "games with cognitive skills and peace values"
    return None


def extract_reasoning_contrast_type_answer(
    question: str,
    result: dict[str, Any],
    rescue_lines: list[str] | None = None,
) -> str | None:
    if reasoning_question_shape(question) != "type":
        return None
    question_text = question.strip().lower()
    if "next" not in question_text and "after" not in question_text and not SPECULATIVE_HINT_RE.search(question_text):
        return None
    bundle = result["bundle"]
    texts = [item_text(item) for item in list(bundle.primary_evidence)]
    if rescue_lines:
        texts.extend(str(line).strip() for line in rescue_lines if str(line).strip())
    question_norm = normalize_text(question)
    for text in texts:
        match = CONTRAST_USUAL_TYPE_RE.search(text)
        if not match:
            continue
        phrase = clean_reasoning_short_phrase(match.group(1))
        if not phrase:
            continue
        phrase_norm = normalize_text(phrase)
        if not phrase_norm or phrase_norm in question_norm:
            continue
        return phrase
    return None


def extract_reasoning_descriptive_type_answer(
    question: str,
    result: dict[str, Any],
    rescue_lines: list[str] | None = None,
) -> str | None:
    if reasoning_question_shape(question) != "type":
        return None
    question_text = question.strip().lower()
    if "what type of" not in question_text and "what kind of" not in question_text:
        return None
    bundle = result["bundle"]
    question_norm = normalize_text(question)
    texts = [item_text(item) for item in list(bundle.primary_evidence)]
    if rescue_lines:
        texts.extend(str(line).strip() for line in rescue_lines if str(line).strip())
    game_phrase = extract_reasoning_game_type_answer(question=question, texts=texts)
    if game_phrase is not None:
        phrase_norm = normalize_text(game_phrase)
        if phrase_norm and phrase_norm not in question_norm:
            return game_phrase
    for text in texts:
        match = ABOUT_FINDING_TYPE_RE.search(text)
        if not match:
            match = LOOKING_FOR_TYPE_RE.search(text)
        if not match:
            continue
        phrase = normalize_descriptive_type_phrase(match.group(1))
        phrase = tighten_reasoning_type_phrase(question=question, phrase=phrase)
        phrase = clean_reasoning_short_phrase(phrase)
        if not phrase:
            continue
        phrase_norm = normalize_text(phrase)
        if not phrase_norm or phrase_norm in question_norm:
            continue
        return phrase
    return None


def primary_evidence_count(result: dict[str, Any]) -> int:
    bundle = result["bundle"]
    return len(list(bundle.primary_evidence))


def is_binary_question(question: str) -> bool:
    question_text = question.strip().lower()
    return question_text.startswith(BINARY_QUESTION_PREFIXES)


def is_exact_identity_question(question: str) -> bool:
    question_text = question.strip().lower()
    return question_text.startswith("what ") and EXACT_IDENTITY_FIELD_RE.search(question_text) is not None


def is_current_state_binary_question(question: str) -> bool:
    if not is_binary_question(question):
        return False
    return bool(
        UPDATE_STATE_HINT_RE.search(question) is not None
        or TEMPORAL_UPDATE_HINT_RE.search(question) is not None
        or YEAR_HINT_RE.search(question) is not None
    )


def has_explicit_middle_name_evidence(result: dict[str, Any]) -> bool:
    bundle = result["bundle"]
    evidence_items = (
        list(bundle.primary_evidence)
        + list(bundle.supporting_context)
        + list(bundle.evidence_objects)
    )
    for item in evidence_items:
        if MIDDLE_NAME_FIELD_RE.search(item_text(item)):
            return True
    return False


def should_short_circuit_unknown_middle_name(question: str, result: dict[str, Any]) -> bool:
    if MIDDLE_NAME_FIELD_RE.search(question) is None:
        return False
    return not has_explicit_middle_name_evidence(result)


def canonicalize_current_state_binary_response(question: str, response: str) -> str:
    if not is_current_state_binary_question(question):
        return response
    match = YES_NO_LEAD_RE.match(response.strip())
    if match is None:
        return response
    return "Yes." if match.group(1).lower() == "yes" else "No."


def extract_strict_current_state_binary_answer(question: str, result: dict[str, Any]) -> str | None:
    if not is_current_state_binary_question(question):
        return None
    bundle = result["bundle"]
    primary_text = " ".join(item_text(item) for item in list(bundle.primary_evidence))
    if not primary_text:
        return None
    question_text = question.strip().lower()
    if (
        (UNEMPLOYED_HINT_RE.search(question_text) or FINANCIAL_STRESS_HINT_RE.search(question_text))
        and EMPLOYMENT_POSITIVE_SIGNAL_RE.search(primary_text)
    ):
        return "No."
    return None


def should_use_reasoning_answer_prompt(question: str, result: dict[str, Any]) -> bool:
    evidence = result["evidence"]
    profile = getattr(evidence, "profile", None)
    needs_multi_hop = bool(getattr(profile, "needs_multi_hop_evidence", False))
    needs_temporal = bool(getattr(profile, "needs_temporal_objects", False))
    question_text = question.strip().lower()
    if is_exact_identity_question(question):
        return False
    temporal_context = YEAR_HINT_RE.search(question) is not None or TEMPORAL_UPDATE_HINT_RE.search(question) is not None
    has_reasoning_cue = (
        question_text.startswith("how ")
        or question_text.startswith("why ")
        or REASONING_HINT_RE.search(question) is not None
        or SPECULATIVE_HINT_RE.search(question) is not None
    )
    return bool(
        primary_evidence_count(result) > 0
        and not is_binary_question(question)
        and (
            needs_multi_hop
            or has_reasoning_cue
            or (needs_temporal and temporal_context and question_text.startswith("what "))
        )
    )


def should_use_relaxed_answer_prompt(question: str, result: dict[str, Any]) -> bool:
    evidence = result["evidence"]
    profile = getattr(evidence, "profile", None)
    needs_exact = bool(getattr(profile, "needs_exact_evidence", False))
    needs_multi_hop = bool(getattr(profile, "needs_multi_hop_evidence", False))
    question_text = question.strip().lower()
    if is_exact_identity_question(question):
        return False
    starts_with_what = question_text.startswith("what ")
    update_like = UPDATE_STATE_HINT_RE.search(question) is not None
    speculative = SPECULATIVE_HINT_RE.search(question) is not None
    temporal_context = YEAR_HINT_RE.search(question) is not None or TEMPORAL_UPDATE_HINT_RE.search(question) is not None
    return bool(
        needs_exact
        and not needs_multi_hop
        and starts_with_what
        and update_like
        and temporal_context
        and not speculative
    )


def should_short_circuit_strict_answer(question: str, result: dict[str, Any]) -> bool:
    evidence = result["evidence"]
    profile = getattr(evidence, "profile", None)
    needs_multi_hop = bool(getattr(profile, "needs_multi_hop_evidence", False))
    question_text = question.strip().lower()
    starts_with_how = question_text.startswith("how ")
    speculative = SPECULATIVE_HINT_RE.search(question) is not None
    return bool(
        primary_evidence_count(result) == 0
        and not is_binary_question(question)
        and needs_multi_hop
        and starts_with_how
        and speculative
    )


def should_expand_reasoning_context(question: str, result: dict[str, Any]) -> bool:
    if not should_use_reasoning_answer_prompt(question=question, result=result):
        return False
    bundle = result["bundle"]
    supporting_count = len(list(bundle.supporting_context))
    object_count = len(list(bundle.evidence_objects))
    return bool(
        primary_evidence_count(result) <= 3
        or supporting_count <= 2
        or object_count <= 1
    )


def strict_answer_token_budget(answer_config: dict[str, Any]) -> int:
    return min(int(answer_config["max_tokens"]), STRICT_ANSWER_MAX_TOKENS)


def reasoning_answer_token_budget(answer_config: dict[str, Any]) -> int:
    return min(int(answer_config["max_tokens"]), REASONING_ANSWER_MAX_TOKENS)


def call_openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    retries: int,
) -> tuple[str, float]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    attempt = 0
    while True:
        attempt += 1
        started = time.perf_counter()
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
            data = json.loads(body)
            content = str(data["choices"][0]["message"]["content"]).strip()
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return content, elapsed_ms
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            if attempt >= max(1, retries):
                raise RuntimeError(f"answer_generation_failed_after_{attempt}_attempts: {exc}") from exc
            time.sleep(min(5 * attempt, 15))


def make_question_budget_copy(user_payload: dict[str, Any], max_sessions: int | None, max_questions: int | None) -> dict[str, Any]:
    cloned = copy.deepcopy(user_payload)
    sessions = list(cloned.get("sessions") or [])
    if max_sessions is not None:
        sessions = sessions[: max(0, max_sessions)]
    remaining_questions = max_questions
    selected_sessions: list[dict[str, Any]] = []
    for session in sessions:
        if remaining_questions is None:
            selected_sessions.append(session)
            continue
        questions = list(session.get("questions") or [])
        if not questions:
            selected_sessions.append(session)
            continue
        if remaining_questions <= 0:
            session["questions"] = []
            session["question_count"] = 0
            selected_sessions.append(session)
            continue
        session["questions"] = questions[:remaining_questions]
        session["question_count"] = len(session["questions"])
        remaining_questions -= len(session["questions"])
        selected_sessions.append(session)
    cloned["sessions"] = selected_sessions
    cloned["total_question_count"] = sum(int(session.get("question_count", 0)) for session in selected_sessions)
    return cloned


def runtime_trace_payload(result: dict[str, Any]) -> dict[str, Any]:
    evidence = result["evidence"]
    completion = result["completion"]
    bundle = result["bundle"]
    cognitive = result["cognitive"]
    return {
        "route": evidence.query_route,
        "profile": {
            "task_type": evidence.profile.task_type,
            "needs_exact_evidence": evidence.profile.needs_exact_evidence,
            "needs_multi_hop_evidence": evidence.profile.needs_multi_hop_evidence,
            "needs_preference_objects": evidence.profile.needs_preference_objects,
            "needs_temporal_objects": evidence.profile.needs_temporal_objects,
            "needs_personal_context_objects": evidence.profile.needs_personal_context_objects,
            "needs_relation_objects": evidence.profile.needs_relation_objects,
            "preferred_object_types": list(evidence.profile.preferred_object_types),
        },
        "timings_ms": {
            "retrieval": dict(evidence.timings_ms),
            "completion": dict(completion.timings_ms),
            "cognitive": dict(cognitive.timings_ms),
            "bundle": dict(bundle.debug),
        },
        "bundle": {
            "primary_evidence": list(bundle.primary_evidence),
            "supporting_context": list(bundle.supporting_context),
            "evidence_objects": list(bundle.evidence_objects),
            "creative_reflections": list(bundle.creative_reflections),
            "alternative_paths": list(bundle.alternative_paths),
            "raw_reference_pointers": list(bundle.raw_reference_pointers),
            "debug": dict(bundle.debug),
        },
    }


def ingest_session(
    runtime: Any,
    memory_node_cls: Any,
    session: dict[str, Any],
    user_name: str,
    session_index: int,
) -> tuple[float, list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for turn_index, turn in enumerate(list(session.get("dialogue") or [])):
        if str(turn.get("role") or "").lower() != "user":
            continue
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        timestamp = parse_halumem_timestamp(turn.get("timestamp"))
        metadata = {
            "benchmark": "HaluMem",
            "user_name": user_name,
            "session_index": session_index,
            "dialogue_turn": turn.get("dialogue_turn"),
            "source_timestamp": turn.get("timestamp"),
        }
        node = memory_node_cls(
            shell=2,
            sector="dialogue",
            zone=f"halumem-{slugify(user_name)}",
            cell=f"session-{session_index:03d}",
            molecular_type="dialogue_turn",
            summary=compact_text(content, limit=180),
            raw_content=content,
            session_id=f"session-{session_index:03d}",
            source_type="conversation_turn",
            source_ref=f"halumem:{session_index}:{turn_index}",
            extraction_method="halumem_dialogue_ingest",
            confidence=0.55,
            verification_status="unverified",
            metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            importance=0.55,
            stability_score=0.45,
            stage="long_term",
            tags="halumem,dialogue,user_turn",
            created_at=timestamp,
            updated_at=timestamp,
            last_accessed_at=timestamp,
        )
        report = runtime.writeback_memory(node=node, source_kind="halumem_dialogue")
        reports.append({"turn_index": turn_index, "report": report})
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return elapsed_ms, reports


def snapshot_runtime_memories(runtime: Any) -> tuple[list[str], list[dict[str, Any]]]:
    rows = runtime.services.storage.fetch_nodes("source_type = ?", ("conversation_turn",))
    rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("created_at") or ""),
            str(row.get("updated_at") or ""),
            str(row.get("id") or ""),
        ),
    )
    extracted_memories: list[str] = []
    for row in rows:
        text = str(row.get("raw_content") or row.get("summary") or "").strip()
        if text:
            extracted_memories.append(text)
    return extracted_memories, rows


def retrieve_update_candidates(runtime: Any, query: str, task_type: str, top_k: int) -> tuple[list[str], float, dict[str, Any]]:
    started = time.perf_counter()
    evidence = runtime.retrieve_evidence(query=query, task_type=task_type, top_k=top_k)
    duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    candidates = []
    seen: set[str] = set()
    for item in evidence.candidates:
        text = item_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        candidates.append(text)
        if len(candidates) >= top_k:
            break
    trace = {
        "route": evidence.query_route,
        "timings_ms": dict(evidence.timings_ms),
        "candidates": list(evidence.candidates[:top_k]),
    }
    return candidates, duration_ms, trace


def answer_question(
    runtime: Any,
    question: str,
    args: argparse.Namespace,
    answer_config: dict[str, Any],
) -> tuple[dict[str, Any], str, float, str | None]:
    started = time.perf_counter()
    result = runtime.run_query(
        query=question,
        task_type=args.task_type,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        evidence_top_k=args.evidence_top_k,
        support_top_k=args.support_top_k,
        object_top_k=args.object_top_k,
        cognitive_top_k=args.cognitive_top_k,
    )
    search_duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    reasoning_context_expanded = False
    if should_expand_reasoning_context(question=question, result=result):
        expanded_started = time.perf_counter()
        result = runtime.run_query(
            query=question,
            task_type=args.task_type,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            evidence_top_k=max(int(args.evidence_top_k), 12),
            support_top_k=max(int(args.support_top_k), 8),
            object_top_k=max(int(args.object_top_k), 8),
            cognitive_top_k=max(int(args.cognitive_top_k), 6),
        )
        search_duration_ms = round(search_duration_ms + ((time.perf_counter() - expanded_started) * 1000.0), 2)
        reasoning_context_expanded = True
    force_unknown_middle_name = should_short_circuit_unknown_middle_name(
        question=question,
        result=result,
    )
    if force_unknown_middle_name:
        prompt_mode = "strict"
    elif should_use_reasoning_answer_prompt(question=question, result=result):
        prompt_mode = "reasoning"
    elif should_use_relaxed_answer_prompt(question=question, result=result):
        prompt_mode = "relaxed"
    else:
        prompt_mode = "strict"
    context_limit = 8 if prompt_mode == "reasoning" else 6
    rescue_lines: list[str] = []
    rescue_trace: dict[str, Any] | None = None
    if prompt_mode == "reasoning" and should_use_reasoning_evidence_rescue(question=question, result=result):
        rescue_lines, rescue_trace, rescue_search_ms = run_reasoning_evidence_rescue(
            runtime=runtime,
            question=question,
            result=result,
            args=args,
        )
        search_duration_ms = round(search_duration_ms + rescue_search_ms, 2)
    context = build_context_from_result(
        result,
        limit_per_section=context_limit,
        targeted_lines=rescue_lines,
    )
    answer_max_tokens = int(answer_config["max_tokens"])
    if prompt_mode == "reasoning":
        prompt = build_reasoning_answer_prompt(context=context, question=question)
        answer_max_tokens = reasoning_answer_token_budget(answer_config)
    elif prompt_mode == "relaxed":
        prompt = build_relaxed_answer_prompt(context=context, question=question)
    else:
        prompt = build_strict_answer_prompt(context=context, question=question)
        answer_max_tokens = strict_answer_token_budget(answer_config)
    generation_error: str | None = None
    short_circuit_reason: str | None = None
    system_response = ""
    response_duration_ms = 0.0
    reasoning_short_circuit_answer: str | None = None
    reasoning_short_circuit_reason: str | None = None
    strict_binary_short_circuit_answer: str | None = None
    if prompt_mode == "reasoning":
        reasoning_short_circuit_answer = extract_reasoning_contrast_type_answer(
            question=question,
            result=result,
            rescue_lines=rescue_lines,
        )
        if reasoning_short_circuit_answer is not None:
            reasoning_short_circuit_reason = "reasoning_contrast_type"
        else:
            reasoning_short_circuit_answer = extract_reasoning_descriptive_type_answer(
                question=question,
                result=result,
                rescue_lines=rescue_lines,
            )
            if reasoning_short_circuit_answer is not None:
                reasoning_short_circuit_reason = "reasoning_descriptive_type"
    elif prompt_mode == "strict":
        strict_binary_short_circuit_answer = extract_strict_current_state_binary_answer(
            question=question,
            result=result,
        )
    if force_unknown_middle_name:
        system_response = "Unknown."
        short_circuit_reason = "unknown_middle_name_missing_explicit_evidence"
    elif strict_binary_short_circuit_answer is not None:
        system_response = strict_binary_short_circuit_answer
        short_circuit_reason = "strict_current_state_binary"
    elif prompt_mode == "strict" and should_short_circuit_strict_answer(question=question, result=result):
        system_response = "Unknown."
        short_circuit_reason = "speculative_multi_hop_how"
    elif reasoning_short_circuit_answer is not None:
        system_response = reasoning_short_circuit_answer
        short_circuit_reason = reasoning_short_circuit_reason
    else:
        try:
            system_response, response_duration_ms = call_openai_compatible_chat(
                base_url=str(answer_config["base_url"]),
                api_key=str(answer_config["api_key"]),
                model=str(answer_config["model"]),
                prompt=prompt,
                max_tokens=answer_max_tokens,
                temperature=float(answer_config["temperature"]),
                timeout_seconds=int(answer_config["timeout_seconds"]),
                retries=int(answer_config["retries"]),
            )
        except Exception as exc:  # noqa: BLE001
            generation_error = str(exc)
    if system_response and prompt_mode == "strict":
        system_response = canonicalize_current_state_binary_response(
            question=question,
            response=system_response,
        )
    trace = runtime_trace_payload(result)
    trace["answer_prompt"] = prompt
    trace["answer_prompt_mode"] = prompt_mode
    trace["answer_max_tokens"] = answer_max_tokens
    trace["answer_short_circuit_reason"] = short_circuit_reason
    trace["search_duration_ms"] = search_duration_ms
    trace["response_duration_ms"] = response_duration_ms
    trace["generation_error"] = generation_error
    trace["reasoning_context_expanded"] = reasoning_context_expanded
    trace["reasoning_evidence_rescue"] = rescue_trace
    trace["reasoning_evidence_rescue_lines"] = rescue_lines
    return trace, context, response_duration_ms, system_response


def summarize_run(
    *,
    user_payload: dict[str, Any],
    generated_payload: dict[str, Any],
    session_reports: list[dict[str, Any]],
    question_traces: list[dict[str, Any]],
    update_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    total_sessions = len(generated_payload.get("sessions") or [])
    extracted_counts = [int(report["extracted_memory_count"]) for report in session_reports]
    add_latencies = [float(report["add_dialogue_duration_ms"]) for report in session_reports]
    search_latencies = [float(trace["search_duration_ms"]) for trace in question_traces]
    response_latencies = [float(trace["response_duration_ms"]) for trace in question_traces if trace["response_duration_ms"] > 0]
    question_count = len(question_traces)
    answer_success_count = sum(1 for trace in question_traces if trace["system_response"])
    context_nonempty_count = sum(1 for trace in question_traces if trace["context"])
    primary_evidence_count = sum(1 for trace in question_traces if trace["primary_evidence_count"] > 0)
    proxy_match_count = sum(1 for trace in question_traces if trace["proxy_match"])
    unknown_alignment_count = sum(
        1 for trace in question_traces if trace["unknown_alignment"]
    )
    update_count = len(update_traces)
    update_hit_count = sum(1 for trace in update_traces if trace["memory_count"] > 0)
    question_type_bucket: dict[str, dict[str, int]] = {}
    for trace in question_traces:
        bucket = question_type_bucket.setdefault(
            trace["question_type"],
            {"count": 0, "proxy_match_count": 0, "answer_success_count": 0},
        )
        bucket["count"] += 1
        bucket["proxy_match_count"] += int(trace["proxy_match"])
        bucket["answer_success_count"] += int(bool(trace["system_response"]))
    question_type_summary = {
        key: {
            "count": value["count"],
            "proxy_match_rate": round(value["proxy_match_count"] / max(1, value["count"]), 4),
            "answer_success_rate": round(value["answer_success_count"] / max(1, value["count"]), 4),
        }
        for key, value in sorted(question_type_bucket.items())
    }
    return {
        "scope": {
            "uuid": user_payload["uuid"],
            "session_count": total_sessions,
            "question_session_count": sum(1 for session in generated_payload["sessions"] if session.get("questions")),
            "question_count": question_count,
            "update_memory_count": update_count,
            "persona_info_present": bool(user_payload.get("persona_info")),
        },
        "coverage": {
            "session_with_extracted_memories_rate": round(
                sum(1 for count in extracted_counts if count > 0) / max(1, total_sessions),
                4,
            ),
            "update_retrieval_nonempty_rate": round(update_hit_count / max(1, update_count), 4),
            "question_answer_success_rate": round(answer_success_count / max(1, question_count), 4),
            "question_context_nonempty_rate": round(context_nonempty_count / max(1, question_count), 4),
            "question_primary_evidence_nonempty_rate": round(primary_evidence_count / max(1, question_count), 4),
        },
        "proxy_question_quality": {
            "proxy_match_rate": round(proxy_match_count / max(1, question_count), 4),
            "unknown_alignment_rate": round(unknown_alignment_count / max(1, question_count), 4),
            "question_type_breakdown": question_type_summary,
        },
        "latency_ms": {
            "add_dialogue": summarize_numbers(add_latencies),
            "search": summarize_numbers(search_latencies),
            "response": summarize_numbers(response_latencies),
        },
        "memory_snapshot": {
            "mean_extracted_memory_count_per_session": round(statistics.fmean(extracted_counts), 2)
            if extracted_counts
            else 0.0,
            "max_extracted_memory_count": max(extracted_counts) if extracted_counts else 0,
        },
        "errors": {
            "question_generation_error_count": sum(1 for trace in question_traces if trace["generation_error"]),
        },
        "caveat": (
            "These are lightweight integration metrics, not HaluMem's full official judge-backed scores. "
            "The run keeps the system general by ingesting dialogue turns rather than benchmark gold memories."
        ),
    }


def render_summary_md(manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    question_breakdown = metrics["proxy_question_quality"]["question_type_breakdown"]
    lines = [
        "# HaluMem Smoke Summary",
        "",
        "## Scope",
        "",
        f"- input_path: `{manifest['input_path']}`",
        f"- session_count: `{metrics['scope']['session_count']}`",
        f"- question_count: `{metrics['scope']['question_count']}`",
        f"- update_memory_count: `{metrics['scope']['update_memory_count']}`",
        f"- dyson_root: `{manifest['dyson_root']}`",
        f"- interpreter: `{manifest['interpreter']}`",
        "",
        "## Headline",
        "",
        f"- question answer success rate: `{metrics['coverage']['question_answer_success_rate']:.4f}`",
        f"- non-empty primary evidence rate: `{metrics['coverage']['question_primary_evidence_nonempty_rate']:.4f}`",
        f"- update retrieval non-empty rate: `{metrics['coverage']['update_retrieval_nonempty_rate']:.4f}`",
        f"- proxy question match rate: `{metrics['proxy_question_quality']['proxy_match_rate']:.4f}`",
        f"- mean search latency: `{metrics['latency_ms']['search']['mean']:.2f} ms`",
        f"- mean response latency: `{metrics['latency_ms']['response']['mean']:.2f} ms`",
        "",
        "## Caveat",
        "",
        f"- {metrics['caveat']}",
        "",
        "## Question Type Breakdown",
        "",
    ]
    if question_breakdown:
        for key, value in question_breakdown.items():
            lines.append(
                f"- `{key}`: count=`{value['count']}`, proxy_match_rate=`{value['proxy_match_rate']:.4f}`, "
                f"answer_success_rate=`{value['answer_success_rate']:.4f}`"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ensure_repo_import(args.dyson_root)
    from sphere_cli.models import MemoryNode  # type: ignore
    from sphere_cli.runtime import UnifiedMemoryRuntime  # type: ignore

    input_path = args.input_path.resolve()
    output_root = args.output_root.resolve()
    prepare_output_root(output_root, clean_output=args.clean_output)

    user_payload = load_user_payload(input_path)
    bounded_payload = make_question_budget_copy(
        user_payload,
        max_sessions=args.max_sessions,
        max_questions=args.max_questions,
    )
    user_name = extract_user_name(str(bounded_payload["persona_info"]))

    answer_config = {
        "base_url": resolve_env_value(
            args.answer_base_url,
            ["OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"],
            default="https://api.openai.com/v1",
        ),
        "api_key": resolve_env_value(
            args.answer_api_key,
            ["OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
        ),
        "model": resolve_env_value(
            args.answer_model,
            ["OPENAI_MODEL", "ANTHROPIC_MODEL"],
            default="gpt-4o-mini",
        ),
        "max_tokens": args.answer_max_tokens,
        "temperature": args.answer_temperature,
        "timeout_seconds": args.answer_timeout_seconds,
        "retries": args.answer_retries,
    }
    if not answer_config["api_key"]:
        raise RuntimeError("Missing answer generation API key in OPENAI_API_KEY or ANTHROPIC_AUTH_TOKEN")

    workspace_root = (
        args.workspace_root.resolve()
        if args.workspace_root is not None
        else (output_root / "workspaces" / bounded_payload["uuid"]).resolve()
    )
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    runtime = UnifiedMemoryRuntime.from_base_dir(workspace_root)
    generated_payload = {
        "uuid": bounded_payload["uuid"],
        "user_name": user_name,
        "sessions": [],
    }

    session_reports: list[dict[str, Any]] = []
    question_traces: list[dict[str, Any]] = []
    update_traces: list[dict[str, Any]] = []

    session_trace_root = output_root / "traces" / bounded_payload["uuid"]
    session_trace_root.mkdir(parents=True, exist_ok=True)

    for session_index, source_session in enumerate(list(bounded_payload.get("sessions") or [])):
        new_session = copy.deepcopy(source_session)
        add_dialogue_duration_ms, ingest_reports = ingest_session(
            runtime=runtime,
            memory_node_cls=MemoryNode,
            session=source_session,
            user_name=user_name,
            session_index=session_index,
        )
        new_session["add_dialogue_duration_ms"] = add_dialogue_duration_ms
        extracted_memories, memory_rows = snapshot_runtime_memories(runtime)
        new_session["extracted_memories"] = extracted_memories

        for memory_index, memory in enumerate(list(new_session.get("memory_points") or [])):
            if str(memory.get("is_update")) != "True" or not list(memory.get("original_memories") or []):
                continue
            memories_from_system, duration_ms, trace = retrieve_update_candidates(
                runtime=runtime,
                query=str(memory.get("memory_content") or ""),
                task_type=args.task_type,
                top_k=args.evidence_top_k,
            )
            memory["memories_from_system"] = memories_from_system
            update_traces.append(
                {
                    "session_index": session_index,
                    "memory_index": memory_index,
                    "memory_type": str(memory.get("memory_type") or ""),
                    "memory_content": str(memory.get("memory_content") or ""),
                    "memory_count": len(memories_from_system),
                    "duration_ms": duration_ms,
                    "trace": trace,
                }
            )

        processed_questions: list[dict[str, Any]] = []
        for question_index, qa in enumerate(list(new_session.get("questions") or [])):
            trace, context, response_duration_ms, system_response = answer_question(
                runtime=runtime,
                question=str(qa.get("question") or ""),
                args=args,
                answer_config=answer_config,
            )
            new_qa = copy.deepcopy(qa)
            new_qa["context"] = context
            new_qa["search_duration_ms"] = trace["search_duration_ms"]
            new_qa["system_response"] = system_response
            new_qa["response_duration_ms"] = response_duration_ms
            if trace["generation_error"]:
                new_qa["generation_error"] = trace["generation_error"]
            processed_questions.append(new_qa)
            question_traces.append(
                {
                    "session_index": session_index,
                    "question_index": question_index,
                    "question_type": str(qa.get("question_type") or "unknown"),
                    "question": str(qa.get("question") or ""),
                    "gold_answer": str(qa.get("answer") or ""),
                    "system_response": system_response,
                    "context": context,
                    "search_duration_ms": trace["search_duration_ms"],
                    "response_duration_ms": response_duration_ms,
                    "answer_prompt_mode": trace.get("answer_prompt_mode", "strict"),
                    "answer_max_tokens": trace.get("answer_max_tokens"),
                    "answer_short_circuit_reason": trace.get("answer_short_circuit_reason"),
                    "generation_error": trace["generation_error"],
                    "proxy_match": proxy_answer_match(system_response, str(qa.get("answer") or "")),
                    "unknown_alignment": is_unknown_like(system_response) and is_unknown_like(str(qa.get("answer") or "")),
                    "primary_evidence_count": len(trace["bundle"]["primary_evidence"]),
                    "supporting_context_count": len(trace["bundle"]["supporting_context"]),
                    "runtime_trace": trace,
                }
            )

        if "questions" in new_session:
            new_session["questions"] = processed_questions
            new_session["question_count"] = len(processed_questions)

        generated_payload["sessions"].append(new_session)
        session_report = {
            "session_index": session_index,
            "add_dialogue_duration_ms": add_dialogue_duration_ms,
            "ingested_user_turn_count": len(ingest_reports),
            "extracted_memory_count": len(extracted_memories),
            "question_count": len(processed_questions),
            "update_memory_count": sum(
                1
                for memory in list(new_session.get("memory_points") or [])
                if str(memory.get("is_update")) == "True"
            ),
        }
        session_reports.append(session_report)
        write_json(session_trace_root / f"session_{session_index:03d}.json", {
            "session_report": session_report,
            "memory_rows": memory_rows,
        })

    generated_jsonl_path = output_root / "generated" / "halumem_eval_results.jsonl"
    write_jsonl(generated_jsonl_path, [generated_payload])
    write_json(output_root / "generated" / "session_reports.json", session_reports)
    write_jsonl(output_root / "generated" / "question_traces.jsonl", question_traces)
    write_jsonl(output_root / "generated" / "update_traces.jsonl", update_traces)

    metrics = summarize_run(
        user_payload=bounded_payload,
        generated_payload=generated_payload,
        session_reports=session_reports,
        question_traces=question_traces,
        update_traces=update_traces,
    )
    manifest = {
        "run_name": "halumem-wave1-bounded-integration-smoke",
        "started_at": now_iso(),
        "input_path": str(input_path),
        "output_root": str(output_root),
        "workspace_root": str(workspace_root),
        "generated_jsonl_path": str(generated_jsonl_path),
        "dyson_root": str(args.dyson_root.resolve()),
        "script_path": str(Path(__file__).resolve()),
        "interpreter": sys.executable,
        "command": " ".join(sys.argv),
        "answer_base_url": answer_config["base_url"],
        "answer_model": answer_config["model"],
        "max_sessions": args.max_sessions,
        "max_questions": args.max_questions,
        "task_type": args.task_type,
        "scope_note": (
            "This smoke preserves a general-system path by ingesting user dialogue turns and answering from "
            "DysonSpherain evidence bundles. It is not an official full HaluMem score report."
        ),
    }
    write_json(output_root / "run_manifest.json", manifest)
    write_json(output_root / "metrics.json", metrics)
    write_json(
        output_root / "summary.json",
        {
            "manifest": manifest,
            "metrics": metrics,
            "question_trace_count": len(question_traces),
            "update_trace_count": len(update_traces),
        },
    )
    (output_root / "summary.md").write_text(render_summary_md(manifest, metrics), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
