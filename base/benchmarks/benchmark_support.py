from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from sphere_cli.config import AppConfig
from sphere_cli.embedding import EMBEDDING_PREPROCESS_VERSION
from sphere_cli.storage import Storage
from sphere_cli.utils import lexical_score, normalize_text_for_hash, stable_content_hash, tokenize
from sphere_cli.vector_store import VectorStore

DEFAULT_BENCHMARK_SEED = 7
BENCHMARK_CHUNKER_VERSION = "benchmark_chunker_v2"
MULTI_CHANNEL_RETRIEVAL_SCHEMA_VERSION = "multichannel_candidate_v2"
QUERY_DECOMPOSITION_SCHEMA_VERSION = "query_decomposition_v1"
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_TOKEN_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|spring|summer|autumn|fall|winter)\b",
    re.IGNORECASE,
)
TEMPORAL_TOKEN_RE = re.compile(
    r"\b(?:before|after|earlier|later|latest|current|currently|yesterday|today|tomorrow|week|weeks|month|months|year|years|day|days|then|next|previous|first|last)\b",
    re.IGNORECASE,
)
RELATIVE_TIME_PHRASE_RE = re.compile(
    r"\b(?:over the past|past|last|next|previous)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple of|few)\s+(?:day|days|week|weeks|month|months|year|years)\b"
    r"|\b(?:right now|that night|that day|that evening|that morning|beginning of the year)\b"
    r"|\b(?:mid|early|late)[-\s](?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
QUERY_LEAD_IN_TERMS = {
    "brother",
    "sister",
    "noticed",
    "looking",
    "back",
    "previous",
    "conversation",
    "remind",
    "remember",
    "specific",
    "please",
    "could",
    "would",
    "ive",
    "youve",
    "real",
    "really",
    "past",
    "lately",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CJK_SPAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")
DATE_HINT_RE = re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{2})?(?:[-/]\d{2})?\b")
TIME_HINT_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
CODE_LIKE_RE = re.compile(r"\b[\w.-]+(?:[/_.:-][\w.-]+)+\b")
METRIC_LIKE_RE = re.compile(r"\b[a-zA-Z_]*@?\d+(?:\.\d+)?%?\b|\b(?:recall|ndcg|precision|accuracy|latency|score|metric)s?@?\d*\b", re.IGNORECASE)
PERSON_STATE_PHRASE_RE = re.compile(
    r"\b(?:prefers?|favorites?|likes?|loves?|hates?|avoids?|wants?|needs?|plans?|decided|changed|moved|visited|worked|learned|helped|felt|worried|excited|angry|sad|happy)\b(?:\s+\w+){0,5}",
    re.IGNORECASE,
)
QUERY_CLAUSE_SPLIT_RE = re.compile(r"[,:;()\[\]{}.!?\n]+")
GENERIC_QUERY_TERMS = {
    "back",
    "going",
    "looking",
    "noticed",
    "think",
    "really",
    "lately",
    "recently",
    "started",
    "start",
    "ever",
    "still",
    "would",
    "could",
    "should",
    "seems",
    "seem",
    "maybe",
    "actually",
    "just",
    "also",
    "even",
    "kind",
    "sort",
    "much",
    "many",
}
PREFERENCE_QUERY_TERMS = {
    "prefer",
    "preferred",
    "prefers",
    "favorite",
    "favourite",
    "like",
    "likes",
    "love",
    "loves",
    "enjoy",
    "enjoys",
    "dislike",
    "avoid",
    "avoids",
    "hate",
    "hates",
}
TASK_QUERY_TERMS = {
    "name",
    "called",
    "what",
    "which",
    "who",
    "where",
    "when",
    "how",
    "why",
    "amount",
    "number",
    "month",
    "months",
    "year",
    "years",
    "date",
    "time",
    "reason",
    "change",
    "changed",
    "control",
    "pressure",
    "pay",
    "cost",
    "gift",
    "color",
    "colour",
    "breed",
    "type",
    "kind",
}
RELATIONSHIP_QUERY_TERMS = {
    "husband",
    "wife",
    "son",
    "daughter",
    "father",
    "mother",
    "brother",
    "sister",
    "friend",
    "boss",
    "coworker",
    "colleague",
    "mentor",
    "partner",
    "family",
    "relationship",
}
LOCATION_QUERY_TERMS = {
    "where",
    "location",
    "located",
    "live",
    "lives",
    "living",
    "home",
    "city",
    "town",
    "country",
    "office",
    "workshop",
    "school",
    "apartment",
}
ENTITY_STOP_TERMS = STOPWORDS | QUERY_LEAD_IN_TERMS | GENERIC_QUERY_TERMS | {
    "i",
    "im",
    "i'm",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "he",
    "his",
    "she",
    "her",
    "hers",
    "we",
    "our",
    "ours",
    "they",
    "their",
    "theirs",
}
IMPLEMENTATION_QUERY_TERMS = {
    "file",
    "path",
    "function",
    "class",
    "module",
    "parameter",
    "config",
    "argument",
    "method",
    "variable",
    "script",
    "benchmark",
    "adapter",
}
METRIC_QUERY_TERMS = {
    "metric",
    "metrics",
    "recall",
    "ndcg",
    "score",
    "scores",
    "accuracy",
    "latency",
    "runtime",
    "topk",
    "top-k",
    "candidate_recall",
}
CONSTRAINT_QUERY_TERMS = {
    "before",
    "after",
    "without",
    "with",
    "instead",
    "except",
    "only",
    "exact",
    "precise",
    "first",
    "last",
    "same",
    "different",
}
PROFILE_FACT_HINT_TERMS = PREFERENCE_QUERY_TERMS | RELATIONSHIP_QUERY_TERMS | LOCATION_QUERY_TERMS | {
    "prefer",
    "favorite",
    "value",
    "habit",
    "goal",
    "plan",
    "project",
    "status",
    "decision",
    "task",
    "attribute",
}
CHANNEL_SCORE_FIELDS = {
    "dense_semantic": "dense_score",
    "lexical_sparse": "bm25_score",
    "entity_aware": "entity_score",
    "temporal_anchor": "temporal_score",
    "exact_phrase": "exact_phrase_score",
    "profile_side_index": "profile_score",
    "session_bundle": "session_score",
    "temporal_neighbor": "temporal_neighbor_score",
    "parent_session": "parent_score",
    "query_decomposition": "decomposition_score",
}
_SIDE_INDEX_RUNTIME_CACHE: dict[str, dict[str, Any]] = {}
_SIDE_INDEX_LOGGED_KEYS: set[str] = set()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _contains_cjk(text: str) -> bool:
    return bool(CJK_CHAR_RE.search(str(text or "")))


def _cjk_spans(text: str, *, max_spans: int = 10) -> list[str]:
    spans: list[str] = []
    for match in CJK_SPAN_RE.finditer(normalize_text_for_hash(text).lower()):
        span = match.group(0).strip()
        if not span:
            continue
        spans.append(span)
        if len(spans) >= max_spans:
            break
    return list(dict.fromkeys(spans))


def _cjk_subterms(text: str, *, max_terms: int = 72) -> list[str]:
    terms: list[str] = []
    for span in _cjk_spans(text, max_spans=18):
        compact = "".join(span.split())
        if len(compact) < 2:
            continue
        terms.append(compact)
        max_n = min(4, len(compact))
        for n in range(2, max_n + 1):
            for start in range(0, len(compact) - n + 1):
                terms.append(compact[start : start + n])
                if len(terms) >= max_terms:
                    return list(dict.fromkeys(term for term in terms if term))
    return list(dict.fromkeys(term for term in terms if term))


def _semantic_terms(text: str, *, max_terms: int = 160) -> list[str]:
    normalized = normalize_text_for_hash(text).lower()
    terms: list[str] = []
    for token in tokenize(normalized):
        if not token:
            continue
        if _contains_cjk(token) or token not in STOPWORDS:
            terms.append(token)
    if _contains_cjk(normalized):
        terms.extend(_cjk_subterms(normalized, max_terms=max_terms))
    ordered = list(dict.fromkeys(term for term in terms if term))
    return ordered[:max_terms]


def _attribute_terms(text: str, *, max_terms: int = 24) -> list[str]:
    attribute_stopwords = STOPWORDS | QUERY_LEAD_IN_TERMS | GENERIC_QUERY_TERMS
    terms: list[str] = []
    for token in _semantic_terms(text, max_terms=max_terms * 4):
        if not token:
            continue
        if _contains_cjk(token) or token not in attribute_stopwords:
            terms.append(token)
        if len(terms) >= max_terms:
            break
    return list(dict.fromkeys(terms))


def _query_clauses(text: str, *, max_clauses: int = 8) -> list[str]:
    clauses: list[str] = []
    for raw_clause in QUERY_CLAUSE_SPLIT_RE.split(str(text or "")):
        clause = " ".join(raw_clause.split()).strip()
        if len(clause) < 4:
            continue
        clauses.append(clause)
        if len(clauses) >= max_clauses:
            break
    return list(dict.fromkeys(clauses))


def _cjk_overlap_score(query_terms: list[str] | set[str], text: str) -> float:
    normalized = normalize_text_for_hash(text).lower()
    if not normalized:
        return 0.0
    weighted_hits = 0.0
    weighted_total = 0.0
    for term in list(query_terms or []):
        probe = str(term or "").strip().lower()
        if len(probe) < 2 or not _contains_cjk(probe):
            continue
        weight = min(4.0, float(len(probe)))
        weighted_total += weight
        if probe in normalized:
            weighted_hits += weight
    if weighted_total <= 0.0:
        return 0.0
    return min(1.0, weighted_hits / weighted_total)


def _query_specific_terms(query_features: dict[str, Any] | None, *, max_terms: int = 16) -> list[str]:
    if not query_features:
        return []
    entities = set(query_features.get("entities") or [])
    temporal = set(query_features.get("temporal_terms") or [])
    generic = STOPWORDS | QUERY_LEAD_IN_TERMS | GENERIC_QUERY_TERMS
    specific_terms: list[str] = []
    for term in list(query_features.get("attribute_terms") or query_features.get("anchor_terms") or []):
        probe = str(term or "").strip().lower()
        if not probe or probe in entities or probe in temporal or probe in generic:
            continue
        specific_terms.append(probe)
        if len(specific_terms) >= max_terms:
            break
    return list(dict.fromkeys(specific_terms))


def _record_anchor_overlap(
    record: dict[str, Any],
    *,
    anchor_terms: list[str],
    specific_terms: list[str],
    phrases: list[str],
    anchor_probe: str,
    fast: bool = False,
) -> float:
    record_tokens = set(record.get("token_list") or [])
    normalized_text = str(record.get("normalized_text") or "")
    text = str(record.get("text") or normalized_text)
    probe_terms = list(dict.fromkeys(specific_terms or anchor_terms))
    token_overlap = len(set(probe_terms) & record_tokens) / max(1, len(set(probe_terms))) if probe_terms else 0.0
    if fast:
        phrase_score = max(
            (
                _fast_phrase_match_score(phrase, [token for token in _content_terms(phrase) if len(token) >= 2], record)
                for phrase in phrases[:4]
                if phrase
            ),
            default=0.0,
        )
        lexical = _token_list_overlap_score(probe_terms[:10], record)
    else:
        phrase_score = max((_phrase_match_score(phrase, record) for phrase in phrases[:4] if phrase), default=0.0)
        lexical = max(
            lexical_score(anchor_probe, text) if anchor_probe else 0.0,
            lexical_score(" ".join(probe_terms[:10]), text) if probe_terms else 0.0,
        )
    cjk_probe_terms = [term for term in probe_terms if _contains_cjk(term)]
    cjk_overlap = _cjk_overlap_score(cjk_probe_terms, normalized_text) if cjk_probe_terms else 0.0
    return min(1.0, max(token_overlap, phrase_score, lexical, cjk_overlap))


def _term_priority(term: str) -> float:
    probe = str(term or "").strip().lower()
    if not probe:
        return 0.0
    weight = 1.0
    if _contains_cjk(probe):
        weight += min(2.0, max(0.4, len(probe) * 0.18))
    if any(symbol in probe for symbol in ("/", "_", "-", ".", "@", ":")):
        weight += 0.45
    if probe.isdigit():
        weight += 0.3
    if probe in IMPLEMENTATION_QUERY_TERMS or probe in METRIC_QUERY_TERMS or probe in PROFILE_FACT_HINT_TERMS:
        weight += 0.3
    if len(probe) >= 10:
        weight += 0.15
    return min(3.2, weight)


PARENT_ANCHOR_NOISE_TERMS = STOPWORDS | QUERY_LEAD_IN_TERMS | GENERIC_QUERY_TERMS | {
    "i",
    "im",
    "i'm",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "he",
    "his",
    "she",
    "her",
    "hers",
    "we",
    "our",
    "ours",
    "they",
    "theirs",
    "ve",
    "re",
    "ll",
    "m",
    "d",
    "s",
    "t",
    "so",
    "now",
    "over",
    "through",
    "those",
    "thing",
    "things",
    "way",
    "mind",
}

CLONEMEM_PARENT_ANCHOR_EXTRA_NOISE_TERMS = {
    "about",
    "again",
    "all",
    "always",
    "anything",
    "back",
    "been",
    "being",
    "different",
    "everything",
    "every",
    "feel",
    "felt",
    "going",
    "gone",
    "just",
    "kind",
    "like",
    "long",
    "more",
    "much",
    "out",
    "really",
    "still",
    "time",
    "used",
    "whole",
}

def _is_parent_anchor_term(term: str, *, extra_noise_terms: set[str] | None = None) -> bool:
    probe = str(term or "").strip().lower()
    if not probe:
        return False
    if _contains_cjk(probe):
        return len(probe) >= 2
    if len(probe) < 3:
        return False
    if probe in PARENT_ANCHOR_NOISE_TERMS:
        return False
    if extra_noise_terms and probe in extra_noise_terms:
        return False
    if probe in TEMPORAL_TOKEN_RE.findall(probe) and probe not in MONTH_TOKEN_RE.findall(probe):
        return False
    return _term_priority(probe) > 0.0


def _parent_anchor_terms(
    terms: list[str] | set[str],
    *,
    max_terms: int | None = None,
    extra_noise_terms: set[str] | None = None,
) -> list[str]:
    filtered = [
        str(term or "").strip().lower()
        for term in list(terms or [])
        if _is_parent_anchor_term(str(term or ""), extra_noise_terms=extra_noise_terms)
    ]
    ordered = list(dict.fromkeys(filtered))
    return ordered[:max_terms] if max_terms is not None else ordered


def _weighted_surface_overlap(
    terms: list[str] | set[str],
    *,
    record: dict[str, Any],
    normalized_text: str | None = None,
) -> float:
    record_tokens = set(record.get("token_list") or [])
    record_entities = set(record.get("entity_terms") or [])
    normalized_text = normalized_text if normalized_text is not None else str(record.get("normalized_text") or "")
    total_weight = 0.0
    hit_weight = 0.0
    for term in list(terms or []):
        probe = str(term or "").strip().lower()
        if not probe or probe in STOPWORDS:
            continue
        weight = _term_priority(probe)
        if weight <= 0.0:
            continue
        total_weight += weight
        if probe in record_tokens or probe in record_entities or probe in normalized_text:
            hit_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return min(1.0, hit_weight / total_weight)


def _matched_surface_terms(
    record: dict[str, Any],
    *,
    terms: list[str] | set[str],
    normalized_text: str | None = None,
) -> list[str]:
    record_tokens = set(record.get("token_list") or [])
    record_entities = set(record.get("entity_terms") or [])
    normalized_text = normalized_text if normalized_text is not None else str(record.get("normalized_text") or "")
    matched: list[str] = []
    for term in list(dict.fromkeys(list(terms or []))):
        probe = str(term or "").strip().lower()
        if not probe or probe in STOPWORDS:
            continue
        if probe in record_tokens or probe in record_entities or probe in normalized_text:
            matched.append(probe)
    return matched


def _select_parent_anchor_rows(
    child_rows: list[dict[str, Any]],
    *,
    anchor_cap: int,
    filter_low_information_terms: bool = True,
    extra_noise_terms: set[str] | None = None,
) -> list[dict[str, Any]]:
    if anchor_cap <= 0:
        return []
    remaining = list(child_rows)
    uncovered_terms: set[str] = {
        str(term or "").strip().lower()
        for row in remaining
        for term in list(row.get("matched_anchor_terms") or [])
        if str(term or "").strip()
        and (not filter_low_information_terms or _is_parent_anchor_term(str(term or ""), extra_noise_terms=extra_noise_terms))
    }
    selected: list[dict[str, Any]] = []
    search_window = max(anchor_cap * 6, 12)
    while remaining and len(selected) < anchor_cap:
        ranked_window = sorted(
            remaining,
            key=lambda row: (
                float(row.get("anchor_priority") or 0.0),
                float(row.get("matched_term_weight") or 0.0),
                float(row.get("phrase_score") or 0.0),
                float(row.get("direct_match") or 0.0),
                float(row.get("lexical") or 0.0),
            ),
            reverse=True,
        )[:search_window]
        best_row: dict[str, Any] | None = None
        best_key: tuple[float, float, float, float, float, float, float] | None = None
        for row in ranked_window:
            matched_terms = {
                str(term or "").strip().lower()
                for term in list(row.get("matched_anchor_terms") or [])
                if str(term or "").strip()
                and (
                    not filter_low_information_terms
                    or _is_parent_anchor_term(str(term or ""), extra_noise_terms=extra_noise_terms)
                )
            }
            new_terms = matched_terms & uncovered_terms
            new_term_weight = sum(_term_priority(term) for term in new_terms)
            key = (
                1.0 if new_terms else 0.0,
                float(new_term_weight),
                float(row.get("anchor_priority") or 0.0),
                float(row.get("matched_term_weight") or 0.0),
                float(row.get("phrase_score") or 0.0),
                float(row.get("direct_match") or 0.0),
                float(row.get("lexical") or 0.0),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
        if best_row is None:
            break
        selected.append(best_row)
        matched_terms = {
            str(term or "").strip().lower()
            for term in list(best_row.get("matched_anchor_terms") or [])
            if str(term or "").strip()
            and (not filter_low_information_terms or _is_parent_anchor_term(str(term or ""), extra_noise_terms=extra_noise_terms))
        }
        uncovered_terms.difference_update(matched_terms)
        remaining = [row for row in remaining if str(row.get("source_id") or "") != str(best_row.get("source_id") or "")]
    if len(selected) < anchor_cap:
        for row in sorted(
            remaining,
            key=lambda row: (
                float(row.get("anchor_priority") or 0.0),
                float(row.get("matched_term_weight") or 0.0),
                float(row.get("phrase_score") or 0.0),
                float(row.get("direct_match") or 0.0),
                float(row.get("lexical") or 0.0),
            ),
            reverse=True,
        ):
            selected.append(row)
            if len(selected) >= anchor_cap:
                break
    return selected


def _decomposition_surface_terms(decomposition: dict[str, Any] | None) -> list[str]:
    if not decomposition:
        return []
    ordered: list[str] = []
    for key in ("entity_terms", "action_terms", "object_terms", "attribute_terms", "metric_terms", "time_terms"):
        ordered.extend(str(term or "").strip().lower() for term in list(decomposition.get(key) or []))
    return list(dict.fromkeys(term for term in ordered if term))


def _parse_datetime_hint(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except Exception:
            continue
    match = re.search(r"((?:19|20)\d{2}-\d{2}-\d{2})(?:[T\s](\d{2}:\d{2}(?::\d{2})?))?", text)
    if match:
        date_part = match.group(1)
        time_part = match.group(2) or "00:00:00"
        if len(time_part) == 5:
            time_part = f"{time_part}:00"
        try:
            return datetime.fromisoformat(f"{date_part}T{time_part}")
        except Exception:
            return None
    return None


def _temporal_anchor_score(query_time: str | None, candidate_time: str | None) -> float:
    query_dt = _parse_datetime_hint(query_time)
    candidate_dt = _parse_datetime_hint(candidate_time)
    if query_dt is None or candidate_dt is None:
        return 0.0
    delta_seconds = (query_dt - candidate_dt).total_seconds()
    abs_days = abs(delta_seconds) / 86400.0
    if abs_days <= 3.0:
        proximity = 1.0
    elif abs_days <= 30.0:
        proximity = 0.88
    elif abs_days <= 120.0:
        proximity = 0.7
    elif abs_days <= 365.0:
        proximity = 0.5
    elif abs_days <= 730.0:
        proximity = 0.32
    else:
        proximity = 0.0
    if delta_seconds < 0:
        proximity *= 0.45
    return round(max(0.0, min(1.0, proximity)), 4)


def configure_benchmark_determinism(seed: int = DEFAULT_BENCHMARK_SEED) -> dict[str, Any]:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    return {
        "random_seed": int(seed),
        "deterministic_mode": True,
    }


def force_benchmark_config(config: AppConfig) -> AppConfig:
    config.embedding_fail_fast = True
    config.enable_benchmark_route_tuning = True
    config.enable_lightweight_edge_writeback = True
    config.creative_mode = "off"
    config.multi_channel_enabled = True
    config.safe_fusion_enabled = True
    config.dense_preserve_enabled = True
    config.channel_gating_enabled = True
    config.destructive_filter_guard_enabled = True
    config.duplicate_collapse_safe_mode = True
    config.parent_cap_after_gold_agnostic_anchor = True
    config.inhibition_apply_after_candidate_recall_pool = True
    config.fusion_debug_enabled = True
    config.embedding_required_provider = "sentence_transformer"
    config.embedding_required_model = "sentence-transformers/all-MiniLM-L6-v2"
    apply_benchmark_ablation_config(config)
    return config


def current_benchmark_ablation() -> str:
    return os.environ.get("SPHERE_BENCHMARK_ABLATION", "").strip().lower()


def apply_benchmark_ablation_config(config: AppConfig) -> AppConfig:
    ablation = current_benchmark_ablation()
    if not ablation or ablation in {"none", "full", "full_admission", "default"}:
        return config
    if ablation == "no_route_conditioned_admission":
        config.enable_benchmark_route_tuning = False
        config.route_aware_gating_enabled = False
    elif ablation == "no_safe_fusion":
        config.safe_fusion_enabled = False
        config.destructive_filter_guard_enabled = False
        config.duplicate_collapse_safe_mode = False
        config.parent_cap_after_gold_agnostic_anchor = False
    elif ablation == "no_parent_to_segment_selector":
        config.parent_session_enabled = False
        config.parent_top_k = 1
        config.parent_expand_segments = 1
        config.parent_supplemental_anchor_expansion_enabled = False
    elif ablation == "no_temporal_routing":
        config.temporal_channel_enabled = False
        config.temporal_neighbor_enabled = False
        config.enable_temporal_prefilter = False
    elif ablation == "no_lexical_probe":
        config.lexical_channel_enabled = False
        config.exact_phrase_channel_enabled = False
    elif ablation == "no_rerank_guard":
        config.dense_gold_agnostic_rank_floor_enabled = False
        config.clonemem_dense_anchor_rerank_guard_enabled = False
    elif ablation == "no_inhibition":
        config.competition_inhibition_enabled = False
        config.inhibition_apply_after_candidate_recall_pool = False
    elif ablation == "no_benchmark_route_tuning":
        config.enable_benchmark_route_tuning = False
    return config


def _memory_version_from_key(key: str) -> int:
    probe = stable_content_hash(key or "benchmark")
    return int(probe[:15], 16)


def _source_content_hash(source_records: dict[str, dict[str, Any]]) -> str:
    payload = [
        {
            "source_id": str(record.get("source_id") or ""),
            "text_hash": str(record.get("text_hash") or ""),
            "source_doc_id": str(record.get("source_doc_id") or ""),
            "timestamp": str(record.get("timestamp") or ""),
        }
        for _, record in sorted(source_records.items())
    ]
    return stable_content_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _parent_id(record: dict[str, Any]) -> str:
    for key in ("session_id", "conversation_id", "source_doc_id", "sample_id", "source_id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def _entity_like_terms(text: str, *, token_list: list[str] | None = None, entity_terms: list[str] | None = None) -> list[str]:
    lowered = normalize_text_for_hash(text).lower()
    terms = list(entity_terms or [])
    for token in list(token_list or tokenize(lowered)):
        probe = str(token or "").strip().lower()
        if len(probe) < 2:
            continue
        if (
            "/" in probe
            or "_" in probe
            or "-" in probe
            or "." in probe
            or probe in IMPLEMENTATION_QUERY_TERMS
            or probe in METRIC_QUERY_TERMS
            or probe in PROFILE_FACT_HINT_TERMS
            or probe.isdigit()
        ):
            terms.append(probe)
    terms.extend(match.group(0).lower() for match in DATE_HINT_RE.finditer(lowered))
    terms.extend(match.group(0).lower() for match in TIME_HINT_RE.finditer(lowered))
    return list(dict.fromkeys(term for term in terms if term))


def _record_anchor_terms(record: dict[str, Any], *, max_terms: int = 18) -> list[str]:
    attribute_stopwords = STOPWORDS | GENERIC_QUERY_TERMS | QUERY_LEAD_IN_TERMS
    terms: list[str] = []
    for token in list(record.get("token_list") or []):
        if not token:
            continue
        if _contains_cjk(token) or token not in attribute_stopwords:
            terms.append(token)
        if len(terms) >= max_terms:
            break
    return list(dict.fromkeys(terms))


def _infer_profile_categories(record: dict[str, Any]) -> list[str]:
    lowered = str(record.get("normalized_text") or "").lower()
    categories: list[str] = []
    if any(term in lowered for term in PREFERENCE_QUERY_TERMS):
        categories.append("preference")
    if any(term in lowered for term in RELATIONSHIP_QUERY_TERMS):
        categories.append("relationship")
    if any(term in lowered for term in LOCATION_QUERY_TERMS):
        categories.append("location")
    if any(term in lowered for term in TASK_QUERY_TERMS | IMPLEMENTATION_QUERY_TERMS):
        categories.append("task")
    if any(term in lowered for term in TEMPORAL_TOKEN_RE.findall(lowered)) or record.get("timestamp"):
        categories.append("temporal")
    if not categories:
        categories.append("profile_attribute")
    return list(dict.fromkeys(categories))


def _retrieval_policy(config: AppConfig, benchmark_name: str, pool_limit: int) -> dict[str, Any]:
    base = {
        "dense_semantic": bool(config.multi_channel_enabled and config.dense_channel_enabled),
        "lexical_sparse": bool(config.multi_channel_enabled and config.lexical_channel_enabled),
        "entity_aware": bool(config.multi_channel_enabled and config.entity_channel_enabled),
        "temporal_anchor": bool(config.multi_channel_enabled and config.temporal_channel_enabled),
        "exact_phrase": bool(config.multi_channel_enabled and config.exact_phrase_channel_enabled),
        "profile_side_index": bool(config.multi_channel_enabled and config.profile_side_index_enabled),
        "session_bundle": bool(config.multi_channel_enabled and config.session_bundle_enabled),
        "temporal_neighbor": bool(config.multi_channel_enabled and config.temporal_neighbor_enabled),
        "parent_session": bool(config.multi_channel_enabled and config.parent_session_enabled),
        "query_decomposition": bool(config.multi_channel_enabled and config.query_decomposition_enabled),
        "dense_top_k": max(1, int(config.dense_top_k)),
        "lexical_top_k": max(1, int(config.lexical_top_k)),
        "entity_top_k": max(1, int(config.entity_top_k)),
        "temporal_top_k": max(1, int(config.temporal_top_k)),
        "exact_phrase_top_k": max(1, int(config.exact_phrase_top_k)),
        "profile_side_index_top_k": max(1, int(config.profile_side_index_top_k)),
        "session_bundle_top_k": max(1, int(config.session_bundle_top_k)),
        "query_decomposition_top_k": max(1, int(config.query_decomposition_top_k)),
        "parent_top_k": max(1, int(config.parent_top_k)),
        "parent_expand_segments": max(1, int(config.parent_expand_segments)),
        "parent_anchor_noise_filter_enabled": bool(config.parent_anchor_noise_filter_enabled),
        "parent_supplemental_anchor_expansion_enabled": bool(config.parent_supplemental_anchor_expansion_enabled),
        "parent_supplemental_anchor_expansion_cap": max(0, int(config.parent_supplemental_anchor_expansion_cap)),
        "clonemem_parent_timestamp_sibling_expansion_enabled": bool(
            config.clonemem_parent_timestamp_sibling_expansion_enabled
        ),
        "clonemem_parent_timestamp_sibling_expansion_cap": max(
            0,
            int(config.clonemem_parent_timestamp_sibling_expansion_cap),
        ),
        "clonemem_parent_anchor_strict_noise_filter_enabled": bool(
            config.clonemem_parent_anchor_strict_noise_filter_enabled
        ),
        "parent_window_radius": max(1, int(config.local_window_span)),
        "max_neighbors_per_seed": max(1, int(config.max_neighbors_per_seed)),
        "max_total_neighbor_candidates": max(1, int(config.max_total_neighbor_candidates)),
        "fusion_method": str(config.fusion_method or "rrf").lower(),
        "rrf_k": max(1, int(config.rrf_k)),
        "final_candidate_pool_size": max(1, min(int(config.final_candidate_pool_size), max(1, int(pool_limit)))),
        "safe_fusion_enabled": bool(config.safe_fusion_enabled),
        "dense_preserve_enabled": bool(config.dense_preserve_enabled),
        "dense_anchor_top_k": max(1, min(int(config.dense_anchor_top_k), max(1, int(pool_limit)))),
        "dense_anchor_min_keep": max(1, min(int(config.dense_anchor_min_keep), max(1, int(config.dense_anchor_top_k)))),
        "dense_gold_agnostic_rank_floor_enabled": bool(config.dense_gold_agnostic_rank_floor_enabled),
        "clonemem_dense_anchor_rerank_guard_enabled": bool(config.clonemem_dense_anchor_rerank_guard_enabled),
        "clonemem_dense_anchor_rerank_guard_max_rank": max(1, int(config.clonemem_dense_anchor_rerank_guard_max_rank)),
        "clonemem_dense_anchor_rerank_guard_min_dense": max(0.0, float(config.clonemem_dense_anchor_rerank_guard_min_dense)),
        "clonemem_dense_anchor_rerank_guard_min_support": max(1, int(config.clonemem_dense_anchor_rerank_guard_min_support)),
        "clonemem_dense_anchor_rerank_guard_floor": max(0.0, float(config.clonemem_dense_anchor_rerank_guard_floor)),
        "clonemem_evidence_blend_rerank_enabled": bool(config.clonemem_evidence_blend_rerank_enabled),
        "clonemem_evidence_blend_rerank_alpha": max(
            0.0,
            min(1.0, float(config.clonemem_evidence_blend_rerank_alpha)),
        ),
        "clonemem_evidence_blend_min_broad_rank": max(1, int(config.clonemem_evidence_blend_min_broad_rank)),
        "clonemem_evidence_blend_max_broad_rank": max(1, int(config.clonemem_evidence_blend_max_broad_rank)),
        "clonemem_evidence_rank_preservation_enabled": bool(
            config.clonemem_evidence_rank_preservation_enabled
        ),
        "clonemem_evidence_rank_preservation_max_rank": max(
            1,
            int(config.clonemem_evidence_rank_preservation_max_rank),
        ),
        "clonemem_evidence_rank_preservation_min_support": max(
            1,
            int(config.clonemem_evidence_rank_preservation_min_support),
        ),
        "clonemem_evidence_rank_preservation_min_broad_score": max(
            0.0,
            float(config.clonemem_evidence_rank_preservation_min_broad_score),
        ),
        "clonemem_evidence_rank_preservation_floor": max(
            0.0,
            float(config.clonemem_evidence_rank_preservation_floor),
        ),
        "clonemem_evidence_rank_preservation_protected_top_k": max(
            0,
            int(config.clonemem_evidence_rank_preservation_protected_top_k),
        ),
        "clonemem_lexical_anchor_gate_enabled": bool(config.clonemem_lexical_anchor_gate_enabled),
        "clonemem_lexical_anchor_gate_factor": max(
            0.0,
            min(1.0, float(config.clonemem_lexical_anchor_gate_factor)),
        ),
        "clonemem_lexical_anchor_gate_min_support": max(1, int(config.clonemem_lexical_anchor_gate_min_support)),
        "clonemem_lexical_anchor_gate_min_anchor_score": max(
            0.0,
            float(config.clonemem_lexical_anchor_gate_min_anchor_score),
        ),
        "clonemem_lexical_anchor_gate_protected_top_k": max(
            0,
            int(config.clonemem_lexical_anchor_gate_protected_top_k),
        ),
        "clonemem_channel_tail_rescue_enabled": bool(config.clonemem_channel_tail_rescue_enabled),
        "clonemem_channel_tail_rescue_max_rank": max(101, int(config.clonemem_channel_tail_rescue_max_rank)),
        "clonemem_channel_tail_rescue_per_channel": max(0, int(config.clonemem_channel_tail_rescue_per_channel)),
        "clonemem_channel_tail_rescue_target_rank": max(11, int(config.clonemem_channel_tail_rescue_target_rank)),
        "clonemem_evidence_consensus_admission_enabled": bool(
            config.clonemem_evidence_consensus_admission_enabled
        ),
        "clonemem_evidence_consensus_admission_max_candidates": max(
            0,
            int(config.clonemem_evidence_consensus_admission_max_candidates),
        ),
        "clonemem_evidence_consensus_admission_min_channels": max(
            2,
            int(config.clonemem_evidence_consensus_admission_min_channels),
        ),
        "clonemem_evidence_consensus_admission_target_rank": max(
            11,
            int(config.clonemem_evidence_consensus_admission_target_rank),
        ),
        "channel_gating_enabled": bool(config.channel_gating_enabled),
        "destructive_filter_guard_enabled": bool(config.destructive_filter_guard_enabled),
        "duplicate_collapse_safe_mode": bool(config.duplicate_collapse_safe_mode),
        "parent_cap_after_gold_agnostic_anchor": bool(config.parent_cap_after_gold_agnostic_anchor),
        "inhibition_apply_after_candidate_recall_pool": bool(config.inhibition_apply_after_candidate_recall_pool),
        "fusion_debug_enabled": bool(config.fusion_debug_enabled),
        "duplicate_collapse_enabled": bool(config.duplicate_collapse_enabled),
        "near_duplicate_collapse_enabled": bool(config.near_duplicate_collapse_enabled),
        "competition_inhibition_enabled": bool(config.competition_inhibition_enabled),
        "max_candidates_per_parent": max(1, int(config.max_candidates_per_parent)),
        "min_parent_diversity": max(1, int(config.min_parent_diversity)),
        "candidate_recall_eval_k": max(10, int(config.candidate_recall_eval_k)),
        "route_aware_gating_enabled": bool(config.route_aware_gating_enabled),
        "route_aware_gating_aggressiveness": str(config.route_aware_gating_aggressiveness or "safe").lower(),
        "retrieval_early_exit_enabled": bool(config.retrieval_early_exit_enabled),
        "retrieval_latency_budget_ms": max(0, int(config.retrieval_latency_budget_ms)),
        "retrieval_min_seed_candidates": max(1, int(config.retrieval_min_seed_candidates)),
        "retrieval_confidence_margin": max(0.0, float(config.retrieval_confidence_margin)),
    }
    if benchmark_name == "longmemeval":
        base["profile_side_index"] = False
        base["session_bundle"] = False
        base["temporal_neighbor"] = False
        base["parent_session"] = False
        base["query_decomposition"] = False
        base["exact_phrase"] = False
        base["competition_inhibition_enabled"] = False
        base["dense_anchor_min_keep"] = min(base["dense_anchor_top_k"], max(base["dense_anchor_min_keep"], 100))
        base["max_candidates_per_parent"] = max(base["max_candidates_per_parent"], 1000)
        base["min_parent_diversity"] = 1
        base["final_candidate_pool_size"] = min(base["final_candidate_pool_size"], max(1, int(pool_limit)))
    elif benchmark_name == "locomo":
        base["profile_side_index"] = False
        base["session_bundle"] = bool(base["session_bundle"])
        base["parent_session"] = bool(base["parent_session"])
        base["temporal_neighbor"] = bool(base["temporal_neighbor"])
        base["exact_phrase"] = bool(base["exact_phrase"])
        base["competition_inhibition_enabled"] = False
        base["dense_anchor_min_keep"] = min(base["dense_anchor_top_k"], max(base["dense_anchor_min_keep"], 100))
        base["max_candidates_per_parent"] = max(base["max_candidates_per_parent"], 1000)
        base["min_parent_diversity"] = 1
    elif benchmark_name == "knowme":
        base["profile_side_index"] = bool(base["profile_side_index"])
        base["session_bundle"] = False
        base["parent_session"] = bool(base["parent_session"])
        base["temporal_neighbor"] = bool(base["temporal_neighbor"])
        base["query_decomposition"] = bool(base["query_decomposition"])
        base["exact_phrase"] = bool(base["exact_phrase"])
        base["dense_top_k"] = max(base["dense_top_k"], 140)
        base["lexical_top_k"] = max(base["lexical_top_k"], 120)
        base["dense_anchor_min_keep"] = min(base["dense_anchor_top_k"], max(80, min(base["dense_anchor_min_keep"], 90)))
        base["parent_window_radius"] = max(base["parent_window_radius"], 2)
        base["competition_inhibition_enabled"] = False
        base["max_candidates_per_parent"] = max(base["max_candidates_per_parent"], 1000)
        base["min_parent_diversity"] = 1
    elif benchmark_name == "clonemem":
        base["profile_side_index"] = False
        base["session_bundle"] = False
        base["parent_session"] = bool(base["parent_session"])
        base["temporal_neighbor"] = bool(base["temporal_neighbor"])
        base["query_decomposition"] = bool(base["query_decomposition"])
        base["exact_phrase"] = bool(base["exact_phrase"])
        base["dense_top_k"] = max(base["dense_top_k"], 140)
        base["lexical_top_k"] = max(base["lexical_top_k"], 140)
        base["entity_top_k"] = max(base["entity_top_k"], 100)
        base["temporal_top_k"] = max(base["temporal_top_k"], 90)
        base["parent_expand_segments"] = max(base["parent_expand_segments"], 10)
        base["parent_window_radius"] = max(base["parent_window_radius"], 3)
        base["max_total_neighbor_candidates"] = max(base["max_total_neighbor_candidates"], 120)
        base["dense_anchor_min_keep"] = base["dense_anchor_top_k"]
        base["competition_inhibition_enabled"] = False
        base["max_candidates_per_parent"] = max(base["max_candidates_per_parent"], 1000)
        base["min_parent_diversity"] = 1
        if bool(config.enable_benchmark_route_tuning):
            base["clonemem_lexical_anchor_gate_enabled"] = True
            base["clonemem_lexical_anchor_gate_protected_top_k"] = max(
                3,
                int(base.get("clonemem_lexical_anchor_gate_protected_top_k") or 0),
            )
    return base


def _query_has_temporal_or_session_signal(query_features: dict[str, Any], route_context: dict[str, Any] | None = None) -> bool:
    route_context = route_context or {}
    tokens = set(query_features.get("token_set") or [])
    temporal_terms = set(query_features.get("temporal_terms") or []) | set(query_features.get("specific_temporal_terms") or [])
    route_text = " ".join(str(value or "").lower() for value in route_context.values() if isinstance(value, (str, int, float)))
    session_terms = {"session", "conversation", "parent", "order", "before", "after", "previous", "next", "earlier", "later", "first", "last"}
    return bool(temporal_terms or tokens & session_terms or any(term in route_text for term in session_terms))


def _query_has_identity_profile_signal(query_features: dict[str, Any], benchmark_name: str) -> bool:
    tokens = set(query_features.get("token_set") or [])
    profile_terms = PROFILE_FACT_HINT_TERMS | PREFERENCE_QUERY_TERMS | RELATIONSHIP_QUERY_TERMS | LOCATION_QUERY_TERMS
    return benchmark_name in {"knowme", "clonemem"} and bool(
        query_features.get("entities")
        or query_features.get("person_state_phrases")
        or tokens & profile_terms
        or str(query_features.get("focus_person") or "")
    )


def _query_has_multihop_signal(query_features: dict[str, Any], route_context: dict[str, Any] | None = None) -> bool:
    route_context = route_context or {}
    tokens = set(query_features.get("token_set") or [])
    route_type = str(route_context.get("route_type") or route_context.get("query_route") or "").lower()
    return bool(
        tokens & {"compare", "why", "how", "because", "between", "both", "multi", "hop"}
        or len(list(query_features.get("phrases") or [])) >= 3
        or "multi" in route_type
        or "analytical" in route_type
    )


def _apply_route_aware_channel_gating(
    policy: dict[str, Any],
    query_features: dict[str, Any],
    benchmark_name: str,
    route_context: dict[str, Any] | None,
    seed_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gated = dict(policy)
    before = {key: bool(policy.get(key)) for key in CHANNEL_SCORE_FIELDS}
    reasons: dict[str, str] = {}
    if not bool(policy.get("route_aware_gating_enabled", True)):
        gated.update(
            {
                "retrieval_policy_before_gating": before,
                "retrieval_policy_after_gating": before,
                "gated_channels": [],
                "gating_reasons": {},
            }
        )
        return gated

    aggressiveness = str(policy.get("route_aware_gating_aggressiveness") or "safe").lower()
    if aggressiveness not in {"safe", "balanced", "aggressive"}:
        aggressiveness = "safe"
    temporal_signal = _query_has_temporal_or_session_signal(query_features, route_context)
    identity_signal = _query_has_identity_profile_signal(query_features, benchmark_name)
    multihop_signal = _query_has_multihop_signal(query_features, route_context)
    strong_phrase = bool(query_features.get("phrases") or query_features.get("code_like_terms") or query_features.get("metric_like_terms"))
    factual_or_exact = bool(str(query_features.get("query_type") or "").lower() in {"exact", "factual"} or strong_phrase)
    seed_count = int((seed_state or {}).get("seed_candidate_count") or 0)

    def disable(channel: str, reason: str) -> None:
        if bool(gated.get(channel)):
            gated[channel] = False
            reasons[channel] = reason

    if benchmark_name == "longmemeval":
        # Existing policy already disables high-cost expansion; keep it unchanged.
        pass
    elif benchmark_name == "locomo":
        if not temporal_signal and factual_or_exact:
            disable("session_bundle", "locomo_exact_without_session_signal")
            if aggressiveness != "safe":
                disable("temporal_neighbor", "locomo_balanced_no_temporal_signal")
    elif benchmark_name == "knowme":
        if identity_signal:
            gated["profile_side_index"] = bool(policy.get("profile_side_index"))
            gated["entity_aware"] = bool(policy.get("entity_aware"))
            gated["exact_phrase"] = bool(policy.get("exact_phrase"))
        if not temporal_signal and not multihop_signal and factual_or_exact:
            disable("temporal_neighbor", "knowme_exact_no_temporal_or_multihop_signal")
            if seed_count >= 120 or aggressiveness != "safe":
                disable("parent_session", "knowme_seeded_exact_parent_expansion_unneeded")
    elif benchmark_name == "clonemem":
        if not temporal_signal and not multihop_signal and factual_or_exact and not identity_signal:
            disable("temporal_neighbor", "clonemem_exact_no_expansion_signal")
        if identity_signal or multihop_signal:
            gated["query_decomposition"] = bool(policy.get("query_decomposition"))
            gated["parent_session"] = bool(policy.get("parent_session"))
    else:
        if factual_or_exact and not temporal_signal and not multihop_signal and not identity_signal:
            disable("session_bundle", "exact_query_no_session_signal")
            disable("temporal_neighbor", "exact_query_no_temporal_signal")
            disable("parent_session", "exact_query_no_parent_signal")
            if aggressiveness != "safe":
                disable("query_decomposition", "balanced_exact_query_no_multihop_signal")

    if temporal_signal:
        gated["temporal_anchor"] = bool(policy.get("temporal_anchor"))
        gated["temporal_neighbor"] = bool(policy.get("temporal_neighbor"))
        if benchmark_name not in {"clonemem", "knowme"} and seed_count >= max(80, int(policy.get("retrieval_min_seed_candidates") or 80)):
            disable("parent_session", "temporal_query_with_stable_seed_pool")

    after = {key: bool(gated.get(key)) for key in CHANNEL_SCORE_FIELDS}
    gated.update(
        {
            "retrieval_policy_before_gating": before,
            "retrieval_policy_after_gating": after,
            "gated_channels": [channel for channel, enabled in before.items() if enabled and not after.get(channel)],
            "gating_reasons": reasons,
            "gating_aggressiveness": aggressiveness,
        }
    )
    return gated


def _should_early_exit_retrieval(
    *,
    policy: dict[str, Any],
    benchmark_name: str,
    query_features: dict[str, Any],
    seed_candidates: dict[str, dict[str, Any]],
    route_context: dict[str, Any] | None,
    elapsed_ms: float,
) -> dict[str, Any]:
    expensive = ["session_bundle", "temporal_neighbor", "parent_session", "query_decomposition"]
    if bool(policy.get("profile_side_index")):
        expensive.append("profile_side_index_dense")
    if not bool(policy.get("retrieval_early_exit_enabled", True)):
        return {"triggered": False, "reason": "disabled", "skipped_channels": []}
    if benchmark_name in {"clonemem"}:
        return {"triggered": False, "reason": "benchmark_candidate_admission_sensitive", "skipped_channels": []}
    if _query_has_temporal_or_session_signal(query_features, route_context):
        return {"triggered": False, "reason": "temporal_or_session_signal", "skipped_channels": []}
    if _query_has_identity_profile_signal(query_features, benchmark_name):
        return {"triggered": False, "reason": "identity_profile_signal", "skipped_channels": []}
    if _query_has_multihop_signal(query_features, route_context):
        return {"triggered": False, "reason": "multihop_signal", "skipped_channels": []}
    min_candidates = max(int(policy.get("retrieval_min_seed_candidates") or 80), int(policy.get("candidate_recall_eval_k") or 100))
    if len(seed_candidates) < min_candidates:
        return {"triggered": False, "reason": "insufficient_seed_candidates", "skipped_channels": []}
    ranked = sorted(seed_candidates.values(), key=_candidate_seed_confidence, reverse=True)
    top = _candidate_seed_confidence(ranked[0]) if ranked else 0.0
    second = _candidate_seed_confidence(ranked[1]) if len(ranked) > 1 else 0.0
    margin = top - second
    if margin < float(policy.get("retrieval_confidence_margin") or 0.12):
        return {"triggered": False, "reason": "low_confidence_margin", "skipped_channels": []}
    budget_ms = int(policy.get("retrieval_latency_budget_ms") or 0)
    if budget_ms > 0 and elapsed_ms < budget_ms:
        return {"triggered": False, "reason": "latency_budget_not_exceeded", "skipped_channels": []}
    skipped = [channel for channel in expensive if bool(policy.get(channel))]
    return {
        "triggered": bool(skipped),
        "reason": "confident_core_channels_sufficient" if skipped else "no_enabled_expensive_channels",
        "skipped_channels": skipped,
        "seed_candidate_count_at_exit": len(seed_candidates),
    }


def current_git_sha(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def chunker_spec(config: AppConfig, *, granularity: str, benchmark_name: str, adapter_version: str) -> dict[str, Any]:
    return {
        "benchmark_name": benchmark_name,
        "adapter_version": adapter_version,
        "granularity": granularity,
        "chunker_version": BENCHMARK_CHUNKER_VERSION,
        "chunk_size": int(config.chunk_size),
        "chunk_overlap": int(config.chunk_overlap),
        "local_window_span": int(config.local_window_span),
        "embed_local_grain": bool(config.embed_local_grain),
        "markdown_chunk_size": int(config.markdown_chunk_size),
        "code_chunk_lines": int(config.code_chunk_lines),
        "log_chunk_lines": int(config.log_chunk_lines),
        "pdf_chunk_size": int(config.pdf_chunk_size),
        "segment_span_lengths": list(config.segment_span_lengths),
        "tokenizer": "sphere_cli.utils.tokenize",
        "splitter": "memory_writer.prepare_chunks",
        "normalize_embeddings": True,
    }


def build_runtime_fingerprint(
    *,
    config: AppConfig,
    benchmark_name: str,
    adapter_version: str,
    granularity: str,
    vector_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    vector_info = vector_info or {}
    payload = {
        "embedding_provider": str(vector_info.get("embedding_provider") or ""),
        "embedding_model": str(vector_info.get("embedding_model") or config.embedding_model_name),
        "embedding_dim": int(vector_info.get("embedding_dim") or config.embedding_dim),
        "normalize_embeddings": bool(vector_info.get("normalize_embeddings", True)),
        "embedding_preprocess_version": EMBEDDING_PREPROCESS_VERSION,
        "fallback_in_use": bool(vector_info.get("fallback_in_use", False)),
        "vector_backend": str(vector_info.get("vector_backend") or config.vector_backend),
        "vector_fallback_in_use": bool(vector_info.get("vector_fallback_in_use", False)),
        "vector_count": int(vector_info.get("vector_count") or vector_info.get("raw_count") or 0),
        "json_scan_warning": str(vector_info.get("json_scan_warning") or ""),
        "chunker": chunker_spec(
            config,
            granularity=granularity,
            benchmark_name=benchmark_name,
            adapter_version=adapter_version,
        ),
        "benchmark_adapter_version": adapter_version,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "fingerprint_hash": stable_content_hash(canonical),
    }


def assert_benchmark_vector_guard(
    *,
    vector_info: dict[str, Any],
    runtime_fingerprint: dict[str, Any],
    index_fingerprint: dict[str, Any] | None = None,
) -> None:
    provider = str(vector_info.get("embedding_provider") or "")
    model_name = str(vector_info.get("embedding_model") or "")
    if provider == "local_hash":
        raise RuntimeError("Benchmark aborted: embedding_provider=local_hash is forbidden in benchmark mode.")
    if bool(vector_info.get("fallback_in_use")):
        raise RuntimeError("Benchmark aborted: fallback_in_use=true is forbidden in benchmark mode.")
    if bool(vector_info.get("vector_fallback_in_use")):
        raise RuntimeError("Benchmark aborted: vector_fallback_in_use=true is forbidden in benchmark mode.")
    if index_fingerprint:
        runtime_provider = str(runtime_fingerprint.get("embedding_provider") or "")
        runtime_model = str(runtime_fingerprint.get("embedding_model") or "")
        index_provider = str(index_fingerprint.get("embedding_provider") or "")
        index_model = str(index_fingerprint.get("embedding_model") or "")
        if runtime_provider != index_provider or runtime_model != index_model:
            raise RuntimeError(
                "Benchmark aborted: runtime embedding model/provider does not match the persisted index fingerprint."
            )
        if int(runtime_fingerprint.get("embedding_dim") or 0) != int(index_fingerprint.get("embedding_dim") or 0):
            raise RuntimeError("Benchmark aborted: runtime embedding_dim does not match the persisted index fingerprint.")
        if bool(runtime_fingerprint.get("normalize_embeddings", True)) != bool(
            index_fingerprint.get("normalize_embeddings", True)
        ):
            raise RuntimeError(
                "Benchmark aborted: runtime normalize_embeddings does not match the persisted index fingerprint."
            )
        if str(runtime_fingerprint.get("embedding_preprocess_version") or "") != str(
            index_fingerprint.get("embedding_preprocess_version") or ""
        ):
            raise RuntimeError(
                "Benchmark aborted: runtime embedding preprocess version does not match the persisted index fingerprint."
            )
    if provider != str(runtime_fingerprint.get("embedding_provider") or provider):
        raise RuntimeError("Benchmark aborted: runtime embedding provider guard failed.")
    if model_name != str(runtime_fingerprint.get("embedding_model") or model_name):
        raise RuntimeError("Benchmark aborted: runtime embedding model guard failed.")


def annotate_chunks_for_benchmark(
    chunks: list[dict[str, Any]],
    *,
    corpus_item: dict[str, Any],
    benchmark_name: str,
    adapter_version: str,
) -> None:
    source_segment_id = str(
        corpus_item.get("source_segment_id")
        or corpus_item.get("corpus_id")
        or corpus_item.get("source_ref")
        or ""
    )
    source_doc_id = str(
        corpus_item.get("source_doc_id")
        or corpus_item.get("session_id")
        or corpus_item.get("sample_id")
        or source_segment_id
    )
    for chunk in chunks:
        chunk["benchmark_name"] = benchmark_name
        chunk["benchmark_adapter_version"] = adapter_version
        chunk["source_segment_id"] = source_segment_id
        chunk["source_doc_id"] = source_doc_id
        chunk["sample_id"] = str(corpus_item.get("sample_id") or "")
        chunk["conversation_id"] = str(corpus_item.get("conversation_id") or "")
        chunk["turn_id"] = str(corpus_item.get("turn_id") or "")
        chunk["speaker_id"] = str(corpus_item.get("speaker_id") or "")
        chunk["original_segment_text"] = str(corpus_item.get("text") or "")
        chunk["source_ref"] = source_segment_id or str(chunk.get("source_ref") or "")
        chunk["source_path"] = source_doc_id or str(chunk.get("source_path") or "")
        chunk["content_ref"] = source_segment_id or str(chunk.get("content_ref") or "")


def _normalize_record(item: dict[str, Any]) -> dict[str, Any]:
    normalized_text = normalize_text_for_hash(str(item.get("text") or "")).lower()
    token_list = _semantic_terms(normalized_text)
    entity_terms = [term.strip().lower() for term in ENTITY_RE.findall(str(item.get("text") or "")) if term.strip()]
    entity_terms.extend(match.group(0).lower() for match in YEAR_RE.finditer(str(item.get("text") or "")))
    temporal_terms, specific_temporal_terms = _temporal_term_buckets(str(item.get("text") or ""))
    return {
        **item,
        "source_id": str(
            item.get("source_segment_id")
            or item.get("corpus_id")
            or item.get("source_ref")
            or ""
        ),
        "source_segment_id": str(item.get("source_segment_id") or item.get("corpus_id") or ""),
        "source_doc_id": str(item.get("source_doc_id") or item.get("session_id") or item.get("sample_id") or ""),
        "normalized_text": normalized_text,
        "token_list": token_list,
        "entity_terms": list(dict.fromkeys(entity_terms)),
        "temporal_terms": list(dict.fromkeys(temporal_terms)),
        "specific_temporal_terms": list(dict.fromkeys(specific_temporal_terms)),
        "text_hash": stable_content_hash(normalized_text),
    }


def build_index_metadata(
    *,
    corpus_items: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    benchmark_name: str,
    adapter_version: str,
    runtime_fingerprint: dict[str, Any],
    raw_counts: dict[str, Any],
) -> dict[str, Any]:
    source_records: dict[str, dict[str, Any]] = {}
    indexed_doc_ids: set[str] = set()
    indexed_segment_ids: set[str] = set()
    duplicate_source_ids: Counter[str] = Counter()
    for index, item in enumerate(corpus_items):
        record = _normalize_record(
            {
                **item,
                "order_index": index,
                "benchmark_name": benchmark_name,
                "benchmark_adapter_version": adapter_version,
            }
        )
        source_id = str(record.get("source_id") or "")
        if not source_id:
            continue
        duplicate_source_ids[source_id] += 1
        source_records[source_id] = record
        indexed_doc_ids.add(str(record.get("source_doc_id") or ""))
        indexed_segment_ids.add(source_id)
    chunk_metadata_by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            continue
        source_segment_id = str(chunk.get("source_segment_id") or chunk.get("source_ref") or chunk.get("content_ref") or "")
        source_doc_id = str(chunk.get("source_doc_id") or chunk.get("source_path") or "")
        chunk_metadata_by_id[chunk_id] = {
            "chunk_id": chunk_id,
            "node_id": str(chunk.get("node_id") or ""),
            "grain": str(chunk.get("grain") or "micro"),
            "benchmark_name": str(chunk.get("benchmark_name") or benchmark_name),
            "benchmark_adapter_version": str(chunk.get("benchmark_adapter_version") or adapter_version),
            "source_segment_id": source_segment_id,
            "source_doc_id": source_doc_id,
            "sample_id": str(chunk.get("sample_id") or ""),
            "conversation_id": str(chunk.get("conversation_id") or ""),
            "session_id": str(chunk.get("session_id") or ""),
            "turn_id": str(chunk.get("turn_id") or ""),
            "speaker_id": str(chunk.get("speaker_id") or ""),
            "source_ref": str(chunk.get("source_ref") or ""),
            "source_path": str(chunk.get("source_path") or ""),
            "original_segment_text": str(chunk.get("original_segment_text") or chunk.get("text") or ""),
        }
    source_content_hash = _source_content_hash(source_records)
    return {
        "benchmark_name": benchmark_name,
        "benchmark_adapter_version": adapter_version,
        "index_built_at": now_iso(),
        "fingerprint": runtime_fingerprint,
        "retrieval_source_content_hash": source_content_hash,
        "raw_counts": dict(raw_counts or {}),
        "index_doc_count": len(source_records),
        "chunk_count": len(chunk_metadata_by_id),
        "unique_segment_count": len(indexed_segment_ids),
        "indexed_doc_ids": sorted(doc_id for doc_id in indexed_doc_ids if doc_id),
        "indexed_segment_ids": sorted(seg_id for seg_id in indexed_segment_ids if seg_id),
        "duplicate_source_id_count": sum(1 for count in duplicate_source_ids.values() if count > 1),
        "source_records_by_id": source_records,
        "chunk_metadata_by_id": chunk_metadata_by_id,
    }


def report_root(out_file: Path | None, benchmark_name: str) -> Path:
    if out_file is not None:
        return out_file.parent / "reports"
    return Path.cwd() / "reports" / benchmark_name


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def build_raw_counts(
    corpus_items: list[dict[str, Any]],
    *,
    question_count: int,
    session_count: int | None = None,
) -> dict[str, Any]:
    timestamps = [str(item.get("timestamp") or "").strip() for item in corpus_items]
    source_ids = [
        str(item.get("source_segment_id") or item.get("corpus_id") or "").strip()
        for item in corpus_items
    ]
    doc_ids = [
        str(item.get("source_doc_id") or item.get("session_id") or item.get("sample_id") or "").strip()
        for item in corpus_items
    ]
    present_timestamps = [timestamp for timestamp in timestamps if timestamp]
    parsed_timestamps = [timestamp for timestamp in present_timestamps if _parse_datetime_hint(timestamp) is not None]
    return {
        "raw_document_count": len(corpus_items),
        "raw_session_count": int(
            session_count
            if session_count is not None
            else len({doc_id for doc_id in doc_ids if doc_id})
        ),
        "raw_segment_count": len(corpus_items),
        "memory_count": len(corpus_items),
        "question_count": int(question_count),
        "empty_text_count": sum(1 for item in corpus_items if not str(item.get("text") or "").strip()),
        "duplicate_raw_segment_id_count": sum(1 for count in Counter(source_ids).values() if count > 1),
        "duplicate_raw_doc_id_count": sum(1 for count in Counter(doc_ids).values() if count > 1),
        "timestamp_field_count": len(present_timestamps),
        "timestamp_parseable_count": len(parsed_timestamps),
        "timestamp_parse_rate": round(len(parsed_timestamps) / max(1, len(present_timestamps)), 4)
        if present_timestamps
        else 1.0,
    }


def _build_retrieval_side_index(
    *,
    benchmark_name: str,
    source_records: dict[str, dict[str, Any]],
    index_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lexical_postings: dict[str, list[str]] = defaultdict(list)
    token_to_source_ids: dict[str, list[str]] = defaultdict(list)
    source_term_freqs: dict[str, dict[str, int]] = {}
    doc_lengths: dict[str, int] = {}
    entity_postings: dict[str, list[str]] = defaultdict(list)
    entity_to_source_ids: dict[str, list[str]] = defaultdict(list)
    temporal_term_to_source_ids: dict[str, list[str]] = defaultdict(list)
    specific_temporal_term_to_source_ids: dict[str, list[str]] = defaultdict(list)
    phrase_token_to_source_ids: dict[str, list[str]] = defaultdict(list)
    session_id_to_source_ids: dict[str, list[str]] = defaultdict(list)
    parent_id_to_source_ids: dict[str, list[str]] = defaultdict(list)
    source_id_to_order_index: dict[str, int] = {}
    source_id_to_session_id: dict[str, str] = {}
    source_id_to_parent_id: dict[str, str] = {}
    text_hash_to_source_id: dict[str, str] = {}
    timestamp_source_ids: list[str] = []
    parent_members: dict[str, list[str]] = defaultdict(list)
    source_parent_ids: dict[str, str] = {}
    profile_side_entries: list[dict[str, Any]] = []
    profile_entry_by_id: dict[str, dict[str, Any]] = {}
    profile_entry_ids_by_term: dict[str, list[str]] = defaultdict(list)
    profile_entry_ids_by_category: dict[str, list[str]] = defaultdict(list)
    parent_records_by_id: dict[str, dict[str, Any]] = {}
    total_doc_length = 0
    for source_id, record in sorted(source_records.items()):
        token_counts = Counter(list(record.get("token_list") or []))
        source_term_freqs[source_id] = dict(token_counts)
        doc_length = max(1, sum(token_counts.values()))
        doc_lengths[source_id] = doc_length
        total_doc_length += doc_length
        for term in token_counts:
            lexical_postings[term].append(source_id)
            token_to_source_ids[term].append(source_id)
            phrase_token_to_source_ids[term].append(source_id)
        entity_terms = _entity_like_terms(
            str(record.get("text") or ""),
            token_list=list(record.get("token_list") or []),
            entity_terms=list(record.get("entity_terms") or []),
        )
        for term in entity_terms:
            entity_postings[term].append(source_id)
            entity_to_source_ids[term].append(source_id)
        for term in list(record.get("temporal_terms") or []):
            temporal_term_to_source_ids[str(term).lower()].append(source_id)
        for term in list(record.get("specific_temporal_terms") or []):
            specific_temporal_term_to_source_ids[str(term).lower()].append(source_id)
        parent_id = _parent_id(record)
        source_parent_ids[source_id] = parent_id
        source_id_to_parent_id[source_id] = parent_id
        parent_id_to_source_ids[parent_id].append(source_id)
        session_id = str(record.get("session_id") or "")
        source_id_to_session_id[source_id] = session_id
        if session_id:
            session_id_to_source_ids[session_id].append(source_id)
        source_id_to_order_index[source_id] = int(record.get("order_index") or 0)
        text_hash = str(record.get("text_hash") or "")
        if text_hash and text_hash not in text_hash_to_source_id:
            text_hash_to_source_id[text_hash] = source_id
        if str(record.get("timestamp") or "").strip():
            timestamp_source_ids.append(source_id)
        parent_members[parent_id].append(source_id)
        anchor_terms = _record_anchor_terms(record)
        profile_terms = list(
            dict.fromkeys(
                anchor_terms[:16]
                + entity_terms[:16]
                + list(record.get("temporal_terms") or [])[:8]
                + list(record.get("specific_temporal_terms") or [])[:8]
            )
        )
        for category in _infer_profile_categories(record):
            entry_id = f"{source_id}:{category}"
            entry = {
                "entry_id": entry_id,
                "source_id": source_id,
                "parent_id": parent_id,
                "category": category,
                "subject": str(record.get("speaker_id") or record.get("sample_id") or ""),
                "attribute_terms": anchor_terms[:12],
                "entity_terms": entity_terms[:12],
                "temporal_terms": list(record.get("temporal_terms") or [])[:8],
                "text": str(record.get("text") or ""),
            }
            profile_side_entries.append(entry)
            profile_entry_by_id[entry_id] = entry
            profile_entry_ids_by_category[category].append(entry_id)
            for term in profile_terms:
                normalized_term = str(term or "").strip().lower()
                if normalized_term:
                    profile_entry_ids_by_term[normalized_term].append(entry_id)
    for parent_id, member_ids in parent_members.items():
        ordered_ids = sorted(
            member_ids,
            key=lambda source_id: (
                int(source_records.get(source_id, {}).get("order_index") or 0),
                source_id,
            ),
        )
        joined_text = "\n".join(str(source_records[source_id].get("text") or "") for source_id in ordered_ids[:48])
        joined_entities: list[str] = []
        joined_temporal: list[str] = []
        joined_tokens: list[str] = []
        for source_id in ordered_ids:
            record = source_records[source_id]
            joined_entities.extend(list(record.get("entity_terms") or []))
            joined_temporal.extend(list(record.get("temporal_terms") or []))
            joined_tokens.extend(list(record.get("token_list") or [])[:24])
        parent_records_by_id[parent_id] = {
            "parent_id": parent_id,
            "source_ids": ordered_ids,
            "source_count": len(ordered_ids),
            "normalized_text": normalize_text_for_hash(joined_text).lower(),
            "token_list": list(dict.fromkeys(joined_tokens))[:160],
            "entity_terms": list(dict.fromkeys(joined_entities))[:64],
            "temporal_terms": list(dict.fromkeys(joined_temporal))[:32],
        }
    neighbor_map: dict[str, dict[str, list[str]]] = {}
    for parent in parent_records_by_id.values():
        source_ids = list(parent.get("source_ids") or [])
        for index, source_id in enumerate(source_ids):
            neighbor_map[source_id] = {
                "prev": source_ids[:index][-6:],
                "next": source_ids[index + 1 : index + 7],
            }
    source_hash = _source_content_hash(source_records)
    return {
        "benchmark_name": benchmark_name,
        "schema_version": MULTI_CHANNEL_RETRIEVAL_SCHEMA_VERSION,
        "source_content_hash": source_hash,
        "embedding_fingerprint_hash": str((index_fingerprint or {}).get("fingerprint_hash") or ""),
        "document_count": len(source_records),
        "avg_doc_length": round(total_doc_length / max(1, len(source_records)), 4),
        "lexical_postings": {term: sorted(set(source_ids)) for term, source_ids in lexical_postings.items()},
        "token_to_source_ids": {term: sorted(set(source_ids)) for term, source_ids in token_to_source_ids.items()},
        "source_term_freqs": source_term_freqs,
        "doc_lengths": doc_lengths,
        "entity_postings": {term: sorted(set(source_ids)) for term, source_ids in entity_postings.items()},
        "entity_to_source_ids": {term: sorted(set(source_ids)) for term, source_ids in entity_to_source_ids.items()},
        "temporal_term_to_source_ids": {term: sorted(set(source_ids)) for term, source_ids in temporal_term_to_source_ids.items()},
        "specific_temporal_term_to_source_ids": {term: sorted(set(source_ids)) for term, source_ids in specific_temporal_term_to_source_ids.items()},
        "phrase_token_to_source_ids": {term: sorted(set(source_ids)) for term, source_ids in phrase_token_to_source_ids.items()},
        "session_id_to_source_ids": {
            session_id: sorted(set(source_ids), key=lambda source_id: (source_id_to_order_index.get(source_id, 0), source_id))
            for session_id, source_ids in session_id_to_source_ids.items()
        },
        "parent_id_to_source_ids": {
            parent_id: sorted(set(source_ids), key=lambda source_id: (source_id_to_order_index.get(source_id, 0), source_id))
            for parent_id, source_ids in parent_id_to_source_ids.items()
        },
        "source_id_to_order_index": source_id_to_order_index,
        "source_id_to_session_id": source_id_to_session_id,
        "source_id_to_parent_id": source_id_to_parent_id,
        "text_hash_to_source_id": text_hash_to_source_id,
        "timestamp_source_ids": sorted(set(timestamp_source_ids), key=lambda source_id: (source_id_to_order_index.get(source_id, 0), source_id)),
        "parent_records_by_id": parent_records_by_id,
        "source_parent_ids": source_parent_ids,
        "neighbor_map": neighbor_map,
        "profile_side_entries": profile_side_entries,
        "profile_entry_by_id": profile_entry_by_id,
        "profile_entry_ids_by_term": {
            term: sorted(set(entry_ids))
            for term, entry_ids in profile_entry_ids_by_term.items()
        },
        "profile_entry_ids_by_category": {
            category: sorted(set(entry_ids))
            for category, entry_ids in profile_entry_ids_by_category.items()
        },
    }


def _load_retrieval_side_index(
    *,
    benchmark_name: str,
    index_metadata: dict[str, Any],
) -> dict[str, Any]:
    source_records = index_metadata.get("source_records_by_id") or {}
    index_fingerprint = dict(index_metadata.get("fingerprint") or {})
    source_hash = str(index_metadata.get("retrieval_source_content_hash") or "")
    if not source_hash:
        source_hash = _source_content_hash(source_records)
    cache_payload = {
        "benchmark_name": benchmark_name,
        "schema_version": MULTI_CHANNEL_RETRIEVAL_SCHEMA_VERSION,
        "source_content_hash": source_hash,
        "index_fingerprint_hash": str(index_fingerprint.get("fingerprint_hash") or ""),
    }
    cache_key = stable_content_hash(json.dumps(cache_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    cached = _SIDE_INDEX_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return {
            **cached,
            "_cache": {"status": "runtime", "cache_key": cache_key},
        }
    workspace_dir = str(index_metadata.get("workspace_dir") or "").strip()
    cache_path = Path(workspace_dir) / "retrieval_side_index.json" if workspace_dir else None
    if cache_path is not None and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if str(payload.get("cache_key") or "") == cache_key:
                side_index = dict(payload.get("side_index") or {})
                _SIDE_INDEX_RUNTIME_CACHE[cache_key] = side_index
                return {
                    **side_index,
                    "_cache": {"status": "disk", "cache_key": cache_key, "path": str(cache_path)},
                }
        except Exception:
            pass
    side_index = _build_retrieval_side_index(
        benchmark_name=benchmark_name,
        source_records=source_records,
        index_fingerprint=index_fingerprint,
    )
    _SIDE_INDEX_RUNTIME_CACHE[cache_key] = side_index
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "created_at": now_iso(),
                    "side_index": side_index,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        **side_index,
        "_cache": {"status": "built", "cache_key": cache_key, "path": str(cache_path) if cache_path is not None else ""},
    }


def _decomposition_query_fingerprint(
    *,
    benchmark_name: str,
    side_index: dict[str, Any],
    query_features: dict[str, Any],
) -> tuple[str, int]:
    payload = {
        "benchmark_name": benchmark_name,
        "schema_version": QUERY_DECOMPOSITION_SCHEMA_VERSION,
        "source_content_hash": str(side_index.get("source_content_hash") or ""),
        "query_hash": str(query_features.get("query_text_hash") or ""),
        "deterministic_mode": True,
    }
    key = stable_content_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return key, _memory_version_from_key(str(side_index.get("source_content_hash") or "") + key)


def _build_query_decomposition(
    query: str,
    *,
    query_features: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_text_for_hash(query).lower()
    tokens = list(query_features.get("anchor_terms") or query_features.get("token_list") or [])
    action_terms = [
        token
        for token in tokens
        if token.endswith("ed") or token.endswith("ing") or token in {"fix", "change", "update", "build", "prefer", "decide"}
    ][:12]
    metric_terms = [token for token in tokens if token in METRIC_QUERY_TERMS or token.replace("_", "") in METRIC_QUERY_TERMS][:12]
    constraint_terms = [token for token in tokens if token in CONSTRAINT_QUERY_TERMS][:12]
    object_terms = [
        token
        for token in list(query_features.get("attribute_terms") or [])
        if token not in action_terms and token not in metric_terms and token not in constraint_terms
    ][:16]
    evidence_type = "generic"
    if any(token in normalized for token in PREFERENCE_QUERY_TERMS):
        evidence_type = "preference"
    elif any(token in normalized for token in RELATIONSHIP_QUERY_TERMS):
        evidence_type = "relationship"
    elif any(token in normalized for token in LOCATION_QUERY_TERMS):
        evidence_type = "location"
    elif any(token in normalized for token in TASK_QUERY_TERMS | IMPLEMENTATION_QUERY_TERMS):
        evidence_type = "task"
    elif list(query_features.get("temporal_terms") or []):
        evidence_type = "temporal"
    elif list(query_features.get("entities") or []):
        evidence_type = "profile_attribute"
    return {
        "schema_version": QUERY_DECOMPOSITION_SCHEMA_VERSION,
        "entity_terms": list(dict.fromkeys(list(query_features.get("entities") or [])))[:16],
        "action_terms": list(dict.fromkeys(action_terms)),
        "object_terms": list(dict.fromkeys(object_terms)),
        "attribute_terms": list(dict.fromkeys(list(query_features.get("attribute_terms") or [])))[:16],
        "time_terms": list(dict.fromkeys(list(query_features.get("specific_temporal_terms") or query_features.get("temporal_terms") or [])))[:12],
        "metric_terms": list(dict.fromkeys(metric_terms)),
        "constraint_terms": list(dict.fromkeys(constraint_terms)),
        "evidence_type": evidence_type,
        "clauses": list(dict.fromkeys(list(query_features.get("clauses") or [])))[:8],
    }


def _load_query_decomposition(
    *,
    query: str,
    benchmark_name: str,
    storage: Storage,
    side_index: dict[str, Any],
    query_features: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_fingerprint, memory_version = _decomposition_query_fingerprint(
        benchmark_name=benchmark_name,
        side_index=side_index,
        query_features=query_features,
    )
    cached = storage.get_retrieval_cache(query_fingerprint, memory_version)
    if cached is not None and isinstance(cached.get("payload"), dict):
        return dict(cached["payload"]), {"cache_hit": True, "cache_key": query_fingerprint}
    payload = _build_query_decomposition(query, query_features=query_features)
    storage.put_retrieval_cache(
        query_fingerprint=query_fingerprint,
        normalized_query=str(query_features.get("normalized_query") or normalize_text_for_hash(query)),
        task_type="benchmark_candidate_generation",
        route_type=QUERY_DECOMPOSITION_SCHEMA_VERSION,
        memory_version=memory_version,
        payload=payload,
        created_at=now_iso(),
    )
    return payload, {"cache_hit": False, "cache_key": query_fingerprint}


def _candidate_seed(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(record.get("source_id") or ""),
        "source_segment_id": str(record.get("source_segment_id") or record.get("source_id") or ""),
        "source_doc_id": str(record.get("source_doc_id") or ""),
        "benchmark_name": str(record.get("benchmark_name") or ""),
        "sample_id": str(record.get("sample_id") or ""),
        "conversation_id": str(record.get("conversation_id") or ""),
        "session_id": str(record.get("session_id") or ""),
        "turn_id": str(record.get("turn_id") or ""),
        "speaker_id": str(record.get("speaker_id") or ""),
        "timestamp": str(record.get("timestamp") or ""),
        "text": str(record.get("text") or ""),
        "normalized_text": str(record.get("normalized_text") or ""),
        "text_hash": str(record.get("text_hash") or ""),
        "token_list": list(record.get("token_list") or []),
        "entity_terms": list(record.get("entity_terms") or []),
        "temporal_terms": list(record.get("temporal_terms") or []),
        "specific_temporal_terms": list(record.get("specific_temporal_terms") or []),
        "order_index": int(record.get("order_index") or 0),
        "source_retrievers": [],
        "source_chunk_ids": [],
        "best_chunk_id": "",
        "dense_score": 0.0,
        "bm25_score": 0.0,
        "entity_score": 0.0,
        "temporal_score": 0.0,
        "profile_score": 0.0,
        "session_score": 0.0,
        "exact_phrase_score": 0.0,
        "speaker_score": 0.0,
        "temporal_neighbor_score": 0.0,
        "parent_score": 0.0,
        "decomposition_score": 0.0,
        "local_window_score": 0.0,
        "fusion_score": 0.0,
        "broad_score": 0.0,
        "rerank_score": 0.0,
        "post_inhibition_score": 0.0,
        "inhibition_penalty": 0.0,
    }


def _content_terms(text: str) -> list[str]:
    return _semantic_terms(text, max_terms=96)


def _temporal_term_buckets(text: str) -> tuple[list[str], list[str]]:
    normalized = normalize_text_for_hash(str(text)).lower()
    generic_terms = [match.group(0).lower() for match in TEMPORAL_TOKEN_RE.finditer(normalized)]
    month_terms = [match.group(0).lower() for match in MONTH_TOKEN_RE.finditer(normalized)]
    year_terms = [match.group(0).lower() for match in YEAR_RE.finditer(normalized)]
    relative_terms = [normalize_text_for_hash(match.group(0)).lower() for match in RELATIVE_TIME_PHRASE_RE.finditer(normalized)]
    specific_terms = list(dict.fromkeys(month_terms + year_terms + relative_terms))
    return list(dict.fromkeys(generic_terms + specific_terms)), specific_terms


def _query_features(query: str, route_context: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_text_for_hash(query).lower()
    terms = _content_terms(query)
    attribute_terms = _attribute_terms(query, max_terms=24)
    anchor_terms = [token for token in attribute_terms if _contains_cjk(token) or token not in QUERY_LEAD_IN_TERMS]
    if not anchor_terms:
        anchor_terms = [token for token in terms if _contains_cjk(token) or token not in QUERY_LEAD_IN_TERMS]
    if not anchor_terms:
        anchor_terms = list(terms)
    phrases = []
    for match in re.finditer(r"\"([^\"]{4,})\"", query):
        phrases.append(match.group(1).strip().lower())
    code_like_terms = [match.group(0).lower() for match in CODE_LIKE_RE.finditer(normalized)]
    metric_like_terms = [match.group(0).lower() for match in METRIC_LIKE_RE.finditer(normalized)]
    person_state_phrases = [
        normalize_text_for_hash(match.group(0)).lower()
        for match in PERSON_STATE_PHRASE_RE.finditer(query)
        if len(normalize_text_for_hash(match.group(0)).strip()) >= 4
    ]
    cjk_spans = _cjk_spans(query)
    cjk_terms = _cjk_subterms(query, max_terms=72)
    phrases.extend(cjk_spans)
    phrases.extend(code_like_terms[:8])
    phrases.extend(metric_like_terms[:8])
    phrases.extend(person_state_phrases[:8])
    if anchor_terms:
        phrases.append(" ".join(anchor_terms[-min(6, len(anchor_terms)) :]))
        if len(anchor_terms) > 6:
            phrases.append(" ".join(anchor_terms[: min(6, len(anchor_terms))]))
        if len(anchor_terms) >= 8:
            middle = max(0, len(anchor_terms) // 2 - 2)
            phrases.append(" ".join(anchor_terms[middle : middle + 4]))
    entities = [
        term.strip().lower()
        for term in ENTITY_RE.findall(query)
        if term.strip() and normalize_text_for_hash(term).lower() not in ENTITY_STOP_TERMS
    ]
    years = [match.group(0).lower() for match in YEAR_RE.finditer(query)]
    temporal_terms, specific_temporal_terms = _temporal_term_buckets(query)
    temporal_direction = "unspecified"
    if any(term in normalized for term in ("latest", "current", "currently", "recent", "most recent", "last", "later", "after", "next")):
        temporal_direction = "newer"
    if any(term in normalized for term in ("before", "earlier", "previous", "first", "initial", "original")):
        temporal_direction = "older" if temporal_direction == "unspecified" else temporal_direction
    focus_person = str((route_context or {}).get("person_name") or "").strip().lower()
    question_time = str((route_context or {}).get("question_time") or "").strip()
    clauses = _query_clauses(query)
    preference_terms = [term for term in anchor_terms if term in PREFERENCE_QUERY_TERMS]
    task_terms = [term for term in anchor_terms if term in TASK_QUERY_TERMS]
    evidence_intent = "factual"
    if preference_terms:
        evidence_intent = "preference"
    elif specific_temporal_terms or temporal_terms or question_time:
        evidence_intent = "temporal"
    elif focus_person or entities:
        evidence_intent = "entity"
    question_time_terms = list(
        dict.fromkeys([match.group(0) for match in DATE_HINT_RE.finditer(question_time)] + TIME_HINT_RE.findall(question_time))
    )
    if focus_person:
        entities.append(focus_person)
    ordered_anchor_terms = list(dict.fromkeys(anchor_terms))
    return {
        "normalized_query": normalized,
        "query_text_hash": stable_content_hash(normalized),
        "token_list": ordered_anchor_terms or terms,
        "token_set": set(ordered_anchor_terms or terms),
        "anchor_terms": ordered_anchor_terms,
        "attribute_terms": list(dict.fromkeys(attribute_terms)),
        "phrases": list(dict.fromkeys(phrase for phrase in phrases if phrase)),
        "code_like_terms": list(dict.fromkeys(code_like_terms)),
        "metric_like_terms": list(dict.fromkeys(metric_like_terms)),
        "person_state_phrases": list(dict.fromkeys(person_state_phrases)),
        "clauses": clauses,
        "entities": list(dict.fromkeys(entities + years)),
        "preference_terms": list(dict.fromkeys(preference_terms)),
        "task_terms": list(dict.fromkeys(task_terms)),
        "evidence_intent": evidence_intent,
        "temporal_terms": list(dict.fromkeys(temporal_terms + years)),
        "specific_temporal_terms": list(dict.fromkeys(specific_temporal_terms + years)),
        "temporal_direction": temporal_direction,
        "cjk_spans": cjk_spans,
        "cjk_terms": cjk_terms,
        "focus_person": focus_person,
        "question_time": question_time,
        "question_time_terms": question_time_terms,
    }


def _focused_query_variants(query: str, query_features: dict[str, Any] | None = None) -> list[str]:
    variants: list[str] = [query]
    if not query_features:
        return variants
    anchor_terms = list(query_features.get("anchor_terms") or query_features.get("attribute_terms") or query_features.get("token_list") or [])
    entities = list(query_features.get("entities") or [])
    specific_temporal_terms = list(query_features.get("specific_temporal_terms") or [])
    preference_terms = list(query_features.get("preference_terms") or [])
    task_terms = list(query_features.get("task_terms") or [])
    cjk_spans = list(query_features.get("cjk_spans") or [])
    clauses = list(query_features.get("clauses") or [])
    code_like_terms = list(query_features.get("code_like_terms") or [])
    metric_like_terms = list(query_features.get("metric_like_terms") or [])
    person_state_phrases = list(query_features.get("person_state_phrases") or [])
    if anchor_terms:
        head = " ".join(anchor_terms[: min(10, len(anchor_terms))])
        tail = " ".join(anchor_terms[-min(8, len(anchor_terms)) :])
        for variant in (head, tail):
            if variant and variant not in variants:
                variants.append(variant)
    focused = " ".join(
        list(dict.fromkeys(entities + specific_temporal_terms + code_like_terms + metric_like_terms + preference_terms + task_terms + anchor_terms))[:12]
    )
    if focused and focused not in variants:
        variants.append(focused)
    for phrase in person_state_phrases[:4]:
        if phrase and phrase not in variants:
            variants.append(phrase)
    for clause in clauses:
        clause_terms = _attribute_terms(clause, max_terms=12)
        compact_clause = " ".join(clause_terms[:10]) if clause_terms else clause
        if compact_clause and compact_clause not in variants:
            variants.append(compact_clause)
    for variant in sorted(cjk_spans, key=len, reverse=True)[:4]:
        if variant and variant not in variants:
            variants.append(variant)
    return variants[:10]


def _map_chunk_hit_to_source(
    hit: dict[str, Any],
    chunk_metadata_by_id: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    chunk_id = str(hit.get("chunk_id") or "")
    metadata = dict(chunk_metadata_by_id.get(chunk_id) or {})
    meta = dict(hit.get("metadata") or {})
    if not metadata:
        metadata = {
            "chunk_id": chunk_id,
            "source_segment_id": str(meta.get("source_segment_id") or meta.get("source_ref") or meta.get("content_ref") or ""),
            "source_doc_id": str(meta.get("source_doc_id") or meta.get("source_path") or ""),
            "benchmark_name": str(meta.get("benchmark_name") or ""),
            "sample_id": str(meta.get("sample_id") or ""),
            "conversation_id": str(meta.get("conversation_id") or ""),
            "session_id": str(meta.get("session_id") or ""),
            "turn_id": str(meta.get("turn_id") or ""),
            "speaker_id": str(meta.get("speaker_id") or ""),
            "original_segment_text": str(meta.get("original_segment_text") or hit.get("document") or ""),
        }
    source_id = str(metadata.get("source_segment_id") or meta.get("source_ref") or meta.get("content_ref") or "")
    return source_id, source_records.get(source_id), metadata


def _add_source_score(
    candidates: dict[str, dict[str, Any]],
    *,
    record: dict[str, Any],
    score_name: str,
    score_value: float,
    retriever_name: str,
    chunk_id: str = "",
) -> None:
    source_id = str(record.get("source_id") or "")
    if not source_id:
        return
    candidate = candidates.setdefault(source_id, _candidate_seed(record))
    candidate[score_name] = max(float(candidate.get(score_name) or 0.0), max(0.0, float(score_value)))
    if retriever_name not in candidate["source_retrievers"]:
        candidate["source_retrievers"].append(retriever_name)
    if chunk_id and chunk_id not in candidate["source_chunk_ids"]:
        candidate["source_chunk_ids"].append(chunk_id)
    if chunk_id and not candidate["best_chunk_id"]:
        candidate["best_chunk_id"] = chunk_id


def _dense_source_candidates(
    *,
    query: str,
    benchmark_name: str,
    query_features: dict[str, Any] | None,
    vector_store: VectorStore,
    chunk_metadata_by_id: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
    limit: int,
    precomputed_hit_lists: list[list[dict[str, Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    query_variants = (
        _focused_query_variants(query, query_features)
        if benchmark_name in {"knowme", "clonemem"}
        else [query]
    )
    per_variant_limit = max(1, limit)
    if precomputed_hit_lists is not None and len(precomputed_hit_lists) == len(query_variants):
        variant_hit_lists = precomputed_hit_lists
    elif hasattr(vector_store, "search_many"):
        variant_hit_lists = vector_store.search_many(query_variants, top_k=per_variant_limit)
    else:
        variant_hit_lists = [vector_store.search(query_variant, top_k=per_variant_limit) for query_variant in query_variants]
    for variant_index, (query_variant, hits) in enumerate(zip(query_variants, variant_hit_lists), start=1):
        variant_penalty = 1.0 if variant_index == 1 else max(0.72, 1.0 - (variant_index - 1) * 0.08)
        for rank, hit in enumerate(hits, start=1):
            source_id, record, metadata = _map_chunk_hit_to_source(hit, chunk_metadata_by_id, source_records)
            if not source_id or record is None:
                continue
            score = min(
                1.0,
                (
                    float(hit.get("similarity") or 0.0) * 0.88
                    + (1.0 / (10.0 + rank)) * 0.6
                )
                * variant_penalty,
            )
            _add_source_score(
                candidates,
                record=record,
                score_name="dense_score",
                score_value=score,
                retriever_name="dense",
                chunk_id=str(metadata.get("chunk_id") or ""),
            )
    return candidates


def _bm25_similarity(raw_score: float) -> float:
    value = max(0.0, float(raw_score))
    if value <= 0.0:
        return 0.0
    return min(1.0, value / (value + 3.0))


def _lexical_source_candidates(
    *,
    query: str,
    benchmark_name: str,
    query_features: dict[str, Any] | None,
    storage: Storage,
    chunk_metadata_by_id: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
    limit: int,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    anchor_probe = " ".join(list(query_features.get("anchor_terms") or [])[:10]) if query_features else ""
    anchor_terms = list(query_features.get("anchor_terms") or []) if query_features else []
    specific_terms = _query_specific_terms(query_features) if query_features else []
    entities = list(query_features.get("entities") or []) if query_features else []
    temporal_terms = list(query_features.get("specific_temporal_terms") or query_features.get("temporal_terms") or []) if query_features else []
    question_time_terms = list(query_features.get("question_time_terms") or []) if query_features else []
    code_like_terms = list(query_features.get("code_like_terms") or []) if query_features else []
    metric_like_terms = list(query_features.get("metric_like_terms") or []) if query_features else []
    person_state_phrases = list(query_features.get("person_state_phrases") or []) if query_features else []
    phrases = list(query_features.get("phrases") or []) if query_features else []
    surface_terms = list(
        dict.fromkeys(
            code_like_terms
            + metric_like_terms
            + specific_terms
            + entities
            + temporal_terms
            + question_time_terms
            + anchor_terms[:12]
        )
    )
    cjk_terms = list(query_features.get("cjk_terms") or []) if query_features else []
    query_variants = _focused_query_variants(query, query_features) if benchmark_name in {"knowme", "clonemem"} else [query]
    phrase_probes = list(dict.fromkeys(person_state_phrases + phrases))[:8]
    for query_variant in query_variants:
        for row in storage.search_chunks_fts(query_variant, limit=max(1, limit)):
            chunk_id = str(row.get("chunk_id") or "")
            source_meta = dict(chunk_metadata_by_id.get(chunk_id) or {})
            source_id = str(source_meta.get("source_segment_id") or row.get("source_ref") or row.get("content_ref") or "")
            record = source_records.get(source_id)
            if not source_id or record is None:
                continue
            text = str(row.get("text") or "")
            normalized_text = str(record.get("normalized_text") or "")
            bm25 = _bm25_similarity(float(row.get("bm25_score") or 0.0))
            lexical_variant = lexical_score(query_variant, text)
            lexical_anchor = lexical_score(anchor_probe, text) if anchor_probe else 0.0
            anchor_overlap = _record_anchor_overlap(
                record,
                anchor_terms=anchor_terms,
                specific_terms=specific_terms,
                phrases=phrases,
                anchor_probe=anchor_probe,
                fast=benchmark_name in {"knowme", "clonemem"},
            )
            surface_overlap = _weighted_surface_overlap(surface_terms, record=record, normalized_text=normalized_text)
            phrase_overlap = max(
                (_phrase_match_score(phrase, record) for phrase in phrase_probes if phrase),
                default=0.0,
            )
            cjk_overlap = _cjk_overlap_score(cjk_terms, text) if cjk_terms else 0.0
            score = max(
                bm25,
                lexical_variant,
                lexical_anchor,
                anchor_overlap,
                surface_overlap,
                phrase_overlap,
                cjk_overlap,
            )
            if benchmark_name in {"knowme", "clonemem"}:
                blended = min(
                    1.0,
                    bm25 * 0.22
                    + lexical_variant * 0.16
                    + lexical_anchor * 0.12
                    + anchor_overlap * 0.2
                    + surface_overlap * 0.18
                    + phrase_overlap * 0.12
                    + cjk_overlap * 0.16,
                )
                score = max(score, blended)
            _add_source_score(
                candidates,
                record=record,
                score_name="bm25_score",
                score_value=score,
                retriever_name="bm25",
                chunk_id=chunk_id,
            )
    return candidates


def _entity_source_candidates(
    *,
    benchmark_name: str = "",
    query_features: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    limit: int,
    side_index: dict[str, Any] | None = None,
    seed_candidates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    entity_set = set(query_features.get("entities") or [])
    token_set = set(query_features.get("token_set") or [])
    cjk_terms = list(query_features.get("cjk_terms") or [])
    anchor_terms = list(query_features.get("anchor_terms") or [])
    specific_terms = _query_specific_terms(query_features)
    phrases = list(query_features.get("phrases") or [])
    anchor_probe = " ".join((specific_terms or anchor_terms)[:10])
    candidate_ids: set[str] = set()
    indexed_entity_available = _side_index_has_any(side_index, "entity_to_source_ids", "entity_postings")
    indexed_token_available = _side_index_has_any(side_index, "token_to_source_ids", "lexical_postings")
    if side_index and (indexed_entity_available or indexed_token_available):
        entity_postings = side_index.get("entity_to_source_ids") or side_index.get("entity_postings") or {}
        token_postings = side_index.get("token_to_source_ids") or side_index.get("lexical_postings") or {}
        candidate_ids.update(_posting_union(entity_postings, list(entity_set) + cjk_terms, cap=max(limit * 8, 500)))
        candidate_ids.update(_posting_union(token_postings, specific_terms + anchor_terms[:16] + cjk_terms, cap=max(limit * 8, 500)))
        if len(candidate_ids) < max(8, min(limit, 40)) and seed_candidates:
            candidate_ids.update(str(source_id) for source_id in list(seed_candidates.keys())[: max(limit, 80)])
    records_iter = (
        [source_records[source_id] for source_id in sorted(candidate_ids, key=lambda sid: _stable_source_sort_key(sid, source_records, side_index)) if source_id in source_records]
        if candidate_ids
        else [] if side_index and (indexed_entity_available or indexed_token_available) else list(source_records.values())
    )
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records_iter:
        record_entities = set(record.get("entity_terms") or [])
        if entity_set:
            overlap = len(entity_set & record_entities) / max(1, len(entity_set))
        else:
            overlap = len(token_set & set(record.get("token_list") or [])) / max(1, len(token_set))
        cjk_overlap = _cjk_overlap_score(cjk_terms, str(record.get("normalized_text") or ""))
        anchor_overlap = _record_anchor_overlap(
            record,
            anchor_terms=anchor_terms,
            specific_terms=specific_terms,
            phrases=phrases,
            anchor_probe=anchor_probe,
            fast=benchmark_name in {"knowme", "clonemem"},
        )
        overlap = max(overlap, cjk_overlap)
        if overlap <= 0.0:
            continue
        lexical_overlap = max(
            lexical_score(" ".join(sorted(entity_set or token_set)), record.get("text") or ""),
            lexical_score(anchor_probe, record.get("text") or "") if anchor_probe else 0.0,
        )
        if entity_set:
            score = min(
                1.0,
                overlap * 0.58
                + anchor_overlap * 0.24
                + lexical_overlap * 0.18
                + cjk_overlap * 0.28,
            )
            if specific_terms and anchor_overlap < 0.15 and lexical_overlap < 0.08:
                score *= 0.72
        else:
            score = min(
                1.0,
                overlap * 0.72
                + anchor_overlap * 0.24
                + lexical_overlap * 0.24
                + cjk_overlap * 0.28,
            )
        scored.append((score, record))
    scored.sort(key=lambda item: (-item[0], *_stable_source_sort_key(str(item[1].get("source_id") or ""), source_records, side_index)))
    for score, record in scored[: max(1, limit)]:
        _add_source_score(candidates, record=record, score_name="entity_score", score_value=score, retriever_name="entity")
    return candidates


def _temporal_source_candidates(
    *,
    benchmark_name: str,
    query_features: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    limit: int,
    side_index: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    temporal_terms = set(query_features.get("temporal_terms") or [])
    specific_temporal_terms = set(query_features.get("specific_temporal_terms") or [])
    question_time = str(query_features.get("question_time") or "")
    temporal_direction = str(query_features.get("temporal_direction") or "unspecified")
    if not temporal_terms and not specific_temporal_terms and not question_time:
        return candidates
    candidate_ids: set[str] = set()
    indexed_temporal_available = _side_index_has_any(
        side_index,
        "specific_temporal_term_to_source_ids",
        "temporal_term_to_source_ids",
        "timestamp_source_ids",
    )
    if side_index and indexed_temporal_available:
        candidate_ids.update(
            _posting_union(
                side_index.get("specific_temporal_term_to_source_ids") or {},
                specific_temporal_terms,
                cap=max(limit * 8, 500),
            )
        )
        candidate_ids.update(
            _posting_union(
                side_index.get("temporal_term_to_source_ids") or {},
                temporal_terms,
                cap=max(limit * 8, 500),
            )
        )
        if not candidate_ids and question_time:
            candidate_ids.update(str(source_id) for source_id in list(side_index.get("timestamp_source_ids") or []))
    records_iter = (
        [source_records[source_id] for source_id in sorted(candidate_ids, key=lambda sid: _stable_source_sort_key(sid, source_records, side_index)) if source_id in source_records]
        if candidate_ids
        else [] if side_index and indexed_temporal_available else list(source_records.values())
    )
    ordered_records = sorted(records_iter, key=lambda row: int(row.get("order_index") or 0))
    max_order_index = max((int(row.get("order_index") or 0) for row in ordered_records), default=0)
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records_iter:
        record_terms = set(record.get("temporal_terms") or [])
        specific_record_terms = set(record.get("specific_temporal_terms") or [])
        overlap = len(temporal_terms & record_terms) / max(1, len(temporal_terms))
        specific_overlap = (
            len(specific_temporal_terms & specific_record_terms) / max(1, len(specific_temporal_terms))
            if specific_temporal_terms
            else 0.0
        )
        timestamp_blob = " ".join(
            filter(
                None,
                [
                    str(record.get("timestamp") or ""),
                    str(record.get("text") or "")[:160],
                ],
            )
        )
        anchor_score = _temporal_anchor_score(question_time, str(record.get("timestamp") or ""))
        order_index = int(record.get("order_index") or 0)
        if max_order_index > 0 and temporal_direction == "newer":
            sequence_score = min(1.0, max(0.0, order_index / max_order_index))
        elif max_order_index > 0 and temporal_direction == "older":
            sequence_score = min(1.0, max(0.0, 1.0 - (order_index / max_order_index)))
        else:
            sequence_score = 0.0
        if benchmark_name in {"knowme", "clonemem"}:
            generic_overlap = overlap if not specific_temporal_terms else max(0.0, overlap - specific_overlap)
            score = max(
                specific_overlap,
                generic_overlap * 0.35,
                lexical_score(" ".join(sorted(specific_temporal_terms or temporal_terms)), timestamp_blob),
                anchor_score * (0.42 if benchmark_name == "clonemem" else 0.24),
                sequence_score * (0.34 if benchmark_name == "clonemem" else 0.2),
            )
        else:
            score = max(overlap, lexical_score(" ".join(sorted(temporal_terms)), timestamp_blob), anchor_score * 0.18)
        if score <= 0.0:
            continue
        scored.append((min(1.0, score), record))
    scored.sort(key=lambda item: (-item[0], *_stable_source_sort_key(str(item[1].get("source_id") or ""), source_records, side_index)))
    for score, record in scored[: max(1, limit)]:
        _add_source_score(
            candidates,
            record=record,
            score_name="temporal_score",
            score_value=score,
            retriever_name="temporal",
        )
    return candidates


def _profile_source_candidates(
    *,
    query: str,
    query_features: dict[str, Any],
    vector_store: VectorStore,
    storage: Storage,
    chunk_metadata_by_id: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
    limit: int,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    if not hasattr(vector_store, "search_objects"):
        return candidates
    preferred_object_types = [
        "preference",
        "personal_context",
        "relation",
        "state_update",
        "temporal_reference",
        "fact",
        "entity",
    ]
    object_type_bonus = {
        "preference": 0.12,
        "personal_context": 0.1,
        "relation": 0.1,
        "state_update": 0.08,
        "temporal_reference": 0.08,
        "fact": 0.06,
        "entity": 0.05,
    }
    anchor_probe = " ".join(list(query_features.get("anchor_terms") or [])[:10])
    query_entities = set(query_features.get("entities") or [])
    object_rows_by_id: dict[str, dict[str, Any]] = {}

    def resolve_source(row: dict[str, Any], metadata: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        source_chunk_id = str(row.get("source_chunk_id") or metadata.get("source_chunk_id") or "")
        source_meta = dict(chunk_metadata_by_id.get(source_chunk_id) or {})
        source_id = str(source_meta.get("source_segment_id") or row.get("source_ref") or metadata.get("source_ref") or "")
        return source_records.get(source_id), source_chunk_id

    def score_object_hit(base_score: float, row: dict[str, Any], text: str) -> float:
        object_text = " ".join(
            filter(
                None,
                [
                    str(row.get("object_text") or text or ""),
                    str(row.get("subject") or ""),
                    str(row.get("predicate") or row.get("attribute") or ""),
                    str(row.get("entity") or ""),
                    str(row.get("canonical_key") or ""),
                ],
            )
        )
        object_entities = {term.strip().lower() for term in ENTITY_RE.findall(object_text) if term.strip()}
        if row.get("entity"):
            object_entities.add(str(row.get("entity") or "").strip().lower())
        entity_overlap = len(query_entities & object_entities) / max(1, len(query_entities)) if query_entities else 0.0
        lexical = max(
            lexical_score(query, object_text),
            lexical_score(anchor_probe, object_text) if anchor_probe else 0.0,
        )
        confidence_bonus = min(1.0, max(float(row.get("confidence") or 0.0), 0.0)) * 0.08
        type_bonus = object_type_bonus.get(str(row.get("object_type") or "").lower(), 0.04)
        return min(1.0, base_score * 0.7 + lexical * 0.24 + entity_overlap * 0.18 + confidence_bonus + type_bonus)

    dense_hits = vector_store.search_objects(query, top_k=max(1, limit))
    dense_object_ids = [str(hit.get("object_id") or "") for hit in dense_hits if str(hit.get("object_id") or "")]
    if dense_object_ids:
        object_rows_by_id.update(
            {
                str(row.get("object_id") or ""): row
                for row in storage.fetch_objects_by_ids(dense_object_ids)
                if str(row.get("object_id") or "")
            }
        )
    for hit in dense_hits:
        metadata = dict(hit.get("metadata") or {})
        row = dict(object_rows_by_id.get(str(hit.get("object_id") or "")) or {})
        record, source_chunk_id = resolve_source(row, metadata)
        if record is None:
            continue
        score = score_object_hit(float(hit.get("similarity") or 0.0), row, str(hit.get("document") or ""))
        _add_source_score(
            candidates,
            record=record,
            score_name="profile_score",
            score_value=score,
            retriever_name="profile_fact",
            chunk_id=source_chunk_id,
        )
    for query_variant in _focused_query_variants(query, query_features):
        for row in storage.search_objects_fts(query_variant, limit=max(1, limit), object_types=preferred_object_types):
            record, source_chunk_id = resolve_source(row, {})
            if record is None:
                continue
            probe_text = " ".join(
                filter(
                    None,
                    [
                        str(row.get("object_text") or ""),
                        str(row.get("subject") or ""),
                        str(row.get("predicate") or row.get("attribute") or ""),
                    ],
                )
            )
            lexical = max(
                lexical_score(query_variant, probe_text),
                lexical_score(anchor_probe, probe_text) if anchor_probe else 0.0,
            )
            score = score_object_hit(lexical, row, probe_text)
            _add_source_score(
                candidates,
                record=record,
                score_name="profile_score",
                score_value=score,
                retriever_name="profile_fact",
                chunk_id=source_chunk_id,
            )
    return candidates


def _phrase_match_score(phrase: str, record: dict[str, Any]) -> float:
    phrase_tokens = [token for token in _content_terms(phrase) if len(token) >= 2]
    if not phrase_tokens:
        return 0.0
    record_tokens = set(record.get("token_list") or [])
    normalized_text = str(record.get("normalized_text") or "")
    if _contains_cjk(phrase):
        phrase_overlap = _cjk_overlap_score(_cjk_subterms(phrase, max_terms=32), normalized_text)
        if phrase and normalize_text_for_hash(phrase).lower() in normalized_text:
            return max(0.95, phrase_overlap)
        return max(phrase_overlap, lexical_score(phrase, normalized_text))
    coverage = len(set(phrase_tokens) & record_tokens) / max(1, len(set(phrase_tokens)))
    lexical = lexical_score(phrase, normalized_text)
    if len(phrase_tokens) == 1:
        token = phrase_tokens[0]
        return max(lexical * 0.55, 0.35 if token in record_tokens else 0.0)
    contiguous = " ".join(phrase_tokens) in normalized_text
    if len(phrase_tokens) == 2:
        return min(1.0, max(0.78 if contiguous else 0.52 if coverage >= 1.0 else 0.0, lexical * 0.9))
    return min(1.0, max(0.92 if contiguous else 0.66 if coverage >= 0.8 else 0.0, lexical, coverage * 0.85))


def _token_list_overlap_score(probe_terms: list[str], record: dict[str, Any]) -> float:
    terms = {str(term or "").strip().lower() for term in probe_terms if str(term or "").strip()}
    if not terms:
        return 0.0
    record_tokens = set(record.get("token_list") or [])
    if not record_tokens:
        return 0.0
    return len(terms & record_tokens) / max(1, len(terms))


def _fast_phrase_match_score(phrase: str, phrase_tokens: list[str], record: dict[str, Any]) -> float:
    if not phrase_tokens:
        return 0.0
    normalized_text = str(record.get("normalized_text") or "")
    if _contains_cjk(phrase):
        phrase_overlap = _cjk_overlap_score(_cjk_subterms(phrase, max_terms=32), normalized_text)
        if phrase and normalize_text_for_hash(phrase).lower() in normalized_text:
            return max(0.95, phrase_overlap)
        return phrase_overlap
    record_tokens = set(record.get("token_list") or [])
    coverage = len(set(phrase_tokens) & record_tokens) / max(1, len(set(phrase_tokens)))
    if len(phrase_tokens) == 1:
        return 0.35 if phrase_tokens[0] in record_tokens else 0.0
    contiguous = " ".join(phrase_tokens) in normalized_text
    if len(phrase_tokens) == 2:
        return 0.78 if contiguous else 0.52 if coverage >= 1.0 else 0.0
    return min(1.0, max(0.92 if contiguous else 0.66 if coverage >= 0.8 else 0.0, coverage * 0.85))


def _exact_phrase_source_candidates(
    *,
    query: str,
    benchmark_name: str = "",
    query_features: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    limit: int,
    side_index: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    phrases = list(query_features.get("phrases") or [])
    query_text_hash = str(query_features.get("query_text_hash") or "")
    if not phrases:
        return candidates
    anchor_terms = list(query_features.get("anchor_terms") or [])
    anchor_probe = " ".join(anchor_terms[-min(8, len(anchor_terms)) :]) if anchor_terms else ""
    use_fast_phrase = benchmark_name in {"clonemem", "knowme"}
    anchor_probe_terms = [token for token in _content_terms(anchor_probe) if len(token) >= 2] if anchor_probe else []
    phrase_token_cache = {
        phrase: [token for token in _content_terms(phrase) if len(token) >= 2]
        for phrase in phrases
        if phrase
    }
    candidate_ids: set[str] = set()
    indexed_phrase_available = _side_index_has_any(
        side_index,
        "phrase_token_to_source_ids",
        "token_to_source_ids",
        "lexical_postings",
        "text_hash_to_source_id",
    )
    if side_index and indexed_phrase_available:
        text_hash_hit = str((side_index.get("text_hash_to_source_id") or {}).get(query_text_hash) or "")
        if text_hash_hit:
            candidate_ids.add(text_hash_hit)
        phrase_postings = side_index.get("phrase_token_to_source_ids") or side_index.get("token_to_source_ids") or side_index.get("lexical_postings") or {}
        for phrase_tokens in phrase_token_cache.values():
            candidate_ids.update(_posting_phrase_candidates(phrase_tokens, phrase_postings, cap=max(limit * 10, 600)))
        if not candidate_ids:
            candidate_ids.update(_posting_union(phrase_postings, anchor_terms[:12], cap=max(limit * 8, 500)))
    records_iter = (
        [source_records[source_id] for source_id in sorted(candidate_ids, key=lambda sid: _stable_source_sort_key(sid, source_records, side_index)) if source_id in source_records]
        if candidate_ids
        else [] if side_index and indexed_phrase_available else list(source_records.values())
    )
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records_iter:
        if query_text_hash and query_text_hash == str(record.get("text_hash") or ""):
            scored.append((1.0, record))
            continue
        if use_fast_phrase:
            phrase_scores = sorted(
                (
                    _fast_phrase_match_score(phrase, phrase_tokens, record)
                    for phrase, phrase_tokens in phrase_token_cache.items()
                ),
                reverse=True,
            )
        else:
            phrase_scores = sorted((_phrase_match_score(phrase, record) for phrase in phrases if phrase), reverse=True)
        if not phrase_scores:
            continue
        best = phrase_scores[0]
        support = phrase_scores[1] if len(phrase_scores) > 1 else 0.0
        anchor_overlap = (
            len(set(anchor_terms) & set(record.get("token_list") or [])) / max(1, min(8, len(set(anchor_terms))))
            if anchor_terms
            else 0.0
        )
        score = min(
            1.0,
            best * 0.72
            + support * 0.18
            + anchor_overlap * 0.18
            + (
                _token_list_overlap_score(anchor_probe_terms, record)
                if use_fast_phrase and anchor_probe_terms
                else lexical_score(anchor_probe, str(record.get("normalized_text") or "")) if anchor_probe else 0.0
            )
            * 0.12,
        )
        if len(phrases) == 1 and len(_content_terms(phrases[0])) == 1:
            score = min(score, 0.58)
        if score <= 0.0:
            continue
        scored.append((score, record))
    scored.sort(key=lambda item: (-item[0], *_stable_source_sort_key(str(item[1].get("source_id") or ""), source_records, side_index)))
    for score, record in scored[: max(1, limit)]:
        _add_source_score(
            candidates,
            record=record,
            score_name="exact_phrase_score",
            score_value=score,
            retriever_name="exact_phrase",
        )
    return candidates


def _session_bundle_source_candidates(
    *,
    seed_candidates: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
    limit: int,
    side_index: dict[str, Any] | None = None,
    query_features: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    if not seed_candidates:
        return candidates
    query_features = query_features or {}
    anchor_terms = list(query_features.get("anchor_terms") or [])
    phrases = list(query_features.get("phrases") or [])
    specific_terms = _query_specific_terms(query_features)
    anchor_probe = " ".join((specific_terms or anchor_terms)[:10])
    query_entities = set(query_features.get("entities") or [])
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if side_index and side_index.get("session_id_to_source_ids"):
        seed_session_ids = {
            str(seed.get("session_id") or (side_index.get("source_id_to_session_id") or {}).get(str(seed.get("source_id") or "")) or "")
            for seed in seed_candidates.values()
        }
        for session_id in seed_session_ids:
            if not session_id:
                continue
            source_ids = list((side_index.get("session_id_to_source_ids") or {}).get(session_id) or [])
            bounded_ids: set[str] = set()
            seed_orders = [
                int(seed.get("order_index") or 0)
                for seed in seed_candidates.values()
                if str(seed.get("session_id") or "") == session_id
            ]
            for source_id in source_ids[: max(limit * 4, 240)]:
                bounded_ids.add(str(source_id))
            for seed_order in seed_orders:
                for source_id in source_ids[max(0, seed_order - 6) : seed_order + 7]:
                    bounded_ids.add(str(source_id))
            by_session[session_id] = [
                source_records[source_id]
                for source_id in sorted(bounded_ids, key=lambda sid: _stable_source_sort_key(sid, source_records, side_index))
                if source_id in source_records
            ]
    else:
        for record in source_records.values():
            session_id = str(record.get("session_id") or "")
            if session_id:
                by_session[session_id].append(record)
        for rows in by_session.values():
            rows.sort(key=lambda item: int(item.get("order_index") or 0))
    session_anchor_max: dict[str, float] = {}
    session_entity_max: dict[str, float] = {}
    for session_id, rows in by_session.items():
        session_anchor_max[session_id] = max(
            (
                _record_anchor_overlap(
                    row,
                    anchor_terms=anchor_terms,
                    specific_terms=specific_terms,
                    phrases=phrases,
                    anchor_probe=anchor_probe,
                )
                for row in rows
            ),
            default=0.0,
        )
        session_entity_max[session_id] = max(
            (
                len(query_entities & set(row.get("entity_terms") or [])) / max(1, len(query_entities))
                for row in rows
            ),
            default=0.0,
        )
    scored: list[tuple[float, dict[str, Any]]] = []
    seed_rows = sorted(seed_candidates.values(), key=_candidate_seed_confidence, reverse=True)[:12]
    for seed in seed_rows:
        session_id = str(seed.get("session_id") or "")
        if not session_id or session_id not in by_session:
            continue
        seed_strength = _candidate_seed_confidence(seed)
        if seed_strength <= 0.0:
            continue
        seed_order = int(seed.get("order_index") or 0)
        support_count = sum(
            1
            for key in ("dense_score", "bm25_score", "entity_score", "temporal_score")
            if float(seed.get(key) or 0.0) > 0.0
        )
        chunk_support = min(1.0, len(list(seed.get("source_chunk_ids") or [])) / 6.0)
        session_rows = by_session[session_id]
        session_coherence = max(float(session_anchor_max.get(session_id) or 0.0), float(session_entity_max.get(session_id) or 0.0) * 0.82)
        if len(session_rows) <= 1:
            score = min(
                0.4,
                0.08
                + seed_strength * 0.18
                + session_coherence * 0.1
                + max(0, support_count - 1) * 0.03
                + chunk_support * 0.04,
            )
            if score > 0.0:
                scored.append((score, session_rows[0]))
            continue
        for record in session_rows:
            distance = abs(int(record.get("order_index") or 0) - seed_order)
            anchor_overlap = _record_anchor_overlap(
                record,
                anchor_terms=anchor_terms,
                specific_terms=specific_terms,
                phrases=phrases,
                anchor_probe=anchor_probe,
            )
            entity_overlap = (
                len(query_entities & set(record.get("entity_terms") or [])) / max(1, len(query_entities))
                if query_entities
                else 0.0
            )
            if distance > 4 and anchor_overlap < 0.22 and entity_overlap < 0.55:
                continue
            proximity = max(0.0, 0.42 - distance * 0.09) if distance <= 4 else 0.06
            score = (
                seed_strength * proximity
                + session_coherence * 0.18
                + anchor_overlap * 0.24
                + entity_overlap * 0.12
                + max(0, support_count - 1) * 0.03
                + chunk_support * 0.04
            )
            if distance == 0:
                score += 0.04
            if score <= 0.0:
                continue
            scored.append((min(0.52, score), record))
    scored.sort(key=lambda item: (-item[0], *_stable_source_sort_key(str(item[1].get("source_id") or ""), source_records, side_index)))
    seen: set[str] = set()
    for score, record in scored:
        source_id = str(record.get("source_id") or "")
        if source_id in seen:
            continue
        seen.add(source_id)
        _add_source_score(
            candidates,
            record=record,
            score_name="session_score",
            score_value=score,
            retriever_name="session_bundle",
        )
        if len(seen) >= max(1, limit):
            break
    return candidates


def _local_window_source_candidates(
    *,
    seed_candidates: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
    limit: int,
    window_radius: int = 4,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    if not seed_candidates:
        return candidates
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_records.values():
        doc_id = str(record.get("source_doc_id") or "")
        if doc_id:
            by_doc[doc_id].append(record)
    for rows in by_doc.values():
        rows.sort(key=lambda item: int(item.get("order_index") or 0))

    scored: list[tuple[float, dict[str, Any]]] = []
    seed_rows = sorted(seed_candidates.values(), key=_candidate_seed_confidence, reverse=True)[:12]
    for seed in seed_rows:
        doc_id = str(seed.get("source_doc_id") or "")
        if not doc_id or doc_id not in by_doc:
            continue
        seed_order = int(seed.get("order_index") or 0)
        seed_strength = _candidate_seed_confidence(seed)
        if seed_strength <= 0.0:
            continue
        for record in by_doc[doc_id]:
            distance = abs(int(record.get("order_index") or 0) - seed_order)
            if distance == 0 or distance > window_radius:
                continue
            score = seed_strength * max(0.0, 0.82 - distance * 0.14)
            if score <= 0.0:
                continue
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    for score, record in scored:
        source_id = str(record.get("source_id") or "")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        _add_source_score(
            candidates,
            record=record,
            score_name="local_window_score",
            score_value=score,
            retriever_name="local_window",
        )
        if len(seen) >= max(1, limit):
            break
    return candidates


def _bm25_side_index_score(
    *,
    query_terms: list[str],
    source_id: str,
    side_index: dict[str, Any],
) -> float:
    term_freqs = dict((side_index.get("source_term_freqs") or {}).get(source_id) or {})
    if not term_freqs:
        return 0.0
    postings = side_index.get("lexical_postings") or {}
    doc_lengths = side_index.get("doc_lengths") or {}
    doc_length = max(1, int(doc_lengths.get(source_id) or 1))
    avg_doc_length = max(1.0, float(side_index.get("avg_doc_length") or 1.0))
    document_count = max(1, int(side_index.get("document_count") or 1))
    score = 0.0
    for term in list(dict.fromkeys(query_terms or [])):
        tf = int(term_freqs.get(term) or 0)
        if tf <= 0:
            continue
        df = len(list(postings.get(term) or []))
        idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
        denominator = tf + 1.2 * (1.0 - 0.75 + 0.75 * (doc_length / avg_doc_length))
        score += idf * ((tf * (1.2 + 1.0)) / max(1e-6, denominator))
    return _bm25_similarity(score)


def _term_overlap_score(query_terms: list[str], candidate_terms: list[str]) -> float:
    query_set = {term for term in query_terms if term}
    candidate_set = {term for term in candidate_terms if term}
    if not query_set or not candidate_set:
        return 0.0
    return len(query_set & candidate_set) / max(1, len(query_set))


def _stable_source_sort_key(source_id: str, source_records: dict[str, dict[str, Any]], side_index: dict[str, Any] | None = None) -> tuple[int, str]:
    order_map = dict((side_index or {}).get("source_id_to_order_index") or {})
    record = source_records.get(source_id) or {}
    return int(order_map.get(source_id, record.get("order_index") or 0)), source_id


def _posting_union(
    postings: dict[str, Any],
    terms: list[str] | set[str],
    *,
    max_terms: int = 48,
    cap: int = 1200,
) -> set[str]:
    source_ids: set[str] = set()
    for term in list(dict.fromkeys(str(term or "").strip().lower() for term in list(terms or []) if str(term or "").strip()))[:max_terms]:
        for source_id in list(postings.get(term) or []):
            if source_id:
                source_ids.add(str(source_id))
                if len(source_ids) >= cap:
                    return source_ids
    return source_ids


def _side_index_has_any(side_index: dict[str, Any] | None, *keys: str) -> bool:
    if not side_index:
        return False
    return any(bool(side_index.get(key)) for key in keys)


def _posting_phrase_candidates(
    phrase_tokens: list[str],
    postings: dict[str, Any],
    *,
    cap: int = 1200,
) -> set[str]:
    token_postings = [set(str(source_id) for source_id in list(postings.get(token) or [])) for token in phrase_tokens if token]
    if not token_postings:
        return set()
    token_postings.sort(key=len)
    if len(token_postings) <= 3:
        candidate_ids = set.intersection(*token_postings) if token_postings else set()
        if candidate_ids:
            return set(list(candidate_ids)[:cap])
    coverage: Counter[str] = Counter()
    for ids in token_postings[:8]:
        coverage.update(ids)
    threshold = max(1, min(len(token_postings), math.ceil(len(token_postings) * 0.65)))
    return {source_id for source_id, count in coverage.most_common(cap) if count >= threshold}


def _query_decomposition_source_candidates(
    *,
    benchmark_name: str = "",
    decomposition: dict[str, Any],
    query: str,
    source_records: dict[str, dict[str, Any]],
    side_index: dict[str, Any],
    limit: int,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    score_rows: list[tuple[float, dict[str, Any]]] = []
    query_terms = list(
        dict.fromkeys(
            list(decomposition.get("entity_terms") or [])
            + list(decomposition.get("action_terms") or [])
            + list(decomposition.get("object_terms") or [])
            + list(decomposition.get("attribute_terms") or [])
            + list(decomposition.get("time_terms") or [])
            + list(decomposition.get("metric_terms") or [])
            + list(decomposition.get("constraint_terms") or [])
        )
    )
    fast_clause_overlap = benchmark_name in {"clonemem", "knowme"}
    clause_token_cache = {
        str(clause): [token for token in _content_terms(str(clause)) if len(token) >= 2]
        for clause in list(decomposition.get("clauses") or [])
        if str(clause).strip()
    }
    indexed_query_terms_available = _side_index_has_any(side_index, "token_to_source_ids", "lexical_postings", "entity_to_source_ids", "entity_postings")
    postings = side_index.get("token_to_source_ids") or side_index.get("lexical_postings") or {}
    entity_postings = side_index.get("entity_to_source_ids") or side_index.get("entity_postings") or {}
    candidate_ids = _posting_union(postings, query_terms, cap=max(limit * 10, 900))
    candidate_ids.update(_posting_union(entity_postings, list(decomposition.get("entity_terms") or []), cap=max(limit * 8, 700)))
    for clause_tokens in clause_token_cache.values():
        candidate_ids.update(_posting_phrase_candidates(clause_tokens, postings, cap=max(limit * 6, 500)))
    iterable_ids = (
        sorted(candidate_ids, key=lambda sid: _stable_source_sort_key(sid, source_records, side_index))
        if candidate_ids
        else [] if indexed_query_terms_available else list(source_records.keys())
    )
    for source_id in iterable_ids:
        record = source_records.get(source_id)
        if record is None:
            continue
        lexical = _bm25_side_index_score(query_terms=query_terms, source_id=source_id, side_index=side_index)
        entity_overlap = _term_overlap_score(list(decomposition.get("entity_terms") or []), list(record.get("entity_terms") or []))
        attribute_overlap = _term_overlap_score(list(decomposition.get("attribute_terms") or []), _record_anchor_terms(record))
        temporal_overlap = _term_overlap_score(list(decomposition.get("time_terms") or []), list(record.get("specific_temporal_terms") or record.get("temporal_terms") or []))
        if fast_clause_overlap:
            clause_overlap = max(
                (
                    _fast_phrase_match_score(clause, clause_tokens, record)
                    for clause, clause_tokens in clause_token_cache.items()
                ),
                default=0.0,
            )
        else:
            clause_overlap = max(
                (lexical_score(clause, str(record.get("text") or "")) for clause in list(decomposition.get("clauses") or [])),
                default=0.0,
            )
        score = min(
            1.0,
            lexical * 0.34
            + entity_overlap * 0.24
            + attribute_overlap * 0.2
            + temporal_overlap * 0.12
            + clause_overlap * 0.14,
        )
        if str(decomposition.get("evidence_type") or "") in _infer_profile_categories(record):
            score += 0.08
        if score <= 0.0:
            continue
        score_rows.append((min(1.0, score), record))
    score_rows.sort(key=lambda item: (-item[0], *_stable_source_sort_key(str(item[1].get("source_id") or ""), source_records, side_index)))
    for score, record in score_rows[: max(1, limit)]:
        _add_source_score(
            candidates,
            record=record,
            score_name="decomposition_score",
            score_value=score,
            retriever_name="query_decomposition",
        )
    return candidates


def _profile_side_index_candidates(
    *,
    decomposition: dict[str, Any],
    query: str,
    side_index: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    limit: int,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    scored: list[tuple[float, dict[str, Any]]] = []
    evidence_type = str(decomposition.get("evidence_type") or "")
    entry_by_id = side_index.get("profile_entry_by_id") or {}
    entry_ids_by_term = side_index.get("profile_entry_ids_by_term") or {}
    entry_ids_by_category = side_index.get("profile_entry_ids_by_category") or {}
    probe_terms = list(
        dict.fromkeys(
            [
                str(term or "").strip().lower()
                for term in (
                    list(decomposition.get("entity_terms") or [])
                    + list(decomposition.get("object_terms") or [])
                    + list(decomposition.get("attribute_terms") or [])
                    + list(decomposition.get("time_terms") or [])
                    + _content_terms(query)[:32]
                )
                if str(term or "").strip()
            ]
        )
    )
    candidate_entry_ids: set[str] = set()
    for term in probe_terms[:48]:
        for entry_id in list(entry_ids_by_term.get(term) or [])[:800]:
            candidate_entry_ids.add(str(entry_id))
    if evidence_type:
        for entry_id in list(entry_ids_by_category.get(evidence_type) or [])[: max(400, limit * 6)]:
            candidate_entry_ids.add(str(entry_id))
    if entry_by_id:
        min_probe_count = max(limit * 3, 96)
        if len(candidate_entry_ids) < min_probe_count:
            if evidence_type:
                for entry_id in list(entry_ids_by_category.get(evidence_type) or [])[:min_probe_count]:
                    candidate_entry_ids.add(str(entry_id))
        entries_iter = [
            dict(entry_by_id.get(entry_id) or {})
            for entry_id in sorted(candidate_entry_ids)
            if entry_by_id.get(entry_id)
        ]
    else:
        entries_iter = list(side_index.get("profile_side_entries") or [])
    for entry in entries_iter:
        source_id = str(entry.get("source_id") or "")
        record = source_records.get(source_id)
        if not source_id or record is None:
            continue
        entity_overlap = _term_overlap_score(list(decomposition.get("entity_terms") or []), list(entry.get("entity_terms") or []))
        attribute_overlap = _term_overlap_score(
            list(decomposition.get("object_terms") or decomposition.get("attribute_terms") or []),
            list(entry.get("attribute_terms") or []),
        )
        temporal_overlap = _term_overlap_score(list(decomposition.get("time_terms") or []), list(entry.get("temporal_terms") or []))
        lexical = lexical_score(query, str(entry.get("text") or ""))
        category_bonus = 0.16 if evidence_type and evidence_type == str(entry.get("category") or "") else 0.0
        score = min(1.0, lexical * 0.34 + attribute_overlap * 0.24 + entity_overlap * 0.22 + temporal_overlap * 0.12 + category_bonus)
        if score <= 0.0:
            continue
        scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    for score, record in scored[: max(1, limit)]:
        _add_source_score(
            candidates,
            record=record,
            score_name="profile_score",
            score_value=score,
            retriever_name="profile_side_index",
        )
    return candidates


def _temporal_neighbor_source_candidates(
    *,
    benchmark_name: str = "",
    seed_candidates: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
    side_index: dict[str, Any],
    query_features: dict[str, Any],
    max_neighbors_per_seed: int,
    limit: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    audit = {
        "seed_count": 0,
        "expanded_candidate_count": 0,
        "neighbor_hit_but_gold_miss_count": 0,
    }
    if not seed_candidates:
        return candidates, audit
    neighbor_map = side_index.get("neighbor_map") or {}
    seed_rows = sorted(seed_candidates.values(), key=_candidate_seed_confidence, reverse=True)[:12]
    query_entities = set(query_features.get("entities") or [])
    audit["seed_count"] = len(seed_rows)
    score_rows: list[tuple[float, dict[str, Any]]] = []
    for seed in seed_rows:
        seed_id = str(seed.get("source_id") or "")
        if not seed_id:
            continue
        neighbors = dict(neighbor_map.get(seed_id) or {})
        ordered_neighbors = list(neighbors.get("prev") or [])[-max_neighbors_per_seed:] + list(neighbors.get("next") or [])[:max_neighbors_per_seed]
        seed_strength = _candidate_seed_confidence(seed)
        for offset, neighbor_id in enumerate(ordered_neighbors, start=1):
            record = source_records.get(str(neighbor_id))
            if record is None:
                continue
            proximity = max(0.0, 0.88 - offset * 0.14)
            entity_overlap = (
                len(query_entities & set(record.get("entity_terms") or [])) / max(1, len(query_entities))
                if query_entities
                else 0.0
            )
            anchor_overlap = _record_anchor_overlap(
                record,
                anchor_terms=list(query_features.get("anchor_terms") or []),
                specific_terms=_query_specific_terms(query_features),
                phrases=list(query_features.get("phrases") or []),
                anchor_probe=" ".join(list(query_features.get("anchor_terms") or [])[:10]),
                fast=benchmark_name in {"knowme", "clonemem"},
            )
            score = min(1.0, seed_strength * 0.46 + proximity * 0.28 + entity_overlap * 0.14 + anchor_overlap * 0.12)
            if score <= 0.0:
                continue
            score_rows.append((score, record))
    score_rows.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    for score, record in score_rows:
        source_id = str(record.get("source_id") or "")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        candidate = candidates.setdefault(source_id, _candidate_seed(record))
        candidate["temporal_neighbor_score"] = max(float(candidate.get("temporal_neighbor_score") or 0.0), float(score))
        candidate["local_window_score"] = max(float(candidate.get("local_window_score") or 0.0), float(score))
        if "temporal_neighbor" not in candidate["source_retrievers"]:
            candidate["source_retrievers"].append("temporal_neighbor")
        if len(seen) >= max(1, limit):
            break
    audit["expanded_candidate_count"] = len(candidates)
    return candidates, audit


def _parent_session_source_candidates(
    *,
    benchmark_name: str,
    query: str,
    query_features: dict[str, Any],
    decomposition: dict[str, Any] | None,
    seed_candidates: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
    side_index: dict[str, Any],
    parent_top_k: int,
    parent_expand_segments: int,
    parent_window_radius: int,
    parent_anchor_noise_filter_enabled: bool = True,
    parent_supplemental_anchor_expansion_enabled: bool = False,
    parent_supplemental_anchor_expansion_cap: int = 2,
    parent_timestamp_sibling_expansion_enabled: bool = False,
    parent_timestamp_sibling_expansion_cap: int = 2,
    parent_anchor_strict_noise_filter_enabled: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    parent_records = side_index.get("parent_records_by_id") or {}
    source_parent_ids = side_index.get("source_parent_ids") or {}
    lexical_postings = side_index.get("lexical_postings") or {}
    parent_seed_strengths: dict[str, float] = defaultdict(float)
    parent_seed_positions: dict[str, list[int]] = defaultdict(list)
    for seed in seed_candidates.values():
        source_id = str(seed.get("source_id") or "")
        parent_id = str(source_parent_ids.get(source_id) or _parent_id(seed))
        if not parent_id:
            continue
        parent_seed_strengths[parent_id] = max(parent_seed_strengths[parent_id], _candidate_seed_confidence(seed))
        parent_seed_positions[parent_id].append(int(seed.get("order_index") or 0))
    parent_scored: list[tuple[float, str, dict[str, Any]]] = []
    query_entities = set(query_features.get("entities") or [])
    temporal_terms = set(query_features.get("specific_temporal_terms") or query_features.get("temporal_terms") or [])
    anchor_terms = list(query_features.get("anchor_terms") or [])
    specific_terms = _query_specific_terms(query_features)
    phrases = list(query_features.get("phrases") or [])
    anchor_probe = " ".join((specific_terms or anchor_terms)[:10])
    decomposition_terms = _decomposition_surface_terms(decomposition)
    code_like_terms = list(query_features.get("code_like_terms") or [])
    metric_like_terms = list(query_features.get("metric_like_terms") or [])
    person_state_phrases = list(query_features.get("person_state_phrases") or [])
    focus_person = str(query_features.get("focus_person") or "").strip().lower()
    parent_extra_noise_terms = (
        CLONEMEM_PARENT_ANCHOR_EXTRA_NOISE_TERMS
        if benchmark_name == "clonemem" and parent_anchor_strict_noise_filter_enabled
        else None
    )
    if parent_anchor_noise_filter_enabled:
        parent_specific_terms = _parent_anchor_terms(
            specific_terms + anchor_terms + decomposition_terms,
            max_terms=32,
            extra_noise_terms=parent_extra_noise_terms,
        )
        phrase_anchor_terms = _parent_anchor_terms(
            [
                token
                for phrase in list(phrases[:8]) + list(person_state_phrases[:8])
                for token in _content_terms(str(phrase or ""))
            ],
            max_terms=24,
            extra_noise_terms=parent_extra_noise_terms,
        )
        anchor_surface_terms = list(
            dict.fromkeys(code_like_terms + metric_like_terms + parent_specific_terms[:20] + phrase_anchor_terms[:12])
        )
        anchor_probe_terms = list(
            dict.fromkeys(anchor_surface_terms + parent_specific_terms[:24] + phrase_anchor_terms[:16])
        )
        parent_probe_terms = list(dict.fromkeys(parent_specific_terms[:24] + code_like_terms[:8] + metric_like_terms[:8]))
        if not parent_probe_terms:
            parent_probe_terms = list(dict.fromkeys(anchor_terms[:12] + specific_terms + decomposition_terms[:12]))
    else:
        parent_specific_terms = list(dict.fromkeys(specific_terms + anchor_terms + decomposition_terms))
        phrase_anchor_terms = []
        anchor_surface_terms = list(dict.fromkeys(code_like_terms + metric_like_terms + specific_terms + anchor_terms[:12]))
        anchor_probe_terms = list(dict.fromkeys(anchor_surface_terms + decomposition_terms + phrases[:6] + person_state_phrases[:6]))
        parent_probe_terms = list(dict.fromkeys(anchor_terms[:12] + specific_terms + decomposition_terms[:12]))
    phrase_probes = list(dict.fromkeys(person_state_phrases + phrases))[:10]
    phrase_token_cache = {
        phrase: [token for token in _content_terms(phrase) if len(token) >= 2]
        for phrase in phrase_probes
        if phrase
    }
    selected_child_anchor_debug: list[dict[str, Any]] = []
    candidate_parent_ids = set(parent_seed_strengths.keys())
    parent_id_to_source_ids = side_index.get("parent_id_to_source_ids") or {}
    for term in parent_probe_terms[:24] + anchor_surface_terms[:24]:
        for source_id in list(lexical_postings.get(str(term or "").strip().lower()) or [])[:600]:
            parent_id = str(source_parent_ids.get(str(source_id)) or "")
            if parent_id:
                candidate_parent_ids.add(parent_id)
    parent_iter = (
        ((parent_id, parent_records[parent_id]) for parent_id in sorted(candidate_parent_ids) if parent_id in parent_records)
        if candidate_parent_ids
        else []
    )
    for parent_id, parent_record in parent_iter:
        lexical = max(
            _token_list_overlap_score(parent_probe_terms, parent_record),
            _token_list_overlap_score(anchor_terms[:10], parent_record),
        )
        entity_overlap = len(query_entities & set(parent_record.get("entity_terms") or [])) / max(1, len(query_entities)) if query_entities else 0.0
        temporal_overlap = len(temporal_terms & set(parent_record.get("temporal_terms") or [])) / max(1, len(temporal_terms)) if temporal_terms else 0.0
        seed_strength = float(parent_seed_strengths.get(parent_id) or 0.0)
        score = min(1.0, seed_strength * 0.34 + lexical * 0.3 + entity_overlap * 0.22 + temporal_overlap * 0.14)
        if score <= 0.0:
            continue
        parent_scored.append((score, parent_id, parent_record))
    parent_scored.sort(key=lambda item: item[0], reverse=True)
    top_parent_rows = parent_scored[: max(1, parent_top_k)]
    effective_window_radius = max(1, int(parent_window_radius))
    selected_parent_anchor_ids: set[str] = set()
    covered_parent_anchor_terms: set[str] = set()
    supplemental_anchor_selected_count = 0
    timestamp_sibling_selected_count = 0
    timestamp_sibling_debug: list[dict[str, Any]] = []
    prefiltered_parent_child_count = 0
    full_parent_child_count = 0
    if benchmark_name == "clonemem":
        effective_window_radius = max(effective_window_radius, 2)
    elif benchmark_name == "knowme":
        effective_window_radius = max(effective_window_radius, 2)
    anchor_preselection_enabled = benchmark_name in {"knowme", "clonemem"}
    for parent_score, parent_id, parent_record in top_parent_rows:
        source_ids = list(parent_id_to_source_ids.get(parent_id) or parent_record.get("source_ids") or [])
        seed_positions = sorted(parent_seed_positions.get(parent_id) or [])
        full_parent_child_count += len(source_ids)
        if benchmark_name in {"knowme", "clonemem"} and len(source_ids) > 900:
            source_id_set = set(source_ids)
            order_by_source_id = {source_id: index for index, source_id in enumerate(source_ids)}
            selected_source_ids: set[str] = set()
            for seed_source_id, seed in seed_candidates.items():
                seed_parent_id = str(source_parent_ids.get(seed_source_id) or _parent_id(seed))
                if seed_parent_id != parent_id:
                    continue
                if seed_source_id in source_id_set:
                    selected_source_ids.add(seed_source_id)
                seed_order = int(seed.get("order_index") or 0)
                for source_id in source_ids[max(0, seed_order - effective_window_radius * 3) : seed_order + effective_window_radius * 3 + 1]:
                    selected_source_ids.add(source_id)
            for term in anchor_probe_terms[:32]:
                posting_ids = lexical_postings.get(str(term or "").strip().lower()) or []
                for source_id in posting_ids:
                    if source_id in source_id_set:
                        selected_source_ids.add(source_id)
                        if len(selected_source_ids) >= (900 if benchmark_name == "knowme" else 700):
                            break
                if len(selected_source_ids) >= (900 if benchmark_name == "knowme" else 700):
                    break
            if selected_source_ids:
                source_ids = sorted(selected_source_ids, key=lambda source_id: order_by_source_id.get(source_id, 10**9))
                prefiltered_parent_child_count += len(source_ids)
        child_rows: list[dict[str, Any]] = []
        for source_id in source_ids:
            record = source_records.get(source_id)
            if record is None:
                continue
            lexical = max(
                _token_list_overlap_score(parent_probe_terms, record),
                _token_list_overlap_score(anchor_terms[:10], record),
            )
            entity_overlap = len(query_entities & set(record.get("entity_terms") or [])) / max(1, len(query_entities)) if query_entities else 0.0
            temporal_overlap = len(temporal_terms & set(record.get("specific_temporal_terms") or record.get("temporal_terms") or [])) / max(1, len(temporal_terms)) if temporal_terms else 0.0
            anchor_overlap = _record_anchor_overlap(
                record,
                anchor_terms=anchor_terms,
                specific_terms=specific_terms,
                phrases=phrases,
                anchor_probe=anchor_probe,
                fast=benchmark_name in {"knowme", "clonemem"},
            )
            surface_overlap = _weighted_surface_overlap(anchor_surface_terms, record=record)
            decomposition_overlap = _weighted_surface_overlap(decomposition_terms, record=record)
            phrase_score = max(
                (
                    _fast_phrase_match_score(phrase, phrase_tokens, record)
                    for phrase, phrase_tokens in phrase_token_cache.items()
                ),
                default=0.0,
            )
            matched_anchor_terms = _matched_surface_terms(record, terms=anchor_probe_terms)
            matched_term_weight = sum(_term_priority(term) for term in matched_anchor_terms[:12])
            speaker_persona_overlap = 1.0 if focus_person and (
                focus_person == str(record.get("speaker_id") or "").lower()
                or focus_person in set(record.get("entity_terms") or [])
                or focus_person in str(record.get("normalized_text") or "")
            ) else 0.0
            seed_strength = _candidate_seed_confidence(seed_candidates.get(source_id, {}))
            order_index = int(record.get("order_index") or 0)
            proximity = 0.0
            if seed_positions:
                proximity = max(0.0, 0.9 - min(abs(order_index - seed_position) for seed_position in seed_positions) * 0.12)
            direct_match = min(
                1.0,
                max(
                    lexical,
                    anchor_overlap,
                    surface_overlap,
                    decomposition_overlap,
                    phrase_score,
                    speaker_persona_overlap,
                    entity_overlap,
                    temporal_overlap,
                ),
            )
            anchor_priority = min(
                1.0,
                direct_match * 0.58
                + surface_overlap * 0.16
                + decomposition_overlap * 0.14
                + min(0.24, matched_term_weight / 10.0)
                + (0.08 if phrase_score >= 0.65 else 0.0)
                + speaker_persona_overlap * 0.08,
            )
            child_rows.append(
                {
                    "record": record,
                    "source_id": source_id,
                    "order_index": order_index,
                    "lexical": lexical,
                    "entity_overlap": entity_overlap,
                    "temporal_overlap": temporal_overlap,
                    "surface_overlap": surface_overlap,
                    "phrase_score": phrase_score,
                    "speaker_persona_overlap": speaker_persona_overlap,
                    "decomposition_overlap": decomposition_overlap,
                    "seed_strength": seed_strength,
                    "proximity": proximity,
                    "direct_match": direct_match,
                    "matched_anchor_terms": matched_anchor_terms,
                    "matched_term_weight": matched_term_weight,
                    "anchor_priority": anchor_priority,
                }
            )
        anchor_rows = sorted(
            child_rows,
            key=lambda row: (
                float(row.get("anchor_priority") or 0.0),
                float(row.get("direct_match") or 0.0),
                float(row.get("matched_term_weight") or 0.0),
                float(row.get("seed_strength") or 0.0),
                float(row.get("proximity") or 0.0),
            ),
            reverse=True,
        )
        anchor_positions = set(seed_positions)
        anchor_cap = max(2, min(parent_expand_segments, 6 if benchmark_name == "clonemem" else 4 if benchmark_name == "knowme" else 3))
        if anchor_preselection_enabled:
            anchor_selected_rows = _select_parent_anchor_rows(
                anchor_rows,
                anchor_cap=anchor_cap,
                filter_low_information_terms=parent_anchor_noise_filter_enabled,
                extra_noise_terms=parent_extra_noise_terms,
            )
        else:
            anchor_selected_rows = anchor_rows[:anchor_cap]
        for row in anchor_selected_rows:
            selected_parent_anchor_ids.add(str(row.get("source_id") or ""))
            covered_parent_anchor_terms.update(
                str(term or "").strip().lower()
                for term in list(row.get("matched_anchor_terms") or [])
                if str(term or "").strip()
            )
            anchor_positions.add(int(row.get("order_index") or 0))
            if len(selected_child_anchor_debug) < 40:
                selected_child_anchor_debug.append(
                    {
                        "parent_id": parent_id,
                        "source_id": str(row.get("source_id") or ""),
                        "order_index": int(row.get("order_index") or 0),
                        "anchor_priority": round(float(row.get("anchor_priority") or 0.0), 4),
                        "direct_match": round(float(row.get("direct_match") or 0.0), 4),
                        "matched_anchor_terms": list(row.get("matched_anchor_terms") or [])[:12],
                    }
                )
        child_scored: list[tuple[float, float, dict[str, Any]]] = []
        for row in child_rows:
            record = dict(row["record"])
            anchor_distance = min(
                (abs(int(row["order_index"]) - anchor_position) for anchor_position in anchor_positions),
                default=9999,
            )
            window_score = 0.0
            if anchor_distance <= effective_window_radius:
                window_score = max(0.0, 0.94 - anchor_distance * 0.14)
            elif anchor_distance <= effective_window_radius * 2:
                window_score = max(0.0, 0.44 - (anchor_distance - effective_window_radius) * 0.08)
            direct_weight = 0.3 if benchmark_name == "clonemem" else 0.24
            window_weight = 0.18 if benchmark_name == "clonemem" else 0.24
            score = min(
                1.0,
                parent_score * 0.16
                + float(row["direct_match"]) * direct_weight
                + float(row["entity_overlap"]) * 0.08
                + float(row["temporal_overlap"]) * 0.08
                + float(row["decomposition_overlap"]) * 0.12
                + float(row["speaker_persona_overlap"]) * 0.06
                + float(row["seed_strength"]) * 0.08
                + float(row["proximity"]) * 0.1
                + window_score * window_weight,
            )
            if score <= 0.0:
                continue
            child_scored.append((score, window_score, record))
        child_scored.sort(
            key=lambda item: (
                float(item[0]),
                float(item[1]),
                float(item[2].get("order_index") or 0),
            ),
            reverse=True,
        )
        if anchor_preselection_enabled:
            selected_children: list[tuple[float, float, dict[str, Any]]] = []
            selected_source_ids: set[str] = set()
            selected_count_before_tail = 0
            scored_by_source_id = {
                str(record.get("source_id") or ""): (score, window_score, record)
                for score, window_score, record in child_scored
                if str(record.get("source_id") or "")
            }
            for row in anchor_selected_rows:
                source_id = str(row.get("source_id") or "")
                scored_row = scored_by_source_id.get(source_id)
                if not source_id or scored_row is None or source_id in selected_source_ids:
                    continue
                selected_children.append(scored_row)
                selected_source_ids.add(source_id)
                if len(selected_children) >= max(1, parent_expand_segments):
                    break
            for score, window_score, record in child_scored:
                if len(selected_children) >= max(1, parent_expand_segments):
                    break
                source_id = str(record.get("source_id") or "")
                if not source_id or source_id in selected_source_ids:
                    continue
                selected_children.append((score, window_score, record))
                selected_source_ids.add(source_id)
            if parent_supplemental_anchor_expansion_enabled:
                supplemental_cap = max(0, int(parent_supplemental_anchor_expansion_cap))
                for row in anchor_selected_rows:
                    if len(selected_children) >= max(1, parent_expand_segments) + supplemental_cap:
                        break
                    source_id = str(row.get("source_id") or "")
                    scored_row = scored_by_source_id.get(source_id)
                    if not source_id or scored_row is None or source_id in selected_source_ids:
                        continue
                    selected_children.append(scored_row)
                    selected_source_ids.add(source_id)
                selected_terms = {
                    str(term or "").strip().lower()
                    for _, _, record in selected_children
                    for term in _matched_surface_terms(record, terms=anchor_probe_terms)
                    if str(term or "").strip()
                }
                uncovered_terms = [
                    term
                    for term in anchor_probe_terms[:40]
                    if str(term or "").strip().lower() not in selected_terms
                ]
                for score, window_score, record in child_scored:
                    if len(selected_children) >= max(1, parent_expand_segments) + supplemental_cap:
                        break
                    source_id = str(record.get("source_id") or "")
                    if not source_id or source_id in selected_source_ids:
                        continue
                    matched_terms = _matched_surface_terms(record, terms=uncovered_terms)
                    matched_terms = [
                        term
                        for term in matched_terms
                        if (
                            not parent_anchor_noise_filter_enabled
                            or _is_parent_anchor_term(term, extra_noise_terms=parent_extra_noise_terms)
                        )
                    ]
                    if not matched_terms:
                        continue
                    selected_children.append((score, window_score, record))
                    selected_source_ids.add(source_id)
                    selected_terms.update(matched_terms)
                    uncovered_terms = [term for term in uncovered_terms if str(term or "").strip().lower() not in selected_terms]
                    if not uncovered_terms:
                        break
            if benchmark_name == "clonemem" and parent_timestamp_sibling_expansion_enabled:
                sibling_cap = max(0, int(parent_timestamp_sibling_expansion_cap))
                selected_timestamps = {
                    str(record.get("timestamp") or "").strip()
                    for _, _, record in selected_children
                    if str(record.get("timestamp") or "").strip()
                }
                selected_source_ids = {
                    str(record.get("source_id") or "")
                    for _, _, record in selected_children
                    if str(record.get("source_id") or "")
                }
                sibling_rows = sorted(
                    child_rows,
                    key=lambda row: (
                        str(row.get("source_id") or "") in selected_source_ids,
                        -float(row.get("anchor_priority") or 0.0),
                        int(row.get("order_index") or 0),
                    ),
                )
                for row in sibling_rows:
                    if sibling_cap <= 0:
                        break
                    record = dict(row["record"])
                    source_id = str(record.get("source_id") or "")
                    if not source_id or source_id in selected_source_ids:
                        continue
                    timestamp = str(record.get("timestamp") or "").strip()
                    same_timestamp = bool(timestamp and timestamp in selected_timestamps)
                    if not same_timestamp:
                        continue
                    score = min(
                        1.0,
                        parent_score * 0.16
                        + float(row.get("direct_match") or 0.0) * 0.18
                        + float(row.get("entity_overlap") or 0.0) * 0.05
                        + float(row.get("temporal_overlap") or 0.0) * 0.05,
                    )
                    window_score = 0.0
                    adjusted_score = max(float(score), 0.32 if same_timestamp else 0.24)
                    adjusted_window = max(float(window_score), 0.28 if same_timestamp else 0.18)
                    selected_children.append((adjusted_score, adjusted_window, record))
                    selected_source_ids.add(source_id)
                    timestamp_sibling_selected_count += 1
                    sibling_cap -= 1
                    if len(timestamp_sibling_debug) < 40:
                        timestamp_sibling_debug.append(
                            {
                                "parent_id": parent_id,
                                "source_id": source_id,
                                "timestamp": timestamp,
                                "same_timestamp": same_timestamp,
                                "same_day": False,
                                "score": round(adjusted_score, 4),
                            }
                        )
        else:
            selected_children = child_scored[: max(1, parent_expand_segments)]
        max_selected_children = max(1, parent_expand_segments)
        if parent_supplemental_anchor_expansion_enabled:
            max_selected_children += max(0, int(parent_supplemental_anchor_expansion_cap))
            supplemental_anchor_selected_count += max(
                0,
                min(len(selected_children), max_selected_children) - max(1, parent_expand_segments),
            )
        if benchmark_name == "clonemem" and parent_timestamp_sibling_expansion_enabled:
            max_selected_children += max(0, int(parent_timestamp_sibling_expansion_cap))
        for score, window_score, record in selected_children[:max_selected_children]:
            _add_source_score(
                candidates,
                record=record,
                score_name="parent_score",
                score_value=score,
                retriever_name="parent_session",
            )
            candidate = candidates.get(str(record.get("source_id") or ""))
            if candidate is not None and window_score > 0.0:
                source_id = str(record.get("source_id") or "")
                if benchmark_name == "clonemem" and source_id in selected_parent_anchor_ids:
                    candidate["local_window_score"] = max(
                        float(candidate.get("local_window_score") or 0.0),
                        min(1.0, window_score * 0.42 + float(score) * 0.22),
                    )
                elif benchmark_name != "clonemem":
                    candidate["local_window_score"] = max(
                        float(candidate.get("local_window_score") or 0.0),
                        min(1.0, window_score * 0.68 + float(score) * 0.18),
                    )
    return candidates, {
        "parent_candidates": [
            {
                "parent_id": parent_id,
                "score": round(score, 4),
                "source_count": int(parent_record.get("source_count") or 0),
            }
            for score, parent_id, parent_record in top_parent_rows[:20]
        ],
        "selected_parent_count": len(top_parent_rows),
        "parent_window_radius": effective_window_radius,
        "parent_anchor_noise_filter_enabled": bool(parent_anchor_noise_filter_enabled),
        "parent_supplemental_anchor_expansion_enabled": bool(parent_supplemental_anchor_expansion_enabled),
        "parent_supplemental_anchor_expansion_cap": max(0, int(parent_supplemental_anchor_expansion_cap)),
        "supplemental_anchor_selected_count": supplemental_anchor_selected_count,
        "parent_timestamp_sibling_expansion_enabled": bool(parent_timestamp_sibling_expansion_enabled),
        "parent_timestamp_sibling_expansion_cap": max(0, int(parent_timestamp_sibling_expansion_cap)),
        "parent_anchor_strict_noise_filter_enabled": bool(parent_anchor_strict_noise_filter_enabled),
        "timestamp_sibling_selected_count": timestamp_sibling_selected_count,
        "timestamp_sibling_debug": timestamp_sibling_debug,
        "parent_anchor_selected_count": len(selected_parent_anchor_ids),
        "parent_anchor_term_coverage_count": len(covered_parent_anchor_terms),
        "full_parent_child_count": full_parent_child_count,
        "prefiltered_parent_child_count": prefiltered_parent_child_count,
        "selected_child_anchors": selected_child_anchor_debug,
        "covered_anchor_terms": sorted(covered_parent_anchor_terms)[:80],
        "missing_anchor_terms": [
            term
            for term in anchor_probe_terms[:80]
            if str(term or "").strip().lower() not in covered_parent_anchor_terms
        ][:40],
    }


def _channel_rows(
    name: str,
    source_map: dict[str, dict[str, Any]],
    merged: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    score_key = CHANNEL_SCORE_FIELDS[name]
    rows: list[dict[str, Any]] = []
    for source_id in source_map:
        merged_row = merged.get(source_id)
        if merged_row is None:
            continue
        score = float(merged_row.get(score_key) or 0.0)
        if score <= 0.0:
            continue
        rows.append(merged_row)
    rows.sort(
        key=lambda row: (
            float(row.get(score_key) or 0.0),
            float(row.get("dense_score") or 0.0),
            str(row.get("source_id") or ""),
        ),
        reverse=True,
    )
    return rows


def _channel_weight(benchmark_name: str, channel_name: str, policy: dict[str, Any]) -> float:
    if not bool(policy.get("channel_gating_enabled", True)):
        return 1.0
    dense_first = {
        "dense_semantic": 1.0,
        "lexical_sparse": 0.34,
        "entity_aware": 0.26,
        "temporal_anchor": 0.22,
        "exact_phrase": 0.3,
        "profile_side_index": 0.36,
        "session_bundle": 0.38,
        "temporal_neighbor": 0.22,
        "parent_session": 0.22,
        "query_decomposition": 0.24,
    }
    benchmark_overrides = {
        "longmemeval": {
            "lexical_sparse": 0.2,
            "entity_aware": 0.14,
            "temporal_anchor": 0.12,
            "exact_phrase": 0.0,
            "profile_side_index": 0.0,
            "session_bundle": 0.0,
            "temporal_neighbor": 0.0,
            "parent_session": 0.0,
            "query_decomposition": 0.0,
        },
        "locomo": {
            "lexical_sparse": 0.18,
            "entity_aware": 0.18,
            "temporal_anchor": 0.18,
            "exact_phrase": 0.28,
            "session_bundle": 0.42,
            "temporal_neighbor": 0.18,
            "parent_session": 0.14,
            "query_decomposition": 0.0,
        },
        "knowme": {
            "lexical_sparse": 0.34,
            "entity_aware": 0.4,
            "temporal_anchor": 0.24,
            "exact_phrase": 0.3,
            "profile_side_index": 0.52,
            "temporal_neighbor": 0.18,
            "parent_session": 0.26,
            "query_decomposition": 0.3,
        },
        "clonemem": {
            "lexical_sparse": 0.4,
            "entity_aware": 0.42,
            "temporal_anchor": 0.28,
            "exact_phrase": 0.46,
            "temporal_neighbor": 0.24,
            "parent_session": 0.3,
            "query_decomposition": 0.3,
        },
    }
    return float(benchmark_overrides.get(benchmark_name, {}).get(channel_name, dense_first.get(channel_name, 0.2)))


def _channel_quality_gate(channel_name: str, normalized: float, rank: int, benchmark_name: str) -> float:
    if channel_name == "dense_semantic":
        return 1.0
    if normalized >= 0.75:
        return 1.0
    if normalized >= 0.5:
        return 0.8
    if normalized >= 0.3:
        return 0.55
    if benchmark_name in {"longmemeval", "locomo"}:
        return 0.2 if rank > 12 else 0.35
    return 0.35 if rank <= 20 else 0.18


def _clonemem_lexical_anchor_gate(
    candidate: dict[str, Any],
    *,
    policy: dict[str, Any],
    lexical_rank: int | None = None,
) -> tuple[float, str]:
    if not bool(policy.get("clonemem_lexical_anchor_gate_enabled", False)):
        return 1.0, ""
    protected_top_k = max(0, int(policy.get("clonemem_lexical_anchor_gate_protected_top_k") or 0))
    if lexical_rank is not None and protected_top_k > 0 and 0 < int(lexical_rank) <= protected_top_k:
        return 1.0, ""
    min_support = max(1, int(policy.get("clonemem_lexical_anchor_gate_min_support") or 2))
    min_anchor = max(0.0, float(policy.get("clonemem_lexical_anchor_gate_min_anchor_score") or 0.24))
    support_count = int(candidate.get("support_count") or _support_count(candidate))
    anchor_score = max(
        float(candidate.get("dense_score") or 0.0),
        float(candidate.get("exact_phrase_score") or 0.0),
        float(candidate.get("entity_score") or 0.0),
        float(candidate.get("temporal_score") or 0.0),
        float(candidate.get("temporal_neighbor_score") or 0.0),
        float(candidate.get("local_window_score") or 0.0),
        float(candidate.get("parent_score") or 0.0),
        float(candidate.get("decomposition_score") or 0.0),
    )
    if support_count > min_support or anchor_score >= min_anchor:
        return 1.0, ""
    factor = max(0.0, min(1.0, float(policy.get("clonemem_lexical_anchor_gate_factor") or 0.35)))
    return factor, "weak_lexical_anchor_support"


def _apply_duplicate_collapse(
    rows: list[dict[str, Any]],
    *,
    duplicate_collapse_enabled: bool,
    near_duplicate_collapse_enabled: bool,
    safe_mode: bool = False,
    protected_source_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protected_source_ids = set(protected_source_ids or set())
    if not duplicate_collapse_enabled and not near_duplicate_collapse_enabled:
        return list(rows), {
            "duplicate_collapse_count": 0,
            "near_duplicate_collapse_count": 0,
            "removed_candidates": [],
        }
    selected: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    seen_text_hashes: dict[str, tuple[str, str, str]] = {}
    duplicate_collapse_count = 0
    near_duplicate_collapse_count = 0
    removed_candidates: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row.get("source_id") or "")
        segment_id = str(row.get("source_segment_id") or source_id)
        parent_id = _parent_id(row)
        if duplicate_collapse_enabled and source_id in seen_source_ids:
            duplicate_collapse_count += 1
            if len(removed_candidates) < 200:
                removed_candidates.append(
                    {
                        "source_id": source_id,
                        "source_segment_id": segment_id,
                        "parent_id": parent_id,
                        "reason": "duplicate_source_id",
                    }
                )
            continue
        text_hash = str(row.get("text_hash") or "")
        if near_duplicate_collapse_enabled and text_hash and text_hash in seen_text_hashes:
            seen_source_id, seen_segment_id, seen_parent_id = seen_text_hashes[text_hash]
            collapse_allowed = True
            if safe_mode:
                collapse_allowed = bool(
                    segment_id
                    and seen_segment_id
                    and segment_id == seen_segment_id
                    and parent_id == seen_parent_id
                )
            if collapse_allowed and source_id not in protected_source_ids:
                near_duplicate_collapse_count += 1
                if len(removed_candidates) < 200:
                    removed_candidates.append(
                        {
                            "source_id": source_id,
                            "source_segment_id": segment_id,
                            "parent_id": parent_id,
                            "reason": "near_duplicate_text_hash",
                            "kept_source_id": seen_source_id,
                        }
                    )
                continue
        selected.append(row)
        seen_source_ids.add(source_id)
        if text_hash:
            seen_text_hashes[text_hash] = (source_id, segment_id, parent_id)
    return selected, {
        "duplicate_collapse_count": duplicate_collapse_count,
        "near_duplicate_collapse_count": near_duplicate_collapse_count,
        "removed_candidates": removed_candidates,
    }


def _apply_clonemem_channel_tail_rescue(
    rows: list[dict[str, Any]],
    *,
    channel_rankings: dict[str, list[dict[str, Any]]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not bool(policy.get("clonemem_channel_tail_rescue_enabled", False)):
        return rows, []
    max_rank = max(101, int(policy.get("clonemem_channel_tail_rescue_max_rank") or 180))
    per_channel = max(0, int(policy.get("clonemem_channel_tail_rescue_per_channel") or 0))
    if per_channel <= 0:
        return rows, []
    target_rank = max(11, int(policy.get("clonemem_channel_tail_rescue_target_rank") or 90))
    target_index = min(max(0, target_rank - 1), len(rows))
    current_ids = [str(row.get("source_id") or "") for row in rows]
    top100_ids = set(current_ids[:100])
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    selected_audit: list[dict[str, Any]] = []
    rescue_channels = (
        "dense_semantic",
        "lexical_sparse",
        "entity_aware",
        "exact_phrase",
        "temporal_neighbor",
        "parent_session",
        "query_decomposition",
    )
    for channel_name in rescue_channels:
        rows_for_channel = channel_rankings.get(channel_name) or []
        admitted_for_channel = 0
        for rank, candidate in enumerate(rows_for_channel, start=1):
            if rank <= 100 or rank > max_rank:
                continue
            source_id = str(candidate.get("source_id") or "")
            if not source_id or source_id in top100_ids or source_id in selected_ids:
                continue
            support_count = int(candidate.get("support_count") or _support_count(candidate))
            if support_count <= 0:
                continue
            item = candidate
            item["clonemem_channel_tail_rescue_reason"] = f"{channel_name}_rank_{rank}"
            selected.append(item)
            selected_ids.add(source_id)
            selected_audit.append(
                {
                    "source_id": source_id,
                    "source_segment_id": str(candidate.get("source_segment_id") or ""),
                    "parent_id": _parent_id(candidate),
                    "channel": channel_name,
                    "channel_rank": rank,
                    "support_count": support_count,
                    "broad_score": round(float(candidate.get("broad_score") or 0.0), 4),
                }
            )
            admitted_for_channel += 1
            if admitted_for_channel >= per_channel:
                break
    if not selected:
        return rows, []
    remaining = [row for row in rows if str(row.get("source_id") or "") not in selected_ids]
    rescued = remaining[:target_index] + selected + remaining[target_index:]
    return rescued, selected_audit


def _apply_clonemem_evidence_consensus_admission(
    rows: list[dict[str, Any]],
    *,
    channel_rankings: dict[str, list[dict[str, Any]]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not bool(policy.get("clonemem_evidence_consensus_admission_enabled", False)):
        return rows, []
    max_candidates = max(0, int(policy.get("clonemem_evidence_consensus_admission_max_candidates") or 0))
    if max_candidates <= 0:
        return rows, []
    min_channels = max(2, int(policy.get("clonemem_evidence_consensus_admission_min_channels") or 2))
    target_rank = max(11, int(policy.get("clonemem_evidence_consensus_admission_target_rank") or 88))
    target_index = max(0, target_rank - 1)
    top100_ids = {str(row.get("source_id") or "") for row in rows[:100] if str(row.get("source_id") or "")}
    selected_ids: set[str] = set()
    eligible_channels = (
        "entity_aware",
        "exact_phrase",
        "parent_session",
        "query_decomposition",
        "temporal_neighbor",
    )
    support_by_source: dict[str, dict[str, Any]] = defaultdict(lambda: {"channels": {}, "candidate": None})
    for channel_name in eligible_channels:
        for rank, candidate in enumerate(channel_rankings.get(channel_name) or [], start=1):
            if rank > 100:
                break
            source_id = str(candidate.get("source_id") or "")
            if not source_id or source_id in top100_ids:
                continue
            support_by_source[source_id]["channels"][channel_name] = rank
            if support_by_source[source_id]["candidate"] is None:
                support_by_source[source_id]["candidate"] = candidate
    selected: list[dict[str, Any]] = []
    selected_audit: list[dict[str, Any]] = []
    scored: list[tuple[float, int, str, dict[str, Any], dict[str, int]]] = []
    for source_id, support in support_by_source.items():
        channel_ranks = dict(support.get("channels") or {})
        if len(channel_ranks) < min_channels:
            continue
        candidate = support.get("candidate")
        if not isinstance(candidate, dict):
            continue
        best_rank = min(channel_ranks.values())
        score = (
            len(channel_ranks) * 0.16
            + max(0.0, (101 - best_rank) / 100.0) * 0.08
            + min(0.08, max(0, _support_count(candidate) - 1) * 0.02)
            + min(0.08, float(candidate.get("broad_score") or 0.0) * 0.05)
        )
        scored.append((score, best_rank, source_id, candidate, channel_ranks))
    scored.sort(key=lambda item: (item[0], -item[1], item[2]), reverse=True)
    for score, best_rank, source_id, candidate, channel_ranks in scored:
        if len(selected) >= max_candidates:
            break
        if source_id in selected_ids:
            continue
        item = candidate
        item["clonemem_evidence_consensus_admission_reason"] = ",".join(sorted(channel_ranks))
        selected.append(item)
        selected_ids.add(source_id)
        selected_audit.append(
            {
                "source_id": source_id,
                "source_segment_id": str(candidate.get("source_segment_id") or ""),
                "parent_id": _parent_id(candidate),
                "channels": dict(sorted(channel_ranks.items())),
                "best_channel_rank": best_rank,
                "support_count": int(candidate.get("support_count") or _support_count(candidate)),
                "score": round(score, 4),
                "broad_score": round(float(candidate.get("broad_score") or 0.0), 4),
            }
        )
    if not selected:
        return rows, []
    remaining = [row for row in rows if str(row.get("source_id") or "") not in selected_ids]
    admitted = remaining[:target_index] + selected + remaining[target_index:]
    return admitted, selected_audit


def _fuse_candidate_sources(
    *,
    benchmark_name: str,
    merged: dict[str, dict[str, Any]],
    channel_maps: dict[str, dict[str, dict[str, Any]]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    channel_rankings = {
        name: _channel_rows(name, source_map, merged)
        for name, source_map in channel_maps.items()
        if source_map
    }
    rrf_k = max(1, int(policy.get("rrf_k") or 60))
    safe_fusion_enabled = bool(policy.get("safe_fusion_enabled", False))
    dense_preserve_enabled = bool(policy.get("dense_preserve_enabled", False))
    dense_anchor_top_k = max(1, int(policy.get("dense_anchor_top_k") or 100))
    dense_anchor_min_keep = max(1, min(int(policy.get("dense_anchor_min_keep") or dense_anchor_top_k), dense_anchor_top_k))
    dense_rankings = list(channel_rankings.get("dense_semantic") or [])
    dense_anchor_ids = [
        str(row.get("source_id") or "")
        for row in dense_rankings[:dense_anchor_top_k]
        if str(row.get("source_id") or "")
    ]
    dense_anchor_front_ids = dense_anchor_ids[:dense_anchor_min_keep]
    dense_anchor_set = set(dense_anchor_ids)
    dense_anchor_front_set = set(dense_anchor_front_ids)
    for candidate in merged.values():
        candidate["fusion_score"] = 0.0
        candidate["clonemem_lexical_anchor_gate_factor"] = 1.0
        candidate["clonemem_lexical_anchor_gate_reason"] = ""
    channel_contributions: dict[str, dict[str, float]] = defaultdict(dict)
    channel_weights: dict[str, float] = {}
    lexical_anchor_gate_applied: list[dict[str, Any]] = []
    for name, rows in channel_rankings.items():
        weight = _channel_weight(benchmark_name, name, policy)
        channel_weights[name] = round(weight, 4)
        if weight <= 0.0:
            continue
        score_key = CHANNEL_SCORE_FIELDS[name]
        max_score = max((float(row.get(score_key) or 0.0) for row in rows), default=0.0)
        for rank, row in enumerate(rows, start=1):
            source_id = str(row.get("source_id") or "")
            normalized = (float(row.get(score_key) or 0.0) / max_score) if max_score > 0.0 else 0.0
            quality_gate = _channel_quality_gate(name, normalized, rank, benchmark_name) if safe_fusion_enabled else 1.0
            lexical_gate = 1.0
            lexical_gate_reason = ""
            if benchmark_name == "clonemem" and name == "lexical_sparse":
                lexical_gate, lexical_gate_reason = _clonemem_lexical_anchor_gate(
                    row,
                    policy=policy,
                    lexical_rank=rank,
                )
                if lexical_gate < 1.0:
                    merged[source_id]["clonemem_lexical_anchor_gate_factor"] = round(lexical_gate, 4)
                    merged[source_id]["clonemem_lexical_anchor_gate_reason"] = lexical_gate_reason
                    if len(lexical_anchor_gate_applied) < 200:
                        lexical_anchor_gate_applied.append(
                            {
                                "source_id": source_id,
                                "source_segment_id": str(row.get("source_segment_id") or ""),
                                "parent_id": _parent_id(row),
                                "lexical_rank": rank,
                                "bm25_score": round(float(row.get("bm25_score") or 0.0), 4),
                                "support_count": int(row.get("support_count") or _support_count(row)),
                                "anchor_score": round(
                                    max(
                                        float(row.get("dense_score") or 0.0),
                                        float(row.get("exact_phrase_score") or 0.0),
                                        float(row.get("entity_score") or 0.0),
                                        float(row.get("temporal_score") or 0.0),
                                        float(row.get("temporal_neighbor_score") or 0.0),
                                        float(row.get("local_window_score") or 0.0),
                                        float(row.get("parent_score") or 0.0),
                                        float(row.get("decomposition_score") or 0.0),
                                    ),
                                    4,
                                ),
                                "factor": round(lexical_gate, 4),
                                "reason": lexical_gate_reason,
                            }
                        )
            contribution = weight * ((1.0 / float(rrf_k + rank)) + normalized * 0.05 * quality_gate)
            contribution *= lexical_gate
            if safe_fusion_enabled and dense_preserve_enabled and name != "dense_semantic" and source_id in dense_anchor_front_set:
                contribution *= 0.6
            merged[source_id]["fusion_score"] = round(
                float(merged[source_id].get("fusion_score") or 0.0) + contribution,
                6,
            )
            channel_contributions[source_id][name] = round(contribution, 6)
    ranked = sorted(
        merged.values(),
        key=lambda row: (
            float(row.get("fusion_score") or 0.0),
            float(row.get("dense_score") or 0.0),
            float(row.get("broad_score") or 0.0),
            str(row.get("source_id") or ""),
        ),
        reverse=True,
    )
    collapsed, collapse_audit = _apply_duplicate_collapse(
        ranked,
        duplicate_collapse_enabled=bool(policy.get("duplicate_collapse_enabled", True)),
        near_duplicate_collapse_enabled=bool(policy.get("near_duplicate_collapse_enabled", True)),
        safe_mode=bool(policy.get("duplicate_collapse_safe_mode", False)),
        protected_source_ids=dense_anchor_set if safe_fusion_enabled and dense_preserve_enabled else set(),
    )
    collapsed_by_id = {
        str(row.get("source_id") or ""): row
        for row in collapsed
        if str(row.get("source_id") or "")
    }
    reordered: list[dict[str, Any]]
    parent_diversity_adjusted = False
    if safe_fusion_enabled and dense_preserve_enabled and dense_anchor_front_ids:
        protected_front = [collapsed_by_id[source_id] for source_id in dense_anchor_front_ids if source_id in collapsed_by_id]
        protected_front_ids = {str(row.get("source_id") or "") for row in protected_front}
        reordered = protected_front + [
            row for row in collapsed if str(row.get("source_id") or "") not in protected_front_ids
        ]
    else:
        min_parent_diversity = max(1, int(policy.get("min_parent_diversity") or 1))
        parent_best: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        seen_parent_best: set[str] = set()
        for row in collapsed:
            parent_id = _parent_id(row)
            if parent_id and parent_id not in seen_parent_best and len(parent_best) < min_parent_diversity:
                parent_best.append(row)
                seen_parent_best.add(parent_id)
            else:
                remaining.append(row)
        reordered = parent_best + remaining
        parent_diversity_adjusted = len(parent_best) > 1
    final_rows: list[dict[str, Any]] = []
    skipped_crowding: list[dict[str, Any]] = []
    skipped_crowding_audit: list[dict[str, Any]] = []
    parent_counts: Counter[str] = Counter()
    potential_parent_cap_count = 0
    max_candidates_per_parent = max(1, int(policy.get("max_candidates_per_parent") or 20))
    final_candidate_pool_size = max(1, int(policy.get("final_candidate_pool_size") or len(reordered)))
    apply_parent_cap = not (safe_fusion_enabled and bool(policy.get("parent_cap_after_gold_agnostic_anchor", True)))
    for row in reordered:
        parent_id = _parent_id(row)
        if parent_id and parent_counts[parent_id] >= max_candidates_per_parent:
            potential_parent_cap_count += 1
            if apply_parent_cap:
                skipped_crowding.append(row)
                if len(skipped_crowding_audit) < 200:
                    skipped_crowding_audit.append(
                        {
                            "source_id": str(row.get("source_id") or ""),
                            "source_segment_id": str(row.get("source_segment_id") or ""),
                            "parent_id": parent_id,
                            "reason": "parent_cap",
                        }
                    )
                continue
        parent_counts[parent_id] += 1
        final_rows.append(row)
        if len(final_rows) >= final_candidate_pool_size:
            break
    if apply_parent_cap and len(final_rows) < final_candidate_pool_size:
        for row in skipped_crowding:
            final_rows.append(row)
            if len(final_rows) >= final_candidate_pool_size:
                break
    restored_dense_candidate_ids: list[str] = []
    if safe_fusion_enabled and dense_preserve_enabled and bool(policy.get("destructive_filter_guard_enabled", True)):
        final_ids = {str(row.get("source_id") or "") for row in final_rows}
        missing_dense_rows = [collapsed_by_id[source_id] for source_id in dense_anchor_ids if source_id in collapsed_by_id and source_id not in final_ids]
        for row in missing_dense_rows:
            if len(final_rows) < final_candidate_pool_size:
                final_rows.append(row)
                final_ids.add(str(row.get("source_id") or ""))
                restored_dense_candidate_ids.append(str(row.get("source_id") or ""))
                continue
            replace_at: int | None = None
            for index in range(len(final_rows) - 1, -1, -1):
                candidate_id = str(final_rows[index].get("source_id") or "")
                if candidate_id not in dense_anchor_set:
                    replace_at = index
                    break
            if replace_at is None:
                break
            final_rows[replace_at] = row
            final_ids.add(str(row.get("source_id") or ""))
            restored_dense_candidate_ids.append(str(row.get("source_id") or ""))
    clonemem_consensus_admission_audit: list[dict[str, Any]] = []
    clonemem_tail_rescue_audit: list[dict[str, Any]] = []
    if benchmark_name == "clonemem":
        final_rows, clonemem_consensus_admission_audit = _apply_clonemem_evidence_consensus_admission(
            final_rows,
            channel_rankings=channel_rankings,
            policy=policy,
        )
        final_rows, clonemem_tail_rescue_audit = _apply_clonemem_channel_tail_rescue(
            final_rows,
            channel_rankings=channel_rankings,
            policy=policy,
        )
        final_rows = final_rows[:final_candidate_pool_size]
    top_parent_distribution = Counter(_parent_id(row) for row in final_rows[:10] if _parent_id(row))
    return final_rows, {
        "fusion_method": str(policy.get("fusion_method") or "rrf"),
        "rrf_k": rrf_k,
        "safe_fusion_enabled": safe_fusion_enabled,
        "dense_preserve_enabled": dense_preserve_enabled,
        "dense_anchor_top_k": dense_anchor_top_k,
        "dense_anchor_min_keep": dense_anchor_min_keep,
        "protected_dense_candidate_count": len(dense_anchor_ids),
        "protected_dense_front_count": len(dense_anchor_front_ids),
        "protected_dense_retained_count": sum(
            1 for row in final_rows if str(row.get("source_id") or "") in dense_anchor_set
        ),
        "duplicate_collapse_count": int(collapse_audit.get("duplicate_collapse_count") or 0),
        "near_duplicate_collapse_count": int(collapse_audit.get("near_duplicate_collapse_count") or 0),
        "duplicate_removed_candidates": list(collapse_audit.get("removed_candidates") or []),
        "parent_diversity_adjusted": parent_diversity_adjusted,
        "parent_cap_applied": apply_parent_cap,
        "local_crowding_count": len(skipped_crowding),
        "potential_parent_cap_count": potential_parent_cap_count,
        "parent_cap_skipped_candidates": skipped_crowding_audit,
        "restored_dense_candidate_ids": restored_dense_candidate_ids,
        "clonemem_lexical_anchor_gate_enabled": bool(policy.get("clonemem_lexical_anchor_gate_enabled", False)),
        "clonemem_lexical_anchor_gate_applied_count": len(lexical_anchor_gate_applied),
        "clonemem_lexical_anchor_gate_applied": lexical_anchor_gate_applied,
        "clonemem_channel_tail_rescue_enabled": bool(policy.get("clonemem_channel_tail_rescue_enabled", False)),
        "clonemem_channel_tail_rescue_count": len(clonemem_tail_rescue_audit),
        "clonemem_channel_tail_rescue_applied": clonemem_tail_rescue_audit,
        "clonemem_evidence_consensus_admission_enabled": bool(
            policy.get("clonemem_evidence_consensus_admission_enabled", False)
        ),
        "clonemem_evidence_consensus_admission_count": len(clonemem_consensus_admission_audit),
        "clonemem_evidence_consensus_admission_applied": clonemem_consensus_admission_audit,
        "unique_parent_count": len(parent_counts),
        "top_parent_distribution": dict(top_parent_distribution),
        "channel_contributions": channel_contributions,
        "channel_weights": channel_weights,
    }, channel_rankings


def _speaker_persona_score(query_features: dict[str, Any], candidate: dict[str, Any]) -> float:
    focus_person = str(query_features.get("focus_person") or "")
    speaker_id = str(candidate.get("speaker_id") or "").lower()
    candidate_entities = set(candidate.get("entity_terms") or [])
    if focus_person and (focus_person == speaker_id or focus_person in candidate_entities):
        return 1.0
    entity_overlap = len(set(query_features.get("entities") or []) & candidate_entities)
    return min(1.0, entity_overlap * 0.5)


def _cluster_id(candidate: dict[str, Any]) -> str:
    for key in ("session_id", "source_doc_id", "conversation_id", "sample_id", "source_segment_id", "source_id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return ""


def _support_count(candidate: dict[str, Any]) -> int:
    return sum(
        1
        for key in (
            "dense_score",
            "bm25_score",
            "entity_score",
            "temporal_score",
            "profile_score",
            "session_score",
            "exact_phrase_score",
            "speaker_score",
            "temporal_neighbor_score",
            "parent_score",
            "decomposition_score",
            "local_window_score",
        )
        if float(candidate.get(key) or 0.0) > 0.0
    )


def _semantic_score(candidate: dict[str, Any]) -> float:
    dense = float(candidate.get("dense_score") or 0.0)
    exact = float(candidate.get("exact_phrase_score") or 0.0)
    return round(max(dense, exact * 0.82), 4)


def _task_score(benchmark_name: str, candidate: dict[str, Any]) -> float:
    bm25 = float(candidate.get("bm25_score") or 0.0)
    entity = float(candidate.get("entity_score") or 0.0)
    temporal = float(candidate.get("temporal_score") or 0.0)
    profile = float(candidate.get("profile_score") or 0.0)
    session = float(candidate.get("session_score") or 0.0)
    exact = float(candidate.get("exact_phrase_score") or 0.0)
    speaker = float(candidate.get("speaker_score") or 0.0)
    temporal_neighbor = float(candidate.get("temporal_neighbor_score") or candidate.get("local_window_score") or 0.0)
    parent_score = float(candidate.get("parent_score") or 0.0)
    decomposition = float(candidate.get("decomposition_score") or 0.0)
    local_window = float(candidate.get("local_window_score") or 0.0)
    if benchmark_name == "locomo":
        return round(
            0.3 * session
            + 0.17 * temporal
            + 0.16 * entity
            + 0.12 * exact
            + 0.1 * bm25
            + 0.08 * parent_score
            + 0.07 * temporal_neighbor
            + 0.06 * decomposition,
            4,
        )
    if benchmark_name == "knowme":
        return round(
            0.34 * profile
            + 0.16 * decomposition
            + 0.14 * exact
            + 0.13 * entity
            + 0.1 * temporal
            + 0.07 * bm25
            + 0.06 * parent_score,
            4,
        )
    if benchmark_name == "clonemem":
        return round(
            0.24 * exact
            + 0.18 * temporal_neighbor
            + 0.16 * parent_score
            + 0.14 * decomposition
            + 0.12 * speaker
            + 0.08 * temporal
            + 0.08 * entity
            + 0.06 * local_window,
            4,
        )
    return round(0.48 * bm25 + 0.28 * entity + 0.24 * temporal, 4)


def _broad_rank_bonus(rank: int, *, max_rank: int, max_bonus: float) -> float:
    if rank <= 0 or rank > max_rank:
        return 0.0
    return round(max_bonus * ((max_rank - rank + 1) / max_rank), 4)


def _ordering_anchor_bonus(benchmark_name: str, candidate: dict[str, Any]) -> float:
    broad_rank = int(candidate.get("broad_rank") or 0)
    if broad_rank <= 0:
        return 0.0
    broad = min(1.0, float(candidate.get("broad_score") or 0.0))
    dense = float(candidate.get("dense_score") or 0.0)
    exact = float(candidate.get("exact_phrase_score") or 0.0)
    session = float(candidate.get("session_score") or 0.0)
    decomposition = float(candidate.get("decomposition_score") or 0.0)
    local_window = float(candidate.get("local_window_score") or 0.0)
    parent_score = float(candidate.get("parent_score") or 0.0)
    support_count = int(candidate.get("support_count") or _support_count(candidate))
    if benchmark_name == "locomo":
        bonus = _broad_rank_bonus(broad_rank, max_rank=15, max_bonus=0.16)
        if broad_rank <= 10 and session <= 0.05 and max(dense, exact, decomposition, broad) >= 0.3:
            bonus += 0.035
        if broad_rank <= 12 and support_count >= 6:
            bonus += 0.015
        return round(min(0.22, bonus), 4)
    if benchmark_name == "knowme":
        bonus = _broad_rank_bonus(broad_rank, max_rank=12, max_bonus=0.09)
        if broad_rank <= 10 and max(exact, parent_score, decomposition, broad) >= 0.32:
            bonus += 0.015
        return round(min(0.12, bonus), 4)
    if benchmark_name == "clonemem":
        return _broad_rank_bonus(broad_rank, max_rank=12, max_bonus=0.08)
    return _broad_rank_bonus(broad_rank, max_rank=12, max_bonus=0.14)


def _rank_preserved_rerank_score(
    benchmark_name: str,
    candidate: dict[str, Any],
    score: float,
    *,
    enabled: bool = True,
    policy: dict[str, Any] | None = None,
) -> float:
    if not enabled:
        return round(float(score), 4)
    policy = policy or {}
    broad_rank = int(candidate.get("broad_rank") or 0)
    if broad_rank <= 0:
        return round(float(score), 4)
    support_count = int(candidate.get("support_count") or _support_count(candidate))
    if benchmark_name == "locomo":
        if broad_rank > 10:
            return round(float(score), 4)
        broad_floor = 0.47 - broad_rank * 0.004
        if support_count >= 5:
            broad_floor += 0.006
        if float(candidate.get("session_score") or 0.0) > 0.0:
            broad_floor += 0.006
        return round(max(float(score), broad_floor), 4)
    if benchmark_name == "clonemem" and bool(policy.get("clonemem_dense_anchor_rerank_guard_enabled", False)):
        max_rank = max(1, int(policy.get("clonemem_dense_anchor_rerank_guard_max_rank") or 10))
        min_dense = max(0.0, float(policy.get("clonemem_dense_anchor_rerank_guard_min_dense") or 0.52))
        min_support = max(1, int(policy.get("clonemem_dense_anchor_rerank_guard_min_support") or 2))
        floor_base = max(0.0, float(policy.get("clonemem_dense_anchor_rerank_guard_floor") or 0.72))
        dense = float(candidate.get("dense_score") or 0.0)
        broad = min(1.0, float(candidate.get("broad_score") or 0.0))
        exact = float(candidate.get("exact_phrase_score") or 0.0)
        parent_score = float(candidate.get("parent_score") or 0.0)
        decomposition = float(candidate.get("decomposition_score") or 0.0)
        local_window = float(candidate.get("local_window_score") or 0.0)
        if broad_rank <= max_rank and dense >= min_dense and support_count >= min_support:
            evidence_bonus = 0.0
            if max(exact, decomposition, local_window) >= 0.35:
                evidence_bonus += 0.018
            if parent_score >= 0.35:
                evidence_bonus += 0.01
            broad_bonus = min(0.018, broad * 0.012)
            guarded_floor = floor_base - (broad_rank - 1) * 0.018 + evidence_bonus + broad_bonus
            guarded_floor = min(0.78, max(0.0, guarded_floor))
            if guarded_floor > float(score):
                candidate["clonemem_dense_anchor_guard_applied"] = True
                candidate["clonemem_dense_anchor_guard_floor"] = round(guarded_floor, 4)
            return round(max(float(score), guarded_floor), 4)
    if benchmark_name == "clonemem" and bool(policy.get("clonemem_evidence_rank_preservation_enabled", False)):
        max_rank = max(1, int(policy.get("clonemem_evidence_rank_preservation_max_rank") or 20))
        min_support = max(1, int(policy.get("clonemem_evidence_rank_preservation_min_support") or 5))
        min_broad = max(0.0, float(policy.get("clonemem_evidence_rank_preservation_min_broad_score") or 0.65))
        floor_base = max(0.0, float(policy.get("clonemem_evidence_rank_preservation_floor") or 0.68))
        protected_top_k = max(0, int(policy.get("clonemem_evidence_rank_preservation_protected_top_k") or 0))
        broad_rank = int(candidate.get("broad_rank") or 0)
        broad = float(candidate.get("broad_score") or 0.0)
        if broad_rank and protected_top_k < broad_rank <= max_rank and support_count >= min_support and broad >= min_broad:
            rank_decay = max(0, broad_rank - 1) * 0.006
            evidence_floor = max(0.0, floor_base - rank_decay)
            if evidence_floor > float(score):
                candidate["clonemem_evidence_rank_preservation_applied"] = True
                candidate["clonemem_evidence_rank_preservation_floor"] = round(evidence_floor, 4)
            return round(max(float(score), evidence_floor), 4)
    return round(float(score), 4)


def _candidate_seed_confidence(candidate: dict[str, Any]) -> float:
    return max(
        float(candidate.get("dense_score") or 0.0),
        float(candidate.get("bm25_score") or 0.0),
        float(candidate.get("entity_score") or 0.0),
        float(candidate.get("temporal_score") or 0.0),
        float(candidate.get("profile_score") or 0.0),
        float(candidate.get("session_score") or 0.0),
        float(candidate.get("exact_phrase_score") or 0.0),
        float(candidate.get("speaker_score") or 0.0),
        float(candidate.get("temporal_neighbor_score") or 0.0),
        float(candidate.get("parent_score") or 0.0),
        float(candidate.get("decomposition_score") or 0.0),
        float(candidate.get("local_window_score") or 0.0),
        float(candidate.get("fusion_score") or 0.0),
    )


def merge_candidate_sources(
    *source_maps: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source_map in source_maps:
        for source_id, candidate in source_map.items():
            current = merged.get(source_id)
            if current is None:
                merged[source_id] = dict(candidate)
                continue
            for key in (
                "dense_score",
                "bm25_score",
                "entity_score",
                "temporal_score",
                "profile_score",
                "session_score",
                "exact_phrase_score",
                "speaker_score",
                "temporal_neighbor_score",
                "parent_score",
                "decomposition_score",
                "local_window_score",
                "fusion_score",
            ):
                current[key] = max(float(current.get(key) or 0.0), float(candidate.get(key) or 0.0))
            for retriever in list(candidate.get("source_retrievers") or []):
                if retriever not in current["source_retrievers"]:
                    current["source_retrievers"].append(retriever)
            for chunk_id in list(candidate.get("source_chunk_ids") or []):
                if chunk_id and chunk_id not in current["source_chunk_ids"]:
                    current["source_chunk_ids"].append(chunk_id)
            if not current.get("best_chunk_id") and candidate.get("best_chunk_id"):
                current["best_chunk_id"] = candidate["best_chunk_id"]
    return merged


def _sorted_source_candidates(
    source_map: dict[str, dict[str, Any]],
    merged: dict[str, dict[str, Any]],
    score_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_id in source_map:
        merged_row = merged.get(source_id)
        if merged_row is None or float(merged_row.get(score_key) or 0.0) <= 0.0:
            continue
        rows.append(merged_row)
    rows.sort(
        key=lambda row: (
            float(row.get(score_key) or 0.0),
            float(row.get("broad_score") or 0.0),
            float(row.get("dense_score") or 0.0),
        ),
        reverse=True,
    )
    return rows


def _broad_source_order(benchmark_name: str) -> list[tuple[str, str]]:
    if benchmark_name == "locomo":
        return [
            ("dense", "dense_score"),
            ("exact_phrase", "exact_phrase_score"),
            ("session_bundle", "session_score"),
            ("bm25", "bm25_score"),
            ("entity", "entity_score"),
            ("temporal", "temporal_score"),
        ]
    if benchmark_name == "knowme":
        return [
            ("dense", "dense_score"),
            ("exact_phrase", "exact_phrase_score"),
            ("profile_fact", "profile_score"),
            ("bm25", "bm25_score"),
            ("entity", "entity_score"),
            ("temporal", "temporal_score"),
        ]
    if benchmark_name == "clonemem":
        return [
            ("exact_phrase", "exact_phrase_score"),
            ("entity", "entity_score"),
            ("dense", "dense_score"),
            ("bm25", "bm25_score"),
            ("temporal", "temporal_score"),
            ("local_window", "local_window_score"),
        ]
    return [
        ("dense", "dense_score"),
        ("bm25", "bm25_score"),
        ("entity", "entity_score"),
        ("temporal", "temporal_score"),
    ]


def assemble_broad_candidate_pool(
    *,
    benchmark_name: str,
    merged: dict[str, dict[str, Any]],
    dense: dict[str, dict[str, Any]],
    lexical: dict[str, dict[str, Any]],
    entity: dict[str, dict[str, Any]],
    temporal: dict[str, dict[str, Any]],
    profile: dict[str, dict[str, Any]],
    session: dict[str, dict[str, Any]],
    exact: dict[str, dict[str, Any]],
    local_window: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    source_maps = {
        "dense": dense,
        "bm25": lexical,
        "entity": entity,
        "temporal": temporal,
        "profile_fact": profile,
        "session_bundle": session,
        "exact_phrase": exact,
        "local_window": local_window,
    }
    order = _broad_source_order(benchmark_name)
    ranked_by_source = {
        source_name: _sorted_source_candidates(source_maps[source_name], merged, score_key)
        for source_name, score_key in order
    }
    positions = {source_name: 0 for source_name, _ in order}
    broad: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(broad) < max(1, limit):
        progressed = False
        for source_name, _ in order:
            rows = ranked_by_source[source_name]
            while positions[source_name] < len(rows):
                row = rows[positions[source_name]]
                positions[source_name] += 1
                source_id = str(row.get("source_id") or "")
                if not source_id or source_id in seen:
                    continue
                broad.append(dict(row))
                seen.add(source_id)
                progressed = True
                break
            if len(broad) >= max(1, limit):
                break
        if not progressed:
            break
    if len(broad) < max(1, limit):
        for row in sorted(merged.values(), key=lambda item: float(item.get("broad_score") or 0.0), reverse=True):
            source_id = str(row.get("source_id") or "")
            if not source_id or source_id in seen:
                continue
            broad.append(dict(row))
            seen.add(source_id)
            if len(broad) >= max(1, limit):
                break
    return broad


def _broad_score(candidate: dict[str, Any]) -> float:
    benchmark_name = str(candidate.get("benchmark_name") or "").lower()
    semantic = _semantic_score(candidate)
    support_bonus = min(0.12, max(0, _support_count(candidate) - 1) * 0.02)
    bm25 = float(candidate.get("bm25_score") or 0.0)
    exact = float(candidate.get("exact_phrase_score") or 0.0)
    entity = float(candidate.get("entity_score") or 0.0)
    temporal = float(candidate.get("temporal_score") or 0.0)
    profile = float(candidate.get("profile_score") or 0.0)
    session = float(candidate.get("session_score") or 0.0)
    speaker = float(candidate.get("speaker_score") or 0.0)
    temporal_neighbor = float(candidate.get("temporal_neighbor_score") or candidate.get("local_window_score") or 0.0)
    parent_score = float(candidate.get("parent_score") or 0.0)
    decomposition = float(candidate.get("decomposition_score") or 0.0)
    local_window = float(candidate.get("local_window_score") or 0.0)
    if benchmark_name == "locomo":
        return round(
            semantic
            + bm25 * 0.12
            + exact * 0.18
            + entity * 0.18
            + temporal * 0.2
            + session * 0.3
            + temporal_neighbor * 0.16
            + parent_score * 0.18
            + decomposition * 0.1
            + support_bonus,
            4,
        )
    if benchmark_name == "knowme":
        return round(
            semantic
            + bm25 * 0.12
            + exact * 0.22
            + entity * 0.16
            + temporal * 0.14
            + profile * 0.32
            + parent_score * 0.16
            + decomposition * 0.16
            + support_bonus,
            4,
        )
    if benchmark_name == "clonemem":
        return round(
            semantic
            + bm25 * 0.12
            + exact * 0.24
            + entity * 0.16
            + temporal * 0.14
            + local_window * 0.24
            + temporal_neighbor * 0.24
            + parent_score * 0.2
            + decomposition * 0.14
            + speaker * 0.1
            + support_bonus,
            4,
        )
    return round(
        semantic
        + bm25 * 0.18
        + exact * 0.16
        + entity * 0.22
        + temporal * 0.18
        + profile * 0.2
        + session * 0.18
        + local_window * 0.18
        + support_bonus,
        4,
    )


def _clonemem_evidence_blend_score(
    base_score: float,
    candidate: dict[str, Any],
    alpha: float,
    *,
    min_broad_rank: int = 6,
    max_broad_rank: int = 20,
) -> float:
    broad_rank = int(candidate.get("broad_rank") or 0)
    min_broad_rank = max(1, int(min_broad_rank))
    max_broad_rank = max(min_broad_rank, int(max_broad_rank))
    if broad_rank <= 0 or broad_rank < min_broad_rank or broad_rank > max_broad_rank:
        return round(float(base_score), 4)
    dense = float(candidate.get("dense_score") or 0.0)
    bm25 = float(candidate.get("bm25_score") or 0.0)
    exact = float(candidate.get("exact_phrase_score") or 0.0)
    parent_score = float(candidate.get("parent_score") or 0.0)
    decomposition = float(candidate.get("decomposition_score") or 0.0)
    broad = min(1.0, float(candidate.get("broad_score") or 0.0))
    support_bonus = min(0.06, max(0, int(candidate.get("support_count") or _support_count(candidate)) - 1) * 0.012)
    evidence_score = (
        0.38 * dense
        + 0.24 * broad
        + 0.1 * bm25
        + 0.08 * exact
        + 0.08 * parent_score
        + 0.06 * decomposition
        + support_bonus
    )
    alpha = max(0.0, min(1.0, float(alpha)))
    return round((1.0 - alpha) * float(base_score) + alpha * evidence_score, 4)


def _rerank_score(benchmark_name: str, candidate: dict[str, Any], policy: dict[str, Any] | None = None) -> float:
    policy = policy or {}
    dense = float(candidate.get("dense_score") or 0.0)
    bm25 = float(candidate.get("bm25_score") or 0.0)
    entity = float(candidate.get("entity_score") or 0.0)
    temporal = float(candidate.get("temporal_score") or 0.0)
    profile = float(candidate.get("profile_score") or 0.0)
    session = float(candidate.get("session_score") or 0.0)
    exact = float(candidate.get("exact_phrase_score") or 0.0)
    temporal_neighbor = float(candidate.get("temporal_neighbor_score") or candidate.get("local_window_score") or 0.0)
    parent_score = float(candidate.get("parent_score") or 0.0)
    decomposition = float(candidate.get("decomposition_score") or 0.0)
    local_window = float(candidate.get("local_window_score") or 0.0)
    broad = min(1.0, float(candidate.get("broad_score") or 0.0))
    semantic = float(candidate.get("semantic_score") or _semantic_score(candidate))
    task = float(candidate.get("task_score") or _task_score(benchmark_name, candidate))
    support_bonus = min(0.08, max(0, int(candidate.get("support_count") or _support_count(candidate)) - 1) * 0.012)
    ordering_bonus = _ordering_anchor_bonus(benchmark_name, candidate)
    if benchmark_name == "locomo":
        return round(
            0.48 * semantic
            + 0.12 * task
            + 0.1 * broad
            + 0.07 * entity
            + 0.08 * temporal
            + 0.02 * session
            + 0.04 * parent_score
            + 0.03 * temporal_neighbor
            + 0.08 * decomposition
            + 0.1 * exact
            + 0.12 * bm25
            + support_bonus
            + ordering_bonus,
            4,
        )
    if benchmark_name == "knowme":
        return round(
            0.41 * semantic
            + 0.16 * task
            + 0.08 * broad
            + 0.09 * profile
            + 0.07 * exact
            + 0.06 * entity
            + 0.05 * temporal
            + 0.04 * parent_score
            + 0.04 * decomposition
            + 0.04 * bm25
            + support_bonus
            + ordering_bonus,
            4,
        )
    if benchmark_name == "clonemem":
        lexical_gate = max(0.0, min(1.0, float(candidate.get("clonemem_lexical_anchor_gate_factor") or 1.0)))
        bm25_component = 0.08 * bm25 * lexical_gate
        base_score = round(
            0.4 * semantic
            + 0.28 * task
            + 0.12 * broad
            + bm25_component
            + 0.06 * exact
            + 0.06 * local_window
            + 0.05 * temporal_neighbor
            + 0.05 * parent_score
            + 0.04 * decomposition
            + support_bonus
            + ordering_bonus,
            4,
        )
        if bool(policy.get("clonemem_evidence_blend_rerank_enabled", False)):
            return _clonemem_evidence_blend_score(
                base_score,
                candidate,
                float(policy.get("clonemem_evidence_blend_rerank_alpha", 0.35)),
                min_broad_rank=int(policy.get("clonemem_evidence_blend_min_broad_rank", 6)),
                max_broad_rank=int(policy.get("clonemem_evidence_blend_max_broad_rank", 20)),
            )
        return base_score
    if benchmark_name == "longmemeval":
        return round(
            0.58 * semantic
            + 0.16 * task
            + 0.12 * broad
            + 0.06 * bm25
            + 0.04 * dense
            + support_bonus
            + ordering_bonus,
            4,
        )
    return round(
        0.56 * semantic
        + 0.18 * task
        + 0.12 * broad
        + 0.08 * bm25
        + support_bonus,
        4,
    )


def _redundancy(a: dict[str, Any], b: dict[str, Any]) -> float:
    if str(a.get("source_id") or "") == str(b.get("source_id") or ""):
        return 0.0
    text_a = str(a.get("normalized_text") or "")
    text_b = str(b.get("normalized_text") or "")
    if not text_a or not text_b:
        return 0.0
    score = lexical_score(text_a[:500], text_b[:500])
    if str(a.get("source_doc_id") or "") and str(a.get("source_doc_id") or "") == str(b.get("source_doc_id") or ""):
        score += 0.12
    if str(a.get("session_id") or "") and str(a.get("session_id") or "") == str(b.get("session_id") or ""):
        score += 0.08
    return min(1.0, score)


def apply_inhibition_audit(
    benchmark_name: str,
    reranked: list[dict[str, Any]],
    *,
    safe_mode: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    adjusted: list[dict[str, Any]] = []
    if safe_mode:
        base_penalty = 0.012 if benchmark_name in {"knowme", "clonemem"} else 0.01
    else:
        base_penalty = 0.05 if benchmark_name == "clonemem" else 0.035
    route_gated_skip_count = 0
    for rank, candidate in enumerate(reranked, start=1):
        item = dict(candidate)
        item["rank_before_inhibition"] = rank
        penalty = 0.0
        for chosen in selected[:6]:
            redundancy = _redundancy(item, chosen)
            same_parent = bool(_parent_id(item) and _parent_id(item) == _parent_id(chosen))
            if safe_mode and not same_parent and redundancy < 0.82:
                route_gated_skip_count += 1
                continue
            pair_penalty = redundancy * base_penalty
            if safe_mode and same_parent:
                pair_penalty *= 0.65 if benchmark_name == "clonemem" else 0.8
            penalty = max(penalty, pair_penalty)
        if int(item.get("broad_rank") or 0) and int(item.get("broad_rank") or 0) <= 10:
            penalty *= 0.35 if safe_mode else 0.6
        if int(item.get("support_count") or 0) >= 3:
            penalty *= 0.5 if safe_mode else 0.7
        item["inhibition_penalty"] = round(min(0.08 if safe_mode else 0.18, penalty), 4)
        item["post_inhibition_score"] = round(float(item.get("rerank_score") or 0.0) - float(item["inhibition_penalty"]), 4)
        adjusted.append(item)
        selected.append(item)
    adjusted.sort(key=lambda row: float(row.get("post_inhibition_score") or 0.0), reverse=True)
    for rank, item in enumerate(adjusted, start=1):
        item["rank_after_inhibition"] = rank
    return adjusted, {
        "base_penalty": base_penalty,
        "candidate_count": len(adjusted),
        "positive_penalty_count": sum(1 for row in adjusted if float(row.get("inhibition_penalty") or 0.0) > 0.0),
        "safe_mode": safe_mode,
        "route_gated_skip_count": route_gated_skip_count,
    }


def candidate_rows(
    candidates: list[dict[str, Any]],
    limit: int = 200,
    *,
    gold_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gold_ids = set(gold_ids or set())
    for rank, candidate in enumerate(candidates[: max(0, limit)], start=1):
        preview = " ".join(str(candidate.get("text") or "").split())
        if len(preview) > 220:
            preview = preview[:217].rstrip() + "..."
        source_id = str(candidate.get("source_id") or "")
        rows.append(
            {
                "rank": rank,
                "source_id": source_id,
                "source_segment_id": str(candidate.get("source_segment_id") or ""),
                "source_doc_id": str(candidate.get("source_doc_id") or ""),
                "benchmark_name": str(candidate.get("benchmark_name") or ""),
                "sample_id": str(candidate.get("sample_id") or ""),
                "conversation_id": str(candidate.get("conversation_id") or ""),
                "session_id": str(candidate.get("session_id") or ""),
                "turn_id": str(candidate.get("turn_id") or ""),
                "speaker_id": str(candidate.get("speaker_id") or ""),
                "timestamp": str(candidate.get("timestamp") or ""),
                "order_index": int(candidate.get("order_index") or 0),
                "best_chunk_id": str(candidate.get("best_chunk_id") or ""),
                "source_chunk_ids": list(candidate.get("source_chunk_ids") or []),
                "source_retrievers": list(candidate.get("source_retrievers") or []),
                "cluster_id": str(candidate.get("cluster_id") or _cluster_id(candidate)),
                "support_count": int(candidate.get("support_count") or _support_count(candidate)),
                "broad_rank": int(candidate.get("broad_rank") or 0),
                "dense_score": round(float(candidate.get("dense_score") or 0.0), 4),
                "bm25_score": round(float(candidate.get("bm25_score") or 0.0), 4),
                "entity_score": round(float(candidate.get("entity_score") or 0.0), 4),
                "temporal_score": round(float(candidate.get("temporal_score") or 0.0), 4),
                "profile_score": round(float(candidate.get("profile_score") or 0.0), 4),
                "session_score": round(float(candidate.get("session_score") or 0.0), 4),
                "local_window_score": round(float(candidate.get("local_window_score") or 0.0), 4),
                "temporal_neighbor_score": round(float(candidate.get("temporal_neighbor_score") or 0.0), 4),
                "parent_score": round(float(candidate.get("parent_score") or 0.0), 4),
                "decomposition_score": round(float(candidate.get("decomposition_score") or 0.0), 4),
                "exact_phrase_score": round(float(candidate.get("exact_phrase_score") or 0.0), 4),
                "speaker_score": round(float(candidate.get("speaker_score") or 0.0), 4),
                "fusion_score": round(float(candidate.get("fusion_score") or 0.0), 4),
                "semantic_score": round(float(candidate.get("semantic_score") or _semantic_score(candidate)), 4),
                "task_score": round(float(candidate.get("task_score") or _task_score(str(candidate.get("benchmark_name") or ""), candidate)), 4),
                "broad_score": round(float(candidate.get("broad_score") or 0.0), 4),
                "rerank_score": round(float(candidate.get("rerank_score") or 0.0), 4),
                "clonemem_dense_anchor_guard_applied": bool(candidate.get("clonemem_dense_anchor_guard_applied")),
                "clonemem_dense_anchor_guard_floor": round(float(candidate.get("clonemem_dense_anchor_guard_floor") or 0.0), 4),
                "clonemem_lexical_anchor_gate_factor": round(
                    float(candidate.get("clonemem_lexical_anchor_gate_factor") or 1.0),
                    4,
                ),
                "clonemem_lexical_anchor_gate_reason": str(candidate.get("clonemem_lexical_anchor_gate_reason") or ""),
                "clonemem_channel_tail_rescue_reason": str(candidate.get("clonemem_channel_tail_rescue_reason") or ""),
                "post_inhibition_score": round(float(candidate.get("post_inhibition_score") or 0.0), 4),
                "inhibition_penalty": round(float(candidate.get("inhibition_penalty") or 0.0), 4),
                "rank_before_inhibition": int(candidate.get("rank_before_inhibition") or 0),
                "rank_after_inhibition": int(candidate.get("rank_after_inhibition") or 0),
                "final_score": round(
                    float(candidate.get("post_inhibition_score") or candidate.get("rerank_score") or 0.0),
                    4,
                ),
                "is_gold": source_id in gold_ids if gold_ids else False,
                "preview": preview,
            }
        )
    return rows


def rank_benchmark_sources(
    *,
    query: str,
    benchmark_name: str,
    vector_store: VectorStore,
    storage: Storage,
    index_metadata: dict[str, Any],
    route_context: dict[str, Any] | None = None,
    config: AppConfig | None = None,
    pool_limit: int = 240,
    precomputed_dense_hit_lists: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    config = force_benchmark_config(config or AppConfig(base_dir=Path.cwd()))
    if benchmark_name == "clonemem":
        pool_limit = max(300, int(pool_limit))
    pool_limit = max(int(pool_limit), int(config.candidate_recall_eval_k))
    source_records = index_metadata.get("source_records_by_id") or {}
    chunk_metadata_by_id = index_metadata.get("chunk_metadata_by_id") or {}
    timings_ms: dict[str, float] = {}
    stage_start = perf_counter()
    query_meta = _query_features(query, route_context=route_context)
    timings_ms["query_features_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    stage_start = perf_counter()
    side_index = _load_retrieval_side_index(benchmark_name=benchmark_name, index_metadata=index_metadata)
    timings_ms["side_index_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    cache_state = dict(side_index.get("_cache") or {})
    cache_key = str(cache_state.get("cache_key") or "")
    if cache_key and cache_key not in _SIDE_INDEX_LOGGED_KEYS:
        cache_status = str(cache_state.get("status") or "unknown")
        cache_path = str(cache_state.get("path") or "")
        print(
            f"[{benchmark_name}] retrieval side index cache={cache_status}"
            + (f" path={cache_path}" if cache_path else "")
        )
        _SIDE_INDEX_LOGGED_KEYS.add(cache_key)
    policy = _retrieval_policy(config, benchmark_name, pool_limit)
    policy = _apply_route_aware_channel_gating(
        policy,
        query_meta,
        benchmark_name,
        route_context,
    )
    stage_start = perf_counter()
    decomposition, decomposition_cache = _load_query_decomposition(
        query=query,
        benchmark_name=benchmark_name,
        storage=storage,
        side_index=side_index,
        query_features=query_meta,
    )
    timings_ms["query_decomposition_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)

    stage_start = perf_counter()
    dense = (
        _dense_source_candidates(
            query=query,
            benchmark_name=benchmark_name,
            query_features=query_meta,
            vector_store=vector_store,
            chunk_metadata_by_id=chunk_metadata_by_id,
            source_records=source_records,
            limit=int(policy["dense_top_k"]),
            precomputed_hit_lists=precomputed_dense_hit_lists,
        )
        if policy["dense_semantic"]
        else {}
    )
    timings_ms["dense_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    if precomputed_dense_hit_lists is not None:
        timings_ms["dense_prefetched"] = 1.0
    stage_start = perf_counter()
    lexical = (
        _lexical_source_candidates(
            query=query,
            benchmark_name=benchmark_name,
            query_features=query_meta,
            storage=storage,
            chunk_metadata_by_id=chunk_metadata_by_id,
            source_records=source_records,
            limit=int(policy["lexical_top_k"]),
        )
        if policy["lexical_sparse"]
        else {}
    )
    timings_ms["lexical_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    stage_start = perf_counter()
    entity = (
        _entity_source_candidates(
            benchmark_name=benchmark_name,
            query_features=query_meta,
            source_records=source_records,
            side_index=side_index,
            seed_candidates=merge_candidate_sources(dense, lexical),
            limit=int(policy["entity_top_k"]),
        )
        if policy["entity_aware"]
        else {}
    )
    timings_ms["entity_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    stage_start = perf_counter()
    temporal = (
        _temporal_source_candidates(
            benchmark_name=benchmark_name,
            query_features=query_meta,
            source_records=source_records,
            side_index=side_index,
            limit=int(policy["temporal_top_k"]),
        )
        if policy["temporal_anchor"]
        else {}
    )
    timings_ms["temporal_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    stage_start = perf_counter()
    exact = (
        _exact_phrase_source_candidates(
            query=query,
            benchmark_name=benchmark_name,
            query_features=query_meta,
            source_records=source_records,
            side_index=side_index,
            limit=int(policy["exact_phrase_top_k"]),
        )
        if policy["exact_phrase"]
        else {}
    )
    timings_ms["exact_phrase_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    stage_start = perf_counter()
    seed_candidates = merge_candidate_sources(dense, lexical, entity, temporal, exact)
    early_exit = _should_early_exit_retrieval(
        policy=policy,
        benchmark_name=benchmark_name,
        query_features=query_meta,
        seed_candidates=seed_candidates,
        route_context=route_context,
        elapsed_ms=sum(float(value or 0.0) for value in timings_ms.values()),
    )
    if bool(early_exit.get("triggered")):
        for channel in list(early_exit.get("skipped_channels") or []):
            if channel == "profile_side_index_dense":
                policy["profile_side_index_dense"] = False
            else:
                policy[channel] = False

    profile_dense = (
        _profile_source_candidates(
            query=query,
            query_features=query_meta,
            vector_store=vector_store,
            storage=storage,
            chunk_metadata_by_id=chunk_metadata_by_id,
            source_records=source_records,
            limit=int(policy["profile_side_index_top_k"]),
        )
        if policy["profile_side_index"] and bool(policy.get("profile_side_index_dense", True))
        else {}
    )
    profile_side = (
        _profile_side_index_candidates(
            decomposition=decomposition,
            query=query,
            side_index=side_index,
            source_records=source_records,
            limit=int(policy["profile_side_index_top_k"]),
        )
        if policy["profile_side_index"]
        else {}
    )
    profile = merge_candidate_sources(profile_dense, profile_side)
    timings_ms["profile_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    stage_start = perf_counter()
    decomposition_candidates = (
        _query_decomposition_source_candidates(
            benchmark_name=benchmark_name,
            decomposition=decomposition,
            query=query,
            source_records=source_records,
            side_index=side_index,
            limit=int(policy["query_decomposition_top_k"]),
        )
        if policy["query_decomposition"]
        else {}
    )
    timings_ms["decomposition_channel_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)

    seed_candidates = merge_candidate_sources(seed_candidates, profile, decomposition_candidates)
    stage_start = perf_counter()
    session = (
        _session_bundle_source_candidates(
            seed_candidates=seed_candidates,
            source_records=source_records,
            side_index=side_index,
            limit=int(policy["session_bundle_top_k"]),
            query_features=query_meta,
        )
        if policy["session_bundle"]
        else {}
    )
    timings_ms["session_bundle_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    stage_start = perf_counter()
    temporal_neighbor, temporal_neighbor_audit = (
        _temporal_neighbor_source_candidates(
            benchmark_name=benchmark_name,
            seed_candidates=merge_candidate_sources(seed_candidates, session),
            source_records=source_records,
            side_index=side_index,
            query_features=query_meta,
            max_neighbors_per_seed=int(policy["max_neighbors_per_seed"]),
            limit=int(policy["max_total_neighbor_candidates"]),
        )
        if policy["temporal_neighbor"]
        else ({}, {"seed_count": 0, "expanded_candidate_count": 0, "neighbor_hit_but_gold_miss_count": 0})
    )
    timings_ms["temporal_neighbor_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    stage_start = perf_counter()
    parent_candidates, parent_audit = (
        _parent_session_source_candidates(
            benchmark_name=benchmark_name,
            query=query,
            query_features=query_meta,
            decomposition=decomposition,
            seed_candidates=merge_candidate_sources(seed_candidates, session, temporal_neighbor),
            source_records=source_records,
            side_index=side_index,
            parent_top_k=int(policy["parent_top_k"]),
            parent_expand_segments=int(policy["parent_expand_segments"]),
            parent_window_radius=int(policy["parent_window_radius"]),
            parent_anchor_noise_filter_enabled=bool(policy.get("parent_anchor_noise_filter_enabled", True)),
            parent_supplemental_anchor_expansion_enabled=bool(
                policy.get("parent_supplemental_anchor_expansion_enabled", False)
            ),
            parent_supplemental_anchor_expansion_cap=int(policy.get("parent_supplemental_anchor_expansion_cap", 2)),
            parent_timestamp_sibling_expansion_enabled=bool(
                policy.get("clonemem_parent_timestamp_sibling_expansion_enabled", False)
            ),
            parent_timestamp_sibling_expansion_cap=int(
                policy.get("clonemem_parent_timestamp_sibling_expansion_cap", 2)
            ),
            parent_anchor_strict_noise_filter_enabled=bool(
                policy.get("clonemem_parent_anchor_strict_noise_filter_enabled", False)
            ),
        )
        if policy["parent_session"]
        else ({}, {"parent_candidates": [], "selected_parent_count": 0})
    )
    timings_ms["parent_session_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)

    merged = merge_candidate_sources(
        dense,
        lexical,
        entity,
        temporal,
        profile,
        session,
        exact,
        temporal_neighbor,
        parent_candidates,
        decomposition_candidates,
    )
    query_text_hash = str(query_meta.get("query_text_hash") or "")
    for candidate in merged.values():
        if query_text_hash and query_text_hash == str(candidate.get("text_hash") or ""):
            candidate["exact_phrase_score"] = max(float(candidate.get("exact_phrase_score") or 0.0), 1.0)
            candidate["dense_score"] = max(float(candidate.get("dense_score") or 0.0), 0.98)
        candidate["speaker_score"] = _speaker_persona_score(query_meta, candidate)
        candidate["support_count"] = _support_count(candidate)
        candidate["semantic_score"] = _semantic_score(candidate)
        candidate["task_score"] = _task_score(benchmark_name, candidate)
        candidate["cluster_id"] = _cluster_id(candidate)
        candidate["broad_score"] = _broad_score(candidate)
    stage_start = perf_counter()
    broad, fusion_audit, channel_rankings = _fuse_candidate_sources(
        benchmark_name=benchmark_name,
        merged=merged,
        channel_maps={
            "dense_semantic": dense,
            "lexical_sparse": lexical,
            "entity_aware": entity,
            "temporal_anchor": temporal,
            "exact_phrase": exact,
            "profile_side_index": profile,
            "session_bundle": session,
            "temporal_neighbor": temporal_neighbor,
            "parent_session": parent_candidates,
            "query_decomposition": decomposition_candidates,
        },
        policy=policy,
    )
    timings_ms["fusion_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    broad_rank_by_id = {
        str(row.get("source_id") or ""): rank
        for rank, row in enumerate(broad, start=1)
        if str(row.get("source_id") or "")
    }
    for source_id, candidate in merged.items():
        candidate["broad_rank"] = int(broad_rank_by_id.get(source_id) or 0)
        candidate["rerank_score"] = _rank_preserved_rerank_score(
            benchmark_name,
            candidate,
            _rerank_score(benchmark_name, candidate, policy=policy),
            enabled=bool(policy.get("dense_gold_agnostic_rank_floor_enabled", True)),
            policy=policy,
        )
        candidate["post_inhibition_score"] = candidate["rerank_score"]
    stage_start = perf_counter()
    reranked = sorted(broad, key=lambda row: float(row.get("rerank_score") or 0.0), reverse=True)[: max(1, pool_limit)]
    timings_ms["rerank_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    if bool(policy.get("competition_inhibition_enabled", True)):
        stage_start = perf_counter()
        inhibited, inhibition_audit = apply_inhibition_audit(
            benchmark_name,
            reranked[: max(1, pool_limit)],
            safe_mode=bool(policy.get("safe_fusion_enabled", False)),
        )
        final_ranked = sorted(inhibited, key=lambda row: float(row.get("post_inhibition_score") or 0.0), reverse=True)[
            : max(1, pool_limit)
        ]
        timings_ms["inhibition_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
    else:
        inhibition_audit = {
            "base_penalty": 0.0,
            "candidate_count": len(reranked),
            "positive_penalty_count": 0,
            "disabled": True,
        }
        final_ranked = list(reranked)
        timings_ms["inhibition_ms"] = 0.0
    channel_stats: dict[str, Any] = {}
    for channel_name, rows in channel_rankings.items():
        score_key = CHANNEL_SCORE_FIELDS[channel_name]
        scores = [float(row.get(score_key) or 0.0) for row in rows[: max(1, min(50, len(rows)))]]
        channel_stats[channel_name] = {
            "candidate_count": len(rows),
            "top_score": round(max(scores), 4) if scores else 0.0,
            "bottom_score": round(min(scores), 4) if scores else 0.0,
            "top_source_ids": [str(row.get("source_id") or "") for row in rows[:10]],
        }
    high_cost_channels_executed = [
        channel
        for channel, rows in {
            "session_bundle": session,
            "temporal_neighbor": temporal_neighbor,
            "parent_session": parent_candidates,
            "query_decomposition": decomposition_candidates,
            "profile_side_index": profile,
        }.items()
        if rows
    ]
    early_skipped = set(str(channel) for channel in list(early_exit.get("skipped_channels") or []))
    gated_skipped = set(str(channel) for channel in list(policy.get("gated_channels") or []))
    channel_audit_specs = {
        "entity": {
            "enabled": bool(policy.get("entity_aware")),
            "rows": entity,
            "keys": ("entity_to_source_ids", "entity_postings", "token_to_source_ids", "lexical_postings"),
        },
        "temporal": {
            "enabled": bool(policy.get("temporal_anchor")),
            "rows": temporal,
            "keys": ("temporal_term_to_source_ids", "specific_temporal_term_to_source_ids", "timestamp_source_ids"),
            "query_signal": bool(
                query_meta.get("temporal_terms")
                or query_meta.get("specific_temporal_terms")
                or query_meta.get("question_time")
            ),
        },
        "exact_phrase": {
            "enabled": bool(policy.get("exact_phrase")),
            "rows": exact,
            "keys": ("phrase_token_to_source_ids", "token_to_source_ids", "text_hash_to_source_id"),
        },
        "profile_side_index": {
            "enabled": bool(policy.get("profile_side_index")),
            "rows": profile,
            "keys": ("profile_entry_by_id", "profile_entry_ids_by_term", "profile_entry_ids_by_category", "profile_side_entries"),
            "legacy_full_scan_capable": False,
        },
        "query_decomposition": {
            "enabled": bool(policy.get("query_decomposition")),
            "rows": decomposition_candidates,
            "keys": ("token_to_source_ids", "lexical_postings", "entity_to_source_ids", "entity_postings"),
        },
        "session_bundle": {
            "enabled": bool(policy.get("session_bundle")),
            "rows": session,
            "keys": ("session_id_to_source_ids", "source_id_to_session_id"),
        },
        "temporal_neighbor": {
            "enabled": bool(policy.get("temporal_neighbor")),
            "rows": temporal_neighbor,
            "keys": ("neighbor_map",),
            "legacy_full_scan_capable": False,
        },
        "parent_session": {
            "enabled": bool(policy.get("parent_session")),
            "rows": parent_candidates,
            "keys": ("parent_id_to_source_ids", "source_id_to_parent_id", "parent_records_by_id"),
        },
    }
    side_index_audit: dict[str, dict[str, Any]] = {}
    for channel_name, spec in channel_audit_specs.items():
        available = _side_index_has_any(side_index, *tuple(spec.get("keys") or ()))
        enabled = bool(spec.get("enabled"))
        rows = dict(spec.get("rows") or {})
        skipped_by_early_exit = channel_name in early_skipped or (
            channel_name == "profile_side_index" and "profile_side_index_dense" in early_skipped and not rows
        )
        skipped_by_gating = channel_name in gated_skipped
        legacy_fallback_capable = bool(spec.get("legacy_full_scan_capable", True))
        query_signal = bool(spec.get("query_signal", True))
        legacy_fallback_used = bool(
            legacy_fallback_capable
            and query_signal
            and enabled
            and not available
            and not skipped_by_gating
            and not skipped_by_early_exit
        )
        full_scan_count = len(source_records) if legacy_fallback_used else 0
        side_index_audit[channel_name] = {
            "indexed_fast_path_available": bool(available),
            "indexed_fast_path_used": bool(enabled and available and not skipped_by_gating and not skipped_by_early_exit),
            "indexed_candidate_count": len(rows) if available else 0,
            "legacy_fallback_used": legacy_fallback_used,
            "legacy_fallback_reason": "side_index_missing_or_incompatible" if legacy_fallback_used else "",
            "full_scan_record_count": full_scan_count,
            "scored_record_count": len(rows) if available else full_scan_count,
            "returned_candidate_count": len(rows),
            "skipped_by_gating": skipped_by_gating,
            "skipped_by_early_exit": skipped_by_early_exit,
        }
    side_index_audit_summary = {
        "full_scan_channels": [
            channel for channel, audit in side_index_audit.items() if bool(audit.get("legacy_fallback_used"))
        ],
        "full_scan_total_records_scored": sum(int(audit.get("full_scan_record_count") or 0) for audit in side_index_audit.values()),
        "indexed_fast_path_channel_count": sum(1 for audit in side_index_audit.values() if bool(audit.get("indexed_fast_path_used"))),
        "legacy_fallback_channel_count": sum(1 for audit in side_index_audit.values() if bool(audit.get("legacy_fallback_used"))),
    }
    return {
        "broad_candidates": broad,
        "reranked_candidates": reranked,
        "final_candidates": final_ranked,
        "candidate_source_stats": {
            "dense_semantic": len(dense),
            "lexical_sparse": len(lexical),
            "entity_aware": len(entity),
            "temporal_anchor": len(temporal),
            "profile_side_index": len(profile),
            "session_bundle": len(session),
            "temporal_neighbor": len(temporal_neighbor),
            "parent_session": len(parent_candidates),
            "query_decomposition": len(decomposition_candidates),
            "exact_phrase": len(exact),
            "merged": len(merged),
            "dense_chunks": len(dense),
            "sparse_chunks": len(lexical),
            "dense_objects": len(profile_dense),
            "sparse_objects": len(profile_side),
            "merged_chunks": len(merged),
            "rerank_pool_requested": max(1, pool_limit),
            "rerank_pool": len(broad),
            "local_window": len(temporal_neighbor),
        },
        "inhibition_audit": inhibition_audit,
        "fusion_audit": fusion_audit,
        "parent_audit": parent_audit,
        "temporal_neighbor_audit": temporal_neighbor_audit,
        "channel_rankings": {
            name: candidate_rows(rows, limit=max(1, min(200, pool_limit)))
            for name, rows in channel_rankings.items()
        },
        "channel_stats": channel_stats,
        "query_features": {
            "entities": list(query_meta.get("entities") or []),
            "temporal_terms": list(query_meta.get("temporal_terms") or []),
            "specific_temporal_terms": list(query_meta.get("specific_temporal_terms") or []),
            "temporal_direction": str(query_meta.get("temporal_direction") or "unspecified"),
            "phrases": list(query_meta.get("phrases") or []),
            "anchor_terms": list(query_meta.get("anchor_terms") or []),
            "code_like_terms": list(query_meta.get("code_like_terms") or []),
            "metric_like_terms": list(query_meta.get("metric_like_terms") or []),
            "person_state_phrases": list(query_meta.get("person_state_phrases") or []),
        },
        "query_decomposition": decomposition,
        "retrieval_policy": policy,
        "retrieval_policy_before_gating": dict(policy.get("retrieval_policy_before_gating") or {}),
        "retrieval_policy_after_gating": dict(policy.get("retrieval_policy_after_gating") or {}),
        "gated_channels": list(policy.get("gated_channels") or []),
        "gating_reasons": dict(policy.get("gating_reasons") or {}),
        "high_cost_channels_executed": high_cost_channels_executed,
        "early_exit": {
            "early_exit_enabled": bool(policy.get("retrieval_early_exit_enabled", True)),
            "early_exit_triggered": bool(early_exit.get("triggered")),
            "early_exit_reason": str(early_exit.get("reason") or ""),
            "skipped_expensive_channels": list(early_exit.get("skipped_channels") or []),
            "seed_candidate_count_at_exit": int(early_exit.get("seed_candidate_count_at_exit") or len(seed_candidates)),
        },
        "side_index_cache": dict(side_index.get("_cache") or {}),
        "side_index_audit": side_index_audit,
        "side_index_audit_summary": side_index_audit_summary,
        "query_decomposition_cache": decomposition_cache,
        "timings_ms": timings_ms,
    }


def best_gold_rank(rows: list[dict[str, Any]], gold_ids: set[str]) -> int | None:
    for rank, row in enumerate(rows, start=1):
        if str(row.get("source_id") or row.get("source_segment_id") or "") in gold_ids:
            return rank
    return None


def build_oracle_retrieval_report(
    *,
    benchmark_name: str,
    oracle_items: list[dict[str, Any]],
    vector_store: Any,
    storage: Any,
    index_metadata: dict[str, Any] | None,
    config: Any,
    route_context_base: dict[str, Any] | None = None,
    pool_limit: int = 50,
    retrieval_cache: dict[str, dict[str, Any]] | None = None,
    retrieval_mode: str | None = None,
) -> dict[str, Any]:
    oracle_mode = str(retrieval_mode or os.getenv("SPHERE_ORACLE_RETRIEVAL_MODE") or "self_retrieval").strip().lower()
    direct_index_mode = oracle_mode in {"direct_index", "index", "indexed"}
    source_records = (index_metadata or {}).get("source_records_by_id") or {}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    cache_hits = 0
    retrieval_count = 0
    direct_index_count = 0
    for item in oracle_items:
        gold_ids = {str(value) for value in item.get("gold_segment_ids") or [] if str(value)}
        for gold_id in sorted(gold_ids):
            row_key = (str(item.get("query_id") or ""), gold_id)
            if row_key in seen:
                continue
            seen.add(row_key)
            source_record = dict(source_records.get(gold_id) or {})
            query_text = str(source_record.get("text") or source_record.get("original_segment_text") or "").strip()
            base_row = {
                "query_id": str(item.get("query_id") or ""),
                "sample_id": str(item.get("sample_id") or ""),
                "question_type": str(item.get("question_type") or ""),
                "gold_segment_id": gold_id,
            }
            if not query_text:
                rows.append(
                    {
                        **base_row,
                        "found_in_index": False,
                        "self_rank": None,
                        "top1_hit": False,
                        "top5_hit": False,
                        "top10_hit": False,
                    }
                )
                continue
            cached = retrieval_cache.get(gold_id) if retrieval_cache is not None else None
            if cached is None:
                if direct_index_mode:
                    cached = {
                        "found_in_index": True,
                        "self_rank": 1,
                        "top1_hit": True,
                        "top5_hit": True,
                        "top10_hit": True,
                        "oracle_retrieval_mode": "direct_index",
                    }
                    direct_index_count += 1
                else:
                    route_context = {
                        **dict(route_context_base or {}),
                        **dict(item.get("route_context") or {}),
                        "benchmark": benchmark_name,
                        "question_type": "oracle",
                    }
                    oracle_trace = rank_benchmark_sources(
                        query=query_text,
                        benchmark_name=benchmark_name,
                        vector_store=vector_store,
                        storage=storage,
                        index_metadata=index_metadata,
                        config=config,
                        route_context=route_context,
                        pool_limit=pool_limit,
                    )
                    oracle_final_rows = candidate_rows(oracle_trace["final_candidates"], limit=pool_limit)
                    self_rank = best_gold_rank(oracle_final_rows, {gold_id})
                    cached = {
                        "found_in_index": True,
                        "self_rank": self_rank,
                        "top1_hit": bool(self_rank is not None and self_rank <= 1),
                        "top5_hit": bool(self_rank is not None and self_rank <= 5),
                        "top10_hit": bool(self_rank is not None and self_rank <= 10),
                        "oracle_retrieval_mode": "self_retrieval",
                    }
                    retrieval_count += 1
                if retrieval_cache is not None:
                    retrieval_cache[gold_id] = cached
            else:
                cache_hits += 1
            rows.append(
                {
                    **base_row,
                    **cached,
                }
            )
    return {
        "benchmark_name": benchmark_name,
        "oracle_query_count": len(rows),
        "oracle_recall@1": round(sum(1.0 for row in rows if row.get("top1_hit")) / max(1, len(rows)), 4),
        "oracle_recall@5": round(sum(1.0 for row in rows if row.get("top5_hit")) / max(1, len(rows)), 4),
        "oracle_recall@10": round(sum(1.0 for row in rows if row.get("top10_hit")) / max(1, len(rows)), 4),
        "oracle_self_retrieval_count": retrieval_count,
        "oracle_self_retrieval_cache_hits": cache_hits,
        "oracle_self_retrieval_cache_size": len(retrieval_cache or {}),
        "oracle_direct_index_count": direct_index_count,
        "oracle_retrieval_mode": "direct_index" if direct_index_mode else "self_retrieval",
        "rows": rows,
    }


def recall_at(rows: list[dict[str, Any]], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    top = {
        str(row.get("source_id") or row.get("source_segment_id") or "")
        for row in rows[: max(0, k)]
    }
    return len(top & gold_ids) / max(1, len(gold_ids))


def recall_any_at(rows: list[dict[str, Any]], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    top = {
        str(row.get("source_id") or row.get("source_segment_id") or "")
        for row in rows[: max(0, k)]
    }
    return 1.0 if top & gold_ids else 0.0


def ndcg_at(rows: list[dict[str, Any]], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    score = 0.0
    for index, row in enumerate(rows[: max(0, k)], start=1):
        source_id = str(row.get("source_id") or row.get("source_segment_id") or "")
        if source_id in gold_ids:
            score += 1.0 / math.log2(index + 1)
    ideal = 0.0
    for index in range(1, min(len(gold_ids), max(0, k)) + 1):
        ideal += 1.0 / math.log2(index + 1)
    return 0.0 if ideal <= 0.0 else score / ideal


def _gold_parent_ids(gold_ids: set[str], index_metadata: dict[str, Any] | None) -> set[str]:
    if not gold_ids or not index_metadata:
        return set()
    source_records = (index_metadata or {}).get("source_records_by_id") or {}
    parent_ids: set[str] = set()
    for gold_id in gold_ids:
        record = dict(source_records.get(gold_id) or {})
        parent_id = _parent_id(record)
        if parent_id:
            parent_ids.add(parent_id)
    return parent_ids


def _channel_gold_stats(
    *,
    gold_segment_ids: set[str],
    channel_rankings: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    rankings = channel_rankings or {}
    stats: dict[str, Any] = {}
    for channel_name, rows in rankings.items():
        gold_rank = best_gold_rank(rows, gold_segment_ids)
        stats[channel_name] = {
            "candidate_count": len(rows),
            "gold_hit": bool(gold_rank is not None),
            "gold_rank": gold_rank,
        }
    return stats


def _distribution_from_candidates(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    field_names: tuple[str, ...],
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows[: max(0, limit)]:
        for field_name in field_names:
            value = row.get(field_name)
            if isinstance(value, list):
                counter.update(str(item) for item in value if str(item))
            else:
                text = str(value or "")
                if text:
                    counter[text] += 1
    return dict(counter)


def _fusion_gold_filter_flags(
    *,
    gold_segment_ids: set[str],
    channel_stats: dict[str, Any],
    broad_rows: list[dict[str, Any]],
    reranked_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    fusion_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    fusion_audit = dict(fusion_audit or {})
    dense_rank = channel_stats.get("dense_semantic", {}).get("gold_rank")
    fused_rank = best_gold_rank(broad_rows, gold_segment_ids)
    rerank_rank = best_gold_rank(reranked_rows, gold_segment_ids)
    final_rank = best_gold_rank(final_rows, gold_segment_ids)
    removed_candidates = list(fusion_audit.get("duplicate_removed_candidates") or [])
    parent_cap_skipped = list(fusion_audit.get("parent_cap_skipped_candidates") or [])
    gold_ids = {str(gold_id) for gold_id in gold_segment_ids if str(gold_id)}

    def removed_for_reason(items: list[dict[str, Any]], reason: str) -> bool:
        for item in items:
            if str(item.get("reason") or "") != reason:
                continue
            source_id = str(item.get("source_id") or "")
            source_segment_id = str(item.get("source_segment_id") or "")
            if source_id in gold_ids or source_segment_id in gold_ids:
                return True
        return False

    return {
        "dense_gold_rank": dense_rank,
        "fused_gold_rank": fused_rank,
        "dense_hit@100": bool(dense_rank is not None and int(dense_rank) <= 100),
        "fused_hit@100": bool(fused_rank is not None and int(fused_rank) <= 100),
        "gold_removed_by_duplicate_collapse": removed_for_reason(removed_candidates, "duplicate_source_id"),
        "gold_removed_by_near_duplicate_collapse": removed_for_reason(removed_candidates, "near_duplicate_text_hash"),
        "gold_removed_by_parent_cap": removed_for_reason(parent_cap_skipped, "parent_cap"),
        "gold_downranked_by_inhibition": bool(
            rerank_rank is not None and final_rank is not None and int(final_rank) > int(rerank_rank)
        ),
        "channels_that_hit_gold": sorted(
            channel_name
            for channel_name, channel_payload in channel_stats.items()
            if bool(channel_payload.get("gold_hit"))
        ),
    }


def build_query_diagnostic(
    *,
    benchmark_name: str,
    query_id: str,
    query_text: str,
    answer_text: str,
    gold_segment_ids: set[str],
    gold_evidence_ids: set[str],
    broad_rows: list[dict[str, Any]],
    reranked_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    trace: dict[str, Any] | None = None,
    index_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    broad_rank = best_gold_rank(broad_rows, gold_segment_ids)
    rerank_rank = best_gold_rank(reranked_rows, gold_segment_ids)
    final_rank = best_gold_rank(final_rows, gold_segment_ids)
    channel_rankings = dict((trace or {}).get("channel_rankings") or {})
    channel_stats = _channel_gold_stats(gold_segment_ids=gold_segment_ids, channel_rankings=channel_rankings)
    gold_parent_ids = _gold_parent_ids(gold_segment_ids, index_metadata)
    parent_audit = dict((trace or {}).get("parent_audit") or {})
    broad_parent_ids = {str(row.get("cluster_id") or row.get("source_doc_id") or "") for row in broad_rows[:50] if str(row.get("cluster_id") or row.get("source_doc_id") or "")}
    top_parent_distribution = Counter(
        str(row.get("cluster_id") or row.get("source_doc_id") or "")
        for row in broad_rows[:10]
        if str(row.get("cluster_id") or row.get("source_doc_id") or "")
    )
    fusion_audit = dict((trace or {}).get("fusion_audit") or {})
    inhibition_audit = dict((trace or {}).get("inhibition_audit") or {})
    dense_vs_fused = _fusion_gold_filter_flags(
        gold_segment_ids=gold_segment_ids,
        channel_stats=channel_stats,
        broad_rows=broad_rows,
        reranked_rows=reranked_rows,
        final_rows=final_rows,
        fusion_audit=fusion_audit,
    )
    return {
        "benchmark_name": benchmark_name,
        "query_id": query_id,
        "query_text": query_text,
        "answer_text": answer_text,
        "gold_evidence_ids": sorted(gold_evidence_ids),
        "gold_segment_ids": sorted(gold_segment_ids),
        "candidate_recall@10": recall_at(broad_rows, gold_segment_ids, 10),
        "candidate_recall@50": recall_at(broad_rows, gold_segment_ids, 50),
        "candidate_recall@100": recall_at(broad_rows, gold_segment_ids, 100),
        "candidate_recall@200": recall_at(broad_rows, gold_segment_ids, 200),
        "candidate_ndcg@10": ndcg_at(broad_rows, gold_segment_ids, 10),
        "final_recall@10": recall_at(final_rows, gold_segment_ids, 10),
        "final_ndcg@10": ndcg_at(final_rows, gold_segment_ids, 10),
        "gold_rank_before_rerank": broad_rank,
        "gold_rank_after_rerank": rerank_rank,
        "gold_rank_after_inhibition": final_rank,
        "rerank_delta": (broad_rank or 9999) - (rerank_rank or 9999),
        "inhibition_delta": (rerank_rank or 9999) - (final_rank or 9999),
        "channel_stats": channel_stats,
        "dense_gold_rank": dense_vs_fused["dense_gold_rank"],
        "fused_gold_rank": dense_vs_fused["fused_gold_rank"],
        "dense_hit@100": dense_vs_fused["dense_hit@100"],
        "fused_hit@100": dense_vs_fused["fused_hit@100"],
        "gold_removed_by_duplicate_collapse": dense_vs_fused["gold_removed_by_duplicate_collapse"],
        "gold_removed_by_parent_cap": dense_vs_fused["gold_removed_by_parent_cap"],
        "gold_removed_by_near_duplicate_collapse": dense_vs_fused["gold_removed_by_near_duplicate_collapse"],
        "gold_downranked_by_inhibition": dense_vs_fused["gold_downranked_by_inhibition"],
        "channels_that_hit_gold": dense_vs_fused["channels_that_hit_gold"],
        "gold_parent_ids": sorted(gold_parent_ids),
        "gold_parent_hit": bool(gold_parent_ids & broad_parent_ids) if gold_parent_ids else False,
        "parent_hit_segment_miss": bool(gold_parent_ids & broad_parent_ids) and broad_rank is None,
        "parent_segment_debug": {
            "parent_candidates": list(parent_audit.get("parent_candidates") or [])[:20],
            "selected_child_anchors": list(parent_audit.get("selected_child_anchors") or [])[:40],
            "missing_anchor_terms": list(parent_audit.get("missing_anchor_terms") or [])[:40],
        },
        "top_parent_distribution": dict(top_parent_distribution),
        "final_top20_channel_distribution": _distribution_from_candidates(
            final_rows,
            limit=20,
            field_names=("source_retrievers",),
        ),
        "final_top20_parent_distribution": _distribution_from_candidates(
            final_rows,
            limit=20,
            field_names=("source_doc_id", "cluster_id"),
        ),
        "final_top20_session_distribution": _distribution_from_candidates(
            final_rows,
            limit=20,
            field_names=("session_id", "conversation_id"),
        ),
        "fusion_audit": fusion_audit,
        "inhibition_audit": inhibition_audit,
        "temporal_neighbor_audit": dict((trace or {}).get("temporal_neighbor_audit") or {}),
        "parent_audit": parent_audit,
        "timings_ms": dict((trace or {}).get("timings_ms") or {}),
        "failure_type": "ok",
    }


def build_topk_debug_record(
    *,
    benchmark_name: str,
    query_id: str,
    query_text: str,
    answer_text: str,
    gold_segment_ids: set[str],
    failure_type: str,
    broad_rows: list[dict[str, Any]],
    reranked_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    trace: dict[str, Any] | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    return {
        "benchmark_name": benchmark_name,
        "query_id": query_id,
        "query_text": query_text,
        "answer_text": answer_text,
        "gold_segment_ids": sorted(gold_segment_ids),
        "failure_type": failure_type,
        "topk_before_rerank": candidate_rows(broad_rows, limit=top_k, gold_ids=gold_segment_ids),
        "topk_after_rerank": candidate_rows(reranked_rows, limit=top_k, gold_ids=gold_segment_ids),
        "topk_after_inhibition": candidate_rows(final_rows, limit=top_k, gold_ids=gold_segment_ids),
        "channel_stats": dict((trace or {}).get("channel_stats") or {}),
        "channel_rankings": dict((trace or {}).get("channel_rankings") or {}),
        "fusion_audit": dict((trace or {}).get("fusion_audit") or {}),
        "inhibition_audit": dict((trace or {}).get("inhibition_audit") or {}),
        "parent_audit": dict((trace or {}).get("parent_audit") or {}),
        "parent_segment_debug": {
            "parent_candidates": list(dict((trace or {}).get("parent_audit") or {}).get("parent_candidates") or [])[:20],
            "selected_child_anchors": list(dict((trace or {}).get("parent_audit") or {}).get("selected_child_anchors") or [])[:40],
            "missing_anchor_terms": list(dict((trace or {}).get("parent_audit") or {}).get("missing_anchor_terms") or [])[:40],
        },
        "temporal_neighbor_audit": dict((trace or {}).get("temporal_neighbor_audit") or {}),
        "query_features": dict((trace or {}).get("query_features") or {}),
        "query_decomposition": dict((trace or {}).get("query_decomposition") or {}),
    }


def build_query_failure(
    *,
    benchmark_name: str,
    query_id: str,
    query_text: str,
    answer_text: str,
    gold_segment_ids: set[str],
    gold_evidence_ids: set[str],
    broad_rows: list[dict[str, Any]],
    reranked_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    index_metadata: dict[str, Any],
    trace: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    indexed_segment_ids = set(index_metadata.get("indexed_segment_ids") or [])
    channel_rankings = dict((trace or {}).get("channel_rankings") or {})
    channel_stats = _channel_gold_stats(gold_segment_ids=gold_segment_ids, channel_rankings=channel_rankings)
    gold_parent_ids = _gold_parent_ids(gold_segment_ids, index_metadata)
    broad_parent_ids = {
        str(row.get("cluster_id") or row.get("source_doc_id") or "")
        for row in broad_rows[:200]
        if str(row.get("cluster_id") or row.get("source_doc_id") or "")
    }
    if not gold_segment_ids:
        failure_type = "query_gold_mapping_empty"
        return {
            "benchmark_name": benchmark_name,
            "query_id": query_id,
            "query_text": query_text,
            "answer_text": answer_text,
            "gold_evidence_ids": sorted(gold_evidence_ids),
            "gold_segment_ids": [],
            "top10_before_rerank": broad_rows[:10],
            "top10_after_rerank": reranked_rows[:10],
            "top10_after_inhibition": final_rows[:10],
            "top50_candidate_ids": [str(row.get("source_id") or "") for row in broad_rows[:50]],
            "gold_rank_before_rerank": None,
            "gold_rank_after_rerank": None,
            "gold_rank_after_inhibition": None,
            "whether_gold_in_index": False,
            "whether_gold_in_candidate_100": False,
            "whether_gold_in_candidate_200": False,
            "channel_stats": channel_stats,
            "failure_type": failure_type,
            "failure_reason": "query-answer-gold mapping is empty for this benchmark item",
            "top_wrong_candidates": final_rows[:5],
        }
    whether_gold_in_index = bool(gold_segment_ids and gold_segment_ids <= indexed_segment_ids)
    chunk_metadata_by_id = index_metadata.get("chunk_metadata_by_id") or {}
    broad_rank = best_gold_rank(broad_rows, gold_segment_ids)
    rerank_rank = best_gold_rank(reranked_rows, gold_segment_ids)
    final_rank = best_gold_rank(final_rows, gold_segment_ids)
    if final_rank is not None and final_rank <= 10:
        return None
    failure_type = "gold_missing_from_candidate_pool"
    if not whether_gold_in_index:
        failure_type = "gold_missing_from_index"
    elif gold_parent_ids and gold_parent_ids & broad_parent_ids and broad_rank is None:
        failure_type = "parent_hit_segment_miss"
    elif any(
        (not str(meta.get("source_segment_id") or "")) or (not str(meta.get("source_doc_id") or ""))
        for meta in chunk_metadata_by_id.values()
    ):
        failure_type = "chunk_id_segment_id_mapping_error"
    elif broad_rank is not None and broad_rank <= 10 and (rerank_rank is None or rerank_rank > 10):
        failure_type = "reranker_dropped_gold"
    elif rerank_rank is not None and rerank_rank <= 10 and (final_rank is None or final_rank > 10):
        failure_type = "inhibition_suppressed_gold"
    elif benchmark_name == "locomo" and not any(
        "session_bundle" in row.get("source_retrievers", []) for row in broad_rows[:100]
    ):
        failure_type = "session_bundle_missing"
    elif benchmark_name == "knowme" and not any(
        "profile_fact" in row.get("source_retrievers", []) for row in broad_rows[:100]
    ):
        failure_type = "profile_fact_missing"
    elif broad_rank is None or broad_rank > 200:
        failure_type = "gold_missing_from_candidate_pool"
    elif benchmark_name == "clonemem" and any(
        str(row.get("source_id") or "") and not str(row.get("source_segment_id") or "")
        for row in broad_rows[:50]
    ):
        failure_type = "segment_id_mismatch"
    elif benchmark_name == "clonemem" and not bool(channel_stats.get("dense_semantic", {}).get("gold_hit")):
        failure_type = "dense_semantic_miss"
    elif benchmark_name == "clonemem" and not bool(channel_stats.get("lexical_sparse", {}).get("gold_hit")):
        failure_type = "lexical_miss"
    elif benchmark_name == "clonemem" and not bool(channel_stats.get("entity_aware", {}).get("gold_hit")):
        failure_type = "entity_miss"
    elif benchmark_name == "clonemem" and not bool(channel_stats.get("temporal_anchor", {}).get("gold_hit")):
        failure_type = "temporal_miss"
    elif broad_rank is not None and broad_rank <= 50 and not any(
        "bm25" in row.get("source_retrievers", []) or "exact_phrase" in row.get("source_retrievers", [])
        for row in broad_rows[:50]
    ):
        failure_type = "lexical_miss"
    elif benchmark_name == "clonemem" and (
        [
            int(row.get("order_index") or 0)
            for row in broad_rows[:10]
            if row.get("order_index") is not None
        ]
        and (
            max(int(row.get("order_index") or 0) for row in broad_rows[:10] if row.get("order_index") is not None)
            - min(int(row.get("order_index") or 0) for row in broad_rows[:10] if row.get("order_index") is not None)
        )
        <= 16
    ):
        failure_type = "local_candidate_crowding"
    elif any(term in (query_text or "").lower() for term in ("before", "after", "earlier", "later", "first", "last")):
        failure_type = "temporal_wrong_anchor"
    elif not any("entity" in row.get("source_retrievers", []) for row in broad_rows[:50]):
        failure_type = "entity_miss"
    top_wrong = [row for row in final_rows[:10] if str(row.get("source_id") or "") not in gold_segment_ids]
    return {
        "benchmark_name": benchmark_name,
        "query_id": query_id,
        "query_text": query_text,
        "answer_text": answer_text,
        "gold_evidence_ids": sorted(gold_evidence_ids),
        "gold_segment_ids": sorted(gold_segment_ids),
        "top10_before_rerank": broad_rows[:10],
        "top10_after_rerank": reranked_rows[:10],
        "top10_after_inhibition": final_rows[:10],
        "top50_candidate_ids": [str(row.get("source_id") or "") for row in broad_rows[:50]],
        "gold_rank_before_rerank": broad_rank,
        "gold_rank_after_rerank": rerank_rank,
        "gold_rank_after_inhibition": final_rank,
        "whether_gold_in_index": whether_gold_in_index,
        "whether_gold_in_candidate_100": recall_any_at(broad_rows, gold_segment_ids, 100) > 0.0,
        "whether_gold_in_candidate_200": recall_any_at(broad_rows, gold_segment_ids, 200) > 0.0,
        "channel_stats": channel_stats,
        "gold_parent_ids": sorted(gold_parent_ids),
        "gold_parent_hit": bool(gold_parent_ids & broad_parent_ids) if gold_parent_ids else False,
        "failure_type": failure_type,
        "failure_reason": (
            "gold evidence ids are absent from the built index"
            if failure_type == "gold_missing_from_index"
            else "gold parent context was retrieved but the target segment was never admitted"
            if failure_type == "parent_hit_segment_miss"
            else "retrieved chunks cannot be mapped back to stable source ids"
            if failure_type == "chunk_id_segment_id_mapping_error"
            else "gold evidence is missing from the broad candidate pool"
            if failure_type == "gold_missing_from_candidate_pool"
            else "gold evidence was present before rerank but dropped after rerank"
            if failure_type == "reranker_dropped_gold"
            else "gold evidence was present before inhibition but dropped after inhibition"
            if failure_type == "inhibition_suppressed_gold"
            else "session bundle expansion did not contribute candidates"
            if failure_type == "session_bundle_missing"
            else "profile fact retrieval did not contribute candidates"
            if failure_type == "profile_fact_missing"
            else "query-answer-gold mapping is empty for this benchmark item"
            if failure_type == "query_gold_mapping_empty"
            else "segment-level ids were not preserved consistently through retrieval"
            if failure_type == "segment_id_mismatch"
            else "dense semantic retrieval did not admit the gold segment"
            if failure_type == "dense_semantic_miss"
            else "broad pool lacked lexical anchoring for this query"
            if failure_type == "lexical_miss"
            else "broad pool lacked temporal anchoring for this query"
            if failure_type == "temporal_miss"
            else "top candidates crowded around the wrong local neighborhood"
            if failure_type == "local_candidate_crowding"
            else "temporal anchoring pulled the query to the wrong point in time"
            if failure_type == "temporal_wrong_anchor"
            else "broad pool lacked entity anchoring for this query"
        ),
        "top_wrong_candidates": top_wrong[:5],
    }


def build_integrity_report(
    *,
    benchmark_name: str,
    raw_counts: dict[str, Any],
    index_metadata: dict[str, Any],
    gold_segment_ids: set[str],
    gold_document_ids: set[str],
) -> dict[str, Any]:
    chunk_metadata_by_id = index_metadata.get("chunk_metadata_by_id") or {}
    indexed_segment_ids = set(index_metadata.get("indexed_segment_ids") or [])
    indexed_doc_ids = set(index_metadata.get("indexed_doc_ids") or [])
    source_segment_counter = Counter(
        str(meta.get("source_segment_id") or "") for meta in chunk_metadata_by_id.values()
    )
    chunk_id_counter = Counter(chunk_metadata_by_id.keys())
    missing_gold_segments = sorted(seg_id for seg_id in gold_segment_ids if seg_id not in indexed_segment_ids)
    missing_gold_documents = sorted(doc_id for doc_id in gold_document_ids if doc_id not in indexed_doc_ids)
    chunks_without_segment = sum(1 for meta in chunk_metadata_by_id.values() if not str(meta.get("source_segment_id") or ""))
    chunks_without_doc = sum(1 for meta in chunk_metadata_by_id.values() if not str(meta.get("source_doc_id") or ""))
    orphan_chunk_count = sum(
        1
        for meta in chunk_metadata_by_id.values()
        if str(meta.get("source_segment_id") or "") not in indexed_segment_ids
    )
    benchmark_mismatch_count = sum(
        1
        for meta in chunk_metadata_by_id.values()
        if str(meta.get("benchmark_name") or "").lower() != benchmark_name.lower()
    )
    p0_bugs: list[str] = []
    if missing_gold_segments:
        p0_bugs.append("gold_segment_not_in_index")
    if chunks_without_segment:
        p0_bugs.append("chunk_missing_source_segment_id")
    if orphan_chunk_count:
        p0_bugs.append("orphan_chunk_detected")
    if benchmark_mismatch_count:
        p0_bugs.append("chunk_benchmark_name_mismatch")
    gold_evidence_count = len(gold_segment_ids)
    gold_coverage = (
        round((gold_evidence_count - len(missing_gold_segments)) / max(1, gold_evidence_count), 4)
        if gold_evidence_count
        else 1.0
    )
    return {
        "benchmark_name": benchmark_name,
        "memory_count": int(raw_counts.get("memory_count") or raw_counts.get("raw_segment_count") or 0),
        "question_count": int(raw_counts.get("question_count") or 0),
        "raw_document_count": int(raw_counts.get("raw_document_count") or 0),
        "raw_session_count": int(raw_counts.get("raw_session_count") or 0),
        "raw_segment_count": int(raw_counts.get("raw_segment_count") or 0),
        "ingested_document_count": int(index_metadata.get("index_doc_count") or 0),
        "ingested_segment_count": int(index_metadata.get("unique_segment_count") or 0),
        "indexed_chunk_count": int(index_metadata.get("chunk_count") or 0),
        "indexed_segment_count": int(index_metadata.get("unique_segment_count") or 0),
        "gold_evidence_count": gold_evidence_count,
        "gold_evidence_ids_count": gold_evidence_count,
        "gold_evidence_coverage": gold_coverage,
        "missing_gold_evidence_count": len(missing_gold_segments),
        "missing_gold_evidence_ids": missing_gold_segments[:100],
        "duplicate_segment_id_count": sum(1 for count in source_segment_counter.values() if count > 1),
        "duplicate_chunk_id_count": sum(1 for count in chunk_id_counter.values() if count > 1),
        "duplicate_raw_segment_id_count": int(raw_counts.get("duplicate_raw_segment_id_count") or 0),
        "duplicate_raw_doc_id_count": int(raw_counts.get("duplicate_raw_doc_id_count") or 0),
        "empty_text_count": int(raw_counts.get("empty_text_count") or 0),
        "timestamp_field_count": int(raw_counts.get("timestamp_field_count") or 0),
        "timestamp_parseable_count": int(raw_counts.get("timestamp_parseable_count") or 0),
        "timestamp_parse_rate": float(raw_counts.get("timestamp_parse_rate") or 0.0),
        "orphan_chunk_count": orphan_chunk_count,
        "chunks_without_source_segment_id_count": chunks_without_segment,
        "chunks_without_source_doc_id_count": chunks_without_doc,
        "target_segment_ids_not_found_in_index": missing_gold_segments[:100],
        "target_document_ids_not_found_in_index": missing_gold_documents[:100],
        "index_benchmark_name_mismatch_count": benchmark_mismatch_count,
        "p0_bugs": p0_bugs,
        "fingerprint": dict(index_metadata.get("fingerprint") or {}),
    }


def build_candidate_recall_summary(
    *,
    benchmark_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_buckets: dict[str, list[float]] = defaultdict(list)
    failure_counter: Counter[str] = Counter()
    failure_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_channel_hit_counter: Counter[str] = Counter()
    per_channel_rank_sum: defaultdict[str, float] = defaultdict(float)
    per_channel_candidate_count: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        for key in (
            "candidate_recall@10",
            "candidate_recall@50",
            "candidate_recall@100",
            "candidate_recall@200",
            "candidate_ndcg@10",
            "final_recall@10",
            "final_ndcg@10",
        ):
            metric_buckets[key].append(float(row.get(key) or 0.0))
        failure_type = str(row.get("failure_type") or "ok")
        failure_counter[failure_type] += 1
        for channel_name, channel_payload in dict(row.get("channel_stats") or {}).items():
            if bool(channel_payload.get("gold_hit")):
                per_channel_hit_counter[channel_name] += 1
            if channel_payload.get("gold_rank") is not None:
                per_channel_rank_sum[channel_name] += float(channel_payload["gold_rank"])
            per_channel_candidate_count[channel_name] += float(channel_payload.get("candidate_count") or 0.0)
        if failure_type != "ok" and len(failure_examples[failure_type]) < 5:
            failure_examples[failure_type].append(
                {
                    "query_id": str(row.get("query_id") or ""),
                    "query_text": str(row.get("query_text") or ""),
                    "gold_rank_before_rerank": row.get("gold_rank_before_rerank"),
                    "gold_rank_after_rerank": row.get("gold_rank_after_rerank"),
                    "gold_rank_after_inhibition": row.get("gold_rank_after_inhibition"),
                    "candidate_recall@100": round(float(row.get("candidate_recall@100") or 0.0), 4),
                    "final_recall@10": round(float(row.get("final_recall@10") or 0.0), 4),
                }
            )
    question_count = max(1, len(rows))
    dense_hit_count = sum(1 for row in rows if bool(row.get("dense_hit@100")))
    fused_hit_count = sum(1 for row in rows if bool(row.get("fused_hit@100")))
    duplicate_removed_count = sum(1 for row in rows if bool(row.get("gold_removed_by_duplicate_collapse")))
    near_duplicate_removed_count = sum(1 for row in rows if bool(row.get("gold_removed_by_near_duplicate_collapse")))
    parent_cap_removed_count = sum(1 for row in rows if bool(row.get("gold_removed_by_parent_cap")))
    inhibition_downrank_count = sum(1 for row in rows if bool(row.get("gold_downranked_by_inhibition")))
    dense_rank_sum = sum(float(row.get("dense_gold_rank") or 0.0) for row in rows if row.get("dense_gold_rank") is not None)
    dense_rank_hits = sum(1 for row in rows if row.get("dense_gold_rank") is not None)
    fused_rank_sum = sum(float(row.get("fused_gold_rank") or 0.0) for row in rows if row.get("fused_gold_rank") is not None)
    fused_rank_hits = sum(1 for row in rows if row.get("fused_gold_rank") is not None)
    return {
        "benchmark_name": benchmark_name,
        "question_count": len(rows),
        "candidate_recall@10": round(sum(metric_buckets["candidate_recall@10"]) / question_count, 4),
        "candidate_recall@50": round(sum(metric_buckets["candidate_recall@50"]) / question_count, 4),
        "candidate_recall@100": round(sum(metric_buckets["candidate_recall@100"]) / question_count, 4),
        "candidate_recall@200": round(sum(metric_buckets["candidate_recall@200"]) / question_count, 4),
        "candidate_ndcg@10": round(sum(metric_buckets["candidate_ndcg@10"]) / question_count, 4),
        "final_recall@10": round(sum(metric_buckets["final_recall@10"]) / question_count, 4),
        "final_ndcg@10": round(sum(metric_buckets["final_ndcg@10"]) / question_count, 4),
        "dense_hit@100": round(dense_hit_count / question_count, 4),
        "fused_hit@100": round(fused_hit_count / question_count, 4),
        "avg_dense_gold_rank_when_hit": round(dense_rank_sum / max(1, dense_rank_hits), 4),
        "avg_fused_gold_rank_when_hit": round(fused_rank_sum / max(1, fused_rank_hits), 4),
        "reranker_dropped_gold_count": int(failure_counter.get("reranker_dropped_gold", 0)),
        "inhibition_suppressed_gold_count": int(failure_counter.get("inhibition_suppressed_gold", 0)),
        "gold_missing_from_index_count": int(failure_counter.get("gold_missing_from_index", 0)),
        "gold_removed_by_duplicate_collapse_count": duplicate_removed_count,
        "gold_removed_by_near_duplicate_collapse_count": near_duplicate_removed_count,
        "gold_removed_by_parent_cap_count": parent_cap_removed_count,
        "gold_downranked_by_inhibition_count": inhibition_downrank_count,
        "failure_type_distribution": dict(failure_counter),
        "failure_type_ratios": {
            key: round(count / question_count, 4)
            for key, count in failure_counter.items()
        },
        "per_channel_gold_hit_rate": {
            channel_name: round(count / question_count, 4)
            for channel_name, count in per_channel_hit_counter.items()
        },
        "per_channel_avg_gold_rank": {
            channel_name: round(per_channel_rank_sum[channel_name] / max(1, per_channel_hit_counter.get(channel_name, 0)), 4)
            for channel_name in per_channel_rank_sum
        },
        "per_channel_avg_candidate_count": {
            channel_name: round(total / question_count, 4)
            for channel_name, total in per_channel_candidate_count.items()
        },
        "representative_failures": dict(failure_examples),
        "queries": rows,
    }


def build_per_channel_contribution_report(
    *,
    benchmark_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    channel_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    channel_counts: Counter[str] = Counter()
    channel_gold_hits: Counter[str] = Counter()
    channel_fused_top10_hits: Counter[str] = Counter()
    channel_gold_rank_sum: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        fused_hit = float(row.get("candidate_recall@10") or 0.0) > 0.0
        for channel_name, channel_payload in dict(row.get("channel_stats") or {}).items():
            channel_counts[channel_name] += 1
            gold_hit = bool(channel_payload.get("gold_hit"))
            if gold_hit:
                channel_gold_hits[channel_name] += 1
                channel_gold_rank_sum[channel_name] += float(channel_payload.get("gold_rank") or 0.0)
                if fused_hit:
                    channel_fused_top10_hits[channel_name] += 1
            if gold_hit and len(channel_examples[channel_name]) < 5:
                channel_examples[channel_name].append(
                    {
                        "query_id": str(row.get("query_id") or ""),
                        "query_text": str(row.get("query_text") or ""),
                        "gold_rank": channel_payload.get("gold_rank"),
                        "fusion_gold_rank": row.get("gold_rank_before_rerank"),
                    }
                )
    return {
        "benchmark_name": benchmark_name,
        "question_count": len(rows),
        "channels": {
            channel_name: {
                "question_coverage": int(channel_counts[channel_name]),
                "gold_hit_rate": round(channel_gold_hits[channel_name] / max(1, channel_counts[channel_name]), 4),
                "avg_gold_rank_when_hit": round(channel_gold_rank_sum[channel_name] / max(1, channel_gold_hits[channel_name]), 4),
                "fused_top10_hit_rate_when_channel_hits": round(channel_fused_top10_hits[channel_name] / max(1, channel_gold_hits[channel_name]), 4),
                "representative_hits": channel_examples[channel_name],
            }
            for channel_name in sorted(channel_counts)
        },
    }


def classify_knowme_query(query_text: str) -> str:
    lowered = normalize_text_for_hash(query_text).lower()
    if any(term in lowered for term in PREFERENCE_QUERY_TERMS):
        return "preference query"
    if any(term in lowered for term in RELATIONSHIP_QUERY_TERMS):
        return "relationship query"
    if any(term in lowered for term in LOCATION_QUERY_TERMS):
        return "location query"
    if any(term in lowered for term in TASK_QUERY_TERMS | IMPLEMENTATION_QUERY_TERMS):
        return "task/project query"
    if any(term in lowered for term in TEMPORAL_TOKEN_RE.findall(lowered)) or DATE_HINT_RE.search(lowered):
        return "temporal event query"
    entity_hits = len(_entity_like_terms(lowered, token_list=tokenize(lowered)))
    if entity_hits >= 2 and " and " in lowered:
        return "ambiguous / multi-hop profile query"
    if entity_hits >= 1:
        return "profile attribute query"
    return "generic semantic query"


def build_knowme_category_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = str(row.get("category") or classify_knowme_query(str(row.get("query_text") or "")))
        buckets[category].append(row)
    payload = {
        "benchmark_name": "knowme",
        "question_count": len(rows),
        "categories": {},
    }
    for category, bucket in sorted(buckets.items()):
        question_count = len(bucket)
        top_channels = Counter()
        failure_modes = Counter()
        representative_examples: list[dict[str, Any]] = []
        for row in bucket:
            failure_modes[str(row.get("failure_type") or "ok")] += 1
            sorted_channels = sorted(
                dict(row.get("channel_stats") or {}).items(),
                key=lambda item: (
                    bool(item[1].get("gold_hit")),
                    -int(item[1].get("gold_rank") or 9999),
                ),
                reverse=True,
            )
            for channel_name, payload_row in sorted_channels[:2]:
                if bool(payload_row.get("gold_hit")):
                    top_channels[channel_name] += 1
            if len(representative_examples) < 5:
                representative_examples.append(
                    {
                        "query_id": str(row.get("query_id") or ""),
                        "query_text": str(row.get("query_text") or ""),
                        "candidate_recall@100": round(float(row.get("candidate_recall@100") or 0.0), 4),
                        "recall_frac@10": round(float(row.get("recall_frac@10") or 0.0), 4),
                        "recall_any@10": round(float(row.get("recall_any@10") or 0.0), 4),
                        "ndcg_any@10": round(float(row.get("ndcg_any@10") or 0.0), 4),
                        "failure_type": str(row.get("failure_type") or "ok"),
                    }
                )
        payload["categories"][category] = {
            "question_count": question_count,
            "candidate_recall@100": round(sum(float(row.get("candidate_recall@100") or 0.0) for row in bucket) / max(1, question_count), 4),
            "recall_frac@10": round(sum(float(row.get("recall_frac@10") or 0.0) for row in bucket) / max(1, question_count), 4),
            "recall_any@10": round(sum(float(row.get("recall_any@10") or 0.0) for row in bucket) / max(1, question_count), 4),
            "ndcg_any@10": round(sum(float(row.get("ndcg_any@10") or 0.0) for row in bucket) / max(1, question_count), 4),
            "top_contributing_channels": [channel for channel, _ in top_channels.most_common(4)],
            "common_failure_modes": dict(failure_modes.most_common(6)),
            "representative_examples": representative_examples,
        }
    return payload


def build_clonemem_failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    taxonomy_counter: Counter[str] = Counter()
    representative_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dense_only_hits = 0
    fused_hits = 0
    parent_hit_segment_miss = 0
    neighbor_hit_gold_miss = 0
    local_crowding = 0
    fusion_up = 0
    fusion_down = 0
    for row in rows:
        failure_type = str(row.get("failure_type") or "ok")
        channel_stats = dict(row.get("channel_stats") or {})
        if failure_type == "ok":
            taxonomy = "ok"
        elif failure_type == "parent_hit_segment_miss" or bool(row.get("parent_hit_segment_miss")):
            taxonomy = "parent_hit_segment_miss"
            parent_hit_segment_miss += 1
        elif not bool(channel_stats.get("dense_semantic", {}).get("gold_hit")):
            taxonomy = "dense_semantic_miss"
        elif not bool(channel_stats.get("lexical_sparse", {}).get("gold_hit")):
            taxonomy = "lexical_miss"
        elif not bool(channel_stats.get("entity_aware", {}).get("gold_hit")):
            taxonomy = "entity_miss"
        elif not bool(channel_stats.get("temporal_anchor", {}).get("gold_hit")):
            taxonomy = "temporal_miss"
        elif int(dict(row.get("fusion_audit") or {}).get("local_crowding_count") or 0) > 0:
            taxonomy = "local_candidate_crowding"
            local_crowding += 1
        elif failure_type == "segment_id_mismatch":
            taxonomy = "granularity_mismatch"
        elif failure_type == "inhibition_suppressed_gold":
            taxonomy = "fusion_downrank"
        else:
            taxonomy = "unknown"
        taxonomy_counter[taxonomy] += 1
        if bool(row.get("dense_hit@100")):
            dense_only_hits += 1
        if bool(row.get("fused_hit@100")) or float(row.get("candidate_recall@100") or 0.0) > 0.0:
            fused_hits += 1
        if bool(channel_stats.get("temporal_neighbor", {}).get("gold_hit")) and not bool(row.get("whether_gold_in_candidate_100")):
            neighbor_hit_gold_miss += 1
        if row.get("gold_rank_before_rerank") is not None and row.get("gold_rank_after_rerank") is not None:
            if int(row["gold_rank_after_rerank"]) < int(row["gold_rank_before_rerank"]):
                fusion_up += 1
            elif int(row["gold_rank_after_rerank"]) > int(row["gold_rank_before_rerank"]):
                fusion_down += 1
        if len(representative_examples[taxonomy]) < 5:
            representative_examples[taxonomy].append(
                {
                    "query_id": str(row.get("query_id") or ""),
                    "query_text": str(row.get("query_text") or ""),
                    "gold_rank_before_rerank": row.get("gold_rank_before_rerank"),
                    "gold_rank_after_rerank": row.get("gold_rank_after_rerank"),
                    "gold_rank_after_inhibition": row.get("gold_rank_after_inhibition"),
                    "top_parent_distribution": dict(row.get("top_parent_distribution") or {}),
                    "channel_stats": channel_stats,
                }
            )
    question_count = max(1, len(rows))
    return {
        "benchmark_name": "clonemem",
        "question_count": len(rows),
        "failure_type_distribution": dict(taxonomy_counter),
        "failure_type_ratios": {
            key: round(count / question_count, 4)
            for key, count in taxonomy_counter.items()
        },
        "dense_only_candidate_hit_rate": round(dense_only_hits / question_count, 4),
        "fused_candidate_hit_rate": round(fused_hits / question_count, 4),
        "parent_hit_segment_miss_ratio": round(parent_hit_segment_miss / question_count, 4),
        "neighbor_hit_gold_miss_ratio": round(neighbor_hit_gold_miss / question_count, 4),
        "local_crowding_ratio": round(local_crowding / question_count, 4),
        "fusion_gold_up_ratio": round(fusion_up / question_count, 4),
        "fusion_gold_down_ratio": round(fusion_down / question_count, 4),
        "representative_examples": dict(representative_examples),
    }


def build_performance_cache_report(
    *,
    benchmark_name: str,
    timing_summary: dict[str, Any],
    reuse_summary: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "benchmark_name": benchmark_name,
        "timing_summary": dict(timing_summary or {}),
        "reuse_summary": dict(reuse_summary or {}),
        "runtime_config": dict(runtime_config or {}),
        "indexing_time_ms": float(timing_summary.get("ingest_ms", 0.0)),
        "retrieval_time_ms": float(timing_summary.get("retrieval_ms", 0.0)),
        "fusion_time_ms": float(timing_summary.get("fusion_ms", 0.0)),
        "rerank_time_ms": float(timing_summary.get("rerank_ms", 0.0)),
        "analysis_time_ms": float(timing_summary.get("json_write_ms", 0.0)),
    }


def build_result_metadata(
    *,
    project_root: Path,
    benchmark_name: str,
    question_count: int,
    vector_info: dict[str, Any],
    index_metadata: dict[str, Any],
    runtime_fingerprint: dict[str, Any],
    determinism: dict[str, Any],
) -> dict[str, Any]:
    index_fingerprint = dict(index_metadata.get("fingerprint") or {})
    ablation = current_benchmark_ablation()
    return {
        "git_sha": current_git_sha(project_root),
        "benchmark_name": benchmark_name,
        "question_count": int(question_count),
        "embedding_provider": str(vector_info.get("embedding_provider") or ""),
        "embedding_model": str(vector_info.get("embedding_model") or ""),
        "embedding_dim": int(vector_info.get("embedding_dim") or 0),
        "embedding_preprocess_version": str(vector_info.get("embedding_preprocess_version") or EMBEDDING_PREPROCESS_VERSION),
        "fallback_in_use": bool(vector_info.get("fallback_in_use", False)),
        "vector_backend": str(vector_info.get("vector_backend") or ""),
        "vector_fallback_in_use": bool(vector_info.get("vector_fallback_in_use", False)),
        "vector_count": int(vector_info.get("vector_count") or vector_info.get("raw_count") or 0),
        "json_scan_warning": str(vector_info.get("json_scan_warning") or ""),
        "index_embedding_provider": str(index_fingerprint.get("embedding_provider") or ""),
        "index_embedding_model": str(index_fingerprint.get("embedding_model") or ""),
        "runtime_embedding_provider": str(runtime_fingerprint.get("embedding_provider") or ""),
        "runtime_embedding_model": str(runtime_fingerprint.get("embedding_model") or ""),
        "chunker_version": str((index_fingerprint.get("chunker") or {}).get("chunker_version") or BENCHMARK_CHUNKER_VERSION),
        "index_built_at": str(index_metadata.get("index_built_at") or ""),
        "index_doc_count": int(index_metadata.get("index_doc_count") or 0),
        "chunk_count": int(index_metadata.get("chunk_count") or 0),
        "unique_segment_count": int(index_metadata.get("unique_segment_count") or 0),
        "random_seed": int(determinism.get("random_seed") or DEFAULT_BENCHMARK_SEED),
        "deterministic_mode": bool(determinism.get("deterministic_mode", True)),
        "ablation": ablation or "full_admission",
        "run_type": ablation or "full_admission",
        "fingerprint": index_fingerprint,
    }
