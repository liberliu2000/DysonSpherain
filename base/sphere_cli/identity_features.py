from __future__ import annotations

import re
from typing import Any

from .utils import token_tuple


COMMON_NAME_STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "who",
    "how",
    "that",
    "this",
    "these",
    "those",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "medium",
    "event",
    "date",
    "time",
    "location",
}
RELATION_TERMS = {
    "father",
    "mother",
    "daughter",
    "son",
    "wife",
    "husband",
    "friend",
    "colleague",
    "boss",
    "manager",
    "teacher",
    "mentor",
    "family",
    "brother",
    "sister",
    "uncle",
    "aunt",
}
ROLE_TERMS = {
    "teacher",
    "manager",
    "engineer",
    "doctor",
    "lawyer",
    "student",
    "director",
    "coach",
    "clerk",
    "worker",
    "professor",
    "designer",
    "developer",
    "father",
    "mother",
    "daughter",
    "son",
}
SOURCE_TERMS = {
    "diary",
    "memo",
    "chat",
    "call",
    "calendar",
    "email",
    "letter",
    "work_log",
    "note",
    "message",
    "phone",
    "wechat",
}
DOMAIN_TERMS: dict[str, set[str]] = {
    "management": {"manager", "management", "team", "work", "project", "deadline", "meeting", "authority"},
    "family": {"family", "father", "mother", "son", "daughter", "wife", "husband", "home"},
    "self_control": {"discipline", "routine", "habit", "self-control", "myself", "restraint", "personal"},
    "control_over_others": {"control", "command", "authority", "supervise", "direct", "manage", "others"},
    "anxiety": {"anxious", "anxiety", "worried", "worry", "nervous", "fear", "panic"},
    "authority": {"authority", "manager", "boss", "director", "teacher", "professor", "official"},
}
SUBTHEME_TERMS: dict[str, set[str]] = {
    "family-control": {"family", "parent", "father", "mother", "control", "home"},
    "self-control": {"discipline", "habit", "restraint", "routine", "self-control"},
    "control-over-others": {"authority", "supervise", "manager", "boss", "direct", "others"},
    "anxiety-driven": {"anxious", "anxiety", "worried", "panic", "fear"},
    "authority-driven": {"authority", "boss", "manager", "order", "command"},
}
STANCE_TERMS: dict[str, set[str]] = {
    "self": {"myself", "self", "own", "habit", "routine"},
    "others": {"others", "team", "staff", "family", "children", "employees"},
}
CAUSAL_DRIVER_TERMS: dict[str, set[str]] = {
    "anxiety": {"because", "anxious", "worried", "fear", "panic"},
    "authority": {"ordered", "required", "authority", "manager", "boss", "command"},
}
IDENTITY_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "with",
    "from",
    "into",
    "about",
    "after",
    "before",
    "during",
    "while",
    "have",
    "has",
    "had",
    "been",
    "that",
    "they",
    "them",
    "their",
    "there",
    "would",
    "could",
    "should",
    "really",
    "over",
    "past",
    "month",
    "months",
    "week",
    "weeks",
    "year",
    "years",
}
NAME_RE = re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Old|Lao)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
HEADER_NAME_RE = re.compile(r"^(?:to|from|with|speaker|caller|recipient)\s*:\s*(?P<name>[A-Z][^\n:]{1,48})$", re.IGNORECASE | re.MULTILINE)
TIME_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?|last\s+\w+|next\s+\w+|today|yesterday|tomorrow)\b")
QUOTED_RE = re.compile(r"\"([^\"]{2,48})\"|“([^”]{2,48})”")


def _normalize_name(raw: str) -> str:
    normalized = re.sub(r"\b(?:mr|mrs|ms|dr|old|lao)\.?\s+", "", raw.strip(), flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.lower()


def _name_candidates(text: str) -> list[str]:
    names: list[str] = []
    for match in NAME_RE.finditer(text):
        value = _normalize_name(match.group(1))
        if not value:
            continue
        parts = value.split()
        if any(part in COMMON_NAME_STOPWORDS for part in parts):
            continue
        if value not in names:
            names.append(value)
    for match in HEADER_NAME_RE.finditer(text):
        value = _normalize_name(match.group("name"))
        if value and value not in names:
            names.append(value)
    return names


def _time_anchors(text: str, metadata: dict[str, Any] | None) -> list[str]:
    anchors: list[str] = []
    for match in TIME_RE.finditer(text):
        item = match.group(0).strip().lower()
        if item and item not in anchors:
            anchors.append(item)
    for key in ("created_at", "timestamp", "time_bucket"):
        value = str((metadata or {}).get(key) or "").strip().lower()
        if value and value not in anchors:
            anchors.append(value)
    return anchors[:8]


def _metadata_entities(metadata: dict[str, Any] | None) -> list[str]:
    entities: list[str] = []
    entity_tags = str((metadata or {}).get("entity_tags") or "").strip().lower()
    for item in entity_tags.split(","):
        value = item.strip()
        if value and value not in entities:
            entities.append(value)
    for key in ("entity", "canonical_key", "source_identity", "speaker_identity"):
        value = str((metadata or {}).get(key) or "").strip().lower()
        if value and value not in entities:
            entities.append(value)
    return entities[:10]


def _discriminative_tokens(text: str) -> list[str]:
    tokens = [
        token
        for token in token_tuple(text.lower())
        if len(token) > 3
        and token not in IDENTITY_STOPWORDS
        and token not in COMMON_NAME_STOPWORDS
    ]
    seen: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
        if len(seen) >= 16:
            break
    return seen


def _term_buckets(tokens: set[str], mapping: dict[str, set[str]]) -> list[str]:
    matched: list[str] = []
    for label, terms in mapping.items():
        if tokens & terms:
            matched.append(label)
    return matched[:6]


def build_identity_features(
    text: str,
    metadata: dict[str, Any] | None = None,
    *,
    focus_names: list[str] | None = None,
) -> dict[str, Any]:
    text = str(text or "")
    lowered = text.lower()
    lowered_tokens = token_tuple(lowered)
    names = _name_candidates(text)
    relations = [token for token in RELATION_TERMS if token in lowered_tokens]
    roles = [token for token in ROLE_TERMS if token in lowered_tokens]
    sources = [token for token in SOURCE_TERMS if token in lowered_tokens]
    for key in ("source_kind", "medium"):
        value = str((metadata or {}).get(key) or "").strip().lower()
        if value and value not in sources:
            sources.append(value)
    entities = _metadata_entities(metadata)
    quoted_terms = [
        str(match.group(1) or match.group(2) or "").strip().lower()
        for match in QUOTED_RE.finditer(text)
        if str(match.group(1) or match.group(2) or "").strip()
    ]
    life_domains = _term_buckets(set(lowered_tokens), DOMAIN_TERMS)
    subthemes = _term_buckets(set(lowered_tokens), SUBTHEME_TERMS)
    interaction_stance = _term_buckets(set(lowered_tokens), STANCE_TERMS)
    causal_drivers = _term_buckets(set(lowered_tokens), CAUSAL_DRIVER_TERMS)
    role_targets = sorted({token for token in relations + roles if token})[:6]
    name_tokens: list[str] = []
    for item in names + list(focus_names or []):
        for token in item.split():
            if token and token not in name_tokens:
                name_tokens.append(token)
    focus_name_tokens: list[str] = []
    for item in focus_names or []:
        normalized = _normalize_name(item)
        for token in normalized.split():
            if token and token not in focus_name_tokens:
                focus_name_tokens.append(token)
    return {
        "names": names,
        "name_tokens": name_tokens,
        "focus_names": [_normalize_name(item) for item in focus_names or [] if str(item).strip()],
        "focus_name_tokens": focus_name_tokens,
        "entities": entities,
        "entity_tokens": sorted({token for item in entities for token in item.split()}),
        "relations": relations,
        "roles": roles,
        "sources": sources[:8],
        "life_domains": life_domains,
        "subthemes": subthemes,
        "interaction_stance": interaction_stance,
        "causal_drivers": causal_drivers,
        "role_targets": role_targets,
        "time_anchors": _time_anchors(text, metadata),
        "quoted_terms": quoted_terms[:8],
        "discriminative_tokens": _discriminative_tokens(text),
        "identity_signature": "|".join(
            filter(
                None,
                [
                    ",".join(names[:4]),
                    ",".join(entities[:4]),
                    ",".join(relations[:4]),
                    ",".join(roles[:4]),
                    ",".join(sources[:4]),
                ],
            )
        ),
    }


def score_identity_alignment(
    query_features: dict[str, Any],
    candidate_features: dict[str, Any],
    *,
    text_similarity: float = 0.0,
) -> dict[str, float]:
    query_name_tokens = set(query_features.get("focus_name_tokens") or query_features.get("name_tokens") or [])
    candidate_name_tokens = set(candidate_features.get("name_tokens") or [])
    query_relations = set(query_features.get("relations") or [])
    candidate_relations = set(candidate_features.get("relations") or [])
    query_roles = set(query_features.get("roles") or [])
    candidate_roles = set(candidate_features.get("roles") or [])
    query_sources = set(query_features.get("sources") or [])
    candidate_sources = set(candidate_features.get("sources") or [])
    query_domains = set(query_features.get("life_domains") or [])
    candidate_domains = set(candidate_features.get("life_domains") or [])
    query_subthemes = set(query_features.get("subthemes") or [])
    candidate_subthemes = set(candidate_features.get("subthemes") or [])
    query_role_targets = set(query_features.get("role_targets") or [])
    candidate_role_targets = set(candidate_features.get("role_targets") or [])
    query_stance = set(query_features.get("interaction_stance") or [])
    candidate_stance = set(candidate_features.get("interaction_stance") or [])
    query_drivers = set(query_features.get("causal_drivers") or [])
    candidate_drivers = set(candidate_features.get("causal_drivers") or [])
    query_times = set(query_features.get("time_anchors") or [])
    candidate_times = set(candidate_features.get("time_anchors") or [])
    query_quotes = set(query_features.get("quoted_terms") or [])
    candidate_quotes = set(candidate_features.get("quoted_terms") or [])
    query_terms = set(query_features.get("discriminative_tokens") or [])
    candidate_terms = set(candidate_features.get("discriminative_tokens") or [])

    name_overlap = len(query_name_tokens & candidate_name_tokens)
    relation_overlap = len(query_relations & candidate_relations)
    role_overlap = len(query_roles & candidate_roles)
    source_overlap = len(query_sources & candidate_sources)
    domain_overlap = len(query_domains & candidate_domains)
    subtheme_overlap = len(query_subthemes & candidate_subthemes)
    role_target_overlap = len(query_role_targets & candidate_role_targets)
    stance_overlap = len(query_stance & candidate_stance)
    driver_overlap = len(query_drivers & candidate_drivers)
    time_overlap = len(query_times & candidate_times)
    quote_overlap = len(query_quotes & candidate_quotes)
    attr_overlap = len(query_terms & candidate_terms) / max(1, min(len(query_terms) or 1, len(candidate_terms) or 1))

    reward = 0.0
    reward += min(0.22, name_overlap * 0.08)
    reward += min(0.08, relation_overlap * 0.04)
    reward += min(0.08, role_overlap * 0.04)
    reward += min(0.06, source_overlap * 0.03)
    reward += min(0.06, domain_overlap * 0.03)
    reward += min(0.07, subtheme_overlap * 0.035)
    reward += min(0.05, role_target_overlap * 0.025)
    reward += min(0.05, stance_overlap * 0.025)
    reward += min(0.05, driver_overlap * 0.025)
    reward += min(0.06, time_overlap * 0.02)
    reward += min(0.06, quote_overlap * 0.03)
    reward += min(0.08, attr_overlap * 0.12)

    wrong_entity_penalty = 0.0
    same_topic_different_entity_penalty = 0.0
    role_source_mismatch_penalty = 0.0
    wrong_domain_penalty = 0.0
    wrong_role_target_penalty = 0.0
    wrong_subtheme_penalty = 0.0
    wrong_stance_penalty = 0.0
    wrong_driver_penalty = 0.0
    quoted_mismatch_penalty = 0.0
    generic_topic_penalty = 0.0

    if query_name_tokens:
        if not name_overlap and candidate_name_tokens:
            wrong_entity_penalty += 0.12
            if text_similarity >= 0.28:
                same_topic_different_entity_penalty += min(0.16, 0.04 + text_similarity * 0.28)
        elif name_overlap == 0 and text_similarity >= 0.38 and attr_overlap >= 0.18:
            wrong_entity_penalty += 0.06
            same_topic_different_entity_penalty += 0.06

    if query_roles and candidate_roles and not role_overlap:
        role_source_mismatch_penalty += 0.06
    if query_sources and candidate_sources and not source_overlap:
        role_source_mismatch_penalty += 0.05
    if query_domains and candidate_domains and not domain_overlap:
        wrong_domain_penalty += 0.05
    if query_subthemes and candidate_subthemes and not subtheme_overlap:
        wrong_subtheme_penalty += 0.06
    if query_role_targets and candidate_role_targets and not role_target_overlap:
        wrong_role_target_penalty += 0.05
    if query_stance and candidate_stance and not stance_overlap:
        wrong_stance_penalty += 0.04
    if query_drivers and candidate_drivers and not driver_overlap:
        wrong_driver_penalty += 0.04
    if query_quotes and candidate_quotes and not quote_overlap:
        quoted_mismatch_penalty += 0.05
    if text_similarity >= 0.22 and len(candidate_terms) <= 2 and len(query_terms) >= 4:
        generic_topic_penalty += 0.03

    return {
        "identity_match_reward": round(reward, 4),
        "wrong_entity_penalty": round(wrong_entity_penalty, 4),
        "same_topic_different_entity_penalty": round(same_topic_different_entity_penalty, 4),
        "role_source_mismatch_penalty": round(role_source_mismatch_penalty, 4),
        "wrong_domain_penalty": round(wrong_domain_penalty, 4),
        "wrong_role_target_penalty": round(wrong_role_target_penalty, 4),
        "wrong_subtheme_penalty": round(wrong_subtheme_penalty, 4),
        "wrong_stance_penalty": round(wrong_stance_penalty, 4),
        "wrong_driver_penalty": round(wrong_driver_penalty, 4),
        "quoted_mismatch_penalty": round(quoted_mismatch_penalty, 4),
        "generic_topic_penalty": round(generic_topic_penalty, 4),
        "name_overlap_count": float(name_overlap),
        "relation_overlap_count": float(relation_overlap),
        "role_overlap_count": float(role_overlap),
        "source_overlap_count": float(source_overlap),
        "domain_overlap_count": float(domain_overlap),
        "subtheme_overlap_count": float(subtheme_overlap),
        "role_target_overlap_count": float(role_target_overlap),
        "stance_overlap_count": float(stance_overlap),
        "driver_overlap_count": float(driver_overlap),
        "time_overlap_count": float(time_overlap),
        "quote_overlap_count": float(quote_overlap),
        "attribute_overlap_ratio": round(attr_overlap, 4),
    }
