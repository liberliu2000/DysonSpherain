from __future__ import annotations

import re
from typing import Any

from .identity_features import build_identity_features, score_identity_alignment
from .models import QueryProfile
from .utils import lexical_score, token_tuple


SEGMENT_SPLIT_RE = re.compile(r"[\n\r]+|(?<=[.!?])\s+")
NEGATION_TERMS = {"not", "never", "no", "without", "avoid", "avoids", "dislike", "dislikes", "hate", "hates"}
PREVIOUS_MARKERS = {"previously", "formerly", "before", "used", "earlier"}
LATEST_MARKERS = {"now", "currently", "latest", "today"}


def candidate_spans(
    text: str,
    *,
    max_spans: int = 10,
    span_lengths: tuple[int, ...] = (1, 2, 3),
) -> list[str]:
    units = [unit.strip() for unit in SEGMENT_SPLIT_RE.split(str(text or "")) if unit and unit.strip()]
    if not units:
        compact = str(text or "").strip()
        return [compact] if compact else []
    spans: list[str] = []
    for span_len in sorted({max(1, int(length)) for length in span_lengths}):
        if span_len == 1:
            for unit in units:
                if unit not in spans:
                    spans.append(unit)
                if len(spans) >= max_spans:
                    return spans[:max_spans]
            continue
        for index in range(0, len(units) - span_len + 1):
            span = " ".join(units[index : index + span_len]).strip()
            if span and span not in spans:
                spans.append(span)
            if len(spans) >= max_spans:
                return spans[:max_spans]
    return spans[:max_spans]


def score_candidate_segments(
    *,
    query: str,
    profile: QueryProfile,
    candidate: dict[str, Any],
    query_features: dict[str, Any],
    focus_names: list[str] | None = None,
    span_lengths: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    effective_span_lengths = sorted({max(1, int(length)) for length in span_lengths})
    if profile.query_person_names or profile.needs_relation_objects or profile.needs_temporal_objects:
        effective_span_lengths = sorted({*effective_span_lengths, 4})
    best: dict[str, Any] = {
        "segment_rerank_score": 0.0,
        "segment_semantic_score": 0.0,
        "segment_identity_score": 0.0,
        "segment_temporal_score": 0.0,
        "segment_attribute_score": 0.0,
        "segment_contradiction_penalty": 0.0,
        "best_span_text": "",
    }
    for span in candidate_spans(str(candidate.get("text") or ""), span_lengths=tuple(effective_span_lengths)):
        semantic_score = lexical_score(query, span)
        span_features = build_identity_features(span, candidate, focus_names=focus_names)
        identity_components = score_identity_alignment(query_features, span_features, text_similarity=semantic_score)
        temporal_score = 0.0
        lowered = span.lower()
        if profile.temporal_reference_terms:
            temporal_score += min(0.12, sum(1 for term in profile.temporal_reference_terms if term in lowered) * 0.05)
        if profile.temporal_mode == "latest" and any(marker in lowered for marker in LATEST_MARKERS):
            temporal_score += 0.06
        if profile.temporal_mode == "previous" and any(marker in lowered for marker in PREVIOUS_MARKERS):
            temporal_score += 0.06
        query_terms = set(profile.attribute_terms or [])
        span_terms = set(token_tuple(lowered))
        attribute_score = 0.0
        if query_terms and span_terms:
            attribute_score = min(0.16, len(query_terms & span_terms) / max(1, len(query_terms)) * 0.18)
        contradiction_penalty = 0.0
        if profile.preference_polarity_hint is not None and (NEGATION_TERMS & span_terms):
            if profile.preference_polarity_hint > 0:
                contradiction_penalty += 0.04
        if profile.temporal_mode == "latest" and any(marker in lowered for marker in PREVIOUS_MARKERS):
            contradiction_penalty += 0.03
        if profile.temporal_mode == "previous" and any(marker in lowered for marker in LATEST_MARKERS):
            contradiction_penalty += 0.03
        identity_score = (
            float(identity_components.get("identity_match_reward") or 0.0)
            - float(identity_components.get("wrong_entity_penalty") or 0.0)
            - float(identity_components.get("same_topic_different_entity_penalty") or 0.0)
            - float(identity_components.get("role_source_mismatch_penalty") or 0.0)
            - float(identity_components.get("wrong_domain_penalty") or 0.0)
            - float(identity_components.get("wrong_role_target_penalty") or 0.0)
            - float(identity_components.get("wrong_subtheme_penalty") or 0.0)
            - float(identity_components.get("wrong_stance_penalty") or 0.0)
            - float(identity_components.get("wrong_driver_penalty") or 0.0)
            - float(identity_components.get("quoted_mismatch_penalty") or 0.0)
            - float(identity_components.get("generic_topic_penalty") or 0.0)
        )
        identity_weight = 0.85 if (profile.query_person_names or profile.needs_relation_objects or profile.needs_temporal_objects) else 0.75
        semantic_weight = 0.54 if identity_weight > 0.75 else 0.58
        final_score = semantic_score * semantic_weight + identity_score * identity_weight + temporal_score + attribute_score - contradiction_penalty
        if final_score > float(best["segment_rerank_score"]):
            best = {
                "segment_rerank_score": round(final_score, 4),
                "segment_semantic_score": round(semantic_score, 4),
                "segment_identity_score": round(identity_score, 4),
                "segment_temporal_score": round(temporal_score, 4),
                "segment_attribute_score": round(attribute_score, 4),
                "segment_contradiction_penalty": round(contradiction_penalty, 4),
                "best_span_text": span[:280],
            }
    return best
