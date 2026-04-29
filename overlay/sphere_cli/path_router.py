from __future__ import annotations

from dataclasses import replace
import os
import re

from .models import QueryProfile, QueryRouteDecision, TaskRoute
from .utils import token_tuple


DEFAULT_ROUTES = {
    "qa": TaskRoute("qa", [0, 1, 2], ["user", "project", "knowledge", "method"], 0.1, "evidence_first"),
    "design": TaskRoute("design", [1, 2, 3], ["project", "knowledge", "method", "creative", "case"], 0.5, "balanced"),
    "debug": TaskRoute("debug", [1, 2, 3, 4], ["project", "case", "method", "raw"], 0.35, "evidence_first"),
    "creative": TaskRoute("creative", [1, 2, 3, 4], ["creative", "case", "knowledge", "project", "raw"], 0.8, "exploratory"),
    "trace": TaskRoute("trace", [1, 2, 4], ["project", "raw", "case", "knowledge"], 0.15, "raw_friendly"),
}

_TEMPORAL_TERMS = {
    "latest",
    "current",
    "currently",
    "now",
    "previous",
    "before",
    "after",
    "yesterday",
    "today",
    "timeline",
    "when",
    "during",
    "last",
    "next",
}
_PERSONA_STATE_TERMS = {
    "prefer",
    "preference",
    "preferences",
    "favorite",
    "favourite",
    "like",
    "likes",
    "avoid",
    "persona",
    "profile",
    "state",
    "status",
    "currently",
    "habit",
    "relation",
    "relationship",
}
_DEBUG_DESIGN_TERMS = {
    "debug",
    "bug",
    "error",
    "exception",
    "traceback",
    "stack",
    "log",
    "logs",
    "design",
    "architecture",
    "refactor",
    "tradeoff",
    "latency",
}
_CREATIVE_TERMS = {
    "idea",
    "ideas",
    "explore",
    "creative",
    "transfer",
    "analogy",
    "contrast",
    "compose",
    "composition",
    "novel",
    "alternative",
}
_EXACT_TERMS = {"what", "which", "who", "where", "exact", "name", "version", "path"}
_RELATION_TERMS = {
    "relation",
    "relationship",
    "partner",
    "spouse",
    "daughter",
    "son",
    "father",
    "mother",
    "wife",
    "husband",
    "friend",
    "colleague",
    "boss",
    "manager",
    "teacher",
    "mentor",
}
_YES_NO_TERMS = {"did", "does", "is", "was", "were", "has", "have", "had", "can", "could"}


def _confusing_cluster_enabled() -> bool:
    raw = str(os.environ.get("SPHERE_ENABLE_CONFUSING_CLUSTER_RERANK", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _temporal_route_enabled() -> bool:
    raw = str(os.environ.get("SPHERE_ENABLE_TEMPORAL_ROUTE", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _effective_profile(profile: QueryProfile | None) -> QueryProfile | None:
    if profile is None or _temporal_route_enabled() or not profile.needs_temporal_objects:
        return profile
    # Disable temporal-route conditioning without mutating the caller-owned profile.
    return replace(profile, needs_temporal_objects=False, temporal_mode="none", temporal_reference_terms=[])


class PathRouter:
    def resolve(self, task_type: str, override_temperature: float | None = None) -> TaskRoute:
        route = DEFAULT_ROUTES.get(task_type, DEFAULT_ROUTES["design"])
        if override_temperature is None:
            return route
        return TaskRoute(
            task_type=route.task_type,
            preferred_shells=route.preferred_shells,
            preferred_sectors=route.preferred_sectors,
            creative_temperature=override_temperature,
            compression_policy=route.compression_policy,
        )

    def route_query(
        self,
        query: str,
        task_type: str,
        profile: QueryProfile | None = None,
    ) -> QueryRouteDecision:
        profile = _effective_profile(profile)
        task_key = str(task_type or "qa").strip().lower() or "qa"
        query_lower = query.lower()
        tokens = set(token_tuple(query_lower))
        temporal_hits = len(tokens & _TEMPORAL_TERMS)
        persona_hits = len(tokens & _PERSONA_STATE_TERMS)
        debug_hits = len(tokens & _DEBUG_DESIGN_TERMS)
        creative_hits = len(tokens & _CREATIVE_TERMS)
        exact_hits = len(tokens & _EXACT_TERMS)
        relation_hits = len(tokens & _RELATION_TERMS)
        has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", query_lower))
        temporal_signal = _temporal_route_enabled() and (temporal_hits >= 1 or has_year or "as of" in query_lower)
        yes_no_temporal_check = bool(profile and profile.needs_exact_evidence and temporal_signal and (tokens & _YES_NO_TERMS))
        preferred_types = set(profile.preferred_object_types) if profile is not None else set()
        explicit_exact_object_lookup = bool(preferred_types & {"artifact", "open_loop"}) and not (
            profile is not None and (profile.needs_preference_objects or profile.needs_personal_context_objects or profile.needs_relation_objects)
        )

        route_type = "exact_factual"
        confidence = 0.54
        retrieval_intensity = "medium"
        prefer_object_shortcut = False
        prefer_temporal_prefilter = False
        prefer_light_rerank = False
        prefer_identity_rerank = False
        prefer_segment_rerank = False
        prefer_confusing_cluster = False
        allow_creative = task_key in {"creative", "design", "debug"}

        if task_key in {"creative"} or creative_hits >= 2:
            route_type = "open_creative_transfer"
            confidence = 0.82 if task_key == "creative" else min(0.94, 0.58 + creative_hits * 0.12)
            retrieval_intensity = "high"
        elif task_key in {"design", "debug"} or debug_hits >= 2:
            route_type = "debug_design"
            confidence = 0.8 if task_key in {"design", "debug"} else min(0.92, 0.58 + debug_hits * 0.11)
            retrieval_intensity = "high" if task_key == "debug" else "medium"
        elif explicit_exact_object_lookup:
            route_type = "exact_factual"
            confidence = min(0.92, 0.68 + exact_hits * 0.05 + (0.08 if "open_loop" in preferred_types else 0.04))
            retrieval_intensity = "light"
            prefer_object_shortcut = True
            prefer_light_rerank = True
            prefer_segment_rerank = True
            prefer_identity_rerank = bool(profile and profile.query_person_names)
            allow_creative = False
        elif (profile is not None and profile.needs_temporal_objects) or temporal_signal:
            route_type = "temporal"
            confidence = min(
                0.96,
                0.64
                + temporal_hits * 0.08
                + (0.1 if has_year else 0.0)
                + (0.06 if profile and profile.needs_temporal_objects else 0.0),
            )
            retrieval_intensity = "medium"
            prefer_object_shortcut = bool(profile and profile.temporal_mode in {"latest", "previous", "range"})
            prefer_temporal_prefilter = True
            prefer_segment_rerank = True
            prefer_identity_rerank = bool(profile and (profile.query_person_names or profile.needs_relation_objects))
            prefer_confusing_cluster = bool(relation_hits or yes_no_temporal_check)
            allow_creative = task_key in {"design", "creative"} and not (profile and profile.needs_exact_evidence)
        elif (
            (profile is not None and (profile.needs_preference_objects or profile.needs_personal_context_objects or profile.needs_relation_objects))
            or persona_hits >= 1
        ):
            route_type = "persona_preference_state"
            confidence = min(
                0.95,
                0.62
                + persona_hits * 0.08
                + (0.12 if profile and profile.needs_preference_objects else 0.0)
                + (0.08 if profile and profile.needs_relation_objects else 0.0),
            )
            retrieval_intensity = "light"
            prefer_object_shortcut = True
            prefer_light_rerank = True
            prefer_segment_rerank = True
            prefer_identity_rerank = bool(profile and (profile.needs_relation_objects or profile.query_person_names))
            allow_creative = False
        else:
            route_type = "exact_factual"
            confidence = min(0.92, 0.6 + exact_hits * 0.06 + (0.08 if profile and profile.needs_exact_evidence else 0.0))
            retrieval_intensity = "light"
            prefer_light_rerank = True
            prefer_identity_rerank = bool(profile and profile.query_person_names)
            prefer_segment_rerank = exact_hits >= 1
            allow_creative = False

        if profile is not None and profile.needs_multi_hop_evidence and route_type not in {"open_creative_transfer", "debug_design"}:
            retrieval_intensity = "medium"
            confidence = min(0.95, confidence + 0.04)
            prefer_confusing_cluster = True
            prefer_segment_rerank = True
        if yes_no_temporal_check and route_type in {"temporal", "exact_factual"}:
            retrieval_intensity = "medium"
            confidence = min(0.95, confidence + 0.03)
            prefer_confusing_cluster = True
            prefer_segment_rerank = True
        if profile is not None and profile.needs_relation_objects:
            prefer_identity_rerank = True
            prefer_confusing_cluster = True
            prefer_segment_rerank = True
        lexical_strength = min(
            1.0,
            (
                temporal_hits
                + persona_hits
                + relation_hits
                + debug_hits
                + creative_hits
                + exact_hits
                + (1 if has_year else 0)
                + (1 if explicit_exact_object_lookup else 0)
                + (1 if profile and profile.needs_exact_evidence else 0)
                + (1 if profile and profile.needs_temporal_objects else 0)
            )
            / 7.0,
        )
        suggested_config = {
            "route_type": route_type,
            "prefer_object_shortcut": prefer_object_shortcut,
            "prefer_temporal_prefilter": prefer_temporal_prefilter,
            "prefer_light_rerank": prefer_light_rerank,
            "prefer_identity_rerank": prefer_identity_rerank,
            "prefer_segment_rerank": prefer_segment_rerank,
            "prefer_confusing_cluster": prefer_confusing_cluster,
            "allow_creative": allow_creative,
            "retrieval_intensity": retrieval_intensity,
            "coarse_topk": 16 if retrieval_intensity == "light" else 24 if retrieval_intensity == "medium" else 36,
            "fine_topk": 6 if retrieval_intensity == "light" else 8 if retrieval_intensity == "medium" else 12,
            "rerank_mode": "light" if prefer_light_rerank else "full",
            "segment_topk": 10 if prefer_segment_rerank else 0,
            "cluster_topk": 16 if prefer_confusing_cluster else 0,
        }
        if not _confusing_cluster_enabled():
            prefer_confusing_cluster = False
            suggested_config["prefer_confusing_cluster"] = False
            suggested_config["cluster_topk"] = 0
        if not _temporal_route_enabled():
            suggested_config["prefer_temporal_prefilter"] = False
        return QueryRouteDecision(
            route_type=route_type,
            confidence=round(confidence, 4),
            normalized_task_type=task_key,
            lexical_strength=round(lexical_strength, 4),
            prefer_object_shortcut=prefer_object_shortcut,
            prefer_temporal_prefilter=prefer_temporal_prefilter,
            prefer_light_rerank=prefer_light_rerank,
            allow_creative=allow_creative,
            retrieval_intensity=retrieval_intensity,
            suggested_config=suggested_config,
            prefer_identity_rerank=prefer_identity_rerank,
            prefer_segment_rerank=prefer_segment_rerank,
            prefer_confusing_cluster=prefer_confusing_cluster,
        )
