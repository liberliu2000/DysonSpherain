from __future__ import annotations

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
        task_key = str(task_type or "qa").strip().lower() or "qa"
        tokens = set(token_tuple(query.lower()))
        temporal_hits = len(tokens & _TEMPORAL_TERMS)
        persona_hits = len(tokens & _PERSONA_STATE_TERMS)
        debug_hits = len(tokens & _DEBUG_DESIGN_TERMS)
        creative_hits = len(tokens & _CREATIVE_TERMS)
        exact_hits = len(tokens & _EXACT_TERMS)
        relation_hits = len(tokens & _RELATION_TERMS)
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
        elif (profile is not None and profile.needs_temporal_objects) or temporal_hits >= 1:
            route_type = "temporal"
            confidence = min(0.95, 0.62 + temporal_hits * 0.08 + (0.12 if profile and profile.needs_temporal_objects else 0.0))
            retrieval_intensity = "medium"
            prefer_object_shortcut = bool(profile and profile.temporal_mode in {"latest", "previous", "range"})
            prefer_temporal_prefilter = True
            prefer_segment_rerank = True
            prefer_identity_rerank = bool(profile and profile.query_person_names)
            allow_creative = task_key in {"design", "creative"} and not (profile and profile.needs_exact_evidence)
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
