from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Protocol, Sequence

from .activation_engine import ActivationEngine
from .benchmark_routes import BenchmarkRouteTuning, resolve_benchmark_route_tuning
from .config import AppConfig
from .creative_reflection_engine import CreativeReflectionEngine
from .identity_features import build_identity_features, score_identity_alignment
from .models import CognitiveAugmentationResult, EvidenceRetrievalResult, QueryProfile, QueryRouteDecision, StructuredCompletionResult
from .path_router import PathRouter
from .prism_propagation_engine import PrismPropagationEngine
from .segment_reranker import score_candidate_segments
from .storage import Storage
from .utils import lexical_score, normalize_text_for_hash, stable_content_hash, token_tuple, tokenize
from .vector_store import VectorStore
from .workspace import WorkspaceContext

TEMPORAL_TERMS = {
    "before",
    "after",
    "latest",
    "current",
    "currently",
    "now",
    "previous",
    "prior",
    "earlier",
    "later",
    "when",
    "timeline",
    "changed",
    "update",
    "updated",
    "moved",
    "switched",
    "ago",
    "yesterday",
    "today",
    "tomorrow",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "day",
    "days",
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
}
PREFERENCE_TERMS = {
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
PREFERENCE_POSITIVE_TERMS = {"prefer", "preferred", "prefers", "favorite", "favourite", "like", "likes", "love", "loves", "enjoy", "enjoys"}
PREFERENCE_NEGATIVE_TERMS = {"dislike", "dislikes", "avoid", "avoids", "hate", "hates", "never"}
PREFERENCE_QUERY_STOPWORDS = {
    "what",
    "which",
    "who",
    "where",
    "does",
    "do",
    "did",
    "is",
    "are",
    "the",
    "a",
    "an",
    "user",
    "my",
    "i",
    "their",
    "preference",
    "preferences",
    "prefers",
    "prefer",
    "likes",
    "like",
    "favorite",
    "favourite",
    "avoid",
    "avoids",
    "dislike",
}
EXACT_TERMS = {
    "exact",
    "which",
    "what",
    "who",
    "where",
    "name",
    "error",
    "version",
    "path",
}
DIAGNOSTIC_TERMS = {"warn", "warning", "error", "exception", "traceback", "stack", "log", "logs"}
ARTIFACT_TERMS = {"artifact", "artifacts", "file", "files", "path", "paths", "document", "documents", "report", "reports", "result", "results"}
OPEN_LOOP_STATUS_TERMS = {"open", "pending", "blocked", "deferred"}
OPEN_LOOP_TERMS = {"todo", "todos", "backlog", "followup", "follow-up", "task", "tasks"} | OPEN_LOOP_STATUS_TERMS
_COMMON_NON_NAMES = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "The", "What", "Which", "Where", "When", "Who", "How", "Does", "Did",
    "This", "That", "Have", "Has", "Are", "Was", "Were", "Can", "Could",
    "Would", "Should", "Will", "Not", "Also", "But", "And", "For",
    "From", "Back", "On",
}
_PERSON_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,15})\b")
ADVICE_TERMS = {"advice", "advise", "suggest", "suggestion", "tips", "tip", "recommend", "recommended", "help", "should"}
TEMPORAL_LATEST_TERMS = {"latest", "current", "currently", "now", "today"}
TEMPORAL_PREVIOUS_TERMS = {"previous", "prior", "before", "earlier", "former"}
TEMPORAL_RANGE_TERMS = {"between", "passed", "since", "until", "from", "to", "duration", "long", "days", "weeks", "months", "years", "after"}
TRACE_REFERENCE_PHRASES = (
    "previous conversation",
    "previous chat",
    "earlier conversation",
    "earlier chat",
    "going back to our previous",
    "looking back at our previous",
    "follow up on our previous conversation",
    "follow-up on our previous conversation",
    "going back to our earlier",
)
FUTURE_PLANNING_PHRASES = (
    "this weekend",
    "next weekend",
    "next week",
    "tonight",
    "tomorrow",
    "upcoming",
)
POINT_TEMPORAL_CUES = (
    "when ",
    "what date",
    "what day",
    "which day",
    "during ",
    "on ",
    "ago",
    "last ",
    "next ",
)
TEMPORAL_QUERY_STOPWORDS = {
    "what",
    "which",
    "who",
    "when",
    "was",
    "were",
    "is",
    "are",
    "did",
    "do",
    "the",
    "a",
    "an",
    "my",
    "i",
    "me",
    "how",
    "many",
    "much",
}
NEGATION_TERMS = {"not", "never", "no", "without", "avoid", "avoids", "dislike", "dislikes", "hate", "hates"}
FORMER_STATE_TERMS = {"used", "previously", "formerly", "prior", "before", "earlier", "old"}
LATEST_STATE_MARKERS = ("is now", "now", "current", "currently", "latest", "updated to", "changed to", "switched to")
PREVIOUS_STATE_MARKERS = ("used to", "previously", "formerly", "prior", "before", "earlier", "was")
PERSONAL_CONTEXT_QUERY_STOPWORDS = {
    "what",
    "which",
    "who",
    "should",
    "would",
    "could",
    "any",
    "tips",
    "tip",
    "help",
    "idea",
    "good",
    "best",
    "for",
    "with",
    "my",
    "i",
    "me",
}
COMMON_QUERY_STOPWORDS = {
    "what", "which", "who", "where", "when", "why", "how",
    "is", "are", "was", "were", "do", "does", "did",
    "the", "a", "an", "to", "of", "for", "with", "and", "or",
    "i", "me", "my", "mine", "you", "your", "their", "our",
}
WEEKDAY_PATTERN = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
MONTH_PATTERN = r"january|february|march|april|may|june|july|august|september|october|november|december"
NUMBER_WORD_PATTERN = r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
TEMPORAL_REFERENCE_PATTERNS = [
    re.compile(r"\b(yesterday|today|tomorrow|tonight|this morning|this afternoon|this evening)\b", re.IGNORECASE),
    re.compile(rf"\b(last\s+(?:{WEEKDAY_PATTERN}|week|month|year|night|weekend))\b", re.IGNORECASE),
    re.compile(rf"\b(next\s+(?:{WEEKDAY_PATTERN}|week|month|year|weekend))\b", re.IGNORECASE),
    re.compile(rf"\b((?:\d+|{NUMBER_WORD_PATTERN})\s+(?:day|days|week|weeks|month|months|year|years)\s+ago)\b", re.IGNORECASE),
    re.compile(rf"\b(in\s+(?:{MONTH_PATTERN})(?:\s+\d{{4}})?)\b", re.IGNORECASE),
    re.compile(rf"\b(on\s+(?:{WEEKDAY_PATTERN}|{MONTH_PATTERN}\s+\d{{1,2}}(?:,\s*\d{{4}})?|\d{{4}}-\d{{2}}-\d{{2}}))\b", re.IGNORECASE),
    re.compile(rf"\b((?:{MONTH_PATTERN})\s+\d{{1,2}}(?:,\s*\d{{4}})?|(?:{MONTH_PATTERN})\s+\d{{4}})\b", re.IGNORECASE),
]
ARTIFACT_FILE_PATTERN = re.compile(r"\.(?:md|markdown|txt|json|yaml|yml|csv|py|ipynb|log|pdf)\b", re.IGNORECASE)
ARTIFACT_QUERY_PHRASES = (
    "which file",
    "what file",
    "which path",
    "what path",
    "which document",
    "what document",
    "which artifact",
    "what artifact",
    "where is the file",
    "where is the artifact",
    "stored in",
    "stores the",
)
OPEN_LOOP_QUERY_PHRASES = (
    "open loop",
    "open-loop",
    "open task",
    "pending task",
    "blocked task",
    "deferred task",
    "next action",
    "next step",
    "remaining task",
    "remaining todo",
    "to-do",
    "todo",
    "still open",
)
TEMPORAL_TASK_TYPES = {"temporal_reasoning", "knowledge_update", "status_lookup", "update_lookup"}
PREFERENCE_TASK_TYPES = {"preference_lookup", "preference", "user_preference"}
EXACT_TASK_TYPES = {"factual_lookup", "trace", "qa"}
ANALYTICAL_TASK_TYPES = {"design", "debug"}
RELATION_TERMS = {
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


class PairPredictor(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Any: ...


class QueryParser:
    def parse(self, query: str, task_type: str = "qa") -> QueryProfile:
        lowered_query = query.lower()
        tokens = set(tokenize(query))
        task_key = task_type.strip().lower()
        relation = self._has_relation_intent(lowered_query, tokens)
        query_person_names = [name for name in _PERSON_NAME_RE.findall(query) if name not in _COMMON_NON_NAMES]
        attribute_terms = [
            token
            for token in tokenize(query)
            if len(token) > 2
            and token not in COMMON_QUERY_STOPWORDS
            and token not in TEMPORAL_QUERY_STOPWORDS
            and token not in PREFERENCE_QUERY_STOPWORDS
        ][:12]
        temporal_reference_terms = self._extract_temporal_reference_terms(lowered_query)
        advice = bool(tokens & ADVICE_TERMS) or "good idea" in lowered_query
        trace_reference = self._looks_like_trace_reference(lowered_query)
        preference = bool(tokens & PREFERENCE_TERMS) or task_key in PREFERENCE_TASK_TYPES
        artifact = self._has_artifact_intent(lowered_query, tokens)
        open_loop = self._has_open_loop_intent(lowered_query, tokens)
        open_loop_status_lookup = self._is_open_loop_status_lookup(
            lowered_query,
            tokens,
            open_loop=open_loop,
            temporal_reference_terms=temporal_reference_terms,
        )
        temporal_mode = self._detect_temporal_mode(
            tokens,
            lowered_query,
            temporal_reference_terms,
            task_key,
            open_loop_status_lookup=open_loop_status_lookup,
        )
        temporal = self._has_temporal_intent(
            tokens=tokens,
            lowered_query=lowered_query,
            temporal_reference_terms=temporal_reference_terms,
            task_key=task_key,
            temporal_mode=temporal_mode,
            preference=preference,
            advice=advice,
            trace_reference=trace_reference,
            open_loop_status_lookup=open_loop_status_lookup,
        )
        if not temporal:
            temporal_mode = "none"
            temporal_reference_terms = []
        preference = bool(tokens & PREFERENCE_TERMS) or task_key in PREFERENCE_TASK_TYPES
        personal_context = advice or task_key in PREFERENCE_TASK_TYPES
        exact = temporal or preference or artifact or open_loop or trace_reference or bool(tokens & EXACT_TERMS) or task_key in EXACT_TASK_TYPES
        multi_hop = temporal or task_key in ANALYTICAL_TASK_TYPES or any(term in tokens for term in ("compare", "difference", "why", "how", "because"))
        if task_key in {"creative"}:
            cognitive = "high"
        elif task_key in ANALYTICAL_TASK_TYPES:
            cognitive = "medium"
        else:
            cognitive = "low"

        granularity_bias = ["micro", "local"] if (temporal or preference or exact) else ["local", "macro"]
        if task_key in ANALYTICAL_TASK_TYPES and "micro" not in granularity_bias:
            granularity_bias = ["local", "micro", "macro"]
        lexical_priority = 0.75 if (temporal or preference or exact) else 0.45
        semantic_priority = 0.9 if task_key in {"qa", "trace", "debug", "temporal_reasoning", "preference_lookup"} else 0.75
        preferred_object_types: list[str] = []
        if preference:
            preferred_object_types.append("preference")
        if temporal:
            preferred_object_types.append("state_update")
            preferred_object_types.append("temporal_reference")
        if personal_context or preference:
            preferred_object_types.append("personal_context")
        if relation:
            preferred_object_types.append("relation")
        if artifact:
            preferred_object_types.append("artifact")
        if open_loop:
            preferred_object_types.append("open_loop")
        if task_key in ANALYTICAL_TASK_TYPES:
            preferred_object_types.append("solution_card")
        preference_polarity_hint: float | None = None
        if (tokens & PREFERENCE_NEGATIVE_TERMS) and not (tokens & PREFERENCE_POSITIVE_TERMS):
            preference_polarity_hint = -1.0
        elif (tokens & PREFERENCE_POSITIVE_TERMS) and not (tokens & PREFERENCE_NEGATIVE_TERMS):
            preference_polarity_hint = 1.0
        return QueryProfile(
            task_type=task_type,
            needs_exact_evidence=exact,
            needs_multi_hop_evidence=multi_hop,
            needs_preference_objects=preference,
            needs_temporal_objects=temporal,
            needs_cognitive_expansion=cognitive,
            granularity_bias=granularity_bias,
            lexical_priority=lexical_priority,
            semantic_priority=semantic_priority,
            preferred_object_types=preferred_object_types,
            preference_polarity_hint=preference_polarity_hint,
            needs_personal_context_objects=personal_context,
            needs_relation_objects=relation,
            temporal_mode=temporal_mode,
            temporal_reference_terms=temporal_reference_terms,
            query_person_names=query_person_names,
            attribute_terms=attribute_terms,
        )

    @staticmethod
    def _has_artifact_intent(lowered_query: str, tokens: set[str]) -> bool:
        if ARTIFACT_FILE_PATTERN.search(lowered_query):
            return True
        if any(phrase in lowered_query for phrase in ARTIFACT_QUERY_PHRASES):
            return True
        if tokens & {"artifact", "artifacts"}:
            return True
        if tokens & ARTIFACT_TERMS and (tokens & EXACT_TERMS or "where" in tokens or "show" in tokens or "find" in tokens):
            return True
        return False

    @staticmethod
    def _has_open_loop_intent(lowered_query: str, tokens: set[str]) -> bool:
        if any(phrase in lowered_query for phrase in OPEN_LOOP_QUERY_PHRASES):
            return True
        if tokens & OPEN_LOOP_STATUS_TERMS:
            if tokens & {"task", "tasks", "item", "items", "loop", "loops", "todo", "todos", "backlog"}:
                return True
            if {"what", "which", "still", "remaining"} & tokens:
                return True
        if "task" in tokens or "tasks" in tokens:
            if tokens & OPEN_LOOP_STATUS_TERMS:
                return True
            if "remaining" in tokens or "unfinished" in tokens:
                return True
        if tokens & {"todo", "todos", "backlog"}:
            return True
        return False

    @staticmethod
    def _is_open_loop_status_lookup(
        lowered_query: str,
        tokens: set[str],
        *,
        open_loop: bool,
        temporal_reference_terms: list[str],
    ) -> bool:
        if not open_loop or temporal_reference_terms:
            return False
        if any(
            phrase in lowered_query
            for phrase in ("before ", "after ", "between ", "since ", "until ", "used to", "changed", "updated", "switched", "timeline")
        ):
            return False
        status_terms = tokens & OPEN_LOOP_STATUS_TERMS
        structure_terms = tokens & {"task", "tasks", "item", "items", "loop", "loops", "todo", "todos", "backlog", "step", "steps", "action", "actions"}
        urgency_terms = {"current", "currently", "now", "right", "still", "remaining", "remains"} & tokens
        phrase_match = any(
            phrase in lowered_query
            for phrase in (
                "right now",
                "currently blocked",
                "currently open",
                "currently deferred",
                "still blocked",
                "still open",
                "remaining task",
                "remaining todo",
                "next action",
                "next step",
            )
        )
        return bool(status_terms or structure_terms) and (bool(urgency_terms) or phrase_match or {"what", "which"} & tokens)

    @staticmethod
    def _has_relation_intent(lowered_query: str, tokens: set[str]) -> bool:
        relation_terms = tokens & RELATION_TERMS
        if not relation_terms:
            return False
        if "relationship" in lowered_query or "relation" in tokens:
            return True
        if "whose" in tokens:
            return True
        explicit_relation_starters = (
            "who is ",
            "who was ",
            "which person is ",
            "which person was ",
            "what is the name of the ",
            "what was the name of the ",
        )
        if any(lowered_query.startswith(prefix) for prefix in explicit_relation_starters):
            return True
        explicit_relation_phrases = (
            "what is his relationship",
            "what was his relationship",
            "what is her relationship",
            "what was her relationship",
            "what is their relationship",
            "what was their relationship",
            "who is his ",
            "who was his ",
            "who is her ",
            "who was her ",
            "who is their ",
            "who was their ",
            "who is my ",
            "who was my ",
            "who is our ",
            "who was our ",
        )
        return any(phrase in lowered_query for phrase in explicit_relation_phrases)

    @staticmethod
    def _looks_like_trace_reference(lowered_query: str) -> bool:
        return any(phrase in lowered_query for phrase in TRACE_REFERENCE_PHRASES)

    def _has_temporal_intent(
        self,
        *,
        tokens: set[str],
        lowered_query: str,
        temporal_reference_terms: list[str],
        task_key: str,
        temporal_mode: str,
        preference: bool,
        advice: bool,
        trace_reference: bool,
        open_loop_status_lookup: bool,
    ) -> bool:
        if task_key in TEMPORAL_TASK_TYPES:
            return True
        if open_loop_status_lookup and not temporal_reference_terms:
            return False
        if any(marker in lowered_query for marker in ("used to", "no longer", "switched to", "switched from", "changed to", "updated to", "currently", "current state")):
            return True
        if trace_reference and temporal_mode in {"previous", "point"}:
            explicit_temporal_lookup = any(cue in lowered_query for cue in ("when ", "what date", "what day", "during ", "ago", "before ", "after ", "latest", "current"))
            if not explicit_temporal_lookup:
                return False
        if temporal_mode in {"latest", "previous", "range"}:
            return True
        if temporal_mode == "point":
            if trace_reference and not any(cue in lowered_query for cue in ("before ", "after ", "ago", "last ", "next ", "earliest", "latest", "during ")):
                return False
            if (preference or advice) and any(phrase in lowered_query for phrase in FUTURE_PLANNING_PHRASES):
                strong_point_lookup = any(cue in lowered_query for cue in ("when ", "what date", "what day", "during ", "ago"))
                if not strong_point_lookup:
                    return False
            return bool(temporal_reference_terms) or any(cue in lowered_query for cue in POINT_TEMPORAL_CUES)
        return False

    def _extract_temporal_reference_terms(self, lowered_query: str) -> list[str]:
        terms: list[str] = []
        for pattern in TEMPORAL_REFERENCE_PATTERNS:
            for match in pattern.finditer(lowered_query):
                term = re.sub(r"\s+", " ", match.group(1)).strip().lower()
                if term and term not in terms:
                    terms.append(term)
        return terms

    def _detect_temporal_mode(
        self,
        tokens: set[str],
        lowered_query: str,
        temporal_reference_terms: list[str],
        task_key: str,
        *,
        open_loop_status_lookup: bool,
    ) -> str:
        if open_loop_status_lookup:
            return "none"
        if task_key in {"knowledge_update", "status_lookup", "update_lookup"} or (tokens & TEMPORAL_LATEST_TERMS):
            return "latest"
        if (tokens & TEMPORAL_PREVIOUS_TERMS) or any(phrase in lowered_query for phrase in ("used to", "no longer", "stopped", "switched from")):
            return "previous"
        range_tokens = tokens & (TEMPORAL_RANGE_TERMS - {"from", "to", "after"})
        if range_tokens or any(phrase in lowered_query for phrase in ("how long", "how many days", "how many months", "how many years", "between ", "since ", "until ", "before ", "after ")):
            return "range"
        if temporal_reference_terms or task_key in TEMPORAL_TASK_TYPES:
            return "point"
        return "none"


class EvidencePipeline:
    def __init__(
        self,
        storage: Storage,
        vector_store: VectorStore,
        activation: ActivationEngine,
        router: PathRouter,
        cross_encoder: PairPredictor | None = None,
        config: AppConfig | None = None,
        creative_engine: CreativeReflectionEngine | None = None,
        prism_engine: PrismPropagationEngine | None = None,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.activation = activation
        self.router = router
        self.parser = QueryParser()
        self.cross_encoder = cross_encoder
        self.config = config or AppConfig()
        self.creative_engine = creative_engine or CreativeReflectionEngine()
        self.prism = prism_engine or PrismPropagationEngine(storage, vector_store, self.creative_engine, self.config)
        self._object_lookup_ms = 0.0
        self._object_support_join_ms = 0.0
        self._object_shortcut_cache: dict[str, dict[str, Any]] = {}
        self._identity_feature_cache: dict[str, dict[str, Any]] = {}
        self._segment_feature_cache: dict[str, dict[str, Any]] = {}
        self._confusing_cluster_cache: dict[str, dict[str, Any]] = {}

    def _snapshot_runtime_stats(self) -> dict[str, Any]:
        storage_stats = self.storage.snapshot_stats(reset=False) if hasattr(self.storage, "snapshot_stats") else {}
        vector_stats = self.vector_store.snapshot_stats(reset=False) if hasattr(self.vector_store, "snapshot_stats") else {}
        return {
            "storage": storage_stats,
            "vector": vector_stats,
        }

    @staticmethod
    def _stats_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
        def diff_bucket(after_bucket: dict[str, Any], before_bucket: dict[str, Any]) -> dict[str, Any]:
            after_ops = after_bucket.get("ops", {}) if after_bucket else {}
            before_ops = before_bucket.get("ops", {}) if before_bucket else {}
            keys = set(after_ops) | set(before_ops)
            ops: dict[str, Any] = {}
            for key in keys:
                a = after_ops.get(key, {})
                b = before_ops.get(key, {})
                total_ms = round(float(a.get("total_ms", 0.0)) - float(b.get("total_ms", 0.0)), 2)
                calls = int(a.get("calls", 0)) - int(b.get("calls", 0))
                rows = int(a.get("rows", 0)) - int(b.get("rows", 0))
                if calls > 0 or rows > 0 or abs(total_ms) > 0.01:
                    ops[key] = {"total_ms": total_ms, "calls": calls, "rows": rows}
            return {
                "total_ms": round(float(after_bucket.get("total_ms", 0.0)) - float(before_bucket.get("total_ms", 0.0)), 2),
                "calls": int(after_bucket.get("calls", 0)) - int(before_bucket.get("calls", 0)),
                "rows": int(after_bucket.get("rows", 0)) - int(before_bucket.get("rows", 0)),
                "ops": ops,
            }

        def diff_numeric_map(after_map: dict[str, Any], before_map: dict[str, Any]) -> dict[str, float]:
            keys = set(after_map) | set(before_map)
            delta: dict[str, float] = {}
            for key in keys:
                value = float(after_map.get(key, 0.0)) - float(before_map.get(key, 0.0))
                if abs(value) > 0.0001:
                    delta[key] = round(value, 2)
            return delta

        return {
            "storage": diff_bucket(after.get("storage", {}), before.get("storage", {})),
            "vector": diff_bucket(after.get("vector", {}), before.get("vector", {})),
            "vector_counters": diff_numeric_map(after.get("vector", {}).get("counters", {}), before.get("vector", {}).get("counters", {})),
            "embedding_cache": diff_numeric_map(after.get("vector", {}).get("embedding_cache", {}), before.get("vector", {}).get("embedding_cache", {})),
        }

    @staticmethod
    def _grain_distribution(candidates: list[dict[str, Any]]) -> dict[str, int]:
        distribution: dict[str, int] = {}
        for candidate in candidates:
            grain = str(candidate.get("grain") or "unknown")
            distribution[grain] = distribution.get(grain, 0) + 1
        return distribution

    @staticmethod
    def _should_use_object_support(
        profile: QueryProfile,
        ranked_candidates: list[dict[str, Any]],
        evidence_top_k: int,
    ) -> tuple[bool, str]:
        if profile.needs_temporal_objects and profile.needs_preference_objects:
            return True, "enabled_temporal_and_preference_profile"
        if profile.needs_temporal_objects:
            return True, "enabled_temporal_profile"
        if profile.needs_preference_objects:
            return True, "enabled_preference_profile"
        if profile.preferred_object_types or profile.needs_multi_hop_evidence:
            return True, "enabled_profile_requires_object_support"
        if not ranked_candidates:
            return True, "no_ranked_candidates"
        top_score = float(ranked_candidates[0].get("evidence_score") or 0.0)
        second_score = float(ranked_candidates[1].get("evidence_score") or 0.0) if len(ranked_candidates) > 1 else 0.0
        third_score = float(ranked_candidates[2].get("evidence_score") or 0.0) if len(ranked_candidates) > 2 else second_score
        top_dense = float(ranked_candidates[0].get("dense_score") or 0.0)
        top_lexical = float(ranked_candidates[0].get("query_lexical") or 0.0)
        margin_12 = top_score - second_score
        margin_13 = top_score - third_score
        if profile.needs_exact_evidence and (margin_12 < 0.07 or margin_13 < 0.1):
            return True, f"enabled_exact_small_margin_m12_{margin_12:.3f}_m13_{margin_13:.3f}"
        if margin_12 < 0.06:
            return True, f"enabled_top2_margin_m12_{margin_12:.3f}"
        if top_score >= 0.78 and margin_12 >= 0.11 and margin_13 >= 0.15 and top_dense >= 0.54 and top_lexical >= 0.2:
            return False, "skipped_clear_dense_lexical_lead"
        if evidence_top_k <= 5 and top_score >= 0.82 and margin_12 >= 0.14 and margin_13 >= 0.18 and top_dense >= 0.58:
            return False, "skipped_small_topk_clear_lead"
        return True, f"enabled_default_margin_m12_{margin_12:.3f}_m13_{margin_13:.3f}"

    @staticmethod
    def _should_use_cross_encoder(profile: QueryProfile, ranked_candidates: list[dict[str, Any]]) -> tuple[bool, str]:
        if not ranked_candidates:
            return False, "no_ranked_candidates"
        if profile.needs_preference_objects or profile.needs_temporal_objects or profile.needs_multi_hop_evidence:
            return True, "required_by_complex_profile"
        top_score = float(ranked_candidates[0].get("evidence_score") or 0.0)
        second_score = float(ranked_candidates[1].get("evidence_score") or 0.0) if len(ranked_candidates) > 1 else 0.0
        if top_score >= 0.8 and (top_score - second_score) >= 0.1:
            return False, "skipped_high_confidence_margin"
        return True, "enabled_uncertain_rank"

    def _resolve_task_route(self, task_type: str) -> Any:
        resolver = getattr(self.router, "resolve", None)
        if callable(resolver):
            return resolver(task_type)
        return type(
            "FallbackTaskRoute",
            (),
            {
                "task_type": task_type,
                "preferred_shells": [],
                "preferred_sectors": [],
                "creative_temperature": 0.2,
                "compression_policy": "balanced",
            },
        )()

    def _decide_query_route(self, query: str, task_type: str, profile: QueryProfile) -> QueryRouteDecision:
        if self.config.enable_task_router:
            router_fn = getattr(self.router, "route_query", None)
            if callable(router_fn):
                try:
                    decision = router_fn(query, task_type, profile=profile)
                    if isinstance(decision, QueryRouteDecision):
                        return decision
                except TypeError:
                    pass
        route_type = "exact_factual"
        preferred_types = set(profile.preferred_object_types)
        explicit_exact_object_lookup = bool(preferred_types & {"artifact", "open_loop"}) and not (
            profile.needs_preference_objects or profile.needs_personal_context_objects or profile.needs_relation_objects
        )
        if explicit_exact_object_lookup:
            route_type = "exact_factual"
        elif profile.needs_temporal_objects:
            route_type = "temporal"
        elif profile.needs_preference_objects or profile.needs_personal_context_objects:
            route_type = "persona_preference_state"
        elif task_type in {"debug", "design"}:
            route_type = "debug_design"
        elif task_type == "creative":
            route_type = "open_creative_transfer"
        return QueryRouteDecision(
            route_type=route_type,
            confidence=0.55,
            normalized_task_type=str(task_type or "qa"),
            lexical_strength=round(min(1.0, lexical_score(query, query)), 4),
            prefer_object_shortcut=route_type == "persona_preference_state"
            or (route_type == "temporal" and profile.temporal_mode in {"latest", "previous", "range"})
            or explicit_exact_object_lookup,
            prefer_temporal_prefilter=route_type == "temporal",
            prefer_light_rerank=route_type in {"exact_factual", "persona_preference_state"},
            allow_creative=route_type in {"debug_design", "open_creative_transfer"},
            retrieval_intensity="high" if route_type in {"debug_design", "open_creative_transfer"} else "medium" if route_type == "temporal" else "light",
            suggested_config={
                "coarse_topk": self.config.retrieval_topk_coarse,
                "fine_topk": self.config.retrieval_topk_fine,
                "rerank_mode": "light" if route_type in {"exact_factual", "persona_preference_state"} else "full",
            },
        )

    def _get_memory_version(self) -> int:
        getter = getattr(self.storage, "get_memory_version", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                return 0
        return 0

    def _cache_is_fresh(self, created_at: str | None, ttl_seconds: int) -> bool:
        if ttl_seconds <= 0:
            return True
        if not created_at:
            return True
        created_epoch = self._timestamp_to_epoch(created_at)
        if created_epoch <= 0:
            return True
        return (datetime.now(timezone.utc).timestamp() - created_epoch) <= float(ttl_seconds)

    @staticmethod
    def _query_route_payload(route: QueryRouteDecision) -> dict[str, Any]:
        return route.to_dict() if hasattr(route, "to_dict") else dict(route or {})

    @staticmethod
    def _route_context_key(route_context: dict[str, Any] | None) -> str:
        if not route_context:
            return ""
        return stable_content_hash(json.dumps(route_context, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _route_query_variants(query: str, route_context: dict[str, Any] | None) -> list[str]:
        variants = [query]
        seen = {normalize_text_for_hash(query).lower()}
        raw_variants: list[Any] = []
        if route_context:
            query_variants = route_context.get("query_variants")
            if isinstance(query_variants, (list, tuple, set)):
                raw_variants.extend(query_variants)
            elif query_variants:
                raw_variants.append(query_variants)
            focused_query = route_context.get("focused_query")
            if focused_query:
                raw_variants.append(focused_query)
        for raw_variant in raw_variants:
            variant = str(raw_variant or "").strip()
            if not variant:
                continue
            normalized_variant = normalize_text_for_hash(variant).lower()
            if not normalized_variant or normalized_variant in seen:
                continue
            seen.add(normalized_variant)
            variants.append(variant)
        return variants

    @staticmethod
    def _route_variant_label(query_variant: str, query: str, route_context: dict[str, Any] | None) -> str:
        normalized_variant = normalize_text_for_hash(query_variant).lower()
        if normalized_variant == normalize_text_for_hash(query).lower():
            return "primary"
        focused_query = str((route_context or {}).get("focused_query") or "").strip()
        if focused_query and normalized_variant == normalize_text_for_hash(focused_query).lower():
            return "focused"
        return "route_context"

    @staticmethod
    def _tag_route_variant_hits(hits: list[dict[str, Any]], route_variant: str) -> list[dict[str, Any]]:
        tagged_hits: list[dict[str, Any]] = []
        for hit in hits:
            tagged_hit = dict(hit)
            tagged_hit["route_variant"] = route_variant
            tagged_hits.append(tagged_hit)
        return tagged_hits

    @staticmethod
    def _route_focus_names(route_context: dict[str, Any] | None, profile: QueryProfile | None) -> list[str]:
        focus_names: list[str] = []
        if route_context:
            for key in ("person_name", "speaker_name", "focus_name", "subject_name", "target_name", "person_aliases"):
                raw_value = route_context.get(key)
                if isinstance(raw_value, (list, tuple, set)):
                    values = [str(item).strip() for item in raw_value if str(item).strip()]
                else:
                    values = re.split(r"\s*(?:,|/|;|\||\band\b)\s*", str(raw_value or "").strip())
                for value in values:
                    cleaned = value.strip()
                    if not cleaned or cleaned.lower() in {"none", "unknown", "n/a"}:
                        continue
                    if cleaned not in focus_names:
                        focus_names.append(cleaned)
        if profile is not None:
            for value in profile.query_person_names:
                if value and value not in focus_names:
                    focus_names.append(value)
        return focus_names

    @staticmethod
    def _workspace_context(route_context: dict[str, Any] | None) -> WorkspaceContext:
        payload = dict(route_context or {})
        return WorkspaceContext.from_values(
            workspace=str(payload.get("workspace") or "").strip() or None,
            project=str(payload.get("project") or "").strip() or None,
            session_id=str(payload.get("session_id") or "").strip() or None,
            scope=str(payload.get("scope") or "").strip() or None,
            scope_order=payload.get("scope_order"),
            mode=str(payload.get("mode") or "").strip() or None,
        )

    def _candidate_in_scope_context(self, candidate: dict[str, Any], route_context: dict[str, Any] | None) -> bool:
        workspace_context = self._workspace_context(route_context)
        rank, _ = workspace_context.candidate_scope_rank(
            scope=str(candidate.get("scope") or ""),
            project=str(candidate.get("project") or ""),
            session_id=str(candidate.get("session_id") or ""),
            workspace=str(candidate.get("workspace") or ""),
        )
        return rank >= -1

    def _filter_hits_by_scope_context(
        self,
        hits: list[dict[str, Any]],
        route_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        workspace_context = self._workspace_context(route_context)
        if not any([workspace_context.workspace, workspace_context.project, workspace_context.session_id]) and workspace_context.scope == "global":
            return hits
        filtered: list[dict[str, Any]] = []
        for hit in hits:
            meta = dict(hit.get("metadata") or {})
            rank, _ = workspace_context.candidate_scope_rank(
                scope=str(meta.get("scope") or ""),
                project=str(meta.get("project") or ""),
                session_id=str(meta.get("session_id") or ""),
                workspace=str(meta.get("workspace") or ""),
            )
            if rank >= -1:
                filtered.append(hit)
        return filtered

    def _persist_recall_trace(
        self,
        *,
        query: str,
        route_context: dict[str, Any] | None,
        diagnostics: dict[str, Any],
        timings_ms: dict[str, float],
        candidates: list[dict[str, Any]],
    ) -> None:
        setter = getattr(self.storage, "set_runtime_state", None)
        if not callable(setter):
            return
        payload = {
            "query": query,
            "route_context": dict(route_context or {}),
            "timings_ms": timings_ms,
            "diagnostics": diagnostics,
            "top_candidates": self._candidate_diagnostic_rows(candidates, limit=min(8, len(candidates))),
        }
        try:
            setter("last_recall_trace", json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            return

    def _identity_cache_key(
        self,
        *,
        text: str,
        metadata: dict[str, Any] | None,
        focus_names: list[str],
        prefix: str,
    ) -> str:
        metadata_signature = ""
        if metadata:
            metadata_signature = stable_content_hash(
                json.dumps(
                    {
                        "entity_tags": metadata.get("entity_tags"),
                        "time_bucket": metadata.get("time_bucket"),
                        "created_at": metadata.get("created_at"),
                        "source_kind": metadata.get("source_kind"),
                        "medium": metadata.get("medium"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return stable_content_hash(
            json.dumps(
                {
                    "prefix": prefix,
                    "text": normalize_text_for_hash(text),
                    "focus_names": list(focus_names),
                    "metadata_signature": metadata_signature,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    def _identity_features(
        self,
        *,
        text: str,
        metadata: dict[str, Any] | None,
        focus_names: list[str],
        prefix: str,
    ) -> dict[str, Any]:
        cache_key = self._identity_cache_key(text=text, metadata=metadata, focus_names=focus_names, prefix=prefix)
        if self.config.enable_identity_feature_cache and cache_key in self._identity_feature_cache:
            return dict(self._identity_feature_cache[cache_key])
        features = build_identity_features(text, metadata, focus_names=focus_names)
        if self.config.enable_identity_feature_cache:
            self._identity_feature_cache[cache_key] = dict(features)
        return features

    def _resolve_route_tuning(
        self,
        *,
        query_route: QueryRouteDecision,
        profile: QueryProfile,
        evidence_top_k: int,
        route_context: dict[str, Any] | None,
    ) -> BenchmarkRouteTuning:
        if not self.config.enable_benchmark_route_tuning:
            return BenchmarkRouteTuning()
        return resolve_benchmark_route_tuning(
            query_route=query_route,
            profile=profile,
            evidence_top_k=evidence_top_k,
            route_context=route_context,
        )

    def _retrieval_cache_key(
        self,
        query: str,
        task_type: str,
        route: QueryRouteDecision,
        route_context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        normalized_query = normalize_text_for_hash(query).lower()
        fingerprint = stable_content_hash(
            json.dumps(
                {
                    "query": normalized_query,
                    "task_type": str(task_type or "qa"),
                    "route_type": route.route_type,
                    "benchmark_profile": str(getattr(route, "benchmark_profile", "") or ""),
                    "route_context": self._route_context_key(route_context),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return normalized_query, fingerprint

    def _completion_cache_key(self, query: str, task_type: str, evidence_signature: str) -> str:
        normalized_query = normalize_text_for_hash(query).lower()
        return stable_content_hash(
            json.dumps(
                {
                    "query": normalized_query,
                    "task_type": str(task_type or "qa"),
                    "evidence_signature": evidence_signature,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    def _object_shortcut_cache_key(
        self,
        *,
        query: str,
        profile: QueryProfile,
        route: QueryRouteDecision,
        limit: int,
        memory_version: int,
        route_context: dict[str, Any] | None = None,
    ) -> str:
        return stable_content_hash(
            json.dumps(
                {
                    "query": normalize_text_for_hash(query).lower(),
                    "task_type": profile.task_type,
                    "route_type": route.route_type,
                    "temporal_mode": profile.temporal_mode,
                    "preferred_types": list(profile.preferred_object_types),
                    "limit": int(limit),
                    "memory_version": int(memory_version),
                    "route_context": self._route_context_key(route_context),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    def _fetch_profile_snapshot_payload(self, snapshot_type: str, memory_version: int = 0) -> dict[str, Any] | None:
        fetcher = getattr(self.storage, "fetch_profile_snapshot", None)
        if not callable(fetcher):
            return None
        try:
            snapshot = fetcher(snapshot_type, memory_version=memory_version or None)
        except TypeError:
            snapshot = fetcher(snapshot_type)
        if not snapshot:
            return None
        payload = snapshot.get("payload")
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _flatten_snapshot_items(snapshot_type: str, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not payload:
            return []
        if snapshot_type == "profile":
            rows: list[dict[str, Any]] = []
            for key in ("preferences", "state", "persona", "relations"):
                items = payload.get(key)
                if isinstance(items, list):
                    rows.extend(dict(item) for item in items if isinstance(item, dict))
            for key in ("preference_history", "state_history", "persona_history", "relation_history"):
                history = payload.get(key)
                if isinstance(history, dict):
                    for values in history.values():
                        if isinstance(values, list):
                            rows.extend(dict(item) for item in values if isinstance(item, dict))
            recent_deltas = payload.get("recent_deltas")
            if isinstance(recent_deltas, list):
                rows.extend(dict(item) for item in recent_deltas if isinstance(item, dict))
            return rows
        items = payload.get("items")
        if isinstance(items, list):
            return [dict(item) for item in items if isinstance(item, dict)]
        history = payload.get("history")
        if isinstance(history, dict):
            rows: list[dict[str, Any]] = []
            for values in history.values():
                if isinstance(values, list):
                    rows.extend(dict(item) for item in values if isinstance(item, dict))
            return rows
        return []

    def _score_shortcut_object(self, query: str, profile: QueryProfile, obj: dict[str, Any], route: QueryRouteDecision) -> float:
        text = " ".join(
            filter(
                None,
                [
                    str(obj.get("object_text") or ""),
                    str(obj.get("entity") or ""),
                    str(obj.get("attribute") or ""),
                    str(obj.get("new_value") or ""),
                    str(obj.get("old_value") or ""),
                    str(obj.get("temporal_marker") or ""),
                ],
            )
        )
        score = lexical_score(query, text)
        object_type = str(obj.get("object_type") or "")
        preferred_types = set(profile.preferred_object_types)
        if route.route_type == "persona_preference_state":
            if object_type in {"preference", "persona", "personal_context", "constraint", "state_update", "relation"}:
                score += 0.12
        if route.route_type == "temporal":
            if object_type in {"state_update", "temporal_reference", "event"}:
                score += 0.16
        if route.route_type == "exact_factual" and object_type in {"artifact", "open_loop"} and object_type in preferred_types:
            score += 0.12
        if profile.needs_relation_objects and object_type == "relation":
            score += 0.14
        confidence = float(obj.get("confidence") or 0.0)
        score += min(0.12, max(0.0, confidence) * 0.12)
        if profile.needs_preference_objects and obj.get("polarity") is not None:
            score += 0.04
        if profile.temporal_mode == "latest" and str(obj.get("temporal_marker") or "").lower() in {"latest", "point"}:
            score += 0.08
        if profile.temporal_mode == "previous" and str(obj.get("temporal_marker") or "").lower() in {"previous", "point"}:
            score += 0.06
        if str(obj.get("snapshot_state") or "").lower() == "current":
            score += 0.04 if profile.temporal_mode != "previous" else -0.02
        if str(obj.get("snapshot_state") or "").lower() == "history":
            score += 0.03 if profile.temporal_mode == "previous" else 0.0
        return round(score, 4)

    def _score_shortcut_object_with_scope(
        self,
        query: str,
        profile: QueryProfile,
        obj: dict[str, Any],
        route: QueryRouteDecision,
        *,
        workspace_context: WorkspaceContext,
    ) -> dict[str, Any]:
        scored = dict(obj)
        base_score = self._score_shortcut_object(query, profile, scored, route)
        scope_bonus, scope_match = workspace_context.candidate_scope_bonus(
            scope=str(scored.get("scope") or ""),
            project=str(scored.get("project") or ""),
            session_id=str(scored.get("session_id") or ""),
            workspace=str(scored.get("workspace") or ""),
            weight=self.config.scope_priority_weight if self.config.enable_scope_priority else 0.0,
        )
        scored["scope_bonus"] = round(scope_bonus, 4)
        scored["scope_match"] = scope_match
        scored["object_score"] = round(base_score + scope_bonus, 4)
        return scored

    def _hydrate_shortcut_chunks(
        self,
        ranked_objects: list[dict[str, Any]],
        *,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        chunk_ids: list[str] = []
        pending_node_ids: list[str] = []
        for obj in ranked_objects:
            source_chunk_id = str(obj.get("source_chunk_id") or "")
            source_node_id = str(obj.get("source_node_id") or "")
            if source_chunk_id and source_chunk_id not in chunk_ids:
                chunk_ids.append(source_chunk_id)
            elif source_node_id and source_node_id not in pending_node_ids:
                pending_node_ids.append(source_node_id)
            if len(chunk_ids) >= limit:
                break
        chunks = self.storage.fetch_chunks_with_node_metadata_by_ids(chunk_ids) if chunk_ids else []
        if len(chunks) < limit:
            preferred_by_node = self.storage.fetch_preferred_chunks_for_nodes(pending_node_ids)
            for node_id in pending_node_ids:
                chosen = preferred_by_node.get(node_id)
                if not chosen:
                    continue
                chunk_id = str(chosen.get("chunk_id") or "")
                if chunk_id and chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)
                if len(chunk_ids) >= limit:
                    break
            chunks = self.storage.fetch_chunks_with_node_metadata_by_ids(chunk_ids) if chunk_ids else []
        packed = self._pack_chunks_with_metadata(chunks, already_hydrated=True)
        by_chunk_id = {str(item.get("chunk_id") or ""): item for item in packed}
        for obj in ranked_objects:
            chunk_id = str(obj.get("source_chunk_id") or "")
            if not chunk_id or chunk_id not in by_chunk_id:
                continue
            chunk = by_chunk_id[chunk_id]
            chunk["object_support"] = max(float(chunk.get("object_support") or 0.0), float(obj.get("object_score") or 0.0) * 0.12)
            chunk["shortcut_object_type"] = obj.get("object_type")
            chunk["shortcut_object_score"] = obj.get("object_score")
            chunk["shortcut_source"] = "profile_snapshot"
            chunk["query_lexical"] = max(float(chunk.get("query_lexical") or 0.0), lexical_score(str(obj.get("object_text") or ""), query))
        return packed[:limit]

    def _object_shortcut(
        self,
        query: str,
        profile: QueryProfile,
        route: QueryRouteDecision,
        *,
        limit: int,
        memory_version: int,
        route_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cache_key = self._object_shortcut_cache_key(
            query=query,
            profile=profile,
            route=route,
            limit=limit,
            memory_version=memory_version,
            route_context=route_context,
        )
        if self.config.enable_object_shortcut_cache and cache_key in self._object_shortcut_cache:
            return dict(self._object_shortcut_cache[cache_key])
        workspace_context = self._workspace_context(route_context)
        snapshot_order: list[str]
        preferred_types = set(profile.preferred_object_types)
        exact_object_shortcut = route.route_type == "exact_factual" and bool(preferred_types & {"artifact", "open_loop"})
        if route.route_type == "temporal":
            snapshot_order = ["state", "profile"]
        elif route.route_type == "persona_preference_state":
            snapshot_order = ["preference", "persona", "relation", "state", "profile"]
        elif exact_object_shortcut:
            snapshot_order = []
        else:
            return {
                "reason": "not_applicable",
                "shortcut_hit": False,
                "objects": [],
                "candidates": [],
                "snapshot_type": "",
            }
        ranked_objects: list[dict[str, Any]] = []
        snapshot_type_used = ""
        for snapshot_type in snapshot_order:
            payload = self._fetch_profile_snapshot_payload(snapshot_type, memory_version=memory_version)
            items = self._flatten_snapshot_items(snapshot_type, payload)
            if not items:
                continue
            scoped_items: list[dict[str, Any]] = []
            for item in items:
                candidate = {
                    "scope": item.get("scope"),
                    "workspace": item.get("workspace"),
                    "project": item.get("project"),
                    "session_id": item.get("session_id"),
                }
                if not self._candidate_in_scope_context(candidate, route_context):
                    continue
                scoped_items.append(
                    self._score_shortcut_object_with_scope(
                        query,
                        profile,
                        item,
                        route,
                        workspace_context=workspace_context,
                    )
                )
            scoped_items = [item for item in scoped_items if float(item.get("object_score") or 0.0) > 0.08]
            scoped_items.sort(key=lambda item: float(item.get("object_score") or 0.0), reverse=True)
            if scoped_items:
                ranked_objects = scoped_items
                snapshot_type_used = snapshot_type
                break
        if not ranked_objects and hasattr(self.storage, "fetch_objects"):
            fallback_types = tuple(dict.fromkeys(profile.preferred_object_types or ["preference", "persona", "state_update", "temporal_reference", "relation"]))
            if fallback_types:
                where = "object_type IN (" + ",".join(["?"] * len(fallback_types)) + ")"
                ranked_objects = []
                for item in self.storage.fetch_objects(where, fallback_types):
                    candidate = {
                        "scope": item.get("scope"),
                        "workspace": item.get("workspace"),
                        "project": item.get("project"),
                        "session_id": item.get("session_id"),
                    }
                    if not self._candidate_in_scope_context(candidate, route_context):
                        continue
                    ranked_objects.append(
                        self._score_shortcut_object_with_scope(
                            query,
                            profile,
                            dict(item),
                            route,
                            workspace_context=workspace_context,
                        )
                    )
                snapshot_type_used = "objects_fallback"
        ranked_objects = [item for item in ranked_objects if float(item.get("object_score") or 0.0) > 0.08]
        ranked_objects.sort(key=lambda item: float(item.get("object_score") or 0.0), reverse=True)
        if not ranked_objects:
            return {
                "reason": "no_matching_snapshot_objects",
                "shortcut_hit": False,
                "objects": [],
                "candidates": [],
                "snapshot_type": snapshot_type_used,
            }
        candidates = self._hydrate_shortcut_chunks(ranked_objects[: max(limit * 2, 4)], query=query, limit=limit)
        top_score = float(ranked_objects[0].get("object_score") or 0.0)
        second_score = float(ranked_objects[1].get("object_score") or 0.0) if len(ranked_objects) > 1 else 0.0
        explicit_object_query = bool(
            profile.needs_preference_objects
            or profile.needs_relation_objects
            or profile.needs_personal_context_objects
            or bool(preferred_types & {"artifact", "open_loop"})
        )
        temporal_snapshot_query = bool(profile.temporal_mode in {"latest", "previous", "range"})
        shortcut_can_be_primary = True
        focused_benchmark_followup = bool(
            route_context
            and str(route_context.get("benchmark") or "").lower() == "longmemeval"
            and str(route_context.get("focused_query") or "").strip()
        )
        if snapshot_type_used == "objects_fallback" and not (explicit_object_query or temporal_snapshot_query):
            shortcut_can_be_primary = False
        if focused_benchmark_followup and snapshot_type_used == "objects_fallback":
            shortcut_can_be_primary = False
        if profile.temporal_mode == "point":
            shortcut_can_be_primary = False
        sufficient = (
            bool(candidates)
            and shortcut_can_be_primary
            and (top_score >= 0.24 or (top_score >= 0.18 and (top_score - second_score) >= 0.05))
        )
        shortcut_reason = "shortcut_sufficient" if sufficient else "shortcut_partial"
        if focused_benchmark_followup and snapshot_type_used == "objects_fallback":
            shortcut_reason = "shortcut_deferred_focused_query"
        payload = {
            "reason": shortcut_reason,
            "shortcut_hit": sufficient,
            "objects": ranked_objects[: max(limit, 4)],
            "candidates": candidates,
            "snapshot_type": snapshot_type_used,
        }
        if self.config.enable_object_shortcut_cache:
            self._object_shortcut_cache[cache_key] = dict(payload)
        return payload

    def _proxy_hits_to_chunk_hits(self, proxy_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not proxy_hits:
            return []
        chunk_rows: dict[str, dict[str, Any]] = {}
        direct_chunk_ids = [str(hit.get("metadata", {}).get("parent_id") or "") for hit in proxy_hits if str(hit.get("metadata", {}).get("parent_type") or "chunk") == "chunk"]
        if direct_chunk_ids:
            for row in self.storage.fetch_chunks_with_node_metadata_by_ids(list(dict.fromkeys([item for item in direct_chunk_ids if item]))):
                chunk_rows[str(row.get("chunk_id") or "")] = row
        node_chunks: dict[str, dict[str, Any]] = {}
        node_parent_ids = [
            str((hit.get("metadata") or {}).get("parent_id") or "")
            for hit in proxy_hits
            if str((hit.get("metadata") or {}).get("parent_type") or "chunk") == "node"
        ]
        preferred_by_node = self.storage.fetch_preferred_chunks_for_nodes(node_parent_ids)
        for hit in proxy_hits:
            meta = hit.get("metadata") or {}
            if str(meta.get("parent_type") or "chunk") != "node":
                continue
            node_id = str(meta.get("parent_id") or "")
            if not node_id or node_id in node_chunks:
                continue
            hydrated = preferred_by_node.get(node_id)
            if hydrated:
                node_chunks[node_id] = hydrated
                chunk_rows[str(hydrated.get("chunk_id") or "")] = hydrated
        items: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for hit in proxy_hits:
            meta = hit.get("metadata") or {}
            parent_type = str(meta.get("parent_type") or "chunk")
            if parent_type == "chunk":
                chunk_id = str(meta.get("parent_id") or "")
                row = chunk_rows.get(chunk_id)
            else:
                row = node_chunks.get(str(meta.get("parent_id") or ""))
                chunk_id = str((row or {}).get("chunk_id") or "")
            if not row or not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            items.append(
                {
                    "chunk_id": chunk_id,
                    "document": row.get("text") or hit.get("document") or row.get("retrieval_summary") or row.get("structured_summary") or "",
                    "metadata": {
                        "node_id": row.get("node_id"),
                        "shell": row.get("shell"),
                        "sector": row.get("sector"),
                        "zone": row.get("zone"),
                        "cell": row.get("cell"),
                        "chunk_index": row.get("chunk_index"),
                        "grain": row.get("grain", "micro"),
                        "scope": row.get("scope") or meta.get("scope"),
                        "workspace": row.get("workspace") or meta.get("workspace"),
                        "project": row.get("project") or meta.get("project"),
                        "session_id": row.get("session_id") or meta.get("session_id"),
                        "summary": row.get("retrieval_summary") or row.get("summary") or row.get("structured_summary"),
                        "retrieval_summary": row.get("retrieval_summary"),
                        "structured_summary": row.get("structured_summary"),
                        "retrieval_signature": row.get("retrieval_signature"),
                        "time_bucket": row.get("time_bucket"),
                        "entity_tags": row.get("entity_tags"),
                        "task_type_tag": row.get("task_type_tag"),
                        "content_ref": row.get("content_ref"),
                        "source_type": row.get("source_type"),
                        "source_ref": row.get("source_ref"),
                        "access_count": row.get("access_count"),
                        "neighbor_count": row.get("neighbor_count"),
                        "created_at": row.get("created_at"),
                    },
                    "similarity": round(float(hit.get("similarity") or 0.0), 4),
                }
            )
        return items

    def _search_proxy_chunks(
        self,
        query: str,
        profile: QueryProfile,
        route: QueryRouteDecision,
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], str]:
        if not self.config.enable_retrieval_proxy_index or not hasattr(self.vector_store, "search_proxies"):
            return [], "proxy_disabled"
        proxy_kinds = ["summary", "structured", "signature"]
        where: dict[str, Any] | None = None
        reason = "proxy_general"
        if self.config.enable_temporal_prefilter and route.prefer_temporal_prefilter and profile.needs_temporal_objects:
            if profile.temporal_mode in {"latest", "previous"}:
                where = {"time_bucket": profile.temporal_mode}
                reason = f"temporal_bucket_{profile.temporal_mode}"
            else:
                where = {"task_type_tag": "temporal"}
                reason = "temporal_task_prefilter"
        hits = self.vector_store.search_proxies(query, top_k=max(1, top_k), where=where, proxy_kinds=proxy_kinds)
        if not hits and where is not None:
            hits = self.vector_store.search_proxies(query, top_k=max(1, top_k), proxy_kinds=proxy_kinds)
            reason = "proxy_relaxed_after_prefilter_miss"
        return self._proxy_hits_to_chunk_hits(hits), reason

    def _apply_temporal_candidate_prefilter(
        self,
        candidates: list[dict[str, Any]],
        profile: QueryProfile,
        *,
        keep_limit: int,
    ) -> tuple[list[dict[str, Any]], str]:
        if not profile.needs_temporal_objects or not candidates:
            return candidates, "skipped_non_temporal"
        explicit_refs = [term.lower() for term in profile.temporal_reference_terms if term]
        matched: list[dict[str, Any]] = []
        fallback: list[dict[str, Any]] = []
        for candidate in candidates:
            text = " ".join(
                filter(
                    None,
                    [
                        str(candidate.get("text") or ""),
                        str(candidate.get("summary") or ""),
                        str(candidate.get("retrieval_summary") or ""),
                        str(candidate.get("structured_summary") or ""),
                        str(candidate.get("time_bucket") or ""),
                    ],
                )
            ).lower()
            bucket = str(candidate.get("time_bucket") or "").lower()
            matches_explicit = any(term in text for term in explicit_refs)
            matches_mode = (
                (profile.temporal_mode == "latest" and bucket in {"latest", "point"})
                or (profile.temporal_mode == "previous" and bucket in {"previous", "point"})
                or (profile.temporal_mode in {"range", "point"} and (matches_explicit or bool(explicit_refs)))
            )
            if matches_explicit or matches_mode:
                matched.append(candidate)
            else:
                fallback.append(candidate)
        if not matched:
            return candidates, "prefilter_no_match_keep_all"
        filtered = matched + fallback[: max(0, keep_limit - len(matched))]
        return filtered[:keep_limit], f"prefilter_kept_{len(matched)}_temporal_matches"

    def _decide_rerank_mode(
        self,
        query: str,
        profile: QueryProfile,
        route: QueryRouteDecision,
        ranked_candidates: list[dict[str, Any]],
        *,
        cache_hit: bool,
        shortcut_hit: bool,
    ) -> tuple[str, str]:
        if not self.config.enable_conditional_rerank or not ranked_candidates:
            return "full", "default_full"
        top_score = float(ranked_candidates[0].get("evidence_score") or 0.0)
        second_score = float(ranked_candidates[1].get("evidence_score") or 0.0) if len(ranked_candidates) > 1 else 0.0
        gap = top_score - second_score
        top_lexical = float(ranked_candidates[0].get("query_lexical") or 0.0)
        top_anchor = float(ranked_candidates[0].get("direct_anchor_score") or 0.0)
        temporal_conf = 0.0 if not profile.needs_temporal_objects else min(1.0, len(profile.temporal_reference_terms) * 0.4 + (0.25 if profile.temporal_mode != "none" else 0.0))
        if route.route_type in {"debug_design", "open_creative_transfer"} or profile.needs_multi_hop_evidence:
            return "full", "full_complex_route"
        if profile.needs_temporal_objects and temporal_conf < 0.25:
            return "full", "full_low_temporal_confidence"
        if cache_hit or shortcut_hit:
            return "light", "light_cache_or_shortcut"
        if route.prefer_light_rerank and top_anchor >= 0.36 and top_lexical >= 0.16 and gap >= 0.1:
            return "skip", "skip_clear_exact_lead"
        if route.route_type == "temporal":
            return "light", "light_temporal_route"
        return "light" if top_score >= 0.72 and gap >= 0.08 else "full", "light_confident_pool" if top_score >= 0.72 and gap >= 0.08 else "full_uncertain_pool"

    def _compress_candidate_pool(
        self,
        candidates: list[dict[str, Any]],
        *,
        target_limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not candidates:
            return [], {
                "input_count": 0,
                "dedup_count": 0,
                "cluster_seed_count": 0,
                "dedup_hit_rate": 0.0,
            }
        deduped: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        semantic_dedup_count = 0
        for candidate in candidates:
            novelty_key = "|".join(
                [
                    str(candidate.get("node_id") or ""),
                    str(candidate.get("time_bucket") or ""),
                    str(candidate.get("task_type_tag") or ""),
                    normalize_text_for_hash((candidate.get("retrieval_signature") or candidate.get("summary") or candidate.get("text") or "")[:220]).lower(),
                ]
            )
            duplicate = False
            if self.config.enable_semantic_dedup and novelty_key in seen_keys:
                duplicate = True
            elif self.config.enable_semantic_dedup:
                candidate_text = self._candidate_novelty_text(candidate)
                for chosen in deduped[-6:]:
                    if lexical_score(candidate_text[:500], self._candidate_novelty_text(chosen)[:500]) >= 0.86:
                        duplicate = True
                        break
            if duplicate:
                semantic_dedup_count += 1
                continue
            seen_keys.add(novelty_key)
            deduped.append(candidate)
        if not self.config.enable_seed_clustering:
            return deduped[:target_limit], {
                "input_count": len(candidates),
                "dedup_count": semantic_dedup_count,
                "cluster_seed_count": min(len(deduped), target_limit),
                "dedup_hit_rate": round(semantic_dedup_count / max(1, len(candidates)), 4),
            }
        clustered: dict[str, dict[str, Any]] = {}
        for candidate in deduped:
            cluster_key = "|".join(
                [
                    str(candidate.get("node_id") or candidate.get("content_ref") or candidate.get("chunk_id") or ""),
                    str(candidate.get("time_bucket") or ""),
                    str(candidate.get("task_type_tag") or ""),
                ]
            )
            current = clustered.get(cluster_key)
            if current is None or float(candidate.get("evidence_score") or 0.0) > float(current.get("evidence_score") or 0.0):
                updated = dict(candidate)
                updated["cluster_member_chunk_ids"] = list(current.get("cluster_member_chunk_ids") or []) if current else []
                clustered[cluster_key] = updated
                current = updated
            member_chunk_ids = list(current.get("cluster_member_chunk_ids") or [])
            chunk_id = str(candidate.get("chunk_id") or "")
            if chunk_id and chunk_id not in member_chunk_ids:
                member_chunk_ids.append(chunk_id)
            current["cluster_member_chunk_ids"] = member_chunk_ids
            current["cluster_size"] = len(member_chunk_ids)
        seeds = list(clustered.values())
        seeds.sort(key=lambda item: float(item.get("evidence_score") or 0.0), reverse=True)
        return seeds[:target_limit], {
            "input_count": len(candidates),
            "dedup_count": semantic_dedup_count,
            "cluster_seed_count": len(seeds[:target_limit]),
            "dedup_hit_rate": round(semantic_dedup_count / max(1, len(candidates)), 4),
        }

    def retrieve(
        self,
        query: str,
        task_type: str = "qa",
        evidence_top_k: int = 8,
        support_top_k: int = 4,
        cognitive_top_k: int = 4,
    ) -> dict[str, Any]:
        evidence = self.retrieve_evidence(query, task_type=task_type, evidence_top_k=evidence_top_k)
        completion = self.complete_with_objects(
            query=query,
            evidence=evidence,
            support_top_k=support_top_k,
            object_top_k=max(4, support_top_k),
        )
        cognitive = self.augment_cognitively(
            query=query,
            task_type=task_type,
            completion=completion,
            cognitive_top_k=cognitive_top_k,
        )
        return {
            "profile": evidence.profile,
            "core_evidence": completion.core_evidence,
            "evidence_objects": completion.evidence_objects,
            "supporting_context": completion.supporting_context,
            "relevant_experience": cognitive.relevant_experience,
            "creative_reflections": cognitive.creative_reflections,
            "alternative_paths": cognitive.alternative_paths,
            "evidence_nodes": completion.evidence_nodes,
        }

    def retrieve_evidence(
        self,
        query: str,
        task_type: str = "qa",
        evidence_top_k: int = 8,
        route_context: dict[str, Any] | None = None,
    ) -> EvidenceRetrievalResult:
        total_start = perf_counter()
        profile = self.parser.parse(query, task_type=task_type)
        route = self._resolve_task_route(profile.task_type)
        query_route = self._decide_query_route(query, task_type, profile)
        route_tuning = self._resolve_route_tuning(
            query_route=query_route,
            profile=profile,
            evidence_top_k=evidence_top_k,
            route_context=route_context,
        )
        if route_tuning.benchmark:
            query_route.benchmark_profile = route_tuning.route_profile
        if route_tuning.prefer_object_shortcut is not None:
            query_route.prefer_object_shortcut = bool(route_tuning.prefer_object_shortcut)
        query_route.prefer_identity_rerank = bool(query_route.prefer_identity_rerank or route_tuning.prefer_identity_rerank)
        query_route.prefer_segment_rerank = bool(query_route.prefer_segment_rerank or route_tuning.prefer_segment_rerank)
        query_route.prefer_confusing_cluster = bool(query_route.prefer_confusing_cluster or route_tuning.prefer_confusing_cluster)
        if route_tuning.coarse_topk:
            query_route.suggested_config["coarse_topk"] = int(route_tuning.coarse_topk)
        if route_tuning.fine_topk:
            query_route.suggested_config["fine_topk"] = int(route_tuning.fine_topk)
        timings_ms: dict[str, float] = {}
        runtime_latency_budget_ms = max(0, int(getattr(self.config, "runtime_retrieval_latency_budget_ms", 0) or 0))
        channel_parallel_enabled = bool(getattr(self.config, "runtime_parallel_channels_enabled", False))
        diagnostics: dict[str, Any] = {
            "candidate_counts": {},
            "decisions": {},
            "cache": {},
            "feature_cache": {"hits": 0, "misses": 0},
        }
        self._active_candidate_feature_cache = {}
        self._active_feature_cache_stats = {"hits": 0, "misses": 0}
        runtime_before = self._snapshot_runtime_stats()
        memory_version = self._get_memory_version()
        coarse_topk = max(8, int(query_route.suggested_config.get("coarse_topk", self.config.retrieval_topk_coarse) or self.config.retrieval_topk_coarse))
        fine_topk = max(6, int(query_route.suggested_config.get("fine_topk", self.config.retrieval_topk_fine) or self.config.retrieval_topk_fine))
        dense_probe_k = int(route_tuning.dense_probe_k or max(coarse_topk, evidence_top_k + max(8, fine_topk)))
        proxy_probe_k = int(route_tuning.proxy_probe_k or max(coarse_topk, evidence_top_k + max(6, fine_topk // 2)))
        sparse_probe_k = int(route_tuning.sparse_probe_k or max(coarse_topk, evidence_top_k + max(8, fine_topk)))
        diagnostics["route"] = self._query_route_payload(query_route)
        diagnostics["route_context"] = dict(route_context or {})
        diagnostics["route_tuning"] = route_tuning.to_dict()
        diagnostics["decisions"]["route_type"] = query_route.route_type
        diagnostics.setdefault("decision_scores", {})["route_confidence"] = query_route.confidence
        diagnostics["candidate_counts"]["coarse_topk"] = coarse_topk
        diagnostics["candidate_counts"]["fine_topk"] = fine_topk
        diagnostics["candidate_counts"]["dense_probe_k"] = dense_probe_k
        diagnostics["candidate_counts"]["proxy_probe_k"] = proxy_probe_k
        diagnostics["candidate_counts"]["sparse_probe_k"] = sparse_probe_k
        diagnostics["runtime_latency_budget_ms"] = runtime_latency_budget_ms
        diagnostics["runtime_latency_budget_exceeded"] = False
        diagnostics["channel_parallel_enabled"] = channel_parallel_enabled

        normalized_query, query_fingerprint = self._retrieval_cache_key(
            query,
            task_type,
            query_route,
            route_context=route_context,
        )
        if (
            self.config.enable_retrieval_cache
            and memory_version > 0
            and hasattr(self.storage, "get_retrieval_cache")
        ):
            stage_start = perf_counter()
            cache_entry = self.storage.get_retrieval_cache(query_fingerprint, memory_version)
            timings_ms["retrieval_cache_lookup_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
            if cache_entry and self._cache_is_fresh(cache_entry.get("created_at"), self.config.retrieval_cache_ttl_seconds):
                payload = cache_entry.get("payload") or {}
                cached_candidates = [dict(item) for item in payload.get("candidates", [])]
                cached_node_ids = list(payload.get("evidence_node_ids") or [])
                evidence_nodes = self.storage.fetch_nodes_by_ids(cached_node_ids) if cached_node_ids else []
                timings_ms.update({str(k): float(v) for k, v in dict(payload.get("timings_ms") or {}).items()})
                timings_ms["retrieval_cache_hit_ms"] = round((perf_counter() - total_start) * 1000.0, 2)
                timings_ms["total_ms"] = round((perf_counter() - total_start) * 1000.0, 2)
                cached_diagnostics = dict(payload.get("diagnostics") or {})
                cached_diagnostics.setdefault("cache", {})
                cached_diagnostics["cache"]["retrieval_hit"] = 1.0
                cached_diagnostics["cache"]["retrieval_miss"] = 0.0
                cached_diagnostics["backend"] = self._stats_delta(self._snapshot_runtime_stats(), runtime_before)
                return EvidenceRetrievalResult(
                    profile=profile,
                    candidates=cached_candidates,
                    evidence_nodes=evidence_nodes,
                    dense_object_hits=[dict(item) for item in payload.get("dense_object_hits", [])],
                    sparse_object_hits=[dict(item) for item in payload.get("sparse_object_hits", [])],
                    query_route=dict(payload.get("query_route") or self._query_route_payload(query_route)),
                    timings_ms=timings_ms,
                    diagnostics=cached_diagnostics,
                )
            diagnostics["cache"]["retrieval_hit"] = 0.0
            diagnostics["cache"]["retrieval_miss"] = 1.0
        else:
            timings_ms["retrieval_cache_lookup_ms"] = 0.0
            diagnostics["cache"]["retrieval_hit"] = 0.0
            diagnostics["cache"]["retrieval_miss"] = 0.0

        dense_chunks: list[dict[str, Any]] = []
        sparse_chunks: list[dict[str, Any]] = []
        proxy_chunks: list[dict[str, Any]] = []
        dense_objects: list[dict[str, Any]] = []
        sparse_objects: list[dict[str, Any]] = []
        chunk_candidates: list[dict[str, Any]] = []
        ranked_candidates: list[dict[str, Any]] = []
        initial_ranked: list[dict[str, Any]] = []
        initial_guard_reason = "not_run"
        exact_guard_reason = "not_run"
        shortcut_hit = False
        object_ms = 0.0
        object_merge_ms = 0.0
        candidate_merge_ms = 0.0
        rank_ms = 0.0

        stage_start = perf_counter()
        shortcut_payload = (
            self._object_shortcut(
                query=query,
                profile=profile,
                route=query_route,
                limit=max(fine_topk * 2, evidence_top_k),
                memory_version=memory_version,
                route_context=route_context,
            )
            if self.config.enable_object_shortcut and query_route.prefer_object_shortcut
            else {"reason": "shortcut_disabled", "shortcut_hit": False, "objects": [], "candidates": [], "snapshot_type": ""}
        )
        timings_ms["object_shortcut_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
        shortcut_hit = bool(shortcut_payload.get("shortcut_hit"))
        diagnostics["decisions"]["object_shortcut"] = str(shortcut_payload.get("reason") or "")
        diagnostics["candidate_counts"]["shortcut_objects"] = len(shortcut_payload.get("objects") or [])
        diagnostics["candidate_counts"]["shortcut_candidates"] = len(shortcut_payload.get("candidates") or [])
        diagnostics["decisions"]["shortcut_snapshot_type"] = str(shortcut_payload.get("snapshot_type") or "")

        if shortcut_hit:
            chunk_candidates = [dict(item) for item in shortcut_payload.get("candidates") or []]
            dense_objects = [
                {
                    "object_id": str(item.get("object_id") or ""),
                    "document": str(item.get("object_text") or ""),
                    "metadata": {
                        "object_type": item.get("object_type"),
                        "source_chunk_id": item.get("source_chunk_id"),
                        "source_node_id": item.get("source_node_id"),
                        "entity": item.get("entity"),
                        "attribute": item.get("attribute"),
                        "session_id": item.get("session_id"),
                        "canonical_key": item.get("canonical_key"),
                        "temporal_marker": item.get("temporal_marker"),
                    },
                    "similarity": min(0.99, float(item.get("object_score") or 0.0)),
                }
                for item in shortcut_payload.get("objects") or []
                if item.get("object_id")
            ]
            chunk_candidates = [item for item in chunk_candidates if self._candidate_in_scope_context(item, route_context)]
            stage_start = perf_counter()
            ranked_candidates = self._rank_chunk_candidates(
                query=query,
                candidates=chunk_candidates,
                profile=profile,
                route=route,
                route_context=route_context,
                route_tuning=route_tuning,
            )
            ranked_candidates, exact_guard_reason = self._apply_exact_evidence_guard(ranked_candidates, profile)
            rank_ms = (perf_counter() - stage_start) * 1000.0
            diagnostics["rank_pass_count"] = 1
            initial_ranked = ranked_candidates
            initial_guard_reason = exact_guard_reason
            diagnostics["decisions"]["object_support"] = "skipped_shortcut_sufficient"
            diagnostics.setdefault("decision_scores", {})["object_support_enabled"] = 0.0
            timings_ms["dense_vector_ms"] = 0.0
            timings_ms["sparse_fts_ms"] = 0.0
            timings_ms["proxy_vector_ms"] = 0.0
            timings_ms["dense_object_ms"] = 0.0
            timings_ms["sparse_object_fts_ms"] = 0.0
        else:
            query_variants = self._route_query_variants(query, route_context)
            before_dedup_count = len(query_variants)
            deduped_variants: list[str] = []
            seen_variant_keys: set[str] = set()
            for variant in query_variants:
                variant_key = normalize_text_for_hash(variant).lower()
                if not variant_key or variant_key in seen_variant_keys:
                    continue
                seen_variant_keys.add(variant_key)
                deduped_variants.append(variant)
            simple_factual_route = (
                not profile.needs_multi_hop_evidence
                and not profile.needs_temporal_objects
                and not profile.needs_preference_objects
                and float(query_route.confidence or 0.0) >= 0.62
            )
            if simple_factual_route:
                deduped_variants = deduped_variants[:2]
            elif profile.needs_temporal_objects or profile.needs_preference_objects:
                deduped_variants = deduped_variants[:3]
            else:
                deduped_variants = deduped_variants[:6]
            query_variants = deduped_variants or [query]
            diagnostics["query_variant_budget"] = len(query_variants)
            diagnostics["query_variant_count_before_dedup"] = before_dedup_count
            diagnostics["query_variant_count_after_dedup"] = len(query_variants)
            diagnostics["queries"] = [
                {
                    "query": variant,
                    "variant_index": index,
                    "source": self._route_variant_label(variant, query, route_context),
                }
                for index, variant in enumerate(query_variants)
            ]
            diagnostics["candidate_counts"]["query_variants"] = len(query_variants)

            dense_vector_ms = 0.0
            proxy_vector_ms = 0.0
            sparse_fts_ms = 0.0
            proxy_reasons: list[str] = []
            dense_chunks = []
            proxy_chunks = []
            sparse_chunks = []

            def _dense_variant(query_variant: str, route_variant: str, variant_dense_k: int) -> tuple[list[dict[str, Any]], float]:
                stage = perf_counter()
                rows: list[dict[str, Any]] = []
                if hasattr(self.vector_store, "search"):
                    rows = self.vector_store.search(query_variant, top_k=variant_dense_k)
                    rows = self._filter_hits_by_scope_context(rows, route_context)
                    rows = self._tag_route_variant_hits(rows, route_variant)
                return rows, (perf_counter() - stage) * 1000.0

            def _proxy_variant(query_variant: str, route_variant: str, variant_proxy_k: int) -> tuple[list[dict[str, Any]], float, str]:
                stage = perf_counter()
                rows, reason = self._search_proxy_chunks(query_variant, profile, query_route, top_k=variant_proxy_k)
                rows = self._filter_hits_by_scope_context(rows, route_context)
                rows = self._tag_route_variant_hits(rows, route_variant)
                return rows, (perf_counter() - stage) * 1000.0, reason

            def _sparse_variant(query_variant: str, route_variant: str, variant_sparse_k: int) -> tuple[list[dict[str, Any]], float]:
                stage = perf_counter()
                rows = self._sparse_chunk_hits(query_variant, limit=variant_sparse_k)
                rows = self._filter_hits_by_scope_context(rows, route_context)
                rows = self._tag_route_variant_hits(rows, route_variant)
                return rows, (perf_counter() - stage) * 1000.0

            for variant_index, query_variant in enumerate(query_variants):
                if runtime_latency_budget_ms > 0 and (perf_counter() - total_start) * 1000.0 > runtime_latency_budget_ms:
                    diagnostics["runtime_latency_budget_exceeded"] = True
                    break
                route_variant = self._route_variant_label(query_variant, query, route_context)
                variant_scale = 1.0 if variant_index == 0 else (0.55 if simple_factual_route else 0.75)
                variant_dense_k = max(12, int(dense_probe_k * variant_scale))
                variant_proxy_k = max(12, int(proxy_probe_k * variant_scale))
                variant_sparse_k = max(12, int(sparse_probe_k * variant_scale))
                if channel_parallel_enabled:
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        dense_future = executor.submit(_dense_variant, query_variant, route_variant, variant_dense_k)
                        proxy_future = executor.submit(_proxy_variant, query_variant, route_variant, variant_proxy_k)
                        sparse_future = executor.submit(_sparse_variant, query_variant, route_variant, variant_sparse_k)
                        variant_dense_chunks, dense_elapsed = dense_future.result()
                        variant_proxy_chunks, proxy_elapsed, variant_proxy_reason = proxy_future.result()
                        variant_sparse_chunks, sparse_elapsed = sparse_future.result()
                    dense_vector_ms += dense_elapsed
                    proxy_vector_ms += proxy_elapsed
                    sparse_fts_ms += sparse_elapsed
                    dense_chunks.extend(variant_dense_chunks)
                    proxy_chunks.extend(variant_proxy_chunks)
                    sparse_chunks.extend(variant_sparse_chunks)
                    proxy_reasons.append(f"{query_variant}: {variant_proxy_reason}")
                    continue
                variant_dense_chunks, dense_elapsed = _dense_variant(query_variant, route_variant, variant_dense_k)
                dense_vector_ms += dense_elapsed
                dense_chunks.extend(variant_dense_chunks)

                variant_proxy_chunks, proxy_elapsed, variant_proxy_reason = _proxy_variant(query_variant, route_variant, variant_proxy_k)
                proxy_vector_ms += proxy_elapsed
                proxy_chunks.extend(variant_proxy_chunks)
                proxy_reasons.append(f"{query_variant}: {variant_proxy_reason}")

                variant_sparse_chunks, sparse_elapsed = _sparse_variant(query_variant, route_variant, variant_sparse_k)
                sparse_fts_ms += sparse_elapsed
                sparse_chunks.extend(variant_sparse_chunks)

            timings_ms["dense_vector_ms"] = round(dense_vector_ms, 2)
            diagnostics["dense_runtime_ms"] = round(dense_vector_ms, 2)
            diagnostics["candidate_counts"]["dense_chunks"] = len(dense_chunks)
            timings_ms["proxy_vector_ms"] = round(proxy_vector_ms, 2)
            diagnostics["proxy_runtime_ms"] = round(proxy_vector_ms, 2)
            diagnostics["candidate_counts"]["proxy_chunks"] = len(proxy_chunks)
            diagnostics["decisions"]["proxy_retrieval"] = " | ".join(proxy_reasons)
            timings_ms["sparse_fts_ms"] = round(sparse_fts_ms, 2)
            diagnostics["sparse_runtime_ms"] = round(sparse_fts_ms, 2)
            diagnostics["candidate_counts"]["sparse_chunks"] = len(sparse_chunks)

            stage_start = perf_counter()
            initial_candidates = self._merge_chunk_candidates(
                query=query,
                dense_chunks=dense_chunks,
                sparse_chunks=sparse_chunks,
                proxy_chunks=proxy_chunks,
                dense_objects=[],
                sparse_objects=[],
                diagnostics=diagnostics,
            )
            candidate_merge_ms = (perf_counter() - stage_start) * 1000.0
            diagnostics["candidate_counts"]["merged_chunks_before_objects"] = len(initial_candidates)
            diagnostics["candidate_counts"]["grain_before_objects"] = self._grain_distribution(initial_candidates)

            if self.config.enable_temporal_prefilter and query_route.prefer_temporal_prefilter:
                stage_start = perf_counter()
                initial_candidates, temporal_reason = self._apply_temporal_candidate_prefilter(
                    initial_candidates,
                    profile,
                    keep_limit=max(max(dense_probe_k, sparse_probe_k), 16),
                )
                timings_ms["temporal_prefilter_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
                diagnostics["decisions"]["temporal_prefilter"] = temporal_reason
                diagnostics["candidate_counts"]["temporal_prefilter_candidates"] = len(initial_candidates)
            else:
                timings_ms["temporal_prefilter_ms"] = 0.0
                diagnostics["decisions"]["temporal_prefilter"] = "disabled_or_not_needed"

            stage_start = perf_counter()
            initial_ranked = self._rank_chunk_candidates(
                query=query,
                candidates=initial_candidates,
                profile=profile,
                route=route,
                route_context=route_context,
                route_tuning=route_tuning,
            )
            initial_ranked, initial_guard_reason = self._apply_exact_evidence_guard(initial_ranked, profile)
            rank_ms = (perf_counter() - stage_start) * 1000.0
            use_object_support, object_reason = self._should_use_object_support(profile, initial_ranked, evidence_top_k)
            initial_margin = self._ranking_diagnostics(initial_ranked).get("top_margin_1_2", 0.0)
            initial_top_score = float(initial_ranked[0].get("evidence_score") or 0.0) if initial_ranked else 0.0
            if use_object_support and initial_top_score >= 0.78 and float(initial_margin or 0.0) >= 0.12 and initial_guard_reason != "failed_exact_guard":
                use_object_support = False
                object_reason = "skipped_high_confidence_dense_sparse_evidence"
            diagnostics["decisions"]["object_support"] = object_reason
            diagnostics["decisions"]["exact_evidence_guard_pre_object"] = initial_guard_reason
            diagnostics.setdefault("decision_scores", {})["object_support_enabled"] = 1.0 if use_object_support else 0.0
            diagnostics["object_support_triggered"] = bool(use_object_support)
            diagnostics["object_support_reason"] = object_reason

            chunk_candidates = initial_candidates
            ranked_candidates = initial_ranked
            rank_pass_count = 1
            if use_object_support:
                stage_start = perf_counter()
                dense_objects = self.vector_store.search_objects(query, top_k=max(fine_topk * 3, 12)) if hasattr(self.vector_store, "search_objects") else []
                dense_objects = self._filter_hits_by_scope_context(dense_objects, route_context)
                dense_object_ms = (perf_counter() - stage_start) * 1000.0
                stage_start = perf_counter()
                sparse_objects = self._sparse_object_hits(query, profile, limit=max(fine_topk * 3, 12))
                sparse_objects = self._filter_hits_by_scope_context(sparse_objects, route_context)
                sparse_object_ms = (perf_counter() - stage_start) * 1000.0
                stage_start = perf_counter()
                chunk_candidates = self._merge_chunk_candidates(
                    query=query,
                    dense_chunks=dense_chunks,
                    sparse_chunks=sparse_chunks,
                    proxy_chunks=proxy_chunks,
                    dense_objects=dense_objects,
                    sparse_objects=sparse_objects,
                    diagnostics=diagnostics,
                )
                object_merge_ms = (perf_counter() - stage_start) * 1000.0
                stage_start = perf_counter()
                ranked_candidates = self._rank_chunk_candidates(
                    query=query,
                    candidates=chunk_candidates,
                    profile=profile,
                    route=route,
                    route_context=route_context,
                    route_tuning=route_tuning,
                )
                ranked_candidates, exact_guard_reason = self._apply_exact_evidence_guard(ranked_candidates, profile)
                rank_ms += (perf_counter() - stage_start) * 1000.0
                rank_pass_count += 1
                object_ms = dense_object_ms + sparse_object_ms
                timings_ms["dense_object_ms"] = round(dense_object_ms, 2)
                timings_ms["sparse_object_fts_ms"] = round(sparse_object_ms, 2)
            else:
                exact_guard_reason = initial_guard_reason
                timings_ms["dense_object_ms"] = 0.0
                timings_ms["sparse_object_fts_ms"] = 0.0
            diagnostics["rank_pass_count"] = rank_pass_count
            diagnostics["object_runtime_ms"] = round(object_ms, 2)
            diagnostics["full_rank_avoided"] = bool(rank_pass_count == 1)
            diagnostics["incremental_rank_used"] = bool(use_object_support and rank_pass_count == 1)

        diagnostics["candidate_counts"]["dense_objects"] = len(dense_objects)
        diagnostics["candidate_counts"]["sparse_objects"] = len(sparse_objects)
        diagnostics["candidate_counts"]["merged_chunks"] = len(chunk_candidates)
        diagnostics.setdefault("query_variant_budget", 0 if shortcut_hit else diagnostics["candidate_counts"].get("query_variants", 0))
        diagnostics.setdefault("dense_runtime_ms", float(timings_ms.get("dense_vector_ms", 0.0)))
        diagnostics.setdefault("proxy_runtime_ms", float(timings_ms.get("proxy_vector_ms", 0.0)))
        diagnostics.setdefault("sparse_runtime_ms", float(timings_ms.get("sparse_fts_ms", 0.0)))
        diagnostics.setdefault("object_runtime_ms", round(object_ms, 2))
        diagnostics.setdefault("full_rank_avoided", bool(int(diagnostics.get("rank_pass_count") or 0) <= 1))
        diagnostics.setdefault("incremental_rank_used", False)
        feature_stats = dict(getattr(self, "_active_feature_cache_stats", {}) or {})
        diagnostics["feature_cache_hits"] = int(feature_stats.get("hits") or 0)
        diagnostics["feature_cache_misses"] = int(feature_stats.get("misses") or 0)
        diagnostics["feature_cache"] = {
            "hits": int(feature_stats.get("hits") or 0),
            "misses": int(feature_stats.get("misses") or 0),
        }
        diagnostics["candidate_counts"]["grain_after_objects"] = self._grain_distribution(chunk_candidates)
        diagnostics["decisions"]["exact_evidence_guard"] = exact_guard_reason
        ranking_summary = self._ranking_diagnostics(ranked_candidates)
        ranking_summary["initial_top_candidates"] = self._candidate_diagnostic_rows(initial_ranked, limit=min(8, max(evidence_top_k, 8)))
        ranking_summary["top_candidates"] = self._candidate_diagnostic_rows(ranked_candidates, limit=min(8, max(evidence_top_k, 8)))
        diagnostics["ranking"] = ranking_summary
        timings_ms["candidate_merge_ms"] = round(candidate_merge_ms + object_merge_ms, 2)
        timings_ms["object_retrieval_ms"] = round(object_ms, 2)
        timings_ms["rank_evidence_ms"] = round(rank_ms, 2)

        rerank_mode, rerank_reason = self._decide_rerank_mode(
            query=query,
            profile=profile,
            route=query_route,
            ranked_candidates=ranked_candidates,
            cache_hit=False,
            shortcut_hit=shortcut_hit,
        )
        diagnostics["decisions"]["conditional_rerank"] = rerank_reason
        diagnostics["decisions"]["rerank_mode"] = rerank_mode
        target_rerank_pool = int(route_tuning.rerank_pool_k or 0)
        if rerank_mode == "skip":
            rerank_pool_size = max(target_rerank_pool or 0, evidence_top_k + 2, fine_topk)
        elif rerank_mode == "light":
            rerank_pool_size = max(target_rerank_pool or 0, evidence_top_k + max(8, fine_topk), 12)
        else:
            base_rerank_pool = max(evidence_top_k * 3, 24)
            rerank_pool_size = max(target_rerank_pool or 0, evidence_top_k * 4, 32) if self.cross_encoder else max(target_rerank_pool or 0, base_rerank_pool)
        diagnostics["candidate_counts"]["rerank_pool_requested"] = rerank_pool_size
        stage_start = perf_counter()
        ranked_candidates, identity_stats = self._apply_identity_awareness(
            query=query,
            profile=profile,
            candidates=ranked_candidates,
            route_context=route_context,
            route_tuning=route_tuning,
        )
        timings_ms["identity_rerank_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
        diagnostics["identity"] = identity_stats
        stage_start = perf_counter()
        ranked_candidates, cluster_stats = self._apply_confusing_cluster_rerank(
            query=query,
            candidates=ranked_candidates,
            route_context=route_context,
            route_tuning=route_tuning,
        )
        timings_ms["confusing_cluster_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
        diagnostics["clusters"] = cluster_stats
        stage_start = perf_counter()
        ranked_candidates, segment_stats = self._apply_segment_rerank(
            query=query,
            profile=profile,
            candidates=ranked_candidates,
            route_context=route_context,
            route_tuning=route_tuning,
        )
        timings_ms["segment_rerank_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
        diagnostics["segments"] = segment_stats
        ranked_evidence, selection_stats = self._select_core_evidence(
            ranked_candidates,
            limit=rerank_pool_size,
            profile=profile,
            route_context=route_context,
        )
        diagnostics["candidate_counts"]["rerank_pool"] = len(ranked_evidence)
        selection_stats["selected_top_candidates"] = self._candidate_diagnostic_rows(ranked_evidence, limit=min(8, evidence_top_k))
        diagnostics["selection"] = selection_stats

        if self.cross_encoder is not None and ranked_evidence and rerank_mode != "skip":
            use_cross_encoder, cross_reason = self._should_use_cross_encoder(profile, ranked_evidence)
            if rerank_mode == "light" and not (profile.needs_temporal_objects or profile.needs_multi_hop_evidence):
                use_cross_encoder = False
                cross_reason = "skipped_light_rerank_mode"
            diagnostics["decisions"]["cross_encoder"] = cross_reason
            diagnostics["decision_scores"]["cross_encoder_enabled"] = 1.0 if use_cross_encoder else 0.0
            if use_cross_encoder:
                stage_start = perf_counter()
                pairs = [(query, (c.get("text") or "")[:4000]) for c in ranked_evidence]
                ce_scores = [float(score) for score in self.cross_encoder.predict(pairs)]
                for c, ce_s in zip(ranked_evidence, ce_scores):
                    c["cross_encoder_score"] = round(ce_s, 4)
                    c["evidence_score"] = round(c.get("evidence_score", 0.0) * 0.55 + ce_s * 0.45, 4)
                ranked_evidence.sort(key=lambda x: x["evidence_score"], reverse=True)
                timings_ms["cross_encoder_rerank_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
            else:
                timings_ms["cross_encoder_rerank_ms"] = 0.0
        else:
            diagnostics["decision_scores"]["cross_encoder_enabled"] = 0.0
            timings_ms["cross_encoder_rerank_ms"] = 0.0

        compression_target = max(evidence_top_k, fine_topk)
        compressed_evidence, compression_stats = self._compress_candidate_pool(ranked_evidence, target_limit=compression_target)
        diagnostics["compression"] = compression_stats
        diagnostics["candidate_counts"]["seed_input"] = len(ranked_evidence)
        diagnostics["candidate_counts"]["seed_output"] = len(compressed_evidence)
        ranked_evidence = compressed_evidence[:evidence_top_k]
        diagnostics["selection"]["final_top_candidates"] = self._candidate_diagnostic_rows(ranked_evidence, limit=min(8, evidence_top_k))
        diagnostics["safety"] = {
            "factual_contamination_rate": self._factual_contamination_rate(ranked_evidence),
        }
        evidence_node_ids: list[str] = []
        for item in ranked_evidence:
            node_id = str(item.get("node_id") or "")
            if node_id and node_id not in evidence_node_ids:
                evidence_node_ids.append(node_id)
        diagnostics["candidate_counts"]["final_evidence"] = len(ranked_evidence)
        stage_start = perf_counter()
        evidence_nodes = self.storage.fetch_nodes_by_ids(evidence_node_ids)
        timings_ms["fetch_evidence_nodes_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
        timings_ms["total_ms"] = round((perf_counter() - total_start) * 1000.0, 2)
        diagnostics["candidate_counts"]["evidence_nodes"] = len(evidence_nodes)
        diagnostics["backend"] = self._stats_delta(self._snapshot_runtime_stats(), runtime_before)
        self._persist_recall_trace(
            query=query,
            route_context=route_context,
            diagnostics=diagnostics,
            timings_ms=timings_ms,
            candidates=ranked_evidence,
        )

        if (
            self.config.enable_retrieval_cache
            and memory_version > 0
            and hasattr(self.storage, "put_retrieval_cache")
        ):
            self.storage.put_retrieval_cache(
                query_fingerprint=query_fingerprint,
                normalized_query=normalized_query,
                task_type=task_type,
                route_type=query_route.route_type,
                memory_version=memory_version,
                payload={
                    "candidates": ranked_evidence,
                    "evidence_node_ids": evidence_node_ids,
                    "dense_object_hits": dense_objects,
                    "sparse_object_hits": sparse_objects,
                    "query_route": self._query_route_payload(query_route),
                    "timings_ms": timings_ms,
                    "diagnostics": diagnostics,
                },
                created_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            )
        return EvidenceRetrievalResult(
            profile=profile,
            candidates=ranked_evidence,
            evidence_nodes=evidence_nodes,
            dense_object_hits=dense_objects,
            sparse_object_hits=sparse_objects,
            query_route=self._query_route_payload(query_route),
            timings_ms=timings_ms,
            diagnostics=diagnostics,
        )

    def complete_with_objects(
        self,
        query: str,
        evidence: EvidenceRetrievalResult,
        support_top_k: int = 4,
        object_top_k: int = 4,
    ) -> StructuredCompletionResult:
        total_start = perf_counter()
        profile = evidence.profile
        ranked_evidence = evidence.candidates
        timings_ms: dict[str, float] = {}
        diagnostics: dict[str, Any] = {"candidate_counts": {}, "decisions": {}, "cache": {}}
        runtime_before = self._snapshot_runtime_stats()
        self._object_lookup_ms = 0.0
        self._object_support_join_ms = 0.0
        query_route = dict(evidence.query_route or {})
        route_context = dict((evidence.diagnostics or {}).get("route_context") or {})
        diagnostics["query_route"] = query_route
        diagnostics["route_context"] = route_context

        evidence_signature = stable_content_hash(
            json.dumps(
                {
                    "chunks": [str(item.get("chunk_id") or "") for item in ranked_evidence],
                    "nodes": [str(item.get("node_id") or "") for item in ranked_evidence],
                    "route_type": str(query_route.get("route_type") or ""),
                    "object_top_k": int(object_top_k),
                    "support_top_k": int(support_top_k),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        memory_version = self._get_memory_version()
        cache_key = self._completion_cache_key(query, profile.task_type, evidence_signature)
        if (
            self.config.enable_completion_cache
            and memory_version > 0
            and hasattr(self.storage, "get_completion_cache")
        ):
            stage_start = perf_counter()
            cache_entry = self.storage.get_completion_cache(cache_key, memory_version)
            timings_ms["completion_cache_lookup_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
            if cache_entry and self._cache_is_fresh(cache_entry.get("created_at"), self.config.completion_cache_ttl_seconds):
                payload = cache_entry.get("payload") or {}
                evidence_nodes = self.storage.fetch_nodes_by_ids(list(payload.get("evidence_node_ids") or []))
                timings_ms.update({str(k): float(v) for k, v in dict(payload.get("timings_ms") or {}).items()})
                timings_ms["completion_cache_hit_ms"] = round((perf_counter() - total_start) * 1000.0, 2)
                timings_ms["total_ms"] = round((perf_counter() - total_start) * 1000.0, 2)
                cached_diagnostics = dict(payload.get("diagnostics") or {})
                cached_diagnostics.setdefault("cache", {})
                cached_diagnostics["cache"]["completion_hit"] = 1.0
                cached_diagnostics["cache"]["completion_miss"] = 0.0
                cached_diagnostics["backend"] = self._stats_delta(self._snapshot_runtime_stats(), runtime_before)
                return StructuredCompletionResult(
                    profile=profile,
                    core_evidence=[dict(item) for item in payload.get("core_evidence", [])],
                    evidence_objects=[dict(item) for item in payload.get("evidence_objects", [])],
                    supporting_context=[dict(item) for item in payload.get("supporting_context", [])],
                    evidence_nodes=evidence_nodes,
                    timings_ms=timings_ms,
                    diagnostics=cached_diagnostics,
                )
            diagnostics["cache"]["completion_hit"] = 0.0
            diagnostics["cache"]["completion_miss"] = 1.0
        else:
            timings_ms["completion_cache_lookup_ms"] = 0.0
            diagnostics["cache"]["completion_hit"] = 0.0
            diagnostics["cache"]["completion_miss"] = 0.0

        dense_objects: list[dict[str, Any]] = []
        sparse_objects: list[dict[str, Any]] = []
        evidence_objects: list[dict[str, Any]] = []
        if object_top_k > 0:
            stage_start = perf_counter()
            dense_objects = evidence.dense_object_hits or self.vector_store.search_objects(query, top_k=max(object_top_k * 4, 12))
            sparse_objects = evidence.sparse_object_hits or self._sparse_object_hits(query, profile, limit=max(object_top_k * 4, 12))
            timings_ms["object_candidate_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
            diagnostics["candidate_counts"]["dense_objects"] = len(dense_objects)
            diagnostics["candidate_counts"]["sparse_objects"] = len(sparse_objects)

            stage_start = perf_counter()
            evidence_objects = self._select_evidence_objects(
                query=query,
                profile=profile,
                ranked_evidence=ranked_evidence,
                dense_objects=dense_objects,
                sparse_objects=sparse_objects,
                limit=object_top_k,
                route_context=route_context,
            )
            timings_ms["object_rank_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
            diagnostics["candidate_counts"]["ranked_evidence_objects"] = len(evidence_objects)
        else:
            timings_ms["object_candidate_ms"] = 0.0
            timings_ms["object_rank_ms"] = 0.0
            diagnostics["candidate_counts"]["dense_objects"] = 0
            diagnostics["candidate_counts"]["sparse_objects"] = 0
        completion_object_limit = max(
            object_top_k,
            6 if (profile.needs_temporal_objects or profile.needs_personal_context_objects or profile.needs_relation_objects) else object_top_k,
        )

        if profile.needs_temporal_objects and completion_object_limit > 0:
            stage_start = perf_counter()
            temporal_objects = self._complete_temporal_objects(
                query=query,
                profile=profile,
                ranked_evidence=ranked_evidence,
                evidence_objects=evidence_objects,
                limit=completion_object_limit,
            )
            evidence_objects = self._merge_object_lists(evidence_objects, temporal_objects, limit=completion_object_limit)
            timings_ms["temporal_completion_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
            diagnostics["candidate_counts"]["temporal_completion_objects"] = len(temporal_objects)
        else:
            timings_ms["temporal_completion_ms"] = 0.0
            diagnostics["candidate_counts"]["temporal_completion_objects"] = 0

        if (profile.needs_personal_context_objects or profile.needs_relation_objects) and completion_object_limit > 0:
            stage_start = perf_counter()
            personal_context_objects = self._complete_personal_context_objects(
                query=query,
                profile=profile,
                ranked_evidence=ranked_evidence,
                evidence_objects=evidence_objects,
                limit=completion_object_limit,
            )
            evidence_objects = self._merge_object_lists(evidence_objects, personal_context_objects, limit=completion_object_limit)
            timings_ms["personal_context_completion_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
            diagnostics["candidate_counts"]["personal_context_completion_objects"] = len(personal_context_objects)
        else:
            timings_ms["personal_context_completion_ms"] = 0.0
            diagnostics["candidate_counts"]["personal_context_completion_objects"] = 0

        stage_start = perf_counter()
        if support_top_k > 0:
            neighbor_context = self._expand_supporting_context(ranked_evidence, limit=support_top_k)
            object_context = self._expand_object_supporting_context(
                evidence_objects=evidence_objects,
                ranked_evidence=ranked_evidence,
                limit=support_top_k,
            )
            supporting_context = self._merge_chunk_contexts(neighbor_context + object_context, limit=support_top_k)
        else:
            supporting_context = []
            neighbor_context = []
            object_context = []
        timings_ms["supporting_context_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
        timings_ms["object_lookup_ms"] = round(self._object_lookup_ms, 2)
        timings_ms["object_support_join_ms"] = round(self._object_support_join_ms, 2)
        diagnostics["candidate_counts"]["neighbor_context"] = len(neighbor_context)
        diagnostics["candidate_counts"]["object_context"] = len(object_context)
        diagnostics["candidate_counts"]["supporting_context"] = len(supporting_context)
        diagnostics["candidate_counts"]["evidence_objects"] = len(evidence_objects)
        diagnostics["top_evidence_objects"] = self._object_diagnostic_rows(evidence_objects, limit=min(8, object_top_k if object_top_k > 0 else 8))
        diagnostics["top_supporting_context"] = self._candidate_diagnostic_rows(supporting_context, limit=min(8, support_top_k if support_top_k > 0 else 8))

        evidence_node_ids: list[str] = []
        for item in ranked_evidence:
            node_id = str(item.get("node_id") or "")
            if node_id and node_id not in evidence_node_ids:
                evidence_node_ids.append(node_id)
        for item in evidence_objects:
            node_id = str(item.get("source_node_id") or "")
            if node_id and node_id not in evidence_node_ids:
                evidence_node_ids.append(node_id)

        stage_start = perf_counter()
        evidence_nodes = self.storage.fetch_nodes_by_ids(evidence_node_ids)
        timings_ms["fetch_completion_nodes_ms"] = round((perf_counter() - stage_start) * 1000.0, 2)
        timings_ms["total_ms"] = round((perf_counter() - total_start) * 1000.0, 2)
        diagnostics["candidate_counts"]["evidence_nodes"] = len(evidence_nodes)
        diagnostics["backend"] = self._stats_delta(self._snapshot_runtime_stats(), runtime_before)
        if (
            self.config.enable_completion_cache
            and memory_version > 0
            and hasattr(self.storage, "put_completion_cache")
        ):
            self.storage.put_completion_cache(
                cache_key=cache_key,
                evidence_signature=evidence_signature,
                task_type=profile.task_type,
                memory_version=memory_version,
                payload={
                    "core_evidence": ranked_evidence,
                    "evidence_objects": evidence_objects,
                    "supporting_context": supporting_context,
                    "evidence_node_ids": evidence_node_ids,
                    "timings_ms": timings_ms,
                    "diagnostics": diagnostics,
                },
                created_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            )
        return StructuredCompletionResult(
            profile=profile,
            core_evidence=ranked_evidence,
            evidence_objects=evidence_objects,
            supporting_context=supporting_context,
            evidence_nodes=evidence_nodes,
            timings_ms=timings_ms,
            diagnostics=diagnostics,
        )

    def augment_cognitively(
        self,
        query: str,
        task_type: str,
        completion: StructuredCompletionResult,
        cognitive_top_k: int = 4,
    ) -> CognitiveAugmentationResult:
        total_start = perf_counter()
        profile = completion.profile
        evidence_nodes = completion.evidence_nodes
        if cognitive_top_k <= 0 or not evidence_nodes:
            return CognitiveAugmentationResult(
                relevant_experience=[],
                creative_reflections=[],
                alternative_paths=[],
                timings_ms={"total_ms": round((perf_counter() - total_start) * 1000.0, 2)},
                diagnostics={
                    "candidate_counts": {
                        "evidence_nodes": len(evidence_nodes),
                        "relevant_experience": 0,
                        "creative_reflections": 0,
                        "alternative_paths": 0,
                    },
                    "decisions": {
                        "executed": False,
                        "reason": "cognitive_top_k_zero" if cognitive_top_k <= 0 else "no_evidence_nodes",
                    },
                    "prism": {},
                },
            )
        runtime_before = self._snapshot_runtime_stats()
        query_route = dict(completion.diagnostics.get("query_route") or {})
        route_type = str(query_route.get("route_type") or "")
        allow_creative = bool(query_route["allow_creative"]) if "allow_creative" in query_route else True
        relevant_experience, creative_reflections = self._expand_cognitive(
            query=query,
            profile=profile,
            evidence_nodes=evidence_nodes,
            limit=cognitive_top_k,
            include_refraction=(not self.config.creative_enabled) and allow_creative,
        )
        alternative_paths: list[dict[str, Any]] = []
        prism_diagnostics: dict[str, Any] = {}
        if self.config.creative_enabled and allow_creative:
            stage_start = perf_counter()
            creative_reflections, alternative_paths, prism_diagnostics = self.prism.propagate(
                query=query,
                task_type=task_type,
                profile=profile,
                core_evidence=completion.core_evidence,
                evidence_nodes=evidence_nodes,
                supporting_context=completion.supporting_context,
                limit=cognitive_top_k,
            )
            prism_total_ms = round((perf_counter() - stage_start) * 1000.0, 2)
        else:
            if self.config.creative_enabled and not allow_creative:
                prism_diagnostics = {"enabled": False, "reason": f"route_suppressed:{route_type or 'unknown'}"}
            prism_total_ms = 0.0
        return CognitiveAugmentationResult(
            relevant_experience=relevant_experience,
            creative_reflections=creative_reflections,
            alternative_paths=alternative_paths,
            timings_ms={
                "prism_total_ms": prism_total_ms,
                "total_ms": round((perf_counter() - total_start) * 1000.0, 2),
            },
            diagnostics={
                "candidate_counts": {
                    "evidence_nodes": len(evidence_nodes),
                    "relevant_experience": len(relevant_experience),
                    "creative_reflections": len(creative_reflections),
                    "alternative_paths": len(alternative_paths),
                },
                "decisions": {
                    "allow_creative": allow_creative,
                    "route_type": route_type,
                },
                "prism": prism_diagnostics,
                "backend": self._stats_delta(self._snapshot_runtime_stats(), runtime_before),
            },
        )

    def _sparse_chunk_hits(self, query: str, limit: int) -> list[dict[str, Any]]:
        rows = self.storage.search_chunks_fts(query, limit=limit)
        scored: list[dict[str, Any]] = []
        for row in rows:
            bm25_score = float(row.get("bm25_score") or 0.0)
            scored.append(
                {
                    "chunk_id": row["chunk_id"],
                    "document": row["text"],
                    "metadata": {
                        "node_id": row["node_id"],
                        "shell": row.get("shell"),
                        "sector": row.get("sector"),
                        "zone": row.get("zone"),
                        "cell": row.get("cell"),
                        "chunk_index": row.get("chunk_index"),
                        "grain": row.get("grain", "micro"),
                        "scope": row.get("scope"),
                        "workspace": row.get("workspace"),
                        "project": row.get("project"),
                        "session_id": row.get("session_id"),
                        "summary": row.get("summary"),
                        "content_ref": row.get("content_ref"),
                        "source_type": row.get("source_type"),
                        "source_ref": row.get("source_ref"),
                        "access_count": row.get("access_count"),
                        "created_at": row.get("created_at"),
                    },
                    "similarity": round(self._bm25_similarity(bm25_score), 4),
                }
            )
        return scored

    @staticmethod
    def _ranking_diagnostics(ranked_candidates: list[dict[str, Any]]) -> dict[str, float]:
        if not ranked_candidates:
            return {
                "top_margin_1_2": 0.0,
                "top_margin_1_3": 0.0,
                "temporal_prior_positive_count": 0.0,
                "temporal_prior_negative_count": 0.0,
                "object_support_positive_count": 0.0,
                "identity_reward_positive_count": 0.0,
                "wrong_entity_penalty_count": 0.0,
                "segment_positive_count": 0.0,
                "confusing_neighbor_count": 0.0,
            }
        top_score = float(ranked_candidates[0].get("evidence_score") or 0.0)
        second_score = float(ranked_candidates[1].get("evidence_score") or 0.0) if len(ranked_candidates) > 1 else 0.0
        third_score = float(ranked_candidates[2].get("evidence_score") or 0.0) if len(ranked_candidates) > 2 else second_score
        return {
            "top_margin_1_2": round(top_score - second_score, 4),
            "top_margin_1_3": round(top_score - third_score, 4),
            "temporal_prior_positive_count": float(sum(1 for item in ranked_candidates if float(item.get("temporal_prior_score") or 0.0) > 0.01)),
            "temporal_prior_negative_count": float(sum(1 for item in ranked_candidates if float(item.get("temporal_prior_score") or 0.0) < -0.01)),
            "object_support_positive_count": float(sum(1 for item in ranked_candidates if float(item.get("object_support_score") or 0.0) > 0.01)),
            "identity_reward_positive_count": float(sum(1 for item in ranked_candidates if float(item.get("identity_match_reward") or 0.0) > 0.01)),
            "wrong_entity_penalty_count": float(
                sum(
                    1
                    for item in ranked_candidates
                    if (
                        float(item.get("wrong_entity_penalty") or 0.0)
                        + float(item.get("same_topic_different_entity_penalty") or 0.0)
                        + float(item.get("role_source_mismatch_penalty") or 0.0)
                        + float(item.get("wrong_domain_penalty") or 0.0)
                        + float(item.get("wrong_role_target_penalty") or 0.0)
                        + float(item.get("wrong_subtheme_penalty") or 0.0)
                        + float(item.get("generic_topic_penalty") or 0.0)
                    )
                    > 0.01
                )
            ),
            "segment_positive_count": float(sum(1 for item in ranked_candidates if float(item.get("segment_rerank_delta") or 0.0) > 0.01)),
            "confusing_neighbor_count": float(sum(1 for item in ranked_candidates if float(item.get("confusing_neighbor_penalty") or 0.0) > 0.01)),
        }

    @staticmethod
    def _candidate_diagnostic_rows(candidates: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates[: max(0, limit)], start=1):
            preview = " ".join(str(candidate.get("text") or "").split())
            if len(preview) > 220:
                preview = preview[:217].rstrip() + "..."
            rows.append(
                {
                    "rank": rank,
                    "chunk_id": str(candidate.get("chunk_id") or ""),
                    "node_id": str(candidate.get("node_id") or ""),
                    "grain": str(candidate.get("grain") or "unknown"),
                    "shell": candidate.get("shell"),
                    "sector": candidate.get("sector"),
                    "zone": candidate.get("zone"),
                    "cell": candidate.get("cell"),
                    "scope": candidate.get("scope"),
                    "workspace": candidate.get("workspace"),
                    "project": candidate.get("project"),
                    "session_id": candidate.get("session_id"),
                    "content_ref": candidate.get("content_ref"),
                    "route_variants": list(candidate.get("route_variants") or []),
                    "focused_query_match": bool(candidate.get("focused_query_match", False)),
                    "focused_query_hits": int(candidate.get("focused_query_hits") or 0),
                    "focused_query_sources": list(candidate.get("focused_query_sources") or []),
                    "focused_reserve": bool(candidate.get("focused_reserve", False)),
                    "evidence_score": round(float(candidate.get("evidence_score") or 0.0), 4),
                    "direct_anchor_score": round(float(candidate.get("direct_anchor_score") or 0.0), 4),
                    "dense_score": round(float(candidate.get("dense_score") or 0.0), 4),
                    "proxy_score": round(float(candidate.get("proxy_score") or 0.0), 4),
                    "query_lexical": round(float(candidate.get("query_lexical") or 0.0), 4),
                    "salience_score": round(float(candidate.get("salience_score") or 0.0), 4),
                    "person_name_score": round(float(candidate.get("person_name_score") or 0.0), 4),
                    "identity_match_reward": round(float(candidate.get("identity_match_reward") or 0.0), 4),
                    "wrong_entity_penalty": round(float(candidate.get("wrong_entity_penalty") or 0.0), 4),
                    "same_topic_different_entity_penalty": round(float(candidate.get("same_topic_different_entity_penalty") or 0.0), 4),
                    "role_source_mismatch_penalty": round(float(candidate.get("role_source_mismatch_penalty") or 0.0), 4),
                    "wrong_domain_penalty": round(float(candidate.get("wrong_domain_penalty") or 0.0), 4),
                    "wrong_role_target_penalty": round(float(candidate.get("wrong_role_target_penalty") or 0.0), 4),
                    "wrong_subtheme_penalty": round(float(candidate.get("wrong_subtheme_penalty") or 0.0), 4),
                    "generic_topic_penalty": round(float(candidate.get("generic_topic_penalty") or 0.0), 4),
                    "identity_rerank_delta": round(float(candidate.get("identity_rerank_delta") or 0.0), 4),
                    "time_score": round(float(candidate.get("time_score") or 0.0), 4),
                    "temporal_prior_score": round(float(candidate.get("temporal_prior_score") or 0.0), 4),
                    "preference_score": round(float(candidate.get("preference_score") or 0.0), 4),
                    "diagnostic_score": round(float(candidate.get("diagnostic_score") or 0.0), 4),
                    "object_support_raw": round(float(candidate.get("object_support_raw") or 0.0), 4),
                    "object_support_gate": round(float(candidate.get("object_support_gate") or 0.0), 4),
                    "object_support_score": round(float(candidate.get("object_support_score") or 0.0), 4),
                    "local_context_gate": round(float(candidate.get("local_context_gate") or 0.0), 4),
                    "local_context_score": round(float(candidate.get("local_context_score") or 0.0), 4),
                    "grain_score": round(float(candidate.get("grain_score") or 0.0), 4),
                    "specificity_score": round(float(candidate.get("specificity_score") or 0.0), 4),
                    "structure_gate": round(float(candidate.get("structure_gate") or 0.0), 4),
                    "structure_score": round(float(candidate.get("structure_score") or 0.0), 4),
                    "scope_bonus": round(float(candidate.get("scope_bonus") or 0.0), 4),
                    "scope_match": str(candidate.get("scope_match") or ""),
                    "confusing_neighbor_penalty": round(float(candidate.get("confusing_neighbor_penalty") or 0.0), 4),
                    "confusing_cluster_size": int(candidate.get("confusing_cluster_size") or 0),
                    "confusing_cluster_role": str(candidate.get("confusing_cluster_role") or ""),
                    "segment_rerank_score": round(float(candidate.get("segment_rerank_score") or 0.0), 4),
                    "segment_semantic_score": round(float(candidate.get("segment_semantic_score") or 0.0), 4),
                    "segment_identity_score": round(float(candidate.get("segment_identity_score") or 0.0), 4),
                    "segment_temporal_score": round(float(candidate.get("segment_temporal_score") or 0.0), 4),
                    "segment_attribute_score": round(float(candidate.get("segment_attribute_score") or 0.0), 4),
                    "segment_contradiction_penalty": round(float(candidate.get("segment_contradiction_penalty") or 0.0), 4),
                    "segment_rerank_delta": round(float(candidate.get("segment_rerank_delta") or 0.0), 4),
                    "identity_signature": str(candidate.get("identity_signature") or ""),
                    "exact_evidence_guard": bool(candidate.get("exact_evidence_guard", False)),
                    "timestamp_source": str(candidate.get("candidate_timestamp_source") or ""),
                    "best_span_text": str(candidate.get("best_span_text") or "")[:220],
                    "preview": preview,
                }
            )
        return rows

    @staticmethod
    def _object_diagnostic_rows(objects: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rank, obj in enumerate(objects[: max(0, limit)], start=1):
            preview = " ".join(str(obj.get("object_text") or "").split())
            if len(preview) > 180:
                preview = preview[:177].rstrip() + "..."
            rows.append(
                {
                    "rank": rank,
                    "object_id": str(obj.get("object_id") or ""),
                    "object_type": str(obj.get("object_type") or ""),
                    "source_node_id": str(obj.get("source_node_id") or ""),
                    "source_chunk_id": str(obj.get("source_chunk_id") or ""),
                    "entity": obj.get("entity"),
                    "attribute": obj.get("attribute"),
                    "temporal_marker": obj.get("temporal_marker"),
                    "object_score": round(float(obj.get("object_score") or 0.0), 4),
                    "dense_score": round(float(obj.get("dense_score") or 0.0), 4),
                    "lexical_score": round(float(obj.get("lexical_score") or 0.0), 4),
                    "preference_object_score": round(float(obj.get("preference_object_score") or 0.0), 4),
                    "temporal_object_score": round(float(obj.get("temporal_object_score") or 0.0), 4),
                    "personal_context_score": round(float(obj.get("personal_context_score") or 0.0), 4),
                    "relation_object_score": round(float(obj.get("relation_object_score") or 0.0), 4),
                    "artifact_object_score": round(float(obj.get("artifact_object_score") or 0.0), 4),
                    "open_loop_object_score": round(float(obj.get("open_loop_object_score") or 0.0), 4),
                    "scope_bonus": round(float(obj.get("scope_bonus") or 0.0), 4),
                    "scope_match": str(obj.get("scope_match") or ""),
                    "scope": str(obj.get("scope") or ""),
                    "project": str(obj.get("project") or ""),
                    "session_id": str(obj.get("session_id") or ""),
                    "status": str(obj.get("status") or ""),
                    "source_type": str(obj.get("source_type") or ""),
                    "source_ref": str(obj.get("source_ref") or ""),
                    "snapshot_state": str(obj.get("snapshot_state") or ""),
                    "effective_time": obj.get("effective_time"),
                    "valid_time": obj.get("valid_time"),
                    "preview": preview,
                }
            )
        return rows

    @staticmethod
    def _is_longmemeval_focused_reserve_candidate(
        candidate: dict[str, Any],
        route_context: dict[str, Any] | None,
    ) -> bool:
        if not route_context:
            return False
        if str(route_context.get("benchmark") or "").lower() != "longmemeval":
            return False
        if not str(route_context.get("focused_query") or "").strip():
            return False
        if not bool(candidate.get("focused_query_match")):
            return False
        return (
            float(candidate.get("query_lexical") or 0.0) >= 0.12
            or float(candidate.get("direct_anchor_score") or 0.0) >= 0.18
        )

    @staticmethod
    def _factual_contamination_rate(candidates: list[dict[str, Any]]) -> float:
        if not candidates:
            return 0.0
        contaminated = 0
        for candidate in candidates:
            direct_anchor = float(candidate.get("direct_anchor_score") or 0.0)
            object_support = float(candidate.get("object_support_score") or 0.0)
            segment_score = float(candidate.get("segment_rerank_score") or 0.0)
            if direct_anchor < 0.12 and object_support > 0.08 and segment_score < 0.08:
                contaminated += 1
        return round(contaminated / max(1, len(candidates)), 4)

    @staticmethod
    def _apply_exact_evidence_guard(
        ranked_candidates: list[dict[str, Any]],
        profile: QueryProfile,
    ) -> tuple[list[dict[str, Any]], str]:
        if not ranked_candidates:
            return ranked_candidates, "skipped_no_candidates"
        guarded = [dict(candidate) for candidate in ranked_candidates]
        for candidate in guarded:
            candidate["exact_evidence_guard"] = False
        if not profile.needs_exact_evidence:
            return guarded, "skipped_non_exact_profile"
        top = guarded[0]
        top_anchor = float(top.get("direct_anchor_score") or 0.0)
        top_lexical = float(top.get("query_lexical") or 0.0)
        if top_anchor >= 0.34 and top_lexical >= 0.18:
            top["exact_evidence_guard"] = True
            return guarded, "kept_existing_direct_lead"

        search_pool = guarded[: min(len(guarded), 12)]
        best_rank, best_direct = max(
            enumerate(search_pool, start=1),
            key=lambda item: (
                float(item[1].get("direct_anchor_score") or 0.0),
                float(item[1].get("query_lexical") or 0.0),
                float(item[1].get("dense_score") or 0.0),
            ),
        )
        direct_anchor = float(best_direct.get("direct_anchor_score") or 0.0)
        score_gap = float(top.get("evidence_score") or 0.0) - float(best_direct.get("evidence_score") or 0.0)
        if best_rank > 1 and direct_anchor >= max(0.32, top_anchor + 0.06) and score_gap <= 0.08:
            best_direct["exact_evidence_guard"] = True
            best_direct["evidence_score"] = round(float(best_direct.get("evidence_score") or 0.0) + 0.04, 4)
            guarded[best_rank - 1] = best_direct
            guarded.sort(key=lambda item: float(item.get("evidence_score") or 0.0), reverse=True)
            return guarded, f"promoted_rank_{best_rank}_direct_anchor"
        return guarded, "not_needed_no_clear_direct_candidate"

    def _sparse_object_hits(self, query: str, profile: QueryProfile, limit: int) -> list[dict[str, Any]]:
        objects = self.storage.search_objects_fts(query, limit=limit, object_types=list(dict.fromkeys(profile.preferred_object_types)) or None)
        scored: list[dict[str, Any]] = []
        for obj in objects:
            score = self._bm25_similarity(float(obj.get("bm25_score") or 0.0))
            scored.append(
                {
                    "object_id": obj["object_id"],
                    "document": obj["object_text"],
                    "metadata": {
                        "object_type": obj.get("object_type"),
                        "source_chunk_id": obj.get("source_chunk_id"),
                        "source_node_id": obj.get("source_node_id"),
                        "scope": obj.get("scope"),
                        "workspace": obj.get("workspace"),
                        "project": obj.get("project"),
                        "session_id": obj.get("session_id"),
                        "status": obj.get("status"),
                        "entity": obj.get("entity"),
                        "attribute": obj.get("attribute"),
                        "canonical_key": obj.get("canonical_key"),
                        "temporal_marker": obj.get("temporal_marker"),
                        "source_type": obj.get("source_type"),
                        "source_ref": obj.get("source_ref"),
                    },
                    "similarity": round(score, 4),
                }
            )
        return scored

    def _merge_chunk_candidates(
        self,
        query: str,
        dense_chunks: list[dict[str, Any]],
        sparse_chunks: list[dict[str, Any]],
        proxy_chunks: list[dict[str, Any]],
        dense_objects: list[dict[str, Any]],
        sparse_objects: list[dict[str, Any]],
        diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        node_to_candidates: dict[str, list[dict[str, Any]]] = {}
        fallback_stats = None
        if diagnostics is not None:
            fallback_stats = diagnostics.setdefault(
                "storage_fallback",
                {
                    "metadata_complete_candidates": 0.0,
                    "node_fallback_candidates": 0.0,
                    "neighbor_fallback_candidates": 0.0,
                    "node_fetch_calls": 0.0,
                    "neighbor_fetch_calls": 0.0,
                },
            )

        def upsert_chunk_hit(hit: dict[str, Any], source_name: str, rank: int) -> None:
            meta = hit.get("metadata") or {}
            chunk_id = str(hit.get("chunk_id") or "")
            hit_similarity = float(hit.get("similarity") or 0.0)
            if not chunk_id:
                return
            current = candidates.get(chunk_id)
            if current is None:
                node_id = str(meta.get("node_id") or "")
                current = {
                    "chunk_id": chunk_id,
                    "node_id": node_id,
                    "text": hit.get("document") or "",
                    "chunk_index": meta.get("chunk_index"),
                    "shell": meta.get("shell"),
                    "sector": meta.get("sector"),
                    "zone": meta.get("zone"),
                    "cell": meta.get("cell"),
                    "grain": meta.get("grain") or "micro",
                    "scope": meta.get("scope") or meta.get("node_scope") or "global",
                    "workspace": meta.get("workspace") or meta.get("node_workspace"),
                    "project": meta.get("project") or meta.get("node_project"),
                    "session_id": meta.get("session_id") or meta.get("node_session_id"),
                    "summary": meta.get("summary"),
                    "retrieval_summary": meta.get("retrieval_summary"),
                    "structured_summary": meta.get("structured_summary"),
                    "retrieval_signature": meta.get("retrieval_signature"),
                    "time_bucket": meta.get("time_bucket"),
                    "entity_tags": meta.get("entity_tags"),
                    "task_type_tag": meta.get("task_type_tag"),
                    "content_ref": meta.get("content_ref") or meta.get("session_id"),
                    "source_type": meta.get("source_type"),
                    "source_ref": meta.get("source_ref"),
                    "access_count": meta.get("access_count"),
                    "neighbor_count": meta.get("neighbor_count"),
                    "created_at": meta.get("created_at") or meta.get("timestamp") or meta.get("session_time") or meta.get("source_time"),
                    "dense_score": 0.0,
                    "proxy_score": 0.0,
                    "lexical_score": 0.0,
                    "fusion_rrf": 0.0,
                    "object_support": 0.0,
                    "route_variants": [],
                    "focused_query_match": False,
                    "focused_query_hits": 0,
                    "focused_query_sources": [],
                }
                candidates[chunk_id] = current
                if node_id:
                    node_to_candidates.setdefault(node_id, []).append(current)
            route_variant = str(hit.get("route_variant") or "")
            if route_variant:
                route_variants = current.setdefault("route_variants", [])
                if route_variant not in route_variants:
                    route_variants.append(route_variant)
                if route_variant == "focused":
                    current["focused_query_match"] = True
                    current["focused_query_hits"] = int(current.get("focused_query_hits") or 0) + 1
                    focused_query_sources = current.setdefault("focused_query_sources", [])
                    if source_name not in focused_query_sources:
                        focused_query_sources.append(source_name)
            current["fusion_rrf"] += 1.0 / (60 + rank)
            if source_name == "dense":
                current["dense_score"] = max(current["dense_score"], hit_similarity)
            elif source_name == "proxy":
                current["proxy_score"] = max(current["proxy_score"], hit_similarity)
                current["dense_score"] = max(current["dense_score"], hit_similarity * 0.92)
            elif source_name == "sparse":
                current["lexical_score"] = max(current["lexical_score"], hit_similarity)

        for rank, hit in enumerate(dense_chunks, start=1):
            upsert_chunk_hit(hit, "dense", rank)
        for rank, hit in enumerate(proxy_chunks, start=1):
            upsert_chunk_hit(hit, "proxy", rank)
        for rank, hit in enumerate(sparse_chunks, start=1):
            upsert_chunk_hit(hit, "sparse", rank)

        object_support_by_chunk_id: dict[str, float] = {}
        object_support_by_node_id: dict[str, float] = {}
        for rank, hit in enumerate(dense_objects, start=1):
            meta = hit.get("metadata") or {}
            source_chunk_id = str(meta.get("source_chunk_id") or "")
            source_node_id = str(meta.get("source_node_id") or "")
            support = 1.0 / (80 + rank)
            if source_chunk_id and source_chunk_id in candidates:
                object_support_by_chunk_id[source_chunk_id] = object_support_by_chunk_id.get(source_chunk_id, 0.0) + support
            elif source_node_id:
                object_support_by_node_id[source_node_id] = object_support_by_node_id.get(source_node_id, 0.0) + support

        for rank, hit in enumerate(sparse_objects, start=1):
            meta = hit.get("metadata") or {}
            source_chunk_id = str(meta.get("source_chunk_id") or "")
            source_node_id = str(meta.get("source_node_id") or "")
            support = 1.0 / (90 + rank)
            if source_chunk_id and source_chunk_id in candidates:
                object_support_by_chunk_id[source_chunk_id] = object_support_by_chunk_id.get(source_chunk_id, 0.0) + support
            elif source_node_id:
                object_support_by_node_id[source_node_id] = object_support_by_node_id.get(source_node_id, 0.0) + support

        for chunk_id, support in object_support_by_chunk_id.items():
            if chunk_id in candidates:
                candidates[chunk_id]["object_support"] += support
        for node_id, support in object_support_by_node_id.items():
            node_candidates = node_to_candidates.get(node_id, [])
            if not node_candidates:
                continue
            preferred_candidates = [candidate for candidate in node_candidates if candidate.get("grain") != "macro"] or node_candidates
            total_weight = sum(1.0 if candidate.get("grain") == "micro" else 0.75 for candidate in preferred_candidates)
            if total_weight <= 0:
                continue
            for candidate in preferred_candidates:
                grain_weight = 1.0 if candidate.get("grain") == "micro" else 0.75
                candidate["object_support"] += support * (grain_weight / total_weight)

        missing_node_ids: list[str] = []
        missing_neighbor_chunk_ids: list[str] = []
        for candidate in candidates.values():
            has_summary = bool(str(candidate.get("summary") or "").strip())
            has_content_ref = bool(str(candidate.get("content_ref") or "").strip())
            has_access_count = candidate.get("access_count") is not None
            has_created_at = bool(str(candidate.get("created_at") or "").strip())
            if has_summary and has_content_ref and has_access_count and has_created_at:
                if fallback_stats is not None:
                    fallback_stats["metadata_complete_candidates"] += 1.0
            elif candidate.get("node_id"):
                missing_node_ids.append(str(candidate["node_id"]))
            if candidate.get("neighbor_count") is None:
                missing_neighbor_chunk_ids.append(str(candidate["chunk_id"]))

        node_map: dict[str, dict[str, Any]] = {}
        if missing_node_ids:
            node_rows = self.storage.fetch_nodes_by_ids(list(dict.fromkeys(missing_node_ids)))
            node_map = {str(row["id"]): row for row in node_rows}
            if fallback_stats is not None:
                fallback_stats["node_fetch_calls"] += 1.0
                fallback_stats["node_fallback_candidates"] += float(len(missing_node_ids))

        neighbor_counts: dict[str, int] = {}
        if missing_neighbor_chunk_ids:
            neighbor_counts = self.storage.fetch_neighbor_counts(list(dict.fromkeys(missing_neighbor_chunk_ids)))
            if fallback_stats is not None:
                fallback_stats["neighbor_fetch_calls"] += 1.0
                fallback_stats["neighbor_fallback_candidates"] += float(len(missing_neighbor_chunk_ids))

        for candidate in candidates.values():
            node = node_map.get(str(candidate.get("node_id") or ""), {})
            if not candidate.get("summary"):
                candidate["summary"] = node.get("summary") or ""
            if not candidate.get("scope"):
                candidate["scope"] = node.get("scope") or "global"
            if not candidate.get("workspace"):
                candidate["workspace"] = node.get("workspace")
            if not candidate.get("project"):
                candidate["project"] = node.get("project")
            if not candidate.get("session_id"):
                candidate["session_id"] = node.get("session_id")
            if not candidate.get("content_ref"):
                candidate["content_ref"] = node.get("content_ref")
            if not candidate.get("source_ref"):
                candidate["source_ref"] = node.get("source_ref")
            if candidate.get("access_count") is None:
                candidate["access_count"] = node.get("access_count") or 0
            if not candidate.get("created_at"):
                candidate["created_at"] = node.get("created_at") or node.get("last_accessed_at") or ""
            candidate["query_lexical"] = lexical_score(query, " ".join(filter(None, [candidate["text"], candidate["summary"], candidate["cell"], candidate["zone"]])))
            if candidate.get("neighbor_count") is None:
                candidate["neighbor_count"] = neighbor_counts.get(str(candidate["chunk_id"]), 0)
        return list(candidates.values())

    def _rank_chunk_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        profile: QueryProfile,
        route: Any,
        route_context: dict[str, Any] | None = None,
        route_tuning: BenchmarkRouteTuning | None = None,
    ) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        best_same_node_non_macro = self._best_non_macro_support_by_node(candidates)
        query_tokens = set(token_tuple(query))
        query_names = self._extract_person_names(query)
        salient_terms = self._salient_query_terms(query_tokens)
        temporal_state = self._build_temporal_prior_state(profile, candidates)
        focus_names = self._route_focus_names(route_context, profile)
        workspace_context = self._workspace_context(route_context)
        tuning = route_tuning or BenchmarkRouteTuning()
        query_identity_features = self._identity_features(
            text=query,
            metadata=route_context,
            focus_names=focus_names,
            prefix="query",
        )
        exact_boost = 1.0 if profile.needs_exact_evidence else 0.0
        temporal_boost = 1.0 if profile.needs_temporal_objects else 0.0
        preference_boost = 1.0 if profile.needs_preference_objects else 0.0
        dense_weight = 0.5 - exact_boost * 0.04 - preference_boost * 0.04
        rrf_weight = 11.5 - exact_boost * 1.4 - temporal_boost * 0.8 - preference_boost * 0.8
        lexical_weight = 0.24 + profile.lexical_priority * 0.12 + exact_boost * 0.08 + temporal_boost * 0.03 + preference_boost * 0.04
        salience_weight = 0.1 + exact_boost * 0.12 + temporal_boost * 0.03 + preference_boost * 0.04
        time_weight = 0.06 + temporal_boost * 0.16
        temporal_prior_weight = 0.04 + temporal_boost * 0.18
        preference_weight = 0.06 + preference_boost * 0.16
        diagnostic_weight = 0.3
        person_name_weight = 0.08 + exact_boost * 0.08
        object_support_weight = 0.2 + exact_boost * 0.04 + temporal_boost * 0.08 + preference_boost * 0.1
        specificity_weight = 1.0 + exact_boost * 0.45 + temporal_boost * 0.15 + preference_boost * 0.18
        grain_weight = 1.0 + exact_boost * 0.2
        structure_weight = 0.18
        for candidate in candidates:
            active_cache = getattr(self, "_active_candidate_feature_cache", None)
            feature_stats = getattr(self, "_active_feature_cache_stats", None)
            cache_key = ""
            if isinstance(active_cache, dict):
                cache_key = stable_content_hash(
                    json.dumps(
                        {
                            "chunk_id": str(candidate.get("chunk_id") or ""),
                            "node_id": str(candidate.get("node_id") or ""),
                            "text_hash": normalize_text_for_hash(str(candidate.get("text") or "")[:1200]).lower(),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            cached_features = active_cache.get(cache_key) if isinstance(active_cache, dict) and cache_key else None
            if cached_features is not None:
                if isinstance(feature_stats, dict):
                    feature_stats["hits"] = int(feature_stats.get("hits") or 0) + 1
                candidate_text = str(cached_features.get("text") or "")
                candidate_lower = str(cached_features.get("lower") or "")
                candidate_tokens = set(cached_features.get("tokens") or [])
            else:
                if isinstance(feature_stats, dict):
                    feature_stats["misses"] = int(feature_stats.get("misses") or 0) + 1
                candidate_text, candidate_lower, candidate_tokens = self._candidate_text_features(candidate)
                if isinstance(active_cache, dict) and cache_key:
                    active_cache[cache_key] = {
                        "text": candidate_text,
                        "lower": candidate_lower,
                        "tokens": list(candidate_tokens),
                        "normalized_text_hash": normalize_text_for_hash(candidate_text).lower(),
                    }
            time_score = self._time_score(profile, query_tokens, candidate_lower, candidate_tokens)
            temporal_prior_score, candidate_epoch, timestamp_source = self._temporal_prior_score(profile, candidate, temporal_state)
            preference_score = self._preference_score(profile, query_tokens, candidate, candidate_lower, candidate_tokens)
            diagnostic_score = self._diagnostic_score(query, query_tokens, candidate_lower, candidate_tokens)
            person_name_score = self._person_name_score(query_names, candidate_text)
            candidate_identity_features = self._identity_features(
                text=candidate_text,
                metadata=candidate,
                focus_names=focus_names,
                prefix=str(candidate.get("chunk_id") or candidate.get("node_id") or "candidate"),
            )
            identity_components = score_identity_alignment(
                query_identity_features,
                candidate_identity_features,
                text_similarity=max(float(candidate.get("dense_score") or 0.0), float(candidate.get("query_lexical") or 0.0)),
            )
            identity_reward = float(identity_components.get("identity_match_reward") or 0.0) * max(0.0, tuning.identity_reward_weight or 0.0)
            wrong_entity_penalty = float(identity_components.get("wrong_entity_penalty") or 0.0) * max(0.0, tuning.wrong_entity_penalty_weight or 0.0)
            same_topic_different_entity_penalty = float(identity_components.get("same_topic_different_entity_penalty") or 0.0) * max(0.0, tuning.wrong_entity_penalty_weight or 0.0)
            role_source_mismatch_penalty = float(identity_components.get("role_source_mismatch_penalty") or 0.0) * max(0.0, tuning.wrong_entity_penalty_weight or 0.0)
            wrong_domain_penalty = float(identity_components.get("wrong_domain_penalty") or 0.0) * max(
                float(self.config.wrong_domain_penalty_weight or 0.0),
                float(tuning.wrong_domain_penalty_weight or 0.0),
            )
            wrong_role_target_penalty = float(identity_components.get("wrong_role_target_penalty") or 0.0) * max(
                float(self.config.wrong_role_target_penalty_weight or 0.0),
                float(tuning.wrong_role_target_penalty_weight or 0.0),
            )
            wrong_subtheme_penalty = float(identity_components.get("wrong_subtheme_penalty") or 0.0) * max(
                float(self.config.wrong_subtheme_penalty_weight or 0.0),
                float(tuning.wrong_subtheme_penalty_weight or 0.0),
            )
            generic_topic_penalty = float(identity_components.get("generic_topic_penalty") or 0.0) * max(
                float(self.config.generic_topic_penalty_weight or 0.0),
                float(tuning.generic_topic_penalty_weight or 0.0),
            )
            salience_score = self._salience_score(salient_terms, candidate_tokens)
            obj_support_cap = 0.22 if profile.needs_preference_objects else 0.18 if (profile.needs_temporal_objects or profile.needs_exact_evidence) else 0.12
            object_support_raw = min(float(candidate.get("object_support", 0.0) or 0.0), obj_support_cap)
            direct_anchor_score = min(
                1.0,
                float(candidate.get("dense_score", 0.0)) * 0.95
                + float(candidate.get("query_lexical", 0.0)) * 1.25
                + salience_score * 0.8
                + person_name_score
                + (preference_score * 0.45 if profile.needs_preference_objects else 0.0)
                + (time_score * 0.35 if profile.temporal_mode != "none" else 0.0),
            )
            object_support_gate = min(1.0, max(0.0, 0.1 + direct_anchor_score * 1.1))
            if direct_anchor_score < 0.12:
                object_support_gate *= 0.35
            object_support_score = object_support_raw * object_support_gate
            raw_local_context = min(candidate.get("neighbor_count", 0) * 0.03, 0.08)
            local_context_gate = min(1.0, max(0.0, direct_anchor_score - 0.08) * 1.35)
            if direct_anchor_score < 0.16 and candidate.get("grain") != "micro":
                local_context_gate *= 0.6
            local_context_score = raw_local_context * local_context_gate
            raw_structure_score = min((candidate.get("access_count") or 0) / 120.0, 0.04)
            grain_score = self._grain_score(candidate.get("grain", "micro"), profile)
            specificity_score = self._specificity_score(
                candidate=candidate,
                profile=profile,
                best_same_node_non_macro=best_same_node_non_macro.get(str(candidate.get("node_id") or ""), 0.0),
            )
            if candidate.get("sector") in route.preferred_sectors:
                raw_structure_score += 0.015
            if candidate.get("shell") in route.preferred_shells:
                raw_structure_score += 0.01
            structure_gate = min(1.0, 0.15 + direct_anchor_score * 0.9)
            if candidate.get("grain") == "macro" and direct_anchor_score < 0.22:
                structure_gate *= 0.75
            structure_score = raw_structure_score * structure_gate
            proxy_score = min(float(candidate.get("proxy_score") or 0.0), 0.18)
            scope_bonus, scope_match_label = workspace_context.candidate_scope_bonus(
                scope=str(candidate.get("scope") or ""),
                project=str(candidate.get("project") or ""),
                session_id=str(candidate.get("session_id") or ""),
                workspace=str(candidate.get("workspace") or ""),
                weight=self.config.scope_priority_weight if self.config.enable_scope_priority else 0.0,
            )

            final_score = (
                candidate.get("dense_score", 0.0) * dense_weight
                + candidate.get("fusion_rrf", 0.0) * rrf_weight
                + candidate.get("query_lexical", 0.0) * lexical_weight
                + salience_score * salience_weight
                + time_score * time_weight
                + temporal_prior_score * temporal_prior_weight
                + preference_score * preference_weight
                + diagnostic_score * diagnostic_weight
                + person_name_score * person_name_weight
                + object_support_score * object_support_weight
                + local_context_score
                + grain_score * grain_weight
                + specificity_score * specificity_weight
                + structure_score * structure_weight
                + proxy_score
                + identity_reward
                + scope_bonus
                - wrong_entity_penalty
                - same_topic_different_entity_penalty
                - role_source_mismatch_penalty
                - wrong_domain_penalty
                - wrong_role_target_penalty
                - wrong_subtheme_penalty
                - generic_topic_penalty
            )
            enriched = dict(candidate)
            enriched["time_score"] = round(time_score, 4)
            enriched["temporal_prior_score"] = round(temporal_prior_score, 4)
            enriched["preference_score"] = round(preference_score, 4)
            enriched["diagnostic_score"] = round(diagnostic_score, 4)
            enriched["person_name_score"] = round(person_name_score, 4)
            enriched["identity_match_reward"] = round(identity_reward, 4)
            enriched["wrong_entity_penalty"] = round(wrong_entity_penalty, 4)
            enriched["same_topic_different_entity_penalty"] = round(same_topic_different_entity_penalty, 4)
            enriched["role_source_mismatch_penalty"] = round(role_source_mismatch_penalty, 4)
            enriched["wrong_domain_penalty"] = round(wrong_domain_penalty, 4)
            enriched["wrong_role_target_penalty"] = round(wrong_role_target_penalty, 4)
            enriched["wrong_subtheme_penalty"] = round(wrong_subtheme_penalty, 4)
            enriched["generic_topic_penalty"] = round(generic_topic_penalty, 4)
            enriched["identity_signature"] = str(candidate_identity_features.get("identity_signature") or "")
            enriched["salience_score"] = round(salience_score, 4)
            enriched["direct_anchor_score"] = round(direct_anchor_score, 4)
            enriched["object_support_raw"] = round(object_support_raw, 4)
            enriched["object_support_gate"] = round(object_support_gate, 4)
            enriched["object_support_score"] = round(object_support_score, 4)
            enriched["local_context_gate"] = round(local_context_gate, 4)
            enriched["local_context_score"] = round(local_context_score, 4)
            enriched["grain_score"] = round(grain_score, 4)
            enriched["specificity_score"] = round(specificity_score, 4)
            enriched["structure_gate"] = round(structure_gate, 4)
            enriched["structure_score"] = round(structure_score, 4)
            enriched["proxy_score"] = round(proxy_score, 4)
            enriched["scope_bonus"] = round(scope_bonus, 4)
            enriched["scope_match"] = scope_match_label
            enriched["candidate_timestamp_epoch"] = round(candidate_epoch, 2) if candidate_epoch > 0 else 0.0
            enriched["candidate_timestamp_source"] = timestamp_source
            enriched["score_weights"] = {
                "dense": round(dense_weight, 4),
                "fusion_rrf": round(rrf_weight, 4),
                "query_lexical": round(lexical_weight, 4),
                "salience": round(salience_weight, 4),
                "time": round(time_weight, 4),
                "temporal_prior": round(temporal_prior_weight, 4),
                "preference": round(preference_weight, 4),
                "diagnostic": round(diagnostic_weight, 4),
                "person_name": round(person_name_weight, 4),
                "object_support": round(object_support_weight, 4),
                "grain": round(grain_weight, 4),
                "specificity": round(specificity_weight, 4),
                "structure": round(structure_weight, 4),
            }
            enriched["evidence_score"] = round(final_score, 4)
            ranked.append(enriched)
        ranked.sort(key=lambda item: item["evidence_score"], reverse=True)
        return ranked

    def _apply_identity_awareness(
        self,
        *,
        query: str,
        profile: QueryProfile,
        candidates: list[dict[str, Any]],
        route_context: dict[str, Any] | None,
        route_tuning: BenchmarkRouteTuning,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not candidates or not self.config.enable_identity_aware_rerank or not route_tuning.prefer_identity_rerank:
            return candidates, {
                "enabled": False,
                "applied_candidates": 0,
                "wrong_entity_penalty_count": 0.0,
                "identity_reward_positive_count": 0.0,
            }
        focus_names = self._route_focus_names(route_context, profile)
        query_features = self._identity_features(
            text=query,
            metadata=route_context,
            focus_names=focus_names,
            prefix="query_identity_rerank",
        )
        pool_limit = min(len(candidates), max(route_tuning.confusing_cluster_topk, route_tuning.segment_rerank_topk, 18))
        reranked = [dict(item) for item in candidates]
        wrong_entity_penalty_count = 0.0
        identity_reward_positive_count = 0.0
        for index, candidate in enumerate(reranked[:pool_limit]):
            candidate_text, _, _ = self._candidate_text_features(candidate)
            candidate_features = self._identity_features(
                text=candidate_text,
                metadata=candidate,
                focus_names=focus_names,
                prefix=f"identity:{candidate.get('chunk_id') or candidate.get('node_id') or index}",
            )
            identity_components = score_identity_alignment(
                query_features,
                candidate_features,
                text_similarity=max(float(candidate.get("query_lexical") or 0.0), float(candidate.get("dense_score") or 0.0)),
            )
            reward = float(identity_components.get("identity_match_reward") or 0.0) * max(0.0, route_tuning.identity_reward_weight) * 0.45
            penalty = (
                float(identity_components.get("wrong_entity_penalty") or 0.0)
                + float(identity_components.get("same_topic_different_entity_penalty") or 0.0)
                + float(identity_components.get("role_source_mismatch_penalty") or 0.0)
            ) * max(0.0, route_tuning.wrong_entity_penalty_weight)
            penalty += float(identity_components.get("wrong_domain_penalty") or 0.0) * max(
                float(self.config.wrong_domain_penalty_weight or 0.0),
                float(route_tuning.wrong_domain_penalty_weight or 0.0),
            )
            penalty += float(identity_components.get("wrong_role_target_penalty") or 0.0) * max(
                float(self.config.wrong_role_target_penalty_weight or 0.0),
                float(route_tuning.wrong_role_target_penalty_weight or 0.0),
            )
            penalty += float(identity_components.get("wrong_subtheme_penalty") or 0.0) * max(
                float(self.config.wrong_subtheme_penalty_weight or 0.0),
                float(route_tuning.wrong_subtheme_penalty_weight or 0.0),
            )
            penalty += float(identity_components.get("generic_topic_penalty") or 0.0) * max(
                float(self.config.generic_topic_penalty_weight or 0.0),
                float(route_tuning.generic_topic_penalty_weight or 0.0),
            )
            candidate["identity_rerank_delta"] = round(reward - penalty, 4)
            candidate["evidence_score"] = round(float(candidate.get("evidence_score") or 0.0) + reward - penalty, 4)
            if penalty > 0.0001:
                wrong_entity_penalty_count += 1.0
            if reward > 0.0001:
                identity_reward_positive_count += 1.0
            reranked[index] = candidate
        reranked.sort(key=lambda item: float(item.get("evidence_score") or 0.0), reverse=True)
        return reranked, {
            "enabled": True,
            "applied_candidates": float(pool_limit),
            "wrong_entity_penalty_count": float(wrong_entity_penalty_count),
            "identity_reward_positive_count": float(identity_reward_positive_count),
        }

    def _apply_confusing_cluster_rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        route_context: dict[str, Any] | None,
        route_tuning: BenchmarkRouteTuning,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not candidates or not self.config.enable_confusing_cluster_rerank or not route_tuning.prefer_confusing_cluster:
            return candidates, {"enabled": False, "cluster_count": 0.0, "avg_cluster_size": 0.0, "cache_hit": 0.0}
        cluster_topk = min(len(candidates), max(4, route_tuning.confusing_cluster_topk or self.config.confusing_cluster_topk_default))
        focus_names = self._route_focus_names(route_context, None)
        cluster_cache_key = stable_content_hash(
            json.dumps(
                {
                    "query": normalize_text_for_hash(query).lower(),
                    "chunk_ids": [str(item.get("chunk_id") or item.get("node_id") or "") for item in candidates[:cluster_topk]],
                    "route_context": self._route_context_key(route_context),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        cache_hit = 0.0
        cached = self._confusing_cluster_cache.get(cluster_cache_key) if self.config.enable_confusing_cluster_cache else None
        if cached is not None:
            clusters = [list(group) for group in cached.get("clusters", [])]
            cache_hit = 1.0
        else:
            clusters: list[list[int]] = []
            identity_feature_rows = [
                self._identity_features(
                    text=self._candidate_novelty_text(candidate),
                    metadata=candidate,
                    focus_names=focus_names,
                    prefix=f"cluster:{candidate.get('chunk_id') or candidate.get('node_id') or index}",
                )
                for index, candidate in enumerate(candidates[:cluster_topk])
            ]
            for index, candidate in enumerate(candidates[:cluster_topk]):
                added = False
                candidate_text = self._candidate_novelty_text(candidate)
                candidate_features = identity_feature_rows[index]
                candidate_names = set(candidate_features.get("name_tokens") or [])
                candidate_terms = set(candidate_features.get("discriminative_tokens") or [])
                for group in clusters:
                    rep = candidates[group[0]]
                    rep_features = identity_feature_rows[group[0]]
                    overlap = max(
                        lexical_score(candidate_text[:500], self._candidate_novelty_text(rep)[:500]),
                        self._token_overlap_ratio(candidate_text, self._candidate_novelty_text(rep)),
                    )
                    shared_identity = bool(candidate_names & set(rep_features.get("name_tokens") or []))
                    shared_terms = len(candidate_terms & set(rep_features.get("discriminative_tokens") or [])) / max(
                        1,
                        min(len(candidate_terms) or 1, len(set(rep_features.get("discriminative_tokens") or [])) or 1),
                    )
                    if overlap >= 0.62 or (overlap >= 0.48 and (shared_identity or shared_terms >= 0.28)):
                        group.append(index)
                        added = True
                        break
                if not added:
                    clusters.append([index])
            if self.config.enable_confusing_cluster_cache:
                self._confusing_cluster_cache[cluster_cache_key] = {"clusters": clusters}

        reranked = [dict(item) for item in candidates]
        cluster_sizes = [len(group) for group in clusters if len(group) > 1]
        for group in clusters:
            if len(group) <= 1:
                continue
            best_index = max(
                group,
                key=lambda idx: (
                    float(reranked[idx].get("evidence_score") or 0.0),
                    float(reranked[idx].get("identity_match_reward") or 0.0),
                    float(reranked[idx].get("specificity_score") or 0.0) * max(0.0, float(self.config.cluster_specificity_weight or 0.0)),
                    float(reranked[idx].get("scope_bonus") or 0.0),
                    -float(reranked[idx].get("generic_topic_penalty") or 0.0),
                    -float(reranked[idx].get("wrong_entity_penalty") or 0.0),
                ),
            )
            representative_score = float(reranked[best_index].get("evidence_score") or 0.0)
            representative_identity = float(reranked[best_index].get("identity_match_reward") or 0.0)
            representative_temporal = (
                float(reranked[best_index].get("temporal_prior_score") or 0.0)
                + float(reranked[best_index].get("segment_temporal_score") or 0.0)
            )
            representative_bucket = str(reranked[best_index].get("time_bucket") or "")
            for idx in group:
                if idx == best_index:
                    reranked[idx]["confusing_cluster_size"] = len(group)
                    reranked[idx]["confusing_cluster_role"] = "representative"
                    continue
                base_penalty = max(0.02, route_tuning.confusing_neighbor_penalty_weight * 0.35) * max(1, len(group) - 1)
                candidate_score = float(reranked[idx].get("evidence_score") or 0.0)
                candidate_identity = float(reranked[idx].get("identity_match_reward") or 0.0)
                candidate_temporal = (
                    float(reranked[idx].get("temporal_prior_score") or 0.0)
                    + float(reranked[idx].get("segment_temporal_score") or 0.0)
                )
                same_time_bucket = bool(
                    representative_bucket and representative_bucket == str(reranked[idx].get("time_bucket") or "")
                )
                dominance_gap = max(0.0, representative_score - candidate_score)
                identity_gap = max(0.0, representative_identity - candidate_identity)
                temporal_gap = max(0.0, representative_temporal - candidate_temporal)
                penalty_scale = 1.0 + min(0.6, dominance_gap * 0.7)
                if same_time_bucket:
                    penalty_scale += 0.18
                if identity_gap > 0.02:
                    penalty_scale += min(0.18, identity_gap * 0.8)
                if temporal_gap > 0.04:
                    penalty_scale += min(0.22, temporal_gap * 0.7)
                penalty = min(0.28, base_penalty * penalty_scale)
                reranked[idx]["confusing_neighbor_penalty"] = round(penalty, 4)
                reranked[idx]["confusing_cluster_size"] = len(group)
                reranked[idx]["confusing_cluster_role"] = "neighbor"
                reranked[idx]["evidence_score"] = round(float(reranked[idx].get("evidence_score") or 0.0) - penalty, 4)
        reranked.sort(key=lambda item: float(item.get("evidence_score") or 0.0), reverse=True)
        return reranked, {
            "enabled": True,
            "cluster_count": float(len(cluster_sizes)),
            "avg_cluster_size": round(sum(cluster_sizes) / len(cluster_sizes), 4) if cluster_sizes else 0.0,
            "cache_hit": cache_hit,
        }

    def _apply_segment_rerank(
        self,
        *,
        query: str,
        profile: QueryProfile,
        candidates: list[dict[str, Any]],
        route_context: dict[str, Any] | None,
        route_tuning: BenchmarkRouteTuning,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not candidates or not self.config.enable_segment_rerank or not route_tuning.prefer_segment_rerank:
            return candidates, {"enabled": False, "applied_candidates": 0.0, "cache_hit": 0.0, "positive_segment_count": 0.0}
        focus_names = self._route_focus_names(route_context, profile)
        query_features = self._identity_features(
            text=query,
            metadata=route_context,
            focus_names=focus_names,
            prefix="query_segment_rerank",
        )
        pool_limit = min(len(candidates), max(1, route_tuning.segment_rerank_topk or self.config.segment_rerank_topk_default))
        reranked = [dict(item) for item in candidates]
        cache_hit = 0.0
        positive_segment_count = 0.0
        for index, candidate in enumerate(reranked[:pool_limit]):
            segment_cache_key = stable_content_hash(
                json.dumps(
                    {
                        "query": normalize_text_for_hash(query).lower(),
                        "chunk_id": str(candidate.get("chunk_id") or candidate.get("node_id") or ""),
                        "text_hash": stable_content_hash(str(candidate.get("text") or "")),
                        "focus_names": focus_names,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            cached = self._segment_feature_cache.get(segment_cache_key) if self.config.enable_segment_feature_cache else None
            if cached is not None:
                segment_result = dict(cached)
                cache_hit += 1.0
            else:
                segment_result = score_candidate_segments(
                    query=query,
                    profile=profile,
                    candidate=candidate,
                    query_features=query_features,
                    focus_names=focus_names,
                    span_lengths=self.config.segment_span_lengths if self.config.enable_three_sentence_spans else (1, 2),
                )
                if self.config.enable_segment_feature_cache:
                    self._segment_feature_cache[segment_cache_key] = dict(segment_result)
            delta = float(segment_result.get("segment_rerank_score") or 0.0) * max(0.0, route_tuning.segment_weight)
            candidate.update(segment_result)
            candidate["segment_rerank_delta"] = round(delta, 4)
            candidate["evidence_score"] = round(float(candidate.get("evidence_score") or 0.0) + delta, 4)
            if delta > 0.0001:
                positive_segment_count += 1.0
            reranked[index] = candidate
        reranked.sort(key=lambda item: float(item.get("evidence_score") or 0.0), reverse=True)
        return reranked, {
            "enabled": True,
            "applied_candidates": float(pool_limit),
            "cache_hit": float(cache_hit),
            "positive_segment_count": float(positive_segment_count),
        }

    def _select_core_evidence(
        self,
        ranked: list[dict[str, Any]],
        limit: int,
        profile: QueryProfile,
        route_context: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if limit <= 0:
            return [], {"selected_unique_nodes": 0.0, "overflow_total": 0.0, "diversity_overflow": 0.0}
        selected: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        selected_chunk_ids: set[str] = set()
        node_cap = 2 if limit <= 10 else max(2, limit // 6)
        per_node_counts: dict[str, int] = {}
        per_node_macro_counts: dict[str, int] = {}
        selection_stats = {
            "node_cap_overflow": 0.0,
            "macro_overflow": 0.0,
            "diversity_overflow": 0.0,
            "overflow_total": 0.0,
            "selected_unique_nodes": 0.0,
            "focused_reserve_considered": 0.0,
            "focused_reserved": 0.0,
        }

        def finalize_selection() -> tuple[list[dict[str, Any]], dict[str, float]]:
            if selection_stats["focused_reserved"] > 0.0:
                selected.sort(key=lambda item: float(item.get("evidence_score") or 0.0), reverse=True)
            selection_stats["selected_unique_nodes"] = float(
                len({str(candidate.get("node_id") or "") for candidate in selected if candidate.get("node_id")})
            )
            selection_stats["overflow_total"] = float(sum(value for key, value in selection_stats.items() if key.endswith("overflow")))
            return selected[:limit], selection_stats

        def try_select(
            item: dict[str, Any],
            *,
            record_overflow: bool,
            ignore_diversity: bool = False,
            focused_reserve: bool = False,
        ) -> bool:
            chunk_id = str(item.get("chunk_id") or "")
            if chunk_id and chunk_id in selected_chunk_ids:
                return False
            node_id = str(item.get("node_id") or "")
            grain = str(item.get("grain") or "micro")
            if node_id:
                if per_node_counts.get(node_id, 0) >= node_cap:
                    if record_overflow:
                        overflow.append(item)
                        selection_stats["node_cap_overflow"] += 1.0
                    return False
                if grain == "macro" and per_node_macro_counts.get(node_id, 0) >= 1:
                    if record_overflow:
                        overflow.append(item)
                        selection_stats["macro_overflow"] += 1.0
                    return False
                if grain == "macro" and profile.needs_exact_evidence and per_node_counts.get(node_id, 0) > 0:
                    if record_overflow:
                        overflow.append(item)
                        selection_stats["macro_overflow"] += 1.0
                    return False
            redundancy_score = self._redundancy_score(item, selected)
            exact_sensitive = self._is_exact_sensitive_candidate(item, profile)
            diversity_threshold = 0.92 if exact_sensitive else 0.82
            same_node_threshold = 0.76 if exact_sensitive else 0.68
            if not ignore_diversity:
                if selected and redundancy_score >= diversity_threshold:
                    if record_overflow:
                        overflow.append(item)
                        selection_stats["diversity_overflow"] += 1.0
                    return False
                if not exact_sensitive and node_id and per_node_counts.get(node_id, 0) >= 1 and redundancy_score >= same_node_threshold:
                    if record_overflow:
                        overflow.append(item)
                        selection_stats["diversity_overflow"] += 1.0
                    return False
            chosen = dict(item) if focused_reserve else item
            if focused_reserve:
                chosen["focused_reserve"] = True
                selection_stats["focused_reserved"] += 1.0
            selected.append(chosen)
            if chunk_id:
                selected_chunk_ids.add(chunk_id)
            if node_id:
                per_node_counts[node_id] = per_node_counts.get(node_id, 0) + 1
                if grain == "macro":
                    per_node_macro_counts[node_id] = per_node_macro_counts.get(node_id, 0) + 1
            return True

        if route_context:
            reserve_limit = min(2, limit)
            reserve_scan_limit = min(len(ranked), max(limit * 3, 18))
            for item in ranked[:reserve_scan_limit]:
                if not self._is_longmemeval_focused_reserve_candidate(item, route_context):
                    continue
                selection_stats["focused_reserve_considered"] += 1.0
                try_select(item, record_overflow=False, ignore_diversity=True, focused_reserve=True)
                if len(selected) >= limit:
                    return finalize_selection()
                if selection_stats["focused_reserved"] >= reserve_limit:
                    break

        for item in ranked:
            try_select(item, record_overflow=True)
            if len(selected) >= limit:
                return finalize_selection()
        for item in overflow:
            chunk_id = str(item.get("chunk_id") or "")
            if chunk_id and chunk_id in selected_chunk_ids:
                continue
            selected.append(item)
            if chunk_id:
                selected_chunk_ids.add(chunk_id)
            if len(selected) >= limit:
                break
        return finalize_selection()

    def _best_non_macro_support_by_node(self, candidates: list[dict[str, Any]]) -> dict[str, float]:
        best: dict[str, float] = {}
        for candidate in candidates:
            if candidate.get("grain") == "macro":
                continue
            node_id = str(candidate.get("node_id") or "")
            if not node_id:
                continue
            support = (
                candidate.get("dense_score", 0.0) * 0.55
                + candidate.get("query_lexical", 0.0) * 0.35
                + min(candidate.get("object_support", 0.0), 0.1)
                + min(candidate.get("neighbor_count", 0) * 0.02, 0.04)
            )
            best[node_id] = max(best.get(node_id, 0.0), float(support))
        return best

    def _candidate_text_features(self, candidate: dict[str, Any]) -> tuple[str, str, set[str]]:
        text = " ".join(
            filter(
                None,
                [
                    str(candidate.get("text") or ""),
                    str(candidate.get("summary") or ""),
                    str(candidate.get("cell") or ""),
                    str(candidate.get("zone") or ""),
                ],
            )
        )
        lowered = text.lower()
        return text, lowered, set(token_tuple(lowered))

    @staticmethod
    def _candidate_novelty_text(candidate: dict[str, Any]) -> str:
        return " ".join(
            filter(
                None,
                [
                    str(candidate.get("text") or ""),
                    str(candidate.get("summary") or ""),
                    str(candidate.get("content_ref") or ""),
                ],
            )
        )

    @staticmethod
    def _token_overlap_ratio(text_a: str, text_b: str) -> float:
        tokens_a = set(token_tuple(text_a.lower()))
        tokens_b = set(token_tuple(text_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))

    def _redundancy_score(self, candidate: dict[str, Any], selected: list[dict[str, Any]]) -> float:
        candidate_text = self._candidate_novelty_text(candidate)
        node_id = str(candidate.get("node_id") or "")
        content_ref = str(candidate.get("content_ref") or "")
        best_overlap = 0.0
        for chosen in selected:
            chosen_text = self._candidate_novelty_text(chosen)
            overlap = max(
                lexical_score(candidate_text[:600], chosen_text[:600]),
                self._token_overlap_ratio(candidate_text, chosen_text),
            )
            if node_id and node_id == str(chosen.get("node_id") or ""):
                overlap += 0.08
            if content_ref and content_ref == str(chosen.get("content_ref") or ""):
                overlap += 0.05
            best_overlap = max(best_overlap, overlap)
        return min(1.0, best_overlap)

    @staticmethod
    def _is_exact_sensitive_candidate(candidate: dict[str, Any], profile: QueryProfile) -> bool:
        if not profile.needs_exact_evidence:
            return False
        return (
            float(candidate.get("query_lexical") or 0.0) >= 0.18
            or float(candidate.get("salience_score") or 0.0) >= 0.12
            or float(candidate.get("person_name_score") or 0.0) >= 0.08
            or float(candidate.get("time_score") or 0.0) >= 0.12
            or float(candidate.get("preference_score") or 0.0) >= 0.12
        )

    @staticmethod
    def _salient_query_terms(query_tokens: set[str]) -> set[str]:
        return {
            token for token in query_tokens
            if token not in COMMON_QUERY_STOPWORDS
            and token not in TEMPORAL_QUERY_STOPWORDS
            and token not in PREFERENCE_QUERY_STOPWORDS
            and len(token) > 2
        }

    @staticmethod
    def _salience_score(salient_terms: set[str], candidate_tokens: set[str]) -> float:
        if not salient_terms:
            return 0.0
        return min(0.24, len(salient_terms & candidate_tokens) / max(1, len(salient_terms)) * 0.24)

    def _specificity_score(
        self,
        candidate: dict[str, Any],
        profile: QueryProfile,
        best_same_node_non_macro: float,
    ) -> float:
        grain = candidate.get("grain", "micro")
        score = 0.0
        if profile.needs_exact_evidence:
            if grain == "micro":
                score += 0.06
            elif grain == "local":
                score += 0.025
            elif grain == "macro":
                score -= 0.12
        if (profile.needs_preference_objects or profile.needs_temporal_objects) and grain == "macro":
            score -= 0.07
        elif grain == "macro":
            score += 0.01

        if grain == "macro" and best_same_node_non_macro > 0.14:
            score -= min(0.18, 0.04 + best_same_node_non_macro * 0.28)
        return score

    def _grain_score(self, grain: str, profile: QueryProfile) -> float:
        if grain in profile.granularity_bias:
            return {"micro": 0.08, "local": 0.06, "macro": 0.03}.get(grain, 0.0)
        return {"micro": 0.03, "local": 0.02, "macro": -0.01}.get(grain, 0.0)

    def _extract_candidate_timestamp(self, candidate: dict[str, Any]) -> tuple[float, str]:
        for key in (
            "timestamp",
            "event_time",
            "session_time",
            "source_time",
            "created_at",
            "updated_at",
            "node_created_at",
            "node_timestamp",
            "last_accessed_at",
        ):
            epoch = self._timestamp_to_epoch(candidate.get(key))
            if epoch > 0:
                return epoch, key
        metadata = candidate.get("metadata")
        if isinstance(metadata, dict):
            for key in (
                "timestamp",
                "event_time",
                "session_time",
                "source_time",
                "created_at",
            ):
                epoch = self._timestamp_to_epoch(metadata.get(key))
                if epoch > 0:
                    return epoch, f"metadata.{key}"
        return 0.0, ""

    def _build_temporal_prior_state(self, profile: QueryProfile, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        epochs: list[float] = []
        for candidate in candidates:
            epoch, _ = self._extract_candidate_timestamp(candidate)
            if epoch > 0:
                epochs.append(epoch)
        explicit_query_epochs = [self._timestamp_to_epoch(term) for term in profile.temporal_reference_terms]
        explicit_query_epochs = [epoch for epoch in explicit_query_epochs if epoch > 0]
        explicit_range = (min(explicit_query_epochs), max(explicit_query_epochs)) if len(explicit_query_epochs) >= 2 else None
        return {
            "sorted_epochs": sorted(set(epochs)),
            "query_epochs": explicit_query_epochs,
            "query_range": explicit_range,
        }

    def _temporal_prior_score(
        self,
        profile: QueryProfile,
        candidate: dict[str, Any],
        temporal_state: dict[str, Any],
    ) -> tuple[float, float, str]:
        candidate_epoch, timestamp_source = self._extract_candidate_timestamp(candidate)
        sorted_epochs = list(temporal_state.get("sorted_epochs") or [])
        if candidate_epoch <= 0 or not sorted_epochs:
            return 0.0, candidate_epoch, timestamp_source
        min_epoch = sorted_epochs[0]
        max_epoch = sorted_epochs[-1]
        span = max(max_epoch - min_epoch, 86400.0)
        latest_epoch = sorted_epochs[-1]
        previous_epoch = sorted_epochs[-2] if len(sorted_epochs) > 1 else None
        score = 0.0

        if profile.temporal_mode == "latest":
            target_window = max(span * 0.18, 86400.0)
            proximity = 1.0 - min(1.0, abs(candidate_epoch - latest_epoch) / target_window)
            relative = (candidate_epoch - min_epoch) / span
            score = proximity * 0.18 + max(-0.05, relative * 0.05 - 0.01)
            if candidate_epoch < latest_epoch - target_window:
                score -= min(0.07, ((latest_epoch - candidate_epoch) / span) * 0.07)
        elif profile.temporal_mode == "previous":
            if previous_epoch is None:
                return 0.0, candidate_epoch, timestamp_source
            target_window = max(span * 0.18, 86400.0)
            proximity = 1.0 - min(1.0, abs(candidate_epoch - previous_epoch) / target_window)
            score = proximity * 0.18
            if abs(candidate_epoch - latest_epoch) < target_window * 0.6:
                score -= 0.08
            elif candidate_epoch < previous_epoch - target_window:
                score -= min(0.06, ((previous_epoch - candidate_epoch) / span) * 0.06)
        elif profile.temporal_mode == "range":
            query_range = temporal_state.get("query_range")
            if query_range is None:
                return 0.0, candidate_epoch, timestamp_source
            range_start, range_end = query_range
            if range_start <= candidate_epoch <= range_end:
                score += 0.16
            else:
                gap = (range_start - candidate_epoch) if candidate_epoch < range_start else (candidate_epoch - range_end)
                score += max(-0.03, 0.1 - min(0.13, (gap / span) * 0.13))
        elif profile.temporal_mode == "point":
            query_epochs = list(temporal_state.get("query_epochs") or [])
            if query_epochs:
                closest_gap = min(abs(candidate_epoch - query_epoch) for query_epoch in query_epochs)
                target_window = max(span * 0.12, 43200.0)
                proximity = 1.0 - min(1.0, closest_gap / target_window)
                score += proximity * 0.14
                if closest_gap > target_window:
                    score -= min(0.04, (closest_gap / span) * 0.04)

        return round(score, 4), candidate_epoch, timestamp_source

    def _time_score(
        self,
        profile: QueryProfile,
        query_tokens: set[str],
        candidate_lower: str,
        candidate_tokens: set[str],
    ) -> float:
        if not profile.needs_temporal_objects:
            return 0.0
        temporal_query_tokens = (query_tokens & TEMPORAL_TERMS) or query_tokens
        overlap = len(temporal_query_tokens & candidate_tokens) / max(1, len(temporal_query_tokens))
        temporal_hits = sum(1 for term in TEMPORAL_TERMS if term in candidate_lower)
        reference_hits = sum(1 for term in profile.temporal_reference_terms if term in candidate_lower)
        latest_hits = sum(1 for term in LATEST_STATE_MARKERS if term in candidate_lower)
        previous_hits = sum(1 for term in PREVIOUS_STATE_MARKERS if term in candidate_lower)
        range_hits = sum(1 for term in ("between", "from", "to", "ago", "since", "until", "before", "after") if term in candidate_lower)
        change_hits = sum(1 for term in ("switched from", "switched to", "changed to", "changed from", "updated to", "no longer", "used to", "stopped") if term in candidate_lower)
        mode_bonus = 0.0
        if profile.temporal_mode == "latest":
            mode_bonus += min(0.12, latest_hits * 0.04 + change_hits * 0.015)
            if previous_hits:
                mode_bonus -= min(0.06, previous_hits * 0.03)
        elif profile.temporal_mode == "previous":
            mode_bonus += min(0.12, previous_hits * 0.04 + change_hits * 0.02)
            if latest_hits:
                mode_bonus -= min(0.05, latest_hits * 0.025)
        elif profile.temporal_mode == "range":
            mode_bonus += min(0.1, range_hits * 0.03 + reference_hits * 0.03)
        return min(0.46, overlap * 0.32 + temporal_hits * 0.03 + reference_hits * 0.1 + change_hits * 0.015 + mode_bonus)

    _PREF_EXPRESSION_RE = re.compile(
        r"\b(?:prefer|prefers|preferred|favorite|favourite|like|likes|love|loves|enjoy|enjoys"
        r"|dislike|dislikes|hate|hates|avoid|avoids|don't like|do not like)\s+",
        re.IGNORECASE,
    )

    @classmethod
    def _infer_preference_polarity(cls, candidate_lower: str) -> float | None:
        positive = bool(re.search(r"\b(?:prefer|prefers|preferred|favorite|favourite|like|likes|love|loves|enjoy|enjoys)\b", candidate_lower))
        negative = bool(re.search(r"\b(?:dislike|dislikes|hate|hates|avoid|avoids|don't like|do not like|no longer like|never)\b", candidate_lower))
        if positive and not negative:
            return 1.0
        if negative and not positive:
            return -1.0
        return None

    def _preference_score(
        self,
        profile: QueryProfile,
        query_tokens: set[str],
        candidate: dict[str, Any],
        candidate_lower: str,
        candidate_tokens: set[str],
    ) -> float:
        if not profile.needs_preference_objects and not (query_tokens & PREFERENCE_TERMS):
            return 0.0
        topic_tokens = query_tokens - PREFERENCE_TERMS - PREFERENCE_QUERY_STOPWORDS
        topic_overlap = len(topic_tokens & candidate_tokens) / max(1, len(topic_tokens)) if topic_tokens else 0.0
        expression_bonus = 0.08 if self._PREF_EXPRESSION_RE.search(candidate_lower) else 0.0
        polarity_bonus = 0.0
        candidate_polarity = self._infer_preference_polarity(candidate_lower)
        if profile.preference_polarity_hint is not None and candidate_polarity is not None:
            polarity_bonus += 0.12 if profile.preference_polarity_hint * candidate_polarity > 0 else -0.14
        elif candidate_polarity is not None:
            polarity_bonus += 0.02
        transition_bonus = 0.0
        if any(marker in candidate_lower for marker in ("used to", "no longer", "stopped", "switched from", "switching from")):
            transition_bonus += 0.05 if profile.needs_preference_objects else 0.03
            if profile.preference_polarity_hint is None and any(marker in candidate_lower for marker in ("used to", "no longer", "stopped")):
                polarity_bonus -= 0.03
        obj_support = min(candidate.get("object_support", 0.0), 0.12)
        return min(0.46, topic_overlap * 0.24 + expression_bonus + transition_bonus + obj_support * 0.6 + polarity_bonus)

    def _diagnostic_score(self, query: str, query_tokens: set[str], candidate_lower: str, candidate_tokens: set[str]) -> float:
        if not (query_tokens & DIAGNOSTIC_TERMS):
            return 0.0
        overlap = len((query_tokens & DIAGNOSTIC_TERMS) & candidate_tokens)
        if overlap == 0:
            return 0.0
        return min(0.3, 0.1 * overlap + lexical_score(query, candidate_lower) * 0.12)

    @staticmethod
    def _extract_person_names(query: str) -> list[str]:
        """Extract likely person names from a query string."""
        candidates = _PERSON_NAME_RE.findall(query)
        return [name for name in candidates if name not in _COMMON_NON_NAMES]

    @staticmethod
    def _person_name_score(query_names: list[str], candidate_text: str) -> float:
        if not query_names:
            return 0.0
        matched = sum(1 for name in query_names if name in candidate_text)
        if matched == 0:
            return 0.0
        return min(0.20, 0.10 * matched)

    def _select_evidence_objects(
        self,
        query: str,
        profile: QueryProfile,
        ranked_evidence: list[dict[str, Any]],
        dense_objects: list[dict[str, Any]],
        sparse_objects: list[dict[str, Any]],
        limit: int,
        route_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        evidence_node_ids = {item["node_id"] for item in ranked_evidence}
        evidence_chunk_ids = {item["chunk_id"] for item in ranked_evidence}
        preferred_types = set(profile.preferred_object_types)
        workspace_context = self._workspace_context(route_context)
        hit_scores: dict[str, dict[str, float]] = {}

        def add_object_hit(hit: dict[str, Any], source_name: str, rank: int) -> None:
            object_id = str(hit.get("object_id") or "")
            hit_similarity = float(hit.get("similarity") or 0.0)
            if not object_id:
                return
            current = hit_scores.get(object_id)
            if current is None:
                current = {"object_score": 0.0, "dense_score": 0.0, "lexical_score": 0.0}
                hit_scores[object_id] = current
            current["object_score"] += 1.0 / (70 + rank)
            if source_name == "dense":
                current["dense_score"] = max(current["dense_score"], hit_similarity)
            else:
                current["lexical_score"] = max(current["lexical_score"], hit_similarity)

        for rank, hit in enumerate(dense_objects, start=1):
            add_object_hit(hit, "dense", rank)
        for rank, hit in enumerate(sparse_objects, start=1):
            add_object_hit(hit, "sparse", rank)

        object_rows = self.storage.fetch_objects_by_ids(list(hit_scores.keys()))
        row_map = {str(row["object_id"]): row for row in object_rows}
        latest_preference_by_entity = self._latest_preference_by_entity(object_rows)
        temporal_history_by_entity = self._temporal_history_by_entity(object_rows)
        ranked: list[dict[str, Any]] = []
        for object_id, scores in hit_scores.items():
            base = row_map.get(object_id)
            if not base:
                continue
            candidate = dict(base)
            if not self._candidate_in_scope_context(
                {
                    "scope": candidate.get("scope"),
                    "workspace": candidate.get("workspace"),
                    "project": candidate.get("project"),
                    "session_id": candidate.get("session_id"),
                },
                route_context,
            ):
                continue
            if candidate.get("source_node_id") not in evidence_node_ids and candidate.get("source_chunk_id") not in evidence_chunk_ids:
                if candidate.get("object_type") not in preferred_types:
                    continue
            type_bonus = 0.0
            if candidate.get("object_type") in preferred_types:
                type_bonus += 0.18
            preference_bonus = self._preference_object_score(
                query=query,
                profile=profile,
                candidate=candidate,
                latest_preference_by_entity=latest_preference_by_entity,
            )
            temporal_bonus = self._temporal_object_score(
                query=query,
                profile=profile,
                candidate=candidate,
                temporal_history_by_entity=temporal_history_by_entity,
            )
            personal_context_bonus = self._personal_context_object_score(
                query=query,
                profile=profile,
                candidate=candidate,
            )
            relation_bonus = self._relation_object_score(
                query=query,
                profile=profile,
                candidate=candidate,
            )
            artifact_bonus = self._artifact_object_score(
                query=query,
                profile=profile,
                candidate=candidate,
            )
            open_loop_bonus = self._open_loop_object_score(
                query=query,
                profile=profile,
                candidate=candidate,
            )
            scope_bonus, scope_match = workspace_context.candidate_scope_bonus(
                scope=str(candidate.get("scope") or ""),
                project=str(candidate.get("project") or ""),
                session_id=str(candidate.get("session_id") or ""),
                workspace=str(candidate.get("workspace") or ""),
                weight=self.config.scope_priority_weight if self.config.enable_scope_priority else 0.0,
            )
            final_score = (
                scores["object_score"] * 15.0
                + scores["dense_score"] * 0.45
                + scores["lexical_score"] * 0.35
                + type_bonus
                + preference_bonus
                + temporal_bonus
                + personal_context_bonus
                + relation_bonus
                + artifact_bonus
                + open_loop_bonus
                + scope_bonus
            )
            enriched = dict(candidate)
            enriched["dense_score"] = round(scores["dense_score"], 4)
            enriched["lexical_score"] = round(scores["lexical_score"], 4)
            enriched["preference_object_score"] = round(preference_bonus, 4)
            enriched["temporal_object_score"] = round(temporal_bonus, 4)
            enriched["personal_context_score"] = round(personal_context_bonus, 4)
            enriched["relation_object_score"] = round(relation_bonus, 4)
            enriched["artifact_object_score"] = round(artifact_bonus, 4)
            enriched["open_loop_object_score"] = round(open_loop_bonus, 4)
            enriched["scope_bonus"] = round(scope_bonus, 4)
            enriched["scope_match"] = scope_match
            enriched["object_score"] = round(final_score, 4)
            ranked.append(enriched)

        ranked.sort(key=lambda item: item["object_score"], reverse=True)
        return ranked[:limit]

    def _complete_temporal_objects(
        self,
        query: str,
        profile: QueryProfile,
        ranked_evidence: list[dict[str, Any]],
        evidence_objects: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        node_ids = [str(item.get("node_id") or "") for item in ranked_evidence if item.get("node_id")]
        candidate_rows = self._fetch_objects_for_nodes(node_ids, ["state_update", "temporal_reference"])
        seen_ids = {str(obj.get("object_id") or "") for obj in candidate_rows}
        for obj in evidence_objects:
            if obj.get("object_type") not in {"state_update", "temporal_reference"}:
                continue
            object_id = str(obj.get("object_id") or "")
            if object_id and object_id not in seen_ids:
                candidate_rows.append(obj)
                seen_ids.add(object_id)

        temporal_history_by_entity = self._temporal_history_by_entity(candidate_rows)
        ranked: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            score = self._temporal_object_score(
                query=query,
                profile=profile,
                candidate=candidate,
                temporal_history_by_entity=temporal_history_by_entity,
            )
            if score <= 0.01:
                continue
            enriched = dict(candidate)
            enriched["object_score"] = round(max(float(candidate.get("object_score") or 0.0), score), 4)
            enriched["temporal_object_score"] = round(score, 4)
            ranked.append(enriched)

        ranked.sort(key=lambda item: item.get("object_score", 0.0), reverse=True)
        if profile.temporal_mode == "range" or profile.needs_multi_hop_evidence:
            return self._select_diverse_objects(ranked, limit=limit)
        return ranked[:limit]

    def _complete_personal_context_objects(
        self,
        query: str,
        profile: QueryProfile,
        ranked_evidence: list[dict[str, Any]],
        evidence_objects: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        node_ids = [str(item.get("node_id") or "") for item in ranked_evidence if item.get("node_id")]
        candidate_rows = self._fetch_objects_for_nodes(node_ids, ["preference", "personal_context", "relation"])
        seen_ids = {str(obj.get("object_id") or "") for obj in candidate_rows}
        for obj in evidence_objects:
            if obj.get("object_type") not in {"preference", "personal_context", "relation"}:
                continue
            object_id = str(obj.get("object_id") or "")
            if object_id and object_id not in seen_ids:
                candidate_rows.append(obj)
                seen_ids.add(object_id)

        latest_preference_by_entity = self._latest_preference_by_entity(candidate_rows)
        ranked: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            score = 0.0
            if candidate.get("object_type") == "preference":
                score = self._preference_object_score(
                    query=query,
                    profile=profile,
                    candidate=candidate,
                    latest_preference_by_entity=latest_preference_by_entity,
                )
            elif candidate.get("object_type") == "personal_context":
                score = self._personal_context_object_score(
                    query=query,
                    profile=profile,
                    candidate=candidate,
                )
            elif candidate.get("object_type") == "relation":
                score = self._relation_object_score(
                    query=query,
                    profile=profile,
                    candidate=candidate,
                )
            if score <= 0.01:
                continue
            enriched = dict(candidate)
            enriched["object_score"] = round(max(float(candidate.get("object_score") or 0.0), score), 4)
            ranked.append(enriched)

        ranked.sort(key=lambda item: item.get("object_score", 0.0), reverse=True)
        return ranked[:limit]

    def _expand_object_supporting_context(
        self,
        evidence_objects: list[dict[str, Any]],
        ranked_evidence: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        evidence_chunk_ids = {str(item.get("chunk_id") or "") for item in ranked_evidence}
        chunk_ids: list[str] = []
        for obj in evidence_objects:
            chunk_id = str(obj.get("source_chunk_id") or "")
            if not chunk_id or chunk_id in evidence_chunk_ids or chunk_id in chunk_ids:
                continue
            chunk_ids.append(chunk_id)
            if len(chunk_ids) >= limit:
                break
        started = perf_counter()
        chunks = self.storage.fetch_chunks_with_node_metadata_by_ids(chunk_ids)
        self._object_support_join_ms += (perf_counter() - started) * 1000.0
        return self._pack_chunks_with_metadata(chunks, already_hydrated=True)

    def _merge_object_lists(
        self,
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in primary + secondary:
            object_id = str(item.get("object_id") or "")
            if not object_id or object_id in seen:
                continue
            merged.append(item)
            seen.add(object_id)
            if limit > 0 and len(merged) >= limit:
                break
        return merged

    def _merge_chunk_contexts(self, chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen:
                continue
            merged.append(chunk)
            seen.add(chunk_id)
            if limit > 0 and len(merged) >= limit:
                break
        return merged

    def _fetch_objects_for_nodes(self, node_ids: list[str], object_types: list[str]) -> list[dict[str, Any]]:
        filtered_node_ids = list(dict.fromkeys(node_id for node_id in node_ids if node_id))
        if not filtered_node_ids or not object_types:
            return []
        started = perf_counter()
        rows = self.storage.fetch_objects_for_nodes(filtered_node_ids, object_types=object_types)
        self._object_lookup_ms += (perf_counter() - started) * 1000.0
        return rows

    def _select_diverse_objects(self, ranked: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        reserve: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()
        for item in ranked:
            node_id = str(item.get("source_node_id") or "")
            if node_id and node_id not in seen_nodes:
                selected.append(item)
                seen_nodes.add(node_id)
            else:
                reserve.append(item)
            if len(selected) >= limit:
                return selected[:limit]
        for item in reserve:
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected[:limit]

    def _expand_supporting_context(self, ranked_evidence: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        links = self.storage.fetch_neighbor_links([item["chunk_id"] for item in ranked_evidence], limit=max(limit * 4, limit))
        if not links:
            return []
        neighbor_ids: list[str] = []
        evidence_ids = {item["chunk_id"] for item in ranked_evidence}
        for link in links:
            neighbor_id = str(link["neighbor_chunk_id"])
            if neighbor_id in evidence_ids or neighbor_id in neighbor_ids:
                continue
            neighbor_ids.append(neighbor_id)
            if len(neighbor_ids) >= limit:
                break
        chunks = self.storage.fetch_chunks_with_node_metadata_by_ids(neighbor_ids)
        return self._pack_chunks_with_metadata(chunks, already_hydrated=True)

    def _pack_chunks_with_metadata(self, chunks: list[dict[str, Any]], already_hydrated: bool = False) -> list[dict[str, Any]]:
        packed: list[dict[str, Any]] = []
        meta_map = {str(chunk["chunk_id"]): chunk for chunk in chunks} if already_hydrated else {
            str(row["chunk_id"]): row
            for row in self.storage.hydrate_chunks_with_node_metadata([str(chunk["chunk_id"]) for chunk in chunks])
        }
        for chunk in chunks:
            meta = meta_map.get(str(chunk["chunk_id"]), {})
            packed.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "node_id": chunk["node_id"],
                    "text": chunk["text"],
                    "summary": chunk.get("summary") or chunk.get("retrieval_summary") or chunk.get("structured_summary"),
                    "retrieval_summary": chunk.get("retrieval_summary"),
                    "structured_summary": chunk.get("structured_summary"),
                    "retrieval_signature": chunk.get("retrieval_signature"),
                    "time_bucket": chunk.get("time_bucket"),
                    "entity_tags": chunk.get("entity_tags"),
                    "task_type_tag": chunk.get("task_type_tag"),
                    "shell": meta.get("shell"),
                    "sector": meta.get("sector"),
                    "zone": meta.get("zone"),
                    "cell": meta.get("cell"),
                    "grain": chunk.get("grain", "micro"),
                    "chunk_index": chunk.get("chunk_index"),
                    "created_at": chunk.get("created_at"),
                }
            )
        return packed

    def _expand_cognitive(
        self,
        query: str,
        profile: QueryProfile,
        evidence_nodes: list[dict[str, Any]],
        limit: int,
        include_refraction: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not evidence_nodes:
            return [], []
        seed_query = query + "\n" + "\n".join((node.get("summary") or "") for node in evidence_nodes[:4])
        if profile.needs_cognitive_expansion == "high":
            temperature = 0.7
            experience_limit = max(4, limit)
            creative_limit = max(3, limit // 2)
        elif profile.needs_cognitive_expansion == "medium":
            temperature = 0.4
            experience_limit = max(3, limit)
            creative_limit = max(2, limit // 2)
        else:
            temperature = 0.15
            experience_limit = min(2, limit)
            creative_limit = 0

        relevant = self.activation.reflection_activation(seed_query, evidence_nodes, temperature, limit=experience_limit)
        creative = []
        if include_refraction and creative_limit > 0:
            creative = self.activation.refraction_activation(seed_query, evidence_nodes, temperature, limit=creative_limit)
        return relevant, creative

    def _bm25_similarity(self, bm25_score: float) -> float:
        return 1.0 / (1.0 + abs(float(bm25_score or 0.0)))

    def _latest_preference_by_entity(self, objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for obj in objects:
            if obj.get("object_type") != "preference":
                continue
            entity = self._normalize_preference_entity(obj)
            if not entity:
                continue
            current = latest.get(entity)
            if current is None or self._preference_sort_key(obj) > self._preference_sort_key(current):
                latest[entity] = obj
        return latest

    def _temporal_history_by_entity(self, objects: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for obj in objects:
            if obj.get("object_type") not in {"state_update", "temporal_reference"}:
                continue
            entity = self._temporal_entity_key(obj)
            if not entity:
                continue
            grouped.setdefault(entity, []).append(obj)
        for entity, rows in grouped.items():
            rows.sort(key=self._temporal_sort_key, reverse=True)
            grouped[entity] = rows
        return grouped

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _preference_object_score(
        self,
        query: str,
        profile: QueryProfile,
        candidate: dict[str, Any],
        latest_preference_by_entity: dict[str, dict[str, Any]],
    ) -> float:
        if candidate.get("object_type") != "preference":
            return 0.0
        score = 0.0
        entity = self._normalize_preference_entity(candidate)
        entity_text = " ".join(filter(None, [entity, str(candidate.get("object_text") or "")]))
        query_terms = {
            token
            for token in tokenize(query)
            if token not in PREFERENCE_QUERY_STOPWORDS and token not in PREFERENCE_TERMS
        }
        entity_terms = set(tokenize(entity_text))
        if query_terms and entity_terms:
            score += (len(query_terms & entity_terms) / max(1, len(query_terms))) * 0.18

        polarity = self._coerce_optional_float(candidate.get("polarity"))
        if profile.preference_polarity_hint is not None and polarity is not None:
            if float(profile.preference_polarity_hint) * polarity > 0:
                score += 0.12
            else:
                score -= 0.08
        elif polarity is not None:
            score += 0.02

        latest = latest_preference_by_entity.get(entity or "")
        latest_polarity = self._coerce_optional_float(latest.get("polarity")) if latest else None
        if latest and str(latest.get("object_id")) == str(candidate.get("object_id")):
            score += 0.08
        elif latest_polarity is not None and polarity is not None and latest_polarity * polarity < 0:
            score -= 0.1

        score += min(max(float(candidate.get("confidence") or 0.0), 0.0), 1.0) * 0.06
        return score

    def _temporal_object_score(
        self,
        query: str,
        profile: QueryProfile,
        candidate: dict[str, Any],
        temporal_history_by_entity: dict[str, list[dict[str, Any]]],
    ) -> float:
        if candidate.get("object_type") not in {"state_update", "temporal_reference"}:
            return 0.0
        score = 0.0
        text = " ".join(
            filter(
                None,
                [
                    str(candidate.get("object_text") or ""),
                    str(candidate.get("entity") or ""),
                    str(candidate.get("attribute") or ""),
                    str(candidate.get("old_value") or ""),
                    str(candidate.get("new_value") or ""),
                    str(candidate.get("event_text") or ""),
                ],
            )
        ).lower()
        query_terms = {
            token
            for token in tokenize(query)
            if token not in TEMPORAL_QUERY_STOPWORDS
        }
        candidate_terms = set(tokenize(text))
        if query_terms and candidate_terms:
            score += (len(query_terms & candidate_terms) / max(1, len(query_terms))) * 0.16
        if profile.temporal_reference_terms:
            matched_refs = sum(1 for term in profile.temporal_reference_terms if term in text)
            score += min(0.22, matched_refs * 0.1)
        if candidate.get("object_type") == "state_update":
            score += 0.05
        if candidate.get("object_type") == "temporal_reference":
            score += 0.07
        marker = str(candidate.get("temporal_marker") or "").lower()
        if profile.temporal_mode == "latest" and marker == "latest":
            score += 0.08
        if profile.temporal_mode == "previous" and marker == "previous":
            score += 0.08

        history = temporal_history_by_entity.get(self._temporal_entity_key(candidate), [])
        if history:
            if profile.temporal_mode == "latest" and str(history[0].get("object_id")) == str(candidate.get("object_id")):
                score += 0.14
            elif profile.temporal_mode == "latest":
                score -= 0.08
            elif profile.temporal_mode == "previous" and len(history) > 1 and str(history[1].get("object_id")) == str(candidate.get("object_id")):
                score += 0.12
            elif profile.temporal_mode == "previous":
                score -= 0.06
            elif profile.temporal_mode == "range" and len(history) > 1:
                score += 0.08
        if profile.needs_multi_hop_evidence:
            score += 0.04
        score += min(max(float(candidate.get("confidence") or 0.0), 0.0), 1.0) * 0.05
        return score

    def _personal_context_object_score(
        self,
        query: str,
        profile: QueryProfile,
        candidate: dict[str, Any],
    ) -> float:
        if candidate.get("object_type") != "personal_context":
            return 0.0
        score = 0.0
        text = " ".join(
            filter(
                None,
                [
                    str(candidate.get("object_text") or ""),
                    str(candidate.get("entity") or ""),
                    str(candidate.get("attribute") or ""),
                    str(candidate.get("event_text") or ""),
                ],
            )
        ).lower()
        query_terms = {
            token
            for token in tokenize(query)
            if token not in PERSONAL_CONTEXT_QUERY_STOPWORDS
        }
        candidate_terms = set(tokenize(text))
        if query_terms and candidate_terms:
            score += (len(query_terms & candidate_terms) / max(1, len(query_terms))) * 0.18
        if profile.needs_personal_context_objects:
            score += 0.08
        if set(tokenize(query)) & ADVICE_TERMS:
            score += 0.05
        score += min(max(float(candidate.get("confidence") or 0.0), 0.0), 1.0) * 0.05
        return score

    def _relation_object_score(
        self,
        query: str,
        profile: QueryProfile,
        candidate: dict[str, Any],
    ) -> float:
        if candidate.get("object_type") != "relation":
            return 0.0
        score = 0.0
        query_tokens = set(tokenize(query))
        relation_label = str(candidate.get("attribute") or candidate.get("predicate") or "").strip().lower()
        relation_text = " ".join(
            filter(
                None,
                [
                    str(candidate.get("object_text") or ""),
                    str(candidate.get("entity") or ""),
                    str(candidate.get("subject") or ""),
                    str(candidate.get("predicate") or ""),
                    str(candidate.get("attribute") or ""),
                    str(candidate.get("source_unit_text") or ""),
                ],
            )
        ).lower()
        candidate_terms = set(tokenize(relation_text))
        query_terms = {
            token
            for token in query_tokens
            if token not in COMMON_QUERY_STOPWORDS
            and token not in TEMPORAL_QUERY_STOPWORDS
            and token not in EXACT_TERMS
        }
        relation_terms = query_tokens & RELATION_TERMS
        if profile.needs_relation_objects:
            score += 0.08
        if relation_terms:
            if relation_label and relation_label in relation_terms:
                score += 0.18
            elif relation_terms & candidate_terms:
                score += 0.1
            else:
                score -= 0.04
        elif relation_label:
            score += 0.02
        if query_terms and candidate_terms:
            score += min(0.18, len(query_terms & candidate_terms) / max(1, len(query_terms)) * 0.2)
        if profile.attribute_terms:
            attribute_overlap = len(set(profile.attribute_terms) & candidate_terms) / max(1, len(set(profile.attribute_terms)))
            score += min(0.12, attribute_overlap * 0.14)
        if profile.query_person_names:
            query_name_tokens = {token for name in profile.query_person_names for token in tokenize(name)}
            candidate_name_tokens = {
                token
                for token in tokenize(
                    " ".join(
                        filter(
                            None,
                            [
                                str(candidate.get("entity") or ""),
                                str(candidate.get("subject") or ""),
                                str(candidate.get("object_text") or ""),
                                str(candidate.get("source_unit_text") or ""),
                            ],
                        )
                    )
                )
                if len(token) > 1
            }
            if query_name_tokens & candidate_name_tokens:
                score += 0.22
            elif candidate_name_tokens:
                score -= 0.12
        snapshot_state = str(candidate.get("snapshot_state") or "").strip().lower()
        if profile.temporal_mode == "latest" and snapshot_state == "current":
            score += 0.06
        elif profile.temporal_mode == "previous" and snapshot_state == "history":
            score += 0.06
        if candidate.get("effective_time") or candidate.get("valid_time"):
            score += 0.03
        score += min(max(float(candidate.get("confidence") or 0.0), 0.0), 1.0) * 0.05
        return score

    def _artifact_object_score(
        self,
        query: str,
        profile: QueryProfile,
        candidate: dict[str, Any],
    ) -> float:
        if candidate.get("object_type") != "artifact":
            return 0.0
        score = 0.0
        query_tokens = set(tokenize(query))
        text = " ".join(
            filter(
                None,
                [
                    str(candidate.get("object_text") or ""),
                    str(candidate.get("entity") or ""),
                    str(candidate.get("attribute") or ""),
                    str(candidate.get("new_value") or ""),
                    str(candidate.get("source_ref") or ""),
                    str(candidate.get("source_unit_text") or ""),
                    str(candidate.get("project") or ""),
                    str(candidate.get("workspace") or ""),
                ],
            )
        ).lower()
        candidate_terms = set(tokenize(text))
        query_terms = {
            token
            for token in query_tokens
            if token not in COMMON_QUERY_STOPWORDS
            and token not in EXACT_TERMS
        }
        if "artifact" in set(profile.preferred_object_types):
            score += 0.08
        if query_tokens & ARTIFACT_TERMS:
            score += 0.08
        if query_terms and candidate_terms:
            score += min(0.22, len(query_terms & candidate_terms) / max(1, len(query_terms)) * 0.24)
        if str(candidate.get("source_type") or "").strip().lower() == "artifact_registry":
            score += 0.05
        score += min(max(float(candidate.get("confidence") or 0.0), 0.0), 1.0) * 0.05
        return score

    def _open_loop_object_score(
        self,
        query: str,
        profile: QueryProfile,
        candidate: dict[str, Any],
    ) -> float:
        if candidate.get("object_type") != "open_loop":
            return 0.0
        score = 0.0
        query_tokens = set(tokenize(query))
        text = " ".join(
            filter(
                None,
                [
                    str(candidate.get("object_text") or ""),
                    str(candidate.get("entity") or ""),
                    str(candidate.get("status") or ""),
                    str(candidate.get("source_unit_text") or ""),
                    str(candidate.get("project") or ""),
                    str(candidate.get("workspace") or ""),
                ],
            )
        ).lower()
        candidate_terms = set(tokenize(text))
        query_terms = {
            token
            for token in query_tokens
            if token not in COMMON_QUERY_STOPWORDS
            and token not in EXACT_TERMS
        }
        status = str(candidate.get("status") or "").strip().lower()
        if "open_loop" in set(profile.preferred_object_types):
            score += 0.12
        if query_terms and candidate_terms:
            score += min(0.22, len(query_terms & candidate_terms) / max(1, len(query_terms)) * 0.24)
        if query_tokens & OPEN_LOOP_TERMS:
            score += 0.06
        if status == "open" and ({"open", "pending"} & query_tokens):
            score += 0.1
        elif status == "blocked" and "blocked" in query_tokens:
            score += 0.28
        elif status == "deferred" and "deferred" in query_tokens:
            score += 0.24
        elif {"open", "pending", "blocked", "deferred"} & query_tokens and status:
            score -= 0.22
        if str(candidate.get("source_type") or "").strip().lower() == "open_loop_registry":
            score += 0.05
        score += min(max(float(candidate.get("confidence") or 0.0), 0.0), 1.0) * 0.05
        return score

    def _normalize_preference_entity(self, candidate: dict[str, Any]) -> str:
        canonical_key = str(candidate.get("canonical_key") or "").strip().lower()
        if canonical_key:
            return canonical_key.split(":", 1)[-1]
        entity = str(candidate.get("entity") or "").strip().lower()
        if entity:
            return entity
        text = str(candidate.get("object_text") or "").lower()
        for prefix in ("user prefers ", "user avoids ", "unknown prefers ", "unknown avoids "):
            if text.startswith(prefix):
                return text[len(prefix):].strip()
        return text.strip()

    def _preference_sort_key(self, candidate: dict[str, Any]) -> tuple[float, int, float, str]:
        return (
            self._timestamp_to_epoch(candidate.get("timestamp")),
            int(candidate.get("sequence_index") or -1),
            int(candidate.get("turn_index") or -1),
            float(candidate.get("confidence") or 0.0),
            str(candidate.get("object_id") or ""),
        )

    def _temporal_entity_key(self, candidate: dict[str, Any]) -> str:
        canonical_key = str(candidate.get("canonical_key") or "").strip().lower()
        if canonical_key:
            return canonical_key
        entity = str(candidate.get("entity") or "").strip().lower()
        if entity:
            return entity
        event_text = str(candidate.get("event_text") or "").strip().lower()
        if event_text:
            return event_text
        return str(candidate.get("new_value") or "").strip().lower()

    def _temporal_sort_key(self, candidate: dict[str, Any]) -> tuple[float, int, float, str]:
        return (
            self._timestamp_to_epoch(candidate.get("timestamp")),
            int(candidate.get("sequence_index") or -1),
            int(candidate.get("turn_index") or -1),
            float(candidate.get("confidence") or 0.0),
            str(candidate.get("object_id") or ""),
        )

    def _timestamp_to_epoch(self, value: Any) -> float:
        raw = str(value or "").strip()
        if not raw:
            return 0.0
        normalized = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        try:
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            pass
        for fmt in (
            "%Y/%m/%d (%a) %H:%M",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%B %d, %Y",
            "%B %d %Y",
            "%b %d, %Y",
            "%b %d %Y",
            "%B %Y",
            "%b %Y",
        ):
            try:
                parsed = datetime.strptime(normalized, fmt).replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except ValueError:
                continue
        if re.fullmatch(r"\d{4}", normalized):
            try:
                parsed = datetime.strptime(normalized, "%Y").replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except ValueError:
                return 0.0
        return 0.0
