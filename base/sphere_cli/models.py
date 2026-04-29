from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import uuid


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class MemoryNode:
    shell: int
    sector: str
    zone: str
    cell: str
    molecular_type: str
    summary: str
    content_hash: str | None = None
    normalized_hash: str | None = None
    content_ref: str | None = None
    raw_content: str | None = None
    scope: str | None = None
    workspace: str | None = None
    project: str | None = None
    session_id: str | None = None
    source_type: str = "memory_note"
    source_ref: str | None = None
    extraction_method: str = "manual"
    confidence: float = 0.0
    verification_status: str = "unverified"
    metadata_json: str | None = None
    seen_count: int = 1
    last_seen_at: str | None = None
    base_node_id: str | None = None
    delta_summary: str | None = None
    changed_fields_json: str | None = None
    retrieval_summary: str | None = None
    structured_summary: str | None = None
    retrieval_signature: str | None = None
    time_bucket: str | None = None
    entity_tags: str | None = None
    task_type_tag: str | None = None
    theta: float | None = None
    phi: float | None = None
    importance: float = 0.0
    creative_score: float = 0.0
    stability_score: float = 0.0
    access_count: int = 0
    compression_level: str = "medium"
    stage: str = "long_term"
    tags: str | None = None
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    last_accessed_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEdge:
    source_id: str
    target_id: str
    semantic_weight: float = 0.0
    task_weight: float = 0.0
    temporal_weight: float = 0.0
    causal_weight: float = 0.0
    creative_weight: float = 0.0
    structural_weight: float = 0.0
    id: str = field(default_factory=lambda: f"edge_{uuid.uuid4().hex[:12]}")
    last_activated_at: str = field(default_factory=now_iso)


@dataclass
class MemoryObject:
    object_type: str
    object_text: str
    source_chunk_id: str
    source_node_id: str
    scope: str | None = None
    workspace: str | None = None
    project: str | None = None
    subject: str | None = None
    predicate: str | None = None
    polarity: float | None = None
    entity: str | None = None
    attribute: str | None = None
    old_value: str | None = None
    new_value: str | None = None
    event_text: str | None = None
    canonical_key: str | None = None
    temporal_marker: str | None = None
    sequence_index: int | None = None
    source_unit_text: str | None = None
    content_hash: str | None = None
    confidence: float = 0.0
    session_id: str | None = None
    status: str = "active"
    turn_index: int | None = None
    timestamp: str | None = None
    snapshot_key: str | None = None
    merge_policy: str | None = None
    source_type: str = "memory_extraction"
    source_ref: str | None = None
    extraction_method: str = "heuristic"
    verification_status: str = "unverified"
    related_artifact_ids_json: str | None = None
    metadata_json: str | None = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    object_id: str = field(default_factory=lambda: f"obj_{uuid.uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActivationBundle:
    task: str
    task_type: str
    temperature: float
    primary_evidence: list[dict[str, Any]]
    core_evidence: list[dict[str, Any]]
    evidence_objects: list[dict[str, Any]]
    supporting_context: list[dict[str, Any]]
    relevant_experience: list[dict[str, Any]]
    creative_reflections: list[dict[str, Any]]
    alternative_paths: list[dict[str, Any]]
    raw_reference_pointers: list[str]
    debug: dict[str, Any]


@dataclass
class QueryProfile:
    task_type: str
    needs_exact_evidence: bool
    needs_multi_hop_evidence: bool
    needs_preference_objects: bool
    needs_temporal_objects: bool
    needs_cognitive_expansion: str
    granularity_bias: list[str]
    lexical_priority: float
    semantic_priority: float
    preferred_object_types: list[str]
    preference_polarity_hint: float | None = None
    needs_personal_context_objects: bool = False
    needs_relation_objects: bool = False
    temporal_mode: str = "none"
    temporal_reference_terms: list[str] = field(default_factory=list)
    query_person_names: list[str] = field(default_factory=list)
    attribute_terms: list[str] = field(default_factory=list)


@dataclass
class EvidenceRetrievalResult:
    profile: QueryProfile
    candidates: list[dict[str, Any]]
    evidence_nodes: list[dict[str, Any]]
    dense_object_hits: list[dict[str, Any]] = field(default_factory=list)
    sparse_object_hits: list[dict[str, Any]] = field(default_factory=list)
    query_route: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredCompletionResult:
    profile: QueryProfile
    core_evidence: list[dict[str, Any]]
    evidence_objects: list[dict[str, Any]]
    supporting_context: list[dict[str, Any]]
    evidence_nodes: list[dict[str, Any]]
    timings_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveAugmentationResult:
    relevant_experience: list[dict[str, Any]]
    creative_reflections: list[dict[str, Any]]
    alternative_paths: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeamPath:
    beam_type: str
    seed_node_id: str
    seed_chunk_id: str
    node_ids: list[str]
    chunk_ids: list[str]
    hop_count: int
    relevance: float
    novelty: float
    support: float
    feasibility: float
    diversity: float
    conflict_risk: float
    redundancy_penalty: float
    score: float
    endpoint_score: float = 0.0
    trajectory_score: float = 0.0
    backflow_score: float = 0.0
    amplified_score: float = 0.0
    mmr_score: float = 0.0
    backflow_gain: float = 0.0
    path_role: str = "alternative"
    signature: str = ""
    summary: str = ""
    reflections: list[str] = field(default_factory=list)
    evidence_anchor_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    path_id: str = field(default_factory=lambda: f"path_{uuid.uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RealTaskCaseResult:
    task_id: str
    task_type: str
    latency_ms: float
    passed: bool
    errors: list[str]
    diagnostics: dict[str, Any]


@dataclass
class ArtifactRecord:
    path: str
    artifact_type: str
    scope: str = "global"
    workspace: str | None = None
    project: str | None = None
    session_id: str | None = None
    title: str | None = None
    summary: str | None = None
    tags_json: str | None = None
    source_type: str = "file"
    source_ref: str | None = None
    related_memory_ids_json: str | None = None
    related_object_ids_json: str | None = None
    related_goal_ids_json: str | None = None
    metadata_json: str | None = None
    artifact_id: str = field(default_factory=lambda: f"art_{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpenLoopItem:
    title: str
    details: str | None = None
    status: str = "open"
    scope: str = "global"
    workspace: str | None = None
    project: str | None = None
    session_id: str | None = None
    priority: str = "normal"
    tags_json: str | None = None
    blocked_reason: str | None = None
    source_type: str = "manual"
    source_ref: str | None = None
    related_memory_ids_json: str | None = None
    related_artifact_ids_json: str | None = None
    metadata_json: str | None = None
    loop_id: str = field(default_factory=lambda: f"loop_{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRoute:
    task_type: str
    preferred_shells: list[int]
    preferred_sectors: list[str]
    creative_temperature: float = 0.3
    compression_policy: str = "balanced"


@dataclass
class QueryRouteDecision:
    route_type: str
    confidence: float
    normalized_task_type: str
    lexical_strength: float
    prefer_object_shortcut: bool
    prefer_temporal_prefilter: bool
    prefer_light_rerank: bool
    allow_creative: bool
    retrieval_intensity: str
    suggested_config: dict[str, Any] = field(default_factory=dict)
    prefer_identity_rerank: bool = False
    prefer_segment_rerank: bool = False
    prefer_confusing_cluster: bool = False
    benchmark_profile: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
