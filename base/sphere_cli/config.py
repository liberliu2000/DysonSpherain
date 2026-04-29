from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from .workspace import DEFAULT_SCOPE_ORDER, apply_mode_profile, normalize_mode, normalize_scope_order


@dataclass
class AppConfig:
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    shared_cache_dir: Path | None = None
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    export_dir: Path = field(init=False)
    vector_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    ingest_state_path: Path = field(init=False)
    embedding_cache_path: Path = field(init=False)
    workspace_name: str | None = None
    project_name: str | None = None
    session_id: str | None = None
    scope: str = "global"
    retrieval_scope_order: tuple[str, ...] = DEFAULT_SCOPE_ORDER
    mode: str = "balanced"
    vector_collection_name: str = "raw_chunks"
    object_collection_name: str = "memory_objects"
    proxy_collection_name: str = "retrieval_proxies"
    vector_backend: str = "auto"
    embedding_dim: int = 384
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_fail_fast: bool = False
    embedding_cache_memory_size: int = 20000
    chunk_size: int = 800
    chunk_overlap: int = 120
    local_window_span: int = 2
    embed_local_grain: bool = False
    markdown_chunk_size: int = 1200
    code_chunk_lines: int = 80
    log_chunk_lines: int = 120
    pdf_chunk_size: int = 1000
    rerank_mode_default: str = "hybrid"
    rerank_top_k: int = 8
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_cross_encoder_rerank: bool = False
    enable_task_router: bool = True
    enable_object_shortcut: bool = True
    enable_temporal_prefilter: bool = True
    retrieval_topk_coarse: int = 24
    retrieval_topk_fine: int = 8
    enable_seed_clustering: bool = True
    enable_semantic_dedup: bool = True
    enable_conditional_rerank: bool = True
    enable_ingest_compression: bool = True
    enable_content_hash_dedup: bool = True
    enable_delta_memory_writer: bool = True
    enable_structured_compression: bool = True
    enable_multires_summaries: bool = True
    enable_retrieval_proxy_index: bool = True
    enable_context_compressor: bool = True
    enable_retrieval_cache: bool = True
    enable_completion_cache: bool = True
    enable_profile_snapshot_cache: bool = True
    enable_prism_path_cache: bool = True
    enable_benchmark_route_tuning: bool = True
    enable_lightweight_edge_writeback: bool = True
    enable_identity_aware_rerank: bool = True
    enable_segment_rerank: bool = True
    enable_confusing_cluster_rerank: bool = True
    enable_object_shortcut_cache: bool = True
    enable_identity_feature_cache: bool = True
    enable_segment_feature_cache: bool = True
    enable_confusing_cluster_cache: bool = True
    enable_scope_priority: bool = True
    scope_priority_weight: float = 0.035
    enable_artifact_registry: bool = True
    enable_note_artifact_auto_register: bool = True
    enable_open_loop_tracking: bool = True
    enable_three_sentence_spans: bool = True
    segment_span_lengths: tuple[int, ...] = (1, 2, 3)
    wrong_domain_penalty_weight: float = 0.0
    wrong_role_target_penalty_weight: float = 0.0
    wrong_subtheme_penalty_weight: float = 0.0
    generic_topic_penalty_weight: float = 0.0
    cluster_specificity_weight: float = 0.04
    retrieval_cache_ttl_seconds: int = 3600
    completion_cache_ttl_seconds: int = 3600
    profile_snapshot_ttl_seconds: int = 3600
    segment_rerank_topk_default: int = 12
    confusing_cluster_topk_default: int = 20
    cold_access_threshold: int = 1
    cold_days_threshold: int = 30
    zone_split_threshold: int = 40
    zone_split_group_size: int = 20
    edge_decay_factor: float = 0.97
    edge_decay_floor: float = 0.05
    watch_poll_seconds: float = 3.0
    creative_mode: str | bool = "off"
    creative_beam_width: int = 6
    creative_max_hops: int = 2
    creative_neighbors_per_hop: int = 4
    creative_enable_analogy: bool = True
    creative_enable_contrast: bool = True
    creative_enable_transfer: bool = True
    creative_enable_temporal: bool = True
    creative_enable_composition: bool = True
    creative_novelty_weight: float = 0.2
    creative_support_weight: float = 0.24
    creative_diversity_weight: float = 0.16
    creative_conflict_penalty: float = 0.22
    creative_reflection_gain: float = 0.18
    creative_max_output_paths: int = 4

    def __post_init__(self) -> None:
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.cache_dir = Path(self.shared_cache_dir) if self.shared_cache_dir is not None else (self.data_dir / "cache")
        self.export_dir = self.data_dir / "exports"
        self.vector_dir = self.data_dir / "vector_db"
        self.db_path = self.data_dir / "memory.db"
        self.ingest_state_path = self.cache_dir / "ingest_state.json"
        self.embedding_cache_path = self.cache_dir / "embedding_cache.sqlite3"
        self.mode = normalize_mode(self.mode)
        self.retrieval_scope_order = normalize_scope_order(self.retrieval_scope_order)
        apply_mode_profile(self)

    def ensure_dirs(self) -> None:
        for path in [self.data_dir, self.raw_dir, self.cache_dir, self.export_dir, self.vector_dir]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def creative_mode_name(self) -> str:
        return _normalize_creative_mode(self.creative_mode)

    @property
    def creative_enabled(self) -> bool:
        return self.creative_mode_name != "off"

    @property
    def creative_is_conservative(self) -> bool:
        return self.creative_mode_name == "conservative"

    @property
    def creative_is_exploratory(self) -> bool:
        return self.creative_mode_name == "exploratory"

    @classmethod
    def from_env(cls, base_dir: Path | None = None, overrides: dict[str, object] | None = None) -> "AppConfig":
        env = os.environ
        config = cls(
            base_dir=base_dir or Path.cwd(),
            shared_cache_dir=Path(env["SPHERE_SHARED_CACHE_DIR"]) if env.get("SPHERE_SHARED_CACHE_DIR") else None,
            workspace_name=env.get("SPHERE_WORKSPACE_NAME") or None,
            project_name=env.get("SPHERE_PROJECT_NAME") or None,
            session_id=env.get("SPHERE_SESSION_ID") or None,
            scope=env.get("SPHERE_SCOPE", "global"),
            retrieval_scope_order=normalize_scope_order(env.get("SPHERE_SCOPE_ORDER", ",".join(DEFAULT_SCOPE_ORDER))),
            mode=normalize_mode(env.get("SPHERE_MODE", "balanced")),
            proxy_collection_name=env.get("SPHERE_PROXY_COLLECTION_NAME", "retrieval_proxies"),
            vector_backend=env.get("SPHERE_VECTOR_BACKEND", "auto").strip().lower() or "auto",
            embedding_fail_fast=_env_bool("SPHERE_EMBEDDING_FAIL_FAST", False),
            enable_task_router=_env_bool("SPHERE_ENABLE_TASK_ROUTER", True),
            enable_object_shortcut=_env_bool("SPHERE_ENABLE_OBJECT_SHORTCUT", True),
            enable_temporal_prefilter=_env_bool("SPHERE_ENABLE_TEMPORAL_PREFILTER", True),
            retrieval_topk_coarse=_env_int("SPHERE_RETRIEVAL_TOPK_COARSE", 24),
            retrieval_topk_fine=_env_int("SPHERE_RETRIEVAL_TOPK_FINE", 8),
            enable_seed_clustering=_env_bool("SPHERE_ENABLE_SEED_CLUSTERING", True),
            enable_semantic_dedup=_env_bool("SPHERE_ENABLE_SEMANTIC_DEDUP", True),
            enable_conditional_rerank=_env_bool("SPHERE_ENABLE_CONDITIONAL_RERANK", True),
            enable_ingest_compression=_env_bool("SPHERE_ENABLE_INGEST_COMPRESSION", True),
            enable_content_hash_dedup=_env_bool("SPHERE_ENABLE_CONTENT_HASH_DEDUP", True),
            enable_delta_memory_writer=_env_bool("SPHERE_ENABLE_DELTA_MEMORY_WRITER", True),
            enable_structured_compression=_env_bool("SPHERE_ENABLE_STRUCTURED_COMPRESSION", True),
            enable_multires_summaries=_env_bool("SPHERE_ENABLE_MULTIRES_SUMMARIES", True),
            enable_retrieval_proxy_index=_env_bool("SPHERE_ENABLE_RETRIEVAL_PROXY_INDEX", True),
            enable_context_compressor=_env_bool("SPHERE_ENABLE_CONTEXT_COMPRESSOR", True),
            enable_retrieval_cache=_env_bool("SPHERE_ENABLE_RETRIEVAL_CACHE", True),
            enable_completion_cache=_env_bool("SPHERE_ENABLE_COMPLETION_CACHE", True),
            enable_profile_snapshot_cache=_env_bool("SPHERE_ENABLE_PROFILE_SNAPSHOT_CACHE", True),
            enable_prism_path_cache=_env_bool("SPHERE_ENABLE_PRISM_PATH_CACHE", True),
            enable_benchmark_route_tuning=_env_bool("SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING", True),
            enable_lightweight_edge_writeback=_env_bool("SPHERE_ENABLE_LIGHTWEIGHT_EDGE_WRITEBACK", True),
            enable_identity_aware_rerank=_env_bool("SPHERE_ENABLE_IDENTITY_AWARE_RERANK", True),
            enable_segment_rerank=_env_bool("SPHERE_ENABLE_SEGMENT_RERANK", True),
            enable_confusing_cluster_rerank=_env_bool("SPHERE_ENABLE_CONFUSING_CLUSTER_RERANK", True),
            enable_object_shortcut_cache=_env_bool("SPHERE_ENABLE_OBJECT_SHORTCUT_CACHE", True),
            enable_identity_feature_cache=_env_bool("SPHERE_ENABLE_IDENTITY_FEATURE_CACHE", True),
            enable_segment_feature_cache=_env_bool("SPHERE_ENABLE_SEGMENT_FEATURE_CACHE", True),
            enable_confusing_cluster_cache=_env_bool("SPHERE_ENABLE_CONFUSING_CLUSTER_CACHE", True),
            enable_scope_priority=_env_bool("SPHERE_ENABLE_SCOPE_PRIORITY", True),
            scope_priority_weight=_env_float("SPHERE_SCOPE_PRIORITY_WEIGHT", 0.035),
            enable_artifact_registry=_env_bool("SPHERE_ENABLE_ARTIFACT_REGISTRY", True),
            enable_note_artifact_auto_register=_env_bool("SPHERE_ENABLE_NOTE_ARTIFACT_AUTO_REGISTER", True),
            enable_open_loop_tracking=_env_bool("SPHERE_ENABLE_OPEN_LOOP_TRACKING", True),
            enable_three_sentence_spans=_env_bool("SPHERE_ENABLE_THREE_SENTENCE_SPANS", True),
            segment_span_lengths=_env_int_tuple("SPHERE_SEGMENT_SPAN_LENGTHS", (1, 2, 3)),
            wrong_domain_penalty_weight=_env_float("SPHERE_WRONG_DOMAIN_PENALTY_WEIGHT", 0.0),
            wrong_role_target_penalty_weight=_env_float("SPHERE_WRONG_ROLE_TARGET_PENALTY_WEIGHT", 0.0),
            wrong_subtheme_penalty_weight=_env_float("SPHERE_WRONG_SUBTHEME_PENALTY_WEIGHT", 0.0),
            generic_topic_penalty_weight=_env_float("SPHERE_GENERIC_TOPIC_PENALTY_WEIGHT", 0.0),
            cluster_specificity_weight=_env_float("SPHERE_CLUSTER_SPECIFICITY_WEIGHT", 0.04),
            retrieval_cache_ttl_seconds=_env_int("SPHERE_RETRIEVAL_CACHE_TTL_SECONDS", 3600),
            completion_cache_ttl_seconds=_env_int("SPHERE_COMPLETION_CACHE_TTL_SECONDS", 3600),
            profile_snapshot_ttl_seconds=_env_int("SPHERE_PROFILE_SNAPSHOT_TTL_SECONDS", 3600),
            segment_rerank_topk_default=_env_int("SPHERE_SEGMENT_RERANK_TOPK_DEFAULT", 12),
            confusing_cluster_topk_default=_env_int("SPHERE_CONFUSING_CLUSTER_TOPK_DEFAULT", 20),
            creative_mode=_env_creative_mode("SPHERE_CREATIVE_MODE", "off"),
            creative_beam_width=_env_int("SPHERE_CREATIVE_BEAM_WIDTH", 6),
            creative_max_hops=_env_int("SPHERE_CREATIVE_MAX_HOPS", 2),
            creative_neighbors_per_hop=_env_int("SPHERE_CREATIVE_NEIGHBORS_PER_HOP", 4),
            creative_enable_analogy=_env_bool("SPHERE_CREATIVE_ENABLE_ANALOGY", True),
            creative_enable_contrast=_env_bool("SPHERE_CREATIVE_ENABLE_CONTRAST", True),
            creative_enable_transfer=_env_bool("SPHERE_CREATIVE_ENABLE_TRANSFER", True),
            creative_enable_temporal=_env_bool("SPHERE_CREATIVE_ENABLE_TEMPORAL", True),
            creative_enable_composition=_env_bool("SPHERE_CREATIVE_ENABLE_COMPOSITION", True),
            creative_novelty_weight=_env_float("SPHERE_CREATIVE_NOVELTY_WEIGHT", 0.2),
            creative_support_weight=_env_float("SPHERE_CREATIVE_SUPPORT_WEIGHT", 0.24),
            creative_diversity_weight=_env_float("SPHERE_CREATIVE_DIVERSITY_WEIGHT", 0.16),
            creative_conflict_penalty=_env_float("SPHERE_CREATIVE_CONFLICT_PENALTY", 0.22),
            creative_reflection_gain=_env_float("SPHERE_CREATIVE_REFLECTION_GAIN", 0.18),
            creative_max_output_paths=_env_int("SPHERE_CREATIVE_MAX_OUTPUT_PATHS", 4),
        )
        for key, value in dict(overrides or {}).items():
            if value is None or not hasattr(config, key):
                continue
            setattr(config, key, value)
        config.mode = normalize_mode(config.mode)
        config.retrieval_scope_order = normalize_scope_order(config.retrieval_scope_order)
        apply_mode_profile(config)
        return config


DEFAULT_SHELL_POLICY = {
    0: {"name": "core", "compression": "very_high", "max_items_hint": 500},
    1: {"name": "active", "compression": "high", "max_items_hint": 2000},
    2: {"name": "stable_knowledge", "compression": "medium", "max_items_hint": 6000},
    3: {"name": "case_experience", "compression": "low_medium", "max_items_hint": 10000},
    4: {"name": "raw_material", "compression": "minimal", "max_items_hint": 50000},
}

DEFAULT_SECTORS = [
    "user",
    "project",
    "knowledge",
    "method",
    "case",
    "raw",
    "creative",
    "archive",
]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_creative_mode(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _normalize_creative_mode(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            continue
    return tuple(values or list(default))


def _normalize_creative_mode(value: str | bool | None) -> str:
    if isinstance(value, bool):
        return "exploratory" if value else "off"
    lowered = str(value or "").strip().lower()
    if lowered in {"1", "true", "yes", "on", "explore", "exploratory"}:
        return "exploratory"
    if lowered in {"safe", "guarded", "conservative"}:
        return "conservative"
    return "off"
