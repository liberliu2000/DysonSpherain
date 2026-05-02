from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .ledger import append_event_payload


@dataclass(frozen=True)
class RuntimeConfig:
    embedding_backend: str = "project_vector_store"
    lexical_backend: str = "ledger_lexical"
    projection_backend: str = "json_projection"
    context_budget: int = 1200
    max_followup_rounds: int = 1
    max_extra_tokens: int = 400
    enabled_operators: list[str] = field(
        default_factory=lambda: [
            "recent_event_scan",
            "lexical_exact_search",
            "decision_lookup",
            "constraint_lookup",
            "artifact_lookup",
            "metric_delta_scan",
            "failure_lookup",
            "patch_lookup",
            "code_region_lookup",
            "hypothesis_lookup",
            "user_preference_scan",
        ]
    )
    operator_weights: dict[str, float] = field(
        default_factory=lambda: {
            "recent_event_scan": 1.0,
            "lexical_exact_search": 0.75,
            "decision_lookup": 0.85,
            "constraint_lookup": 0.8,
            "artifact_lookup": 0.7,
            "metric_delta_scan": 1.0,
            "failure_lookup": 0.85,
            "patch_lookup": 0.75,
        }
    )
    section_limits: dict[str, int] = field(
        default_factory=lambda: {
            "current_goal": 180,
            "constraints": 220,
            "decisions": 220,
            "benchmark_state": 260,
            "files": 180,
            "failures": 220,
            "patches": 220,
            "raw_evidence": 260,
        }
    )
    scheduler_triggers: list[str] = field(
        default_factory=lambda: [
            "session_ended",
            "benchmark_finished",
            "metric_regression_detected",
            "artifact_updated",
            "user_preference_declared",
            "decision_changed",
            "index_staleness_detected",
        ]
    )
    audit_checks: list[str] = field(
        default_factory=lambda: [
            "constraint_coverage_check",
            "freshness_check",
            "supersession_check",
            "provenance_check",
            "contradiction_check",
            "token_efficiency_check",
            "diversity_check",
        ]
    )
    privacy_filters: list[str] = field(default_factory=lambda: ["secret_redaction", "private_block_redaction", "dysonignore"])
    cache_policy: str = "trace_only"
    ui_animation_intensity: str = "medium"
    llm_config: dict[str, Any] = field(
        default_factory=lambda: {
            "provider": "auto",
            "mode": "use_existing_agent_if_available",
            "api_base_url": "",
            "api_key": "",
            "model": "",
            "external_llm_enabled": False,
            "fallback_to_local": True,
            "require_user_config_for_external_api": True,
            "local_only": True,
            "privacy_mode": "strict",
            "allow_raw_memory_external": False,
            "require_source_ids": True,
            "require_verifier": True,
            "use_existing_agent_if_available": True,
        }
    )
    compaction_config: dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "deterministic",
            "exact_hash_enabled": True,
            "near_duplicate_enabled": True,
            "near_duplicate_threshold": 0.9,
            "min_cluster_size": 3,
            "token_saving_threshold": 0.2,
            "auto_commit": False,
            "preserve_raw_memory": True,
            "max_input_memories": 24,
            "max_input_tokens": 12000,
            "max_output_tokens": 700,
            "timeout_seconds": 45,
            "retry_count": 1,
            "external_llm_compaction_enabled": False,
        }
    )
    scoring_config: dict[str, Any] = field(
        default_factory=lambda: {
            "recency_weight": 0.12,
            "importance_weight": 0.18,
            "confidence_weight": 0.1,
            "access_weight": 0.05,
            "redundancy_weight": 0.2,
            "recency_half_life_days": 30,
            "stable_memory_decay_multiplier": 0.25,
            "decision_decay_multiplier": 0.5,
            "active_only_default": True,
            "include_superseded_evidence": False,
            "token_budget": 2000,
        }
    )
    privacy_config: dict[str, Any] = field(
        default_factory=lambda: {
            "local_only": True,
            "allow_raw_memory_external": False,
            "allow_canonical_memory_external": True,
            "redact_secrets": True,
            "require_external_call_confirmation": True,
            "show_external_call_preview": True,
        }
    )
    lifecycle_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "active": 1.0,
            "stable": 1.08,
            "canonical": 1.15,
            "compacted": 0.3,
            "superseded": 0.0,
            "deprecated": 0.0,
            "contradicted": 0.0,
            "archived": 0.0,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_path(base_dir: Path) -> Path:
    return base_dir / "data" / "config" / "memory_runtime_config.json"


def load_runtime_config(base_dir: Path) -> RuntimeConfig:
    path = config_path(base_dir)
    if not path.exists():
        return RuntimeConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return RuntimeConfig()
    defaults = RuntimeConfig().to_dict()
    merged = {**defaults, **payload}
    return RuntimeConfig(**merged)


def _redact_config_patch(patch: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(patch, ensure_ascii=False))
    llm = redacted.get("llm_config")
    if isinstance(llm, dict) and llm.get("api_key"):
        llm["api_key"] = "***redacted***"
    return redacted


def save_runtime_config(base_dir: Path, patch: dict[str, Any], *, project: str = "DysonSpherain", actor: str = "user") -> dict[str, Any]:
    current = load_runtime_config(base_dir).to_dict()
    allowed = set(current)
    clean_patch = {key: value for key, value in patch.items() if key in allowed}
    if "context_budget" in clean_patch:
        clean_patch["context_budget"] = max(100, int(clean_patch["context_budget"]))
    if "max_followup_rounds" in clean_patch:
        clean_patch["max_followup_rounds"] = max(0, min(int(clean_patch["max_followup_rounds"]), 3))
    if "max_extra_tokens" in clean_patch:
        clean_patch["max_extra_tokens"] = max(0, int(clean_patch["max_extra_tokens"]))
    if "operator_weights" in clean_patch:
        clean_patch["operator_weights"] = {str(key): float(value) for key, value in dict(clean_patch["operator_weights"] or {}).items()}
    if "section_limits" in clean_patch:
        clean_patch["section_limits"] = {str(key): max(20, int(value)) for key, value in dict(clean_patch["section_limits"] or {}).items()}
    if "llm_config" in clean_patch:
        previous = dict(current.get("llm_config") or {})
        incoming = dict(clean_patch["llm_config"] or {})
        if "api_key" in incoming and not incoming["api_key"]:
            incoming.pop("api_key", None)
        clean_patch["llm_config"] = {**previous, **incoming}
    if "compaction_config" in clean_patch:
        previous = dict(current.get("compaction_config") or {})
        incoming = dict(clean_patch["compaction_config"] or {})
        if "max_input_memories" in incoming:
            incoming["max_input_memories"] = max(2, min(int(incoming["max_input_memories"]), 100))
        if "max_output_tokens" in incoming:
            incoming["max_output_tokens"] = max(80, min(int(incoming["max_output_tokens"]), 8000))
        if "timeout_seconds" in incoming:
            incoming["timeout_seconds"] = max(3, min(int(incoming["timeout_seconds"]), 300))
        if "retry_count" in incoming:
            incoming["retry_count"] = max(0, min(int(incoming["retry_count"]), 5))
        if "near_duplicate_threshold" in incoming:
            incoming["near_duplicate_threshold"] = max(0.1, min(float(incoming["near_duplicate_threshold"]), 1.0))
        if "min_cluster_size" in incoming:
            incoming["min_cluster_size"] = max(2, min(int(incoming["min_cluster_size"]), 50))
        clean_patch["compaction_config"] = {**previous, **incoming}
    if "scoring_config" in clean_patch:
        previous = dict(current.get("scoring_config") or {})
        incoming = dict(clean_patch["scoring_config"] or {})
        clean_patch["scoring_config"] = {**previous, **incoming}
    if "privacy_config" in clean_patch:
        previous = dict(current.get("privacy_config") or {})
        incoming = dict(clean_patch["privacy_config"] or {})
        clean_patch["privacy_config"] = {**previous, **incoming}
    if "lifecycle_multipliers" in clean_patch:
        clean_patch["lifecycle_multipliers"] = {str(key): float(value) for key, value in dict(clean_patch["lifecycle_multipliers"] or {}).items()}
    updated = {**current, **clean_patch}
    config = RuntimeConfig(**updated)
    path = config_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_event_payload(
        base_dir,
        event_type="constraint_changed",
        payload={"title": "Memory runtime configuration changed", "patch": _redact_config_patch(clean_patch), "config_path": str(path)},
        source="memory_runtime_config",
        actor=actor,
        project=project,
        provenance={"config_path": str(path)},
    )
    return {"status": "ok", "config": config.to_dict(), "path": str(path), "changed_keys": sorted(clean_patch)}
