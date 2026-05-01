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
    updated = {**current, **clean_patch}
    config = RuntimeConfig(**updated)
    path = config_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_event_payload(
        base_dir,
        event_type="constraint_changed",
        payload={"title": "Memory runtime configuration changed", "patch": clean_patch, "config_path": str(path)},
        source="memory_runtime_config",
        actor=actor,
        project=project,
        provenance={"config_path": str(path)},
    )
    return {"status": "ok", "config": config.to_dict(), "path": str(path), "changed_keys": sorted(clean_patch)}
