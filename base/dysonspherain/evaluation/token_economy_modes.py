from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class TokenEconomyMode:
    name: str
    status: str
    description: str
    runtime_overrides: Mapping[str, Any] | None = None
    retrieval_disabled: bool = False
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return self.status == "available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "description": self.description,
            "runtime_overrides": dict(self.runtime_overrides or {}),
            "retrieval_disabled": self.retrieval_disabled,
            "unavailable_reason": self.unavailable_reason,
        }


def _mode(
    name: str,
    *,
    description: str,
    runtime_overrides: dict[str, Any] | None = None,
    retrieval_disabled: bool = False,
    status: str = "available",
    unavailable_reason: str = "",
) -> TokenEconomyMode:
    return TokenEconomyMode(
        name=name,
        status=status,
        description=description,
        runtime_overrides=MappingProxyType(dict(runtime_overrides or {})) if runtime_overrides is not None else None,
        retrieval_disabled=retrieval_disabled,
        unavailable_reason=unavailable_reason,
    )


TOKEN_ECONOMY_MODE_REGISTRY: dict[str, TokenEconomyMode] = {
    "off": _mode(
        "off",
        description="Disable runtime retrieval/context assembly for baseline comparison.",
        retrieval_disabled=True,
    ),
    "conservative": _mode(
        "conservative",
        description="Fast runtime assembly with creative expansion disabled.",
        runtime_overrides={"mode": "fast", "creative_mode": "off"},
    ),
    "exploratory": _mode(
        "exploratory",
        description="Deep runtime assembly with exploratory cognitive context enabled.",
        runtime_overrides={"mode": "deep", "creative_mode": "exploratory"},
    ),
    "fast": _mode("fast", description="Runtime fast mode.", runtime_overrides={"mode": "fast", "creative_mode": "off"}),
    "balanced": _mode("balanced", description="Runtime balanced mode.", runtime_overrides={"mode": "balanced", "creative_mode": "off"}),
    "deep": _mode("deep", description="Runtime deep mode.", runtime_overrides={"mode": "deep", "creative_mode": "off"}),
    "dense_only": _mode(
        "dense_only",
        description="Runtime dense-channel diagnostic mode with non-dense channels disabled.",
        runtime_overrides={
            "mode": "balanced",
            "creative_mode": "off",
            "multi_channel_enabled": False,
            "dense_channel_enabled": True,
            "lexical_channel_enabled": False,
            "temporal_channel_enabled": False,
            "entity_channel_enabled": False,
            "exact_phrase_channel_enabled": False,
        },
    ),
    "safe_fusion": _mode(
        "safe_fusion",
        description="Runtime balanced mode with safe fusion enabled.",
        runtime_overrides={"mode": "balanced", "creative_mode": "off", "safe_fusion_enabled": True},
    ),
    "temporal_edge": _mode(
        "temporal_edge",
        description="Runtime balanced mode with temporal channel and temporal neighbor expansion enabled.",
        runtime_overrides={"mode": "balanced", "creative_mode": "off", "temporal_channel_enabled": True, "temporal_neighbor_enabled": True},
    ),
    "inhibition_on": _mode(
        "inhibition_on",
        description="Runtime balanced mode with competition inhibition enabled.",
        runtime_overrides={"mode": "balanced", "creative_mode": "off", "competition_inhibition_enabled": True},
    ),
    "inhibition_off": _mode(
        "inhibition_off",
        description="Runtime balanced mode with competition inhibition disabled.",
        runtime_overrides={"mode": "balanced", "creative_mode": "off", "competition_inhibition_enabled": False},
    ),
    "profile_enhanced": _mode(
        "profile_enhanced",
        description="Runtime balanced mode with profile side-index support enabled.",
        runtime_overrides={"mode": "balanced", "creative_mode": "off", "profile_side_index_enabled": True},
    ),
}


def resolve_token_economy_mode(mode: str) -> TokenEconomyMode:
    normalized = str(mode or "").strip().lower() or "off"
    registered = TOKEN_ECONOMY_MODE_REGISTRY.get(normalized)
    if registered is not None:
        return registered
    return _mode(
        normalized,
        status="unavailable",
        description="No verified Token Economy runtime adapter is registered for this mode.",
        unavailable_reason="mode_not_registered_or_unverified",
    )


def list_token_economy_modes() -> list[dict[str, Any]]:
    return [TOKEN_ECONOMY_MODE_REGISTRY[name].to_dict() for name in sorted(TOKEN_ECONOMY_MODE_REGISTRY)]
