from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceItem:
    text: str
    memory_id: str = ""
    timestamp: str = ""
    source: str = ""
    confidence: float = 0.0
    stale_possible: bool = False
    uncertain: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelevantFile:
    path: str
    reason: str
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextPack:
    summary: str = ""
    core_evidence: list[EvidenceItem] = field(default_factory=list)
    prior_decisions: list[str] = field(default_factory=list)
    known_failures: list[str] = field(default_factory=list)
    benchmark_state: list[str] = field(default_factory=list)
    relevant_files: list[RelevantFile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended_next_actions: list[str] = field(default_factory=list)
    token_economy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "core_evidence": [item.to_dict() for item in self.core_evidence],
            "prior_decisions": list(self.prior_decisions),
            "known_failures": list(self.known_failures),
            "benchmark_state": list(self.benchmark_state),
            "relevant_files": [item.to_dict() for item in self.relevant_files],
            "warnings": list(self.warnings),
            "recommended_next_actions": list(self.recommended_next_actions),
            "token_economy": dict(self.token_economy),
        }
