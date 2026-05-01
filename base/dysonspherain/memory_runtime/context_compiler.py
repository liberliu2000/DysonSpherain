from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .events import stable_hash
from .evidence_vm import EvidenceCandidate, RecallIntent


@dataclass(frozen=True)
class ContextSection:
    section_type: str
    title: str
    content: str
    candidate_ids: list[str]
    used_tokens: int
    utility_per_token: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OmittedEvidence:
    candidate_id: str
    reason: str
    token_cost: int
    score: float
    source_event_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextPacket:
    packet_id: str
    query: str
    intent: RecallIntent
    budget_tokens: int
    used_tokens: int
    sections: list[ContextSection]
    omitted_candidates: list[OmittedEvidence]
    compiler_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "query": self.query,
            "intent": self.intent.to_dict(),
            "budget_tokens": self.budget_tokens,
            "used_tokens": self.used_tokens,
            "sections": [section.to_dict() for section in self.sections],
            "omitted_candidates": [item.to_dict() for item in self.omitted_candidates],
            "compiler_trace": self.compiler_trace,
        }


DEFAULT_WEIGHTS = {
    "task_utility": 1.0,
    "freshness": 0.25,
    "provenance_strength": 0.25,
    "diversity_contribution": 0.20,
    "user_constraint_priority": 0.30,
    "redundancy_penalty": -0.40,
    "contradiction_risk": -0.60,
}


SECTION_BY_TYPE = {
    "constraint_added": "active_constraints",
    "constraint_changed": "active_constraints",
    "preference_declared": "active_constraints",
    "decision_made": "recent_decisions",
    "failure_observed": "failure_and_recovery",
    "recovery_attempted": "failure_and_recovery",
    "regression_detected": "failure_and_recovery",
    "benchmark_finished": "metric_state",
    "metric_changed": "metric_state",
    "artifact_created": "artifact_references",
    "artifact_updated": "artifact_references",
    "patch_applied": "critical_evidence",
    "file_changed": "critical_evidence",
    "user_instruction_received": "current_task_state",
    "hypothesis_created": "open_questions",
}


def candidate_score(candidate: EvidenceCandidate, weights: dict[str, float] | None = None) -> float:
    active = {**DEFAULT_WEIGHTS, **(weights or {})}
    return sum(float(candidate.scores.get(key, 0.0)) * weight for key, weight in active.items())


def compile_context(
    *,
    query: str,
    intent: RecallIntent,
    candidates: list[EvidenceCandidate],
    budget_tokens: int,
    weights: dict[str, float] | None = None,
    trace: dict[str, Any] | None = None,
) -> ContextPacket:
    selected: list[EvidenceCandidate] = []
    omitted: list[OmittedEvidence] = []
    seen_text: set[str] = set()
    used_tokens = 0
    scored = sorted(((candidate_score(item, weights), item) for item in candidates), key=lambda pair: pair[0], reverse=True)
    for score, candidate in scored:
        normalized = " ".join(candidate.text.lower().split())[:240]
        if normalized in seen_text:
            omitted.append(OmittedEvidence(candidate.candidate_id, "duplicate", candidate.token_cost, score, candidate.source_event_ids))
            continue
        if candidate.provenance.get("superseded_by") or "superseded" in candidate.text.lower():
            omitted.append(OmittedEvidence(candidate.candidate_id, "superseded", candidate.token_cost, score, candidate.source_event_ids))
            continue
        if candidate.scores.get("freshness", 0.0) < 0.08 and packet_intent_needs_freshness(intent):
            omitted.append(OmittedEvidence(candidate.candidate_id, "stale", candidate.token_cost, score, candidate.source_event_ids))
            continue
        if candidate.scores.get("provenance_strength", 0.0) < 0.4:
            omitted.append(OmittedEvidence(candidate.candidate_id, "low_provenance", candidate.token_cost, score, candidate.source_event_ids))
            continue
        if candidate.scores.get("contradiction_risk", 0.0) >= 0.8:
            omitted.append(OmittedEvidence(candidate.candidate_id, "contradiction_risk", candidate.token_cost, score, candidate.source_event_ids))
            continue
        if used_tokens + candidate.token_cost > budget_tokens:
            omitted.append(OmittedEvidence(candidate.candidate_id, "over_budget", candidate.token_cost, score, candidate.source_event_ids))
            continue
        if score <= 0:
            omitted.append(OmittedEvidence(candidate.candidate_id, "low_utility", candidate.token_cost, score, candidate.source_event_ids))
            continue
        selected.append(candidate)
        seen_text.add(normalized)
        used_tokens += candidate.token_cost
    grouped: dict[str, list[EvidenceCandidate]] = {}
    for candidate in selected:
        grouped.setdefault(SECTION_BY_TYPE.get(candidate.evidence_type, "critical_evidence"), []).append(candidate)
    sections: list[ContextSection] = []
    for section_type, items in grouped.items():
        content = "\n".join(f"- {item.text} [{','.join(item.source_event_ids)}]" for item in items)
        total_tokens = sum(item.token_cost for item in items)
        total_score = sum(candidate_score(item, weights) for item in items)
        sections.append(
            ContextSection(
                section_type=section_type,
                title=section_type.replace("_", " ").title(),
                content=content,
                candidate_ids=[item.candidate_id for item in items],
                used_tokens=total_tokens,
                utility_per_token=total_score / max(1, total_tokens),
            )
        )
    packet_id = f"packet_{stable_hash([query, intent.to_dict(), [item.candidate_id for item in selected], budget_tokens])[:18]}"
    compiler_trace = {
        "weights": {**DEFAULT_WEIGHTS, **(weights or {})},
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "omitted_count": len(omitted),
        "utility_per_token": sum(section.utility_per_token for section in sections) / max(1, len(sections)),
        "section_allocation": {section.section_type: section.used_tokens for section in sections},
        **(trace or {}),
    }
    return ContextPacket(packet_id, query, intent, budget_tokens, used_tokens, sections, omitted, compiler_trace)


def packet_intent_needs_freshness(intent: RecallIntent) -> bool:
    return intent.intent_type in {"continue_task", "debug_regression", "recover_interrupted_work", "explain_metric_change"} or intent.freshness_level >= 4


def render_context_packet(packet: ContextPacket) -> str:
    lines = [
        "# DysonSpherain Context Packet",
        "",
        f"- packet_id: `{packet.packet_id}`",
        f"- intent: `{packet.intent.intent_type}`",
        f"- budget_tokens: `{packet.budget_tokens}`",
        f"- used_tokens: `{packet.used_tokens}`",
    ]
    for section in packet.sections:
        lines.extend(["", f"## {section.title}", "", section.content])
    if packet.omitted_candidates:
        lines.extend(["", "## Omitted Evidence", ""])
        for item in packet.omitted_candidates[:20]:
            lines.append(f"- `{item.candidate_id}`: {item.reason} ({item.token_cost} tokens)")
    return "\n".join(lines) + "\n"
