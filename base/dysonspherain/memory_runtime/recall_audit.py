from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .context_compiler import ContextPacket
from .evidence_vm import EvidenceOperatorSpec


@dataclass(frozen=True)
class AuditCheck:
    name: str
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecallAudit:
    packet_id: str
    checks: list[AuditCheck]
    warnings: list[str]
    risk_level: str
    suggested_followup_ops: list[EvidenceOperatorSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "checks": [check.to_dict() for check in self.checks],
            "warnings": self.warnings,
            "risk_level": self.risk_level,
            "suggested_followup_ops": [op.to_dict() for op in self.suggested_followup_ops],
        }


def audit_context_packet(packet: ContextPacket) -> RecallAudit:
    checks: list[AuditCheck] = []
    warnings: list[str] = []
    section_types = {section.section_type for section in packet.sections}
    if "active_constraints" not in section_types and packet.intent.intent_type in {"continue_task", "debug_regression", "find_prior_decision"}:
        checks.append(AuditCheck("constraint_coverage_check", "warn", "No active constraints were included."))
        warnings.append("constraint_coverage_missing")
    else:
        checks.append(AuditCheck("constraint_coverage_check", "ok", "Constraint coverage is acceptable."))
    if packet.used_tokens > packet.budget_tokens:
        checks.append(AuditCheck("token_efficiency_check", "fail", "Packet exceeds budget."))
        warnings.append("over_budget")
    elif packet.used_tokens > int(packet.budget_tokens * 0.9):
        checks.append(AuditCheck("token_efficiency_check", "warn", "Packet is close to budget."))
        warnings.append("budget_pressure")
    else:
        checks.append(AuditCheck("token_efficiency_check", "ok", "Packet is within budget."))
    if any(item.reason == "contradiction_risk" for item in packet.omitted_candidates):
        checks.append(AuditCheck("contradiction_check", "warn", "Contradictory evidence was omitted."))
        warnings.append("contradiction_omitted")
    else:
        checks.append(AuditCheck("contradiction_check", "ok", "No high-risk contradiction selected."))
    if not packet.sections:
        checks.append(AuditCheck("provenance_check", "fail", "No evidence selected."))
        warnings.append("empty_packet")
    else:
        checks.append(AuditCheck("provenance_check", "ok", "Selected evidence includes event provenance."))
    if "recent_decisions" not in section_types and packet.intent.intent_type in {"find_prior_decision", "debug_regression"}:
        checks.append(AuditCheck("supersession_check", "warn", "No decision evidence available to check supersession."))
        warnings.append("decision_context_missing")
    elif any(item.reason == "superseded" for item in packet.omitted_candidates):
        checks.append(AuditCheck("supersession_check", "warn", "Superseded evidence was detected and omitted."))
        warnings.append("superseded_evidence_omitted")
    else:
        checks.append(AuditCheck("supersession_check", "ok", "Supersession risk is bounded by selected evidence."))
    if any(item.reason == "stale" for item in packet.omitted_candidates):
        checks.append(AuditCheck("freshness_check", "warn", "Stale evidence was omitted for a freshness-sensitive intent."))
        warnings.append("stale_evidence_omitted")
    elif packet.intent.freshness_level >= 4 and not packet.sections:
        checks.append(AuditCheck("freshness_check", "warn", "Freshness-sensitive intent has no selected evidence."))
        warnings.append("fresh_context_missing")
    else:
        checks.append(AuditCheck("freshness_check", "ok", "Freshness coverage is acceptable."))
    if len(section_types) <= 1 and sum(len(section.candidate_ids) for section in packet.sections) > 2:
        checks.append(AuditCheck("diversity_check", "warn", "Selected evidence is concentrated in one section."))
        warnings.append("low_diversity")
    else:
        checks.append(AuditCheck("diversity_check", "ok", "Evidence diversity is acceptable."))
    risk = "high" if any(check.status == "fail" for check in checks) else "medium" if warnings else "low"
    followups: list[EvidenceOperatorSpec] = []
    if "constraint_coverage_missing" in warnings:
        followups.append(EvidenceOperatorSpec("constraint_lookup", 1.0, 6))
    if "decision_context_missing" in warnings:
        followups.append(EvidenceOperatorSpec("decision_lookup", 1.0, 6))
    if "fresh_context_missing" in warnings:
        followups.append(EvidenceOperatorSpec("recent_event_scan", 1.0, 6))
    if "low_diversity" in warnings:
        followups.append(EvidenceOperatorSpec("artifact_lookup", 0.7, 4))
    return RecallAudit(packet.packet_id, checks, warnings, risk, followups)
