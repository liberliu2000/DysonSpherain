from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dysonspherain.utils.token_counter import TokenCounter
from sphere_cli.utils import lexical_score


@dataclass(frozen=True)
class TokenEconomyDecision:
    status: str
    decision: str
    estimated_tokens: int
    estimated_saved_tokens: int
    risk: str
    reason: str
    tokenizer_name: str
    fallback_tokenizer_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision": self.decision,
            "estimated_tokens": self.estimated_tokens,
            "estimated_saved_tokens": self.estimated_saved_tokens,
            "risk": self.risk,
            "reason": self.reason,
            "tokenizer_name": self.tokenizer_name,
            "fallback_tokenizer_used": self.fallback_tokenizer_used,
        }


def evaluate(
    *,
    query: str,
    candidate_context: str,
    baseline_context_tokens: int = 0,
    token_budget: int = 1600,
    task_type: str = "unknown",
    counter: TokenCounter | None = None,
) -> TokenEconomyDecision:
    counter = counter or TokenCounter()
    count = counter.count(candidate_context)
    relevance_score = lexical_score(query, candidate_context)
    duplication_score = 0.0
    novelty_score = min(1.0, relevance_score + 0.2) if candidate_context.strip() else 0.0
    risk_score = 0.15
    if task_type in {"benchmark", "debug", "paper"}:
        risk_score = 0.35
    if "fallback" in candidate_context.lower() or "regression" in candidate_context.lower():
        risk_score = min(1.0, risk_score + 0.25)

    if not candidate_context.strip() or relevance_score < 0.05:
        decision = "skip"
        reason = "Candidate context is empty or has very low lexical relevance."
    elif duplication_score > 0.80 and novelty_score < 0.20:
        decision = "skip"
        reason = "Candidate context appears duplicative and low novelty."
    elif count.tokens > token_budget:
        decision = "inject_summary_only"
        reason = "Candidate context exceeds the requested token budget."
    elif risk_score > 0.70:
        decision = "return_file_refs_only"
        reason = "Context risk is high; prefer references over prose injection."
    else:
        decision = "inject"
        reason = "Context is relevant and within budget."
    estimated_saved = max(0, int(baseline_context_tokens) - count.tokens)
    risk = "high" if risk_score > 0.70 else "medium" if risk_score > 0.30 else "low"
    return TokenEconomyDecision(
        status="ok",
        decision=decision,
        estimated_tokens=count.tokens,
        estimated_saved_tokens=estimated_saved,
        risk=risk,
        reason=reason,
        tokenizer_name=count.tokenizer_name,
        fallback_tokenizer_used=count.fallback_used,
    )
