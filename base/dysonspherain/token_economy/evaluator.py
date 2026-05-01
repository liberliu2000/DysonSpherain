from __future__ import annotations

import re
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
    relevance_score: float = 0.0
    duplication_score: float = 0.0
    novelty_score: float = 0.0
    density_score: float = 0.0
    protected_evidence_score: float = 0.0
    reference_sufficiency_score: float = 0.0
    quality_guard_status: str = "unknown"
    source_files: tuple[str, ...] = ()
    diagnostics: dict[str, Any] | None = None

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
            "relevance_score": self.relevance_score,
            "duplication_score": self.duplication_score,
            "novelty_score": self.novelty_score,
            "density_score": self.density_score,
            "protected_evidence_score": self.protected_evidence_score,
            "reference_sufficiency_score": self.reference_sufficiency_score,
            "quality_guard_status": self.quality_guard_status,
            "source_files": list(self.source_files),
            "diagnostics": dict(self.diagnostics or {}),
        }


SECRET_RE = re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{8,}")
PATH_RE = re.compile(r"(?:^|\s)([\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|txt|log))")
PROTECTED_MARKERS = (
    "benchmark",
    "regression",
    "failure",
    "failed",
    "candidate_recall",
    "recall@",
    "ndcg",
    "gold_rank",
    "safe fusion",
    "guard",
    "fallback",
)
EVIDENCE_MARKERS = ("dyson://", "file:", "path:", ".py", ".ts", ".json", "test", "error", "decision", "because")


def _tokens(value: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[\w./-]+", value or "") if len(item) > 1}


def _ngram_set(tokens: list[str], n: int = 4) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set(tuple([token]) for token in tokens)
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def _duplication_score(candidate_context: str, existing_context: str) -> float:
    if not candidate_context.strip() or not existing_context.strip():
        return 0.0
    cand_tokens = [item.lower() for item in re.findall(r"[\w./-]+", candidate_context) if len(item) > 1]
    existing_tokens = [item.lower() for item in re.findall(r"[\w./-]+", existing_context) if len(item) > 1]
    cand_ngrams = _ngram_set(cand_tokens)
    existing_ngrams = _ngram_set(existing_tokens)
    ngram_overlap = len(cand_ngrams & existing_ngrams) / max(1, len(cand_ngrams))
    cand_lines = {" ".join(line.lower().split()) for line in candidate_context.splitlines() if line.strip()}
    existing_lines = {" ".join(line.lower().split()) for line in existing_context.splitlines() if line.strip()}
    line_overlap = len(cand_lines & existing_lines) / max(1, len(cand_lines))
    return max(0.0, min(1.0, max(ngram_overlap, line_overlap)))


def _anchor_overlap(query: str, candidate_context: str) -> float:
    query_paths = set(PATH_RE.findall(query))
    context_paths = set(PATH_RE.findall(candidate_context))
    path_score = len(query_paths & context_paths) / max(1, len(query_paths)) if query_paths else 0.0
    query_entities = {item for item in _tokens(query) if any(char.isupper() for char in item) or "_" in item or "/" in item}
    context_tokens = _tokens(candidate_context)
    entity_score = len(query_entities & context_tokens) / max(1, len(query_entities)) if query_entities else 0.0
    return max(path_score, entity_score)


def _risk_score(candidate_context: str, task_type: str) -> tuple[float, list[str]]:
    lowered = candidate_context.lower()
    reasons: list[str] = []
    score = 0.1
    if SECRET_RE.search(candidate_context):
        score += 0.75
        reasons.append("secret_like_content")
    if any(marker in lowered for marker in ("traceback", "stderr", "stdout", "raw log")):
        score += 0.15
        reasons.append("raw_log_content")
    if any(marker in lowered for marker in ("fallback", "regression", "benchmark")):
        score += 0.15
        reasons.append("sensitive_regression_context")
    if task_type in {"benchmark", "debug", "paper"}:
        score += 0.1
    return min(1.0, score), reasons


def _density_score(candidate_context: str, tokens: int) -> float:
    if tokens <= 0:
        return 0.0
    hits = sum(candidate_context.lower().count(marker) for marker in EVIDENCE_MARKERS)
    return min(1.0, hits / max(1.0, tokens / 120))


def _protected_score(candidate_context: str) -> float:
    lowered = candidate_context.lower()
    hits = sum(1 for marker in PROTECTED_MARKERS if marker in lowered)
    return min(1.0, hits / 4)


def _reference_sufficiency(candidate_context: str) -> tuple[float, list[str]]:
    paths = sorted(set(PATH_RE.findall(candidate_context)))
    prose_tokens = len(re.findall(r"\w+", candidate_context))
    if not paths:
        return 0.0, []
    return min(1.0, len(paths) / max(1.0, prose_tokens / 80)), paths


def evaluate(
    *,
    query: str,
    candidate_context: str,
    existing_context: str = "",
    baseline_context_tokens: int = 0,
    token_budget: int = 1600,
    task_type: str = "unknown",
    mode: str = "conservative",
    source_files: list[str] | None = None,
    quality_signals: dict[str, Any] | None = None,
    counter: TokenCounter | None = None,
) -> TokenEconomyDecision:
    counter = counter or TokenCounter()
    count = counter.count(candidate_context)
    lexical = lexical_score(query, candidate_context)
    relevance_score = min(1.0, lexical + 0.35 * _anchor_overlap(query, candidate_context))
    duplication_score = _duplication_score(candidate_context, existing_context)
    novelty_score = max(0.0, min(1.0, relevance_score * (1.0 - duplication_score)))
    risk_score, risk_reasons = _risk_score(candidate_context, task_type)
    density_score = _density_score(candidate_context, count.tokens)
    protected_evidence_score = _protected_score(candidate_context)
    reference_sufficiency_score, context_paths = _reference_sufficiency(candidate_context)
    source_files = source_files or context_paths
    normalized_mode = str(mode or "conservative").lower()
    budget = max(1, int(token_budget or 1600))

    if normalized_mode == "off":
        decision = "skip"
        reason = "Token economy mode is off; record baseline without injection."
    elif not candidate_context.strip() or relevance_score < 0.05:
        decision = "skip"
        reason = "Candidate context is empty or has very low relevance."
    elif risk_score > 0.70 and reference_sufficiency_score >= 0.1:
        decision = "return_file_refs_only"
        reason = "Context risk is high; prefer file references over prose injection."
    elif duplication_score > 0.80 and novelty_score < 0.20:
        decision = "skip" if protected_evidence_score < 0.3 else "inject_summary_only"
        reason = "Candidate context is highly duplicative; protected evidence is summarized if present."
    elif count.tokens > budget:
        decision = "inject_summary_only"
        reason = "Candidate context exceeds the requested token budget."
    elif normalized_mode == "minimal" and reference_sufficiency_score >= 0.15:
        decision = "return_file_refs_only"
        reason = "Minimal mode prefers file references and key facts."
    elif normalized_mode == "benchmark" and protected_evidence_score >= 0.25 and count.tokens <= budget:
        decision = "inject"
        reason = "Benchmark/debug protected evidence is relevant and within budget."
    elif normalized_mode == "conservative" and risk_score > 0.45 and reference_sufficiency_score >= 0.1:
        decision = "return_file_refs_only"
        reason = "Conservative mode avoids medium-risk prose when references are sufficient."
    else:
        decision = "inject"
        reason = "Context is relevant and within budget."
    estimated_saved = max(0, int(baseline_context_tokens) - count.tokens)
    risk = "high" if risk_score > 0.70 else "medium" if risk_score > 0.30 else "low"
    if count.fallback_used:
        reason = f"{reason} Fallback tokenizer used: {count.tokenizer_name}."
    if risk_reasons:
        reason = f"{reason} Risk signals: {', '.join(risk_reasons)}."
    quality_guard_status = "ok"
    if quality_signals and quality_signals.get("quality_drop") is not None:
        try:
            quality_guard_status = "violation" if float(quality_signals["quality_drop"]) > 0.05 else "ok"
        except (TypeError, ValueError):
            quality_guard_status = "unknown"
    return TokenEconomyDecision(
        status="ok",
        decision=decision,
        estimated_tokens=count.tokens,
        estimated_saved_tokens=estimated_saved,
        risk=risk,
        reason=reason,
        tokenizer_name=count.tokenizer_name,
        fallback_tokenizer_used=count.fallback_used,
        relevance_score=round(relevance_score, 4),
        duplication_score=round(duplication_score, 4),
        novelty_score=round(novelty_score, 4),
        density_score=round(density_score, 4),
        protected_evidence_score=round(protected_evidence_score, 4),
        reference_sufficiency_score=round(reference_sufficiency_score, 4),
        quality_guard_status=quality_guard_status,
        source_files=tuple(source_files),
        diagnostics={
            "mode": normalized_mode,
            "task_type": task_type,
            "token_budget": budget,
            "risk_score": round(risk_score, 4),
            "risk_reasons": risk_reasons,
        },
    )
