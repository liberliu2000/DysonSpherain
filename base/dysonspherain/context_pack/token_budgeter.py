from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dysonspherain.utils.token_counter import TokenCounter

from .renderers import render_markdown
from .schemas import ContextPack, EvidenceItem


DEFAULT_BUDGETS = {
    "quick": 800,
    "coding": 1600,
    "debug": 1800,
    "benchmark": 3000,
    "paper": 2500,
    "planning": 2000,
    "unknown": 1200,
}


@dataclass(frozen=True)
class BudgetPolicy:
    total_budget: int
    core_evidence_ratio: float = 0.42
    file_refs_ratio: float = 0.08
    prior_decisions_ratio: float = 0.12
    known_failures_ratio: float = 0.12
    benchmark_state_ratio: float = 0.10
    actions_ratio: float = 0.06
    warnings_ratio: float = 0.04
    reserve_ratio: float = 0.06

    def __post_init__(self) -> None:
        if self.ratio_sum > 1.000001:
            raise ValueError(f"BudgetPolicy ratios exceed 1.0: {self.ratio_sum:.4f}")

    @property
    def ratio_sum(self) -> float:
        return round(
            self.core_evidence_ratio
            + self.file_refs_ratio
            + self.prior_decisions_ratio
            + self.known_failures_ratio
            + self.benchmark_state_ratio
            + self.actions_ratio
            + self.warnings_ratio
            + self.reserve_ratio,
            6,
        )

    def allocate(self) -> dict[str, int]:
        return {
            "core_evidence": int(self.total_budget * self.core_evidence_ratio),
            "relevant_files": int(self.total_budget * self.file_refs_ratio),
            "prior_decisions": int(self.total_budget * self.prior_decisions_ratio),
            "known_failures": int(self.total_budget * self.known_failures_ratio),
            "benchmark_state": int(self.total_budget * self.benchmark_state_ratio),
            "recommended_next_actions": int(self.total_budget * self.actions_ratio),
            "warnings": int(self.total_budget * self.warnings_ratio),
            "reserve": max(0, self.total_budget - int(self.total_budget * (self.ratio_sum - self.reserve_ratio))),
        }


def budget_policy_for(task_type: str, total_budget: int) -> BudgetPolicy:
    normalized = str(task_type or "unknown").lower()
    if normalized in {"benchmark", "debug"}:
        return BudgetPolicy(total_budget=total_budget, core_evidence_ratio=0.36, known_failures_ratio=0.16, benchmark_state_ratio=0.14, prior_decisions_ratio=0.10)
    if normalized == "paper":
        return BudgetPolicy(total_budget=total_budget, core_evidence_ratio=0.34, prior_decisions_ratio=0.16, benchmark_state_ratio=0.14, actions_ratio=0.04)
    return BudgetPolicy(total_budget=total_budget)


@dataclass
class BudgetResult:
    pack: ContextPack
    estimated_tokens_before: int
    estimated_tokens_after: int
    compression_ratio: float
    dropped_items: list[str] = field(default_factory=list)
    kept_items: list[str] = field(default_factory=list)
    over_budget: bool = False
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_tokens_before": self.estimated_tokens_before,
            "estimated_tokens_after": self.estimated_tokens_after,
            "compression_ratio": self.compression_ratio,
            "dropped_items": list(self.dropped_items),
            "kept_items": list(self.kept_items),
            "over_budget": self.over_budget,
            "manifest": dict(self.manifest),
        }


RAW_LOG_MARKERS = ("traceback", "loading weights", "stdout", "stderr", "debug log", "raw log", "```")
PROMPT_MARKERS = ("user prompt", "assistant", "system prompt", "claude", "codex prompt")
SAFETY_MARKERS = ("do not", "must not", "never", "fallback", "secret", "api key", "password", "local_hash")
REGRESSION_MARKERS = ("regression", "failure", "failed", "error", "blocker", "candidate_recall", "recall@", "ndcg", "gold_rank")


def _compact_text(value: str, max_chars: int, suffix: str = " ...[truncated]") -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= max_chars:
        return text, False
    return text[: max(1, max_chars - len(suffix))].rstrip() + suffix, True


def _dedupe_strings(values: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    result: list[str] = []
    dropped = 0
    for value in values:
        key = " ".join(str(value).lower().split())
        if not key:
            dropped += 1
            continue
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        result.append(value)
    return result, dropped


def _evidence_priority(item: EvidenceItem) -> tuple[int, float, int]:
    text = item.text.lower()
    protected = any(marker in text for marker in REGRESSION_MARKERS + SAFETY_MARKERS)
    low_value = item.uncertain or item.confidence < 0.1 or any(marker in text for marker in RAW_LOG_MARKERS + PROMPT_MARKERS)
    return (1 if protected else 0, float(item.confidence), 0 if low_value else 1)


def _precompress_pack(pack: ContextPack, dropped: list[str]) -> None:
    for attr in ("prior_decisions", "known_failures", "benchmark_state", "warnings", "recommended_next_actions"):
        values, count = _dedupe_strings(list(getattr(pack, attr)))
        setattr(pack, attr, values)
        dropped.extend([f"{attr}:duplicate_or_empty"] * count)

    seen_evidence: set[str] = set()
    evidence: list[EvidenceItem] = []
    for item in pack.core_evidence:
        key = " ".join(item.text.lower().split())
        if not key or key in seen_evidence:
            dropped.append("core_evidence:duplicate_or_empty")
            continue
        seen_evidence.add(key)
        text_lower = item.text.lower()
        max_chars = 900
        if any(marker in text_lower for marker in RAW_LOG_MARKERS):
            max_chars = 360
        elif any(marker in text_lower for marker in PROMPT_MARKERS):
            max_chars = 520
        elif item.uncertain or item.confidence < 0.1:
            max_chars = 480
        compact, did = _compact_text(item.text, max_chars)
        if did:
            dropped.append("core_evidence:low_value_or_long_text_compacted")
            item.text = compact
        evidence.append(item)
    evidence.sort(key=_evidence_priority, reverse=True)
    pack.core_evidence = evidence

    compacted_benchmarks: list[str] = []
    for value in pack.benchmark_state:
        compact, did = _compact_text(value, 700)
        if did:
            dropped.append("benchmark_state:explanatory_tail_compacted")
        compacted_benchmarks.append(compact)
    pack.benchmark_state = compacted_benchmarks

    compacted_failures: list[str] = []
    for value in pack.known_failures:
        compact, did = _compact_text(value, 700)
        if did:
            dropped.append("known_failures:explanatory_tail_compacted")
        compacted_failures.append(compact)
    pack.known_failures = compacted_failures

    for file_ref in pack.relevant_files:
        reason, did = _compact_text(file_ref.reason, 180)
        if did:
            dropped.append("relevant_files:reason_tail_compacted")
        file_ref.reason = reason


def _drop_one_lowest(pack: ContextPack, section: str, dropped: list[str]) -> bool:
    values = getattr(pack, section)
    if not values:
        return False
    if section == "core_evidence":
        if len(values) <= 1:
            return False
        lowest_index = min(range(len(values)), key=lambda idx: _evidence_priority(values[idx]))
        values.pop(lowest_index)
    else:
        if len(values) <= 1:
            return False
        values.pop()
    dropped.append(section)
    return True


def fit_context_pack(pack: ContextPack, token_budget: int, counter: TokenCounter | None = None, policy: BudgetPolicy | None = None) -> BudgetResult:
    counter = counter or TokenCounter()
    policy = policy or BudgetPolicy(total_budget=token_budget)
    dropped: list[str] = []
    kept: list[str] = []
    _precompress_pack(pack, dropped)
    before = counter.count(render_markdown(pack)).tokens
    if before <= token_budget:
        return BudgetResult(pack, before, before, 1.0, dropped, ["all"], False, {"policy": policy.allocate(), "actual_tokens": {"all": before}, "dropped_count": len(dropped)})

    # Drop expendable sections before touching protected benchmark/failure/file-ref evidence.
    drop_order = (
        "warnings",
        "recommended_next_actions",
        "core_evidence",
        "prior_decisions",
        "benchmark_state",
        "known_failures",
        "relevant_files",
    )
    made_progress = True
    while counter.count(render_markdown(pack)).tokens > token_budget and made_progress:
        made_progress = False
        for section in drop_order:
            if counter.count(render_markdown(pack)).tokens <= token_budget:
                break
            if _drop_one_lowest(pack, section, dropped):
                made_progress = True

    for section in ("summary", "core_evidence", "prior_decisions", "known_failures", "benchmark_state", "relevant_files", "warnings", "recommended_next_actions"):
        value = getattr(pack, section)
        if value:
            kept.append(section)

    if counter.count(render_markdown(pack)).tokens > token_budget and pack.summary:
        max_chars = max(120, token_budget)
        pack.summary, did = _compact_text(pack.summary, max_chars)
        if did:
            dropped.append("summary_tail")

    if counter.count(render_markdown(pack)).tokens > token_budget:
        for section, max_chars in (("core_evidence", 300), ("prior_decisions", 240), ("benchmark_state", 240), ("known_failures", 240)):
            if counter.count(render_markdown(pack)).tokens <= token_budget:
                break
            values = getattr(pack, section)
            for idx, value in enumerate(values):
                if counter.count(render_markdown(pack)).tokens <= token_budget:
                    break
                if section == "core_evidence":
                    value.text, did = _compact_text(value.text, max_chars)
                else:
                    values[idx], did = _compact_text(str(value), max_chars)
                if did:
                    dropped.append(f"{section}:protected_tail_compacted")

    after = counter.count(render_markdown(pack)).tokens
    return BudgetResult(
        pack=pack,
        estimated_tokens_before=before,
        estimated_tokens_after=after,
        compression_ratio=(after / before) if before else 0.0,
        dropped_items=dropped,
        kept_items=kept,
        over_budget=after > token_budget,
        manifest={"policy": policy.allocate(), "actual_tokens": {"total": after}, "dropped_count": len(dropped)},
    )
