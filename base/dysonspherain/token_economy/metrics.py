from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievalQuality:
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    ndcg_at_10: float | None = None
    gold_rank: int | None = None
    candidate_recall_at_100: float | None = None


@dataclass
class TokenEconomySample:
    sample_id: str
    query: str
    mode: str
    baseline_type: str
    context_token_budget: int | None
    raw_history_tokens: int
    raw_history_chars: int
    retrieved_context_tokens: int
    retrieved_context_chars: int
    final_prompt_tokens: int
    final_prompt_chars: int
    system_prompt_tokens: int = 0
    user_query_tokens: int = 0
    evidence_tokens: int = 0
    metadata_tokens: int = 0
    instruction_tokens: int = 0
    memory_header_tokens: int = 0
    answer_tokens: int | None = None
    saved_tokens_abs: int = 0
    saved_tokens_ratio: float = 0.0
    compression_ratio: float = 0.0
    context_reduction_ratio: float = 0.0
    retrieved_evidence_count: int = 0
    candidate_count: int = 0
    final_context_item_count: int = 0
    latency_seconds: float = 0.0
    tokenizer_name: str = ""
    fallback_tokenizer_used: bool = False
    retrieval_quality: RetrievalQuality = field(default_factory=RetrievalQuality)
    extra: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> "TokenEconomySample":
        self.saved_tokens_abs = self.raw_history_tokens - self.final_prompt_tokens
        if self.raw_history_tokens > 0:
            self.saved_tokens_ratio = self.saved_tokens_abs / self.raw_history_tokens
            self.compression_ratio = self.final_prompt_tokens / self.raw_history_tokens
            self.context_reduction_ratio = 1.0 - self.compression_ratio
        else:
            self.saved_tokens_ratio = 0.0
            self.compression_ratio = 0.0
            self.context_reduction_ratio = 0.0
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_samples(samples: list[TokenEconomySample]) -> dict[str, Any]:
    if not samples:
        return {
            "num_samples": 0,
            "sample_count": 0,
            "fallback_tokenizer_used_count": 0,
            "modes": [],
            "baseline_types": [],
            "context_token_budgets": [],
            "by_mode": {},
            "failure_case_counts": {},
            "llm_prompt_token_economy": {},
            "local_compute_economy": {},
        }
    failure_cases = detect_token_economy_failures(samples)
    by_mode: dict[str, dict[str, Any]] = {}
    for key in sorted({item.mode for item in samples}):
        group = [item for item in samples if item.mode == key]
        by_mode[key] = _summary_group(group)
    ratios = [item.saved_tokens_ratio for item in samples]
    fallback_count = sum(1 for item in samples if item.fallback_tokenizer_used)
    quality_guard_violations = sum(1 for item in samples if str(item.extra.get("quality_guard_status") or "") == "violation")
    over_budget_count = sum(1 for item in samples if item.context_token_budget is not None and item.final_prompt_tokens > item.context_token_budget)
    decision_distribution: dict[str, int] = {}
    adapter_distribution: dict[str, int] = {}
    for item in samples:
        decision = str(item.extra.get("decision") or "unknown")
        adapter = str(item.extra.get("adapter") or "evaluation")
        decision_distribution[decision] = decision_distribution.get(decision, 0) + 1
        adapter_distribution[adapter] = adapter_distribution.get(adapter, 0) + 1
    raw_total = sum(item.raw_history_tokens for item in samples)
    final_total = sum(item.final_prompt_tokens for item in samples)
    saved_total = sum(item.saved_tokens_abs for item in samples)
    local_compute = _local_compute_summary(samples)
    return {
        "num_samples": len(samples),
        "sample_count": len(samples),
        "tokenizer_name": samples[0].tokenizer_name,
        "fallback_tokenizer_used_count": fallback_count,
        "modes": sorted({item.mode for item in samples}),
        "baselines": sorted({item.baseline_type for item in samples}),
        "baseline_types": sorted({item.baseline_type for item in samples}),
        "context_token_budgets": sorted({item.context_token_budget for item in samples if item.context_token_budget is not None}),
        "avg_saved_tokens_abs": sum(item.saved_tokens_abs for item in samples) / len(samples),
        "mean_saved_tokens_abs": sum(item.saved_tokens_abs for item in samples) / len(samples),
        "avg_saved_tokens_ratio": sum(item.saved_tokens_ratio for item in samples) / len(samples),
        "mean_saved_tokens_ratio": sum(item.saved_tokens_ratio for item in samples) / len(samples),
        "p50_saved_tokens_ratio": _percentile(ratios, 0.5),
        "p90_saved_tokens_ratio": _percentile(ratios, 0.9),
        "avg_final_prompt_tokens": sum(item.final_prompt_tokens for item in samples) / len(samples),
        "mean_final_prompt_tokens": sum(item.final_prompt_tokens for item in samples) / len(samples),
        "avg_quality_drop": _mean([float(item.extra.get("quality_drop") or 0.0) for item in samples]),
        "quality_guard_violation_rate": quality_guard_violations / len(samples),
        "duplicate_token_ratio_avg": _mean([float(item.extra.get("duplicate_token_ratio") or 0.0) for item in samples]),
        "fallback_tokenizer_rate": fallback_count / len(samples),
        "over_budget_rate": over_budget_count / len(samples),
        "decision_distribution": decision_distribution,
        "adapter_distribution": adapter_distribution,
        "fallback_tokenizer_used": any(item.fallback_tokenizer_used for item in samples),
        "token_regression_count": sum(1 for item in samples if item.saved_tokens_abs < 0),
        "by_mode": by_mode,
        "failure_case_counts": {key: len(value) for key, value in failure_cases.items()},
        "llm_prompt_token_economy": {
            "baseline_context_tokens": raw_total,
            "final_injected_tokens": final_total,
            "estimated_saved_tokens": saved_total,
            "saved_ratio": (saved_total / raw_total) if raw_total else 0.0,
        },
        "local_compute_economy": {
            **local_compute,
        },
    }


def _finite(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def _summary_group(samples: list[TokenEconomySample]) -> dict[str, Any]:
    ratios = [item.saved_tokens_ratio for item in samples]
    return {
        "sample_count": len(samples),
        "avg_raw_history_tokens": _mean([float(item.raw_history_tokens) for item in samples]),
        "avg_final_prompt_tokens": _mean([float(item.final_prompt_tokens) for item in samples]),
        "avg_saved_tokens_abs": _mean([float(item.saved_tokens_abs) for item in samples]),
        "avg_saved_tokens_ratio": _mean(ratios),
        "median_saved_tokens_ratio": _percentile(ratios, 0.5),
        "p10_saved_tokens_ratio": _percentile(ratios, 0.1),
        "p90_saved_tokens_ratio": _percentile(ratios, 0.9),
        "avg_compression_ratio": _mean([item.compression_ratio for item in samples]),
        "avg_context_reduction_ratio": _mean([item.context_reduction_ratio for item in samples]),
        "avg_latency_seconds": _mean([item.latency_seconds for item in samples]),
        "avg_retrieved_evidence_count": _mean([float(item.retrieved_evidence_count) for item in samples]),
        "avg_candidate_count": _mean([float(item.candidate_count) for item in samples]),
        "avg_final_context_item_count": _mean([float(item.final_context_item_count) for item in samples]),
    }


def _local_compute_summary(samples: list[TokenEconomySample]) -> dict[str, Any]:
    embedding_hits = 0.0
    embedding_misses = 0.0
    retrieval_hits = 0.0
    retrieval_total = 0.0
    profile_hits = 0.0
    profile_total = 0.0
    saved_ms = 0.0
    for item in samples:
        local = item.extra.get("local_compute_economy") if isinstance(item.extra, dict) else {}
        if not isinstance(local, dict):
            local = {}
        embedding_hits += float(local.get("embedding_cache_hit_count") or local.get("embedding_cache_hits") or 0.0)
        embedding_misses += float(local.get("embedding_cache_miss_count") or local.get("embedding_cache_misses") or 0.0)
        saved_ms += float(local.get("estimated_local_runtime_saved_ms") or local.get("embedding_cache_hit_ms_saved") or 0.0)
        if local.get("retrieval_cache_hit_rate") is not None:
            retrieval_hits += float(local.get("retrieval_cache_hit_rate") or 0.0)
            retrieval_total += 1.0
        if local.get("profile_cache_hit_rate") is not None:
            profile_hits += float(local.get("profile_cache_hit_rate") or 0.0)
            profile_total += 1.0
    embedding_total = embedding_hits + embedding_misses
    return {
        "embedding_cache_hit_rate": (embedding_hits / embedding_total) if embedding_total else 0.0,
        "retrieval_cache_hit_rate": (retrieval_hits / retrieval_total) if retrieval_total else 0.0,
        "profile_cache_hit_rate": (profile_hits / profile_total) if profile_total else 0.0,
        "estimated_local_runtime_saved_ms": round(saved_ms, 2),
    }


def _quality_value(item: TokenEconomySample) -> float | None:
    quality = _finite(item.retrieval_quality.recall_at_10)
    if quality is None:
        quality = _finite(item.retrieval_quality.ndcg_at_10)
    return quality


def detect_token_economy_failures(
    samples: list[TokenEconomySample],
    low_saving_threshold: float = 0.2,
    evidence_bloat_threshold: float = 0.85,
    metadata_bloat_threshold: float = 0.25,
    quality_drop_threshold: float = 0.05,
) -> dict[str, list[dict[str, Any]]]:
    failures: dict[str, list[dict[str, Any]]] = {
        "token_regression": [],
        "low_saving": [],
        "quality_drop_with_high_compression": [],
        "paired_quality_drop": [],
        "evidence_bloat": [],
        "metadata_bloat": [],
    }
    quality_by_mode: dict[str, float] = {}
    for mode in {item.mode for item in samples}:
        values = [
            value
            for item in samples
            if item.mode == mode
            for value in [_finite(item.retrieval_quality.recall_at_10), _finite(item.retrieval_quality.ndcg_at_10)]
            if value is not None
        ]
        quality_by_mode[mode] = _mean(values)
    for item in samples:
        base = {
            "sample_id": item.sample_id,
            "mode": item.mode,
            "baseline_type": item.baseline_type,
            "context_token_budget": item.context_token_budget,
            "raw_history_tokens": item.raw_history_tokens,
            "final_prompt_tokens": item.final_prompt_tokens,
            "saved_tokens_ratio": item.saved_tokens_ratio,
        }
        if item.final_prompt_tokens > item.raw_history_tokens:
            failures["token_regression"].append({**base, "regression_tokens": item.final_prompt_tokens - item.raw_history_tokens})
        if item.saved_tokens_ratio < low_saving_threshold:
            failures["low_saving"].append(base)
        if item.final_prompt_tokens > 0 and item.evidence_tokens / item.final_prompt_tokens > evidence_bloat_threshold:
            failures["evidence_bloat"].append({**base, "evidence_token_ratio": item.evidence_tokens / item.final_prompt_tokens})
        overhead = item.metadata_tokens + item.memory_header_tokens + item.instruction_tokens
        if item.final_prompt_tokens > 0 and overhead / item.final_prompt_tokens > metadata_bloat_threshold:
            failures["metadata_bloat"].append({**base, "metadata_overhead_ratio": overhead / item.final_prompt_tokens})
        quality = _quality_value(item)
        if item.saved_tokens_ratio > 0.7 and quality is not None and quality < quality_by_mode.get(item.mode, quality) - quality_drop_threshold:
            failures["quality_drop_with_high_compression"].append({**base, "quality": quality, "mode_average_quality": quality_by_mode.get(item.mode)})
    paired: dict[tuple[str, int | None], list[TokenEconomySample]] = {}
    for item in samples:
        paired.setdefault((item.sample_id, item.context_token_budget), []).append(item)
    for (_, _), group in paired.items():
        quality_values = [_quality_value(item) for item in group]
        available = [value for value in quality_values if value is not None]
        if not available:
            continue
        best_quality = max(available)
        best_rows = [item for item in group if _quality_value(item) == best_quality]
        best_mode = best_rows[0].mode if best_rows else ""
        best_baseline = best_rows[0].baseline_type if best_rows else ""
        for item in group:
            quality = _quality_value(item)
            if quality is None:
                continue
            if item.saved_tokens_ratio > 0.2 and quality < best_quality - quality_drop_threshold:
                failures["paired_quality_drop"].append(
                    {
                        "sample_id": item.sample_id,
                        "mode": item.mode,
                        "baseline_type": item.baseline_type,
                        "context_token_budget": item.context_token_budget,
                        "quality": quality,
                        "best_paired_quality": best_quality,
                        "quality_delta": quality - best_quality,
                        "best_mode": best_mode,
                        "best_baseline_type": best_baseline,
                        "saved_tokens_ratio": item.saved_tokens_ratio,
                    }
                )
    failures["token_regression"].sort(key=lambda row: int(row.get("regression_tokens") or 0), reverse=True)
    failures["paired_quality_drop"].sort(key=lambda row: float(row.get("quality_delta") or 0.0))
    return {key: value[:20] for key, value in failures.items()}
