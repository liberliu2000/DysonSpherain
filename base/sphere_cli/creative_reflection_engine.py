from __future__ import annotations

from typing import Any

from .utils import clamp


class CreativeReflectionEngine:
    def score_candidates(
        self,
        task_relatedness: float,
        creative_weight: float,
        novelty: float,
        structural_potential: float,
        conflict: float,
    ) -> float:
        alpha, beta, gamma, delta, lam = 0.32, 0.24, 0.18, 0.18, 0.28
        score = (
            alpha * task_relatedness
            + beta * creative_weight
            + gamma * novelty
            + delta * structural_potential
            - lam * conflict
        )
        return clamp(score, 0.0, 1.0)

    def score_path(
        self,
        *,
        relevance: float,
        novelty: float,
        support: float,
        feasibility: float,
        diversity: float,
        conflict_risk: float,
        redundancy_penalty: float,
        novelty_weight: float,
        support_weight: float,
        diversity_weight: float,
        conflict_penalty: float,
        reflection_gain: float = 0.0,
    ) -> float:
        score = (
            relevance * 0.34
            + novelty * novelty_weight
            + support * support_weight
            + feasibility * 0.18
            + diversity * diversity_weight
            + reflection_gain
            - conflict_risk * conflict_penalty
            - redundancy_penalty * 0.16
        )
        return clamp(score, 0.0, 1.6)

    def annotate(self, items: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        annotated = []
        for item in items:
            enriched = dict(item)
            enriched["bucket"] = label
            annotated.append(enriched)
        return annotated
