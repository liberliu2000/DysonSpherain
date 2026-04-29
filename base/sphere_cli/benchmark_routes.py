from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import QueryProfile, QueryRouteDecision


@dataclass
class BenchmarkRouteTuning:
    benchmark: str = ""
    route_profile: str = "default"
    coarse_topk: int | None = None
    fine_topk: int | None = None
    dense_probe_k: int | None = None
    proxy_probe_k: int | None = None
    sparse_probe_k: int | None = None
    rerank_pool_k: int | None = None
    segment_rerank_topk: int = 0
    confusing_cluster_topk: int = 0
    prefer_object_shortcut: bool | None = None
    prefer_identity_rerank: bool = False
    prefer_segment_rerank: bool = False
    prefer_confusing_cluster: bool = False
    identity_reward_weight: float = 0.0
    wrong_entity_penalty_weight: float = 0.0
    wrong_domain_penalty_weight: float = 0.0
    wrong_role_target_penalty_weight: float = 0.0
    wrong_subtheme_penalty_weight: float = 0.0
    generic_topic_penalty_weight: float = 0.0
    confusing_neighbor_penalty_weight: float = 0.0
    segment_weight: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "route_profile": self.route_profile,
            "coarse_topk": self.coarse_topk,
            "fine_topk": self.fine_topk,
            "dense_probe_k": self.dense_probe_k,
            "proxy_probe_k": self.proxy_probe_k,
            "sparse_probe_k": self.sparse_probe_k,
            "rerank_pool_k": self.rerank_pool_k,
            "segment_rerank_topk": self.segment_rerank_topk,
            "confusing_cluster_topk": self.confusing_cluster_topk,
            "prefer_object_shortcut": self.prefer_object_shortcut,
            "prefer_identity_rerank": self.prefer_identity_rerank,
            "prefer_segment_rerank": self.prefer_segment_rerank,
            "prefer_confusing_cluster": self.prefer_confusing_cluster,
            "identity_reward_weight": round(self.identity_reward_weight, 4),
            "wrong_entity_penalty_weight": round(self.wrong_entity_penalty_weight, 4),
            "wrong_domain_penalty_weight": round(self.wrong_domain_penalty_weight, 4),
            "wrong_role_target_penalty_weight": round(self.wrong_role_target_penalty_weight, 4),
            "wrong_subtheme_penalty_weight": round(self.wrong_subtheme_penalty_weight, 4),
            "generic_topic_penalty_weight": round(self.generic_topic_penalty_weight, 4),
            "confusing_neighbor_penalty_weight": round(self.confusing_neighbor_penalty_weight, 4),
            "segment_weight": round(self.segment_weight, 4),
            "notes": list(self.notes),
        }


def resolve_benchmark_route_tuning(
    *,
    query_route: QueryRouteDecision,
    profile: QueryProfile,
    evidence_top_k: int,
    route_context: dict[str, Any] | None,
) -> BenchmarkRouteTuning:
    benchmark = str((route_context or {}).get("benchmark") or "").strip().lower()
    question_type = str((route_context or {}).get("question_type") or "").strip().lower()
    task_name = str((route_context or {}).get("task_name") or "").strip().lower()

    coarse_topk = max(16, evidence_top_k + min(18, max(8, evidence_top_k // 3)))
    fine_topk = max(8, min(16, max(6, evidence_top_k // 4)))
    dense_probe_k = max(coarse_topk, evidence_top_k + max(8, fine_topk))
    proxy_probe_k = max(coarse_topk, evidence_top_k + max(6, fine_topk // 2))
    sparse_probe_k = max(coarse_topk, evidence_top_k + max(8, fine_topk))
    rerank_pool_k = max(evidence_top_k + max(8, fine_topk), min(evidence_top_k * 2, evidence_top_k + 18))

    tuning = BenchmarkRouteTuning(
        benchmark=benchmark,
        route_profile="default",
        coarse_topk=coarse_topk,
        fine_topk=fine_topk,
        dense_probe_k=dense_probe_k,
        proxy_probe_k=proxy_probe_k,
        sparse_probe_k=sparse_probe_k,
        rerank_pool_k=rerank_pool_k,
        segment_rerank_topk=0,
        confusing_cluster_topk=0,
        prefer_object_shortcut=query_route.prefer_object_shortcut,
        prefer_identity_rerank=query_route.prefer_identity_rerank,
        prefer_segment_rerank=query_route.prefer_segment_rerank,
        prefer_confusing_cluster=query_route.prefer_confusing_cluster,
        identity_reward_weight=0.08 if query_route.prefer_identity_rerank else 0.0,
        wrong_entity_penalty_weight=0.08 if query_route.prefer_identity_rerank else 0.0,
        wrong_domain_penalty_weight=0.03 if query_route.prefer_identity_rerank else 0.0,
        wrong_role_target_penalty_weight=0.03 if query_route.prefer_identity_rerank else 0.0,
        wrong_subtheme_penalty_weight=0.03 if query_route.prefer_identity_rerank else 0.0,
        generic_topic_penalty_weight=0.02 if query_route.prefer_identity_rerank else 0.0,
        confusing_neighbor_penalty_weight=0.06 if query_route.prefer_confusing_cluster else 0.0,
        segment_weight=0.1 if query_route.prefer_segment_rerank else 0.0,
    )

    if benchmark == "longmemeval":
        tuning.route_profile = "longmemeval_guarded"
        tuning.segment_rerank_topk = 0
        tuning.confusing_cluster_topk = 0
        tuning.prefer_identity_rerank = bool(query_route.prefer_identity_rerank and profile.query_person_names)
        tuning.prefer_segment_rerank = False
        tuning.prefer_confusing_cluster = False
        tuning.identity_reward_weight = 0.06 if tuning.prefer_identity_rerank else 0.0
        tuning.wrong_entity_penalty_weight = 0.06 if tuning.prefer_identity_rerank else 0.0
        tuning.wrong_domain_penalty_weight = 0.0
        tuning.wrong_role_target_penalty_weight = 0.0
        tuning.wrong_subtheme_penalty_weight = 0.0
        tuning.generic_topic_penalty_weight = 0.0
        tuning.segment_weight = 0.0
        tuning.notes.append("guard_evidence_first")
        return tuning

    if benchmark == "locomo":
        tuning.route_profile = "locomo_local"
        tuning.segment_rerank_topk = 8 if (profile.needs_temporal_objects or profile.needs_exact_evidence) else 4
        tuning.confusing_cluster_topk = 10 if question_type in {"temporal", "temporal_inference"} else 0
        tuning.prefer_segment_rerank = tuning.segment_rerank_topk > 0
        tuning.prefer_identity_rerank = bool(query_route.prefer_identity_rerank or profile.query_person_names)
        tuning.prefer_confusing_cluster = tuning.confusing_cluster_topk > 0
        tuning.identity_reward_weight = 0.1 if tuning.prefer_identity_rerank else 0.0
        tuning.wrong_entity_penalty_weight = 0.08 if tuning.prefer_identity_rerank else 0.0
        tuning.wrong_domain_penalty_weight = 0.03 if tuning.prefer_identity_rerank else 0.0
        tuning.wrong_role_target_penalty_weight = 0.03 if tuning.prefer_identity_rerank else 0.0
        tuning.wrong_subtheme_penalty_weight = 0.03 if tuning.prefer_identity_rerank else 0.0
        tuning.generic_topic_penalty_weight = 0.02 if tuning.prefer_identity_rerank else 0.0
        tuning.segment_weight = 0.12 if tuning.prefer_segment_rerank else 0.0
        tuning.notes.extend(["favor_local_session_sort", "preserve_session_recall"])
        return tuning

    if benchmark == "convomem":
        context_size = int((route_context or {}).get("context_size") or 0)
        preference_like = question_type == "preference" or profile.needs_preference_objects
        temporal_like = question_type in {"changing", "implicit_connection"} or profile.needs_temporal_objects
        tuning.route_profile = "convomem_conservative"
        tuning.coarse_topk = max(18, evidence_top_k + 8)
        tuning.fine_topk = max(8, min(14, evidence_top_k // 5 or 8))
        tuning.dense_probe_k = max(tuning.coarse_topk or 0, evidence_top_k + 8)
        tuning.proxy_probe_k = max(tuning.coarse_topk or 0, evidence_top_k + 6)
        tuning.sparse_probe_k = max(tuning.coarse_topk or 0, evidence_top_k + 8)
        tuning.rerank_pool_k = max(
            evidence_top_k + 10,
            evidence_top_k + (tuning.fine_topk or 8),
            evidence_top_k + 12,
        )
        tuning.segment_rerank_topk = 10 if temporal_like else 6 if preference_like else 4
        tuning.confusing_cluster_topk = 10 if temporal_like and context_size >= 10 else 0
        tuning.prefer_object_shortcut = bool(preference_like and 0 < context_size <= 10)
        tuning.prefer_identity_rerank = bool(query_route.prefer_identity_rerank or profile.query_person_names or preference_like)
        tuning.prefer_segment_rerank = tuning.segment_rerank_topk > 0
        tuning.prefer_confusing_cluster = tuning.confusing_cluster_topk > 0
        tuning.identity_reward_weight = 0.1 if tuning.prefer_identity_rerank else 0.0
        tuning.wrong_entity_penalty_weight = 0.08 if tuning.prefer_identity_rerank else 0.0
        tuning.wrong_domain_penalty_weight = 0.03 if tuning.prefer_identity_rerank else 0.0
        tuning.wrong_role_target_penalty_weight = 0.03 if tuning.prefer_identity_rerank else 0.0
        tuning.wrong_subtheme_penalty_weight = 0.03 if tuning.prefer_identity_rerank else 0.0
        tuning.generic_topic_penalty_weight = 0.02 if tuning.prefer_identity_rerank else 0.0
        tuning.confusing_neighbor_penalty_weight = 0.08 if tuning.prefer_confusing_cluster else 0.0
        tuning.segment_weight = 0.14 if tuning.prefer_segment_rerank else 0.0
        tuning.notes.extend(["favor_conversation_locality", "preserve_message_evidence"])
        if context_size and context_size <= 4:
            tuning.notes.append("small_context_guard")
        if context_size >= 30:
            tuning.notes.append("large_context_shift")
        return tuning

    if benchmark == "knowme":
        tuning.route_profile = "knowme_object_first"
        tuning.coarse_topk = max(18, evidence_top_k + 10)
        tuning.fine_topk = max(8, min(16, evidence_top_k // 4))
        tuning.dense_probe_k = max(tuning.coarse_topk or 0, evidence_top_k + 10)
        tuning.proxy_probe_k = max(tuning.coarse_topk or 0, evidence_top_k + 8)
        tuning.sparse_probe_k = max(tuning.coarse_topk or 0, evidence_top_k + 10)
        tuning.rerank_pool_k = max(evidence_top_k + 12, evidence_top_k + (tuning.fine_topk or 8), evidence_top_k + 14)
        tuning.segment_rerank_topk = 16 if profile.needs_preference_objects or profile.needs_temporal_objects or task_name == "expert-annotated psychoanalysis" else 12
        tuning.confusing_cluster_topk = 16
        tuning.prefer_object_shortcut = bool(
            profile.needs_preference_objects
            or profile.needs_relation_objects
            or profile.needs_personal_context_objects
        )
        tuning.prefer_identity_rerank = True
        tuning.prefer_segment_rerank = True
        tuning.prefer_confusing_cluster = True
        tuning.identity_reward_weight = 0.14
        tuning.wrong_entity_penalty_weight = 0.14
        tuning.wrong_domain_penalty_weight = 0.05
        tuning.wrong_role_target_penalty_weight = 0.05
        tuning.wrong_subtheme_penalty_weight = 0.05
        tuning.generic_topic_penalty_weight = 0.035
        tuning.confusing_neighbor_penalty_weight = 0.1
        tuning.segment_weight = 0.2
        tuning.notes.extend(["boost_profile_state_binding", "favor_segment_precision", "favor_context_fidelity"])
        return tuning

    if benchmark == "clonemem":
        tuning.route_profile = "clonemem_identity"
        tuning.coarse_topk = max(18, evidence_top_k + 8)
        tuning.fine_topk = max(8, min(14, evidence_top_k // 5 or 8))
        tuning.dense_probe_k = max(tuning.coarse_topk or 0, evidence_top_k + 8)
        tuning.proxy_probe_k = max(tuning.coarse_topk or 0, evidence_top_k + 6)
        tuning.sparse_probe_k = max(tuning.coarse_topk or 0, evidence_top_k + 8)
        tuning.rerank_pool_k = max(evidence_top_k + 10, evidence_top_k + (tuning.fine_topk or 8), evidence_top_k + 12)
        tuning.segment_rerank_topk = 20 if question_type in {"trajectory", "pattern", "inference", "comparison"} else 14
        tuning.confusing_cluster_topk = 28
        explicit_object_query = bool(
            profile.needs_preference_objects
            or profile.needs_relation_objects
            or profile.needs_personal_context_objects
        )
        question_type_allows_shortcut = question_type in {"single_point_factual", "comparison", "inference"}
        tuning.prefer_object_shortcut = bool(explicit_object_query and question_type_allows_shortcut)
        tuning.prefer_identity_rerank = True
        tuning.prefer_segment_rerank = True
        tuning.prefer_confusing_cluster = True
        tuning.identity_reward_weight = 0.24
        tuning.wrong_entity_penalty_weight = 0.22
        tuning.wrong_domain_penalty_weight = 0.09
        tuning.wrong_role_target_penalty_weight = 0.09
        tuning.wrong_subtheme_penalty_weight = 0.09
        tuning.generic_topic_penalty_weight = 0.05
        tuning.confusing_neighbor_penalty_weight = 0.15
        tuning.segment_weight = 0.24
        tuning.notes.extend(["narrow_before_heavy_compute", "anti_confusion_priority", "favor_context_fidelity"])
        if not tuning.prefer_object_shortcut:
            tuning.notes.append("guard_shortcut_for_narrative_clone_queries")
        return tuning

    return tuning
