from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

from dysonspherain.utils.token_counter import TokenCounter
from sphere_cli.runtime import UnifiedMemoryRuntime
from sphere_cli.utils import lexical_score

from .config import load_runtime_config
from .events import MemoryEvent, stable_hash
from .ledger import replay_events
from .situation_graph import SituationGraph, rebuild_situation_graph


@dataclass(frozen=True)
class RecallIntent:
    intent_type: str
    project_scope: str | None = None
    temporal_scope: str | None = None
    artifact_scope: str | None = None
    code_scope: str | None = None
    required_evidence_types: list[str] = field(default_factory=list)
    forbidden_evidence_types: list[str] = field(default_factory=list)
    freshness_level: int = 3
    precision_vs_recall: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceOperatorSpec:
    op: str
    weight: float = 1.0
    limit: int = 12
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceProgram:
    program_id: str
    intent: RecallIntent
    operators: list[EvidenceOperatorSpec]
    merge_policy: str
    budget_policy: str
    safety_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "intent": self.intent.to_dict(),
            "operators": [op.to_dict() for op in self.operators],
            "merge_policy": self.merge_policy,
            "budget_policy": self.budget_policy,
            "safety_policy": self.safety_policy,
        }


@dataclass(frozen=True)
class EvidenceCandidate:
    candidate_id: str
    evidence_type: str
    text: str
    source_event_ids: list[str]
    source_node_ids: list[str]
    timestamp: str
    scores: dict[str, float]
    token_cost: int
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_recall_intent(query: str) -> RecallIntent:
    lowered = query.lower()
    if any(token in lowered for token in ("继续", "resume", "continue", "上次", "中断")):
        return RecallIntent("continue_task", required_evidence_types=["Task", "Decision", "Constraint", "Failure", "Patch"], freshness_level=5, precision_vs_recall=0.7)
    if any(token in lowered for token in ("regression", "benchmark", "clonemem", "knowme", "longmemeval", "locomo", "回归", "得分")):
        return RecallIntent("debug_regression", required_evidence_types=["Regression", "Metric", "BenchmarkRun", "Failure", "Patch", "Decision"], freshness_level=4, precision_vs_recall=0.65)
    if any(token in lowered for token in ("decision", "决定", "策略", "之前")):
        return RecallIntent("find_prior_decision", required_evidence_types=["Decision", "Constraint"], freshness_level=3, precision_vs_recall=0.75)
    if any(token in lowered for token in ("paper", "draft", "论文", "revision")):
        return RecallIntent("write_paper_revision", required_evidence_types=["Decision", "Artifact", "BenchmarkRun", "Metric"], freshness_level=3, precision_vs_recall=0.55)
    return RecallIntent("summarize_project_state", required_evidence_types=["Task", "Decision", "Constraint", "Failure", "OpenQuestion"], freshness_level=3, precision_vs_recall=0.6)


def compile_evidence_program(intent: RecallIntent) -> EvidenceProgram:
    if intent.intent_type == "debug_regression":
        ops = [
            EvidenceOperatorSpec("dense_semantic_search", 0.75),
            EvidenceOperatorSpec("metric_delta_scan", 1.0),
            EvidenceOperatorSpec("recent_event_scan", 0.9),
            EvidenceOperatorSpec("failure_lookup", 0.85),
            EvidenceOperatorSpec("patch_lookup", 0.75),
            EvidenceOperatorSpec("artifact_lookup", 0.7),
            EvidenceOperatorSpec("code_region_lookup", 0.65),
            EvidenceOperatorSpec("decision_lookup", 0.65),
            EvidenceOperatorSpec("constraint_lookup", 0.6),
            EvidenceOperatorSpec("causal_neighbor_expand", 0.55),
        ]
    elif intent.intent_type in {"continue_task", "recover_interrupted_work"}:
        ops = [
            EvidenceOperatorSpec("recent_event_scan", 1.0),
            EvidenceOperatorSpec("similar_task_lookup", 0.9),
            EvidenceOperatorSpec("decision_lookup", 0.85),
            EvidenceOperatorSpec("constraint_lookup", 0.8),
            EvidenceOperatorSpec("failure_lookup", 0.75),
            EvidenceOperatorSpec("patch_lookup", 0.65),
        ]
    elif intent.intent_type == "find_prior_decision":
        ops = [EvidenceOperatorSpec("decision_lookup", 1.0), EvidenceOperatorSpec("constraint_lookup", 0.75), EvidenceOperatorSpec("temporal_window_scan", 0.55), EvidenceOperatorSpec("contradiction_scan", 0.5)]
    else:
        ops = [EvidenceOperatorSpec("recent_event_scan", 0.85), EvidenceOperatorSpec("lexical_exact_search", 0.75), EvidenceOperatorSpec("decision_lookup", 0.65), EvidenceOperatorSpec("constraint_lookup", 0.65)]
    program_id = f"program_{stable_hash([intent.to_dict(), [op.to_dict() for op in ops]])[:16]}"
    return EvidenceProgram(program_id, intent, ops, "causal_diverse_rrf", "high_provenance_low_redundancy", "preserve_user_constraints")


def _event_text(event: MemoryEvent) -> str:
    payload = event.payload
    return "\n".join(
        str(value)
        for value in (
            payload.get("title"),
            payload.get("summary"),
            payload.get("content"),
            payload.get("message"),
            payload.get("path"),
            event.event_type,
        )
        if value
    )


OP_EVENT_TYPES = {
    "recent_event_scan": None,
    "temporal_window_scan": None,
    "lexical_exact_search": None,
    "dense_semantic_search": None,
    "decision_lookup": {"decision_made"},
    "constraint_lookup": {"constraint_added", "constraint_changed", "preference_declared"},
    "artifact_lookup": {"artifact_created", "artifact_updated"},
    "metric_delta_scan": {"metric_changed", "benchmark_finished", "regression_detected"},
    "failure_lookup": {"failure_observed", "regression_detected"},
    "patch_lookup": {"patch_applied", "file_changed", "recovery_attempted"},
    "code_region_lookup": {"file_changed", "patch_applied"},
    "hypothesis_lookup": {"hypothesis_created"},
    "similar_task_lookup": {"user_instruction_received"},
    "causal_neighbor_expand": None,
    "contradiction_scan": {"decision_made", "constraint_changed", "failure_observed"},
    "user_preference_scan": {"preference_declared", "constraint_added"},
}


def _token_vector(text: str) -> dict[str, float]:
    vector: dict[str, float] = {}
    for token in text.lower().replace("_", " ").replace("/", " ").replace(".", " ").split():
        if len(token) < 2:
            continue
        vector[token] = vector.get(token, 0.0) + 1.0
    return vector


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _freshness_score(index_from_newest: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return max(0.05, 1.0 - (index_from_newest / max(1, total - 1)))


def _graph_neighbor_event_ids(graph: SituationGraph) -> set[str]:
    ids: set[str] = set()
    for edge in graph.edges:
        ids.update(edge.source_event_ids)
    return ids


def run_operator(query: str, op: EvidenceOperatorSpec, events: list[MemoryEvent], graph: SituationGraph) -> list[EvidenceCandidate]:
    allowed = OP_EVENT_TYPES.get(op.op)
    candidates: list[EvidenceCandidate] = []
    counter = TokenCounter()
    query_vector = _token_vector(query)
    neighbor_event_ids = _graph_neighbor_event_ids(graph)
    ordered = list(reversed(events))
    for idx, event in enumerate(ordered):
        if allowed is not None and event.event_type not in allowed:
            continue
        if op.op == "causal_neighbor_expand" and event.event_id not in neighbor_event_ids:
            continue
        text = _event_text(event)
        if not text.strip():
            continue
        dense_score = _cosine(query_vector, _token_vector(text))
        relevance = lexical_score(query, text)
        if op.op == "dense_semantic_search":
            relevance = max(relevance, dense_score)
        elif op.op in {"recent_event_scan", "temporal_window_scan"}:
            relevance = max(0.1, relevance)
        elif op.op in {"artifact_lookup", "metric_delta_scan", "failure_lookup", "patch_lookup", "code_region_lookup", "hypothesis_lookup", "similar_task_lookup", "causal_neighbor_expand"}:
            relevance = max(0.12, relevance)
        if op.op in {"lexical_exact_search", "dense_semantic_search"} and relevance <= 0:
            continue
        token_cost = counter.count(text).tokens
        freshness = _freshness_score(idx, len(ordered))
        provenance_strength = 0.9 if event.provenance else 0.7 if event.source else 0.45
        contradiction_risk = 0.0
        lowered = text.lower()
        if any(word in lowered for word in ("contradict", "conflict", "superseded", "invalidated", "过期", "冲突")):
            contradiction_risk = 0.85 if op.op == "contradiction_scan" else 0.55
        scores = {
            "operator_weight": float(op.weight),
            "task_utility": min(1.0, relevance + 0.25),
            "freshness": freshness,
            "provenance_strength": provenance_strength,
            "diversity_contribution": 0.5,
            "redundancy_penalty": 0.0,
            "contradiction_risk": contradiction_risk,
            "user_constraint_priority": 1.0 if event.event_type in {"constraint_added", "preference_declared"} else 0.0,
            "dense_semantic": dense_score,
            "lexical": lexical_score(query, text),
        }
        candidates.append(
            EvidenceCandidate(
                candidate_id=f"cand_{stable_hash([op.op, event.event_id])[:18]}",
                evidence_type=event.event_type,
                text=text,
                source_event_ids=[event.event_id],
                source_node_ids=[],
                timestamp=event.timestamp,
                scores=scores,
                token_cost=token_cost,
                provenance={
                    "operator": op.op,
                    "source": event.source,
                    "event_type": event.event_type,
                    "backend": "ledger_token_cosine" if op.op == "dense_semantic_search" else "event_projection",
                    "fallback_in_use": False,
                    **event.provenance,
                },
            )
        )
        if len(candidates) >= op.limit:
            break
    return candidates


def run_dense_vector_operator(base_dir: Path, query: str, op: EvidenceOperatorSpec, *, project: str) -> tuple[list[EvidenceCandidate], dict[str, Any]]:
    counter = TokenCounter()
    try:
        runtime = UnifiedMemoryRuntime.from_base_dir(base_dir, config_overrides={"project_name": project})
        info = runtime.services.vector_store.info()
        rows = runtime.services.vector_store.search(query, top_k=op.limit)
        candidates: list[EvidenceCandidate] = []
        for row in rows:
            text = str(row.get("document") or "")
            if not text.strip():
                continue
            metadata = dict(row.get("metadata") or {})
            chunk_id = str(row.get("chunk_id") or metadata.get("chunk_id") or stable_hash(text)[:16])
            distance = float(row.get("distance") if row.get("distance") is not None else row.get("similarity") or 0.0)
            relevance = max(0.0, 1.0 - distance) if distance else float(row.get("similarity") or 0.5)
            candidates.append(
                EvidenceCandidate(
                    candidate_id=f"cand_vec_{stable_hash([op.op, chunk_id])[:18]}",
                    evidence_type="dense_vector_hit",
                    text=text,
                    source_event_ids=[],
                    source_node_ids=[str(metadata.get("node_id") or "")],
                    timestamp=str(metadata.get("updated_at") or metadata.get("created_at") or ""),
                    scores={
                        "operator_weight": float(op.weight),
                        "task_utility": min(1.0, relevance + 0.2),
                        "freshness": 0.6,
                        "provenance_strength": 0.75,
                        "diversity_contribution": 0.55,
                        "redundancy_penalty": 0.0,
                        "contradiction_risk": 0.0,
                        "user_constraint_priority": 0.0,
                        "dense_semantic": relevance,
                        "lexical": lexical_score(query, text),
                    },
                    token_cost=counter.count(text).tokens,
                    provenance={
                        "operator": op.op,
                        "backend": "project_vector_store",
                        "fallback_in_use": bool(info.get("fallback_in_use") or info.get("vector_fallback_in_use")),
                        "embedding_provider": info.get("embedding_provider"),
                        "embedding_model": info.get("embedding_model"),
                        "vector_backend": info.get("vector_backend"),
                        "chunk_id": chunk_id,
                    },
                )
            )
        return candidates, {"backend": "project_vector_store", "candidate_count": len(candidates), "vector_info": info}
    except Exception as exc:
        return [], {"backend": "project_vector_store", "candidate_count": 0, "error": str(exc)}


def run_evidence_program(base_dir: Path, query: str, program: EvidenceProgram, *, project: str = "DysonSpherain") -> tuple[list[EvidenceCandidate], dict[str, Any]]:
    events = replay_events(base_dir, project=project)
    graph = rebuild_situation_graph(events, project=project)
    config = load_runtime_config(base_dir)
    enabled = set(config.enabled_operators)
    all_candidates: dict[str, EvidenceCandidate] = {}
    trace: dict[str, Any] = {
        "program": program.to_dict(),
        "operators": [],
        "embedding_backend": config.embedding_backend,
        "fallback_in_use": False,
        "index_freshness": "fresh" if events else "empty",
    }
    for op in program.operators:
        if op.op not in enabled and op.op not in {"temporal_window_scan", "causal_neighbor_expand", "contradiction_scan", "similar_task_lookup", "dense_semantic_search"}:
            trace["operators"].append({"op": op.op, "candidate_count": 0, "weight": op.weight, "skipped": "disabled_by_config"})
            continue
        weight = float(config.operator_weights.get(op.op, op.weight))
        configured = EvidenceOperatorSpec(op.op, weight=weight, limit=op.limit, params=op.params)
        vector_trace: dict[str, Any] | None = None
        if configured.op == "dense_semantic_search" and config.embedding_backend in {"project_vector_store", "auto", "existing_project_backend"}:
            results, vector_trace = run_dense_vector_operator(base_dir, query, configured, project=project)
            if not results:
                fallback_results = run_operator(query, configured, events, graph)
                results = fallback_results
                vector_trace = {**(vector_trace or {}), "fallback_backend": "ledger_token_cosine", "fallback_reason": (vector_trace or {}).get("error") or "empty_vector_results"}
        else:
            results = run_operator(query, configured, events, graph)
        for candidate in results:
            all_candidates.setdefault(candidate.candidate_id, candidate)
        trace["operators"].append({"op": op.op, "candidate_count": len(results), "weight": weight, "backend": results[0].provenance.get("backend") if results else "event_projection", **({"vector_trace": vector_trace} if vector_trace else {})})
    ranked = sorted(
        all_candidates.values(),
        key=lambda candidate: (
            candidate.scores.get("task_utility", 0.0) * candidate.scores.get("operator_weight", 1.0)
            + candidate.scores.get("user_constraint_priority", 0.0),
            candidate.timestamp,
        ),
        reverse=True,
    )
    trace["candidate_count"] = len(ranked)
    return ranked, trace
