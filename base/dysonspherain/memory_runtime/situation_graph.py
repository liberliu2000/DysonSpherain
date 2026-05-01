from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

from .events import MemoryEvent, stable_hash
from .ledger import replay_events, write_projection


NODE_TYPES = {
    "Task",
    "Subtask",
    "Decision",
    "Constraint",
    "Hypothesis",
    "BenchmarkRun",
    "Metric",
    "Regression",
    "Patch",
    "Artifact",
    "CodeRegion",
    "Failure",
    "RecoveryAction",
    "UserPreference",
    "OpenQuestion",
}

EDGE_TYPES = {
    "depends_on",
    "caused_by",
    "fixed_by",
    "introduced_by",
    "supports",
    "contradicts",
    "supersedes",
    "derived_from",
    "validated_by",
    "invalidated_by",
    "blocks",
    "unblocks",
    "reopens",
    "similar_to",
}


@dataclass
class SituationNode:
    node_id: str
    node_type: str
    title: str
    summary: str
    status: str
    created_at: str
    updated_at: str
    source_event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SituationEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    confidence: float
    source_event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SituationGraph:
    project: str
    nodes: list[SituationNode]
    edges: list[SituationEdge]
    source_event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "source_event_count": self.source_event_count,
        }


@dataclass(frozen=True)
class GraphMutation:
    mutation_type: str
    node_id: str | None = None
    edge_id: str | None = None
    event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EVENT_NODE_MAP = {
    "user_instruction_received": "Task",
    "decision_made": "Decision",
    "constraint_added": "Constraint",
    "constraint_changed": "Constraint",
    "preference_declared": "UserPreference",
    "hypothesis_created": "Hypothesis",
    "benchmark_started": "BenchmarkRun",
    "benchmark_finished": "BenchmarkRun",
    "metric_changed": "Metric",
    "regression_detected": "Regression",
    "patch_applied": "Patch",
    "artifact_created": "Artifact",
    "artifact_updated": "Artifact",
    "file_changed": "CodeRegion",
    "failure_observed": "Failure",
    "recovery_attempted": "RecoveryAction",
}


def _title(event: MemoryEvent) -> str:
    payload = event.payload
    return str(payload.get("title") or payload.get("task") or payload.get("summary") or payload.get("path") or event.event_type).strip()[:120]


def _summary(event: MemoryEvent) -> str:
    payload = event.payload
    return str(payload.get("summary") or payload.get("content") or payload.get("message") or json.dumps(payload, ensure_ascii=False, sort_keys=True))[:1200]


def _node_id(event: MemoryEvent, node_type: str) -> str:
    explicit = event.payload.get("node_id") or event.payload.get("artifact_id") or event.payload.get("memory_id")
    if explicit:
        return f"{node_type.lower()}_{stable_hash([event.project, node_type, explicit])[:16]}"
    return f"{node_type.lower()}_{event.content_hash[:16]}"


def _edge(edges: dict[str, SituationEdge], source: str, target: str, edge_type: str, event: MemoryEvent, confidence: float = 0.8, metadata: dict[str, Any] | None = None) -> None:
    if not source or not target:
        return
    edge_id = f"edge_{stable_hash([source, target, edge_type])[:16]}"
    edges[edge_id] = SituationEdge(edge_id, source, target, edge_type, confidence, [event.event_id], metadata or {})


def _explicit_relation_edges(edges: dict[str, SituationEdge], node_id: str, event: MemoryEvent) -> None:
    relation_map = {
        "depends_on": "depends_on",
        "caused_by": "caused_by",
        "fixed_by": "fixed_by",
        "introduced_by": "introduced_by",
        "supports": "supports",
        "contradicts": "contradicts",
        "supersedes": "supersedes",
        "derived_from": "derived_from",
        "validated_by": "validated_by",
        "invalidated_by": "invalidated_by",
        "blocks": "blocks",
        "unblocks": "unblocks",
        "reopens": "reopens",
        "similar_to": "similar_to",
    }
    for key, edge_type in relation_map.items():
        value = event.payload.get(key)
        if not value:
            continue
        targets = value if isinstance(value, list) else [value]
        for target in targets:
            _edge(edges, node_id, str(target), edge_type, event, 0.9, {"explicit_payload_key": key})


def rebuild_situation_graph(events: Iterable[MemoryEvent], *, project: str = "DysonSpherain") -> SituationGraph:
    event_list = list(events)
    node_by_id: dict[str, SituationNode] = {}
    edges: dict[str, SituationEdge] = {}
    last_task_id = ""
    for event in sorted([item for item in event_list if item.project == project], key=lambda item: (item.timestamp, item.event_id)):
        node_type = EVENT_NODE_MAP.get(event.event_type)
        if not node_type:
            continue
        node_id = _node_id(event, node_type)
        existing = node_by_id.get(node_id)
        if existing:
            existing.updated_at = max(existing.updated_at, event.timestamp)
            if event.event_id not in existing.source_event_ids:
                existing.source_event_ids.append(event.event_id)
            existing.summary = _summary(event) or existing.summary
        else:
            node_by_id[node_id] = SituationNode(
                node_id=node_id,
                node_type=node_type,
                title=_title(event),
                summary=_summary(event),
                status=str(event.payload.get("status") or "current"),
                created_at=event.timestamp,
                updated_at=event.timestamp,
                source_event_ids=[event.event_id],
                metadata={**event.provenance, **{k: v for k, v in event.payload.items() if k not in {"content", "summary", "message"}}},
            )
        if node_type == "Task":
            last_task_id = node_id
        elif last_task_id and node_type in {"Decision", "Constraint", "Regression", "Patch", "Failure", "RecoveryAction", "BenchmarkRun", "Metric", "Artifact", "CodeRegion", "Hypothesis"}:
            edge_type = "depends_on" if node_type in {"Decision", "Constraint"} else "derived_from"
            if node_type == "Patch":
                edge_type = "fixed_by"
            if node_type == "Failure":
                edge_type = "blocks"
            if node_type == "Regression":
                edge_type = "caused_by"
            if node_type == "RecoveryAction":
                edge_type = "unblocks"
            if node_type == "Metric":
                edge_type = "validated_by"
            _edge(edges, last_task_id, node_id, edge_type, event, 0.75)
        _explicit_relation_edges(edges, node_id, event)
    nodes = sorted(node_by_id.values(), key=lambda node: (node.node_type, node.updated_at, node.node_id))
    return SituationGraph(project=project, nodes=nodes, edges=sorted(edges.values(), key=lambda edge: edge.edge_id), source_event_count=len(event_list))


def update_situation_graph(event: MemoryEvent, graph: SituationGraph) -> list[GraphMutation]:
    rebuilt = rebuild_situation_graph([event], project=graph.project)
    existing_nodes = {node.node_id for node in graph.nodes}
    existing_edges = {edge.edge_id for edge in graph.edges}
    mutations: list[GraphMutation] = []
    for node in rebuilt.nodes:
        mutations.append(GraphMutation("node_updated" if node.node_id in existing_nodes else "node_added", node_id=node.node_id, event_id=event.event_id))
    for edge in rebuilt.edges:
        mutations.append(GraphMutation("edge_updated" if edge.edge_id in existing_edges else "edge_added", edge_id=edge.edge_id, event_id=event.event_id, metadata={"edge_type": edge.edge_type}))
    return mutations


def build_and_save_graph(base_dir: Path, *, project: str = "DysonSpherain") -> SituationGraph:
    events = replay_events(base_dir, project=project)
    graph = rebuild_situation_graph(events, project=project)
    write_projection(base_dir, "task_situation_graph.json", graph.to_dict())
    latest = {
        "project": project,
        "active_tasks": [asdict(node) for node in graph.nodes if node.node_type == "Task" and node.status != "done"][-5:],
        "active_constraints": [asdict(node) for node in graph.nodes if node.node_type == "Constraint" and node.status == "current"][-10:],
        "open_regressions": [asdict(node) for node in graph.nodes if node.node_type == "Regression" and node.status != "resolved"][-10:],
        "recent_decisions": [asdict(node) for node in graph.nodes if node.node_type == "Decision"][-10:],
    }
    write_projection(base_dir, "latest_project_state.json", latest)
    return graph
