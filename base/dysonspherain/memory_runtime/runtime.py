from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .context_compiler import compile_context, render_context_packet
from .config import load_runtime_config
from .events import build_event
from .evidence_vm import compile_evidence_program, infer_recall_intent, run_evidence_program
from .ledger import append_event, replay_events, write_projection
from .recall_audit import audit_context_packet
from .scheduler import run_maintenance_job, schedule_memory_maintenance
from .situation_graph import build_and_save_graph


def append_from_file(base_dir: Path, file_path: Path, *, source: str = "manual", project: str = "DysonSpherain") -> dict[str, Any]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    event = build_event(
        event_type=str(payload.get("event_type") or "user_instruction_received"),
        payload=dict(payload.get("payload") or payload),
        source=str(payload.get("source") or source),
        actor=str(payload.get("actor") or "user"),
        project=str(payload.get("project") or project),
        session_id=str(payload.get("session_id") or "") or None,
        parent_event_id=str(payload.get("parent_event_id") or "") or None,
        timestamp=str(payload.get("timestamp") or "") or None,
        provenance=dict(payload.get("provenance") or {"file": str(file_path)}),
    )
    return append_event(base_dir, event).to_dict()


def recall_runtime(base_dir: Path, query: str, *, project: str = "DysonSpherain", budget: int = 1200, trace: bool = False) -> dict[str, Any]:
    intent = infer_recall_intent(query)
    program = compile_evidence_program(intent)
    candidates, router_trace = run_evidence_program(base_dir, query, program, project=project)
    packet = compile_context(query=query, intent=intent, candidates=candidates, budget_tokens=budget, trace={"router": router_trace})
    audit = audit_context_packet(packet)
    if audit.risk_level in {"medium", "high"} and audit.suggested_followup_ops:
        followup_program = type(program)(
            program_id=program.program_id + "_followup",
            intent=program.intent,
            operators=audit.suggested_followup_ops,
            merge_policy=program.merge_policy,
            budget_policy=program.budget_policy,
            safety_policy=program.safety_policy,
        )
        more_candidates, followup_trace = run_evidence_program(base_dir, query, followup_program, project=project)
        packet = compile_context(
            query=query,
            intent=intent,
            candidates=[*candidates, *more_candidates],
            budget_tokens=budget,
            trace={"router": router_trace, "followup": followup_trace},
        )
        audit = audit_context_packet(packet)
    write_projection(base_dir, "latest_context_packet.json", packet.to_dict())
    write_projection(base_dir, "latest_recall_audit.json", audit.to_dict())
    append_event(
        base_dir,
        build_event(
            event_type="retrieval_performed",
            payload={"query": query, "packet_id": packet.packet_id, "intent": intent.to_dict(), "candidate_count": len(candidates)},
            source="memory_runtime",
            actor="system",
            project=project,
            provenance={"trace_saved": "data/projections/latest_context_packet.json"},
        ),
    )
    append_event(
        base_dir,
        build_event(
            event_type="context_packet_compiled",
            payload={"packet_id": packet.packet_id, "used_tokens": packet.used_tokens, "budget_tokens": packet.budget_tokens},
            source="memory_runtime",
            actor="system",
            project=project,
            provenance={"trace_saved": "data/projections/latest_context_packet.json"},
        ),
    )
    result = {"status": "ok", "packet": packet.to_dict(), "audit": audit.to_dict(), "rendered_context": render_context_packet(packet)}
    if trace:
        result["trace"] = {"router": router_trace, "compiler": packet.compiler_trace}
    return result


def compact_safe(base_dir: Path, *, project: str = "DysonSpherain") -> dict[str, Any]:
    events = replay_events(base_dir, project=project)
    jobs = schedule_memory_maintenance("session_ended", [event.event_id for event in events[-20:]])
    results = [run_maintenance_job(base_dir, job, project=project).to_dict() for job in jobs]
    return {"status": "ok", "job_count": len(jobs), "results": results}


def graph_state(base_dir: Path, *, project: str = "DysonSpherain") -> dict[str, Any]:
    graph = build_and_save_graph(base_dir, project=project)
    return graph.to_dict()


def cockpit_snapshot(base_dir: Path, *, project: str = "DysonSpherain") -> dict[str, Any]:
    events = replay_events(base_dir, project=project)
    graph = build_and_save_graph(base_dir, project=project)
    packet_path = base_dir / "data" / "projections" / "latest_context_packet.json"
    audit_path = base_dir / "data" / "projections" / "latest_recall_audit.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8")) if packet_path.exists() else {"status": "empty"}
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"status": "empty"}
    config = load_runtime_config(base_dir).to_dict()
    node_counts: dict[str, int] = {}
    for node in graph.nodes:
        node_counts[node.node_type] = node_counts.get(node.node_type, 0) + 1
    latest_events = [event.to_dict() for event in events[-20:]]
    active_tasks = [node for node in graph.nodes if node.node_type == "Task" and node.status != "done"][-5:]
    constraints = [node for node in graph.nodes if node.node_type in {"Constraint", "UserPreference"}][-8:]
    regressions = [node for node in graph.nodes if node.node_type == "Regression"][-8:]
    return {
        "status": "ok",
        "project": project,
        "mission_control": {
            "event_count": len(events),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "memory_health": "attention" if audit.get("risk_level") in {"medium", "high"} else "ok",
            "index_freshness": "fresh" if events else "empty",
            "active_tasks": [node.__dict__ for node in active_tasks],
            "active_constraints": [node.__dict__ for node in constraints],
            "open_regressions": [node.__dict__ for node in regressions],
            "node_counts": node_counts,
        },
        "ledger": {"event_count": len(events), "latest_events": latest_events},
        "graph": graph.to_dict(),
        "packet": packet,
        "audit": audit,
        "config": config,
    }
