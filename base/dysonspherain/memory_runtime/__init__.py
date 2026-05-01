"""Event-sourced memory runtime for DysonSpherain.

The runtime is intentionally backend-light: JSONL is used for the first
append-only ledger implementation, while projections and packets remain
deterministically rebuildable from ledger events.
"""

from .context_compiler import ContextPacket, compile_context
from .evidence_vm import EvidenceProgram, RecallIntent, compile_evidence_program, run_evidence_program
from .ledger import AppendResult, append_event, replay_events
from .recall_audit import RecallAudit, audit_context_packet
from .scheduler import MaintenanceJob, enqueue_maintenance_jobs, load_pending_jobs, run_scheduler_daemon, run_scheduler_once, schedule_memory_maintenance
from .situation_graph import SituationGraph, rebuild_situation_graph

__all__ = [
    "AppendResult",
    "ContextPacket",
    "EvidenceProgram",
    "MaintenanceJob",
    "RecallAudit",
    "RecallIntent",
    "SituationGraph",
    "append_event",
    "audit_context_packet",
    "compile_context",
    "compile_evidence_program",
    "enqueue_maintenance_jobs",
    "load_pending_jobs",
    "rebuild_situation_graph",
    "replay_events",
    "run_evidence_program",
    "run_scheduler_daemon",
    "run_scheduler_once",
    "schedule_memory_maintenance",
]
