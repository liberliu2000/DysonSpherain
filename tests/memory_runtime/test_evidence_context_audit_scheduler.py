from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dysonspherain.memory_runtime.context_compiler import compile_context
from dysonspherain.memory_runtime.events import build_event
from dysonspherain.memory_runtime.evidence_vm import EvidenceOperatorSpec, compile_evidence_program, infer_recall_intent, run_dense_vector_operator, run_evidence_program, run_operator
from dysonspherain.memory_runtime.ledger import append_event, replay_events
from dysonspherain.memory_runtime.recall_audit import audit_context_packet
from dysonspherain.memory_runtime.scheduler import enqueue_maintenance_jobs, index_ledger_events, load_pending_jobs, run_maintenance_job, run_scheduler_daemon, run_scheduler_once, schedule_memory_maintenance
from dysonspherain.memory_runtime.situation_graph import rebuild_situation_graph


class EvidenceContextAuditSchedulerTests(unittest.TestCase):
    def _seed(self, base: Path) -> None:
        for event in [
            build_event(event_type="user_instruction_received", payload={"content": "Debug CloneMem regression"}, source="test", actor="user", timestamp="2026-04-30T00:00:00+00:00"),
            build_event(event_type="constraint_added", payload={"content": "Do not disable dense preserve or safe fusion"}, source="test", actor="user", timestamp="2026-04-30T00:01:00+00:00"),
            build_event(event_type="regression_detected", payload={"summary": "CloneMem candidate_recall dropped"}, source="test", actor="system", timestamp="2026-04-30T00:02:00+00:00"),
            build_event(event_type="patch_applied", payload={"summary": "Added route-aware admission guard", "path": "base/sphere_cli/evidence_pipeline.py"}, source="test", actor="assistant", timestamp="2026-04-30T00:03:00+00:00"),
            build_event(event_type="artifact_updated", payload={"summary": "Updated clonemem_candidate_recall.json", "artifact_id": "candidate_recall"}, source="test", actor="system", timestamp="2026-04-30T00:04:00+00:00"),
        ]:
            append_event(base, event)

    def test_evidence_program_compilation(self) -> None:
        intent = infer_recall_intent("debug CloneMem benchmark regression")
        program = compile_evidence_program(intent)
        self.assertEqual(intent.intent_type, "debug_regression")
        self.assertTrue(any(op.op == "metric_delta_scan" for op in program.operators))
        self.assertTrue(any(op.op == "artifact_lookup" for op in program.operators))

    def test_operator_backends_return_distinct_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed(base)
            events = replay_events(base)
            graph = rebuild_situation_graph(events)
            dense = run_operator("candidate recall", EvidenceOperatorSpec("dense_semantic_search", 1.0), events, graph)
            artifact = run_operator("candidate recall", EvidenceOperatorSpec("artifact_lookup", 1.0), events, graph)
            self.assertTrue(dense)
            self.assertTrue(artifact)
            self.assertEqual(dense[0].provenance["backend"], "ledger_token_cosine")

    def test_dense_vector_operator_reports_project_vector_backend_or_explicit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed(base)
            candidates, trace = run_dense_vector_operator(base, "candidate recall", EvidenceOperatorSpec("dense_semantic_search", 1.0), project="DysonSpherain")
            self.assertIn(trace["backend"], {"project_vector_store"})
            if candidates:
                self.assertEqual(candidates[0].provenance["backend"], "project_vector_store")
            else:
                self.assertEqual(trace["candidate_count"], 0)
                self.assertIn("vector_info", trace)

    def test_context_budget_compiler_respects_budget_and_omits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed(base)
            intent = infer_recall_intent("debug CloneMem benchmark regression")
            program = compile_evidence_program(intent)
            candidates, trace = run_evidence_program(base, "debug CloneMem benchmark regression", program)
            packet = compile_context(query="debug CloneMem benchmark regression", intent=intent, candidates=candidates, budget_tokens=35, trace=trace)
            self.assertLessEqual(packet.used_tokens, packet.budget_tokens)
            self.assertGreater(len(packet.omitted_candidates), 0)
            self.assertIn("section_allocation", packet.compiler_trace)

    def test_recall_audit_detects_missing_constraints(self) -> None:
        intent = infer_recall_intent("find prior decision")
        packet = compile_context(query="find prior decision", intent=intent, candidates=[], budget_tokens=100)
        audit = audit_context_packet(packet)
        self.assertIn(audit.risk_level, {"medium", "high"})
        self.assertTrue(audit.suggested_followup_ops)
        self.assertTrue(any(check.name == "freshness_check" for check in audit.checks))

    def test_scheduler_job_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed(base)
            job = schedule_memory_maintenance("benchmark_finished", ["evt1"])[0]
            first = run_maintenance_job(base, job)
            second = run_maintenance_job(base, job)
            self.assertEqual(first.status, "ok")
            self.assertEqual(second.status, "ok")
            self.assertGreaterEqual(len(replay_events(base)), 4)
            self.assertTrue((base / "data" / "maintenance" / "jobs.jsonl").exists())
            self.assertTrue((base / "data" / "projections" / "metric_deltas.json").exists())

    def test_scheduler_queue_and_ledger_vector_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed(base)
            event_ids = [event.event_id for event in replay_events(base)[:2]]
            jobs = enqueue_maintenance_jobs(base, "artifact_updated", event_ids)
            self.assertGreaterEqual(len(jobs), 2)
            self.assertEqual(len(load_pending_jobs(base)), len(jobs))
            report = index_ledger_events(base, event_ids)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["attempted_event_count"], 2)
            drained = run_scheduler_once(base)
            self.assertEqual(drained["status"], "ok")
            self.assertEqual(load_pending_jobs(base), [])
            self.assertTrue((base / "data" / "projections" / "ledger_vector_index_report.json").exists())

    def test_scheduler_daemon_can_run_bounded_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed(base)
            enqueue_maintenance_jobs(base, "benchmark_finished", ["evt1"])
            result = run_scheduler_daemon(base, interval_seconds=0.1, max_loops=1)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["loops"], 1)
            self.assertGreaterEqual(result["total_ran"], 1)


if __name__ == "__main__":
    unittest.main()
