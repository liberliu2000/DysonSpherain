from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.adapters.daemon import DysonMemoryHandler
from dysonspherain.memory_os.observation_store import write_observation, write_token_economy_event
from dysonspherain.memory_runtime.events import build_event
from dysonspherain.memory_runtime.ledger import append_event
from dysonspherain.memory_runtime.runtime import recall_runtime
from dysonspherain.product import remember


class DaemonApiTests(unittest.TestCase):
    def test_health_search_observation_and_web_ui(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = remember(root, project_id="DysonSpherain", evidence_type="decision", text="daemon maintenance duplicate", tags=["daemon"])
            remember(root, project_id="DysonSpherain", evidence_type="decision", text="daemon maintenance duplicate", tags=["daemon"])
            write_observation(root, project="DysonSpherain", kind="note", title="daemon note", content="token economy daemon search", source="unit")
            write_token_economy_event(
                root,
                project="DysonSpherain",
                session_id="s1",
                prompt="daemon token savings",
                decision="inject",
                injected_tokens=100,
                baseline_context_tokens=1000,
                estimated_saved_tokens=900,
                budget_usage_ratio=0.0625,
                local_compute_economy={"embedding_cache_hit_count": 3, "embedding_cache_miss_count": 1, "estimated_local_runtime_saved_ms": 80},
            )
            append_event(
                root,
                build_event(
                    event_type="user_instruction_received",
                    payload={"content": "Continue cockpit UI upgrade"},
                    source="unit",
                    actor="user",
                    timestamp="2026-04-30T00:00:00+00:00",
                ),
            )
            recall_runtime(root, "continue cockpit UI upgrade", budget=500)
            handler = type("TestHandler", (DysonMemoryHandler,), {"base_dir": root, "project": "DysonSpherain"})
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                health = json.loads(urlopen(base + "/api/health", timeout=3).read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")
                maintenance = json.loads(urlopen(base + "/api/maintenance", timeout=3).read().decode("utf-8"))
                duplicate = [item for item in maintenance["suggestions"] if item["type"] == "duplicate_merge"][0]
                lifecycle = json.loads(urlopen(base + "/api/lifecycle/summary", timeout=3).read().decode("utf-8"))
                self.assertEqual(lifecycle["status"], "ok")
                self.assertGreaterEqual(lifecycle["state_counts"]["active"], 1)
                self.assertTrue(lifecycle["retrieval_policy"]["raw_memory_preserved"])
                compaction_candidates = json.loads(urlopen(base + "/api/lifecycle/compaction-candidates", timeout=3).read().decode("utf-8"))
                self.assertEqual(compaction_candidates["status"], "ok")
                self.assertGreaterEqual(compaction_candidates["count"], 1)
                preview_req = Request(
                    base + f"/api/compaction/clusters/{compaction_candidates['candidates'][0]['cluster_id']}/run?project=DysonSpherain",
                    data=json.dumps({"mode": "deterministic", "verifier": "unit"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                preview = json.loads(urlopen(preview_req, timeout=3).read().decode("utf-8"))
                self.assertEqual(preview["status"], "ok")
                self.assertIn("canonical_content", preview["result"])
                verify_req = Request(
                    base + f"/api/compaction/results/{preview['result']['result_id']}/verify?project=DysonSpherain",
                    data=json.dumps({}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                verified = json.loads(urlopen(verify_req, timeout=3).read().decode("utf-8"))
                self.assertTrue(verified["result"]["verifier_passed"])
                reject_req = Request(
                    base + f"/api/compaction/results/{preview['result']['result_id']}/reject?project=DysonSpherain",
                    data=json.dumps({"reason": "daemon preview path tested"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                rejected = json.loads(urlopen(reject_req, timeout=3).read().decode("utf-8"))
                self.assertEqual(rejected["result"]["status"], "rejected")
                compact_req = Request(
                    base + "/api/lifecycle/compact?project=DysonSpherain",
                    data=json.dumps({"cluster_id": compaction_candidates["candidates"][0]["cluster_id"], "mode": "deterministic", "verifier": "unit"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                compacted = json.loads(urlopen(compact_req, timeout=3).read().decode("utf-8"))
                self.assertEqual(compacted["status"], "ok")
                self.assertTrue(compacted["raw_memory_preserved"])
                inspect_req = Request(
                    base + "/api/retrieval/inspect?project=DysonSpherain",
                    data=json.dumps({"query": "daemon maintenance", "limit": 5}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                inspected = json.loads(urlopen(inspect_req, timeout=3).read().decode("utf-8"))
                self.assertIn("stage_counts", inspected)
                self.assertIn("score_breakdown", inspected)
                lifecycle_audit = json.loads(urlopen(base + "/api/lifecycle/audit", timeout=3).read().decode("utf-8"))
                self.assertEqual(lifecycle_audit["events"][0]["event_type"], "memory_compaction_committed")
                rebuild_req = Request(
                    base + "/api/index/rebuild?project=DysonSpherain",
                    data=json.dumps({}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                rebuild = json.loads(urlopen(rebuild_req, timeout=3).read().decode("utf-8"))
                self.assertEqual(rebuild["status"], "ok")
                backends = json.loads(urlopen(base + "/api/index/embedding-backends", timeout=3).read().decode("utf-8"))
                self.assertIn("local_hash_embedding", backends["backends"])
                vector_backends = json.loads(urlopen(base + "/api/index/vector-backends", timeout=3).read().decode("utf-8"))
                self.assertIn("sqlite_inline", vector_backends["backends"])
                apply_req = Request(
                    base + "/api/maintenance/apply?project=DysonSpherain",
                    data=json.dumps({"suggestion_id": duplicate["suggestion_id"], "canonical_id": first["capsule_id"]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                applied = json.loads(urlopen(apply_req, timeout=3).read().decode("utf-8"))
                self.assertEqual(applied["status"], "applied")
                search = json.loads(urlopen(base + "/api/search?query=daemon", timeout=3).read().decode("utf-8"))
                self.assertGreaterEqual(search["count"], 1)
                obs_id = next(item["observation_id"] for item in search["observations"] if item["kind"] == "note")
                detail = json.loads(urlopen(base + f"/api/observations/{obs_id}", timeout=3).read().decode("utf-8"))
                self.assertIn("daemon search", detail["observations"][0]["content"])
                token = json.loads(urlopen(base + "/api/token-economy", timeout=3).read().decode("utf-8"))
                self.assertEqual(token["windows"]["24h"]["estimated_saved_tokens"], 900)
                self.assertAlmostEqual(token["windows"]["24h"]["saving_ratio"], 0.9)
                self.assertEqual(token["decision_distribution"]["inject"], 1)
                self.assertEqual(token["fallback_tokenizer_rate"], 0.0)
                self.assertIn("llm_prompt_token_economy", token)
                self.assertAlmostEqual(token["local_compute_economy"]["embedding_cache_hit_rate"], 0.75)
                resume = json.loads(urlopen(base + "/api/resume-context?session_id=s1", timeout=3).read().decode("utf-8"))
                self.assertEqual(resume["status"], "ok")
                self.assertIn("DysonSpherain Resume Context", resume["rendered_context"])
                cockpit = json.loads(urlopen(base + "/api/runtime/cockpit", timeout=3).read().decode("utf-8"))
                self.assertEqual(cockpit["status"], "ok")
                self.assertIn("mission_control", cockpit)
                self.assertIn("config", cockpit)
                runtime_config = json.loads(urlopen(base + "/api/runtime/config", timeout=3).read().decode("utf-8"))
                self.assertEqual(runtime_config["status"], "ok")
                post = Request(
                    base + "/api/runtime/config",
                    data=json.dumps({"context_budget": 777, "ui_animation_intensity": "low"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                updated = json.loads(urlopen(post, timeout=3).read().decode("utf-8"))
                self.assertEqual(updated["config"]["context_budget"], 777)
                settings_post = Request(
                    base + "/api/settings",
                    data=json.dumps({"llm_config": {"external_llm_enabled": False, "local_only": True}, "compaction_config": {"mode": "deterministic", "max_input_memories": 12}}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                settings = json.loads(urlopen(settings_post, timeout=3).read().decode("utf-8"))
                self.assertFalse(settings["config"]["llm_config"]["external_llm_enabled"])
                self.assertEqual(settings["config"]["compaction_config"]["max_input_memories"], 12)
                queue = Request(
                    base + "/api/runtime/scheduler/enqueue",
                    data=json.dumps({"trigger": "artifact_updated"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                queued = json.loads(urlopen(queue, timeout=3).read().decode("utf-8"))
                self.assertEqual(queued["status"], "ok")
                scheduler = json.loads(urlopen(base + "/api/runtime/scheduler", timeout=3).read().decode("utf-8"))
                self.assertGreaterEqual(scheduler["pending_count"], 1)
                html = urlopen(base + "/", timeout=3).read().decode("utf-8")
                self.assertIn("DysonSpherain Memory", html)
                self.assertIn("Project Dashboard", html)
                self.assertIn("Mission Control", html)
                self.assertIn("Memory Ledger", html)
                self.assertIn("Situation Graph", html)
                self.assertIn("graph-canvas", html)
                self.assertIn("renderGraphCanvas", html)
                self.assertIn("graphTimeline", html)
                self.assertIn("stepGraph", html)
                self.assertIn("Evidence Router", html)
                self.assertIn("Context Compiler", html)
                self.assertIn("Evidence Search", html)
                self.assertIn("Retrieval Trace Viewer", html)
                self.assertIn("Evidence Timeline", html)
                self.assertIn("Evidence Field Graph", html)
                self.assertIn("Benchmark Lab", html)
                self.assertIn("Health Doctor", html)
                self.assertIn("Maintenance", html)
                self.assertIn("renderEvidenceSearch", html)
                self.assertIn("runProductSearch", html)
                self.assertIn("rebuildProductIndex", html)
                self.assertIn("buildProductContext", html)
                self.assertIn("/api/benchmark-dashboard", html)
                self.assertIn("Recall Audit", html)
                self.assertIn("Active Scheduler", html)
                self.assertIn("Configuration Studio", html)
                self.assertIn("Operator Weights", html)
                self.assertIn("Section Limits", html)
                self.assertIn("Import / Export", html)
            finally:
                server.shutdown()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
