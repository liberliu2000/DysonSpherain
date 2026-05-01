from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.adapters.daemon import DysonMemoryHandler
from dysonspherain.product import (
    apply_maintenance_suggestion,
    benchmark_compare,
    benchmark_record,
    apply_retention,
    configure_embedding_backend,
    configure_encryption,
    configure_product_vector_backend,
    create_context_pack,
    dismiss_maintenance_suggestion,
    doctor,
    export_project,
    forget_capsule,
    forget_before,
    get_active_evidence,
    get_decision_chain,
    get_evidence_at_time,
    get_evidence_for_commit,
    get_maintenance_suggestion,
    init_product_store,
    mark_contradicted,
    mark_deprecated,
    mark_reverted,
    mark_superseded,
    maintenance_suggestions,
    migrate_product_db_to_sqlcipher,
    record_source,
    remember,
    privacy_policy,
    product_embedding_backends,
    product_vector_backends,
    rebuild_product_embeddings,
    rebuild_product_vector_index,
    register_alias,
    retrieve,
    resolve_alias,
)
from sphere_cli.cli import app


class ProductMemoryTests(unittest.TestCase):
    def test_product_store_retrieve_pack_benchmark_and_forget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init = init_product_store(root, project_id="P")
            self.assertEqual(init["status"], "ok")
            self.assertTrue(Path(init["db_path"]).exists())

            created = remember(
                root,
                project_id="P",
                text="Keep stable citation ids for benchmark artifacts.",
                evidence_type="decision",
                tags=["benchmark"],
            )
            capsule_id = created["capsule_id"]
            self.assertTrue(capsule_id.startswith("cap_"))

            result = retrieve(root, project_id="P", query="citation benchmark", show_audit=True, context_pack=True, max_tokens=500)
            self.assertGreaterEqual(result["count"], 1)
            self.assertIn("retrieval_trace", result)
            self.assertIn(capsule_id, result["context_pack"]["capsule_ids"])

            pack = create_context_pack(root, project_id="P", query="citation", max_tokens=500)
            self.assertIn("Must-Use Anchors", pack["markdown"])

            metrics = root / "metrics.json"
            metrics.write_text(json.dumps({"benchmark": "unit", "metrics": {"recall": 1.0}, "elapsed_seconds": 1.5}), encoding="utf-8")
            bench = benchmark_record(root, project_id="P", artifact=metrics)
            self.assertEqual(bench["benchmark"], "unit")

            deleted = forget_capsule(root, capsule_id=capsule_id, project_id="P")
            self.assertEqual(deleted["status"], "deleted")
            after = retrieve(root, project_id="P", query="citation benchmark", show_audit=True)
            self.assertNotIn(capsule_id, [item["capsule_id"] for item in after["candidates"]])

    def test_code_error_benchmark_binding_and_health_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            err = root / "traceback.txt"
            err.write_text(
                'Traceback (most recent call last):\n  File "src/app.py", line 12, in run\n    boom()\nValueError: broken input\n',
                encoding="utf-8",
            )
            error_record = record_source(root, project_id="P", source="error", file=err)
            binding = error_record["capsule"]["metadata"]["error_binding"]
            self.assertEqual(binding["exception_type"], "ValueError")
            self.assertEqual(binding["stack_frames"][0]["file"], "src/app.py")
            self.assertIn("src/app.py", error_record["capsule"]["file_refs"])

            git_record = record_source(root, project_id="P", source="code-diff", text="")
            self.assertIn("git_binding", git_record["capsule"]["metadata"])
            self.assertIn("changed_files", git_record["capsule"]["metadata"]["git_binding"])

            current = root / "current_metrics.json"
            baseline = root / "baseline_metrics.json"
            current.write_text(
                json.dumps({"benchmark": "KnowMe", "dataset": "KnowMe", "metrics": {"recall": 0.8, "candidate_recall@100": 0.9, "latency_ms": 120}, "failure_taxonomy": {"miss": 1}}),
                encoding="utf-8",
            )
            baseline.write_text(
                json.dumps({"benchmark": "KnowMe", "dataset": "KnowMe", "metrics": {"recall": 0.7, "candidate_recall@100": 0.95, "latency_ms": 100}, "failure_taxonomy": {"miss": 0}}),
                encoding="utf-8",
            )
            recorded = benchmark_record(root, project_id="P", artifact=current, benchmark="KnowMe")
            self.assertEqual(recorded["binding"]["dataset"], "KnowMe")
            self.assertIn("candidate_recall@100", recorded["binding"]["quality"])
            self.assertTrue(Path(recorded["dashboard_files"]["benchmark_runs"]).exists())

            comparison = benchmark_compare(root, project_id="P", current=current, baseline=baseline)
            self.assertLess(comparison["candidate_recall_deltas"]["candidate_recall@100"], 0)
            self.assertEqual(comparison["latency_deltas"]["latency_ms"], 20)
            self.assertEqual(comparison["failure_taxonomy_deltas"]["miss"], 1)
            self.assertTrue((root / ".memory" / "artifacts" / "benchmark_lab" / "regression_report.json").exists())

            health = doctor(root, project_id="P")
            self.assertIn("embedding_backend", health["checks"])
            self.assertIn("runtime_commands", health["checks"])
            self.assertIn("benchmark_dashboard", health["checks"])

    def test_privacy_ignore_retention_forget_and_export_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".dysonignore").write_text("private/\n*.secret\n", encoding="utf-8")
            secret = root / ".env"
            secret.write_text("api_key=sk-abcdef1234567890", encoding="utf-8")
            skipped = record_source(root, project_id="P", source="manual", file=secret)
            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(skipped["reason"], "ignored_path")

            private_dir = root / "private"
            private_dir.mkdir()
            private_file = private_dir / "note.txt"
            private_file.write_text("private note", encoding="utf-8")
            skipped_private = record_source(root, project_id="P", source="manual", file=private_file)
            self.assertEqual(skipped_private["status"], "skipped")

            allowed = root / "notes.txt"
            allowed.write_text("public note", encoding="utf-8")
            captured = record_source(root, project_id="P", source="manual", file=allowed, allowlist=["*.txt"])
            self.assertEqual(captured["status"], "ok")
            self.assertIn("notes.txt", captured["capsule"]["file_refs"][0])

            old = remember(root, project_id="P", text="old retention item", evidence_type="note")
            new = remember(root, project_id="P", text="new retention item", evidence_type="note")
            create_context_pack(root, project_id="P", query="old retention", max_tokens=800)
            deleted = forget_capsule(root, capsule_id=old["capsule_id"], project_id="P")
            self.assertEqual(deleted["status"], "deleted")
            result = retrieve(root, project_id="P", query="old retention", show_audit=True)
            self.assertNotIn(old["capsule_id"], [item["capsule_id"] for item in result["candidates"]])

            before_result = forget_before(root, project_id="P", before=new["capsule"]["timestamp"])
            self.assertGreaterEqual(before_result["forgotten_count"], 0)
            retention = apply_retention(root, project_id="P", keep_last=1)
            self.assertGreaterEqual(retention["forgotten_count"], 0)

            exported = export_project(root, project_id="P", fmt="json")
            self.assertTrue(Path(exported["output"]).exists())
            self.assertTrue(Path(exported["manifest"]).exists())
            policy = privacy_policy(root)
            self.assertIn(".env", policy["ignore_patterns"])
            self.assertFalse(policy["encryption_at_rest"]["available"])

    def test_product_probe_registry_and_candidate_admission_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = remember(
                root,
                project_id="P",
                text="KnowMe benchmark candidate_recall artifact should keep stable citations.",
                evidence_type="benchmark_result",
                tags=["KnowMe"],
                artifact_refs=["runs/knowme/metrics.json"],
                benchmark_refs=["KnowMe"],
            )
            remember(
                root,
                project_id="P",
                text="Old KnowMe benchmark note that should not enter active context.",
                evidence_type="decision",
                validity_state="superseded",
                tags=["KnowMe"],
                benchmark_refs=["KnowMe"],
            )

            result = retrieve(root, project_id="P", query="KnowMe benchmark metrics", show_audit=True, context_pack=True, max_tokens=700)
            trace = result["retrieval_trace"]

            self.assertEqual(result["route"], "benchmark")
            self.assertIn("sparse_probe", trace["probe_results"])
            self.assertIn("dense_probe", trace["probe_results"])
            self.assertIn("artifact_probe", trace["probe_results"])
            self.assertIn("recent_state_probe", trace["probe_results"])
            self.assertEqual(trace["unavailable_probes"], {})
            self.assertGreaterEqual(trace["probe_results"]["dense_probe"]["count"], 1)
            self.assertGreaterEqual(trace["probe_results"]["sparse_probe"]["count"], 1)
            self.assertGreaterEqual(trace["drop_stage_distribution"]["duplicate_collapse"], 1)
            self.assertEqual(trace["drop_stage_distribution"]["validity_filter"], 1)
            self.assertEqual(trace["final_candidates"][0]["capsule_id"], benchmark["capsule_id"])
            self.assertIn("source_probes", trace["final_candidates"][0]["raw_features"])
            self.assertIn(benchmark["capsule_id"], result["context_pack"]["capsule_ids"])

    def test_dense_probe_alias_maintenance_and_encryption_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = remember(root, project_id="P", text="Alpha semantic dense memory benchmark.", evidence_type="note")
            remember(root, project_id="P", text="Alpha semantic dense memory benchmark.", evidence_type="note")
            old_bench = remember(
                root,
                project_id="P",
                text="KnowMe recall baseline before repair.",
                evidence_type="benchmark_result",
                benchmark_refs=["KnowMe"],
            )
            new_bench = remember(
                root,
                project_id="P",
                text="KnowMe recall latest after repair.",
                evidence_type="benchmark_result",
                benchmark_refs=["KnowMe"],
            )

            result = retrieve(root, project_id="P", query="semantic dense benchmark", show_audit=True)
            trace = result["retrieval_trace"]
            self.assertIn("dense_probe", trace["probe_results"])
            self.assertGreaterEqual(trace["probe_results"]["dense_probe"]["count"], 1)
            self.assertEqual(trace["unavailable_probes"], {})
            self.assertIn(first["capsule_id"], [item["capsule_id"] for item in result["candidates"]])

            registered = register_alias(root, project_id="P", alias="local hash backend", canonical="local_hash_embedding")
            self.assertEqual(registered["status"], "ok")
            resolved = resolve_alias(root, project_id="P", value="local hash backend")
            self.assertEqual(resolved["canonical"], "local_hash_embedding")

            suggestions = maintenance_suggestions(root, project_id="P")
            suggestion_types = {item["type"] for item in suggestions["suggestions"]}
            self.assertIn("duplicate_merge", suggestion_types)
            self.assertIn("invalidate_older_benchmark", suggestion_types)
            invalidation = [item for item in suggestions["suggestions"] if item["type"] == "invalidate_older_benchmark"][0]
            self.assertEqual(invalidation["older_capsule_id"], old_bench["capsule_id"])
            self.assertEqual(invalidation["newer_capsule_id"], new_bench["capsule_id"])

            health = doctor(root, project_id="P")
            self.assertTrue(health["checks"]["embedding_backend"]["dense_probe_available"])
            self.assertGreaterEqual(health["checks"]["maintenance"]["suggestions"], 2)
            policy = privacy_policy(root)
            self.assertEqual(policy["encryption_at_rest"]["status"], "not_configured")
            self.assertFalse(policy["encryption_at_rest"]["available"])

            rebuild = rebuild_product_embeddings(root, project_id="P")
            self.assertEqual(rebuild["status"], "ok")
            self.assertEqual(rebuild["capsules_seen"], 4)
            backends = product_embedding_backends(root)
            self.assertTrue(backends["backends"]["local_hash_embedding"]["available"])
            configured = configure_embedding_backend(root, backend="local_hash_embedding")
            self.assertEqual(configured["config"]["backend"], "local_hash_embedding")
            pending_semantic = configure_embedding_backend(root, backend="sentence_transformers", allow_unavailable=True)
            self.assertEqual(pending_semantic["config"]["backend"], "sentence_transformers")
            configure_embedding_backend(root, backend="local_hash_embedding")
            vector_backends = product_vector_backends(root)
            self.assertTrue(vector_backends["backends"]["sqlite_inline"]["available"])
            vector_config = configure_product_vector_backend(root, backend="chroma", allow_unavailable=True)
            self.assertEqual(vector_config["config"]["backend"], "chroma")
            vector_rebuild = rebuild_product_vector_index(root, project_id="P")
            self.assertIn(vector_rebuild["status"], {"ok", "unavailable"})
            configure_product_vector_backend(root, backend="sqlite_inline")

            duplicate = [item for item in suggestions["suggestions"] if item["type"] == "duplicate_merge"][0]
            fetched = get_maintenance_suggestion(root, project_id="P", suggestion_id=duplicate["suggestion_id"])
            self.assertEqual(fetched["type"], "duplicate_merge")
            applied = apply_maintenance_suggestion(root, project_id="P", suggestion_id=duplicate["suggestion_id"], canonical_id=first["capsule_id"])
            self.assertEqual(applied["status"], "applied")
            active = retrieve(root, project_id="P", query="semantic dense benchmark", show_audit=True)
            self.assertEqual([item["capsule_id"] for item in active["candidates"] if item["capsule_id"] == first["capsule_id"]], [first["capsule_id"]])

            invalidation_id = invalidation["suggestion_id"]
            dismissed = dismiss_maintenance_suggestion(root, project_id="P", suggestion_id=invalidation_id, reason="unit test")
            self.assertEqual(dismissed["status"], "dismissed")
            self.assertEqual(get_maintenance_suggestion(root, project_id="P", suggestion_id=invalidation_id)["suggestion_status"], "dismissed")
            encrypted_marker = configure_encryption(root, provider="external_or_os_managed", scope="unit_test_volume")
            self.assertEqual(encrypted_marker["encryption_at_rest"]["status"], "external_or_os_managed")
            migration = migrate_product_db_to_sqlcipher(root, key_env="DYSON_TEST_SQLCIPHER_KEY")
            self.assertIn(migration["status"], {"ok", "unavailable"})

    def test_temporal_validity_api_and_decision_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = remember(root, project_id="P", text="Use policy A for entity Alpha.", evidence_type="decision", tags=["Alpha"])
            second = remember(root, project_id="P", text="Use policy B for entity Alpha.", evidence_type="decision", tags=["Alpha"])

            superseded = mark_superseded(root, first["capsule_id"], second["capsule_id"], "policy B replaced policy A")
            self.assertEqual(superseded["validity_state"], "superseded")
            active = get_active_evidence(root, project_id="P")
            self.assertEqual([item["id"] for item in active["capsules"]], [second["capsule_id"]])

            chain = get_decision_chain(root, project_id="P", entity="Alpha")
            self.assertEqual([item["id"] for item in chain["decision_chain"]], [first["capsule_id"], second["capsule_id"]])
            self.assertEqual(chain["decision_chain"][0]["superseded_by"], [second["capsule_id"]])
            self.assertEqual(chain["decision_chain"][1]["supersedes"], [first["capsule_id"]])

            at_time = get_evidence_at_time(root, project_id="P", timestamp=second["capsule"]["timestamp"])
            self.assertGreaterEqual(at_time["count"], 2)
            commit = second["capsule"]["git_commit"]
            if commit:
                by_commit = get_evidence_for_commit(root, project_id="P", commit_hash=commit)
                self.assertGreaterEqual(by_commit["count"], 1)

            contradicted = mark_contradicted(root, second["capsule_id"], first["capsule_id"], "test contradiction")
            self.assertEqual(contradicted["validity_state"], "contradicted")
            deprecated = mark_deprecated(root, first["capsule_id"], "old decision")
            self.assertEqual(deprecated["validity_state"], "deprecated")
            reverted = mark_reverted(root, second["capsule_id"], first["capsule_id"], "revert test")
            self.assertEqual(reverted["validity_state"], "reverted")

    def test_context_composer_formats_sections_and_section_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = remember(
                root,
                project_id="P",
                text="Decision Alpha should retain artifact citations and debug trace visibility.",
                evidence_type="decision",
                artifact_refs=["artifacts/alpha/metrics.json"],
            )
            pack = create_context_pack(
                root,
                project_id="P",
                query="Decision Alpha artifact",
                max_tokens=800,
                task_type="benchmark",
                agent_role="benchmarker",
                sections=["Mission State", "Supporting Evidence"],
                section_budget={"Supporting Evidence": 80},
                include_debug_trace=True,
                fmt="yaml",
            )

            self.assertEqual(pack["format"], "yaml")
            self.assertEqual([section["name"] for section in pack["sections"]], ["Mission State", "Supporting Evidence"])
            self.assertIn("sections_omitted", pack["risk_flags"])
            self.assertIn("context_pack_id:", pack["rendered"])
            self.assertIn(created["capsule_id"], pack["markdown"])

            text_pack = create_context_pack(
                root,
                project_id="P",
                query="Decision Alpha artifact",
                max_tokens=800,
                sections=["Mission State"],
                fmt="text",
            )
            self.assertTrue(text_pack["rendered"].startswith("Mission State\n"))
            self.assertNotIn("## Mission State", text_pack["rendered"])

    def test_product_cli_flow(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = Path.cwd()
            os.chdir(root)
            self.addCleanup(os.chdir, previous)

            result = runner.invoke(app, ["init", "--project", "P"])
            self.assertEqual(result.exit_code, 0, result.output)

            result = runner.invoke(app, ["remember", "--project", "P", "--type", "decision", "--text", "Use route-conditioned evidence capsules."])
            self.assertEqual(result.exit_code, 0, result.output)
            capsule_id = json.loads(result.output)["capsule_id"]

            result = runner.invoke(app, ["retrieve", "route-conditioned", "--project", "P", "--show-audit", "--context-pack", "--max-tokens", "500"])
            self.assertEqual(result.exit_code, 0, result.output)
            payload = json.loads(result.output)
            self.assertEqual(payload["candidates"][0]["capsule_id"], capsule_id)
            self.assertIn("context_pack", payload)

            result = runner.invoke(app, ["wake", "--project", "P", "--task", "route-conditioned", "--max-tokens", "500", "--format", "json"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn(capsule_id, json.loads(result.output)["capsule_ids"])

            result = runner.invoke(app, ["index", "rebuild", "--project", "P"])
            self.assertEqual(result.exit_code, 0, result.output)
            rebuild_payload = json.loads(result.output)
            self.assertEqual(rebuild_payload["embedding_rebuild"]["status"], "ok")

            result = runner.invoke(app, ["index", "embedding-backends", "--project", "P"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("local_hash_embedding", json.loads(result.output)["backends"])
            result = runner.invoke(app, ["index", "vector-backends", "--project", "P"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("sqlite_inline", json.loads(result.output)["backends"])
            result = runner.invoke(app, ["index", "configure-vector", "chroma", "--allow-unavailable"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result.output)["config"]["backend"], "chroma")
            result = runner.invoke(app, ["index", "rebuild-vector", "--project", "P"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn(json.loads(result.output)["status"], {"ok", "unavailable"})
            result = runner.invoke(app, ["index", "configure-vector", "sqlite_inline"])
            self.assertEqual(result.exit_code, 0, result.output)
            result = runner.invoke(app, ["index", "configure-embedding", "local_hash_embedding"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result.output)["config"]["backend"], "local_hash_embedding")
            result = runner.invoke(app, ["index", "configure-encryption", "external_or_os_managed", "--scope", "cli-test"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result.output)["encryption_at_rest"]["status"], "external_or_os_managed")
            result = runner.invoke(app, ["index", "migrate-sqlcipher", "--key-env", "DYSON_TEST_SQLCIPHER_KEY"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn(json.loads(result.output)["status"], {"ok", "unavailable"})

            result = runner.invoke(app, ["remember", "--project", "P", "--type", "decision", "--text", "Use route-conditioned evidence capsules."])
            self.assertEqual(result.exit_code, 0, result.output)
            result = runner.invoke(app, ["index", "maintenance", "--project", "P"])
            self.assertEqual(result.exit_code, 0, result.output)
            maintenance_payload = json.loads(result.output)
            duplicate = [item for item in maintenance_payload["suggestions"] if item["type"] == "duplicate_merge"][0]
            result = runner.invoke(app, ["index", "maintenance", "--project", "P", "--dismiss", duplicate["suggestion_id"], "--reason", "cli smoke"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result.output)["status"], "dismissed")

            for args in (
                ["runtime", "during-task", "--project", "P", "--summary", "mid task checkpoint"],
                ["runtime", "before-benchmark", "--project", "P", "--benchmark", "KnowMe", "--profile", "official"],
                ["runtime", "before-commit", "--project", "P", "--summary", "prepare commit"],
                ["runtime", "after-commit", "--project", "P", "--commit", "abc123", "--summary", "commit done"],
                ["runtime", "manual-checkpoint", "--project", "P", "--summary", "manual checkpoint"],
            ):
                result = runner.invoke(app, args)
                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertEqual(payload["status"], "ok")
                self.assertTrue(Path(payload["context_pack_path"]).exists())

    def test_product_daemon_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            handler = type("TestProductHandler", (DysonMemoryHandler,), {"base_dir": root, "project": "P"})
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                post = Request(
                    base + "/api/capsules?project=P",
                    data=json.dumps({"text": "Capsule API keeps raw traces and citations.", "type": "decision", "tags": ["api"]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                created = json.loads(urlopen(post, timeout=3).read().decode("utf-8"))
                capsule_id = created["capsule_id"]

                capsules = json.loads(urlopen(base + "/api/capsules?project=P", timeout=3).read().decode("utf-8"))
                self.assertEqual(capsules["count"], 1)
                detail = json.loads(urlopen(base + f"/api/capsules/{capsule_id}?project=P", timeout=3).read().decode("utf-8"))
                self.assertEqual(detail["capsule"]["id"], capsule_id)

                retrieve_req = Request(
                    base + "/api/retrieve?project=P",
                    data=json.dumps({"query": "raw traces citations", "show_audit": True, "context_pack": True, "max_tokens": 500}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                retrieved = json.loads(urlopen(retrieve_req, timeout=3).read().decode("utf-8"))
                trace_id = retrieved["retrieval_trace"]["trace_id"]
                context_pack_id = retrieved["context_pack"]["context_pack_id"]
                trace = json.loads(urlopen(base + f"/api/retrieval-traces/{trace_id}?project=P", timeout=3).read().decode("utf-8"))
                self.assertEqual(trace["trace"]["query"], "raw traces citations")
                pack = json.loads(urlopen(base + f"/api/context-packs/{context_pack_id}?project=P", timeout=3).read().decode("utf-8"))
                self.assertIn("Supporting Evidence", pack["context_pack"]["markdown"])

                patch = Request(
                    base + f"/api/capsules/{capsule_id}?project=P",
                    data=json.dumps({"validity_state": "superseded"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="PATCH",
                )
                updated = json.loads(urlopen(patch, timeout=3).read().decode("utf-8"))
                self.assertEqual(updated["capsule"]["validity_state"], "superseded")

                projects = json.loads(urlopen(base + "/api/projects", timeout=3).read().decode("utf-8"))
                self.assertEqual(projects["projects"][0]["project_id"], "P")
                health = json.loads(urlopen(base + "/api/health?project=P", timeout=3).read().decode("utf-8"))
                self.assertEqual(health["product"]["status"], "ok")
            finally:
                server.shutdown()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
