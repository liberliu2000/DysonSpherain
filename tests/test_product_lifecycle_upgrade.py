from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.product import (
    commit_compaction_result,
    create_compaction_result,
    find_compaction_candidates,
    get_active_successor,
    get_capsule,
    inspect_retrieval,
    mark_superseded,
    memory_lifecycle_summary,
    migrate_lifecycle_metadata,
    remember,
    run_deterministic_compaction,
    verify_compaction_result_record,
)
from dysonspherain.memory_runtime.config import save_runtime_config


class ProductLifecycleUpgradeTests(unittest.TestCase):
    def test_near_duplicate_compaction_preserves_raw_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = remember(root, project_id="P", text="Use local semantic memory compaction for repeated project decisions.", evidence_type="decision")
            second = remember(root, project_id="P", text="Use local semantic compaction for repeated project decision notes.", evidence_type="decision")

            candidates = find_compaction_candidates(root, project_id="P", near_duplicate_threshold=0.4)
            self.assertGreaterEqual(candidates["count"], 1)
            cluster = candidates["candidates"][0]
            self.assertIn(first["capsule_id"], cluster["memory_ids"])
            self.assertIn(second["capsule_id"], cluster["memory_ids"])

            compacted = run_deterministic_compaction(root, project_id="P", cluster_id=cluster["cluster_id"], mode="local_semantic", verifier="unit")
            self.assertEqual(compacted["status"], "ok")
            self.assertTrue(compacted["raw_memory_preserved"])
            self.assertTrue(compacted["verification"]["verifier_passed"])

            summary = memory_lifecycle_summary(root, project_id="P")
            self.assertEqual(summary["state_counts"]["canonical"], 1)
            self.assertEqual(summary["state_counts"]["compacted"], 2)

    def test_stable_memory_retrieval_and_supersession_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = remember(root, project_id="P", text="Old UI plan uses a plain table.", evidence_type="decision")
            new = remember(root, project_id="P", text="Current UI plan uses Memory OS Cockpit cards.", evidence_type="decision", validity_state="stable")
            mark_superseded(root, old["capsule_id"], new["capsule_id"], reason="newer UI plan")

            inspected = inspect_retrieval(root, project_id="P", query="current UI plan cards", limit=5)
            final_ids = [item["memory_id"] for item in inspected["final_context"]]
            self.assertIn(new["capsule_id"], final_ids)
            self.assertTrue(any(item["memory_id"] == old["capsule_id"] for item in inspected["excluded_evidence"]))

            successor = get_active_successor(root, old["capsule_id"], project_id="P")
            self.assertEqual(successor["successor"]["id"], new["capsule_id"])

    def test_lifecycle_migration_adds_hashes_and_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remember(root, project_id="P", text="Lifecycle migration should add hashes.", evidence_type="note")
            migrated = migrate_lifecycle_metadata(root, project_id="P", backup=True)

            self.assertEqual(migrated["status"], "ok")
            self.assertTrue(migrated["backup_path"])
            cap = get_capsule(root, remember(root, project_id="P", text="Second migrated note.", evidence_type="note")["capsule_id"], project_id="P")
            self.assertNotIn("content_hash", cap["metadata"])
            migrate_lifecycle_metadata(root, project_id="P", backup=False)
            cap = get_capsule(root, cap["id"], project_id="P")
            self.assertIn("content_hash", cap["metadata"])

    def test_settings_drive_scoring_and_compaction_result_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remember(root, project_id="P", text="Low importance matching memory for scoring settings.", evidence_type="note", metadata={"redundancy_score": 0.0})
            save_runtime_config(root, {"scoring_config": {"importance_weight": 0.9, "recency_weight": 0.0, "confidence_weight": 0.0, "access_weight": 0.0, "redundancy_weight": 0.0}, "lifecycle_multipliers": {"active": 2.0}}, project="P")

            inspected = inspect_retrieval(root, project_id="P", query="scoring settings", limit=3)
            breakdown = next(iter(inspected["score_breakdown"].values()))
            self.assertEqual(breakdown["weights"]["importance"], 0.9)
            self.assertEqual(breakdown["lifecycle_multiplier"], 2.0)

            remember(root, project_id="P", text="Compaction result preview should be reviewable.", evidence_type="note")
            remember(root, project_id="P", text="Compaction result preview should be reviewable.", evidence_type="note")
            cluster = find_compaction_candidates(root, project_id="P")["candidates"][0]
            preview = create_compaction_result(root, project_id="P", cluster_id=cluster["cluster_id"], mode="deterministic", verifier="unit")
            self.assertEqual(preview["status"], "ok")
            result_id = preview["result"]["result_id"]
            verified = verify_compaction_result_record(root, result_id=result_id, project_id="P")
            self.assertTrue(verified["result"]["verifier_passed"])
            committed = commit_compaction_result(root, result_id=result_id, project_id="P")
            self.assertEqual(committed["status"], "ok")
            self.assertTrue(committed["canonical_id"].startswith("cap_"))

    def test_llm_compaction_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remember(root, project_id="P", text="Repeated memory that should compact locally.", evidence_type="note")
            remember(root, project_id="P", text="Repeated memory that should compact locally.", evidence_type="note")
            cluster = find_compaction_candidates(root, project_id="P")["candidates"][0]

            with patch("dysonspherain.product.store.urllib.request.urlopen") as urlopen:
                preview = create_compaction_result(root, project_id="P", cluster_id=cluster["cluster_id"], mode="hybrid", verifier="unit")

            urlopen.assert_not_called()
            self.assertIn("external_llm_disabled_by_user", preview["result"]["warnings"])
            self.assertIn("external_llm_compaction_disabled_by_user", preview["result"]["warnings"])
            self.assertIsNone(preview["result"]["external_llm"])

    def test_local_only_blocks_external_llm_even_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remember(root, project_id="P", text="External LLM compaction must stay user controlled.", evidence_type="note")
            remember(root, project_id="P", text="External LLM compaction must stay user controlled.", evidence_type="note")
            save_runtime_config(
                root,
                {
                    "llm_config": {
                        "provider": "openai_compatible",
                        "api_base_url": "https://example.invalid",
                        "api_key": "test-key",
                        "external_llm_enabled": True,
                        "local_only": True,
                    },
                    "compaction_config": {"external_llm_compaction_enabled": True, "mode": "hybrid"},
                },
                project="P",
            )
            cluster = find_compaction_candidates(root, project_id="P")["candidates"][0]

            with patch("dysonspherain.product.store.urllib.request.urlopen") as urlopen:
                preview = create_compaction_result(root, project_id="P", cluster_id=cluster["cluster_id"], mode="hybrid", verifier="unit", confirm_external_call=True)

            urlopen.assert_not_called()
            self.assertIn("local_only_mode_prevents_external_llm", preview["result"]["warnings"])
            self.assertIsNone(preview["result"]["external_llm"])


if __name__ == "__main__":
    unittest.main()
