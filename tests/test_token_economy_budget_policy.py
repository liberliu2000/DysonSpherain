from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.context_pack.token_budgeter import BudgetPolicy, budget_policy_for
from dysonspherain.evaluation.token_economy import _sample_from_payload, main
from dysonspherain.utils.token_counter import TokenCounter
from sphere_cli.context_assembler import ContextAssembler


class TokenEconomyBudgetPolicyTests(unittest.TestCase):
    def test_builtin_budget_policies_do_not_exceed_total_ratio(self) -> None:
        for task_type in ("coding", "debug", "benchmark", "paper", "unknown"):
            self.assertLessEqual(budget_policy_for(task_type, 1000).ratio_sum, 1.0)
        with self.assertRaises(ValueError):
            BudgetPolicy(total_budget=1000, core_evidence_ratio=0.9, reserve_ratio=0.2)

    def test_legacy_context_assembler_uses_non_negative_budget_manifest(self) -> None:
        bundle = ContextAssembler().assemble(
            task="debug benchmark",
            task_type="benchmark",
            temperature=0.0,
            main_nodes=[{"id": "m1", "shell": 1, "sector": "s", "zone": "z", "cell": "c", "molecular_type": "fact", "summary": "failure evidence"}],
            reflected_nodes=[],
            refracted_nodes=[],
            max_tokens=200,
        )
        self.assertLessEqual(bundle.debug["budget_policy_ratio_sum"], 1.0)
        self.assertGreaterEqual(bundle.debug["raw_pointer_budget"], 0)

    def test_default_budgeting_drops_oversized_single_evidence_instead_of_truncating_it(self) -> None:
        counter = TokenCounter()
        payload = {
            "sample_id": "single-evidence",
            "query": "What caused the benchmark regression?",
            "history": "full history " * 200,
            "retrieved_context": "single evidence sentence " * 200,
        }
        sample = _sample_from_payload(
            payload,
            index=0,
            mode="off",
            baseline_type="full_history",
            budget=20,
            recent_k=5,
            counter=counter,
            allow_evidence_truncation=False,
        )
        self.assertTrue(sample.extra["context_truncated"])
        self.assertEqual(sample.retrieved_context_tokens, 0)

    def test_allow_evidence_truncation_compacts_oversized_single_evidence(self) -> None:
        counter = TokenCounter()
        payload = {
            "sample_id": "single-evidence",
            "query": "What caused the benchmark regression?",
            "history": "full history " * 200,
            "retrieved_context": "single evidence sentence " * 200,
        }
        sample = _sample_from_payload(
            payload,
            index=0,
            mode="off",
            baseline_type="full_history",
            budget=20,
            recent_k=5,
            counter=counter,
            allow_evidence_truncation=True,
        )
        self.assertTrue(sample.extra["context_truncated"])
        self.assertGreater(sample.retrieved_context_tokens, 0)
        self.assertGreater(sample.final_prompt_tokens, sample.user_query_tokens)

    def test_multiple_context_budgets_write_distinct_artifact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "samples.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "sample_id": "budgeted",
                        "query": "Continue CloneMem regression analysis.",
                        "history": "full history " * 200,
                        "retrieved_context": "line one evidence\nline two evidence\nline three evidence",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "out"
            main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_dir),
                    "--modes",
                    "off",
                    "--baseline-types",
                    "full_history",
                    "--context-token-budget",
                    "20,40",
                ]
            )
            rows = [
                json.loads(line)
                for line in (output_dir / "per_sample.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual([row["context_token_budget"] for row in rows], [20, 40])
            self.assertEqual(summary["context_token_budgets"], [20, 40])


if __name__ == "__main__":
    unittest.main()
