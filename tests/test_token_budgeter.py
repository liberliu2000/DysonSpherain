from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.context_pack.schemas import ContextPack, EvidenceItem
from dysonspherain.context_pack.token_budgeter import fit_context_pack


class TokenBudgeterTests(unittest.TestCase):
    def test_budgeter_reports_drop_and_budget_state(self) -> None:
        pack = ContextPack(summary="summary", core_evidence=[EvidenceItem(text="evidence " * 200), EvidenceItem(text="second")])
        result = fit_context_pack(pack, 20)
        self.assertGreater(result.estimated_tokens_before, result.estimated_tokens_after)
        self.assertIn("core_evidence", result.dropped_items)
        self.assertIsInstance(result.over_budget, bool)

    def test_budgeter_preserves_benchmark_failure_and_file_refs_before_low_value_context(self) -> None:
        pack = ContextPack(
            summary="current goal: repair benchmark regression",
            core_evidence=[
                EvidenceItem(text="raw log stdout " + ("noise " * 400), confidence=0.01, uncertain=True),
                EvidenceItem(text="candidate_recall@100 dropped after rerank", confidence=0.9),
            ],
            prior_decisions=["Do not use local_hash fallback."],
            known_failures=["regression reason: gold_rank_after_rerank worsened"],
            benchmark_state=["CloneMem recall_frac@10=0.12 candidate_recall@100=0.40 ndcg_any@10=0.08"],
        )
        result = fit_context_pack(pack, 120)
        rendered = "\n".join(
            [
                pack.summary,
                " ".join(item.text for item in pack.core_evidence),
                " ".join(pack.prior_decisions),
                " ".join(pack.known_failures),
                " ".join(pack.benchmark_state),
            ]
        )
        self.assertIn("candidate_recall@100", rendered)
        self.assertIn("regression reason", rendered)
        self.assertIn("Do not use local_hash fallback", rendered)
        self.assertTrue(any("low_value_or_long_text_compacted" in item for item in result.dropped_items))


if __name__ == "__main__":
    unittest.main()
