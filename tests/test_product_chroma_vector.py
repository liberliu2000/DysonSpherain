from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("chromadb")

from dysonspherain.product import configure_product_vector_backend, rebuild_product_vector_index, remember, retrieve


class ProductChromaVectorTests(unittest.TestCase):
    def test_chroma_vector_index_serves_dense_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = remember(
                root,
                project_id="P",
                evidence_type="decision",
                text="Chroma ANN product vector index should retrieve benchmark capsule.",
                tags=["ann", "chroma"],
            )
            configured = configure_product_vector_backend(root, backend="chroma")
            self.assertEqual(configured["config"]["backend"], "chroma")
            rebuilt = rebuild_product_vector_index(root, project_id="P")
            self.assertEqual(rebuilt["status"], "ok")
            self.assertGreaterEqual(rebuilt["indexed_count"], 1)

            result = retrieve(root, project_id="P", query="ANN benchmark capsule", show_audit=True)
            self.assertEqual(result["retrieval_trace"]["probe_results"]["dense_probe"]["status"], "ok")
            self.assertIn(created["capsule_id"], [item["capsule_id"] for item in result["candidates"]])
            dense = [item for item in result["candidates"] if item["capsule_id"] == created["capsule_id"]][0]
            self.assertEqual(dense["raw_features"]["vector_backend"], "chroma")


if __name__ == "__main__":
    unittest.main()
