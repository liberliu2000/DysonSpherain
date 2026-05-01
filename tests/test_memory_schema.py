from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.project_state import get_memory, write_memory


class MemorySchemaTests(unittest.TestCase):
    def test_project_memory_record_has_required_plan_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            payload = write_memory(
                root,
                memory_type="decision",
                project="DysonSpherain",
                content="Use artifact-backed formal evidence only.",
                source="unit-test",
                metadata={"title": "formal evidence rule", "status": "current"},
            )
            record = get_memory(root, "DysonSpherain", payload["memory_id"])

            self.assertIsNotNone(record)
            for key in (
                "memory_id",
                "memory_type",
                "project",
                "title",
                "content",
                "source",
                "status",
                "metadata",
                "created_at",
            ):
                self.assertIn(key, record)
            self.assertEqual(record["memory_type"], "decision")
            self.assertEqual(record["status"], "current")
            self.assertEqual(record["source"], "unit-test")

    def test_secret_fields_are_redacted_before_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            payload = write_memory(
                root,
                memory_type="fact",
                project="DysonSpherain",
                content="api_key=sk-abcdef1234567890",
                source="unit-test",
                metadata={"token": "sk-abcdef1234567890"},
            )
            record = get_memory(root, "DysonSpherain", payload["memory_id"])

            self.assertNotIn("sk-abcdef", str(record))


if __name__ == "__main__":
    unittest.main()
