from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.project_state import (
    archive_memory,
    get_memory,
    list_memories,
    search_memories,
    update_memory,
    write_memory,
)


class MemoryStoreTests(unittest.TestCase):
    def test_create_read_search_update_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            payload = write_memory(
                root,
                memory_type="task",
                project="DysonSpherain",
                content="Refresh formal protocol reports.",
                source="unit-test",
                metadata={"title": "formal refresh"},
            )
            memory_id = payload["memory_id"]

            self.assertEqual(get_memory(root, "DysonSpherain", memory_id)["title"], "formal refresh")
            self.assertEqual(search_memories(root, "DysonSpherain", "protocol")[0]["memory_id"], memory_id)

            updated = update_memory(root, "DysonSpherain", memory_id, {"status": "superseded"})
            self.assertEqual(updated["status"], "superseded")

            archived = archive_memory(root, "DysonSpherain", memory_id)
            self.assertEqual(archived["status"], "archived")
            self.assertEqual(list_memories(root, "DysonSpherain"), [])
            self.assertEqual(list_memories(root, "DysonSpherain", include_archived=True)[0]["memory_id"], memory_id)

    def test_type_filter_keeps_memory_classes_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_memory(root, memory_type="decision", project="DysonSpherain", content="Keep formal rows artifact-backed.", source="unit")
            write_memory(root, memory_type="task", project="DysonSpherain", content="Update paper tables.", source="unit")

            decisions = list_memories(root, "DysonSpherain", memory_type="decision")

            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]["memory_type"], "decision")


if __name__ == "__main__":
    unittest.main()
