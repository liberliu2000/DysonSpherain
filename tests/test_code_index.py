from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.code_index import build_code_index, relevant_files, search_symbol


class CodeIndexTests(unittest.TestCase):
    def test_index_search_and_relevant_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "retrieval.py").write_text(
                "import json\n\nclass RouteConditionedCandidateAdmission:\n    pass\n\ndef safe_fusion():\n    return json.dumps({})\n",
                encoding="utf-8",
            )
            (root / "tests").mkdir()
            (root / "tests" / "test_retrieval.py").write_text("", encoding="utf-8")

            payload = build_code_index(root, root, project="DysonSpherain")
            symbols = search_symbol(root, "RouteConditioned", project="DysonSpherain")
            files = relevant_files(root, "retrieval safe fusion", project="DysonSpherain")

            self.assertEqual(payload["file_count"], 2)
            self.assertEqual(payload["parse_error_count"], 0)
            self.assertEqual(symbols[0]["name"], "RouteConditionedCandidateAdmission")
            self.assertEqual(files[0]["path"], "pkg/retrieval.py")

    def test_syntax_errors_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.py").write_text("def broken(:\n", encoding="utf-8")

            payload = build_code_index(root, root, project="DysonSpherain")

            self.assertEqual(payload["parse_error_count"], 1)
            self.assertIn("SyntaxError", payload["records"][0]["parse_error"])


if __name__ == "__main__":
    unittest.main()
