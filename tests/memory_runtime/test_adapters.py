from __future__ import annotations

import unittest

from dysonspherain.memory_runtime.adapters import ClaudeCodeAdapter, ManualImportAdapter, adapter_for_source


class RuntimeAdapterTests(unittest.TestCase):
    def test_adapter_for_source_and_claude_tool_capture(self) -> None:
        adapter = adapter_for_source("claude_code")
        self.assertIsInstance(adapter, ClaudeCodeAdapter)
        events = adapter.capture_tool_use({"tool_name": "Read", "input": {"file_path": "x.py"}, "project": "DysonSpherain"})
        self.assertEqual(events[0].event_type, "tool_call_observed")
        self.assertEqual(events[0].provenance["adapter"], "claude_code")

    def test_manual_import_adapter_accepts_explicit_event_type(self) -> None:
        adapter = ManualImportAdapter()
        events = adapter.capture_input({"event_type": "decision_made", "summary": "Use runtime ledger"})
        self.assertEqual(events[0].event_type, "decision_made")


if __name__ == "__main__":
    unittest.main()

