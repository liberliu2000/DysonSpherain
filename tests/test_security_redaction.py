from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.context_compiler import compile_context_packet
from sphere_cli.execution_ledger import record_postrun
from sphere_cli.project_state import load_project_state, write_memory
from sphere_cli.security import REDACTION, redact_payload, redact_secrets


class SecurityRedactionTests(unittest.TestCase):
    def test_redact_common_secret_patterns(self) -> None:
        text = "api_key=sk-testsecret123456 Authorization: Bearer abc.def password=hunter2"
        redacted = redact_secrets(text)

        self.assertNotIn("sk-testsecret", redacted)
        self.assertNotIn("abc.def", redacted)
        self.assertNotIn("hunter2", redacted)
        self.assertIn(REDACTION, redacted)

    def test_redact_payload_secret_keys(self) -> None:
        payload = redact_payload({"token": "abc", "nested": {"note": "password=secret"}})

        self.assertEqual(payload["token"], REDACTION)
        self.assertNotIn("secret", payload["nested"]["note"])

    def test_memory_and_ledger_writeback_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            write_memory(root, memory_type="fact", project="DysonSpherain", content="api_key=sk-abcdef1234567890", source="unit")
            run = record_postrun(root, project="DysonSpherain", summary="password=hunter2", source="unit")
            state = load_project_state(root, "DysonSpherain")

            self.assertTrue(state.source_memory_ids)
            self.assertNotIn("hunter2", str(run.metadata))
            self.assertIn(REDACTION, str(run.metadata))

    def test_context_packet_is_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packet = compile_context_packet(
                Path(tmp),
                task="Fix issue with api_key=sk-abcdef1234567890",
                project="DysonSpherain",
            )

            self.assertNotIn("sk-abcdef", packet)
            self.assertIn(REDACTION, packet)


if __name__ == "__main__":
    unittest.main()
