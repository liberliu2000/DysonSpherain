from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class SmokeBenchTests(unittest.TestCase):
    def test_smoke_bench_report_records_required_runtime_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            proc = subprocess.run(
                [sys.executable, "scripts/run_smoke_bench.py", "--dataset", "longmemeval", "--n", "2", "--output-dir", str(out)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["status"], "ok")
            self.assertIn("embedding_backend", payload)
            self.assertIn("fallback_in_use", payload)
            self.assertIn("index_freshness", payload)
            self.assertIn("context_packet_trace", payload)
            self.assertTrue((out / "longmemeval_smoke_report.json").exists())

    def test_smoke_bench_blocks_unallowed_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "ALLOW_EMBEDDING_FALLBACK": "1"}
            proc = subprocess.run(
                [sys.executable, "scripts/run_smoke_bench.py", "--dataset", "locomo", "--n", "2", "--output-dir", tmp],
                cwd=ROOT,
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(proc.returncode, 2)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["error"], "silent_fallback_blocked")

    def test_smoke_bench_can_call_real_runner_interface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "fake_runner.py"
            runner.write_text(
                "import argparse, json\n"
                "from pathlib import Path\n"
                "p=argparse.ArgumentParser(); p.add_argument('data', nargs='?'); p.add_argument('--limit'); p.add_argument('--out', type=Path); a=p.parse_args();\n"
                "a.out.write_text(json.dumps({'status':'ok','limit':a.limit})+'\\n')\n",
                encoding="utf-8",
            )
            data = root / "data.json"
            data.write_text("[]\n", encoding="utf-8")
            out = root / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_smoke_bench.py",
                    "--dataset",
                    "longmemeval",
                    "--n",
                    "2",
                    "--output-dir",
                    str(out),
                    "--run-runner",
                    "--runner-script",
                    str(runner),
                    "--data-path",
                    str(data),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["runner"]["status"], "ok")
            self.assertTrue(payload["runner"]["metrics_exists"])


if __name__ == "__main__":
    unittest.main()
