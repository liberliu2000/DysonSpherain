from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "base"
WEB = ROOT / "web"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

playwright_sync = pytest.importorskip("playwright.sync_api")

from dysonspherain.adapters.daemon import DysonMemoryHandler
from dysonspherain.memory_os.observation_store import write_token_economy_event
from dysonspherain.product import mark_deprecated, remember


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_url(url: str, timeout: float = 25.0) -> None:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # pragma: no cover - diagnostic path
            last = exc
            time.sleep(0.4)
    raise RuntimeError(f"Timed out waiting for {url}: {last}")


class MemoryCockpitNextPlaywrightTests(unittest.TestCase):
    def test_memory_cockpit_core_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remember(root, project_id="DysonSpherain", text="Playwright cockpit should explain lifecycle retrieval.", evidence_type="decision", tags=["ui"])
            remember(root, project_id="DysonSpherain", text="Playwright duplicate compaction memory.", evidence_type="note", tags=["compact"])
            remember(root, project_id="DysonSpherain", text="Playwright duplicate compaction memory.", evidence_type="note", tags=["compact"])
            stale = remember(root, project_id="DysonSpherain", text="Playwright deprecated memory should appear in conflict review.", evidence_type="note")
            mark_deprecated(root, stale["capsule_id"], reason="playwright review")
            write_token_economy_event(
                root,
                project="DysonSpherain",
                session_id="next-ui",
                prompt="memory cockpit",
                decision="inject",
                injected_tokens=120,
                baseline_context_tokens=1000,
                estimated_saved_tokens=880,
                budget_usage_ratio=0.12,
            )

            handler = type("NextUiHandler", (DysonMemoryHandler,), {"base_dir": root, "project": "DysonSpherain"})
            api = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            api_thread = threading.Thread(target=api.serve_forever, daemon=True)
            api_thread.start()
            web_port = free_port()
            env = {**os.environ, "NEXT_PUBLIC_DYSON_API_BASE": f"http://127.0.0.1:{api.server_port}"}
            proc = subprocess.Popen(
                ["npm", "run", "dev", "--", "--hostname", "127.0.0.1", "--port", str(web_port)],
                cwd=WEB,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_for_url(f"http://127.0.0.1:{web_port}")
                with playwright_sync.sync_playwright() as p:
                    try:
                        browser = p.chromium.launch()
                    except Exception as exc:  # pragma: no cover
                        pytest.skip(f"Playwright Chromium is not installed: {exc}")
                    page = browser.new_page(viewport={"width": 1360, "height": 1000})
                    page.goto(f"http://127.0.0.1:{web_port}", wait_until="networkidle")
                    self.assertIn("Control what the system remembers", page.locator("body").inner_text())

                    page.locator("textarea").first.fill("explain lifecycle retrieval")
                    page.get_by_role("button", name="Explain retrieval").click()
                    page.wait_for_function("document.body.innerText.includes('Pipeline')")
                    self.assertIn("Selected", page.locator("body").inner_text())

                    page.get_by_role("button", name="Run deterministic compaction").first.click()
                    page.wait_for_function("document.body.innerText.includes('Compaction result preview')")
                    self.assertIn("sources:", page.locator("body").inner_text())

                    page.get_by_role("button", name="Mark Deprecated").first.click()
                    page.wait_for_function("document.body.innerText.includes('marked deprecated')")

                    page.get_by_role("button", name="Save settings").click()
                    page.wait_for_function("document.body.innerText.includes('Settings saved')")
                    browser.close()
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                api.shutdown()
                api_thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
