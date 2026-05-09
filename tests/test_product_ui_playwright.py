from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

playwright_sync = pytest.importorskip("playwright.sync_api")

from dysonspherain.adapters.daemon import DysonMemoryHandler
from dysonspherain.memory_os.observation_store import write_token_economy_event
from dysonspherain.product import maintenance_suggestions, remember


class ProductUiPlaywrightTests(unittest.TestCase):
    def test_evidence_cockpit_core_interactions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remember(
                root,
                project_id="DysonSpherain",
                evidence_type="decision",
                text="Playwright interaction test should retrieve product dense evidence.",
                tags=["ui", "playwright"],
            )
            remember(
                root,
                project_id="DysonSpherain",
                evidence_type="decision",
                text="Playwright maintenance duplicate confirmation.",
                tags=["ui", "maintenance"],
            )
            remember(
                root,
                project_id="DysonSpherain",
                evidence_type="decision",
                text="Playwright maintenance duplicate confirmation.",
                tags=["ui", "maintenance"],
            )
            maintenance_suggestions(root, project_id="DysonSpherain")
            write_token_economy_event(
                root,
                project="DysonSpherain",
                session_id="ui-test",
                prompt="show token economy",
                decision="inject",
                injected_tokens=200,
                baseline_context_tokens=1000,
                estimated_saved_tokens=800,
                budget_usage_ratio=0.2,
            )

            handler = type("ProductUiHandler", (DysonMemoryHandler,), {"base_dir": root, "project": "DysonSpherain"})
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with playwright_sync.sync_playwright() as p:
                    try:
                        browser = p.chromium.launch()
                    except Exception as exc:  # pragma: no cover - depends on local browser install
                        pytest.skip(f"Playwright Chromium is not installed: {exc}")
                    page = browser.new_page(viewport={"width": 1280, "height": 900})
                    page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")

                    page.get_by_role("button", name="Evidence Search").click()
                    page.locator("#productQuery").fill("dense evidence")
                    page.locator("#evidenceSearch").get_by_role("button", name="Search").click()
                    page.wait_for_function("document.querySelector('#productDetail').innerText.includes('dense_probe')")
                    self.assertIn("dense_probe", page.locator("#productDetail").inner_text())

                    page.get_by_role("button", name="Context Composer").click()
                    page.locator("#ctxQuery").fill("dense evidence")
                    page.locator("#contextComposer").get_by_role("button", name="Build Context").click()
                    page.wait_for_function("document.querySelector('#ctxPreview').innerText.includes('Supporting Evidence')")
                    self.assertIn("Supporting Evidence", page.locator("#ctxPreview").inner_text())

                    page.get_by_role("button", name="Settings").click()
                    self.assertIn("encryption", page.locator("#productSettings").inner_text())

                    page.get_by_role("button", name="Maintenance").click()
                    self.assertIn("Maintenance", page.locator("#maintenance").inner_text())
                    self.assertIn("Rebuild Index", page.locator("#maintenance").inner_text())
                    with page.expect_dialog() as dialog_info:
                        page.locator("#maintenance").get_by_role("button", name="Dismiss").first.click()
                    dialog = dialog_info.value
                    self.assertIn("Dismiss this maintenance suggestion", dialog.message)
                    dialog.dismiss()
                    with page.expect_dialog() as dialog_info:
                        page.locator("#maintenance").get_by_role("button", name="Dismiss").first.click()
                    dialog_info.value.accept()
                    page.wait_for_function("document.querySelector('#maintenanceDetail').innerText.includes('dismissed')")
                    self.assertIn("dismissed", page.locator("#maintenanceDetail").inner_text())
                    page.locator("#vectorBackend").select_option("chroma")
                    page.get_by_role("button", name="Save Vector Backend").click()
                    page.wait_for_function("document.querySelector('#maintenanceDetail').innerText.includes('chroma')")
                    self.assertIn("chroma", page.locator("#maintenanceDetail").inner_text())
                    page.get_by_role("button", name="Rebuild Vector Index").click()
                    page.wait_for_function("document.querySelector('#maintenanceDetail').innerText.includes('backend')")
                    self.assertIn("backend", page.locator("#maintenanceDetail").inner_text())
                    page.locator("#embeddingBackend").select_option("local_hash_embedding")
                    page.get_by_role("button", name="Save Embedding Backend").click()
                    page.wait_for_function("document.querySelector('#maintenanceDetail').innerText.includes('local_hash_embedding')")
                    self.assertIn("local_hash_embedding", page.locator("#maintenanceDetail").inner_text())
                    browser.close()
            finally:
                server.shutdown()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
