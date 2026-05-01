# Product Development

Focused validation:

```bash
python -m py_compile base/dysonspherain/product/store.py base/dysonspherain/adapters/daemon.py base/sphere_cli/cli.py
python -m pytest tests/test_product_memory.py tests/test_daemon_api.py tests/test_memory_cli.py -q
```

Optional browser interaction validation:

```bash
python -m pip install -e ".[ui-test]"
python -m playwright install chromium
python -m pytest tests/test_product_ui_playwright.py -q
```

The GitHub Actions workflow at `.github/workflows/product.yml` installs the
`ui-test` extra, downloads Chromium, and runs the same product smoke/regression
surface in CI.

Implementation boundaries:

- Keep benchmark runners backward compatible.
- Generate benchmark and UI reports from artifacts or runtime state.
- Keep product storage in `.memory/`.
- Add retrieval probes and vector backends through the product repository or
  existing retrieval modules instead of creating unrelated stores.
- Do not require network access for core commands.
