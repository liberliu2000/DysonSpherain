# Codex MCP Setup

Install project-level Codex MCP config and AGENTS policy:

```bash
dysonspherain adapters install-codex-mcp --project .
```

From npm/npx, use the lightweight wrapper:

```bash
npx dysonspherain-memory install --project .
npx dysonspherain-memory doctor --project .
```

Or install a persistent command:

```bash
npm install -g dysonspherain-memory
dyson-memory install --project .
dyson-memory doctor --project .
```

The wrapper calls the Python CLI, installs Codex MCP config, Claude hooks,
plugin manifests, and then runs doctor. On first use it checks whether the
selected Python can import DysonSpherain; when it cannot, the wrapper creates a
package-local `.dyson-quickstart-venv` and installs the bundled Python package
there. Use `--no-bootstrap` or `DYSON_NO_BOOTSTRAP=1` when you want to manage
Python dependencies yourself.

Plugin-only quick start:

```bash
npx dysonspherain-memory plugin install --project .
npx dysonspherain-memory plugin print
```

Optional user-level supervisor install:

```bash
npx dysonspherain-memory supervisor install --project .
npx dysonspherain-memory supervisor status --project .
```

Pass `--activate` to start the generated macOS `launchd` or Linux `systemd --user`
services immediately.

This safely merges `.codex/config.toml` and `AGENTS.md`. Existing files are not overwritten; a `.bak` is created before modifications.

Verify:

```bash
dysonspherain adapters doctor --project .
```

## Token Economy Workflow

For long-running Codex tasks, the generated `AGENTS.md` policy instructs Codex
to call memory tools in this order:

1. `dyson_memory_intent`
2. `dyson_project_state`, `dyson_recall`, and `dyson_context_pack` when the intent says memory is useful
3. `dyson_token_economy_eval` before injecting candidate context
4. `dyson_write_memory` after the task with files changed, tests run, failures, token economy anomalies, and next actions

Evaluator decisions should be applied directly:

- `inject`: use rendered context.
- `inject_summary_only`: use summary plus file refs.
- `return_file_refs_only`: open the referenced files instead of copying long prose.
- `skip`: read local files directly and avoid memory injection.

Do not paste raw recall result arrays into prompts. The goal is to reduce LLM
prompt tokens while preserving the evidence needed for task quality.

The generated MCP entry runs `python -m dysonspherain.adapters.mcp_server`. Install the optional MCP dependency when SDK transport is required:

```bash
pip install -e ".[mcp]"
```

Without that package the server keeps the same tool surface through the JSON-RPC fallback, but SDK lifecycle coverage should be treated as unavailable.

Install cross-IDE/plugin metadata for Codex, Claude plugin-compatible layouts, Gemini CLI, and OpenCode:

```bash
dysonspherain adapters install-plugin-manifests --project .
```

This writes `.codex-plugin/plugin.json`, `.claude-plugin/plugin.json`, `.gemini/dyson-memory.json`, and `.opencode/dyson-memory.json` with the same MCP command and tool list.
## Product Evidence Tools

The MCP server exposes legacy observation tools and product evidence tools.

Product tools:

- `dyson_product_write`
- `dyson_product_search`
- `dyson_product_retrieve`
- `dyson_product_wake`
- `dyson_product_inspect`
- `dyson_product_update_validity`
- `dyson_product_context_pack`

Runtime tools:

- `dyson_runtime_before_task`
- `dyson_runtime_on_error`
- `dyson_runtime_after_task`
- `dyson_runtime_pre_compact`

Benchmark and health tools:

- `dyson_benchmark_record`
- `dyson_benchmark_compare`
- `dyson_health_doctor`

All product tools use the same local `.memory/dyson_product.sqlite3` store as
the CLI and daemon API. Paths are checked against allowed roots before file
access.
