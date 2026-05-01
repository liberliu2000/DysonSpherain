# DysonSpherain Agent Memory Integration

DysonSpherain exposes a local Memory OS layer for coding agents through MCP tools, Claude Code hooks, and project-level policy files.

## MCP Tools

- `dyson_recall`
- `dyson_memory_intent`
- `dyson_context_pack`
- `dyson_write_memory`
- `dyson_project_state`
- `dyson_token_economy_eval`
- `dyson_search_memory`
- `dyson_timeline`
- `dyson_get_observations`
- `dyson_resume_context`

The tools are diagnostic and context assembly adapters. They do not change the default benchmark retrieval pipeline. The server prefers the official Python MCP SDK `FastMCP` stdio transport and falls back to the in-tree JSON-RPC handler only when the SDK package is unavailable.

MCP path access is allowlisted. Tool calls can read/write only under `DYSON_PROJECT_ROOT`, `DYSON_HOME`, the current working directory, or extra roots listed in `DYSON_ALLOWED_PATHS`.

`dyson_memory_intent` is the first-step routing decision: it returns `should_call_memory`, a reason such as `cross_session_continuation`, preferred tools, and a token budget. `dyson_recall` routes through the existing retrieval/context assembly path. `dyson_context_pack` can pack by query, explicit `memory_ids`, or supplied candidate records (`candidates`, `ranked_items`, `memory_objects`) and can render `markdown` or `json` with selected sections. `dyson_project_state` trims the returned state to the requested token budget.

Progressive disclosure uses three tools:

- `dyson_search_memory`: compact search results with stable observation IDs, snippets, citations, and token cost.
- `dyson_timeline`: related events for an observation or session.
- `dyson_get_observations`: full details for selected observation IDs.
- `dyson_resume_context`: compact continuation packet for the latest or selected session.

Observation records are stored in `artifacts/memory_os/observations.sqlite3` with SQLite FTS5, stable `obs_*` IDs, and `dyson://observation/<id>` citations. Existing project memories are projected into this index as `obs_mem_<memory_id>` records.

## Writeback

Session writeback runs sanitizer before dedupe. Exact, normalized, semantic, task-window, and lexical duplicate checks prevent repeated Codex reconnect or Claude hook writes.

`<private>...</private>` regions are redacted before observation storage. `.dysonignore` patterns can skip matching observation content or file refs. Observation maintenance commands:

```bash
dysonspherain memory obs-export --out artifacts/memory_os/observations_export.json
dysonspherain memory obs-delete obs_xxx
dysonspherain memory obs-retention --keep-last 200
```

## Daemon / Web UI

Run the local daemon and minimal Web UI:

```bash
dysonspherain adapters daemon --project-root . --port 37777
```

Install user-level supervisor configs for the daemon and scheduler queue:

```bash
dysonspherain adapters install-supervisor --project .
dysonspherain adapters supervisor-status
```

Add `--activate` on macOS/Linux when you want the installer to call `launchctl`
or `systemctl --user` and start the services immediately. The generated services
are `memory-daemon` for the Web UI/API and `memory-scheduler` for queued
maintenance/index refresh jobs.

HTTP endpoints:

- `/api/health`
- `/api/search?query=...`
- `/api/timeline?observation_id=...`
- `/api/observations/<id>`
- `/api/token-economy`

The root page serves a modern minimal dashboard. It shows:

- a "Resume Last Session" continuation packet;
- per-conversation token savings rows from `token_economy_event` observations;
- total estimated saved tokens for the last 24 hours, 7 days, and 30 days;
- aggregate saving ratio for each window, computed as `sum(saved_tokens) / sum(baseline_context_tokens)`;
- observation search and detail inspection.

`UserPromptSubmit` records token economy events when recalled context is injected or summarized, so the dashboard reflects actual hook usage rather than only offline benchmark reports.

`SessionStart` injects project state plus `dyson_resume_context` when a previous session is available, allowing new windows to pick up recent goal, files changed, tests, failures, next actions, timeline, and token savings.

## Reports

Run:

```bash
dysonspherain adapters write-integration-report --project .
```

The report is written to `artifacts/memory_agent_integration_report.md`.

Current verification should include:

```bash
python -m pytest -q
PYTHONPATH=base python -m dysonspherain.adapters.mcp_server --smoke
```
