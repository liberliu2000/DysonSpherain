# DysonSpherain Memory Runtime Upgrade Report

## Scope

This upgrade follows `dyson_spherain_original_technical_route.md` by adding an original memory runtime layer on top of the existing DysonSpherain codebase. It does not replace existing benchmark runners, retrieval pipelines, MCP tools, or paper artifacts.

## Implemented

- Added `base/dysonspherain/memory_runtime/`.
- Implemented append-only `MemoryEvent` ledger with stable content-hash deduplication and JSONL replay.
- Implemented deterministic `Task Situation Graph` projection and projection exports under `data/projections/`.
- Implemented an `Evidence Routing VM` with intent inference, dynamic evidence programs, operator specs, candidate traces, and a non-fixed retrieval flow.
- Added operator-specific runtime behavior for dense search, artifact lookup, metric delta scan, code-region lookup, causal-neighbor expansion, contradiction scan, similar task lookup, and preference/constraint lookup. `dense_semantic_search` now prefers the project vector store and falls back explicitly to deterministic token-cosine when vector results are unavailable.
- Implemented `Context Budget Compiler` with configurable scoring weights, token budget selection, dynamic sections, and omitted-evidence reasons.
- Implemented `Self-Verifying Recall` audit with constraint coverage, token efficiency, contradiction, provenance, freshness, diversity, supersession, and decision-context checks.
- Implemented `Active Memory Scheduler` with idempotent job ids, a persistent queue, job logs, projection refresh, metric extraction, contradiction report, recovery summary, ledger vector indexing, index freshness report, and safe compaction plans.
- Added generic CLI, Codex, Claude Code, and manual import adapter interfaces.
- Added CLI commands:
  - `dysonspherain memory append`
  - `dysonspherain memory recall`
  - `dysonspherain memory graph`
  - `dysonspherain memory replay`
  - `dysonspherain memory compact --safe`
  - `dysonspherain memory scheduler`
  - extended `dysonspherain memory audit --last`
  - extended `dysonspherain memory explain --last-recall`
- Added daemon APIs for Glass-Cockpit data:
  - `/api/runtime/ledger`
  - `/api/runtime/graph`
  - `/api/runtime/cockpit`
  - `/api/runtime/config`
  - `/api/runtime/scheduler`
  - `/api/runtime/scheduler/enqueue`
  - `/api/runtime/scheduler/run-once`
  - `/api/runtime/latest-packet`
  - `/api/runtime/latest-audit`
- Upgraded the Web UI into a minimal eight-view Retrieval Glass-Cockpit:
  - Mission Control
  - Memory Ledger
  - Situation Graph
  - Evidence Router
  - Context Compiler
  - Recall Audit
  - Active Scheduler
  - Configuration Studio
- Added an interactive SVG Situation Graph canvas with node/edge rendering, timeline playback controls, and click-to-inspect node detail.
- Added runtime configuration persistence under `data/config/memory_runtime_config.json`; config changes write ledger events. Configuration Studio now edits operator weights, section limits, scheduler settings, backend choices, and import/export payloads with high-risk confirmation.
- Wired Claude Code hooks to append structured `MemoryEvent` records and enqueue index refresh jobs while preserving the existing observation/project-memory write paths.
- Added `scripts/run_smoke_bench.py` to create smoke reports with embedding backend, fallback state, index freshness, context packet trace, and optional real benchmark-runner invocation through `--run-runner`. Silent fallback is blocked unless `--allow-fallback` is passed.

## Design Notes

- SQLite, MCP, and the existing observation store remain adapters/backends, not the new architecture center.
- The new runtime centers on ledger events, deterministic projections, evidence programs, budgeted packets, and recall audit traces.
- Context injection now has explicit omission trace, so UI and debugging can show what was not injected and why.
- Maintenance jobs write ledger events and can be rerun safely using stable idempotency keys.
- Scheduler jobs can run one-shot, drain from the persistent queue, or run as a polling daemon with `dysonspherain memory scheduler --daemon`.
- Platform-level supervisor configs can be generated for macOS `launchd` and Linux `systemd --user` with `dysonspherain adapters install-supervisor`; the generated services cover `memory-daemon` and `memory-scheduler`, with optional `--activate` for immediate start.
- Hook-written ledger events enqueue `index_staleness_detected` jobs; queued refresh jobs upsert ledger event chunks into the project vector store and write explicit fallback/error metadata.

## Tests Added

- `tests/memory_runtime/test_ledger.py`
- `tests/memory_runtime/test_situation_graph.py`
- `tests/memory_runtime/test_evidence_context_audit_scheduler.py`
- `tests/memory_runtime/test_config.py`
- `tests/memory_runtime/test_adapters.py`
- `tests/memory_runtime/test_smoke_bench.py`
- `tests/integration/test_full_memory_flow.py`

Covered requirements:

- ledger idempotency
- situation graph replay determinism
- evidence program compilation
- context budget compiler budget enforcement
- context packet omission trace
- recall audit missing-constraint detection
- scheduler job idempotency
- runtime configuration writeback
- agent adapter capture
- smoke report fallback blocking and runtime trace fields
- scheduler queue drain and ledger vector index report
- Claude hook ledger writeback
- cockpit scheduler endpoints, graph timeline controls, and configuration editor fields
- smoke bench real-runner passthrough
- CLI recall command
- end-to-end append -> graph -> recall -> audit -> replay flow

## Validation

- Focused runtime/hook/daemon/smoke tests: `29 passed`
- Full project tests: `276 passed, 24 warnings`
- CLI smoke:
  - `dysonspherain memory append --content ...`: wrote an append-only event under `data/ledger/events_202604.jsonl`
  - `dysonspherain memory scheduler --trigger session_ended --run-once`: scheduled and ran projection/next-session jobs
  - `dysonspherain memory scheduler --trigger artifact_updated --enqueue` plus `--drain-queue`: persists and drains scheduler jobs
  - `dysonspherain memory recall "继续 DysonSpherain memory runtime upgrade" --budget 800 --trace`: produced a budgeted context packet
  - `dysonspherain memory audit --last`: returned a medium-risk audit with a missing-constraint follow-up suggestion
  - `dysonspherain memory graph --table`: rendered the projected task situation graph
- Cockpit API smoke:
  - `/api/runtime/cockpit`: mission control, graph, packet, audit, config snapshot
  - `/api/runtime/config`: read/write runtime configuration
- Benchmark smoke report:
  - `python scripts/run_smoke_bench.py --dataset longmemeval --n 2 --output-dir reports/smoke_runtime_check`
  - `python scripts/run_smoke_bench.py --dataset longmemeval --n 2 --run-runner --data-path ... --out ...`: invokes the real benchmark runner interface and records metrics path/status
  - fallback blocking verified with `ALLOW_EMBEDDING_FALLBACK=1`, returning exit code `2`

## Known Limits

- Dense semantic search is wired to the project vector store when available; empty/unavailable vector results explicitly fall back to deterministic token-cosine with trace metadata.
- Situation Graph edges are substantially richer, including explicit payload relations, and the UI now includes an SVG graph canvas with playback over recent ledger events.
- Follow-up recall remains capped to one round by design.
- Safe compaction writes a compaction plan and preserves all raw ledger events; it does not physically delete events.

## Remaining Follow-Up

- The scheduler queue is durable and drainable from CLI/API/daemon mode, and platform-specific launchd/systemd user-service config generation is implemented. Activation remains explicit through `--activate`.
- Real runner smoke invocation is wired; full benchmark validation still depends on the user-provided dataset paths and runtime budget.
