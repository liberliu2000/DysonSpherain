# Runtime Contract

Agents should call the product memory layer at five lifecycle points.

- Before task: `dyson runtime before-task --task ...`
- During task: `dyson runtime during-task --summary ...`
- On error: `dyson runtime on-error --error-file ...`
- After task: `dyson runtime after-task --summary ...`
- Before context compaction: `dyson runtime pre-compact`
- Before benchmark: `dyson runtime before-benchmark --benchmark ...`
- After benchmark completion: `dyson runtime after-benchmark --metrics ...`
- Before commit: `dyson runtime before-commit --summary ...`
- After commit: `dyson runtime after-commit --commit ...`
- Manual checkpoint: `dyson runtime manual-checkpoint --summary ...`

Each call records a `runtime_events` row and returns a context pack path. Error,
task, commit, manual checkpoint, and benchmark events also create evidence
capsules so future retrieval can cite the event instead of relying on raw logs
only.

The HTTP API provides equivalent endpoints through `POST /api/retrieve` and
`POST /api/context-pack`, plus capsule creation through `POST /api/capsules`.

## Token Economy Events

Every agent adapter that injects memory context should write a token economy
ledger event. The event records adapter, task type, mode, decision, risk,
baseline type, candidate context tokens, final injected tokens, estimated saved
tokens, duplicate ratio, tokenizer fallback state, source files, and quality
guard status.

The runtime distinguishes:

- LLM prompt token saving: `baseline_context_tokens - final_injected_tokens`.
- Local compute saving: cache/runtime work avoided locally.

Local compute saving must not be added to LLM prompt token saving. Use
`GET /api/token-economy` or `dysonspherain evaluate-token-economy` artifacts to
inspect both categories.
