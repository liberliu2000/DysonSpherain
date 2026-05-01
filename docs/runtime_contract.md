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
