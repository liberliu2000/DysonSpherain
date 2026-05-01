# Product UI And API

Start the local UI:

```bash
dyson ui --project DysonSpherain --host 127.0.0.1 --port 37777
```

Core product API endpoints:

- `GET /api/projects`
- `GET /api/capsules`
- `GET /api/capsules/{id}`
- `POST /api/capsules`
- `PATCH /api/capsules/{id}`
- `POST /api/retrieve`
- `GET /api/retrieval-traces/{id}`
- `POST /api/context-pack`
- `GET /api/context-packs/{id}`
- `GET /api/benchmark-runs`
- `GET /api/maintenance`
- `POST /api/index/rebuild`
- `GET /api/index/embedding-backends`
- `GET /api/index/vector-backends`
- `POST /api/index/configure-embedding`
- `POST /api/index/configure-vector`
- `POST /api/index/rebuild-vector`
- `POST /api/index/configure-encryption`
- `POST /api/maintenance/apply`
- `POST /api/maintenance/dismiss`
- `GET /api/health`
- `GET /api/settings`
- `PATCH /api/settings`
- `GET /api/token-economy`

The existing cockpit UI remains available at `/` and continues to show runtime
ledger, graph, scheduler, token economy, and configuration views.

The Next.js console under `web/` focuses on token economy and memory editing.
It shows recent saved-token windows, a saved-token trend chart, the calculation
formula, editable memory records, recall controls, decision/fallback/over-budget
observability, high-risk file-reference cases, quality-guard violations, and a
separate split for LLM prompt token saving vs local compute saving. Empty or
missing artifact data should render as an empty state rather than crashing.

Evidence Cockpit pages now include:

- Project Dashboard
- Evidence Search
- Retrieval Trace Viewer
- Evidence Timeline
- Evidence Field Graph
- Context Composer
- Benchmark Lab
- Health Doctor
- Maintenance
- Settings

These pages call the product API directly. Evidence Search uses
`/api/capsules` and `/api/retrieve`; Retrieval Trace Viewer uses
`/api/retrieve`; Context Composer uses `/api/context-pack`; Benchmark Lab uses
`/api/benchmark-dashboard`; Health Doctor uses `/api/health`; Maintenance uses
`/api/maintenance`, `/api/index/rebuild`, `/api/index/embedding-backends`,
`/api/index/vector-backends`, `/api/index/configure-embedding`,
`/api/index/configure-vector`, `/api/index/rebuild-vector`,
`/api/maintenance/apply`, and `/api/maintenance/dismiss`; Settings uses
`/api/settings`.
