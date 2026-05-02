# UI Memory Cockpit

The Web UI is organized around memory governance rather than benchmark monitoring.

Main surfaces:

- Token savings: recent and total saved-token windows plus trend.
- Lifecycle map: active, stable, canonical, compacted, superseded, deprecated, contradicted, and archived counts.
- Compaction queue: duplicate and near-duplicate clusters with deterministic compaction action.
- Retrieval inspector: query dry-run with selected memories, excluded evidence, and stage counts.
- Memory explorer: searchable/editable memory records with lifecycle state filters.
- Supersession and conflict review: preserved old/conflicting records that are excluded by default.
- Settings: LLM, compaction, scoring, lifecycle, and privacy controls.

The UI intentionally does not show benchmark score cards or leaderboard-style benchmark panels.
