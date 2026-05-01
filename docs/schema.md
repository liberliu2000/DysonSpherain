# Product SQLite Schema

The product database is `.memory/dyson_product.sqlite3`.

Tables:

- `raw_traces`: original text or blob references, source type, project, session,
  task, agent, timestamps, and metadata.
- `evidence_capsules`: canonical evidence records with title, summary, entity,
  tag, file, code, command, artifact, benchmark, git, validity, relation,
  importance, confidence, sparse terms, route features, and metadata fields.
- `evidence_capsules_fts`: FTS5 index over title, summary, raw text, tags, and
  entities.
- `capsule_entities`, `capsule_relations`, `capsule_artifacts`,
  `capsule_embeddings`: normalized auxiliary indexes. Product capsules write
  vectors into `capsule_embeddings` for local `dense_probe` retrieval. The
  default backend is deterministic `local_hash_embedding`; optional semantic
  backends such as `sentence_transformers` store backend, model, version, and
  source-hash metadata for rebuild/staleness checks.
- Optional product ANN indexes are external to SQLite under
  `.memory/indexes/product_chroma` when the `chroma` vector backend is
  configured. SQLite remains the source of truth; ANN indexes are rebuildable.
- `capsule_aliases`: project-scoped alias to canonical-memory mappings.
- `maintenance_suggestions`: generated duplicate-merge and stale-benchmark
  invalidation suggestions for review workflows. Suggestions carry stable
  `suggestion_id` values and `open`, `applied`, or `dismissed` statuses.
- `retrieval_traces`: route decision, probe results, admitted candidates,
  filtered candidates, reranked candidates, and gold-label audit fields.
- `context_packs`: rendered context packs and structured section payloads.
- `runtime_events`: before-task, error, after-task, pre-compact, and benchmark
  hook events.
- `benchmark_runs`: metrics and artifact references for benchmark-lab tracking.
- `health_reports`: doctor output snapshots.
- `schema_migrations`: applied schema version records.

Schema version is currently `1`. The repository layer creates missing tables on
connect and keeps legacy benchmark artifacts outside this database.

Validity transitions are stored on `evidence_capsules` and linked through
`capsule_relations`. Supported states are `active`, `superseded`, `deprecated`,
`contradicted`, `reverted`, and `unknown`. Supersession and contradiction links
are also mirrored in the capsule JSON relation fields for compact API responses.

Code and benchmark binding metadata is stored inside capsule metadata and
benchmark run metrics payloads. Error capsules include parsed exception type,
message, stack frames, and file references when a Python traceback is captured.
Benchmark run payloads include normalized dataset, quality, latency, artifact,
hardware, and git binding fields.
