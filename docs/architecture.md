# Product Architecture

DysonSpherain now has a product evidence layer on top of the existing benchmark
and retrieval code. The product layer is local-first and lives in
`base/dysonspherain/product/`.

Core runtime pieces:

- `product.store`: SQLite-backed raw trace, capsule, retrieval trace, context
  pack, runtime event, benchmark run, and health report repository.
- `sphere_cli.cli`: product CLI commands such as `init`, `record`, `remember`,
  `retrieve`, `wake`, `runtime`, `benchmark-lab`, `index`, and `ui`.
- `dysonspherain.adapters.daemon`: local HTTP API and web UI server.

Storage is under `.memory/` in the active project directory. Small raw traces are
stored inline in SQLite. Large raw traces are written to `.memory/raw/YYYY/MM/DD/`
with SQLite metadata pointing at the blob.

The product layer does not replace the benchmark pipelines. Benchmark scripts and
legacy memory commands remain compatible, while product commands add a stable
evidence-capsule interface for agent and UI workflows.

Retrieval uses a product probe registry. Supported local probes are
`dense_probe`, `sparse_probe`, `proxy_probe`, `temporal_probe`, `entity_probe`,
`artifact_probe`, `code_ref_probe`, and `recent_state_probe`. `dense_probe` uses
deterministic local feature-hash embeddings stored in SQLite, so product memory
search works without network or cloud embedding services. Every retrieval writes
a candidate-admission trace with probe counts, latency, duplicate collapse,
validity filtering, final candidates, and optional gold-label recall fields.

Index maintenance is explicit and auditable. `dyson index embedding-backends`
reports configured and available embedding backends. `local_hash_embedding` is
the dependency-free default, while `sentence_transformers` can be configured
when the optional semantic embedding extra is installed. `dyson index rebuild`
recomputes product embeddings for the configured or selected backend, and
`dyson index repair` combines embedding rebuild with maintenance suggestion
generation. Duplicate-merge and stale-benchmark suggestions have stable IDs,
statuses, and explicit apply/dismiss actions.

For larger product stores, `dyson index vector-backends` reports ANN backend
availability. The default `sqlite_inline` backend scans inline vectors and stays
dependency-free. `chroma` can be configured as a persistent ANN product capsule
index under `.memory/indexes/product_chroma`; `dense_probe` uses it when
available and falls back to SQLite inline search if the optional dependency or
index is missing.

Temporal validity APIs support supersession, contradiction, deprecation,
reversion, active-evidence queries, decision chains, time-slice evidence, and
commit-bound evidence lookup.
