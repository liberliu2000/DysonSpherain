# Limitations

- External reranker and BGE/E5 baseline rows remain artifact-backed blocked
  because the required local model artifacts are unavailable; they are not
  hand-filled.
- CloneMem remains the primary unresolved candidate-admission risk.
- The latest channel-tail rescue probe is rejected: on 100k/en medium
  validation it reduced candidate_recall@100 from 0.573973 to 0.560200.
- Some paper comparisons remain non-comparable across older runs because config
  hash, dataset version, fallback state, or run-scope metadata differ.
- Formal claims must continue to distinguish full non-fallback artifacts from
  smoke, diagnostic, rejected, and local_hash fallback artifacts.
