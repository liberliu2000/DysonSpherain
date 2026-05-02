# Compaction System

Compaction creates an additional canonical memory from redundant source memories. It never deletes or overwrites raw/source memory.

## Modes

- `deterministic`: normalized exact duplicate detection and deterministic canonical summary generation.
- `local_semantic`: deterministic mode plus local lexical/embedding similarity clustering.
- `hybrid`: currently runs the local path first; external LLM use remains disabled unless configured.

## Candidate Detection

Candidates are produced from:

- normalized content hash exact duplicates,
- local similarity clusters using lexical Jaccard and deterministic local embedding cosine similarity.

Each cluster reports source IDs, memory count, average similarity, estimated input/output tokens, estimated saved tokens, and a suggested action.

## Reviewable Result Flow

The product API supports a non-destructive review flow:

```text
cluster -> run preview -> verify -> commit | reject
```

Running a cluster through `/api/compaction/clusters/{cluster_id}/run` creates a persisted compaction result under `.memory/compaction_results/`. It does not mutate source memories. Commit is the only step that writes the canonical capsule and marks source capsules as `compacted`.

## Verification

Before commit, the verifier checks:

- source IDs are present,
- canonical content is not empty,
- output length is within bounds,
- source memories still exist,
- raw memories are preserved.

If verification fails, the result is returned as `needs_review` and no canonical memory is committed.

## APIs and CLI

```bash
GET  /api/compaction/clusters
POST /api/compaction/clusters/{cluster_id}/run
GET  /api/compaction/results/{result_id}
POST /api/compaction/results/{result_id}/verify
POST /api/compaction/results/{result_id}/commit
POST /api/compaction/results/{result_id}/reject

dysonspherain compaction candidates
dysonspherain compaction run <cluster_id> --mode local_semantic
dysonspherain compaction run <cluster_id> --mode local_semantic --commit
```
