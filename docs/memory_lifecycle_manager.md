# Memory Lifecycle Manager

DysonSpherain stores long-horizon project memory as evidence capsules. The lifecycle layer adds governance metadata around those capsules without replacing the existing store.

## States

- `active`: eligible for retrieval by default.
- `stable`: eligible for retrieval by default with long-term preference semantics.
- `canonical`: compacted memory generated from source capsules; represented as `evidence_type=canonical`.
- `compacted`: raw/source memory included in a canonical capsule. It is preserved and traceable, but excluded by default.
- `superseded`: replaced by newer memory.
- `deprecated`: intentionally no longer recommended.
- `contradicted`: conflicts with newer or stronger evidence.
- `archived`: retained but excluded from default retrieval.

Default retrieval prefers `active`, `stable`, and `canonical` memory. Excluded memories remain visible through retrieval traces and lifecycle APIs.

## APIs

- `GET /api/lifecycle/summary`
- `GET /api/memory/summary`
- `GET /api/memory/list`
- `GET /api/memory/{memory_id}`
- `GET /api/memory/{memory_id}/sources`
- `GET /api/memory/{memory_id}/audit`
- `POST /api/memory/{memory_id}/mark-superseded`
- `POST /api/memory/{memory_id}/mark-deprecated`
- `POST /api/memory/{memory_id}/archive`

## CLI

```bash
dysonspherain memory summary --project DysonSpherain
dysonspherain memory list --project DysonSpherain --state active
dysonspherain memory product-inspect <memory_id>
dysonspherain memory mark-superseded <old_id> --by <new_id> --reason "newer plan"
dysonspherain memory mark-deprecated <memory_id> --reason "obsolete"
dysonspherain memory migrate-lifecycle --backup
```
