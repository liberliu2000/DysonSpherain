# Reproducibility

Formal validation requirements:

- `SPHERE_EMBEDDING_FAIL_FAST=1`
- `SPHERE_VECTOR_BACKEND=chroma`
- `fallback_in_use=false`
- exact command recorded
- elapsed time recorded
- dataset path/version recorded
- question/sample count recorded
- route policy/config hash recorded when applicable
- generated tables read from artifacts

Missing baselines, ablations, CI, or held-out runs must be marked pending rather
than hand-filled.

Current formal status:

- protocol: `passed`
- pending formal rows: `0`
- unavailable-model baseline rows: `8` blocked, not completed
- CloneMem formal full run: `clonemem-a40fef2bed79`
- CloneMem route-policy hash:
  `9144917e6cd3c9e3eeef64bc709e02cbf2150b22b33c3a5f99cc05686c8bde51`

Older CloneMem full runs are not matched formal comparisons when config hash,
dataset version, or route policy differs.
