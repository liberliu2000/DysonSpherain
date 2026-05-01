# Route Policy Comparability Note

Paper-facing comparisons follow the formal protocol comparability rule: quality deltas are computed only between runs with matching benchmark, run type, question scope, embedding, fallback status, config hash, dataset version, and route policy.

## CloneMem Route-Only Promotion

- run_id: `clonemem-a40fef2bed79`
- artifact_dir: `/Users/yanbo/DysonSpherain/BenchmarkResult/20260429_phase5_lexical_anchor_gate_protected_top3_full_v1/clonemem`
- config_hash: `f13db2302607acf8ede920f3f0087807fbd97166a00178fa95fed6f44fd0eedd`
- route_policy_hash: `9144917e6cd3c9e3eeef64bc709e02cbf2150b22b33c3a5f99cc05686c8bde51`
- formal_comparison_status: `non_comparable`

The CloneMem protected top-3 lexical anchor gate is promoted only through the CloneMem benchmark route policy. Older CloneMem full runs are not treated as matched formal comparisons when their config hash, dataset version, or route policy differs.
