# DysonSpherain Regression Diagnosis

## Scope

- Repo root audited: `README.md`, `draft.md`, `base/sphere_cli/*`, `base/benchmarks/*`, `tests/*`.
- Artifact roots used for the final read:
  `/Users/yanbo/DysonSpherain/BenchmarkResult/20260425_regression_full_v5`
  `/Users/yanbo/DysonSpherain/BenchmarkResult/20260426_regression_repair_full_v1`

## Current Diagnosis

1. Silent embedding fallback is no longer the active regression source.
   - Full repaired run stayed on `sentence-transformers/all-MiniLM-L6-v2`.
   - `fallback_in_use=false` on all four factual benchmarks.

2. v5's biggest engineering bug was destructive fusion, not missing channels.
   - Safe fusion fixed the earlier `fused_hit@100 < dense_hit@100` pattern.
   - Duplicate collapse, near-duplicate collapse, parent caps, and inhibition are no longer removing broad-pool gold evidence on the repaired route.

3. LongMemEval and LoCoMo are not broad-recall limited.
   - Both run at `candidate_recall@100=1.0000`.
   - The remaining issue is late ordering quality, especially on LoCoMo.

4. KnowMe improved after regression repair, but still shows route-specific admission and mapping gaps.
   - Full repaired result:
     `segment recall_frac@10=0.5767`
     `candidate_recall@100=0.7597`
   - Main residual failures:
     `query_gold_mapping_empty=143`
     `parent_hit_segment_miss=124`
     `gold_missing_from_candidate_pool=99`

5. CloneMem is still dominated by first-stage candidate admission.
   - Full repaired result:
     `segment recall_frac@10=0.0894`
     `candidate_recall@100=0.3434`
   - Oracle remains healthy:
     `oracle_recall@10=1.0000`
   - Main failures:
     `parent_hit_segment_miss=651`
     `lexical_miss=385`
     `temporal_miss=240`

## Conclusion

- Regression repair succeeded for the non-destructive fusion objective.
- KnowMe improved enough to cross the current phase-1 target.
- LongMemEval stayed strong.
- LoCoMo needs ordering recovery.
- CloneMem still needs stronger parent-to-segment expansion and broader candidate admission, not more destructive fusion.
