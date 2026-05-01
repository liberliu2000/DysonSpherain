# REGRESSION REPORT

## Guard Status

- Benchmark mode now hard-fails on:
  `embedding_provider == local_hash`
  `fallback_in_use == true`
  runtime/index fingerprint mismatch
  embedding preprocess-version mismatch
- Current repaired full run:
  all four benchmarks reported `fallback_in_use=false`
  all four integrity reports reported `p0_bugs=[]`

## Safe-Fusion Status

- Hard requirement met:
  `fused_hit@100 >= dense_hit@100` on every repaired full benchmark
- Current full values:
  LongMemEval: `1.0000 >= 1.0000`
  LoCoMo: `0.9980 >= 0.9980`
  KnowMe: `0.7356 >= 0.7030`
  CloneMem: `0.6816 >= 0.6816`

## Target Status

| Benchmark | Guardrail | Current Full | Status |
| --- | --- | --- | --- |
| LongMemEval | `Recall@10 >= 0.95` | `0.9920` | met |
| LoCoMo | `session recall_frac@10 >= 0.90` | `0.8986` | narrowly missed |
| KnowMe | `segment recall_frac@10 >= 0.55` | `0.5767` | met |
| CloneMem | `segment recall_frac@10 >= 0.12` | `0.0894` | missed |

## Remaining Regression Surface

- LongMemEval:
  no broad-recall problem;
  remaining misses are small-tail ordering issues.
- LoCoMo:
  broad recall saturated, but final ordering regressed slightly;
  `reranker_dropped_gold_count=22`.
- KnowMe:
  candidate admission improved and crossed the target;
  residual failures are `query_gold_mapping_empty`, `parent_hit_segment_miss`, and `gold_missing_from_candidate_pool`.
- CloneMem:
  safe fusion no longer harms dense coverage, but broad recall is still weak;
  `candidate_recall@100=0.3434`,
  `parent_hit_segment_miss=651`,
  `lexical_miss=385`,
  `temporal_miss=240`,
  `oracle_recall@10=1.0000`.
