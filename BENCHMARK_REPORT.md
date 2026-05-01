# BENCHMARK REPORT

## Current Full Validation

- Full repaired result root:
  `/Users/yanbo/DysonSpherain/BenchmarkResult/20260426_regression_repair_full_v1`
- Regression baseline root:
  `/Users/yanbo/DysonSpherain/BenchmarkResult/20260425_regression_full_v5`

## Main Results

| Benchmark | Primary | Secondary | candidate_recall@100 | dense_hit@100 | fused_hit@100 |
| --- | --- | --- | --- | --- | --- |
| LongMemEval | `Recall@10=0.9920` | `Recall@5=0.9700`, `NDCG@10=0.9180` | `1.0000` | `1.0000` | `1.0000` |
| LoCoMo | `session recall_frac@10=0.8986` | `session recall_any@10=0.9335`, `session ndcg_any@10=0.7363` | `1.0000` | `0.9980` | `0.9980` |
| KnowMe | `segment recall_frac@10=0.5767` | `segment recall_any@10=0.5515`, `segment ndcg_any@10=0.4160` | `0.7597` | `0.7030` | `0.7356` |
| CloneMem | `segment recall_frac@10=0.0894` | `segment recall_any@10=0.2405`, `segment ndcg_any@10=0.1306` | `0.3434` | `0.6816` | `0.6816` |

## Before / After vs v5

| Benchmark | v5 Primary | Repaired Primary | v5 Cand@100 | Repaired Cand@100 |
| --- | --- | --- | --- | --- |
| LongMemEval | `0.9900` | `0.9920` | `0.9980` | `1.0000` |
| LoCoMo | `0.9048` | `0.8986` | `1.0000` | `1.0000` |
| KnowMe | `0.5453` | `0.5767` | `0.7257` | `0.7597` |
| CloneMem | `0.0937` | `0.0894` | `0.3493` | `0.3434` |

## Smoke / Medium Read

- Safe-fusion smoke and medium slices removed the v5 destructive-fusion symptom.
- KnowMe medium improved from `candidate_recall@100=0.7850` to `0.8500`.
- CloneMem medium improved top-10 metrics under safe fusion without dropping dense candidate hit rate.
- Full-run outcome remained mixed:
  LongMemEval strong,
  LoCoMo slightly down on ordering,
  KnowMe improved,
  CloneMem still bottlenecked by first-stage candidate admission.
