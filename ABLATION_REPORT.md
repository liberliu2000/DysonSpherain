# ABLATION REPORT

## Effective Changes This Round

1. Dense-preserving safe fusion.
   - Fixed the v5 destructive-fusion regression.
   - Full repaired run satisfied `fused_hit@100 >= dense_hit@100` on all four factual benchmarks.

2. Channel gating plus lighter inhibition.
   - Prevented broad-pool gold loss from duplicate collapse, parent caps, and inhibition.
   - `inhibition_suppressed_gold_count=0` on the repaired full runs.

3. Candidate-generation improvements mattered unevenly.
   - KnowMe benefited in both `candidate_recall@100` and the primary metric.
   - CloneMem benefited in smoke / medium safe-fusion behavior, but the full benchmark remained capped by low broad recall.

## Medium Slice Read

- KnowMe:
  `dense_only` `candidate_recall@100=0.7850`
  `full_multichannel_safe` `candidate_recall@100=0.8500`
- CloneMem:
  `dense_only` `segment recall_frac@10=0.0993`
  `full_multichannel_safe` `segment recall_frac@10=0.1762`
  but `candidate_recall@100` stayed at `0.4834`

## Full Read

- LongMemEval: roughly flat to slightly improved.
- LoCoMo: small final-metric regression despite saturated candidate recall.
- KnowMe: clear improvement over v5.
- CloneMem: no destructive-fusion regression anymore, but still not a net full-benchmark win.
