# DysonSpherain Baseline Verification

## Verdict

Accepted with caveats.

This accepted baseline is a mixed audited surface:

- verified current-code LongMemEval on the repaired latest snapshot
- audited historical 2026-04-17 full-rerun metrics for LoCoMo, KnowMe, and CloneMem

## Verified current-code result

- code root:
  - `/home/liber/Projects/DysonSpherain/sphere_memory_cli_next_main_code_20260417_164120`
- dataset:
  - reused LongMemEval cleaned JSON from the old workspace
- command path:
  - `.venv/bin/python benchmarks/longmemeval_benchmark.py ... --mode evidence --granularity session --top-k 50 --rerank-mode rule --shell 2 --sector knowledge --zone longmemeval`
- scope:
  - 500 questions
- metrics:
  - `Recall@5 = 0.8080`
  - `Recall@10 = 0.8460`
  - `NDCG@10 = 0.7683`
- comparison to the audited historical LongMemEval line:
  - `Recall@10`: `0.8180 -> 0.8460` (`+0.0280`)
  - `NDCG@10`: `0.6884 -> 0.7683` (`+0.0799`)
- implication:
  - the repaired latest snapshot is runnable, comparable on LongMemEval, and still clearly below the user-stated `0.982` target

## Audited historical reference metrics

- LongMemEval:
  - `Recall@10 = 0.8180`
  - `NDCG@10 = 0.6884`
- LoCoMo session:
  - `recall_frac@10 = 0.7669`
  - `recall_any@10 = 0.8122`
  - `ndcg_any@10 = 0.5530`
- KnowMe segment:
  - `recall_frac@10 = 0.3211`
  - `recall_any@10 = 0.2901`
  - `ndcg_any@10 = 0.1825`
- CloneMem segment:
  - `recall_frac@10 = 0.0537`
  - `recall_any@10 = 0.1693`
  - `ndcg_any@10 = 0.0849`

## Caveats

- The current accepted surface is mixed-provenance rather than a single full multi-benchmark rerun on one code snapshot.
- Only LongMemEval has been revalidated end-to-end on the repaired latest snapshot in this round.
- LoCoMo, KnowMe, and CloneMem remain trusted historical references until current-snapshot reruns exist.
- The user-stated higher scores are still unverified and should not be treated as the active baseline.

## Evidence paths

- repaired full validation log:
  - `/home/liber/Projects/DeepScientist/.ds-home/quests/002/.ds/bash_exec/bash-ce8636c5/terminal.log`
- repaired full validation JSON:
  - `/home/liber/Projects/DeepScientist/.ds-home/quests/002/tmp/longmemeval_validation_full.json`
- repaired smoke validation note:
  - `/home/liber/Projects/DeepScientist/.ds-home/quests/002/handoffs/longmemeval-smoke-verification.md`
- baseline normalization audit:
  - `/home/liber/Projects/DeepScientist/.ds-home/quests/002/handoffs/intake-audit-baseline-normalization.md`

## Next anchor

Move to `idea`.

The first improvement direction should target the biggest validated gaps rather than trying to polish an already-solved line:

- LongMemEval still trails the user target by `0.1360` on `Recall@10`
- KnowMe and CloneMem historical reference scores remain especially weak
- any cross-benchmark improvement claim must keep provenance explicit until those benchmarks are rerun on the current code line
