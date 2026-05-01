# SPEED REPORT

## 2026-04-28 Formal Evidence Refresh

- Added artifact-level `dense_bm25_rrf` baselines from existing full
  dense/vector and BM25 outputs. This is a reporting/evidence refresh, not a
  runtime shortcut: no retrieval top-k was reduced and no gold IDs were used in
  candidate generation.
- RRF compact artifacts are now available for all four core benchmarks:
  LongMemEval 500, LoCoMo 1986, KnowMe 1010, and CloneMem 2374 questions.
  Baseline coverage is `25/36`.
- Completed four additional CloneMem Phase 11 medium ablations on the matched
  100k/en diagnostic slice. Coverage is now `12/36`; the new rows are
  diagnostic only and do not change default runtime policy.
- Fixed a chunked-runner efficiency bug: empty CloneMem sample shards now skip
  benchmark subprocess execution instead of accidentally running the full sample
  set. This avoids wasted duplicate work in future sample-sharded medium runs.
- Cleaned the regenerated Phase 11 cache directories after metrics were
  written; no benchmark processes remain active.
- Full test validation after the refresh: `156` unittest cases passed.

## Current Full Timing

- Full repaired result root:
  `/Users/yanbo/DysonSpherain/BenchmarkResult/20260426_regression_repair_full_v1`

| Benchmark | `elapsed_seconds` | `indexing_time_ms` | `retrieval_time_ms` | `fusion_time_ms` | `rerank_time_ms` |
| --- | --- | --- | --- | --- | --- |
| LongMemEval | `191.70` | `87.54` | `81.47` | `0.19` | `0.00` |
| LoCoMo | `668.56` | `68.89` | `37.75` | `0.18` | `0.00` |
| KnowMe | `3144.78` | `196.25` | `2229.25` | `2.12` | `0.02` |
| CloneMem | `7978.52` | `208.30` | `1117.23` | `1.67` | `0.03` |

## Timing Read

- LongMemEval and LoCoMo remain cheap once the index is warm.
- KnowMe is dominated by retrieval and query-side reasoning over richer candidate channels.
- CloneMem is the most expensive full benchmark, but the repair round did not introduce a large fusion or rerank overhead.
- The main latency cost is still first-stage retrieval and workspace/index reuse, not the added diagnostics.

## Retrieval Efficiency Update

_Generated: 2026-04-27_

### Bottleneck Position

- The current full-run timing still points at retrieval, not fusion or rerank:
  KnowMe `retrieval_time_ms=2229.25`,
  CloneMem `retrieval_time_ms=1117.23`,
  while `fusion_time_ms` and `rerank_time_ms` stay near zero.
- This optimization keeps dense preserve, safe fusion, candidate recall audit, oracle reporting, and benchmark-specific candidate generation intact.
  It reduces repeated or global work inside channels instead of lowering global `dense_top_k`, `lexical_top_k`, `candidate_recall_eval_k`, or final pool size.

### Implemented Changes

- Added route-aware channel gating, enabled by default through:
  `SPHERE_ROUTE_AWARE_GATING_ENABLED=true`,
  `SPHERE_ROUTE_AWARE_GATING_AGGRESSIVENESS=safe`.
- Added conservative early exit controls:
  `SPHERE_RETRIEVAL_EARLY_EXIT_ENABLED=true`,
  `SPHERE_RETRIEVAL_LATENCY_BUDGET_MS=0`,
  `SPHERE_RETRIEVAL_MIN_SEED_CANDIDATES=80`,
  `SPHERE_RETRIEVAL_CONFIDENCE_MARGIN=0.12`.
- Extended retrieval side-indexes/postings:
  `token_to_source_ids`,
  `entity_to_source_ids`,
  `temporal_term_to_source_ids`,
  `specific_temporal_term_to_source_ids`,
  `phrase_token_to_source_ids`,
  `session_id_to_source_ids`,
  `parent_id_to_source_ids`,
  `source_id_to_order_index`,
  `source_id_to_session_id`,
  `source_id_to_parent_id`,
  `text_hash_to_source_id`.
- Replaced default full-scan work with indexed fast paths for:
  entity,
  temporal,
  exact phrase,
  session bundle,
  query decomposition,
  and parent/session expansion preselection.
  Legacy fallback remains when side-index fields are missing.
- Tightened indexed-miss behavior:
  when a side-index has the required postings but a query has no posting hits,
  entity,
  temporal,
  exact phrase,
  query decomposition,
  and parent/session expansion now return bounded indexed results instead of silently scanning all `source_records`.
  Full-scan fallback is reserved for missing side-index fields or explicit legacy paths.
- Runtime `retrieve_evidence()` now deduplicates normalized query variants, uses smaller per-variant budgets for simple factual routes, records rank pass count, and keeps a per-call candidate feature cache.
- Updated early exit execution order so it is evaluated after core dense/lexical/entity/temporal/exact channels and before profile dense expansion, query decomposition, session bundle, temporal neighbor, and parent-session expansion.
  When safe conditions hold, `query_decomposition` and bounded expansion channels are skipped before execution rather than only reported as skipped after the fact.
- Split `profile_side_index` behavior so conservative early exit can skip the dense profile subpath while keeping indexed profile candidate generation available for KnowMe identity/profile routes.
- Object-support shortcut handling now has a guardrail test proving a sufficient high-confidence shortcut does not trigger dense/sparse retrieval or a second full rank pass.
- JSON vector backend now reports that it is an O(N) small/offline backend; `SPHERE_VECTOR_BACKEND=chroma` fails fast if Chroma is unavailable.
- Full-suite benchmark execution now has hard runtime red lines in `run_all_benchmarks.py`.
  Per-benchmark warning/fail thresholds are:
  LongMemEval `10/10` min for full 500 queries,
  LoCoMo `45/45` min for full 1986 queries,
  KnowMe `35/35` min for full 1010 queries,
  CloneMem `75/90` min for full 2374 queries.
  The combined suite warns after `2.5h` and fails after `3.5h`.
  Fail thresholds kill the benchmark subprocess group and mark the manifest run as `runtime_timeout`.
- Added generic shard/chunked benchmark execution:
  `knowme_benchmark.py`, `clonemem_benchmark.py`, and `locomo_benchmark.py` now accept `--shard-index`, `--shard-count`, allowlists, `--max-questions`, and `--resume-existing`.
  `run_benchmark_chunked.py` runs each shard in an isolated subprocess/cache root and merges outputs through `merge_benchmark_results.py`.
- CloneMem now defaults to sample-level chunking in `run_benchmark_chunked.py`.
  The failed full run showed question-hash chunking made each shard rebuild the same CloneMem sample workspaces; sample chunking writes per-chunk `sample_id_allowlist.txt` files and runs all questions for the assigned samples only.
  `--shard-strategy question` can still reproduce the older behavior.
- `run_all_benchmarks.py` now supports `--chunked`, `--chunks`, `--workers`, `--per-benchmark-workers`, `--resume`, `--force`, and `--merge-only`.
  Default serial behavior is unchanged.
- Added side-index fast-path audit fields for entity, temporal, exact phrase, profile side index, query decomposition, session bundle, temporal neighbor, and parent session.
  The summary reports `full_scan_channels`, `full_scan_total_records_scored`, `indexed_fast_path_channel_count`, and `legacy_fallback_channel_count`.
- Runtime `retrieve_evidence()` now exposes `SPHERE_RUNTIME_RETRIEVAL_LATENCY_BUDGET_MS` and `SPHERE_RUNTIME_PARALLEL_CHANNELS_ENABLED`.
  Parallel dense/proxy/sparse reads are disabled by default.

### Smoke Profiling

Command:

```bash
./.venv312/bin/python scripts/profile_retrieval_efficiency.py
```

Smoke output on the local sample corpus:

```json
{
  "total_ms": 1.42,
  "dense_ms": 0.08,
  "lexical_ms": 0.02,
  "entity_ms": 0.11,
  "temporal_ms": 0.0,
  "exact_phrase_ms": 0.04,
  "profile_ms": 0.14,
  "decomposition_channel_ms": 0.07,
  "session_bundle_ms": 0.0,
  "temporal_neighbor_ms": 0.0,
  "parent_session_ms": 0.2,
  "fusion_ms": 0.06,
  "rerank_ms": 0.0,
  "inhibition_ms": 0.0,
  "skipped_channels": ["temporal_neighbor"],
  "early_exit_triggered": false,
  "side_index_audit_summary": {
    "full_scan_channels": [],
    "full_scan_total_records_scored": 0,
    "indexed_fast_path_channel_count": 6,
    "legacy_fallback_channel_count": 0
  }
}
```

The profiling script also parses recent benchmark artifacts for before/after context when available.
On the current local artifacts it found the latest CloneMem smoke run improved `retrieval_time_ms` by `-7660.94ms` and `elapsed_seconds` by `-22.363s` versus the previous smoke artifact, with `ndcg@10` up `0.007929` and `recall@10` down `0.005208`.
This is smoke-level context only; the new full five-benchmark run is required before treating quality and speed deltas as final.

### Quality Guardrails

- Four-benchmark diagnostic sampling follow-up, run on 2026-04-27:
  baseline `BenchmarkResult/20260427_full_diagnostic_sampling_v1`,
  optimized validation `BenchmarkResult/20260427_full_diagnostic_sampling_v2_after_profile_index`.
  The optimization stores `retrieval_source_content_hash` in index metadata, avoids per-query full metadata deep copies, and changes `profile_side_index` from full profile-entry scan to v2 side-index postings.
- Diagnostic elapsed time improved on all four samples:
  LongMemEval 100 queries `190.76s -> 109.27s` (`1.75x`),
  LoCoMo 250 questions `22.70s -> 19.04s` (`1.19x`),
  KnowMe 200 questions `364.22s -> 274.83s` (`1.33x`),
  CloneMem 80 questions `257.56s -> 226.05s` (`1.14x`).
- Diagnostic quality did not show a candidate-recall regression:
  LongMemEval `candidate_recall@100=1.000 -> 1.000`, R@10/NDCG unchanged;
  LoCoMo `candidate_recall@100=1.000 -> 1.000`, session R@10/NDCG unchanged;
  KnowMe `candidate_recall@100=0.925 -> 0.930`, `ndcg_any@10=0.3938 -> 0.4008`, `recall_frac@10=0.8025 -> 0.8000`;
  CloneMem `candidate_recall@100=0.405 -> 0.405`, R@10/NDCG unchanged.
- The clearest runtime change was KnowMe `profile_ms=333.67ms -> 124.66ms` and `retrieval_ms=1496.04ms -> 1209.56ms`.
  CloneMem and LongMemEval also benefited from less setup/vector-ingest overhead in the repeated diagnostic run; their quality metrics were unchanged.
- Post-change tests passed:
  `python -m unittest tests/test_multichannel_retrieval.py` 29 passed,
  `python -m unittest tests/test_guardrails.py` 28 passed,
  `python -m unittest discover tests` 81 passed.
- Follow-up attempts after this checkpoint tested a lexical side-index fast path and no-copy BM25/side-index map reuse.
  They were not retained because diagnostic sampling showed either quality regression or `candidate_recall@100` regression.
  The failed-attempt artifacts are preserved for audit under:
  `BenchmarkResult/20260427_lexical_fastpath_sampling_v1`,
  `BenchmarkResult/20260427_no_copy_sampling_v1`, and
  `BenchmarkResult/20260427_bm25_nocopy_sampling_v1`.

- Ran `./.venv312/bin/python -m unittest tests/test_guardrails.py tests/test_multichannel_retrieval.py`: 54 tests passed.
- Ran `./.venv312/bin/python -m unittest discover tests`: 63 tests passed.
- Ran `./.venv312/bin/python scripts/profile_retrieval_efficiency.py`: smoke profile passed with `total_ms=1.42`, `full_scan_total_records_scored=0`.
- The previous interrupted full benchmark directories for 2026-04-27 were removed as invalid. A fresh full benchmark run should be used for Recall/NDCG validation.
- The initial 2026-04-27 v3 run was also stopped and should be treated as invalid because it was launched before hard runtime red lines were added.
- Full v4 validation completed LongMemEval and LoCoMo, then correctly failed KnowMe at the 70 minute hard red line.
  LongMemEval: `Recall@10=0.9920`, `NDCG@10=0.9324`, `candidate_recall@100=1.0000`, elapsed about `11.84 min`.
  LoCoMo finished in about `56.91 min`, below its 60 minute warning red line.
  KnowMe reached only about `136/1010` before timeout, with per-question ingest around `26-30s`; the remaining bottleneck is now benchmark ingest/workspace construction, not fusion/rerank.
- Full v5 validation with workspace retention and chunking completed LongMemEval, LoCoMo, and KnowMe under the tighter red lines:
  LongMemEval `7.66 min`, LoCoMo `2.81 min`, KnowMe `18.70 min`.
  LongMemEval R@5/R@10/NDCG@10 matched v4 exactly (`0.976`, `0.992`, `0.9323582923`).
  KnowMe improved versus the available full baseline (`recall_frac@10=0.5895`, `ndcg_any@10=0.4206`).
  CloneMem hit the `90 min` fail red line after only chunks `0-3` completed; diagnostics showed question-hash chunking was duplicating sample workspace construction across chunks.
- CloneMem sample-sharded full rerun `20260427_clonemem_full_sample_sharded_v1` completed all `2374` questions in `3231.02s` (`53.85 min`), below the `75/90 min` warning/fail red line.
  It produced `speedup_estimate=2.81x`, `recall_frac@10=0.08458`, `recall_any@10=0.22831`, `ndcg_any@10=0.12637`, and `candidate_recall@100=0.34412`.
  This is much faster than the old full CloneMem artifacts (`7589s` to `12243s`) and avoids the prior timeout; final R/NDCG is slightly below the targeted-repair v2 artifact but above the multichannel full v1 artifact, while candidate recall remains effectively at the documented CloneMem baseline.
