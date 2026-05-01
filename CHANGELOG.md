# CHANGELOG

## 2026-04-28 Mainline Follow-Up

- Updated `scripts/validate_formal_protocol.py` so full-benchmark quality
  deltas are computed only against comparable previous runs. Previous full
  artifacts with different config hashes, dataset-version hashes, route policy,
  embedding setup, fallback status, run type, or question scope are now marked
  `non_comparable` instead of producing misleading quality-regression warnings.
- Added `scripts/analyze_clonemem_lexical_interference.py` plus regression
  coverage to compare default CloneMem diagnostics against `no_lexical_probe`
  diagnostics at query level.
- Ran the CloneMem lexical interference diagnostic on 186 matched 100k/en
  evidence questions. `no_lexical_probe` improved mean final_recall@10 by
  `+0.016231`, final_ndcg@10 by `+0.013493`, recall_any@10 by `+0.064516`,
  and ndcg_any@10 by `+0.045658`, while candidate_recall@100 changed only
  `+0.000672`. Because 34 queries regressed, this supports a narrow lexical
  anchor-quality gate rather than globally disabling lexical retrieval.
- Re-ingested current 20260428 matched full benchmark artifacts into the
  experiment registry and fixed registry metadata inference so formal protocol
  validation records exact chunked commands, config hashes, dataset-version
  hashes, and the current git commit when metrics files omit those fields.
- Added regression coverage for deriving formal protocol metadata from
  `run_manifest.json`, `source_files`, `embedding_info`, and `vector_info`.
- Ran a matched CloneMem `100k/en` evidence-mode parent-tail admission probe
  against a default 186-question control. The probe was rejected and rolled
  back because it did not reduce `parent_hit_segment_miss` or
  `gold_missing_from_candidate_pool`, and it slightly reduced R@10, NDCG@10,
  and candidate_recall@100.
- Added `reports/phase5_clonemem_parent_tail_admission_rejected.md` to preserve
  the negative result and keep the next CloneMem step focused on better anchor
  selection/query-anchor extraction rather than broader parent-tail expansion.
- Cleaned regenerated benchmark cache directories from the matched probe
  outputs after preserving metrics and diagnostic reports.

## 2026-04-28 Formal Evidence Four-Step Refresh

- Added `scripts/export_rrf_baselines.py` to generate artifact-level
  `dense_bm25_rrf` baselines from existing full dense/vector and BM25 runs
  without rerunning heavy benchmarks or injecting gold IDs into ranking.
- Exported four RRF baseline artifacts under
  `/Users/yanbo/DysonSpherain/BenchmarkResult/20260428_artifact_rrf_baselines_v1`.
  Each run writes full `metrics.json` plus compact `compact_metrics.json` so
  formal scanners can avoid loading oversized per-question artifacts.
- Formal baseline coverage increased from `21/36` to `25/36`.
- Completed the four missing CloneMem Phase 11 medium 100k/en ablations:
  `no_benchmark_route_tuning`, `no_inhibition`, `no_lexical_probe`, and
  `no_temporal_routing`, written to
  `/Users/yanbo/DysonSpherain/BenchmarkResult/20260428_phase11_clonemem_medium_100k_en_v2`.
- Formal ablation coverage increased from `8/36` to `12/36`.
- Fixed `scripts/run_ablation_suite.py` so artifact-level RRF baseline outputs
  are not misclassified as `full_admission` ablations.
- Fixed CloneMem sample-sharded chunk execution so an empty
  `sample_id_allowlist.txt` writes a 0-question metrics artifact and skips the
  subprocess instead of accidentally running the full sample set.
- Refreshed formal gap, baseline, ablation, paper-output, LOBO protocol, final
  summary, and protocol-validation reports. LOBO remains `0/4` because no real
  held-out full artifacts exist; matched or tuned full runs were not promoted
  as held-out evidence.
- Validation: `PYTHONPATH=base ./.venv312/bin/python -m unittest discover tests`
  passed with `156` tests.

## 2026-04-28 Formal Evidence Completion Pass

- Entered the Phase 10/11/14/19/20 formal evidence completion stage.
- Added explicit BM25-only benchmark mode for LongMemEval and LoCoMo. The new
  mode uses the existing SQLite FTS read path and is available only when
  callers pass `--mode bm25`; default benchmark behavior remains `evidence`.
- Extended LongMemEval and generic chunked runners so formal BM25 baselines can
  be run in shard/chunk mode.
- Updated `scripts/run_baselines.py` to discover sibling
  `compact_metrics.json` files, allowing oversized raw benchmark metrics to
  remain intact while compact artifact summaries are used for tables.
- Updated baseline artifact selection so an oversized `merged_metrics.json`
  cannot override an available sibling compact artifact.
- Added sample/full warnings for available baseline rows whose question count is
  below the known full benchmark size.
- Tightened BM25 baseline classification so artifact path names alone no longer
  classify a run as BM25; the artifact must explicitly declare `mode=bm25` or
  a BM25 method/baseline. This prevents evidence-mode sample runs from filling
  formal BM25 baseline rows.
- Extended KnowMe and CloneMem benchmark runners and chunked orchestration with
  explicit `--mode bm25` support.
- Ran LongMemEval BM25 full:
  `../BenchmarkResult/20260428_longmemeval_bm25_full_v1/compact_metrics.json`
  with 500 questions, R@5 `0.9060`, R@10 `0.9620`, and NDCG@10 `0.8242`.
- Ran LoCoMo BM25 full:
  `../BenchmarkResult/20260428_locomo_bm25_full_v1/locomo/merged_metrics.json`
  with 1986 questions, session recall_frac@10 `0.9059`, session recall_any@10
  `0.9431`, and session NDCG@10 `0.7747`.
- Ran KnowMe BM25 full:
  `../BenchmarkResult/20260428_knowme_bm25_full_v1/knowme/merged_metrics.json`
  with 1010 questions, recall_frac@10 `0.4329`, recall_any@10 `0.3941`, and
  NDCG@10 `0.2862`.
- Ran CloneMem BM25 full:
  `../BenchmarkResult/20260428_clonemem_bm25_full_v1/clonemem/merged_metrics.json`
  with 2374 questions, recall_frac@10 `0.0629`, recall_any@10 `0.1794`, and
  NDCG@10 `0.1022`.
- Refreshed baseline, ablation, leave-one-benchmark-out, and paper output
  artifacts from current evidence.
- Added `scripts/export_oracle_baselines.py` to export artifact-derived oracle
  candidate, parent, and segment upper-bound baseline rows from existing full
  diagnostics without hand-filling missing values.
- Generated `artifacts/baselines/oracle_baseline_exports.json`,
  `reports/oracle_baseline_exports.md`, and
  `/Users/yanbo/DysonSpherain/BenchmarkResult/20260428_artifact_oracle_baselines_v1`.
- Formal baseline coverage improved to `21/36`; missing oracle segment rows
  remain pending where no oracle retrieval diagnostic rows exist.
- Added `scripts/build_formal_evidence_gap_report.py` to summarize available
  vs pending evidence across baselines, ablations, leave-one-benchmark-out,
  paired statistics, and efficiency artifacts.
- Generated `reports/formal_evidence_gap_report.md` and
  `artifacts/formal_evidence_gap_report.json`.
- Added regression coverage in `tests/test_formal_evidence_gap_report.py` and
  `tests/test_phase10_artifact_summaries.py`.

## 2026-04-28 Phase 11 CloneMem Mechanism Ablations

- Added `SPHERE_BENCHMARK_ABLATION` support to benchmark forced config so
  formal ablation runs can disable a specific mechanism without changing
  default benchmark behavior.
- Propagated `ablation` and `run_type` through chunk merge outputs.
- Updated `scripts/run_ablation_suite.py` to scan chunked
  `merged_metrics.json` and compact artifacts, and to avoid treating BM25 or
  vector-only baseline artifacts as formal `full_admission` ablations.
- Ran CloneMem 100k/en 372-question matched medium ablations:
  `no_route_conditioned_admission`, `no_parent_to_segment_selector`,
  `no_safe_fusion`, and `no_rerank_guard`.
- Generated
  `reports/phase11_clonemem_medium_100k_en_ablation_comparison.md` and
  `artifacts/ablations/phase11_clonemem_medium_100k_en_comparison.json`.
- The clearest negative ablation on this slice is
  `no_parent_to_segment_selector`, dropping R@10 by `0.0538` and NDCG@10 by
  `0.0294` against the matched full-medium baseline.
- Refreshed ablation reports and formal evidence gap counts; available
  ablations are now `8/36`.

## 2026-04-28 Phase 12 Statistical Artifact Repair

- Fixed `scripts/compute_confidence_intervals.py` so LaTeX CI table run labels
  are derived from compact artifact filenames instead of the parent
  `statistics` directory.
- Added optional paired bootstrap delta CI support with win/tie/loss counts:
  `--paired-a`, `--paired-b`, `--paired-out`, `--paired-report`, and
  `--pair-key`.
- Paired matching is duplicate-aware and preserves repeated rows by pairing
  shared identifiers plus occurrence order, which keeps CloneMem full paired
  comparisons at 2374 rows.
- Regenerated:
  `artifacts/statistics/bootstrap_ci_full_compact.json`,
  `reports/statistical_significance_report.md`,
  `paper/tables/main_results_with_ci.tex`,
  `artifacts/statistics/paired_delta_full_vs_dense.json`, and
  `reports/paired_delta_full_vs_dense_report.md`.
- Added regression coverage in `tests/test_confidence_intervals.py`.

## 2026-04-28 Phase 13 Efficiency Artifact Scanner

- Updated `scripts/build_efficiency_pareto.py` to scan both `metrics.json` and
  nested `merged_metrics.json` artifacts, so chunked/full benchmark summaries
  are included in Pareto data.
- Metric and timing extraction now recurses through nested `metrics`,
  `segment`, `session`, `turn`, `dialog`, `timings`, and timing-summary
  sections.
- Regenerated `artifacts/profiling/efficiency_quality_pareto.json`,
  `reports/efficiency_quality_pareto.md`, and `figures/pareto_curve_data.csv`.
- Added regression coverage in `tests/test_phase13_efficiency_pareto.py`.

## 2026-04-28 CloneMem Admission Iteration

### Phase 5 CloneMem Evidence-Blend Rerank Probe

- Added a configurable CloneMem-only rerank experiment:
  `SPHERE_CLONEMEM_EVIDENCE_BLEND_RERANK_ENABLED`, default `false`, with
  `SPHERE_CLONEMEM_EVIDENCE_BLEND_RERANK_ALPHA` defaulting to `0.35`.
- Extended `scripts/analyze_clonemem_admission_failures.py` so it can aggregate
  multiple chunked `clonemem_candidate_recall.json` and `clonemem_topk_debug.jsonl`
  artifacts.
- Generated the 372-question medium diagnostic report:
  `reports/clonemem_admission_parent_anchor_filter_medium_100k_en_v1.md`.
- Matched 372-question medium A/B with the evidence-blend switch enabled improved
  CloneMem final_recall@10 `0.1543 -> 0.1665`, final_ndcg@10 `0.1246 -> 0.1318`,
  and candidate_recall@100 `0.5741 -> 0.5746`.
- Decision: not promoted to default because Segment R@5 and Any@5 moved down
  slightly and wall-clock increased. A matched alpha=0.25 sample kept the same
  early-rank regression pattern, so this remains an explicit experimental switch
  and should not be default-enabled without a new diagnostic hypothesis.

### Phase 5 CloneMem Parent Supplemental Anchor Probe

- Added a default-off parent-session experiment:
  `SPHERE_PARENT_SUPPLEMENTAL_ANCHOR_EXPANSION_ENABLED=false`, with bounded cap
  `SPHERE_PARENT_SUPPLEMENTAL_ANCHOR_EXPANSION_CAP=2`.
- The switch exposes a bounded number of overflow parent anchor children and
  records `supplemental_anchor_selected_count` diagnostics.
- Matched 156-question sample showed no quality gain: final_recall@10 stayed
  `0.1555`, final_ndcg@10 moved `0.1231 -> 0.1230`, and
  candidate_recall@100 moved `0.5644 -> 0.5635`.
- Decision: not promoted to default. Keep as diagnostic-only and stop broad
  parent overflow expansion without a new hypothesis.
- Report:
  `reports/phase5_clonemem_parent_supplemental_anchor_probe.md`.

### Phase 10/19 Artifact Table Guard

- Fixed `scripts/run_baselines.py` so formal/full/high-question-count artifacts
  outrank newer smoke/sample/medium/probe/alpha/supplemental artifacts when
  generating baseline and paper tables.
- This prevents default-off experiments from overwriting the paper-facing
  `dysonspherain_full` rows.
- Fixed `scripts/run_ablation_suite.py` so nested benchmark metrics under
  `metrics.segment`, `metrics.session`, `metrics.turn`, or `metrics.dialog` are
  extracted recursively instead of producing `available` rows with empty
  metrics.
- Applied the same full-over-sample selection policy to ablation artifacts.
- Added regression coverage in `tests/test_phase10_artifact_summaries.py`.

### Phase 5 CloneMem Parent-Anchor Noise Filter

- Added `scripts/analyze_clonemem_admission_failures.py` to bucket CloneMem
  admission failures from candidate-recall and top-k debug artifacts.
- Generated:
  `reports/clonemem_admission_guard_off_sample_v1.md`,
  `reports/clonemem_admission_full_chunk00.md`, and
  `reports/clonemem_admission_parent_anchor_filter_sample_v1.md`.
- Added a configurable parent-anchor low-information term filter for
  parent-to-segment selection. The rollback switch is
  `SPHERE_PARENT_ANCHOR_NOISE_FILTER_ENABLED`, default `true`.
- The 80-question CloneMem sample improved Segment R@10 `0.1203 -> 0.1232`,
  Any@10 `0.2875 -> 0.3125`, and NDCG@10 `0.1546 -> 0.1758`, while preserving
  candidate_recall@100 at `0.4050`.
- The 100k/en medium evidence run completed with Segment R@10 `0.1543`,
  Any@10 `0.3925`, NDCG@10 `0.2273`, and candidate_recall@100 `0.5741`.
- The rejected CloneMem dense-anchor rerank guard remains default-off because it
  reduced the directly comparable 80-question sample quality.
- Report:
  `reports/phase5_clonemem_parent_anchor_noise_filter_report.md`.

## 2026-04-27 Unified Plan Progress

### Phase 4 Diagnostic Consolidation

- Added `scripts/consolidate_phase4_diagnostics.py` to consolidate existing
  `*_candidate_recall.json` artifacts into Phase 4 diagnostic JSONL files without
  changing retrieval, fusion, reranking, or candidate admission behavior.
- Added `tests/test_phase4_diagnostic_consolidation.py` for candidate recall file
  discovery, dense-preservation violation extraction, reranker-drop extraction,
  and JSONL/report output.
- Generated:
  `reports/diagnostics/fusion_dense_preservation_violations.jsonl`,
  `reports/diagnostics/reranker_dropped_gold_examples.jsonl`, and
  `reports/phase4_diagnostic_consolidation.md`.
- Consolidation inputs were the current stable diagnostic/full artifacts:
  `BenchmarkResult/20260427_full_diagnostic_sampling_v2_after_profile_index`,
  `BenchmarkResult/20260427_other_full_rerun_v1`, and
  `BenchmarkResult/20260427_clonemem_full_sample_sharded_v1`.
- Result: scanned 36 candidate recall files and 6500 query rows, finding
  3 dense-preservation violations and 641 reranker-dropped-gold examples.
- Extended the consolidation to emit Phase 5 benchmark-specific diagnostics:
  `clonemem_parent_hit_segment_miss_examples.jsonl`,
  `clonemem_lexical_miss_examples.jsonl`,
  `clonemem_temporal_miss_examples.jsonl`,
  `clonemem_reranker_dropped_gold_examples.jsonl`,
  `locomo_ordering_failures.jsonl`, and
  `knowme_segment_admission_failures.jsonl`.
- Validation:
  `tests/test_phase4_diagnostic_consolidation.py` 3 passed.

### Phase 5 CloneMem Reranker Guard Experiment

- Tested CloneMem broad-rank floor variants against the same 80-question
  diagnostic slice used by the current valid profile-index baseline.
- Top-5 and Top-10 floor variants improved final Segment R@10 and NDCG@10 on the
  sample, but both worsened the target `reranker_dropped_gold_count` from 8 to
  10 and reduced candidate NDCG@10.
- Decision: rejected and rolled back the CloneMem rerank-floor behavior change.
  The report is `reports/phase5_clonemem_reranker_guard_experiment.md`.
- Kept only the non-behavioral diagnostic work and a configurable guard hook for
  the existing LoCoMo broad-rank floor.
- Validation after rollback:
  `tests/test_reranker_guard.py`, `tests/test_multichannel_retrieval.py`,
  `tests/test_guardrails.py`, and `tests/test_phase4_diagnostic_consolidation.py`
  passed together, 63 tests total.

### Phase 5 CloneMem Parent-Anchor Local-Window Experiment

- Added a narrow CloneMem parent-to-segment admission improvement: child segments
  already selected by the parent-session anchor selector now receive bounded
  `local_window_score` support.
- This does not increase parent fan-out, does not lower top-k, and does not
  disable channels.
- On the same 80-question CloneMem diagnostic slice, final quality improved:
  Segment R@10 `0.1246 -> 0.1552`, Segment Any@10 `0.3250 -> 0.3750`,
  Segment NDCG@10 `0.1694 -> 0.2058`, and candidate_recall@100
  `0.4050 -> 0.4111`.
- Caveat: `parent_hit_segment_miss` did not decrease on this slice, and
  candidate NDCG@10 moved down, so this is kept as a tentative local improvement
  requiring larger-shard validation.
- Report:
  `reports/phase5_clonemem_parent_anchor_local_window_report.md`.

### Phase 6 Project State Manager

- Added `base/sphere_cli/project_state.py` with `ProjectState`, JSON/Markdown
  rendering, registry-backed state updates, and structured memory write APIs:
  `write_memory`, `write_fact`, `write_decision`, `write_experiment`,
  `write_failure`, `write_task`, `write_constraint`,
  `write_conversation_summary`, and `write_agent_run_summary`.
- Added CLI commands:
  `ds memory state show`,
  `ds memory state update --from-latest-memories`,
  `ds memory state set-goal`,
  `ds memory state add-constraint`,
  `ds memory summarize-session`, and
  `ds agent postrun`.
- Generated the current DysonSpherain project state from the artifact registry
  under `artifacts/project_state/`; it records full non-fallback benchmark status
  for LongMemEval, LoCoMo, KnowMe, and CloneMem, with ConvoMem currently missing.
- Added `tests/test_project_state.py` and extended `tests/test_cli_commands.py`.

### Phase 7 Conflict and Lifecycle Management

- Added `base/sphere_cli/memory_lifecycle.py` with explicit conflict detection
  for `fallback_conflict`, `smoke_full_conflict`, `partial_full_conflict`, and
  `metric_conflict`.
- Added append-only lifecycle action recording for resolve, disputed, supersede,
  pin, archive, and merge operations.
- Added CLI commands:
  `ds memory conflicts list`,
  `ds memory conflicts show`,
  `ds memory conflicts resolve`,
  `ds memory conflicts mark-disputed`,
  `ds memory lifecycle review`,
  `ds memory supersede`,
  `ds memory pin`,
  `ds memory archive`, and
  `ds memory merge`.
- Generated the current lifecycle conflict report under
  `artifacts/memory_lifecycle/conflicts.json`; current conflicts are metric
  conflicts for KnowMe and LoCoMo comparable-looking runs.
- Added `tests/test_memory_lifecycle.py` and extended CLI registration tests.

### Phase 8 Runtime Context Compiler and Agent Adapter

- Added `base/sphere_cli/context_compiler.py` with Markdown context packet
  generation for `general`, `codex`, `paper`, `benchmark`, `debug`, `docs`, and
  `project` modes.
- Context packets include project state, benchmark status, constraints,
  relevant artifacts, and conflict warnings from lifecycle review.
- Added CLI commands:
  `ds memory context`,
  `ds agent preflight`, and
  `ds agent status`.
- Verified `ds agent preflight` against the current DysonSpherain project state;
  the generated packet includes current full benchmark status and active
  KnowMe/LoCoMo metric conflicts.
- Added `tests/test_context_compiler.py` and extended CLI registration tests.

### Phase 9 Execution Ledger and Resume Support

- Added `base/sphere_cli/execution_ledger.py` with append-only `ExecutionRun`
  and `ExecutionStep` records under `artifacts/execution_ledger/`.
- Added CLI commands:
  `ds agent ledger start`,
  `ds agent ledger list`,
  `ds agent ledger show`, and
  `ds agent ledger resume-packet`.
- Extended `ds agent postrun` so it still writes an agent memory summary and now
  also creates or updates the execution ledger with status, artifacts, tests,
  benchmarks, changed files, errors, and next-action hints.
- Resume packets now include run status, terminal/non-terminal safety notes,
  unresolved steps, artifact/error references, and a project-state snapshot.
- Added `tests/test_execution_ledger.py` and extended CLI registration tests.
- Report: `reports/phase9_execution_ledger.md`.
- Validation:
  `tests/test_execution_ledger.py`, `tests/test_cli_commands.py`, and
  `tests/test_context_compiler.py` passed together, 10 tests total.

### Phase 10 Baseline and Ablation Artifact Framework

- Added `scripts/run_baselines.py` to build artifact-backed matched-budget
  baseline summaries from discovered `metrics.json` files.
- Added `scripts/run_ablation_suite.py` to build artifact-backed ablation
  summaries from discovered `metrics.json` files.
- Both scripts mark missing methods as `pending` and oversized per-query metric
  files as `oversized_skipped`; they do not fabricate missing baseline,
  ablation, Recall/NDCG, candidate-recall, or failure-bucket values.
- Generated:
  `artifacts/baselines/baseline_runs.json`,
  `artifacts/ablations/ablation_runs.json`,
  `reports/baseline_comparison_table.md`, and
  `reports/ablation_table.md`.
- Added `tests/test_phase10_artifact_summaries.py`.
- Validation:
  `tests/test_phase10_artifact_summaries.py` passed, 2 tests total.
- Ran a non-fallback LongMemEval 50-question matched-budget baseline smoke:
  dense/vector MiniLM R@5 `0.9400`, R@10 `0.9800`, NDCG@10 `0.8680`;
  evidence route R@5 `1.0000`, R@10 `1.0000`, NDCG@10 `0.9656`.
- Added `reports/baseline_smoke_v1_report.md`,
  `reports/baseline_smoke_v1_comparison.md`, and
  `reports/baseline_smoke_v1_ci.md`.

### Phase 11 Mechanism-Level Ablation Outputs

- Extended `scripts/run_ablation_suite.py` to generate mechanism-level
  artifact outputs in addition to the basic ablation table.
- Generated:
  `reports/neurips_component_ablation.md`,
  `reports/failure_bucket_delta_by_ablation.csv`, and
  `figures/ablation_waterfall_data.csv`.
- The generated reports keep unavailable ablations as `pending` and do not
  invent failure-bucket deltas or metric values.

### Phase 12 Statistical Significance and Confidence Intervals

- Added `scripts/compute_confidence_intervals.py` for bootstrap confidence
  intervals over artifact-backed per-question rows.
- Added `scripts/export_compact_metrics.py` to stream compact per-question
  exports from giant full `metrics.json` files using `jq` while dropping large
  ranking payloads.
- The script records sample unit, resample count, random seed, metric CIs, and
  warnings for missing or oversized artifacts.
- Generated:
  `artifacts/statistics/bootstrap_ci.json`,
  `reports/statistical_significance_report.md`, and
  `paper/tables/main_results_with_ci.tex`.
- Generated current full compact artifacts and full bootstrap CI:
  `compact_longmemeval_full.json`, `compact_locomo_full.json`,
  `compact_knowme_full.json`, `compact_clonemem_full.json`, and
  `bootstrap_ci_full_compact.json`.
- Added `tests/test_confidence_intervals.py`.
- Added `tests/test_compact_metrics_export.py`.
- Validation:
  `tests/test_confidence_intervals.py` and
  `tests/test_compact_metrics_export.py` passed together, 4 tests total.

### Phase 13 Efficiency and Pareto Analysis

- Added `scripts/build_efficiency_pareto.py` for artifact-backed
  efficiency-quality Pareto data extraction.
- Generated:
  `artifacts/profiling/efficiency_quality_pareto.json`,
  `reports/efficiency_quality_pareto.md`, and
  `figures/pareto_curve_data.csv`.
- Dedicated candidate-budget and rerank-budget sweeps are explicitly marked
  pending until matched-budget artifacts are produced.

### Phase 14 Leave-One-Benchmark-Out Generalization

- Added `scripts/run_leave_one_benchmark_out.py` to generate the held-out
  generalization protocol, route-policy config artifacts, and report skeleton.
- The script now supports artifact-backed held-out result binding through
  `--heldout-result BENCHMARK=PATH` and optional discovery of explicitly named
  LOBO/heldout runs through `--results-root`; tuned full runs are not
  auto-promoted as held-out evidence.
- Generated:
  `artifacts/lobo/lobo_protocol.json`,
  `artifacts/lobo/route_policy_train_*.json`, and
  `reports/leave_one_benchmark_out_report.md`.
- All held-out result rows remain `pending` until full no-held-out-tuning
  benchmark artifacts are produced.

### Phase 15 Code Intelligence Index

- Added `base/sphere_cli/code_index.py` for offline Python file indexing,
  symbol extraction, imports, parse-error capture, and convention-based test
  file guesses.
- Added CLI commands:
  `ds code index`,
  `ds code search-symbol`, and
  `ds code relevant-files`.
- Generated the current code index under `artifacts/code_index/`.
- Added `tests/test_code_index.py` and extended CLI registration tests.
- Validation:
  `tests/test_code_index.py` and `tests/test_cli_commands.py` passed together,
  8 tests total.

### Phase 16 Observability and Debug Tracing

- Confirmed existing `ds memory trace`, `ds memory explain --last-recall`, and
  `ds runs explain-regression` surfaces.
- Added CLI commands:
  `ds memory debug-context` and `ds memory why-selected`.
- `why-selected` searches the latest stored recall trace for matching memory,
  node, chunk, object, or source ids and returns matched JSON paths.
- Extended CLI registration tests.

### Phase 17 Security and Local-First Guardrails

- Added `base/sphere_cli/security.py` with secret redaction for API keys,
  bearer tokens, password-like assignments, and private keys.
- Applied redaction to memory write-back, execution ledger persistence, and
  runtime context packet generation.
- Added `tests/test_security_redaction.py`.
- Validation:
  `tests/test_security_redaction.py` passed, 4 tests total.

### Phase 18 Memory OS Evaluation Harness

- Added `tests/test_memory_os_eval.py` for deterministic Memory OS checks.
- Covered context packet inclusion of project state, conflict warnings,
  fallback propagation, pinned constraints, token-budget enforcement, and secret
  redaction.
- Validation:
  `tests/test_memory_os_eval.py` passed, 2 tests total.

### Phase 19 Automatic Paper Tables, Figures, and Appendix

- Added `scripts/generate_paper_outputs.py` to generate paper-facing tables,
  figure data, diagnostic case-study references, and reproducibility checklist
  files from existing artifacts.
- Generated:
  `paper/tables/main_results.tex`,
  `paper/tables/ablation_with_ci.tex`,
  `paper/tables/failure_taxonomy.tex`,
  `paper/tables/efficiency.tex`,
  `paper/figures/data/failure_bucket_delta.csv`,
  `paper/figures/data/pareto_curve_data.csv`,
  `paper/figures/data/ablation_waterfall_data.csv`,
  `paper/figures/data/system_overview_nodes.csv`,
  `paper/appendix/diagnostic_case_studies.md`, and
  `paper/appendix/reproducibility_checklist.md`.
- Missing formal baselines, ablations, or diagnostic artifacts remain marked as
  pending; no paper table values are hand-filled.

### Phase 20 Formal Full Benchmark Protocol

- Added `reports/formal_full_benchmark_protocol.md`.
- Added `scripts/validate_formal_protocol.py` to validate full non-fallback
  benchmark artifacts, current runtime redlines, local-hash/fallback exclusion,
  quality deltas against prior full artifacts, formal evidence sections, and
  generated paper outputs.
- Generated `artifacts/formal_protocol_validation.json` and
  `reports/formal_protocol_validation.md`; current status is `pending` because
  baseline, ablation, LOBO, and efficiency evidence still have pending rows.
- The protocol report records the formal execution order, current available
  non-fallback full benchmark artifact status, and pending baseline, ablation,
  CI, Pareto sweep, and leave-one-benchmark-out requirements.
- No smoke/partial/fallback result is promoted to a formal full benchmark.

### Phase 21 Documentation and README

- Added Memory OS and reproducibility docs under `docs/`:
  `memory_os_overview.md`, `memory_schema.md`, `context_compiler.md`,
  `agent_adapter.md`, `benchmark_registry.md`, `conflict_resolution.md`,
  `candidate_admission_method.md`, and `reproducibility.md`.
- Updated `README.md` with External Memory OS commands and artifact-backed
  benchmark/paper workflow helpers.

### Phase 22 Final Reports and Paper Deliverables

- Added `scripts/generate_final_reports.py` to regenerate final reports from
  existing artifacts without hand-filling pending evidence.
- Added final reports:
  `reports/NEURIPS_UPGRADE_SUMMARY.md`,
  `reports/REPRODUCIBILITY_PACKAGE.md`,
  `reports/MEMORY_OS_UPGRADE_SUMMARY.md`, and
  `reports/FINAL_EXECUTION_LEDGER_SUMMARY.md`.
- Added paper deliverables:
  `paper/draft_neurips_ready.md`, `paper/appendix_full.md`,
  `paper/method_algorithms.tex`, and `paper/limitations_neurips.md`.
- Draft/report language explicitly marks pending formal baselines, ablations,
  leave-one-benchmark-out validation, and full CI work instead of inventing
  unsupported claims.

### Full Diagnostic Sampling Follow-Up

- Ran four-benchmark diagnostic sampling before and after the retrieval metadata/profile-side-index optimization:
  LongMemEval 100 queries, LoCoMo 250 questions, KnowMe 200 questions, and CloneMem 80 questions.
- Optimized benchmark retrieval setup in `benchmark_support.py` by storing `retrieval_source_content_hash` in index metadata and avoiding per-query deep copies of full `source_records_by_id` and `chunk_metadata_by_id`.
- Bumped the retrieval side-index schema to `multichannel_candidate_v2` and added profile-side-index postings:
  `profile_entry_by_id`, `profile_entry_ids_by_term`, and `profile_entry_ids_by_category`.
- Changed `profile_side_index` candidate scoring to score a deterministic postings-derived subset instead of scanning every profile side entry when the v2 side index is available.
- Validation on the same diagnostic sample showed wall-clock improvements with candidate recall preserved:
  LongMemEval `190.76s -> 109.27s`, LoCoMo `22.70s -> 19.04s`, KnowMe `364.22s -> 274.83s`, CloneMem `257.56s -> 226.05s`.
- Quality guardrail status on the diagnostic sample:
  LongMemEval, LoCoMo, and CloneMem metrics were unchanged; KnowMe `candidate_recall@100` improved `0.925 -> 0.930`, `ndcg_any@10` improved `0.3938 -> 0.4008`, and `recall_frac@10` changed `0.8025 -> 0.8000`.
- Tests after the change:
  `tests/test_multichannel_retrieval.py` 29 passed, `tests/test_guardrails.py` 28 passed, and full `python -m unittest discover tests` 81 passed.

Rollback anchor:
the pre-change diagnostic baseline is `BenchmarkResult/20260427_full_diagnostic_sampling_v1`; the after-change validation is `BenchmarkResult/20260427_full_diagnostic_sampling_v2_after_profile_index`.

### Phase 4 Guardrail Test Coverage

- Added standalone Phase 4 guardrail tests without changing retrieval ranking or candidate-admission behavior:
  `tests/test_query_anchor_extraction.py`,
  `tests/test_route_conditioned_admission.py`,
  `tests/test_safe_fusion_invariant.py`,
  and `tests/test_reranker_guard.py`.
- The tests cover query anchors, route-conditioned candidate admission, dense-preserving safe fusion, and reranker guard behavior.
- Added `reports/phase4_guardrail_test_progress.md` with scope, validation commands, results, and the next minimal Phase 4 action.
- Validation:
  new Phase 4 tests 7 passed;
  core retrieval/guardrail tests 57 passed;
  full `python -m unittest discover tests` 88 passed.

### Artifact-Backed Experiment Registry

- Added `base/sphere_cli/experiment_registry.py` with artifact-only benchmark run ingestion, comparable run metadata, latest-run selection, comparison tables, and regression explanation helpers.
- Added `ds runs` CLI commands:
  `ingest`, `list`, `latest`, `compare`, and `explain-regression`.
- Extended Phase 3 ingestion with bounded whole-directory discovery, sidecar artifact metadata for oracle/failure/integrity reports, route-policy hash comparability, and default markdown report generation under `reports/registry/`.
- Extended manifest-aware ingestion so chunked runs can recover worker count, shard strategy, representative subprocess commands, and manifest-derived sample coverage.
- Added a CloneMem parent-to-segment diagnostic-only trace writer at `reports/diagnostics/parent_to_segment_selection_traces.jsonl`.
  This records selector evidence without injecting new candidates or changing ranking behavior.
- Generated the initial registry from the current four full benchmark artifacts plus the 20260426 multichannel full baseline:
  `artifacts/registry/benchmark_runs.jsonl` and `reports/artifact_registry_summary.md`.
- Added tests in `tests/test_experiment_registry.py`, `tests/test_artifact_ingest.py`, `tests/test_run_comparability.py`, `tests/test_parent_to_segment_selector.py`, and extended `tests/test_cli_commands.py`.
- Note: whole-directory ingestion now skips oversized discovered legacy `metrics.json` files by default; direct file ingestion remains available for selected official artifacts.

## 2026-04-27

### Retrieval Efficiency

- Added route-aware channel gating with safe default aggressiveness and diagnostics for policy before/after gating, gated channels, reasons, and high-cost channels executed.
- Added conservative retrieval early exit diagnostics and controls without changing global dense/lexical/candidate-recall pool sizes.
- Extended benchmark retrieval side-indexes with token, entity, temporal, phrase, session, parent, order, session/parent metadata, and text-hash postings.
- Added indexed fast paths for entity, temporal, exact phrase, session bundle, query decomposition, and parent-session expansion while retaining legacy fallbacks.
- Tightened indexed fast paths so a populated side-index with no posting hits no longer silently falls back to scanning all source records; legacy scans remain only for missing side-index fields.
- Optimized runtime evidence retrieval with normalized query variant deduplication, simple-route per-variant budgets, rank-pass diagnostics, object-support skip on high-confidence evidence, and per-call candidate feature caching.
- Moved conservative early exit before expensive profile dense expansion, query decomposition, session bundle, temporal neighbor, and parent-session execution so skipped channels are not run first.
- Allowed early exit to skip only the dense subpath of `profile_side_index` while preserving indexed profile candidate generation for identity/profile routes.

### Vector Backend Guardrails

- Added JSON vector backend controls:
  `SPHERE_JSON_VECTOR_MAX_ITEMS`,
  `SPHERE_VECTOR_FAIL_FAST_ON_FALLBACK`,
  `SPHERE_WARN_ON_JSON_VECTOR_BACKEND`.
- Made `SPHERE_VECTOR_BACKEND=chroma` fail fast when Chroma is unavailable instead of silently falling back to JSON.
- Exposed vector backend, fallback state, vector count, and JSON scan warnings in vector-store info.

### Tests and Profiling

- Added tests for route-aware gating determinism, indexed candidate generation, early exit safety, JSON vector backend guards, and runtime evidence rank-pass diagnostics.
- Added regression coverage for early-exit-before-query-decomposition and object shortcut retrieval skipping.
- Added `scripts/profile_retrieval_efficiency.py` for lightweight retrieval smoke profiling and recent artifact discovery.
- Extended the profiling script to parse recent benchmark metrics artifacts and report retrieval/elapsed/Recall/NDCG deltas when comparable runs exist.
- Updated `SPEED_REPORT.md` with the optimization summary, smoke profiling output, and current quality-guard status.

### Benchmark Runtime Red Lines

- Added enforced runtime red lines to `run_all_benchmarks.py`.
- LongMemEval full 500 queries warns/fails after 10 minutes.
- LoCoMo full 1986 queries warns/fails after 45 minutes.
- KnowMe full 1010 queries warns/fails after 35 minutes.
- CloneMem full 2374 queries warns after 75 minutes and fails after 90 minutes.
- The combined benchmark suite warns after 2.5 hours and fails after 3.5 hours.
- Failures terminate the benchmark subprocess group and are written to `run_manifest.json` as `runtime_timeout`.

### Chunked Benchmark Framework

- Added deterministic shard utilities in `base/benchmarks/shard_utils.py`.
- Added shard CLI parameters to KnowMe, CloneMem, and LoCoMo benchmark runners:
  `--shard-index`, `--shard-count`, `--question-id-allowlist`, `--sample-id-allowlist`, `--max-questions`, and `--resume-existing`.
- Added `base/benchmarks/run_benchmark_chunked.py` for subprocess-based chunked execution with independent per-chunk cache roots, retries, resume, force rerun, and merge-only mode.
- Added `base/benchmarks/merge_benchmark_results.py` for question-row-first aggregation, candidate recall summaries, failure taxonomy merging, serial-equivalent elapsed time, wall-clock elapsed time, and speedup estimates.
- Extended `run_all_benchmarks.py` with `--chunked`, `--chunks`, `--workers`, `--per-benchmark-workers`, `--resume`, `--force`, and `--merge-only`; default serial behavior is unchanged.
- Updated `run_benchmark_chunked.py` so CloneMem uses deterministic sample-level chunking by default instead of question-hash chunking.
  This prevents every CloneMem worker from rebuilding the same sample workspaces, while preserving all questions for each assigned sample.
  `--shard-strategy question` remains available as an explicit compatibility mode.

### Efficiency Diagnostics

- Added side-index fast-path audit diagnostics and summaries to benchmark retrieval traces.
- Updated `scripts/profile_retrieval_efficiency.py` to report side-index audit fields and full-scan totals.
- Added runtime retrieval budget and optional read-channel parallelism diagnostics:
  `SPHERE_RUNTIME_RETRIEVAL_LATENCY_BUDGET_MS` and `SPHERE_RUNTIME_PARALLEL_CHANNELS_ENABLED`.
- Added `scripts/run_efficiency_validation.py` for smoke/full/compare efficiency reports.

## 2026-04-25

### Embedding and Cache Guardrails

- Enforced fail-fast embedding loading unless `ALLOW_EMBEDDING_FALLBACK=1` is explicitly set.
- Strengthened embedding cache keys to include provider, model, embedding dimension, preprocess version, and normalized text hash.
- Exposed embedding preprocess version in vector-store metadata.
- Added preprocess-version checking to benchmark fingerprint guardrails.

### Benchmark Fingerprints and Metadata

- Extended benchmark runtime/index fingerprints with embedding preprocess version.
- Kept benchmark-mode creative expansion disabled and isolated from candidate retrieval.

### Integrity and Diagnostics

- Added `build_raw_counts()` to compute ingest audit statistics consistently across benchmarks.
- Extended integrity reports with:
  - `memory_count`
  - `question_count`
  - `gold_evidence_coverage`
  - `missing_gold_evidence_ids`
  - `empty_text_count`
  - `timestamp_field_count`
  - `timestamp_parseable_count`
  - `timestamp_parse_rate`
  - raw duplicate counters
- Added per-query top-k debug JSONL outputs under `reports/debug/`.
- Added explicit per-query diagnostic rows with:
  - candidate recall
  - final recall
  - gold rank before rerank
  - gold rank after rerank
  - gold rank after inhibition
  - rerank delta
  - inhibition delta

### Retrieval and Rerank

- Added multi-probe dense retrieval for `knowme` and `clonemem` using focused query variants.
- Strengthened query feature extraction with attribute terms, clauses, task terms, preference terms, and evidence-intent tagging.
- Added `semantic_score`, `task_score`, `support_count`, `cluster_id`, and `broad_rank` to candidate diagnostics.
- Rebalanced rerank scoring to preserve strong broad candidates and reduce reranker-induced gold drops.
- Weakened inhibition penalties and reduced suppression for well-supported broad top candidates.

### Benchmark CLI / Adapter Surface

- Added `--question-limit` to `locomo_benchmark.py` for true 20-50 question smoke runs.
- Updated LongMemEval, LoCoMo, KnowMe, and CloneMem adapters to emit new audit and debug outputs.

### Regression Tests

- Added tests for:
  - fallback requiring `ALLOW_EMBEDDING_FALLBACK=1`
  - cache key separation by model/dim/preprocess
  - ingest raw-count audit fields
  - top-k debug row shape
  - query diagnostic rank separation
