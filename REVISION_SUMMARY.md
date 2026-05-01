# Latest Paper Revision Summary

Date: 2026-04-30

Source prompt: `../dysonspherain_latest_paper_revision_codex_prompt.md`

## Sections Modified

- Reframed the paper as an artifact-backed diagnostic framework and guarded
  route-conditioned candidate admission system.
- Added real citations and `paper/latex/references.bib`.
- Strengthened Introduction with benchmark and retrieval citations.
- Expanded Method with route trigger rules, dense-preserving fusion details,
  duplicate/diversity guardrails, parent-to-segment repair criteria, and a
  generated route-policy table.
- Rewrote Results around admission recall, final recall, and admission-to-final
  gap.
- Rewrote Ablations as diagnostics instead of claiming monotonic module gains.
- Updated Efficiency Analysis with the 2026-04-30 CloneMem full-run efficiency
  artifact while preserving the existing quality-bottleneck interpretation.
- Strengthened Limitations and Ethics around non-monotonic ablations,
  benchmark-sensitive route policies, baseline admission gaps, stale-profile
  risk, deletion/forgetting, and opt-in AI-clone deployment.

## Table Consistency Fixes

- `paper/latex/tables/main_results.tex` now uses
  `artifacts/formal_protocol_validation.json` as the only source-of-truth for
  formal full rows.
- `paper/latex/tables/baseline_comparison.tex` overwrites the DysonSpherain
  rows from the formal full artifact so KnowMe and CloneMem values no longer
  drift from `main_results.tex`.
- Baseline rows that do not record `candidate_recall@100` are explicitly shown
  as `--` and marked `available; admission may be unrecorded`.
- `paper/latex/tables/ablation_results.tex` now reports diagnostic deltas
  against the ablation-family `full_admission` row, not against the formal full
  run.
- `paper/latex/tables/route_policy_hyperparameters.tex` is generated from
  formal artifact metric files. Missing route metadata is shown as
  `not recorded`.
- `paper/latex/tables/clonemem_efficiency_optimization.tex` and
  `paper/latex/tables/clonemem_efficiency_text.tex` are generated from the
  current CloneMem efficiency artifact and the formal validation artifact, so
  the runtime numbers and the cautionary quality wording remain tied to
  source files.

## Citations Added

Added `paper/latex/references.bib` and cited:

- LongMemEval
- LoCoMo
- KnowMe-Bench
- CloneMem
- Mem0
- MemGPT / Letta
- LangMem
- MemoryBank
- Generative Agents
- Reflexion
- BM25
- Dense Passage Retrieval
- Reciprocal Rank Fusion
- Sentence-BERT / MiniLM embedding lineage
- Cross-encoder reranking
- Retrieval-Augmented Generation

## Experiments Still Not Comparable

- Historical formal full runs remain `non_comparable` when config, dataset
  version, embedding, fallback state, question scope, or route policy differs.
- The eight unavailable-model baseline rows remain blocked:
  `cross_encoder_or_llm_reranker_subset` and `dense_only_bge_or_e5` for all
  four benchmarks.
- Baseline candidate admission metrics for BM25, dense-only MiniLM, and
  dense+BM25 RRF are not fully available in the current baseline table source;
  the paper marks these as unrecorded instead of using them for admission
  claims.

## Ablation Interpretation

Ablations are not monotonic across benchmarks. This is now presented as a
diagnostic result:

- Route-conditioned modules can help one memory surface while hurting another.
- Lexical routes can recover exact anchors but introduce noisy candidates.
- Safe fusion protects dense anchors but can suppress useful sparse evidence.
- Parent expansion can repair parent-hit segment-miss cases but can over-expand
  local clusters.

Therefore the contribution is a diagnostic framework plus guarded route policy,
not a universal additive recipe.

## CloneMem Positioning

CloneMem is positioned as a non-conversational personal-trace stress test and a
residual admission bottleneck. The paper does not claim that DysonSpherain
solves CloneMem. It points to future work on non-conversational trace routing,
profile-state extraction, and temporal object extraction.

## 2026-04-30 Efficiency Update

The manuscript now includes the CloneMem full-run efficiency artifact:

- `/Users/yanbo/DysonSpherain/BenchmarkResult/manual_clonemem_efficiency_v2_20260430_134155/clonemem/clonemem/merged_metrics.json`

This update is framed as runtime evidence only. The generated table/text report
the wall-clock reduction and shard-level speedup, but explicitly keep the
quality guardrail warning because CloneMem candidate recall remains low.

## Files Still Requiring Original Artifacts

To fully regenerate tables and case studies outside this workspace, include:

- `artifacts/formal_protocol_validation.json`
- `paper/tables/main_results.tex`
- `paper/figures/data/ablation_waterfall_data.csv`
- `paper/figures/data/pareto_curve_data.csv`
- `paper/figures/data/failure_bucket_delta.csv`
- `reports/diagnostics/*.jsonl`
- formal metric files under `BenchmarkResult` if route-policy hyperparameters
  should be regenerated exactly

If these files are absent, generated tables will either fail or mark values as
`not recorded`.

## Regeneration Commands

```bash
make paper
```

Equivalent explicit commands:

```bash
.venv312/bin/python -m scripts.generate_paper_tables
.venv312/bin/python -m scripts.generate_paper_figures
.venv312/bin/python -m scripts.generate_case_studies
cd paper/latex && tectonic main.tex
cd ../..
.venv312/bin/python -m scripts.validate_paper_claims
```

## Validation Status

- `make paper`: passed.
- `tectonic paper/latex/main.tex`: passed and generated `paper/latex/main.pdf`.
- `.venv312/bin/python -m scripts.validate_paper_claims`: passed.
