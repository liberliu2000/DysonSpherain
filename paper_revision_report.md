# Paper Revision Report

Date: 2026-04-30

## Summary

Revised the paper according to
`../dysonspherain_paper_revision_codex_prompt.md`. The manuscript is now framed
as a method paper about diagnosing and repairing candidate admission failures,
instead of an internal artifact validation report.

## Changed Sections

- Rewrote title and abstract around route-conditioned candidate admission.
- Rebuilt the introduction around candidate admission failures before
  generation.
- Added formal problem formulation with `D`, `q`, `G_q`, `A_K(q)`, `R_k(q)`,
  `r(q)`, and `C_c(q)`.
- Expanded method with route-conditioned admission, dense-preserving safe
  fusion, parent-to-segment anchoring, and failure attribution.
- Rewrote Algorithm 1 to match the implemented retrieval pipeline.
- Reworked results so CloneMem and KnowMe are residual bottleneck analyses.
- Updated Efficiency Analysis with the current CloneMem full-run runtime
  artifact without changing the main paper structure.
- Added Related Work, Limitations, Ethics and Privacy Considerations.
- Added appendix sections for artifact schema, validation, ablations, failure
  taxonomy, case studies, compute, and checklist.

## Generated Assets

- `paper/latex/tables/main_results.tex`
- `paper/latex/tables/baseline_comparison.tex`
- `paper/latex/tables/ablation_results.tex`
- `paper/latex/tables/failure_taxonomy.tex`
- `paper/latex/tables/artifact_validation_summary.tex`
- `paper/latex/tables/clonemem_efficiency_optimization.tex`
- `paper/latex/tables/clonemem_efficiency_text.tex`
- `paper/latex/figures/pipeline_figure.tex`
- `paper/latex/figures/benchmark_comparison.pgf`
- `paper/latex/figures/efficiency_pareto.pgf`
- `paper/latex/figures/failure_breakdown.pgf`
- `paper/latex/case_studies.tex`
- `paper/latex/tables/case_study_examples.tex`

## Automation Added

- `scripts/generate_paper_tables.py`
- `scripts/generate_paper_figures.py`
- `scripts/generate_case_studies.py`
- `scripts/validate_paper_claims.py`

## Artifact-Supported Claims

- Formal full benchmarks pass without fallback.
- LongMemEval and LoCoMo have admission recall at 100 of 1.0.
- KnowMe and CloneMem remain lower-admission stress surfaces.
- The 2026-04-30 CloneMem full efficiency run supports a runtime update, but
  not a retrieval-quality improvement claim.
- Baseline rows are generated from artifact-backed paper tables.
- Ablation rows are generated from `paper/figures/data/ablation_waterfall_data.csv`.
- Case studies are extracted from `reports/diagnostics/*.jsonl`.

## Blocked Or Excluded Claims

- No universal SOTA claim is made.
- Non-comparable historical runs are not used as improvement evidence.
- Blocked unavailable-model baselines are not treated as completed
  experiments.
- CloneMem is not described as solved.
- The CloneMem efficiency run is not treated as a new quality win.
- The CloneMem route-only policy is not described as a global CLI default.

## Remaining Risks

- Related Work has been strengthened narratively but still lacks a full
  bibliography pass.
- Some efficiency points are descriptive because matched quality metrics are
  incomplete for every budget sweep.
- Case studies are diagnostic and redacted; they support failure attribution,
  not new benchmark claims.
- Some LaTeX overfull warnings remain because of long generated table labels
  and artifact strings, but the PDF compiles.

## Validation

- `tectonic paper/latex/main.tex`: passed and produced `paper/latex/main.pdf`.
- `.venv312/bin/python scripts/validate_paper_claims.py`: passed.
