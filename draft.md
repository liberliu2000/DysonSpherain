guess you may find this haha

# DysonSpherain: Route-Conditioned Candidate Admission for Long-Horizon Agent Memory Retrieval

This draft has been revised according to
`../dysonspherain_paper_revision_codex_prompt.md` and
`../dysonspherain_latest_paper_revision_codex_prompt.md`. The LaTeX manuscript
under `paper/latex/main.tex` is the source-of-truth paper version; this
Markdown file summarizes the same revised narrative.

## Abstract

Long-horizon agent memory systems often fail before generation: the relevant
evidence is never admitted into the candidate pool. This paper studies
candidate admission as a measurable failure layer in long-horizon memory
retrieval and introduces DysonSpherain, a route-conditioned retrieval framework
that combines dense-preserving safe fusion, route-specific candidate admission,
and parent-to-segment anchoring. Across LongMemEval, LoCoMo, KnowMe-Bench, and
CloneMem, DysonSpherain distinguishes final reranking failures from first-stage
admission failures, showing strong coverage on dialogue-style memory tasks
while exposing residual bottlenecks on profile-centric and non-conversational
personal traces. All reported results are validated through an artifact-backed
protocol that detects fallback runs, blocks non-comparable historical
artifacts, and ties tables to reproducible metric files.

## Main Claim

DysonSpherain is not framed as a universal SOTA system. Its contribution is a
method and diagnostic framework for candidate admission failures in long-horizon
memory retrieval. The formal protocol supports credibility, but it is not the
main scientific claim.

## Method

The revised method defines:

- `D = {d_i}` as the memory corpus.
- `q` as the query.
- `G_q` as the benchmark gold evidence set.
- `A_K(q)` as the first-stage admitted candidate set.
- `R_k(q)` as the final ranked output.
- `r(q)` as the route or query type.
- `C_c(q)` as the candidate set from channel `c`.

Candidate admission recall is defined as:

```text
AdmissionRecall@K(q) = 1[G_q intersects A_K(q)]
```

Segment-level recall is defined as:

```text
SegmentRecall@k(q) = |G_q intersects R_k(q)| / |G_q|
```

The paper separates admission miss, fusion/order failure, reranking failure,
and parent-hit segment-miss.

## Mechanisms

DysonSpherain uses three guarded retrieval mechanisms:

1. Route-conditioned candidate admission over semantic/dense, lexical/exact,
   temporal, entity/profile, and parent/session routes.
2. Dense-preserving safe fusion, using a channel-gated reciprocal-rank-style
   score while protecting dense anchors from noisy route-specific channels.
3. Parent-to-segment anchoring, which expands selected parent/session hits into
   bounded child-segment candidates.

CloneMem suggests a future non-conversational trace route, but the paper does
not claim that route as generally implemented.

## Results Boundary

The current formal protocol passes with four full non-fallback benchmark rows:

| Benchmark | Questions | Admission R@100 | Final R@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| LongMemEval | 500 | 1.0000 | 0.9778 | 0.9259 |
| LoCoMo | 1986 | 1.0000 | 0.9070 | 0.7533 |
| KnowMe | 1010 | 0.7266 | 0.5972 | 0.5051 |
| CloneMem | 2374 | 0.3442 | 0.0954 | 0.0752 |

LongMemEval and LoCoMo show strong first-stage coverage. KnowMe and CloneMem
remain residual admission bottlenecks. CloneMem is reported as a stress test,
not a solved success case.

Baseline rows that lack `candidate_recall@100` are now marked as unrecorded
instead of being used for admission-specific claims. Ablation rows are now
reported as diagnostic deltas within the ablation-family run set, not as direct
formal full comparisons.

## Artifact Policy

Blocked unavailable-model rows and non-comparable historical artifacts are not
used for improvement claims. The eight blocked baseline rows remain blocked:

- `cross_encoder_or_llm_reranker_subset` for the four benchmarks.
- `dense_only_bge_or_e5` for the four benchmarks.

The CloneMem protected top-3 lexical anchor gate is route-only and is not a
global CLI default.

## Generated Paper Assets

The revised paper bundle now includes:

- `paper/latex/main.tex`
- `paper/latex/main.pdf`
- `paper/latex/tables/main_results.tex`
- `paper/latex/tables/baseline_comparison.tex`
- `paper/latex/tables/ablation_results.tex`
- `paper/latex/tables/failure_taxonomy.tex`
- `paper/latex/tables/artifact_validation_summary.tex`
- `paper/latex/tables/route_policy_hyperparameters.tex`
- `paper/latex/figures/pipeline_figure.tex`
- `paper/latex/figures/benchmark_comparison.pgf`
- `paper/latex/figures/efficiency_pareto.pgf`
- `paper/latex/case_studies.tex`
- `paper/latex/tables/case_study_examples.tex`
- `paper_revision_report.md`
- `REVISION_SUMMARY.md`
- `paper/latex/references.bib`

The paper also includes Related Work, Limitations, Ethics and Privacy
Considerations, appendix material, and claim validation.
