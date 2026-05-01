# DysonSpherain: Diagnosing Candidate Admission Failures in Long-Horizon CLI Memory Retrieval

## Claim

DysonSpherain focuses on diagnosing and repairing candidate admission failures
under temporal, entity/profile, lexical/code-like, session, and segment
constraints.

## Method

The system combines route-conditioned candidate admission, dense-preserving safe
fusion, and parent-to-segment anchor selection with artifact-backed diagnostics.

## Evidence

Current paper-facing tables are generated from artifacts. The refreshed formal
protocol passes with four full non-fallback benchmark artifacts, 28 available
baseline rows plus 8 artifact-backed blocked unavailable-model rows, 36/36
mechanism ablations, 4/4 leave-one-benchmark-out rows, paired delta statistics,
and 7/7 matched efficiency sweeps.

## Limitations

The current draft must not claim universal SOTA. CloneMem remains the residual
quality bottleneck, and the latest channel-tail rescue probe was rejected
because medium validation reduced candidate_recall@100 despite preserving final
R@10. Unavailable external reranker/BGE-E5 baseline rows stay blocked rather
than hand-filled.
