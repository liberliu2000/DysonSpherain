# DysonSpherain: Diagnosing Candidate Admission Failures in Long-Horizon CLI Memory Retrieval

## Abstract

Long-horizon CLI memory systems often fail before generation: the relevant
evidence never reaches the candidate set. DysonSpherain studies this candidate
admission problem under temporal, entity/profile, lexical/code-like, session,
and segment constraints. The system combines route-conditioned candidate
admission, dense-preserving safe fusion, and parent-to-segment anchor
selection, with artifact-backed diagnostics that distinguish full formal runs
from smoke, fallback, diagnostic, and rejected experiments.

## Introduction

Semantic similarity is not sufficient for long-horizon engineering and research
memory retrieval. Queries frequently depend on time, people, sessions,
lexical/code anchors, and segment-level evidence hidden inside a retrieved
parent document. If the gold evidence is absent from the admitted candidate
pool, downstream fusion and reranking cannot repair the answer.

DysonSpherain therefore treats candidate admission as the main research object.
The paper-facing claim is narrow: diagnose and repair admission failures while
preserving dense anchors and keeping benchmark-specific behavior out of the
general CLI answer path.

## Method

The method is organized around three mechanisms:

1. Route-conditioned Candidate Admission.
2. Dense-preserving Safe Fusion.
3. Parent-to-Segment Anchor Selection.

Algorithm details are maintained in `paper/method_algorithms.tex`. Creativity
or reflection-style modules are secondary and belong in appendix or future
work unless separately supported by benchmark evidence.

## Evidence

Current formal evidence is artifact-backed:

- formal protocol status: `passed`
- full benchmarks: LongMemEval, LoCoMo, KnowMe, CloneMem
- baselines: `28` available, `8` blocked unavailable-model rows, `0` pending
- mechanism ablations: `36/36`
- leave-one-benchmark-out: `4/4`
- paired delta statistics: `4/4`
- efficiency sweeps: `7/7`

CloneMem uses a route-only protected top-3 lexical anchor gate promotion:
`clonemem-a40fef2bed79`. Older full CloneMem runs are not formal matched
comparisons when their config hash, dataset version, or route policy differs.

## Limitations

The paper must not claim universal SOTA. CloneMem remains the primary residual
quality bottleneck, blocked unavailable-model baselines are not completed
experiments, and many older full runs are intentionally non-comparable under
the formal protocol. Formal claims must continue to be generated from
artifacts, not hand-filled numbers.
