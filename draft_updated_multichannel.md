# DysonSpherain: Multi-Channel Candidate Generation for Long-Horizon CLI Memory

## Abstract
DysonSpherain now targets long-horizon CLI memory retrieval as a multi-channel candidate generation problem rather than a dense-only reranking problem. On the latest full factual benchmark package, LongMemEval reaches 0.9800, LoCoMo reaches 0.8572, KnowMe reaches 0.5032, and CloneMem reaches 0.0815 on their primary retrieval metrics. Relative to the available baseline snapshot, the observed trend is: longmemeval regressed, locomo regressed, knowme regressed, clonemem regressed. The updated diagnostics show that LongMemEval and LoCoMo are not currently limited by broad candidate recall, while KnowMe and CloneMem remain candidate-admission sensitive. CloneMem oracle retrieval stays healthy at oracle_recall@10=1.0000, which argues against a broken embedding/index/id pipeline and instead localizes the remaining failure surface to fine-grained segment admission, parent-to-segment expansion, and channel fusion. The latest rerun is therefore a negative or mixed result for the current multi-channel configuration rather than a clean win.

## Introduction
Long-horizon CLI memory retrieval fails in more than one way. Temporal anchoring drift and local candidate crowding still matter, but the latest full rerun shows that they are not the dominant failure source on every benchmark. LongMemEval and LoCoMo keep high candidate recall at large K, while KnowMe and CloneMem expose a different bottleneck: the gold segment often exists in the indexed corpus and may even be recoverable under oracle querying, yet it is not admitted into the first-stage candidate pool at sufficient rank. The current rerun also shows a harder engineering truth: a richer candidate stack can still regress the end metric if fusion, inhibition, and channel budgets are not calibrated carefully.

This update therefore repositions DysonSpherain as a route-conditioned retrieval system with multi-channel candidate generation, competition-aware fusion and inhibition, and optional memory-grounded creative expansion. Route-conditioned temporal retrieval remains part of the system, but it is no longer the only mechanism worth discussing. The key new mechanism is the candidate admission layer:

\[
C(q) = C_{dense}(q) \cup C_{lex}(q) \cup C_{entity}(q) \cup C_{parent}(q) \cup C_{neighbor}(q) \cup C_{decomp}(q) \cup C_{profile}(q)
\]

The latest evidence does not support a universal SOTA claim. Instead, it supports a bounded diagnostic claim: the current multichannel implementation localizes the remaining bottlenecks, but the latest full rerun still regresses against the baseline snapshot on all four factual benchmarks.

## Contributions
1. We formalize multi-channel candidate generation for long-horizon CLI memory retrieval, integrating dense semantic retrieval, lexical sparse retrieval, entity-aware retrieval, parent-session expansion, temporal neighbor expansion, query decomposition, and profile-side retrieval.
2. We unify these channels under a provenance-preserving fusion stage with parent diversity control and lightweight competition-aware inhibition rather than treating late reranking as the only recovery mechanism.
3. We add candidate recall diagnostics, per-channel contribution reports, KnowMe category analysis, and CloneMem failure taxonomy to separate candidate admission failures from reranking failures.
4. We report an updated four-benchmark factual evaluation surface covering LongMemEval, LoCoMo, KnowMe, and CloneMem from current local artifacts rather than from stale draft numbers.
5. We make the limitation boundary explicit: the current full rerun is a negative or mixed result, CloneMem remains the hardest surface, multi-seed significance is still absent, and public-baseline parity is not claimed.

## Method
### Multi-Channel Candidate Generation
DysonSpherain now exposes distinct candidate channels with independent provenance and configurable top-k budgets. The dense channel preserves the existing embedding path. The lexical sparse channel recovers exact or near-exact lexical anchors. The entity-aware channel promotes shared people, projects, tools, metrics, files, paths, and temporal expressions. The parent-session channel retrieves parent blocks before expanding inside them. The temporal-neighbor channel expands around strong seeds without unbounded drift. The query-decomposition channel derives entity, attribute, object, time, metric, constraint, and evidence-type cues without using gold metadata. The profile-side channel targets stable profile, preference, relationship, and state evidence.

### Candidate Fusion With Provenance
Each channel returns stable segment ids and retains provenance. Fusion uses reciprocal-rank-style aggregation plus normalized channel scores, followed by duplicate collapse, near-duplicate collapse, parent diversity control, and only then reranking. Competition-aware inhibition remains present, but it is deliberately light and is applied after broad retrieval rather than during first-stage admission.

### Diagnostics
We track candidate_recall@K, final recall, per-channel gold hit rate, gold rank movement before and after rerank, parent-hit/segment-miss events, and benchmark-specific failure buckets. This matters because the latest reruns show that a low final score can arise either from a poor candidate pool or from a late-stage drop; the two require different fixes.

## Experiments
### Table 1. Benchmark Scope And Metrics
| Benchmark | Scope | Question Count | Primary Metric | Secondary Metrics | Candidate Recall Metric | Integrity Status |
| --- | --- | --- | --- | --- | --- | --- |
| longmemeval | Long-horizon session evidence retrieval | 500 | recall_any@10 | recall_any@5, recall_any@10, ndcg_any@10 | candidate_recall@100 | clean |
| locomo | Conversation and session-level retrieval | 1986 | recall_frac@10 | recall_frac@10, recall_any@10, ndcg_any@10 | candidate_recall@100 | clean |
| knowme | Profile and preference retrieval | 1010 | recall_frac@10 | recall_frac@10, recall_any@10, ndcg_any@10 | candidate_recall@100 | clean |
| clonemem | Fine-grained segment-level autobiographical retrieval | 2374 | recall_frac@10 | recall_frac@10, recall_any@10, ndcg_any@10 | candidate_recall@100 | clean |

### Table 2. Main Full-Run Results
| Benchmark | Primary Metric | Result | Target / Guardrail | Status |
| --- | --- | --- | --- | --- |
| longmemeval | recall_any@10 | 0.9800 | Recall@10 >= 0.95 | regressed |
| locomo | recall_frac@10 | 0.8572 | session recall_frac@10 >= 0.90 | regressed |
| knowme | recall_frac@10 | 0.5032 | segment recall_frac@10 >= 0.55 | regressed |
| clonemem | recall_frac@10 | 0.0815 | segment recall_frac@10 >= 0.12 | regressed |

### Table 3. Supplementary Metrics
| Benchmark | recall_any@5 | recall_any@10 | ndcg_any@10 |
| --- | --- | --- | --- |
| longmemeval | 0.9540 | 0.9800 | 0.8856 |
| locomo | 0.8572 | 0.8953 | 0.7221 |
| knowme | 0.5032 | 0.4762 | 0.3285 |
| clonemem | 0.0815 | 0.2245 | 0.1217 |

### Table 4. Candidate Recall And Bottleneck Diagnosis
| Benchmark | candidate_recall@100 | Broad Recall Bottleneck? | Rerank Bottleneck? | Diagnosis |
| --- | --- | --- | --- | --- |
| longmemeval | 1.0000 | no | no | late-stage bottleneck |
| locomo | 1.0000 | no | no | late-stage bottleneck |
| knowme | 0.7218 | yes | no | candidate admission bottleneck |
| clonemem | 0.3490 | yes | no | candidate admission bottleneck |

### Table 5. Oracle And Integrity Checks
| Benchmark | fallback_in_use | p0 bugs | oracle recall | interpretation |
| --- | --- | --- | --- | --- |
| longmemeval | false | none | n/a | oracle clean |
| locomo | false | none | n/a | oracle clean |
| knowme | false | none | n/a | oracle clean |
| clonemem | false | none | 1.0000 | oracle clean |

### Table 6. Per-Channel Contribution Snapshot
| Benchmark | Top Channels | candidate_recall@100 |
| --- | --- | --- |
| longmemeval | dense_semantic, lexical_sparse, entity_aware, temporal_anchor | 1.0000 |
| locomo | entity_aware, exact_phrase, dense_semantic, query_decomposition | 1.0000 |
| knowme | dense_semantic, lexical_sparse, query_decomposition, exact_phrase | 0.7218 |
| clonemem | dense_semantic, entity_aware, exact_phrase, query_decomposition | 0.3490 |

**Before And After**
| Benchmark | Before Primary | After Primary | Before Cand@100 | After Cand@100 |
| --- | --- | --- | --- | --- |
| longmemeval | 0.9900 | 0.9800 | 0.9980 | 1.0000 |
| locomo | 0.9048 | 0.8572 | 1.0000 | 1.0000 |
| knowme | 0.5453 | 0.5032 | 0.7257 | 0.7218 |
| clonemem | 0.0937 | 0.0815 | 0.3493 | 0.3490 |

## Mechanistic Analysis
The latest artifacts separate candidate admission from late reranking more clearly than the prior draft. LongMemEval and LoCoMo retain candidate_recall@100 close to 1.0, so their observed regressions are not primarily an embedding or indexing failure; they point to fusion, inhibition, or candidate-priority shifts. KnowMe still shows a candidate-pool gap, especially on profile and relationship questions, and its category analysis now exposes different surfaces instead of collapsing everything into a single benchmark number. The strongest KnowMe categories in the latest artifacts are: ambiguous / multi-hop profile query, generic semantic query, location query, preference query.

CloneMem remains the hardest benchmark. The latest failure taxonomy reports {'ok': 533, 'entity_miss': 206, 'local_candidate_crowding': 95, 'temporal_miss': 399, 'parent_hit_segment_miss': 681, 'lexical_miss': 344, 'dense_semantic_miss': 116}, while oracle retrieval remains clean enough to reject the hypothesis of a broken index or broken embedding path. This shifts the mechanistic interpretation away from “the system cannot find the gold segment at all” and toward “the correct segment is representable, but first-stage admission and parent-to-segment expansion still miss too often.” The latest fused candidate recall is also worse than dense-only candidate hit rate, which further implicates channel fusion and admission policy rather than the raw dense model alone.

## Discussion And Limitations
The previous draft's “three benchmark surface” narrative is no longer valid. The current artifact set covers four factual benchmarks with explicit integrity and oracle checks. CloneMem is also no longer outside current-snapshot coverage; it is inside coverage, but it remains the clearest unresolved bottleneck surface.

The main limitation is not absence of evaluation, but quality regression under the current multichannel configuration. CloneMem still underperforms the target guardrail, and candidate recall still constrains its final top-k recall. KnowMe remains sensitive to profile-side extraction and stitching quality, and the latest full rerun shows that the current retrieval stack has not yet converted the richer channel set into net gains. The study also remains mostly single-seed and deterministic, so it does not yet establish confidence intervals. ConvoMem is not part of this updated full rerun, and no public-SOTA claim is made.

## Conclusion
The updated evidence supports a bounded claim. DysonSpherain now has a clearer mechanism story for long-horizon CLI memory because it no longer treats dense retrieval plus late reranking as the only mechanism. However, the latest full rerun is a regression relative to the baseline snapshot, so the present value of the multichannel stack is diagnostic rather than performance-winning. CloneMem remains the most diagnostic hard surface, and the latest artifacts indicate that first-stage candidate generation, parent expansion, segment admission, and safer fusion are the next correct optimization targets.

## Figures And Captions
- Figure 1 should be updated from a single route-conditioned temporal retrieval diagram to a multi-channel candidate generation diagram with fusion, inhibition, and optional creative expansion.
- Figure 2 should emphasize candidate admission failure, parent-hit/segment-miss, and fine-grained segment loss.
- Figure 3 should reflect the current four-benchmark factual surface rather than the old three-benchmark package.
- Figure 4 should be replaced with a failure-taxonomy or channel-contribution case study from the latest reports.
- If final figure assets are not regenerated yet, mark them as TODO rather than implying they are current.

## Consistency Checklist
- Source draft length inspected: 187 lines.
- Result root: /Users/yanbo/DysonSpherain/BenchmarkResult/20260426_multichannel_full_v1
- Benchmarks covered: four-benchmark factual package (longmemeval, locomo, knowme, clonemem).
- Latest status:
  - longmemeval: primary=0.9800, candidate_recall@100=1.0000, integrity=clean, guardrail=Recall@10 >= 0.95
- locomo: primary=0.8572, candidate_recall@100=1.0000, integrity=clean, guardrail=session recall_frac@10 >= 0.90
- knowme: primary=0.5032, candidate_recall@100=0.7218, integrity=clean, guardrail=segment recall_frac@10 >= 0.55
- clonemem: primary=0.0815, candidate_recall@100=0.3490, integrity=clean, guardrail=segment recall_frac@10 >= 0.12
