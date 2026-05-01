# DysonSpherain: Dense-Preserving Multi-Channel Retrieval for Long-Horizon CLI Memory

## Abstract
DysonSpherain now frames long-horizon CLI memory retrieval as a dense-preserving multi-channel candidate-generation problem rather than a dense-only reranking problem. On the latest full factual benchmark package, LongMemEval reaches 0.9920, LoCoMo reaches 0.8986, KnowMe reaches 0.5767, and CloneMem reaches 0.0894 on their primary metrics. Relative to the baseline snapshot, the observed trend is: longmemeval roughly flat, locomo regressed, knowme improved, clonemem regressed. LongMemEval and LoCoMo are not primarily limited by broad candidate recall, while KnowMe and CloneMem remain candidate-admission sensitive. CloneMem oracle retrieval stays healthy at oracle_recall@10=1.0000, which argues against a broken embedding/index/id pipeline and localizes the remaining failure surface to fine-grained segment admission, parent-to-segment expansion, and benchmark-specific routing. The repaired safe-fusion route preserves dense candidate coverage on the evaluated factual benchmarks. The latest rerun is therefore a mixed result: candidate admission improves on the harder surfaces, but not every final metric moves in the same direction.

## Introduction
Long-horizon CLI memory retrieval fails in more than one way. Temporal anchoring drift and local candidate crowding still matter, but the latest rerun shows that they are not the dominant failure source on every benchmark. LongMemEval and LoCoMo keep broad candidate recall near saturation, while KnowMe and CloneMem expose a different bottleneck: the gold segment often exists in the indexed corpus and can remain recoverable under oracle querying, yet it is not admitted or prioritized at sufficient rank in the first-stage pool.

The current repair round also sharpens an engineering lesson. Naive multi-channel fusion can hurt a strong dense baseline if duplicate collapse, parent caps, or inhibition act destructively. DysonSpherain is therefore better described as route-conditioned temporal retrieval plus dense-preserving multi-channel candidate generation, channel-gated safe fusion, and optional memory-grounded creative expansion. Route-conditioned temporal retrieval remains a core subsystem, but it is no longer the only mechanism worth discussing.

\[
C(q) = C_{dense}(q) \cup C_{lex}(q) \cup C_{entity}(q) \cup C_{parent}(q) \cup C_{neighbor}(q) \cup C_{decomp}(q) \cup C_{profile}(q)
\]

The latest evidence does not support a universal SOTA claim. It supports a bounded mechanism claim: safe fusion and channel gating are necessary conditions for robust fine-grained memory retrieval, and candidate admission remains the dominant unresolved bottleneck on the hardest benchmarks.

## Contributions
1. We formalize dense-preserving multi-channel candidate generation for long-horizon CLI memory retrieval, integrating dense semantic retrieval, lexical sparse retrieval, entity-aware retrieval, parent-session expansion, temporal-neighbor expansion, query decomposition, and profile-side retrieval.
2. We introduce safe fusion and channel gating so that additive channels do not silently destroy dense coverage through duplicate collapse, parent caps, or early inhibition.
3. We add candidate recall diagnostics, per-channel contribution reports, KnowMe category analysis, and CloneMem failure taxonomy to separate candidate admission failures from late-stage ranking failures.
4. We report an updated four-benchmark factual evaluation surface covering LongMemEval, LoCoMo, KnowMe, and CloneMem directly from current local artifacts rather than from stale draft numbers.
5. We make the limitation boundary explicit: CloneMem remains the hardest surface, multi-seed significance is absent, public-baseline parity is not claimed, and the latest rerun should be read as a mixed result rather than as a blanket win.

## Method
### Multi-Channel Candidate Generation
DysonSpherain exposes distinct candidate channels with independent provenance and configurable top-k budgets. The dense channel preserves the embedding baseline. The lexical sparse channel targets exact or near-exact lexical anchors. The entity-aware channel promotes shared people, projects, tools, metrics, files, paths, and temporal expressions. The parent-session channel retrieves parent blocks before expanding inside them. The temporal-neighbor channel expands around strong seeds without unbounded drift. The query-decomposition channel derives entity, attribute, object, time, metric, constraint, and evidence-type cues without using gold metadata. The profile-side channel targets stable profile, preference, relationship, and state evidence.

### Safe Fusion With Provenance
Each channel returns stable segment ids and retains provenance. Fusion uses reciprocal-rank-style aggregation plus normalized channel scores, but it now preserves a dense anchor set, keeps destructive filters away from the broad candidate-recall pool, and records channel contributions and restoration events. Competition-aware inhibition is treated as optional local diversity control rather than as an admission-time filter.

### Diagnostics
We track candidate_recall@K, final recall, dense-vs-fused hit rates, gold rank movement before and after rerank, parent-hit/segment-miss events, and benchmark-specific failure buckets. This matters because the latest reruns show that a low final score can arise either from a poor candidate pool or from a late-stage ordering drop; the two require different fixes.

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
| longmemeval | recall_any@10 | 0.9920 | Recall@10 >= 0.95 | met |
| locomo | recall_frac@10 | 0.8986 | session recall_frac@10 >= 0.90 | narrowly missed |
| knowme | recall_frac@10 | 0.5767 | segment recall_frac@10 >= 0.55 | met |
| clonemem | recall_frac@10 | 0.0894 | segment recall_frac@10 >= 0.12 | missed |

### Table 3. Supplementary Metrics
| Benchmark | Supplementary Metrics |
| --- | --- |
| longmemeval | recall_any@5=0.9700, recall_any@10=0.9920, ndcg_any@10=0.9180 |
| locomo | recall_frac@10=0.8986, recall_any@10=0.9335, ndcg_any@10=0.7363 |
| knowme | recall_frac@10=0.5767, recall_any@10=0.5515, ndcg_any@10=0.4160 |
| clonemem | recall_frac@10=0.0894, recall_any@10=0.2405, ndcg_any@10=0.1306 |

### Table 4. Candidate Recall And Bottleneck Diagnosis
| Benchmark | candidate_recall@100 | Broad Recall Bottleneck? | Rerank Bottleneck? | Diagnosis |
| --- | --- | --- | --- | --- |
| longmemeval | 1.0000 | no | no | late-stage bottleneck |
| locomo | 1.0000 | no | no | late-stage bottleneck |
| knowme | 0.7597 | yes | no | candidate admission bottleneck |
| clonemem | 0.3434 | yes | no | candidate admission bottleneck |

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
| locomo | entity_aware, parent_session, exact_phrase, query_decomposition | 1.0000 |
| knowme | dense_semantic, lexical_sparse, query_decomposition, entity_aware | 0.7597 |
| clonemem | dense_semantic, exact_phrase, entity_aware, query_decomposition | 0.3434 |

### Table 7. Dense-Only vs Safe-Fusion Ablation
| Benchmark | Slice | Config | Primary | candidate_recall@100 | dense_hit@100 | fused_hit@100 |
| --- | --- | --- | --- | --- | --- | --- |
| knowme | smoke50 | dense_only | 0.5700 | 0.7500 | n/a | n/a |
| knowme | smoke50 | full_multichannel_safe | 0.6300 | 0.8200 | 0.8200 | 0.8800 |
| clonemem | smoke50 | dense_only | 0.1392 | 0.5653 | n/a | n/a |
| clonemem | smoke50 | full_multichannel_safe | 0.2097 | 0.5720 | 0.8800 | 0.8800 |
| knowme | medium100 | dense_only | 0.6250 | 0.7850 | 0.8400 | 0.8400 |
| knowme | medium100 | full_multichannel_safe | 0.6100 | 0.8500 | 0.8400 | 0.8900 |
| clonemem | medium100 | dense_only | 0.0993 | 0.4834 | 0.8100 | 0.8100 |
| clonemem | medium100 | full_multichannel_safe | 0.1762 | 0.4834 | 0.8100 | 0.8100 |

**Before And After**
| Benchmark | Before Primary | After Primary | Before Cand@100 | After Cand@100 |
| --- | --- | --- | --- | --- |
| longmemeval | 0.9900 | 0.9920 | 0.9980 | 1.0000 |
| locomo | 0.9048 | 0.8986 | 1.0000 | 1.0000 |
| knowme | 0.5453 | 0.5767 | 0.7257 | 0.7597 |
| clonemem | 0.0937 | 0.0894 | 0.3493 | 0.3434 |

## Mechanistic Analysis
The latest artifacts separate candidate admission from late reranking more clearly than the prior draft. LongMemEval and LoCoMo retain candidate_recall@100 close to 1.0, so their remaining movement is not primarily an embedding or indexing failure; it is an ordering problem. KnowMe still shows a candidate-pool gap, especially on profile and task questions, and its category analysis now exposes different surfaces instead of collapsing everything into a single benchmark number. The strongest KnowMe categories in the latest artifacts are: ambiguous / multi-hop profile query, generic semantic query, location query, preference query.

CloneMem remains the hardest benchmark. The latest failure taxonomy reports {'ok': 571, 'entity_miss': 318, 'unknown': 76, 'parent_hit_segment_miss': 651, 'temporal_miss': 299, 'lexical_miss': 459}, while oracle retrieval remains clean enough to reject the hypothesis of a broken index or broken embedding path. This shifts the interpretation away from “the system cannot represent the gold segment” and toward “the segment is representable, but first-stage admission, parent-to-segment expansion, and route-specific prioritization still miss too often.” The repaired safe-fusion route preserves dense candidate coverage on the evaluated factual benchmarks.

## Discussion And Limitations
The previous draft's “three benchmark surface” narrative is no longer valid. The current artifact set covers four factual benchmarks with explicit integrity and oracle checks. CloneMem is also no longer outside current-snapshot coverage; it is inside coverage, but it remains the clearest unresolved bottleneck surface.

The main remaining limitation is not coverage, but uneven conversion from richer candidate generation into final top-k gains. CloneMem still underperforms the target guardrail, and candidate recall still constrains its final recall. KnowMe remains sensitive to profile-side extraction and top-10 ordering even when candidate admission improves. The study also remains mostly single-seed and deterministic, so it does not yet establish confidence intervals. ConvoMem is not part of this updated full rerun, and no public-SOTA claim is made.

## Conclusion
The updated evidence supports a bounded claim. DysonSpherain now has a clearer mechanism story for long-horizon CLI memory because it no longer treats dense retrieval plus late reranking as the only mechanism. The repair round shows that dense-preserving safe fusion and channel gating are necessary for robust multi-channel retrieval. LongMemEval and LoCoMo act as guardrails, KnowMe exposes profile-sensitive admission and ranking gaps, and CloneMem remains the most diagnostic fine-grained segment-level challenge. The next correct optimization targets are still candidate admission, parent expansion, and route-specific ordering rather than blind expansion of every retrieval channel.

## Figures And Captions
- Figure 1 should be updated from a single route-conditioned temporal retrieval diagram to a dense-preserving multi-channel candidate-generation diagram with safe fusion, optional inhibition, and optional creative expansion.
- Figure 2 should emphasize candidate admission failure, parent-hit/segment-miss, and fine-grained segment loss.
- Figure 3 should reflect the current four-benchmark factual surface rather than the old three-benchmark package.
- Figure 4 should be replaced with a failure-taxonomy or channel-contribution case study from the latest reports.
- If final figure assets are not regenerated yet, mark them as TODO rather than implying they are current.

## Consistency Checklist
- Source draft length inspected: 187 lines.
- Result root: /Users/yanbo/DysonSpherain/BenchmarkResult/20260426_regression_repair_full_v1
- Benchmarks covered: four-benchmark factual package (longmemeval, locomo, knowme, clonemem).
- Latest status:
  - longmemeval: primary=0.9920, candidate_recall@100=1.0000, integrity=clean, guardrail=Recall@10 >= 0.95
- locomo: primary=0.8986, candidate_recall@100=1.0000, integrity=clean, guardrail=session recall_frac@10 >= 0.90
- knowme: primary=0.5767, candidate_recall@100=0.7597, integrity=clean, guardrail=segment recall_frac@10 >= 0.55
- clonemem: primary=0.0894, candidate_recall@100=0.3434, integrity=clean, guardrail=segment recall_frac@10 >= 0.12
