guess you may find this haha, too tired to update this

# DysonSpherain: Route-Conditioned Temporal Retrieval for Long-Horizon CLI Memory

## Abstract

Long-horizon CLI memory retrieval often fails not because relevant evidence is absent, but because temporally relevant evidence becomes hard to reach and semantically similar local distractors saturate the ranking. We present a bounded retrieval upgrade that formalizes DysonSpherain as a route-conditioned pipeline with multi-probe candidate construction, temporal prefilter or expansion, and a locally scoped competition-aware rerank layer. We further define the creativity-side evidence as `memory-grounded creative expansion`: an optional second-stage layer that starts from temporally aligned core evidence, explores weak but potentially useful remote neighbors, and then filters the expansion through groundedness, consistency preservation, and budgeted expansion. Under a matched evaluation protocol, the evaluated DysonSpherain configuration beats the audited baseline across all nine required headline metrics, raising LongMemEval to Recall@5 0.8440, Recall@10 0.8740, and NDCG@10 0.8103, while also improving the currently runnable KnowMe information subset (recall_any@10 0.4802, ndcg_any@10 0.3535, recall_frac@10 0.4782) and LoCoMo (recall_any@10 0.8973, ndcg_any@10 0.7144, recall_frac@10 0.8643). Follow-up four-arm checks on LongMemEval, KnowMe, and LoCoMo, however, do not produce a measurable confusing-neighbor event or a positive inhibition delta. Mechanistically, the current evidence therefore supports a bounded asymmetric claim: weak-link expansion and cross-cluster retrieval receive direct support on the creativity surface, whereas competition-aware control is implemented auxiliary logic with an explicit negative benchmark-evidence boundary rather than a validated gain driver. Taken together, these results support a mechanism-oriented retrieval study with a controlled expansion layer rather than a universal SOTA claim; public MemPalace parity, CloneMem recovery, and multi-seed significance remain open evidence gaps.

## 1. Introduction

Long-horizon CLI memory retrieval sits in an awkward regime: relevant evidence may exist in the store, yet still be effectively unreachable because the query needs temporal alignment, entity fidelity, and local disambiguation at the same time. In this regime, flat semantic ranking is often not the main failure. The more frequent failure is that the right evidence lives in the wrong temporal neighborhood, or that the ranking gets crowded by same-session or same-entity near-duplicates before the correct support reaches the top positions.

The current evidence supports interpreting DysonSpherain as a retrieval-stage mechanism study rather than as a broad memory-architecture paper. We do not claim to have solved long-horizon memory in general. Instead, we isolate one bounded question: how much can route-conditioned temporal retrieval improve long-horizon CLI memory on the current runnable benchmark surface without collapsing the LongMemEval guardrail, and what can still be claimed honestly about the auxiliary competition-aware control? Within that bounded scope, retrieval is responsible for finding precise core evidence, while creativity remains a retrieval-grounded widening layer that becomes valid only after the core evidence is already temporally aligned.

Our paper is organized around three research questions. `RQ1`: can route-conditioned temporal retrieval improve the currently weaker runnable benchmarks without materially degrading LongMemEval, and what benchmark evidence remains for the auxiliary competition-aware control? `RQ2`: do the current diagnosis artifacts support the view that the remaining error surface is dominated by temporal reachability mismatch and local redundancy rather than generic semantic weakness? `RQ3`: how much of the remaining error is caused by candidate admission failure rather than final reranking, and which evidence gaps still block a stronger mechanism claim?

The answer is positive but bounded. Under the matched protocol, the current line improves all nine audited headline metrics on LongMemEval, the KnowMe information subset, and LoCoMo. The evidence also establishes a sharper negative boundary: follow-up four-arm checks on LongMemEval, KnowMe, and LoCoMo activate temporal and cluster controls without producing a measurable confusing-neighbor event, so competition-aware control cannot be claimed as a benchmark-validated source of gain on the accepted package. Within this framing, `memory-grounded creative expansion` functions as a second-stage mechanism for looking wider without giving up the requirement to first look right, rather than as a parallel main contribution.

Our contributions are therefore fourfold. First, we recast the current error surface as temporal reachability mismatch plus local redundancy. Second, we formalize the implemented DysonSpherain retrieval stack as a route-conditioned pipeline rather than leaving it at the level of system intuition. Third, we report competition-aware inhibition with an explicit negative-evidence boundary, so the paper does not overclaim it as a validated gain driver on the accepted benchmark package. Fourth, we assemble a mechanism-oriented evidence bundle that combines benchmark results, miss-bucket analysis, bounded ablation, and clear limitation boundaries: public MemPalace parity is still unmet, CloneMem remains outside trustworthy current-snapshot coverage, and multi-seed significance is still missing.

![Figure 1. DysonSpherain overview: route-conditioned temporal retrieval, competition-aware inhibition, and optional memory-grounded creative expansion. The method first builds a multi-probe candidate pool, then rescoring emphasizes temporally reachable evidence, suppresses redundant local neighbors, and preserves an anchored evidence set before any bounded expansion is allowed.{ width=100% }

Figure 1 summarizes the paper's bounded method story. Route-conditioned temporal retrieval decides which evidence should become reachable, competition-aware inhibition controls redundant local crowding, and memory-grounded creative expansion is allowed only after an anchored evidence set is already established.

## 2. Diagnosis by Measurable Indicators

We describe the diagnosis through three measurable indicators. `Temporal alignment rate` asks how often top-ranked evidence lands in the correct temporal neighborhood for the query. `Local redundancy ratio` asks how much of the top-ranked set is consumed by near-duplicate evidence from the same local neighborhood. `Distinct evidence coverage` asks how many genuinely different supporting episodes remain visible in the candidate set after reranking. These indicators define the retrieval problem at the level of observable behavior: not just whether the system finds something relevant, but whether it reaches the right temporal neighborhood and preserves distinct support.

The current evidence already points in a consistent direction. First, the miss-bucket audit shows that the remaining LongMemEval ceiling is dominated by upstream reachability rather than by purely local rerank order: there are 63 remaining miss10 cases, of which 53 fail because the correct answer never enters the top-50 candidate set and only 10 are admitted but ranked below 10. Second, the cross-surface benchmark pattern is asymmetric: LongMemEval improves modestly, while the KnowMe information subset and LoCoMo improve much more strongly. Third, the controlled creativity package shows that removing weak-link expansion collapses creative transfer quality, which is consistent with a reachability story rather than a purely semantic-similarity story.

We therefore diagnose the active error surface as a coupled problem. The first half is temporal reachability: evidence that should be reachable through time-sensitive or cross-session structure is still easy to miss at candidate-construction time. The second half is local redundancy: once the search enters a plausible neighborhood, multiple semantically similar candidates can still crowd out distinct support. This diagnosis is evidence-backed, but it is not yet a fully isolated causal proof. In particular, recent four-arm checks mostly yield control activation without a measurable confusing-neighbor event, so competition-aware control remains a bounded auxiliary explanation rather than a validated benchmark-side gain source.

![Figure 2. Motivating failure surface in long-horizon CLI memory retrieval. Temporal anchoring drift surfaces when a semantically plausible but wrong-time memory outranks the truly binding evidence, while local candidate crowding appears when redundant same-neighborhood memories occupy the top-k list and displace diverse support. { width=100% }

Figure 2 makes the motivating diagnosis concrete. The left panel shows temporal anchoring drift, where a memory can look semantically relevant yet still be bound to the wrong decision point. The right panel shows local candidate crowding, where redundant same-cluster memories fill the top-k list and hide distinct evidence that should survive into the final answer context.

## 3. Method

We now describe the implemented retrieval line as a route-conditioned heuristic algorithm. The goal is not to claim a learned global objective that the current system does not optimize. The goal is to make the current inference pipeline reproducible enough that a reviewer can see where temporal retrieval and competition-aware reranking actually enter the system.

### 3.1 Problem Setup

Let a query be `q`. Let each memory candidate be `m_i = (x_i, t_i, s_i, e_i, z_i)`, where `x_i` is the textual content, `t_i` is timestamp or temporal metadata, `s_i` is session or segment metadata, `e_i` is entity or profile metadata, and `z_i` is neighborhood structure used by the reranker. The system first derives a query profile `p(q)` and a route decision `r(q)` from lexical cues and profile needs such as temporal objects, relation objects, exact evidence, or preference objects. In the current implementation, `p(q)` is a compact feature vector containing temporal-cue indicators, entity/profile cues, segment or session cues, and ambiguity/crowding cues. The route decision is a discrete controller, `r(q) ∈ {temporal, identity, segment, exact_factual, ambiguous}`, that selects which retrieval controls are active. A route table maps `r(q)` to weights and budgets such as `w_t(r)`, `w_id(r)`, `w_seg(r)`, `λ(r)`, `rerank_pool_k`, `segment_rerank_topk`, and `confusing_cluster_topk`; routes with temporal cues receive larger temporal and segment budgets, while routes with local crowding cues enable the confusing-cluster penalty.

The first candidate set is the union of multiple retrieval probes:

`C_0(q) = C_dense(q) ∪ C_proxy(q) ∪ C_sparse(q).`

The route-conditioned tuner then sets a bounded retrieval budget such as `coarse_topk`, `fine_topk`, `rerank_pool_k`, `segment_rerank_topk`, and `confusing_cluster_topk`. These budgets are benchmark- and route-dependent, but they all preserve the same comparison contract: the system uses the same evaluator semantics while changing how the candidate set is assembled and reranked.

### 3.2 Route-Conditioned Temporal Retrieval

Temporal retrieval in the current line is not a free-floating graph walk. It is a route-conditioned candidate-construction stage that uses query profile and benchmark tuning to decide when temporal locality should influence the search. When the route predicts temporal dependence, the system enables temporal prefiltering, expands or preserves temporally relevant local segments, and increases the segment rerank budget so that the final pool contains more temporally plausible support before the last ranking pass. We write this stage as

`C_r(q) = {m ∈ C_0(q) : φ_temp(q, m) ≥ τ_t(r)} ∪ N_temp(C_0(q), q, B_t(r)),`

where `τ_t(r)` is the route-specific temporal gate and `N_temp` returns a bounded set of neighboring sessions or segments around temporally plausible candidates under budget `B_t(r)`. If the route is not temporal, the gate is inactive and `C_r(q) = C_0(q)`.

We write the resulting route-conditioned base score as

`s_base(q, m_i) = s_ret(q, m_i) + w_t(r) φ_temp(q, m_i) + w_id(r) φ_id(q, m_i) + w_seg(r) φ_seg(q, m_i),`

where `s_ret` is the retrieval score from the probe stack. `φ_temp(q,m_i)` combines timestamp or session-window compatibility with temporal-object matches in the query; it is high when `m_i` lies in the query's inferred time window or in an adjacent session selected by `N_temp`. `φ_id(q,m_i)` combines entity overlap, profile-slot agreement, and exact-name or preference consistency. `φ_seg(q,m_i)` rewards segment-level overlap, same-subtask membership, or same discourse unit membership. These terms are normalized to the rerank scale before weighting. The weights are not learned globally; they are heuristic route-conditioned settings chosen by the benchmark tuner.

### 3.3 Competition-Aware Reranking

After candidate construction, the system applies a local crowding penalty through the confusing-cluster rerank layer. This stage tries to preserve distinct evidence by penalizing candidates that are too similar to already favored same-neighborhood competitors. We write the rerank objective as

`s_final(q, m_i | S) = s_base(q, m_i) - λ(r) max_{m_j ∈ S} ψ_conf(m_i, m_j),`

where `S` is the partial selected set, `ψ_conf` measures local redundancy inside the same confusing neighborhood, and `λ(r)` is the route-conditioned crowding penalty. We define the confusing neighborhood by a local assignment `g(m)` derived from segment identity, entity overlap, semantic similarity, and temporal proximity. The penalty is active only for candidates in the same local neighborhood:

`ψ_conf(m_i,m_j) = 1[g(m_i)=g(m_j)] · sim_local(m_i,m_j),`

where `sim_local` combines semantic overlap, entity overlap, and temporal proximity on a bounded candidate pool. The greedy `max` term penalizes the strongest already selected competitor rather than all competitors, which keeps the rerank local and avoids suppressing every member of a large session. In the current code path, this penalty appears through `confusing_neighbor_penalty_weight` together with identity-aware and segment-aware rerank terms. The interpretation is simple: temporal retrieval improves which evidence becomes reachable, while competition-aware reranking is intended to reduce how much of the final list is wasted on near-duplicates. On the current accepted benchmark package, however, follow-up four-arm checks on LongMemEval, KnowMe, and LoCoMo do not yield a nonzero measured confusing-neighbor event, so this term is implemented reranker logic with an explicit negative-evidence boundary rather than an already validated source of gain.

### 3.4 Memory-Grounded Creative Expansion

We do not write creativity here as free-form or human-like ideation. We write it as `memory-grounded creative expansion`: an optional second-stage layer that runs only after core retrieval has already established a plausible temporally aligned evidence set. The operational goal is to let the system look wider without breaking the paper's retrieval-first boundary.

The layer has three stages. `Stage 1: core retrieval` produces the precision-preserving evidence set described above. `Stage 2: reflective expansion` follows weak but useful bridges from that core set toward cross-session, cross-cluster, or analogy-relevant neighbors. `Stage 3: gated integration` does not blindly append every expanded candidate; it keeps only the candidates that remain helpful after grounding and conflict checks.

The expansion layer is gated before scoring. Let `A(C_core,q)` measure whether the retrieved core evidence is sufficiently temporally aligned and internally consistent. Expansion is allowed only when `A(C_core,q) ≥ τ_anchor`; otherwise the system returns the grounded retrieval set without creative widening. The feasible expansion set is

`M_cre(q) = {m : grounded(m,C_core)=1, conflict(m,C_core) ≤ τ_c, cost(m) ≤ B_cre}.`

We score each feasible expansion candidate with a bounded heuristic:

`s_cre(q, m_i | C_core) = α φ_bridge(q, m_i, C_core) + β φ_util(q, m_i) + γ φ_novel(q, m_i, C_core) - λ_conflict φ_conflict(q, m_i, C_core).`

Here `φ_bridge` measures whether the candidate forms a meaningful bridge from already retrieved core evidence, `φ_util` measures likely usefulness for the current task, `φ_novel` rewards non-duplicative widening beyond the existing core set, and `φ_conflict` penalizes identity, temporal, or factual conflict with the established support.

Three control principles keep this layer paper-defensible. `Groundedness`: expansion must start from already retrieved core evidence rather than from free-floating generation. `Consistency preservation`: the expansion cannot override profile, time, or fact constraints already established by the retrieval layer. `Budgeted expansion`: hop depth, candidate count, and context footprint stay bounded so the layer remains a controlled widening pass rather than an uncontrolled second search problem.

This is also why the current claim remains asymmetric. The existing creativity-side evidence directly supports weak-link expansion or cross-cluster routing as a useful widening mechanism, but it does not yet isolate a matched creativity-side toggle for competition-aware control. In this paper, competition-aware control therefore remains implemented auxiliary logic with a negative benchmark-evidence boundary, not a standalone creativity-layer proof or a validated gain driver.

### 3.5 Inference Algorithm

```text
Algorithm 1 Route-Conditioned Temporal Retrieval with Competition-Aware Reranking
Input: query q, memory index M, target top-k
1: derive query profile p(q) and route decision r(q)
2: choose route-conditioned budgets and weights from benchmark tuner
3: retrieve multi-probe candidates C0 = Cdense ∪ Cproxy ∪ Csparse
4: if r(q) prefers temporal retrieval:
5:     preserve or expand temporally relevant local segments to form Cr
6: else:
7:     set Cr = C0
8: compute s_base(q, m) for each m in Cr using retrieval, temporal, identity, and segment terms
9: initialize selected set S = ∅
10: while |S| < k:
11:     pick the candidate with largest s_final(q, m | S)
12:     add that candidate to S
13: return S
```

### 3.6 Complexity and Implementation Details

The route decision itself is linear in the query length, `O(|q|)`. Candidate construction is bounded by the configured probe sizes and rerank pool size, `O(|C_dense| + |C_proxy| + |C_sparse| + |C_r|)`. Greedy reranking is `O(k · c)` when the confusing-neighborhood search is restricted to local cluster size `c = min(|C_r|, confusing_cluster_topk)`, rather than `O(k|C_r|^2)`. Segment reranking is similarly bounded by `segment_rerank_topk`. In practice, this means the method adds bounded inference overhead inside the retrieval stack instead of introducing a new learned module or a separate large-model reranker.

## 4. Experiments

### 4.1 Evaluation Protocol

The paper compares the current retrieval line against one audited baseline on three runnable benchmark surfaces. The protocol reports dataset scope, split, metrics, shared retrieval envelope, and comparator status in the main text; implementation paths and audit files remain reproducibility artifacts rather than part of the paper narrative.

| Surface | Split / scope | Headline metrics | Comparator role | Role in paper |
| --- | --- | --- | --- | --- |
| LongMemEval | validation | Recall@5, Recall@10, NDCG@10 | audited baseline | guardrail surface |
| KnowMe | information subset | recall_any@10, ndcg_any@10, recall_frac@10 | audited baseline | identity and temporal stress surface |
| LoCoMo | validation | recall_any@10, ndcg_any@10, recall_frac@10 | audited baseline | multi-session stress surface |

The current line keeps a shared evaluation envelope across all three runnable surfaces: evidence-mode retrieval, `top_k = 50`, session-level or information-subset scoring as appropriate, `sentence-transformers/all-MiniLM-L6-v2` embeddings, a fixed candidate-pool scale, the same evaluator semantics, no cross-encoder rerank, and no fallback path. Route-conditioned budgets differ by benchmark because the controller is query- and surface-aware, but the evaluator, metric definitions, embedding backend, and comparison target remain fixed.

| Surface | Shared setting | Comparator status | What is exact in the main comparison |
| --- | --- | --- | --- |
| LongMemEval | evidence-mode session retrieval with matched top-k and embedding backend | audited baseline with exact command parity | split, evaluator, retrieval envelope, and current-line budget surface |
| KnowMe information subset | information-query retrieval with matched top-k and embedding backend | audited baseline with accepted historical provenance | evaluator, retrieval envelope, metric definitions, and current-line budget surface |
| LoCoMo | evidence-mode multi-session retrieval with matched top-k and embedding backend | audited baseline with accepted historical provenance | evaluator, retrieval envelope, metric definitions, and current-line budget surface |

This protocol is sufficient for a bounded paper comparison against the accepted baseline. It is not yet a full public-baseline reproduction package because KnowMe and LoCoMo inherit baseline-side provenance from accepted historical reruns rather than from a freshly replayed baseline command in this paper loop.

CloneMem remains visible in the benchmark story, but it is outside current-snapshot claim coverage because a trustworthy canonical data root is unavailable. Public MemPalace raw LongMemEval results are treated as an external ceiling reference, not as a solved parity claim.

### 4.2 Main Comparison

The evaluated DysonSpherain configuration beats the repaired audited baseline on all three runnable benchmark surfaces. On LongMemEval, Recall@5 rises from 0.8080 to 0.8440, Recall@10 rises from 0.8460 to 0.8740, and NDCG@10 rises from 0.7683 to 0.8103. On the KnowMe information subset, recall_any@10 rises from 0.2901 to 0.4802, ndcg_any@10 rises from 0.1825 to 0.3535, and recall_frac@10 rises from 0.3211 to 0.4782. On LoCoMo, recall_any@10 rises from 0.8122 to 0.8973, ndcg_any@10 rises from 0.5530 to 0.7144, and recall_frac@10 rises from 0.7669 to 0.8643.

| Surface | Current line | Audited baseline | Delta | Role in paper |
| --- | --- | --- | --- | --- |
| LongMemEval (Recall@5 / Recall@10 / NDCG@10) | 0.8440 / 0.8740 / 0.8103 | 0.8080 / 0.8460 / 0.7683 | +0.0360 / +0.0280 / +0.0420 | Accepted benchmark anchor |
| KnowMe information subset (recall_any@10 / ndcg_any@10 / recall_frac@10) | 0.4802 / 0.3535 / 0.4782 | 0.2901 / 0.1825 / 0.3211 | +0.1901 / +0.1710 / +0.1571 | Identity-sensitive gain |
| LoCoMo (recall_any@10 / ndcg_any@10 / recall_frac@10) | 0.8973 / 0.7144 / 0.8643 | 0.8122 / 0.5530 / 0.7669 | +0.0851 / +0.1614 / +0.0974 | Multi-session gain |

![Figure 3. Main comparison between the accepted audited baseline and the current DysonSpherain line on the six headline paper metrics surfaced in the main text. The gains stay positive on the LongMemEval guardrail while becoming much larger on KnowMe and LoCoMo, which is the empirical shape that motivates the paper's bounded retrieval-strengthening claim. { width=95% }

Figure 3 makes the benchmark shape explicit: LongMemEval improves modestly but consistently, while the larger absolute gains appear on KnowMe and LoCoMo. This is why the paper frames the result as a balanced retrieval-strengthening package rather than a one-surface optimization story.

The table makes the claim boundary explicit: the current line is strong enough for an accepted-baseline paper story, but still not a universal win narrative because the public MemPalace raw LongMemEval bar remains unmet and CloneMem remains deferred.

### 4.3 Mechanistic Analysis

The first mechanism result is a reachability result. The miss-bucket audit shows that the remaining LongMemEval ceiling is still dominated by candidate admission, not just final order: 53 of the remaining 63 miss10 cases fail because the correct answer never enters the top-50 candidate set, while only 10 are admitted but ranked below 10. This pattern supports the idea that temporal retrieval and session selection matter upstream of the final rerank.

The second mechanism result is an asymmetric ablation result. On the controlled creativity surface, the full system reaches `FCR = 0.6667`, `WLAR = 0.4167`, and `CCCS = 0.9167`, while `nearest_cluster_only` collapses to `0.0`, `0.0`, and `0.3125`, and `no_weak_link_expansion` also collapses to `0.0`, `0.0`, and `0.5000`. This is direct support for writing creativity as a retrieval-grounded widening layer: the system does more than restate the nearest cluster, but the benefit still depends on weak-link expansion being anchored in core evidence.

![Figure 4. Representative top-k comparison between the audited baseline and DysonSpherain. Temporal rescoring pushes the correct evidence upward, competition-aware inhibition suppresses redundant local neighbors, and the final ranked set preserves both correctness and diversity. { width=100% }

Figure 4 illustrates the intended top-k reordering behavior behind the bounded mechanism claim. Relative to the baseline list, DysonSpherain is designed to downweight wrong-time memories, suppress duplicate local neighbors, and move the correct evidence into the top answer context before any optional expansion step.

The third mechanism result is now a negative boundary rather than a positive mechanism gain. Follow-up four-arm checks on LongMemEval, KnowMe, and LoCoMo either deactivate the cluster budget on the surface or activate it without producing a measurable confusing-neighbor event; the bounded LoCoMo pilot, for example, activates cluster processing on 46 of 199 full-arm queries yet keeps `confusing_neighbor_count = 0 / 199` and remains below baseline on `NDCG@10`. The current evidence therefore keeps competition-aware control in the paper as implemented auxiliary logic, not as a benchmark-validated gain driver.

### 4.4 Reporting Boundary

The current paper still reports single audited reruns rather than multi-seed confidence intervals or paired significance tests. We keep that limitation explicit instead of hiding it. The present paper therefore supports a bounded mechanism-oriented comparison, not a final matched-budget stability claim.

## 5. Discussion and Limitations

The first limitation is incomplete benchmark completeness. CloneMem is not a negative result; it is a missing trustworthy asset. Accordingly, the current evidence supports a three-benchmark result rather than complete current-snapshot coverage. The reopen condition is simple: if the canonical data root is restored, CloneMem becomes a legitimate next evidence target.

The second limitation is now both asymmetric and negative on the benchmark side. Weak-link expansion or cross-cluster retrieval still has direct bounded support, but competition-aware control now has an explicit negative benchmark boundary on top of its missing creativity-side matched toggle: the accepted follow-up checks on LongMemEval, KnowMe, and LoCoMo do not produce measurable confusing-neighbor events or a positive inhibition delta. Accordingly, the mechanism claim remains bounded and asymmetric: the creativity-side evidence directly validates weak-link expansion and cross-cluster retrieval, whereas competition-aware control is documented as implemented auxiliary logic with negative or limited benchmark evidence. For this reason, the creativity package appears as a controlled expansion layer rather than as an independent second main story.

The third limitation is reporting strength. We do not yet provide matched-budget multi-seed variance, paired significance, or a stronger public-baseline comparison table. The implication is that the current paper can defend a strong bounded mechanism story, but not a full NeurIPS-level stability or parity claim.

These limitations also discipline the scope of the paper. Richer memory representation, compression-first storage, and broader architecture questions remain interesting, but the present evidence still says the active contribution is retrieval-stage bottleneck repair plus a bounded creative expansion layer that only operates after core evidence is established. The paper is stronger when it states that directly.

## 6. Conclusion

Our results suggest that long-horizon CLI memory retrieval should be studied as a coupled problem of temporal reachability and local redundancy control rather than as semantic similarity ranking alone. In the current DysonSpherain line, a route-conditioned retrieval upgrade is enough to beat the audited baseline across the current runnable benchmark surface and to support a bounded mechanism-oriented interpretation centered on temporal retrieval. The same evidence also justifies a controlled `memory-grounded creative expansion` layer: retrieval finds the precise core evidence, and the creativity layer can then look wider without discarding grounding constraints. At the same time, the evidence base now draws a sharper inhibition boundary: competition-aware control is implemented and may still matter operationally, but the current accepted benchmark package does not validate it as a measurable gain source. Public MemPalace parity is still unmet, CloneMem is still outside trustworthy current-snapshot coverage, and multi-seed significance is still missing. The resulting contribution is therefore a bounded mechanism paper with an explicit inhibition limitation boundary rather than a universal SOTA claim.
