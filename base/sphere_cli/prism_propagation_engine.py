from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .creative_reflection_engine import CreativeReflectionEngine
from .models import BeamPath, QueryProfile
from .utils import clamp, lexical_score, token_tuple, tokenize


@dataclass
class _TransitionCandidate:
    chunk: dict[str, Any]
    node: dict[str, Any]
    source: str
    relation_weight: float
    ann_similarity: float = 0.0
    edge: dict[str, Any] | None = None


class PrismPropagationEngine:
    ALL_OPERATORS = ("semantic", "analogy", "contrast", "transfer", "temporal", "composition")

    def __init__(
        self,
        storage: Any,
        vector_store: Any,
        creative_engine: CreativeReflectionEngine,
        config: AppConfig,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.creative_engine = creative_engine
        self.config = config
        self._reset_call_state()

    def propagate(
        self,
        *,
        query: str,
        task_type: str,
        profile: QueryProfile,
        core_evidence: list[dict[str, Any]],
        evidence_nodes: list[dict[str, Any]],
        supporting_context: list[dict[str, Any]],
        limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        self._reset_call_state()
        diagnostics: dict[str, Any] = {
            "enabled": False,
            "reason": "",
            "operators": [],
            "candidate_counts": {
                "seed_chunks": 0,
                "seed_clusters": 0,
                "ann_shortlist": 0,
                "expanded_paths": 0,
                "selected_paths": 0,
            },
            "path_counts": {
                "support_paths": 0,
                "alternative_paths": 0,
                "creative_reflection_paths": 0,
            },
            "cache": self._cache_stats,
        }
        if not self.config.creative_enabled:
            diagnostics["reason"] = "creative_mode_disabled"
            return [], [], diagnostics
        if limit <= 0 or not core_evidence or not evidence_nodes:
            diagnostics["reason"] = "insufficient_seed_evidence"
            return [], [], diagnostics
        if self._should_skip_creative(profile, task_type):
            diagnostics["reason"] = "exact_evidence_guard"
            return [], [], diagnostics

        operators = self._enabled_operators(profile, task_type)
        diagnostics["operators"] = operators
        if not operators:
            diagnostics["reason"] = "all_operators_gated"
            return [], [], diagnostics

        seed_chunks = self._compress_seed_chunks(
            core_evidence=core_evidence,
            evidence_nodes=evidence_nodes,
            max_groups=max(1, min(self._effective_beam_width(), 8)),
        )
        diagnostics["candidate_counts"]["seed_chunks"] = len(seed_chunks)
        diagnostics["candidate_counts"]["seed_clusters"] = len({str(seed.get("seed_cluster_id") or seed.get("chunk_id") or "") for seed in seed_chunks})
        seed_node_map = {str(node.get("id") or ""): dict(node) for node in evidence_nodes if node.get("id")}
        seed_context = self._build_seed_context(query, seed_chunks, evidence_nodes, supporting_context, seed_node_map)
        ann_shortlist = self._build_ann_shortlist(query, seed_chunks)
        diagnostics["candidate_counts"]["ann_shortlist"] = len(ann_shortlist)
        diagnostics["enabled"] = True

        frontier = self._seed_paths(seed_chunks, operators)
        all_paths: list[BeamPath] = []
        for hop in range(1, self._effective_max_hops() + 1):
            self._round_frontier_hits = {}
            expanded: list[BeamPath] = []
            for path in frontier:
                expanded.extend(
                    self._expand_path(
                        path=path,
                        hop=hop,
                        query=query,
                        profile=profile,
                        seed_context=seed_context,
                        ann_shortlist=ann_shortlist,
                    )
                )
            if not expanded:
                break
            diagnostics["candidate_counts"]["expanded_paths"] += len(expanded)
            self._apply_backflow(frontier, expanded)
            all_paths.extend(expanded)
            frontier = self._select_frontier(expanded)
            if not frontier:
                break

        amplified = self._amplify_paths(all_paths)
        support_paths, creative_paths, alternative_paths_pool = self._bucket_paths(amplified)
        diagnostics["path_counts"]["support_paths"] = len(support_paths)
        diagnostics["path_counts"]["creative_reflection_paths"] = len(creative_paths)
        diagnostics["path_counts"]["alternative_paths"] = len(alternative_paths_pool)
        selected = self._select_output_paths(
            support_paths=support_paths,
            creative_paths=creative_paths,
            alternative_paths=alternative_paths_pool,
            limit=min(max(1, limit), self._effective_max_output_paths()),
            exact_mode=profile.needs_exact_evidence,
        )
        diagnostics["candidate_counts"]["selected_paths"] = len(selected)
        diagnostics["reason"] = "ok" if selected else "no_high_value_paths"

        creative_reflections = [self._path_to_reflection(path) for path in selected if path.path_role == "creative_reflection"]
        if self.config.creative_is_conservative and profile.needs_exact_evidence:
            creative_reflections = []
        alternative_paths = [self._path_to_alternative(path) for path in selected]
        return creative_reflections, alternative_paths, diagnostics

    def _reset_call_state(self) -> None:
        self._edge_cache: dict[str, list[dict[str, Any]]] = {}
        self._beam_edge_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._neighbor_cache: dict[str, list[dict[str, Any]]] = {}
        self._object_cache: dict[str, list[dict[str, Any]]] = {}
        self._chunk_cache: dict[str, dict[str, Any]] = {}
        self._node_cache: dict[str, dict[str, Any]] = {}
        self._node_chunk_cache: dict[str, list[dict[str, Any]]] = {}
        self._path_score_cache: dict[tuple[str, ...], dict[str, float]] = {}
        self._seed_signature_cache: dict[str, set[str]] = {}
        self._round_frontier_hits: dict[str, int] = {}
        self._cache_stats = {
            "adjacency_hits": 0,
            "adjacency_misses": 0,
            "beam_adjacency_hits": 0,
            "beam_adjacency_misses": 0,
            "local_neighborhood_hits": 0,
            "local_neighborhood_misses": 0,
            "path_score_hits": 0,
            "path_score_misses": 0,
        }

    def _should_skip_creative(self, profile: QueryProfile, task_type: str) -> bool:
        task_key = task_type.strip().lower()
        if self.config.creative_is_exploratory:
            return profile.needs_exact_evidence and task_key in {"qa", "trace", "factual_lookup", "status_lookup"}
        if self.config.creative_is_conservative:
            return profile.needs_exact_evidence and not profile.needs_temporal_objects and task_key in {"qa", "trace", "factual_lookup", "status_lookup"}
        return False

    def _effective_beam_width(self) -> int:
        width = max(1, int(self.config.creative_beam_width))
        if self.config.creative_is_conservative:
            return min(width, 1)
        return width

    def _effective_max_hops(self) -> int:
        hops = max(1, int(self.config.creative_max_hops))
        if self.config.creative_is_conservative:
            return min(hops, 1)
        return hops

    def _effective_neighbors_per_hop(self) -> int:
        neighbors = max(1, int(self.config.creative_neighbors_per_hop))
        if self.config.creative_is_conservative:
            return min(neighbors, 1)
        return neighbors

    def _effective_max_output_paths(self) -> int:
        limit = max(1, int(self.config.creative_max_output_paths))
        if self.config.creative_is_conservative:
            return min(limit, 2)
        return limit

    def _enabled_operators(self, profile: QueryProfile, task_type: str) -> list[str]:
        task_key = task_type.strip().lower()
        operators = ["semantic"]
        if self.config.creative_is_conservative:
            if self.config.creative_enable_temporal and (
                profile.needs_temporal_objects or task_key in {"design", "creative", "trace"}
            ):
                operators.append("temporal")
            return list(dict.fromkeys(operators))
        if self.config.creative_enable_analogy and task_key in {"design", "creative", "debug"}:
            operators.append("analogy")
        if self.config.creative_enable_contrast and task_key in {"design", "creative", "debug"}:
            operators.append("contrast")
        if self.config.creative_enable_transfer and task_key in {"design", "creative", "debug"}:
            operators.append("transfer")
        if self.config.creative_enable_temporal and (profile.needs_temporal_objects or task_key in {"design", "creative", "trace"}):
            operators.append("temporal")
        if self.config.creative_enable_composition and (profile.needs_multi_hop_evidence or task_key in {"design", "creative", "debug"}):
            operators.append("composition")
        return list(dict.fromkeys(operators))

    def _compress_seed_chunks(
        self,
        *,
        core_evidence: list[dict[str, Any]],
        evidence_nodes: list[dict[str, Any]],
        max_groups: int,
    ) -> list[dict[str, Any]]:
        candidates = [dict(item) for item in core_evidence[: max(4, max_groups * 3)]]
        node_map = {str(node.get("id") or ""): dict(node) for node in evidence_nodes if node.get("id")}
        selected: list[dict[str, Any]] = []
        cluster_index = 0
        for candidate in candidates:
            node_id = str(candidate.get("node_id") or "")
            candidate_text = self._candidate_text(candidate, node_map.get(node_id, {}))
            same_cluster = False
            for chosen in selected:
                chosen_text = self._candidate_text(chosen, node_map.get(str(chosen.get("node_id") or ""), {}))
                if node_id and node_id == str(chosen.get("node_id") or ""):
                    same_cluster = True
                    break
                if lexical_score(candidate_text[:500], chosen_text[:500]) >= 0.78:
                    same_cluster = True
                    break
            if same_cluster:
                continue
            cluster_index += 1
            candidate["seed_cluster_id"] = f"seed_cluster_{cluster_index}"
            selected.append(candidate)
            if len(selected) >= max_groups:
                break
        if not selected and candidates:
            fallback = dict(candidates[0])
            fallback["seed_cluster_id"] = "seed_cluster_1"
            selected.append(fallback)
        return selected

    def _build_seed_context(
        self,
        query: str,
        seed_chunks: list[dict[str, Any]],
        evidence_nodes: list[dict[str, Any]],
        supporting_context: list[dict[str, Any]],
        seed_node_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        primary_text = "\n".join(
            filter(
                None,
                [str(item.get("text") or item.get("summary") or "") for item in seed_chunks]
                + [str(node.get("summary") or "") for node in evidence_nodes],
            )
        )
        support_pool = [dict(item) for item in supporting_context] + [dict(item) for item in seed_chunks]
        return {
            "query_terms": set(tokenize(query)),
            "primary_text": primary_text,
            "primary_tokens": set(token_tuple(primary_text.lower())),
            "seed_chunk_ids": {str(item.get("chunk_id") or "") for item in seed_chunks if item.get("chunk_id")},
            "seed_node_ids": {str(item.get("node_id") or "") for item in seed_chunks if item.get("node_id")},
            "seed_node_map": seed_node_map,
            "support_pool": support_pool,
            "seed_clusters": {
                str(item.get("chunk_id") or ""): str(item.get("seed_cluster_id") or item.get("chunk_id") or "")
                for item in seed_chunks
                if item.get("chunk_id")
            },
            "anchor_text_by_seed": {
                str(item.get("chunk_id") or ""): " ".join(
                    filter(None, [str(item.get("text") or ""), str(item.get("summary") or "")])
                )
                for item in seed_chunks
                if item.get("chunk_id")
            },
        }

    def _build_ann_shortlist(self, query: str, seed_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not hasattr(self.vector_store, "search"):
            return []
        blend = query + "\n" + "\n".join(str(seed.get("text") or seed.get("summary") or "")[:280] for seed in seed_chunks[:3])
        shortlist_size = max(
            self._effective_beam_width() * self._effective_neighbors_per_hop() * 3,
            8 if self.config.creative_is_conservative else 12,
        )
        hits = self.vector_store.search(blend, top_k=shortlist_size)
        chunk_ids = [str(hit.get("chunk_id") or "") for hit in hits if hit.get("chunk_id")]
        hydrated = {}
        if chunk_ids and hasattr(self.storage, "fetch_chunks_with_node_metadata_by_ids"):
            hydrated = {
                str(chunk["chunk_id"]): dict(chunk)
                for chunk in self.storage.fetch_chunks_with_node_metadata_by_ids(chunk_ids)
            }
        shortlist: list[dict[str, Any]] = []
        for hit in hits:
            chunk_id = str(hit.get("chunk_id") or "")
            if not chunk_id:
                continue
            chunk = dict(hydrated.get(chunk_id) or {})
            metadata = hit.get("metadata") or {}
            chunk.setdefault("chunk_id", chunk_id)
            chunk.setdefault("node_id", str(metadata.get("node_id") or ""))
            chunk.setdefault("text", str(hit.get("document") or chunk.get("text") or ""))
            chunk.setdefault("shell", metadata.get("shell"))
            chunk.setdefault("sector", metadata.get("sector"))
            chunk.setdefault("zone", metadata.get("zone"))
            chunk.setdefault("cell", metadata.get("cell"))
            chunk.setdefault("grain", metadata.get("grain", "micro"))
            chunk["ann_similarity"] = float(hit.get("similarity") or 0.0)
            self._chunk_cache[str(chunk["chunk_id"])] = dict(chunk)
            shortlist.append(chunk)
        return shortlist

    def _seed_paths(self, seed_chunks: list[dict[str, Any]], operators: list[str]) -> list[BeamPath]:
        frontier: list[BeamPath] = []
        for seed in seed_chunks:
            seed_score = clamp(float(seed.get("evidence_score") or seed.get("dense_score") or 0.0), 0.0, 1.0)
            for operator in operators:
                frontier.append(
                    BeamPath(
                        beam_type=operator,
                        seed_node_id=str(seed.get("node_id") or ""),
                        seed_chunk_id=str(seed.get("chunk_id") or ""),
                        node_ids=[str(seed.get("node_id") or "")],
                        chunk_ids=[str(seed.get("chunk_id") or "")],
                        hop_count=0,
                        relevance=round(min(1.0, 0.55 + seed_score * 0.45), 4),
                        novelty=0.06 if operator == "semantic" else 0.14,
                        support=round(min(1.0, 0.48 + seed_score * 0.42), 4),
                        feasibility=0.78,
                        diversity=0.12 if operator == "semantic" else 0.24,
                        conflict_risk=0.0,
                        redundancy_penalty=0.0,
                        score=round(seed_score, 4),
                        endpoint_score=round(seed_score, 4),
                        trajectory_score=round(seed_score, 4),
                        backflow_score=0.0,
                        amplified_score=round(seed_score, 4),
                        mmr_score=round(seed_score, 4),
                        summary=str(seed.get("summary") or seed.get("text") or "")[:220],
                        evidence_anchor_ids=[str(seed.get("chunk_id") or "")],
                        path_role="support" if operator in {"semantic", "temporal"} else "alternative",
                        signature=f"{operator}|{str(seed.get('seed_cluster_id') or seed.get('chunk_id') or '')}",
                        metadata={"seed_cluster_id": str(seed.get("seed_cluster_id") or seed.get("chunk_id") or "")},
                    )
                )
        return frontier

    def _expand_path(
        self,
        *,
        path: BeamPath,
        hop: int,
        query: str,
        profile: QueryProfile,
        seed_context: dict[str, Any],
        ann_shortlist: list[dict[str, Any]],
    ) -> list[BeamPath]:
        candidates = self._collect_transition_candidates(path, query, seed_context, ann_shortlist)
        if not candidates:
            return []
        scored: list[BeamPath] = []
        for candidate in candidates:
            scored_path = self._score_transition(
                path=path,
                candidate=candidate,
                hop=hop,
                query=query,
                profile=profile,
                seed_context=seed_context,
            )
            if scored_path is None:
                continue
            scored.append(scored_path)
        scored.sort(key=lambda item: item.amplified_score, reverse=True)
        return scored[: self._effective_neighbors_per_hop()]

    def _collect_transition_candidates(
        self,
        path: BeamPath,
        query: str,
        seed_context: dict[str, Any],
        ann_shortlist: list[dict[str, Any]],
    ) -> list[_TransitionCandidate]:
        candidate_map: dict[str, _TransitionCandidate] = {}
        current_chunk_id = path.chunk_ids[-1]
        current_node_id = path.node_ids[-1]
        current_chunk = self._chunk_for_id(current_chunk_id)
        current_node = self._node_for_id(current_node_id)

        for item in self._neighbor_chunks(current_chunk_id):
            chunk = item["chunk"]
            if not chunk or str(chunk.get("chunk_id") or "") in path.chunk_ids:
                continue
            if self._cooldown_penalty(str(chunk.get("node_id") or "")) >= 0.22:
                continue
            node = self._node_for_id(str(chunk.get("node_id") or ""))
            self._upsert_transition(candidate_map, chunk, node, "neighbor", float(item.get("weight") or 0.0))

        for edge in self._node_edges(current_node_id, path.beam_type):
            neighbor_node_id = edge.get("neighbor_node_id")
            if neighbor_node_id:
                target_node_id = str(neighbor_node_id)
            elif edge.get("source_id") == current_node_id:
                target_node_id = str(edge.get("target_id") or "")
            else:
                target_node_id = str(edge.get("source_id") or "")
            if not target_node_id or target_node_id in path.node_ids:
                continue
            if self._cooldown_penalty(target_node_id) >= 0.28:
                continue
            node = self._node_for_id(target_node_id)
            chunk = self._best_chunk_for_node(target_node_id, query, ann_shortlist)
            if not chunk:
                continue
            relation_weight = float((edge.get("mode_scores") or {}).get(path.beam_type, edge.get("best_mode_score") or 0.0))
            self._upsert_transition(candidate_map, chunk, node, "edge", relation_weight, edge=edge)

        ann_limit = max(2, self._effective_neighbors_per_hop() * 2)
        for chunk in ann_shortlist[:ann_limit]:
            chunk_id = str(chunk.get("chunk_id") or "")
            node_id = str(chunk.get("node_id") or "")
            if not chunk_id or chunk_id in path.chunk_ids or node_id in path.node_ids:
                continue
            if self._cooldown_penalty(node_id) >= 0.34:
                continue
            if not self._shortlist_candidate_allowed(chunk, current_chunk, current_node, seed_context):
                continue
            node = self._node_for_id(node_id)
            base_weight = float(chunk.get("ann_similarity") or lexical_score(query, self._candidate_text(chunk, node)))
            self._upsert_transition(candidate_map, chunk, node, "ann", base_weight)

        support_pool = seed_context.get("support_pool") or []
        for chunk in support_pool[:ann_limit]:
            chunk_id = str(chunk.get("chunk_id") or "")
            node_id = str(chunk.get("node_id") or "")
            if not chunk_id or chunk_id in path.chunk_ids or node_id in path.node_ids:
                continue
            if self._cooldown_penalty(node_id) >= 0.28:
                continue
            node = self._node_for_id(node_id)
            relation = lexical_score(self._candidate_text(current_chunk, current_node), self._candidate_text(chunk, node))
            if relation < 0.08:
                continue
            self._upsert_transition(candidate_map, dict(chunk), node, "support", relation)

        return sorted(candidate_map.values(), key=lambda item: item.relation_weight + item.ann_similarity, reverse=True)

    def _upsert_transition(
        self,
        candidate_map: dict[str, _TransitionCandidate],
        chunk: dict[str, Any],
        node: dict[str, Any],
        source: str,
        relation_weight: float,
        *,
        edge: dict[str, Any] | None = None,
    ) -> None:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            return
        existing = candidate_map.get(chunk_id)
        ann_similarity = float(chunk.get("ann_similarity") or 0.0)
        if existing is None:
            candidate_map[chunk_id] = _TransitionCandidate(
                chunk=dict(chunk),
                node=dict(node),
                source=source,
                relation_weight=relation_weight,
                ann_similarity=ann_similarity,
                edge=edge,
            )
            return
        if relation_weight > existing.relation_weight:
            existing.relation_weight = relation_weight
            existing.source = source
            existing.edge = edge or existing.edge
        existing.ann_similarity = max(existing.ann_similarity, ann_similarity)

    def _score_transition(
        self,
        *,
        path: BeamPath,
        candidate: _TransitionCandidate,
        hop: int,
        query: str,
        profile: QueryProfile,
        seed_context: dict[str, Any],
    ) -> BeamPath | None:
        chunk_id = str(candidate.chunk.get("chunk_id") or "")
        node_id = str(candidate.node.get("id") or candidate.chunk.get("node_id") or "")
        stable_path_key = path.signature or "|".join(path.node_ids + path.chunk_ids) or path.path_id
        score_key = (stable_path_key, chunk_id, node_id, path.beam_type, str(hop))
        if score_key in self._path_score_cache:
            self._cache_stats["path_score_hits"] += 1
            metrics = self._path_score_cache[score_key]
        else:
            self._cache_stats["path_score_misses"] += 1
            current_node_id = path.node_ids[-1]
            current_objects = self._node_objects(current_node_id)
            candidate_objects = self._node_objects(node_id)
            candidate_text = self._candidate_text(candidate.chunk, candidate.node)
            path_text = self._path_text(path)
            anchor_text = str(seed_context.get("anchor_text_by_seed", {}).get(path.seed_chunk_id, ""))
            step_relevance = min(
                1.0,
                lexical_score(query, candidate_text) * 0.68
                + lexical_score(anchor_text or path_text, candidate_text) * 0.22
                + self._operator_alignment(path.beam_type, candidate, query, profile) * 0.10,
            )
            step_novelty = clamp(
                1.0
                - max(
                    lexical_score(seed_context.get("primary_text", "")[:1200], candidate_text[:1200]),
                    lexical_score(path_text[:1200], candidate_text[:1200]),
                ),
                0.0,
                1.0,
            )
            object_signal = self._object_signal(path.beam_type, current_objects, candidate_objects)
            step_support = clamp(
                candidate.relation_weight * 0.46
                + candidate.ann_similarity * 0.22
                + self._edge_support(candidate.edge) * 0.22
                + object_signal * 0.16,
                0.0,
                1.0,
            )
            step_feasibility = clamp(
                step_support * 0.58
                + (0.18 if candidate.source in {"neighbor", "edge", "support"} else 0.08)
                + (0.12 if candidate.chunk.get("content_ref") or candidate.chunk.get("source_path") else 0.04)
                + max(0.0, 0.16 - hop * 0.05),
                0.0,
                1.0,
            )
            step_diversity = self._diversity_score(path, candidate, seed_context)
            step_conflict = self._conflict_signal(path.beam_type, profile, step_support, step_novelty, current_objects, candidate_objects)
            step_redundancy = self._redundancy_penalty(path, candidate, seed_context)

            metrics = {
                "relevance": round((path.relevance * path.hop_count + step_relevance) / (path.hop_count + 1), 4),
                "novelty": round(max(path.novelty * 0.78, step_novelty), 4),
                "support": round((path.support * path.hop_count + step_support) / (path.hop_count + 1), 4),
                "feasibility": round((path.feasibility * path.hop_count + step_feasibility) / (path.hop_count + 1), 4),
                "diversity": round(max(path.diversity, step_diversity), 4),
                "conflict_risk": round(clamp(max(path.conflict_risk * 0.55, step_conflict), 0.0, 1.0), 4),
                "redundancy_penalty": round(clamp(max(path.redundancy_penalty * 0.7, step_redundancy), 0.0, 1.0), 4),
            }
            endpoint_score = clamp(
                step_relevance * 0.46 + step_support * 0.26 + step_feasibility * 0.18 + candidate.ann_similarity * 0.1,
                0.0,
                1.0,
            )
            trajectory_score = self.creative_engine.score_path(
                relevance=metrics["relevance"],
                novelty=metrics["novelty"],
                support=metrics["support"],
                feasibility=metrics["feasibility"],
                diversity=metrics["diversity"],
                conflict_risk=metrics["conflict_risk"],
                redundancy_penalty=metrics["redundancy_penalty"],
                novelty_weight=self.config.creative_novelty_weight,
                support_weight=self.config.creative_support_weight,
                diversity_weight=self.config.creative_diversity_weight,
                conflict_penalty=self.config.creative_conflict_penalty,
                reflection_gain=path.backflow_gain,
            )
            backflow_score = clamp(path.backflow_gain, 0.0, 1.0)
            metrics["endpoint_score"] = round(endpoint_score, 4)
            metrics["trajectory_score"] = round(trajectory_score, 4)
            metrics["backflow_score"] = round(backflow_score, 4)
            metrics["score"] = round(
                clamp(endpoint_score * 0.42 + trajectory_score * 0.44 + backflow_score * 0.14, 0.0, 1.6),
                4,
            )
            self._path_score_cache[score_key] = metrics

        if metrics["score"] <= 0.18:
            return None

        summary = self._compose_summary(path, candidate)
        path_role = self._classify_path_role(path.beam_type, metrics, candidate)
        signature = self._path_signature(path, candidate, path_role)
        reflections = [
            f"{path.beam_type}:{candidate.source}",
            f"support={metrics['support']:.2f}",
            f"novelty={metrics['novelty']:.2f}",
        ]
        amplified_score = round(metrics["score"] + path.backflow_gain, 4)
        self._round_frontier_hits[node_id] = self._round_frontier_hits.get(node_id, 0) + 1
        return BeamPath(
            beam_type=path.beam_type,
            seed_node_id=path.seed_node_id,
            seed_chunk_id=path.seed_chunk_id,
            node_ids=[*path.node_ids, node_id],
            chunk_ids=[*path.chunk_ids, chunk_id],
            hop_count=hop,
            relevance=metrics["relevance"],
            novelty=metrics["novelty"],
            support=metrics["support"],
            feasibility=metrics["feasibility"],
            diversity=metrics["diversity"],
            conflict_risk=metrics["conflict_risk"],
            redundancy_penalty=metrics["redundancy_penalty"],
            score=metrics["score"],
            endpoint_score=metrics["endpoint_score"],
            trajectory_score=metrics["trajectory_score"],
            backflow_score=metrics["backflow_score"],
            amplified_score=amplified_score,
            mmr_score=amplified_score,
            backflow_gain=path.backflow_gain,
            path_role=path_role,
            signature=signature,
            summary=summary,
            reflections=reflections,
            evidence_anchor_ids=list(dict.fromkeys(path.evidence_anchor_ids + [path.seed_chunk_id])),
            metadata={
                "parent_path_id": path.path_id,
                "source": candidate.source,
                "terminal_node_id": node_id,
                "terminal_chunk_id": chunk_id,
                "seed_cluster_id": path.metadata.get("seed_cluster_id"),
                "score_breakdown": metrics,
            },
        )

    def _operator_alignment(
        self,
        beam_type: str,
        candidate: _TransitionCandidate,
        query: str,
        profile: QueryProfile,
    ) -> float:
        text = self._candidate_text(candidate.chunk, candidate.node).lower()
        current_sector = str(candidate.node.get("sector") or candidate.chunk.get("sector") or "")
        temporal_hint = 1.0 if ("temporal" in beam_type or profile.needs_temporal_objects) and self._looks_temporal(text) else 0.0
        if beam_type == "semantic":
            return clamp(candidate.relation_weight * 0.8 + candidate.ann_similarity * 0.2, 0.0, 1.0)
        if beam_type == "analogy":
            edge_bonus = self._edge_support(candidate.edge)
            bridge_bonus = 1.0 if candidate.source in {"edge", "support"} else 0.0
            return clamp(edge_bonus * 0.55 + bridge_bonus * 0.2 + candidate.ann_similarity * 0.15 + lexical_score(query, text) * 0.1, 0.0, 1.0)
        if beam_type == "contrast":
            contrast_tokens = {"not", "avoid", "versus", "instead", "risk", "tradeoff", "different", "but"}
            contrast_hit = 1.0 if contrast_tokens & set(token_tuple(text)) else 0.0
            return clamp(contrast_hit * 0.45 + (1.0 - min(1.0, candidate.relation_weight)) * 0.2 + candidate.ann_similarity * 0.15 + (0.2 if current_sector else 0.0), 0.0, 1.0)
        if beam_type == "transfer":
            cross_sector = 1.0 if candidate.source in {"edge", "ann"} and current_sector not in {"", "project"} else 0.0
            return clamp(self._edge_support(candidate.edge) * 0.45 + cross_sector * 0.25 + candidate.ann_similarity * 0.2, 0.0, 1.0)
        if beam_type == "temporal":
            return clamp(
                temporal_hint * 0.45
                + candidate.relation_weight * 0.3
                + self._edge_support(candidate.edge) * 0.15
                + candidate.ann_similarity * 0.1,
                0.0,
                1.0,
            )
        if beam_type == "composition":
            return clamp(candidate.relation_weight * 0.35 + candidate.ann_similarity * 0.2 + (0.3 if candidate.source in {"support", "edge"} else 0.0), 0.0, 1.0)
        return 0.0

    def _object_signal(
        self,
        beam_type: str,
        current_objects: list[dict[str, Any]],
        candidate_objects: list[dict[str, Any]],
    ) -> float:
        if not current_objects and not candidate_objects:
            return 0.0
        current_types = {str(obj.get("object_type") or "") for obj in current_objects}
        candidate_types = {str(obj.get("object_type") or "") for obj in candidate_objects}
        current_entities = {
            str(obj.get("canonical_key") or obj.get("entity") or obj.get("attribute") or "")
            for obj in current_objects
            if obj.get("canonical_key") or obj.get("entity") or obj.get("attribute")
        }
        candidate_entities = {
            str(obj.get("canonical_key") or obj.get("entity") or obj.get("attribute") or "")
            for obj in candidate_objects
            if obj.get("canonical_key") or obj.get("entity") or obj.get("attribute")
        }
        shared_entities = len(current_entities & candidate_entities)
        shared_bonus = min(0.18, shared_entities * 0.09)
        if beam_type == "temporal":
            temporal_bonus = 0.16 if candidate_types & {"state_update", "temporal_reference"} else 0.0
            if current_types & {"state_update", "temporal_reference"} and candidate_types & {"state_update", "temporal_reference"}:
                temporal_bonus += 0.1
            if shared_entities and candidate_types & {"state_update", "temporal_reference"}:
                temporal_bonus += min(0.08, shared_entities * 0.04)
            return clamp(shared_bonus + temporal_bonus, 0.0, 1.0)
        if beam_type == "transfer":
            return clamp(shared_bonus + (0.18 if "solution_card" in candidate_types else 0.0), 0.0, 1.0)
        if beam_type == "contrast":
            current_polarities = {float(obj.get("polarity")) for obj in current_objects if obj.get("polarity") is not None}
            candidate_polarities = {float(obj.get("polarity")) for obj in candidate_objects if obj.get("polarity") is not None}
            opposite = any(a * b < 0 for a in current_polarities for b in candidate_polarities)
            return clamp(shared_bonus + (0.22 if opposite else 0.0), 0.0, 1.0)
        if beam_type == "composition":
            return clamp(shared_bonus + (0.14 if candidate_types - current_types else 0.0), 0.0, 1.0)
        if beam_type == "analogy":
            return clamp(shared_bonus + (0.12 if candidate_types & {"solution_card", "personal_context"} else 0.0), 0.0, 1.0)
        return clamp(shared_bonus, 0.0, 1.0)

    def _conflict_signal(
        self,
        beam_type: str,
        profile: QueryProfile,
        support: float,
        novelty: float,
        current_objects: list[dict[str, Any]],
        candidate_objects: list[dict[str, Any]],
    ) -> float:
        risk = 0.0
        if profile.needs_exact_evidence and beam_type in {"analogy", "contrast", "transfer", "composition"}:
            risk += 0.22
        if support < 0.22 and novelty > 0.72:
            risk += 0.18
        current_polarities = {float(obj.get("polarity")) for obj in current_objects if obj.get("polarity") is not None}
        candidate_polarities = {float(obj.get("polarity")) for obj in candidate_objects if obj.get("polarity") is not None}
        if beam_type == "contrast" and current_polarities and candidate_polarities and not any(a * b < 0 for a in current_polarities for b in candidate_polarities):
            risk += 0.08
        return clamp(risk, 0.0, 1.0)

    def _diversity_score(
        self,
        path: BeamPath,
        candidate: _TransitionCandidate,
        seed_context: dict[str, Any],
    ) -> float:
        node_id = str(candidate.node.get("id") or candidate.chunk.get("node_id") or "")
        sector = str(candidate.node.get("sector") or candidate.chunk.get("sector") or "")
        zone = str(candidate.node.get("zone") or candidate.chunk.get("zone") or "")
        seed_node = seed_context.get("seed_node_map", {}).get(path.seed_node_id, {})
        score = 0.0
        if node_id and node_id not in path.node_ids:
            score += 0.28
        if sector and sector != str(seed_node.get("sector") or ""):
            score += 0.18
        if zone and zone != str(seed_node.get("zone") or ""):
            score += 0.14
        if candidate.source in {"edge", "support"}:
            score += 0.08
        return clamp(score, 0.0, 1.0)

    def _redundancy_penalty(
        self,
        path: BeamPath,
        candidate: _TransitionCandidate,
        seed_context: dict[str, Any],
    ) -> float:
        candidate_text = self._candidate_text(candidate.chunk, candidate.node)
        overlap = lexical_score(candidate_text[:600], seed_context.get("primary_text", "")[:600])
        overlap = max(overlap, lexical_score(candidate_text[:600], self._path_text(path)[:600]))
        if str(candidate.node.get("id") or candidate.chunk.get("node_id") or "") in path.node_ids:
            overlap += 0.12
        return clamp(overlap, 0.0, 1.0)

    def _compose_summary(self, path: BeamPath, candidate: _TransitionCandidate) -> str:
        seed_summary = self._node_for_id(path.seed_node_id).get("summary") or self._chunk_for_id(path.seed_chunk_id).get("text") or ""
        candidate_summary = candidate.node.get("summary") or candidate.chunk.get("text") or ""
        summary = f"{path.beam_type} path: {seed_summary[:90]} -> {candidate_summary[:120]}"
        return summary.strip()

    def _classify_path_role(
        self,
        beam_type: str,
        metrics: dict[str, float],
        candidate: _TransitionCandidate,
    ) -> str:
        if beam_type in {"semantic", "temporal"} and metrics["support"] >= metrics["novelty"]:
            return "support"
        if beam_type in {"analogy", "composition"} or metrics["novelty"] >= 0.36:
            return "creative_reflection"
        if beam_type in {"contrast", "transfer"} or metrics["diversity"] >= 0.32:
            return "alternative"
        if candidate.source == "support":
            return "support"
        return "alternative"

    def _path_signature(self, path: BeamPath, candidate: _TransitionCandidate, path_role: str) -> str:
        seed_cluster = str(path.metadata.get("seed_cluster_id") or path.seed_chunk_id or "")
        endpoint_sector = str(candidate.node.get("sector") or candidate.chunk.get("sector") or "")
        endpoint_zone = str(candidate.node.get("zone") or candidate.chunk.get("zone") or "")
        terminal_tag = "temporal" if self._looks_temporal(self._candidate_text(candidate.chunk, candidate.node).lower()) else "static"
        sources = [str(note.split(":", 1)[0]) for note in path.reflections[:2] if ":" in note]
        source_sig = ",".join(sorted(set(sources + [candidate.source]))[:2])
        return f"{path_role}|{path.beam_type}|{seed_cluster}|{source_sig}|{endpoint_sector}|{endpoint_zone}|{terminal_tag}"

    def _apply_backflow(self, parents: list[BeamPath], children: list[BeamPath]) -> None:
        grouped: dict[str, list[BeamPath]] = {}
        for child in children:
            parent_id = str(child.metadata.get("parent_path_id") or "")
            if not parent_id:
                continue
            grouped.setdefault(parent_id, []).append(child)
        for parent in parents:
            descendants = grouped.get(parent.path_id, [])
            if not descendants:
                continue
            best_child = max(descendants, key=lambda item: item.score)
            gain = max(0.0, best_child.score - parent.score) * self.config.creative_reflection_gain
            parent.backflow_gain = round(parent.backflow_gain + gain, 4)
            parent.backflow_score = round(clamp(parent.backflow_gain, 0.0, 1.0), 4)
            parent.amplified_score = round(clamp(parent.score + parent.backflow_gain, 0.0, 1.6), 4)

    def _amplify_paths(self, paths: list[BeamPath]) -> list[BeamPath]:
        if not paths:
            return []
        pivot_scores = sorted(path.amplified_score for path in paths)
        pivot = pivot_scores[len(pivot_scores) // 2]
        rounds = max(1, min(3, self.config.creative_max_hops))
        for _ in range(rounds):
            for path in paths:
                quality = (path.endpoint_score + path.trajectory_score + path.support) / 3.0
                if path.amplified_score >= pivot and path.support >= 0.22:
                    boost = self.config.creative_reflection_gain * quality * (1.0 - path.conflict_risk * 0.7)
                    path.amplified_score = round(clamp(path.amplified_score + boost, 0.0, 1.6), 4)
                else:
                    decay = 0.03 * path.conflict_risk
                    path.amplified_score = round(max(path.score, path.amplified_score - decay), 4)
                path.backflow_score = round(clamp(path.backflow_gain, 0.0, 1.0), 4)
        return paths

    def _select_frontier(self, paths: list[BeamPath]) -> list[BeamPath]:
        selected: list[BeamPath] = []
        seen_terminal_nodes: set[str] = set()
        seen_signatures: set[str] = set()
        for path in sorted(paths, key=lambda item: item.amplified_score, reverse=True):
            terminal_node = path.node_ids[-1] if path.node_ids else ""
            if terminal_node and terminal_node in seen_terminal_nodes:
                continue
            if path.signature and path.signature in seen_signatures:
                continue
            if selected and any(self._path_overlap(path, other) >= 0.9 for other in selected):
                continue
            selected.append(path)
            if terminal_node:
                seen_terminal_nodes.add(terminal_node)
            if path.signature:
                seen_signatures.add(path.signature)
            if len(selected) >= self._effective_beam_width():
                break
        return selected

    def _bucket_paths(self, paths: list[BeamPath]) -> tuple[list[BeamPath], list[BeamPath], list[BeamPath]]:
        support_paths: list[BeamPath] = []
        creative_paths: list[BeamPath] = []
        alternative_paths: list[BeamPath] = []
        for path in paths:
            if path.hop_count <= 0 or path.conflict_risk >= 0.5:
                continue
            if self.config.creative_is_conservative and path.beam_type in {"semantic", "temporal"}:
                support_paths.append(path)
                continue
            if path.path_role == "support":
                support_paths.append(path)
            elif path.path_role == "creative_reflection":
                creative_paths.append(path)
            else:
                alternative_paths.append(path)
        return support_paths, creative_paths, alternative_paths

    def _select_output_paths(
        self,
        *,
        support_paths: list[BeamPath],
        creative_paths: list[BeamPath],
        alternative_paths: list[BeamPath],
        limit: int,
        exact_mode: bool,
    ) -> list[BeamPath]:
        selected: list[BeamPath] = []
        quota_support = 1 if self.config.creative_is_conservative or exact_mode else max(1, min(2, limit // 2 or 1))
        chosen_support = self._mmr_select(support_paths, limit=min(quota_support, limit), lambda_relevance=0.78)
        if self.config.creative_is_conservative:
            for path in chosen_support:
                path.path_role = "support"
                path.signature = path.signature.replace("creative_reflection|", "support|", 1) if path.signature.startswith("creative_reflection|") else path.signature
        selected.extend(chosen_support)
        remaining = max(0, limit - len(selected))
        if remaining <= 0:
            return selected[:limit]
        exploratory_pool = creative_paths + alternative_paths
        if self.config.creative_is_conservative and exact_mode:
            exploratory_pool = [path for path in alternative_paths if path.beam_type in {"semantic", "temporal"}]
        selected.extend(self._mmr_select(exploratory_pool, limit=remaining, lambda_relevance=0.68, existing=selected))
        return selected[:limit]

    def _mmr_select(
        self,
        paths: list[BeamPath],
        *,
        limit: int,
        lambda_relevance: float,
        existing: list[BeamPath] | None = None,
    ) -> list[BeamPath]:
        if limit <= 0 or not paths:
            return []
        selected: list[BeamPath] = list(existing or [])
        local_selected: list[BeamPath] = []
        candidates = sorted(paths, key=lambda item: item.amplified_score, reverse=True)
        seen_signatures = {path.signature for path in selected if path.signature}
        while candidates and len(local_selected) < limit:
            best_index = -1
            best_score = -999.0
            for index, path in enumerate(candidates):
                if path.signature and path.signature in seen_signatures:
                    continue
                similarity = max((self._path_similarity(path, other) for other in selected), default=0.0)
                mmr_score = lambda_relevance * path.amplified_score - (1.0 - lambda_relevance) * similarity
                if best_index < 0 or mmr_score > best_score:
                    best_index = index
                    best_score = mmr_score
            if best_index < 0:
                break
            chosen = candidates.pop(best_index)
            chosen.mmr_score = round(best_score, 4)
            selected.append(chosen)
            local_selected.append(chosen)
            if chosen.signature:
                seen_signatures.add(chosen.signature)
        return local_selected

    def _path_overlap(self, path_a: BeamPath, path_b: BeamPath) -> float:
        node_overlap = len(set(path_a.node_ids) & set(path_b.node_ids)) / max(1, min(len(path_a.node_ids), len(path_b.node_ids)))
        chunk_overlap = len(set(path_a.chunk_ids) & set(path_b.chunk_ids)) / max(1, min(len(path_a.chunk_ids), len(path_b.chunk_ids)))
        text_overlap = lexical_score(path_a.summary[:600], path_b.summary[:600])
        return clamp(max(node_overlap, chunk_overlap, text_overlap), 0.0, 1.0)

    def _path_similarity(self, path_a: BeamPath, path_b: BeamPath) -> float:
        signature_overlap = 1.0 if path_a.signature and path_a.signature == path_b.signature else 0.0
        role_overlap = 0.22 if path_a.path_role == path_b.path_role else 0.0
        return clamp(max(self._path_overlap(path_a, path_b), signature_overlap, role_overlap + lexical_score(path_a.summary[:500], path_b.summary[:500])), 0.0, 1.0)

    def _path_text(self, path: BeamPath) -> str:
        parts: list[str] = []
        for chunk_id in path.chunk_ids:
            chunk = self._chunk_for_id(chunk_id)
            if chunk.get("text"):
                parts.append(str(chunk["text"]))
        return "\n".join(parts)

    def _cooldown_penalty(self, node_id: str) -> float:
        if not node_id:
            return 0.0
        hits = self._round_frontier_hits.get(node_id, 0)
        return min(0.4, max(0, hits - 1) * 0.12)

    def _neighbor_chunks(self, chunk_id: str) -> list[dict[str, Any]]:
        if chunk_id in self._neighbor_cache:
            self._cache_stats["local_neighborhood_hits"] += 1
            return self._neighbor_cache[chunk_id]
        self._cache_stats["local_neighborhood_misses"] += 1
        grouped_links = self.storage.get_local_chunk_neighbors(
            [chunk_id],
            limit_per_chunk=max(self._effective_neighbors_per_hop() * 4, 4 if self.config.creative_is_conservative else 8),
            min_weight=0.0,
        )
        links = grouped_links.get(chunk_id, [])
        neighbor_ids: list[str] = []
        weights: dict[str, float] = {}
        for link in links:
            neighbor_id = str(link.get("neighbor_chunk_id") or "")
            if not neighbor_id or neighbor_id in neighbor_ids:
                continue
            neighbor_ids.append(neighbor_id)
            weights[neighbor_id] = float(link.get("weight") or 0.0)
        chunks = self.storage.fetch_chunks_with_node_metadata_by_ids(neighbor_ids) if neighbor_ids else []
        rows = [{"chunk": dict(chunk), "weight": weights.get(str(chunk.get("chunk_id") or ""), 0.0)} for chunk in chunks]
        self._neighbor_cache[chunk_id] = rows
        for chunk in chunks:
            self._chunk_cache[str(chunk["chunk_id"])] = dict(chunk)
        return rows

    def _node_edges(self, node_id: str, beam_type: str) -> list[dict[str, Any]]:
        cache_key = (node_id, beam_type)
        if cache_key in self._beam_edge_cache:
            self._cache_stats["beam_adjacency_hits"] += 1
            return self._beam_edge_cache[cache_key]
        self._cache_stats["beam_adjacency_misses"] += 1
        grouped_edges = self.storage.get_top_neighbor_edges(
            [node_id],
            edge_modes=[beam_type],
            limit_per_node=max(self._effective_neighbors_per_hop() * 3, 4 if self.config.creative_is_conservative else 6),
            min_weight=0.06,
        )
        edges = grouped_edges.get(node_id, [])
        self._beam_edge_cache[cache_key] = edges
        self._edge_cache[node_id] = edges
        return edges

    def _node_objects(self, node_id: str) -> list[dict[str, Any]]:
        if node_id in self._object_cache:
            return self._object_cache[node_id]
        rows = self.storage.fetch_objects_for_nodes(
            [node_id],
            object_types=["preference", "state_update", "temporal_reference", "solution_card", "personal_context"],
        )
        self._object_cache[node_id] = rows
        return rows

    def _chunk_for_id(self, chunk_id: str) -> dict[str, Any]:
        if chunk_id in self._chunk_cache:
            return self._chunk_cache[chunk_id]
        if hasattr(self.storage, "fetch_chunks_with_node_metadata_by_ids"):
            rows = self.storage.fetch_chunks_with_node_metadata_by_ids([chunk_id])
            if rows:
                self._chunk_cache[chunk_id] = dict(rows[0])
                return self._chunk_cache[chunk_id]
        return {"chunk_id": chunk_id}

    def _node_for_id(self, node_id: str) -> dict[str, Any]:
        if node_id in self._node_cache:
            return self._node_cache[node_id]
        rows = self.storage.get_nodes_by_ids([node_id]) if node_id and hasattr(self.storage, "get_nodes_by_ids") else {}
        if rows:
            self._node_cache.update({key: dict(value) for key, value in rows.items()})
            return self._node_cache.get(node_id, {"id": node_id})
        legacy_rows = self.storage.fetch_nodes_by_ids([node_id]) if node_id else []
        if legacy_rows:
            self._node_cache[node_id] = dict(legacy_rows[0])
            return self._node_cache[node_id]
        return {"id": node_id}

    def _best_chunk_for_node(self, node_id: str, query: str, ann_shortlist: list[dict[str, Any]]) -> dict[str, Any]:
        cached = self._node_chunk_cache.get(node_id)
        if cached:
            return dict(cached[0])
        shortlist = [dict(chunk) for chunk in ann_shortlist if str(chunk.get("node_id") or "") == node_id]
        if not shortlist and hasattr(self.storage, "fetch_chunks_for_node"):
            shortlist = [dict(chunk) for chunk in self.storage.fetch_chunks_for_node(node_id)]
        if not shortlist:
            node = self._node_for_id(node_id)
            if node.get("summary"):
                shortlist = [
                    {
                        "chunk_id": f"node:{node_id}",
                        "node_id": node_id,
                        "text": str(node.get("summary") or ""),
                        "shell": node.get("shell"),
                        "sector": node.get("sector"),
                        "zone": node.get("zone"),
                        "cell": node.get("cell"),
                        "grain": "macro",
                    }
                ]
        if not shortlist:
            return {}
        shortlist.sort(
            key=lambda item: (
                float(item.get("ann_similarity") or 0.0),
                lexical_score(query, self._candidate_text(item, self._node_for_id(node_id))),
            ),
            reverse=True,
        )
        self._node_chunk_cache[node_id] = shortlist
        best = dict(shortlist[0])
        self._chunk_cache[str(best.get("chunk_id") or "")] = best
        return best

    def _candidate_text(self, chunk: dict[str, Any], node: dict[str, Any]) -> str:
        return " ".join(
            filter(
                None,
                [
                    str(chunk.get("text") or ""),
                    str(node.get("summary") or ""),
                    str(chunk.get("cell") or node.get("cell") or ""),
                    str(chunk.get("zone") or node.get("zone") or ""),
                ],
            )
        )

    def _shortlist_candidate_allowed(
        self,
        chunk: dict[str, Any],
        current_chunk: dict[str, Any],
        current_node: dict[str, Any],
        seed_context: dict[str, Any],
    ) -> bool:
        if str(chunk.get("zone") or "") == str(current_chunk.get("zone") or current_node.get("zone") or ""):
            return True
        if str(chunk.get("sector") or "") == str(current_chunk.get("sector") or current_node.get("sector") or ""):
            return True
        text = self._candidate_text(chunk, self._node_for_id(str(chunk.get("node_id") or "")))
        return lexical_score(text[:500], seed_context.get("primary_text", "")[:500]) >= 0.14

    def _edge_weight(self, edge: dict[str, Any] | None, beam_type: str) -> float:
        if not edge:
            return 0.0
        semantic = float(edge.get("semantic_weight") or 0.0)
        task = float(edge.get("task_weight") or 0.0)
        temporal = float(edge.get("temporal_weight") or 0.0)
        causal = float(edge.get("causal_weight") or 0.0)
        creative = float(edge.get("creative_weight") or 0.0)
        structural = float(edge.get("structural_weight") or 0.0)
        if beam_type == "semantic":
            return semantic * 0.58 + task * 0.14 + structural * 0.16 + causal * 0.12
        if beam_type == "analogy":
            return creative * 0.45 + structural * 0.35 + semantic * 0.2
        if beam_type == "contrast":
            return structural * 0.25 + semantic * 0.15 + creative * 0.2 + task * 0.1
        if beam_type == "transfer":
            return structural * 0.34 + task * 0.22 + creative * 0.18 + semantic * 0.12 + causal * 0.14
        if beam_type == "temporal":
            return temporal * 0.45 + causal * 0.3 + semantic * 0.1 + structural * 0.1 + task * 0.05
        if beam_type == "composition":
            return structural * 0.32 + semantic * 0.18 + task * 0.18 + creative * 0.14 + causal * 0.1
        return semantic

    def _edge_support(self, edge: dict[str, Any] | None) -> float:
        if not edge:
            return 0.0
        return clamp(
            float(edge.get("semantic_weight") or 0.0) * 0.22
            + float(edge.get("task_weight") or 0.0) * 0.14
            + float(edge.get("temporal_weight") or 0.0) * 0.18
            + float(edge.get("causal_weight") or 0.0) * 0.18
            + float(edge.get("creative_weight") or 0.0) * 0.12
            + float(edge.get("structural_weight") or 0.0) * 0.16,
            0.0,
            1.0,
        )

    def _looks_temporal(self, text: str) -> bool:
        temporal_terms = {"before", "after", "latest", "current", "timeline", "today", "yesterday", "tomorrow", "update", "updated"}
        tokens = set(token_tuple(text))
        return bool(tokens & temporal_terms)

    def _path_to_reflection(self, path: BeamPath) -> dict[str, Any]:
        terminal_node = self._node_for_id(path.node_ids[-1])
        return {
            "id": str(terminal_node.get("id") or path.node_ids[-1] or path.path_id),
            "node_id": str(terminal_node.get("id") or path.node_ids[-1] or ""),
            "shell": terminal_node.get("shell"),
            "sector": terminal_node.get("sector"),
            "zone": terminal_node.get("zone"),
            "cell": terminal_node.get("cell"),
            "molecular_type": terminal_node.get("molecular_type", "creative_reflection"),
            "content_ref": terminal_node.get("content_ref"),
            "raw_content": terminal_node.get("raw_content"),
            "path_id": path.path_id,
            "beam_type": path.beam_type,
            "path_role": path.path_role,
            "signature": path.signature,
            "summary": str(terminal_node.get("summary") or path.summary),
            "reflection_score": round(path.amplified_score, 4),
            "refraction_score": round(path.score, 4),
            "endpoint_score": round(path.endpoint_score, 4),
            "trajectory_score": round(path.trajectory_score, 4),
            "backflow_score": round(path.backflow_score, 4),
            "support_score": round(path.support, 4),
            "novelty_score": round(path.novelty, 4),
            "feasibility_score": round(path.feasibility, 4),
            "diversity_score": round(path.diversity, 4),
            "conflict_risk": round(path.conflict_risk, 4),
            "hop_count": path.hop_count,
            "node_ids": list(path.node_ids),
            "chunk_ids": list(path.chunk_ids),
            "evidence_anchor_ids": list(path.evidence_anchor_ids),
            "reflection_notes": list(path.reflections),
        }

    def _path_to_alternative(self, path: BeamPath) -> dict[str, Any]:
        return {
            "path_id": path.path_id,
            "beam_type": path.beam_type,
            "path_role": path.path_role,
            "signature": path.signature,
            "seed_node_id": path.seed_node_id,
            "seed_chunk_id": path.seed_chunk_id,
            "hop_count": path.hop_count,
            "summary": path.summary,
            "score": round(path.amplified_score, 4),
            "base_score": round(path.score, 4),
            "endpoint_score": round(path.endpoint_score, 4),
            "trajectory_score": round(path.trajectory_score, 4),
            "backflow_score": round(path.backflow_score, 4),
            "mmr_score": round(path.mmr_score, 4),
            "reflection_backflow": round(path.backflow_gain, 4),
            "node_ids": list(path.node_ids),
            "chunk_ids": list(path.chunk_ids),
            "score_breakdown": {
                "relevance": round(path.relevance, 4),
                "novelty": round(path.novelty, 4),
                "support": round(path.support, 4),
                "feasibility": round(path.feasibility, 4),
                "diversity": round(path.diversity, 4),
                "conflict_risk": round(path.conflict_risk, 4),
                "redundancy_penalty": round(path.redundancy_penalty, 4),
            },
            "reflection_notes": list(path.reflections),
        }
