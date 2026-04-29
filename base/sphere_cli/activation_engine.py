from __future__ import annotations

from typing import Any

from .models import now_iso
from .storage import Storage
from .utils import lexical_score, clamp, tokenize
from .vector_store import VectorStore


class ActivationEngine:
    def __init__(self, storage: Storage, vector_store: VectorStore) -> None:
        self.storage = storage
        self.vector_store = vector_store

    def _score_main(self, task: str, node: dict[str, Any], preferred_shells: list[int], preferred_sectors: list[str]) -> float:
        text = " ".join(filter(None, [node.get("summary"), node.get("raw_content"), node.get("cell"), node.get("zone")]))
        semantic = lexical_score(task, text)
        shell_bonus = 0.12 if node["shell"] in preferred_shells else 0.0
        sector_bonus = 0.08 if node["sector"] in preferred_sectors else 0.0
        importance = 0.2 * float(node.get("importance") or 0.0)
        recency_like = min((node.get("access_count") or 0) / 50.0, 0.15)
        return clamp(semantic + shell_bonus + sector_bonus + importance + recency_like, 0.0, 1.8)

    def main_activation(self, task: str, preferred_shells: list[int], preferred_sectors: list[str], top_k: int = 8) -> list[dict[str, Any]]:
        graph_candidates = self._graph_candidates(task, preferred_shells, preferred_sectors, top_k=max(top_k * 2, 8))
        vector_candidates = self._vector_candidates(task, preferred_shells, preferred_sectors, top_k=max(top_k * 2, 8))

        merged: dict[str, dict[str, Any]] = {}
        for node in graph_candidates:
            item = dict(node)
            item.setdefault("activation_score", 0.0)
            item.setdefault("vector_score", 0.0)
            item["fusion_score"] = round(item["activation_score"] * 0.7, 4)
            merged[item["id"]] = item

        for node in vector_candidates:
            if node["id"] in merged:
                merged[node["id"]]["vector_score"] = max(merged[node["id"]].get("vector_score", 0.0), node.get("vector_score", 0.0))
                merged[node["id"]]["matched_chunk_id"] = node.get("matched_chunk_id")
                merged[node["id"]]["matched_chunk_text"] = node.get("matched_chunk_text")
                merged[node["id"]]["fusion_score"] = round(
                    merged[node["id"]].get("activation_score", 0.0) * 0.65 + node.get("vector_score", 0.0) * 0.55,
                    4,
                )
            else:
                item = dict(node)
                item.setdefault("activation_score", 0.0)
                item["fusion_score"] = round(item.get("vector_score", 0.0) * 0.75, 4)
                merged[item["id"]] = item

        ranked = sorted(merged.values(), key=lambda x: x.get("fusion_score", 0.0), reverse=True)[:top_k]
        ts = now_iso()
        for node in ranked:
            self.storage.update_node_access(node["id"], ts)
        return ranked

    def cognitive_expansion(
        self,
        task: str,
        evidence_nodes: list[dict[str, Any]],
        temperature: float,
        reflection_limit: int = 6,
        refraction_limit: int = 4,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        reflected = self.reflection_activation(task, evidence_nodes, temperature, limit=reflection_limit)
        refracted = self.refraction_activation(task, evidence_nodes, temperature, limit=refraction_limit)
        return reflected, refracted

    def _graph_candidates(self, task: str, preferred_shells: list[int], preferred_sectors: list[str], top_k: int) -> list[dict[str, Any]]:
        nodes = self.storage.fetch_nodes("stage != ?", ("archive",))
        scored = []
        for node in nodes:
            score = self._score_main(task, node, preferred_shells, preferred_sectors)
            if score > 0.08:
                node = dict(node)
                node["activation_score"] = round(score, 4)
                scored.append(node)
        scored.sort(key=lambda x: x["activation_score"], reverse=True)
        return scored[:top_k]

    def _vector_candidates(self, task: str, preferred_shells: list[int], preferred_sectors: list[str], top_k: int) -> list[dict[str, Any]]:
        results = self.vector_store.search(task, top_k=top_k * 2)
        best_by_node: dict[str, dict[str, Any]] = {}
        for item in results:
            metadata = item.get("metadata") or {}
            node_id = metadata.get("node_id")
            if not node_id:
                continue
            shell = int(metadata.get("shell", -1))
            sector = str(metadata.get("sector", ""))
            if shell not in preferred_shells and sector not in preferred_sectors:
                continue
            current = best_by_node.get(node_id)
            if current is None or item["similarity"] > current["similarity"]:
                best_by_node[node_id] = item
        node_ids = list(best_by_node.keys())[:top_k]
        nodes = self.storage.fetch_nodes_by_ids(node_ids)
        enriched = []
        for node in nodes:
            match = best_by_node[node["id"]]
            item = dict(node)
            item["vector_score"] = round(match["similarity"], 4)
            item["matched_chunk_id"] = match["chunk_id"]
            item["matched_chunk_text"] = match["document"][:240]
            enriched.append(item)
        enriched.sort(key=lambda x: x.get("vector_score", 0.0), reverse=True)
        return enriched[:top_k]

    def reflection_activation(self, task: str, main_nodes: list[dict[str, Any]], temperature: float, limit: int = 6) -> list[dict[str, Any]]:
        if not main_nodes:
            return []
        main_ids = {n["id"] for n in main_nodes}
        zone_set = {n["zone"] for n in main_nodes}
        neighbor_shells = {s for n in main_nodes for s in (n["shell"] - 1, n["shell"], n["shell"] + 1) if 0 <= s <= 4}
        candidate_nodes = self.storage.fetch_nodes("stage != ?", ("archive",))
        edges = self.storage.fetch_edges()
        creative_neighbors: dict[str, float] = {}
        for edge in edges:
            if edge["source_id"] in main_ids:
                creative_neighbors[edge["target_id"]] = max(creative_neighbors.get(edge["target_id"], 0.0), edge.get("creative_weight") or 0.0)
            if edge["target_id"] in main_ids:
                creative_neighbors[edge["source_id"]] = max(creative_neighbors.get(edge["source_id"], 0.0), edge.get("creative_weight") or 0.0)

        main_token_set = set(tokenize(" ".join(n.get("summary") or "" for n in main_nodes)))
        scored: list[dict[str, Any]] = []
        for node in candidate_nodes:
            if node["id"] in main_ids:
                continue
            text = " ".join(filter(None, [node.get("summary"), node.get("raw_content"), node.get("cell"), node.get("zone")]))
            base_rel = lexical_score(task, text)
            same_zone_bonus = 0.12 if node["zone"] in zone_set else 0.0
            adjacent_shell_bonus = 0.08 if node["shell"] in neighbor_shells else 0.0
            creative_edge_bonus = 0.25 * creative_neighbors.get(node["id"], 0.0)
            novelty = 1.0 - lexical_score(" ".join(main_token_set), text)
            novelty_bonus = 0.18 * novelty * temperature
            low_freq_bonus = 0.08 if (node.get("access_count") or 0) <= 2 else 0.0
            score = base_rel + same_zone_bonus + adjacent_shell_bonus + creative_edge_bonus + novelty_bonus + low_freq_bonus
            if score > 0.14:
                item = dict(node)
                item["reflection_score"] = round(score, 4)
                scored.append(item)
        scored.sort(key=lambda x: x["reflection_score"], reverse=True)
        return scored[:limit]

    def refraction_activation(self, task: str, main_nodes: list[dict[str, Any]], temperature: float, limit: int = 4) -> list[dict[str, Any]]:
        if not main_nodes or temperature <= 0.05:
            return []
        abstraction_tags = self._infer_abstractions(task)
        nodes = self.storage.fetch_nodes("stage != ?", ("archive",))
        results = []
        for node in nodes:
            text = " ".join(filter(None, [node.get("summary"), node.get("raw_content"), node.get("tags"), node.get("cell")])).lower()
            matches = sum(1 for tag in abstraction_tags if tag in text)
            if matches == 0:
                continue
            structural = float(node.get("creative_score") or 0.0) + float(node.get("stability_score") or 0.0) * 0.2
            score = clamp(matches * 0.22 + structural * 0.35 + temperature * 0.2, 0.0, 1.5)
            item = dict(node)
            item["refraction_score"] = round(score, 4)
            results.append(item)
        results.sort(key=lambda x: x["refraction_score"], reverse=True)
        main_ids = {n["id"] for n in main_nodes}
        return [r for r in results if r["id"] not in main_ids][:limit]

    def _infer_abstractions(self, task: str) -> list[str]:
        task_lower = task.lower()
        abstractions = []
        mapping = {
            "缓存": ["cache", "caching", "storage", "tiering"],
            "token": ["token", "budget", "compression", "context"],
            "并行": ["parallel", "concurrency", "scheduler", "queue"],
            "检索": ["retrieval", "routing", "index", "query"],
            "创意": ["analogy", "reflection", "creative", "weak link"],
            "记忆": ["memory", "graph", "activation", "hierarchy"],
            "日志": ["log", "trace", "event timeline", "observability"],
            "数据库": ["sqlite", "database", "write lock", "transaction"],
        }
        for key, vals in mapping.items():
            if key in task or any(v in task_lower for v in vals):
                abstractions.extend(vals)
        return abstractions or ["memory", "structure", "optimization"]
