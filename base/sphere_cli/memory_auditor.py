from __future__ import annotations

from collections import defaultdict
from typing import Any

from .storage import Storage
from .utils import lexical_score
from .vector_store import VectorStore


class MemoryAuditor:
    def __init__(self, storage: Storage, vector_store: VectorStore) -> None:
        self.storage = storage
        self.vector_store = vector_store

    def audit(self) -> dict[str, Any]:
        nodes = self.storage.fetch_nodes()
        edges = self.storage.fetch_edges()
        duplicates = self._find_duplicates(nodes)
        orphan_nodes = self._find_orphans(nodes, edges)
        zone_density = self._zone_density(nodes)
        weak_summaries = [n["id"] for n in nodes if len((n.get("summary") or "").strip()) < 12]
        tracked_files = self.storage.fetch_ingest_files()
        return {
            "node_count": len(nodes),
            "chunk_count": self.storage.count_chunks(),
            "object_count": len(self.storage.fetch_objects()),
            "vector_doc_count": self.vector_store.count(),
            "vector_info": self.vector_store.info(),
            "edge_count": len(edges),
            "tracked_file_count": len(tracked_files),
            "duplicate_pairs": duplicates,
            "orphan_nodes": orphan_nodes,
            "dense_zones": zone_density,
            "weak_summaries": weak_summaries,
        }

    def _find_duplicates(self, nodes: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
        pairs = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                sim = lexical_score(nodes[i].get("summary") or "", nodes[j].get("summary") or "")
                if sim > 0.82 and nodes[i]["zone"] == nodes[j]["zone"]:
                    pairs.append((nodes[i]["id"], nodes[j]["id"], round(sim, 4)))
        return pairs[:30]

    def _find_orphans(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
        touched = set()
        for edge in edges:
            touched.add(edge["source_id"])
            touched.add(edge["target_id"])
        return [n["id"] for n in nodes if n["id"] not in touched]

    def _zone_density(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counter: dict[tuple[int, str, str], int] = defaultdict(int)
        for n in nodes:
            counter[(n["shell"], n["sector"], n["zone"])] += 1
        dense = []
        for (shell, sector, zone), count in counter.items():
            if count >= 20:
                dense.append({"shell": shell, "sector": sector, "zone": zone, "count": count})
        dense.sort(key=lambda x: x["count"], reverse=True)
        return dense
