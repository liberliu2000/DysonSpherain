from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from .compression_elevator import CompressionElevator
from .config import AppConfig
from .memory_writer import MemoryWriter
from .models import now_iso
from .storage import Storage
from .vector_store import VectorStore


class BackgroundTaskRunner:
    def __init__(self, config: AppConfig, storage: Storage, vector_store: VectorStore, writer: MemoryWriter) -> None:
        self.config = config
        self.storage = storage
        self.vector_store = vector_store
        self.writer = writer
        self.compressor = CompressionElevator()

    def reembed_all(self, batch_size: int = 200) -> dict[str, Any]:
        chunks = self.storage.fetch_all_chunks()
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            hydrated = self.storage.fetch_chunks_with_node_metadata_by_ids([str(chunk["chunk_id"]) for chunk in batch])
            self.vector_store.upsert_chunks(hydrated)
            self.storage.mark_chunks_vector_synced([c["chunk_id"] for c in batch], now_iso())
        objects = self.storage.fetch_objects()
        for i in range(0, len(objects), batch_size):
            self.vector_store.upsert_objects(objects[i : i + batch_size])
        return {
            "task": "reembed_all",
            "chunks_reembedded": total,
            "objects_reembedded": len(objects),
            "batch_size": batch_size,
        }

    def compress_cold_data(self, cold_days: int | None = None, access_threshold: int | None = None, limit: int = 200) -> dict[str, Any]:
        cold_days = cold_days or self.config.cold_days_threshold
        access_threshold = access_threshold if access_threshold is not None else self.config.cold_access_threshold
        cutoff = datetime.now().astimezone() - timedelta(days=cold_days)
        nodes = self.storage.fetch_nodes("stage != ?", ("archive",))
        touched: list[str] = []
        for node in nodes:
            last_accessed = node.get("last_accessed_at") or node.get("created_at")
            try:
                dt = datetime.fromisoformat(str(last_accessed))
            except Exception:
                continue
            if dt > cutoff or int(node.get("access_count") or 0) > access_threshold:
                continue
            summary = self.compressor.compress_text((node.get("raw_content") or node.get("summary") or ""), target_ratio=0.25)
            self.storage.update_node_compression(
                node["id"],
                summary=summary,
                raw_content=None if int(node.get("shell") or 4) <= 3 else node.get("raw_content"),
                compression_level="high",
            )
            touched.append(node["id"])
            if len(touched) >= limit:
                break
        return {"task": "compress_cold_data", "nodes_compressed": len(touched), "node_ids": touched}

    def decay_edges(self, factor: float | None = None, floor: float | None = None) -> dict[str, Any]:
        factor = factor or self.config.edge_decay_factor
        floor = floor if floor is not None else self.config.edge_decay_floor
        count = self.storage.decay_edges(factor=factor, floor=floor)
        return {"task": "decay_edges", "edges_updated": count, "factor": factor, "floor": floor}

    def split_large_zones(self, threshold: int | None = None, group_size: int | None = None, apply_changes: bool = True) -> dict[str, Any]:
        threshold = threshold or self.config.zone_split_threshold
        group_size = group_size or self.config.zone_split_group_size
        zone_stats = self.storage.zone_counts()
        changes: list[dict[str, Any]] = []
        for item in zone_stats:
            if item["count"] <= threshold:
                continue
            nodes = self.storage.fetch_nodes("zone = ?", (item["zone"],))
            groups = defaultdict(list)
            for node in nodes:
                key = (node.get("cell") or "misc").split("_")[0] or "misc"
                groups[key].append(node)
            generated = 0
            for key, members in groups.items():
                for idx in range(0, len(members), group_size):
                    batch = members[idx : idx + group_size]
                    suffix = key if idx == 0 else f"{key}_{idx//group_size+1}"
                    new_zone = f"{item['zone']}__{suffix}"
                    change = {"old_zone": item["zone"], "new_zone": new_zone, "node_ids": [m["id"] for m in batch]}
                    changes.append(change)
                    generated += 1
                    if apply_changes:
                        self.storage.bulk_update_zone(change["node_ids"], new_zone)
            if apply_changes and generated:
                self.storage.rebuild_zone_index()
        if apply_changes and changes:
            self.reembed_all(batch_size=200)
        return {"task": "split_large_zones", "changes": changes, "applied": apply_changes}
