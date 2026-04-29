from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config import DEFAULT_SHELL_POLICY
from .models import MemoryNode
from .storage import Storage
from .utils import deterministic_angle


class SphereMemoryManager:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def assign_coordinates(self, node: MemoryNode) -> None:
        if node.theta is None or node.phi is None:
            theta, phi = deterministic_angle(f"{node.shell}|{node.sector}|{node.zone}|{node.cell}|{node.summary}")
            node.theta = theta
            node.phi = phi

    def add_node(self, node: MemoryNode) -> MemoryNode:
        self.assign_coordinates(node)
        self.storage.insert_node(node.to_dict())
        zone_id = f"{node.shell}:{node.sector}:{node.zone}"
        self.storage.upsert_zone_index(
            zone_id=zone_id,
            shell=node.shell,
            sector=node.sector,
            zone=node.zone,
            zone_summary=node.summary[:160],
            centroid_theta=node.theta,
            centroid_phi=node.phi,
        )
        return node

    def promote_node(self, node_id: str) -> None:
        self.storage.update_node_stage(node_id, "long_term")

    def demote_node(self, node_id: str) -> None:
        self.storage.update_node_stage(node_id, "archive")

    def shell_stats(self) -> dict[int, int]:
        rows = self.storage.fetch_nodes()
        counts: dict[int, int] = defaultdict(int)
        for row in rows:
            counts[row["shell"]] += 1
        return counts

    def capacity_report(self) -> list[dict[str, Any]]:
        counts = self.shell_stats()
        report = []
        for shell, cfg in DEFAULT_SHELL_POLICY.items():
            count = counts.get(shell, 0)
            report.append(
                {
                    "shell": shell,
                    "name": cfg["name"],
                    "count": count,
                    "max_items_hint": cfg["max_items_hint"],
                    "utilization": round(count / cfg["max_items_hint"], 4),
                }
            )
        return report
