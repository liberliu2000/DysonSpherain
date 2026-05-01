from __future__ import annotations

import unittest

from dysonspherain.memory_runtime.events import build_event
from dysonspherain.memory_runtime.situation_graph import rebuild_situation_graph, update_situation_graph


class SituationGraphTests(unittest.TestCase):
    def test_situation_graph_replay_determinism(self) -> None:
        events = [
            build_event(event_type="user_instruction_received", payload={"content": "Fix regression"}, source="test", actor="user", timestamp="2026-04-30T00:00:00+00:00"),
            build_event(event_type="constraint_added", payload={"content": "Do not break benchmark quality"}, source="test", actor="user", timestamp="2026-04-30T00:01:00+00:00"),
            build_event(event_type="decision_made", payload={"summary": "Use event-sourced runtime"}, source="test", actor="assistant", timestamp="2026-04-30T00:02:00+00:00"),
        ]
        first = rebuild_situation_graph(events).to_dict()
        second = rebuild_situation_graph(list(reversed(events))).to_dict()
        self.assertEqual(first, second)
        self.assertTrue(any(node["node_type"] == "Constraint" for node in first["nodes"]))

    def test_graph_uses_explicit_relation_edges_and_mutations(self) -> None:
        source = build_event(event_type="user_instruction_received", payload={"content": "Root task", "node_id": "task-a"}, source="test", actor="user", timestamp="2026-04-30T00:00:00+00:00")
        patch = build_event(event_type="patch_applied", payload={"summary": "Fix root task", "fixed_by": "patch-node", "depends_on": "task_task-a"}, source="test", actor="assistant", timestamp="2026-04-30T00:01:00+00:00")
        graph = rebuild_situation_graph([source, patch])
        edge_types = {edge["edge_type"] for edge in graph.to_dict()["edges"]}
        self.assertIn("depends_on", edge_types)
        mutations = update_situation_graph(patch, graph)
        self.assertTrue(any(item.mutation_type.startswith("node_") for item in mutations))


if __name__ == "__main__":
    unittest.main()
