from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .memory_manager import SphereMemoryManager
from .memory_writer import MemoryWriter
from .models import ArtifactRecord, MemoryNode, now_iso
from .storage import Storage
from .utils import exact_content_hash, lexical_score, normalize_text_for_hash, stable_content_hash
from .vector_store import VectorStore
from .workspace import compose_scope


class MemoryWritebackService:
    def __init__(
        self,
        storage: Storage,
        vector_store: VectorStore,
        manager: SphereMemoryManager,
        writer: MemoryWriter,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.manager = manager
        self.writer = writer

    def writeback_memory(
        self,
        node: MemoryNode,
        source_kind: str | None = None,
        source_path: str | None = None,
        replace_node_id: str | None = None,
        skip_edges: bool | None = None,
    ) -> dict[str, Any]:
        timestamp = now_iso()
        should_write_edges = self.writer.config.enable_lightweight_edge_writeback if skip_edges is None else not skip_edges
        self._apply_node_defaults(node, source_kind=source_kind, source_path=source_path, timestamp=timestamp)
        source_text = (node.raw_content or "").strip() or node.summary
        node.content_hash = exact_content_hash(source_text)
        node.normalized_hash = stable_content_hash(source_text)
        node.last_seen_at = timestamp

        if replace_node_id:
            self._delete_existing_node_payload(replace_node_id)

        draft_chunks = self.writer.prepare_chunks(node, source_kind=source_kind, source_path=source_path)
        draft_objects = self.writer.extract_objects(node, draft_chunks) if self.writer.config.enable_structured_compression else []
        dedup = self._classify_write(node, draft_objects)
        if dedup["action"] == "exact_duplicate":
            existing_id = str(dedup["existing_node_id"])
            self.storage.update_node_seen(existing_id, timestamp, salience_boost=0.01)
            version = self.storage.bump_memory_version()
            self._refresh_profile_snapshots(version)
            return {
                "node_id": existing_id,
                "chunk_count": 0,
                "object_count": 0,
                "representation_count": 0,
                "neighbor_count": 0,
                "edge_count": 0,
                "dedup_action": "exact_duplicate",
                "memory_version": version,
            }
        if dedup["action"] == "near_duplicate_without_new_info":
            existing_id = str(dedup["existing_node_id"])
            self.storage.update_node_seen(existing_id, timestamp, salience_boost=0.02)
            version = self.storage.bump_memory_version()
            self._refresh_profile_snapshots(version)
            return {
                "node_id": existing_id,
                "chunk_count": 0,
                "object_count": 0,
                "representation_count": 0,
                "neighbor_count": 0,
                "edge_count": 0,
                "dedup_action": "near_duplicate_without_new_info",
                "memory_version": version,
            }

        effective_objects = draft_objects
        delta_payload: dict[str, Any] | None = None
        if dedup["action"] == "delta_write":
            delta_payload = self._apply_delta_rewrite(node, dedup, timestamp)
            draft_chunks = self.writer.prepare_chunks(node, source_kind=source_kind, source_path=source_path)
            effective_objects = list(dedup.get("delta_objects") or draft_objects)

        self._auto_register_artifacts(node, effective_objects, source_path=source_path, timestamp=timestamp)
        representations = self.writer.build_representations(node, draft_chunks, effective_objects)

        self.manager.add_node(node)
        self.storage.insert_chunks(draft_chunks)
        neighbors = self.writer.build_chunk_neighbors(draft_chunks)
        self.storage.insert_chunk_neighbors(neighbors)
        self.vector_store.upsert_chunks(draft_chunks)

        self.storage.insert_objects(effective_objects)
        self.vector_store.upsert_objects(effective_objects)

        self.storage.insert_representations(representations)
        if self.writer.config.enable_retrieval_proxy_index:
            self.vector_store.upsert_representations(representations)

        if delta_payload is not None:
            self.storage.insert_memory_delta(delta_payload)

        edge_count = 0
        if should_write_edges:
            edges = self.writer.create_edges_for_new_node(node)
            for edge in edges:
                self.storage.insert_edge(asdict(edge))
            edge_count = len(edges)

        version = self.storage.bump_memory_version()
        self._refresh_profile_snapshots(version)
        return {
            "node_id": node.id,
            "chunk_count": len(draft_chunks),
            "object_count": len(effective_objects),
            "representation_count": len(representations),
            "neighbor_count": len(neighbors),
            "edge_count": edge_count,
            "dedup_action": dedup["action"],
            "memory_version": version,
        }

    def writeback_batch(
        self,
        nodes: Sequence[tuple[MemoryNode, str | None, str | None]],
        skip_edges: bool | None = None,
    ) -> list[dict[str, Any]]:
        should_write_edges = self.writer.config.enable_lightweight_edge_writeback if skip_edges is None else not skip_edges
        node_dicts: list[dict[str, Any]] = []
        all_chunks: list[dict[str, Any]] = []
        all_edges: list[dict[str, Any]] = []
        all_objects: list[dict[str, Any]] = []
        all_neighbors: list[dict[str, Any]] = []
        all_representations: list[dict[str, Any]] = []
        per_node: list[dict[str, Any]] = []

        for node, source_kind, source_path in nodes:
            source_text = (node.raw_content or "").strip() or node.summary
            self._apply_node_defaults(node, source_kind=source_kind, source_path=source_path, timestamp=node.created_at)
            node.content_hash = exact_content_hash(source_text)
            node.normalized_hash = stable_content_hash(source_text)
            node.last_seen_at = node.created_at
            self.manager.assign_coordinates(node)

            chunks = self.writer.prepare_chunks(node, source_kind=source_kind, source_path=source_path)
            objects = self.writer.extract_objects(node, chunks)
            self._auto_register_artifacts(node, objects, source_path=source_path, timestamp=node.created_at)
            representations = self.writer.build_representations(node, chunks, objects)
            neighbors = self.writer.build_chunk_neighbors(chunks)

            node_dicts.append(node.to_dict())
            all_chunks.extend(chunks)
            all_neighbors.extend(neighbors)
            all_objects.extend(objects)
            all_representations.extend(representations)
            edges = self.writer.create_edges_for_new_node(node) if should_write_edges else []
            all_edges.extend(asdict(edge) for edge in edges)
            per_node.append(
                {
                    "node_id": node.id,
                    "chunk_count": len(chunks),
                    "object_count": len(objects),
                    "representation_count": len(representations),
                    "neighbor_count": len(neighbors),
                    "edge_count": len(edges),
                    "dedup_action": "batch_insert",
                }
            )

        self.storage.insert_nodes(node_dicts)
        for node in [item[0] for item in nodes]:
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
        self.storage.insert_chunks(all_chunks)
        self.storage.insert_chunk_neighbors(all_neighbors)
        self.storage.insert_objects(all_objects)
        self.storage.insert_representations(all_representations)
        for edge in all_edges:
            self.storage.insert_edge(edge)
        self.vector_store.upsert_chunks(all_chunks)
        self.vector_store.upsert_objects(all_objects)
        if self.writer.config.enable_retrieval_proxy_index:
            self.vector_store.upsert_representations(all_representations)

        version = self.storage.bump_memory_version()
        self._refresh_profile_snapshots(version)
        for row in per_node:
            row["memory_version"] = version
        return per_node

    def _delete_existing_node_payload(self, replace_node_id: str) -> None:
        old_chunk_ids = self.storage.delete_chunks_for_node(replace_node_id)
        old_object_ids = self.storage.delete_objects_for_node(replace_node_id)
        old_representation_ids = self.storage.delete_representations_for_parent_ids([replace_node_id] + old_chunk_ids)
        self.storage.delete_chunk_neighbors(old_chunk_ids)
        self.storage.delete_memory_deltas_for_node(replace_node_id)
        self.vector_store.delete_chunks(old_chunk_ids)
        self.vector_store.delete_objects(old_object_ids)
        self.vector_store.delete_representations(old_representation_ids)
        self.storage.delete_node(replace_node_id)

    def _classify_write(self, node: MemoryNode, new_objects: list[dict[str, Any]]) -> dict[str, Any]:
        if self.writer.config.enable_ingest_compression and self.writer.config.enable_content_hash_dedup:
            exact = [
                row
                for row in self.storage.fetch_nodes_by_normalized_hash(str(node.normalized_hash or ""))
                if self._same_scope(node, row)
            ]
            if exact:
                return {"action": "exact_duplicate", "existing_node_id": exact[0]["id"]}

        if not self.writer.config.enable_ingest_compression:
            return {"action": "substantive_new_info"}

        recent_nodes = [
            row
            for row in self.storage.fetch_recent_nodes(limit=20, sector=node.sector, zone=node.zone)
            if self._same_scope(node, row)
        ]
        best_row: dict[str, Any] | None = None
        best_score = 0.0
        new_raw_text = normalize_text_for_hash((node.raw_content or "").strip() or node.summary)
        new_blended_text = normalize_text_for_hash(
            "\n".join(part for part in ((node.raw_content or "").strip(), node.summary.strip()) if part)
        )
        for candidate in recent_nodes:
            candidate_blended_text = normalize_text_for_hash(
                ((candidate.get("raw_content") or "") + "\n" + (candidate.get("summary") or "")).strip()
            )
            candidate_raw_text = normalize_text_for_hash((candidate.get("raw_content") or "").strip() or candidate.get("summary") or "")
            score = max(
                lexical_score(new_raw_text, candidate_raw_text),
                lexical_score(new_raw_text, candidate_blended_text),
                lexical_score(new_blended_text, candidate_blended_text),
            )
            if score > best_score:
                best_row = candidate
                best_score = score
        if best_row is None or best_score < 0.84:
            return {"action": "substantive_new_info"}

        base_objects = self.storage.fetch_objects_for_nodes([str(best_row["id"])])
        delta_info = self._diff_objects(new_objects, base_objects)
        if best_score >= 0.92 and not delta_info["changed_fields"]:
            return {
                "action": "near_duplicate_without_new_info",
                "existing_node_id": best_row["id"],
                "similarity": round(best_score, 4),
            }
        if self.writer.config.enable_delta_memory_writer and delta_info["changed_fields"]:
            return {
                "action": "delta_write",
                "existing_node_id": best_row["id"],
                "base_node": best_row,
                "similarity": round(best_score, 4),
                "changed_fields": delta_info["changed_fields"],
                "delta_objects": delta_info["delta_objects"],
            }
        return {"action": "substantive_new_info"}

    def _apply_node_defaults(
        self,
        node: MemoryNode,
        *,
        source_kind: str | None,
        source_path: str | None,
        timestamp: str,
    ) -> None:
        config = self.writer.config
        node.workspace = node.workspace or config.workspace_name
        node.project = node.project or config.project_name
        node.session_id = node.session_id or config.session_id
        node.scope = compose_scope(scope=node.scope, project=node.project, session_id=node.session_id)
        if source_kind and str(node.source_type or "").strip() in {"", "memory_note"}:
            node.source_type = source_kind
        else:
            node.source_type = node.source_type or source_kind or node.molecular_type or "memory_note"
        node.source_ref = node.source_ref or source_path or node.content_ref
        node.extraction_method = node.extraction_method or "manual"
        node.verification_status = node.verification_status or "unverified"
        node.updated_at = timestamp

    def _auto_register_artifacts(
        self,
        node: MemoryNode,
        objects: list[dict[str, Any]],
        *,
        source_path: str | None,
        timestamp: str,
    ) -> None:
        if not self.writer.config.enable_artifact_registry or not self.writer.config.enable_note_artifact_auto_register:
            return
        goal_ids = [str(item.get("object_id") or "") for item in objects if str(item.get("object_type") or "") == "goal" and item.get("object_id")]
        source_anchor = source_path or node.source_ref or node.content_ref
        for item in objects:
            if str(item.get("object_type") or "") != "artifact":
                continue
            resolved_path = self._resolve_existing_artifact_path(str(item.get("source_ref") or ""), source_anchor)
            if resolved_path is None:
                continue
            artifact_path = str(resolved_path)
            existing = self.storage.fetch_artifact_by_path(artifact_path)
            artifact_id = str((existing or {}).get("artifact_id") or f"art_{stable_content_hash(artifact_path)[:12]}")
            related_memory_ids = self._merge_json_list((existing or {}).get("related_memory_ids_json"), [node.id])
            related_object_ids = self._merge_json_list((existing or {}).get("related_object_ids_json"), [str(item.get("object_id") or "")])
            related_goal_ids = self._merge_json_list((existing or {}).get("related_goal_ids_json"), goal_ids)
            metadata = self._merge_json_object(
                (existing or {}).get("metadata_json"),
                {
                    "auto_registered": True,
                    "resolved_path": artifact_path,
                    "raw_source_ref": str(item.get("source_ref") or ""),
                },
            )
            record = ArtifactRecord(
                artifact_id=artifact_id,
                path=artifact_path,
                artifact_type=str(item.get("new_value") or item.get("attribute") or "file"),
                scope=str(item.get("scope") or node.scope or "global"),
                workspace=str(item.get("workspace") or node.workspace or "") or None,
                project=str(item.get("project") or node.project or "") or None,
                session_id=str(item.get("session_id") or node.session_id or "") or None,
                title=self._artifact_title_from_object(item, artifact_path),
                summary=str((existing or {}).get("summary") or item.get("source_unit_text") or item.get("object_text") or "").strip()[:280] or None,
                tags_json=(existing or {}).get("tags_json") or json.dumps([], ensure_ascii=False),
                source_type=str((existing or {}).get("source_type") or "memory_note_auto"),
                source_ref=artifact_path,
                related_memory_ids_json=json.dumps(related_memory_ids, ensure_ascii=False),
                related_object_ids_json=json.dumps(related_object_ids, ensure_ascii=False),
                related_goal_ids_json=json.dumps(related_goal_ids, ensure_ascii=False),
                metadata_json=json.dumps(metadata, ensure_ascii=False),
                created_at=str((existing or {}).get("created_at") or timestamp),
                updated_at=timestamp,
            )
            self.storage.insert_artifact(record.to_dict())
            item["related_artifact_ids_json"] = json.dumps([artifact_id], ensure_ascii=False)
            item["metadata_json"] = json.dumps(
                self._merge_json_object(
                    item.get("metadata_json"),
                    {"artifact_id": artifact_id, "resolved_path": artifact_path, "auto_registered": True},
                ),
                ensure_ascii=False,
            )

    def _resolve_existing_artifact_path(self, raw_path: str, source_anchor: str | None) -> Path | None:
        cleaned = raw_path.strip()
        if not cleaned:
            return None
        candidate = Path(cleaned).expanduser()
        candidates: list[Path] = []
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            if source_anchor:
                anchor_path = Path(source_anchor).expanduser()
                anchor_base = anchor_path if anchor_path.is_dir() else anchor_path.parent
                candidates.append(anchor_base / candidate)
            candidates.append(self.writer.config.base_dir / candidate)
        for option in candidates:
            try:
                resolved = option.resolve()
            except OSError:
                continue
            if resolved.exists() and resolved.is_file():
                return resolved
        return None

    @staticmethod
    def _merge_json_list(raw_value: Any, new_items: list[str]) -> list[str]:
        items: list[str] = []
        if isinstance(raw_value, str) and raw_value.strip():
            try:
                loaded = json.loads(raw_value)
            except json.JSONDecodeError:
                loaded = []
            if isinstance(loaded, list):
                items.extend(str(item) for item in loaded if str(item or "").strip())
        for item in new_items:
            if item and item not in items:
                items.append(item)
        return items

    @staticmethod
    def _merge_json_object(raw_value: Any, updates: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if isinstance(raw_value, str) and raw_value.strip():
            try:
                loaded = json.loads(raw_value)
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                payload.update(loaded)
        for key, value in updates.items():
            if value is not None and value != "":
                payload[key] = value
        return payload

    @staticmethod
    def _artifact_title_from_object(item: dict[str, Any], artifact_path: str) -> str:
        title = str(item.get("entity") or "").strip()
        if title:
            return title
        return Path(artifact_path).name

    @staticmethod
    def _same_scope(node: MemoryNode, row: dict[str, Any]) -> bool:
        row_scope = str(row.get("scope") or "global")
        row_workspace = str(row.get("workspace") or "") or None
        row_project = str(row.get("project") or "") or None
        row_session = str(row.get("session_id") or "") or None
        return (
            row_scope == str(node.scope or "global")
            and row_workspace == (node.workspace or None)
            and row_project == (node.project or None)
            and row_session == (node.session_id or None)
        )

    def _diff_objects(self, new_objects: list[dict[str, Any]], base_objects: list[dict[str, Any]]) -> dict[str, Any]:
        base_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for item in base_objects:
            key = (
                str(item.get("object_type") or ""),
                str(item.get("canonical_key") or item.get("entity") or item.get("object_text") or "").strip().lower(),
            )
            if key[1]:
                base_by_key[key] = dict(item)

        changed_fields: list[dict[str, Any]] = []
        delta_objects: list[dict[str, Any]] = []
        for item in new_objects:
            key = (
                str(item.get("object_type") or ""),
                str(item.get("canonical_key") or item.get("entity") or item.get("object_text") or "").strip().lower(),
            )
            previous = base_by_key.get(key)
            if previous is None:
                changed_fields.append(
                    {
                        "object_type": item.get("object_type"),
                        "canonical_key": key[1],
                        "change": "new_object",
                        "new_value": item.get("new_value") or item.get("object_text"),
                    }
                )
                delta_objects.append(dict(item))
                continue
            old_value = previous.get("new_value") or previous.get("old_value") or previous.get("object_text")
            new_value = item.get("new_value") or item.get("old_value") or item.get("object_text")
            old_polarity = previous.get("polarity")
            new_polarity = item.get("polarity")
            if str(old_value) != str(new_value) or str(old_polarity) != str(new_polarity):
                changed_fields.append(
                    {
                        "object_type": item.get("object_type"),
                        "canonical_key": key[1],
                        "change": "value_changed",
                        "old_value": old_value,
                        "new_value": new_value,
                        "old_polarity": old_polarity,
                        "new_polarity": new_polarity,
                    }
                )
                delta_objects.append(dict(item))
        return {"changed_fields": changed_fields, "delta_objects": delta_objects}

    def _apply_delta_rewrite(self, node: MemoryNode, dedup: dict[str, Any], timestamp: str) -> dict[str, Any]:
        changed_fields = list(dedup.get("changed_fields") or [])
        base_node = dict(dedup.get("base_node") or {})
        delta_summary = self._summarize_delta(base_node, changed_fields)
        node.base_node_id = str(dedup.get("existing_node_id") or "")
        node.delta_summary = delta_summary
        node.changed_fields_json = json.dumps(changed_fields, ensure_ascii=False)
        node.raw_content = delta_summary
        node.summary = delta_summary[:220]
        node.compression_level = "high"
        return {
            "delta_id": f"delta_{stable_content_hash(node.id + node.base_node_id + delta_summary)[:16]}",
            "node_id": node.id,
            "base_node_id": node.base_node_id,
            "object_type": ",".join(sorted({str(item.get('object_type') or '') for item in changed_fields if item.get('object_type')})),
            "changed_fields_json": node.changed_fields_json,
            "delta_summary": delta_summary,
            "effective_time": timestamp,
            "merge_policy": "base_plus_delta",
            "created_at": timestamp,
        }

    def _summarize_delta(self, base_node: dict[str, Any], changed_fields: list[dict[str, Any]]) -> str:
        if not changed_fields:
            return str(base_node.get("summary") or "small update")
        clauses: list[str] = []
        for item in changed_fields[:4]:
            object_type = str(item.get("object_type") or "field")
            canonical_key = str(item.get("canonical_key") or "unknown")
            if item.get("change") == "new_object":
                clauses.append(f"{object_type} {canonical_key} added as {item.get('new_value')}")
            else:
                clauses.append(f"{object_type} {canonical_key} changed from {item.get('old_value')} to {item.get('new_value')}")
        base_summary = str(base_node.get("summary") or "").strip()
        if base_summary:
            return f"Delta from {base_summary}: " + "; ".join(clauses)
        return "Delta update: " + "; ".join(clauses)

    def _refresh_profile_snapshots(self, memory_version: int) -> None:
        if not self.writer.config.enable_profile_snapshot_cache:
            return
        relevant = self.storage.fetch_objects(
            "object_type IN (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("preference", "state_update", "temporal_reference", "personal_context", "persona", "constraint", "relation", "fact", "event"),
        )
        preference_rows = [row for row in relevant if row.get("object_type") == "preference"]
        state_rows = [row for row in relevant if row.get("object_type") in {"state_update", "temporal_reference"}]
        persona_rows = [row for row in relevant if row.get("object_type") in {"persona", "personal_context", "constraint"}]
        relation_rows = [row for row in relevant if row.get("object_type") == "relation"]
        preferences = self._decorate_snapshot_rows(self._latest_by_key(preference_rows), snapshot_state="current")
        states = self._decorate_snapshot_rows(self._latest_by_key(state_rows), snapshot_state="current")
        persona = self._decorate_snapshot_rows(self._latest_by_key(persona_rows), snapshot_state="current")
        relations = self._decorate_snapshot_rows(self._latest_by_key(relation_rows), snapshot_state="current")
        preference_history = self._history_by_key(preference_rows, limit=3)
        state_history = self._history_by_key(state_rows, limit=3)
        persona_history = self._history_by_key(persona_rows, limit=3)
        relation_history = self._history_by_key(relation_rows, limit=3)
        recent_deltas = self.storage.fetch_memory_deltas()[:12]

        self.storage.upsert_profile_snapshot(
            snapshot_type="preference",
            payload_json=json.dumps({"items": preferences, "history": preference_history}, ensure_ascii=False),
            memory_version=memory_version,
            source_object_ids=[str(item.get("object_id") or "") for item in preferences],
            updated_at=now_iso(),
        )
        self.storage.upsert_profile_snapshot(
            snapshot_type="state",
            payload_json=json.dumps({"items": states, "history": state_history}, ensure_ascii=False),
            memory_version=memory_version,
            source_object_ids=[str(item.get("object_id") or "") for item in states],
            updated_at=now_iso(),
        )
        self.storage.upsert_profile_snapshot(
            snapshot_type="persona",
            payload_json=json.dumps({"items": persona, "history": persona_history}, ensure_ascii=False),
            memory_version=memory_version,
            source_object_ids=[str(item.get("object_id") or "") for item in persona],
            updated_at=now_iso(),
        )
        self.storage.upsert_profile_snapshot(
            snapshot_type="relation",
            payload_json=json.dumps({"items": relations, "history": relation_history}, ensure_ascii=False),
            memory_version=memory_version,
            source_object_ids=[str(item.get("object_id") or "") for item in relations],
            updated_at=now_iso(),
        )
        self.storage.upsert_profile_snapshot(
            snapshot_type="profile",
            payload_json=json.dumps(
                {
                    "preferences": preferences,
                    "preference_history": preference_history,
                    "state": states,
                    "state_history": state_history,
                    "persona": persona,
                    "persona_history": persona_history,
                    "relations": relations,
                    "relation_history": relation_history,
                    "recent_deltas": recent_deltas,
                },
                ensure_ascii=False,
            ),
            memory_version=memory_version,
            source_object_ids=[
                *(str(item.get("object_id") or "") for item in preferences),
                *(str(item.get("object_id") or "") for item in states),
                *(str(item.get("object_id") or "") for item in persona),
                *(str(item.get("object_id") or "") for item in relations),
            ],
            updated_at=now_iso(),
        )

    @staticmethod
    def _latest_by_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row.get("snapshot_key") or row.get("canonical_key") or row.get("entity") or row.get("object_id") or "")
            if not key:
                continue
            current = latest.get(key)
            row_timestamp = str(row.get("timestamp") or row.get("created_at") or "")
            current_timestamp = str((current or {}).get("timestamp") or (current or {}).get("created_at") or "")
            if current is None or row_timestamp >= current_timestamp:
                latest[key] = dict(row)
        ordered = list(latest.values())
        ordered.sort(key=lambda item: str(item.get("timestamp") or item.get("created_at") or ""), reverse=True)
        return ordered

    @staticmethod
    def _decorate_snapshot_rows(rows: list[dict[str, Any]], *, snapshot_state: str) -> list[dict[str, Any]]:
        return [
            {
                **dict(row),
                "snapshot_state": snapshot_state,
                "effective_time": row.get("timestamp") or row.get("created_at"),
                "valid_time": row.get("timestamp") or row.get("created_at"),
            }
            for row in rows
        ]

    def _history_by_key(self, rows: list[dict[str, Any]], limit: int = 3) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in sorted(rows, key=lambda item: str(item.get("timestamp") or item.get("created_at") or ""), reverse=True):
            key = str(row.get("snapshot_key") or row.get("canonical_key") or row.get("entity") or row.get("object_id") or "")
            if not key:
                continue
            items = grouped.setdefault(key, [])
            if len(items) >= max(1, limit):
                continue
            items.append(
                {
                    **dict(row),
                    "snapshot_state": "history",
                    "effective_time": row.get("timestamp") or row.get("created_at"),
                    "valid_time": row.get("timestamp") or row.get("created_at"),
                }
            )
        return grouped
