from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import sphere_cli as sphere_cli_package

from .utils import stable_content_hash


CURRENT_FILE = Path(__file__).resolve()
REPLAY_OBJECT_WEIGHTS = {
    "constraint": 1.35,
    "decision": 1.3,
    "goal": 1.2,
    "relation": 1.15,
    "temporal": 1.1,
    "state": 1.0,
    "preference": 0.95,
    "project": 0.9,
    "artifact": 0.9,
    "fact": 0.8,
}
MAX_REPLAY_CHUNKS = 2
MAX_REPLAY_OBJECTS = 4
_MEMORY_OS_TRACE_ENV = "SPHERE_MEMORY_OS_TRACE_ROOT"
_MEMORY_OS_CONSOLIDATE_ENV = "SPHERE_MEMORY_OS_ENABLE_REPLAY_CONSOLIDATE"


def _load_base_writeback_module() -> Any:
    for package_path in list(getattr(sphere_cli_package, "__path__", [])):
        candidate = Path(package_path) / "writeback.py"
        if not candidate.exists():
            continue
        if candidate.resolve() == CURRENT_FILE:
            continue
        module_name = "sphere_cli._base_writeback_overlay"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError("Unable to locate the base sphere_cli.writeback module for overlay extension.")


def _env_flag(name: str, default: bool = True) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _trace_root() -> Path | None:
    raw = str(os.environ.get(_MEMORY_OS_TRACE_ENV, "")).strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _append_trace_event(filename: str, payload: dict[str, Any]) -> None:
    root = _trace_root()
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    trace_path = root / filename
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


_BASE_WRITEBACK = _load_base_writeback_module()


class MemoryWritebackService(_BASE_WRITEBACK.MemoryWritebackService):
    def writeback_memory(
        self,
        node,
        source_kind: str | None = None,
        source_path: str | None = None,
        replace_node_id: str | None = None,
        skip_edges: bool | None = None,
    ) -> dict[str, Any]:
        report = super().writeback_memory(
            node,
            source_kind=source_kind,
            source_path=source_path,
            replace_node_id=replace_node_id,
            skip_edges=skip_edges,
        )
        replay_representations, consolidate_trace = self._build_replay_representations_for_node(node)
        if replay_representations:
            self.storage.insert_representations(replay_representations)
            self.vector_store.upsert_representations(replay_representations)
            version = self.storage.bump_memory_version()
            self._refresh_profile_snapshots(version)
            report["memory_version"] = version
            report["representation_count"] = int(report.get("representation_count") or 0) + len(replay_representations)
            report["replay_representation_count"] = len(replay_representations)
        else:
            report["replay_representation_count"] = 0
        report["memory_os_trace"] = self._build_memory_os_trace(
            node=node,
            report=report,
            source_kind=source_kind,
            source_path=source_path,
            skip_edges=skip_edges,
            consolidate_trace=consolidate_trace,
        )
        self._emit_memory_os_trace(report["memory_os_trace"])
        return report

    def writeback_batch(
        self,
        nodes,
        skip_edges: bool | None = None,
    ) -> list[dict[str, Any]]:
        reports = super().writeback_batch(nodes, skip_edges=skip_edges)
        replay_representations: list[dict[str, Any]] = []
        replay_counts: dict[str, int] = {}
        node_lookup = {str(node.id): node for node, _, _ in nodes}
        pending_trace_rows: list[dict[str, Any]] = []
        for row in reports:
            node_id = str(row.get("node_id") or "")
            node = node_lookup.get(node_id)
            if node is None:
                row["replay_representation_count"] = 0
                continue
            additions, consolidate_trace = self._build_replay_representations_for_node(node)
            if not additions:
                row["replay_representation_count"] = 0
                row["memory_os_trace"] = self._build_memory_os_trace(
                    node=node,
                    report=row,
                    source_kind=None,
                    source_path=None,
                    skip_edges=skip_edges,
                    consolidate_trace=consolidate_trace,
                )
                self._emit_memory_os_trace(row["memory_os_trace"])
                continue
            replay_representations.extend(additions)
            replay_counts[node_id] = len(additions)
            row["representation_count"] = int(row.get("representation_count") or 0) + len(additions)
            row["replay_representation_count"] = len(additions)
            row["memory_os_trace"] = self._build_memory_os_trace(
                node=node,
                report=row,
                source_kind=None,
                source_path=None,
                skip_edges=skip_edges,
                consolidate_trace=consolidate_trace,
            )
            pending_trace_rows.append(row)
        if replay_representations:
            self.storage.insert_representations(replay_representations)
            self.vector_store.upsert_representations(replay_representations)
            version = self.storage.bump_memory_version()
            self._refresh_profile_snapshots(version)
            for row in reports:
                if replay_counts.get(str(row.get("node_id") or "")):
                    row["memory_version"] = version
                    if isinstance(row.get("memory_os_trace"), dict):
                        row["memory_os_trace"].setdefault("write", {})["memory_version"] = version
        for row in pending_trace_rows:
            trace_payload = row.get("memory_os_trace")
            if isinstance(trace_payload, dict):
                self._emit_memory_os_trace(trace_payload)
        return reports

    def _build_replay_representations_for_node(self, node) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        consolidate_enabled = bool(
            self.writer.config.enable_retrieval_proxy_index
            and _env_flag(_MEMORY_OS_CONSOLIDATE_ENV, default=True)
        )
        consolidate_trace: dict[str, Any] = {
            "consolidate_operator": "replay_proxy_index",
            "consolidate_enabled": consolidate_enabled,
            "operator_off_via_env": not _env_flag(_MEMORY_OS_CONSOLIDATE_ENV, default=True),
            "candidate_chunk_count": 0,
            "selected_chunk_count": 0,
            "selected_parent_chunk_ids": [],
            "generated_proxy_kinds": [],
            "generated_representation_count": 0,
        }
        if not consolidate_enabled:
            return [], consolidate_trace
        node_id = str(node.id or "")
        if not node_id:
            return [], consolidate_trace
        chunks = self.storage.fetch_chunks_for_node(node_id)
        if not chunks:
            return [], consolidate_trace
        objects = self.storage.fetch_objects_for_nodes([node_id])
        object_index: dict[str, list[dict[str, Any]]] = {}
        for obj in objects:
            source_chunk_id = str(obj.get("source_chunk_id") or "")
            if source_chunk_id:
                object_index.setdefault(source_chunk_id, []).append(dict(obj))
        candidates: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
        for chunk in chunks:
            if str(chunk.get("grain") or "") == "macro":
                continue
            chunk_id = str(chunk.get("chunk_id") or "")
            related_objects = list(object_index.get(chunk_id, []))
            if not related_objects:
                continue
            score = self._score_replay_candidate(chunk, related_objects)
            if score < 2.35:
                continue
            candidates.append((score, dict(chunk), related_objects))
        consolidate_trace["candidate_chunk_count"] = len(candidates)
        if not candidates:
            return [], consolidate_trace
        candidates.sort(key=lambda item: item[0], reverse=True)
        replay_representations: list[dict[str, Any]] = []
        generated_proxy_kinds: set[str] = set()
        selected_parent_chunk_ids: list[str] = []
        for score, chunk, related_objects in candidates[:MAX_REPLAY_CHUNKS]:
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id:
                selected_parent_chunk_ids.append(chunk_id)
            time_bucket = str(chunk.get("time_bucket") or getattr(node, "time_bucket", "") or "")
            entity_tags = str(chunk.get("entity_tags") or getattr(node, "entity_tags", "") or "")
            task_type_tag = str(chunk.get("task_type_tag") or getattr(node, "task_type_tag", "") or "")
            created_at = str(chunk.get("created_at") or getattr(node, "created_at", "") or "")
            replay_payloads = {
                "summary": self._build_replay_summary(chunk, related_objects),
                "structured": self._build_replay_structured(chunk, related_objects),
                "signature": self._build_replay_signature(chunk, related_objects, score),
            }
            for proxy_kind, text in replay_payloads.items():
                if not text:
                    continue
                generated_proxy_kinds.add(proxy_kind)
                representation_seed = f"{chunk_id}|replay|{proxy_kind}|{text}"
                replay_representations.append(
                    {
                        "representation_id": f"repr_{stable_content_hash(representation_seed)[:16]}",
                        "parent_id": chunk_id,
                        "parent_type": "chunk",
                        "proxy_kind": proxy_kind,
                        "text": text,
                        "content_hash": stable_content_hash(text),
                        "scope": str(chunk.get("scope") or getattr(node, "scope", "") or ""),
                        "workspace": str(chunk.get("workspace") or getattr(node, "workspace", "") or ""),
                        "project": str(chunk.get("project") or getattr(node, "project", "") or ""),
                        "session_id": str(chunk.get("session_id") or getattr(node, "session_id", "") or ""),
                        "time_bucket": time_bucket,
                        "entity_tags": entity_tags,
                        "task_type_tag": task_type_tag,
                        "created_at": created_at,
                    }
                )
        consolidate_trace["selected_chunk_count"] = len(selected_parent_chunk_ids)
        consolidate_trace["selected_parent_chunk_ids"] = selected_parent_chunk_ids
        consolidate_trace["generated_proxy_kinds"] = sorted(generated_proxy_kinds)
        consolidate_trace["generated_representation_count"] = len(replay_representations)
        return replay_representations, consolidate_trace

    def _build_memory_os_trace(
        self,
        *,
        node,
        report: dict[str, Any],
        source_kind: str | None,
        source_path: str | None,
        skip_edges: bool,
        consolidate_trace: dict[str, Any],
    ) -> dict[str, Any]:
        entity_tags = str(getattr(node, "entity_tags", "") or "")
        task_type_tag = str(getattr(node, "task_type_tag", "") or "")
        time_bucket = str(getattr(node, "time_bucket", "") or "")
        typed_state_labels = ["episode"]
        if entity_tags:
            typed_state_labels.append("entity")
        if task_type_tag:
            typed_state_labels.append("task")
        if time_bucket or str(source_kind or "") in {"conversation_turn", "halumem_dialogue", "preference"}:
            typed_state_labels.append("constraint")
        if int(report.get("replay_representation_count") or 0) > 0:
            typed_state_labels.append("belief")
        typed_state_labels = list(dict.fromkeys(typed_state_labels))
        return {
            "operator_family": "memory_os",
            "operator_stage": "writeback",
            "node_id": str(report.get("node_id") or getattr(node, "id", "") or ""),
            "source_kind": source_kind or str(getattr(node, "source_type", "") or ""),
            "source_path": source_path or "",
            "skip_edges": bool(skip_edges),
            "typed_state_labels": typed_state_labels,
            "bind_signals": {
                "entity_tags": entity_tags,
                "task_type_tag": task_type_tag,
                "time_bucket": time_bucket,
            },
            "write": {
                "memory_version": report.get("memory_version"),
                "representation_count": int(report.get("representation_count") or 0),
                "edge_count": int(report.get("edge_count") or 0),
            },
            "consolidate": consolidate_trace,
        }

    def _emit_memory_os_trace(self, trace_payload: dict[str, Any]) -> None:
        _append_trace_event("writeback_operator_trace.jsonl", trace_payload)

    def _score_replay_candidate(self, chunk: dict[str, Any], related_objects: list[dict[str, Any]]) -> float:
        text = " ".join(str(chunk.get("text") or "").split())
        if not text:
            return 0.0
        token_count = len(text.split())
        normalized_tokens = [token.strip(".,!?;:()[]{}\"'").lower() for token in text.split() if token.strip(".,!?;:()[]{}\"'")]
        unique_ratio = len(set(normalized_tokens)) / max(1, len(normalized_tokens))
        type_bonus = sum(REPLAY_OBJECT_WEIGHTS.get(str(obj.get("object_type") or ""), 0.45) for obj in related_objects)
        unique_types = len({str(obj.get("object_type") or "") for obj in related_objects if obj.get("object_type")})
        weak_anchor_bonus = 0.8 if token_count <= 26 else 0.4 if token_count <= 40 else 0.0
        if unique_ratio < 0.72:
            weak_anchor_bonus += 0.4
        if str(chunk.get("grain") or "") == "micro":
            weak_anchor_bonus += 0.25
        return type_bonus + min(1.2, 0.22 * len(related_objects)) + 0.18 * unique_types + weak_anchor_bonus

    def _build_replay_summary(self, chunk: dict[str, Any], related_objects: list[dict[str, Any]]) -> str:
        base = str(chunk.get("retrieval_summary") or chunk.get("structured_summary") or chunk.get("text") or "").strip()
        if not base:
            return ""
        focus_bits = self._top_object_bits(related_objects, limit=2)
        if focus_bits:
            return f"Replay focus: {base} Key memory cues: {'; '.join(focus_bits)}"[:360]
        return f"Replay focus: {base}"[:320]

    def _build_replay_structured(self, chunk: dict[str, Any], related_objects: list[dict[str, Any]]) -> str:
        focus_bits = self._top_object_bits(related_objects, limit=MAX_REPLAY_OBJECTS)
        if not focus_bits:
            return ""
        return f"Replay structured cues: {'; '.join(focus_bits)}"[:360]

    def _build_replay_signature(
        self,
        chunk: dict[str, Any],
        related_objects: list[dict[str, Any]],
        score: float,
    ) -> str:
        object_types = sorted({str(obj.get("object_type") or "") for obj in related_objects if obj.get("object_type")})
        entity_tags = str(chunk.get("entity_tags") or "")
        time_bucket = str(chunk.get("time_bucket") or "")
        parts = [f"replay_score={score:.2f}"]
        if object_types:
            parts.append(f"types={','.join(object_types[:4])}")
        if entity_tags:
            parts.append(f"entities={entity_tags}")
        if time_bucket:
            parts.append(f"time={time_bucket}")
        return f"Replay signature: {' | '.join(parts)}"[:280]

    def _top_object_bits(self, related_objects: list[dict[str, Any]], limit: int) -> list[str]:
        ranked = sorted(
            related_objects,
            key=lambda obj: (
                REPLAY_OBJECT_WEIGHTS.get(str(obj.get("object_type") or ""), 0.45),
                float(obj.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        seen: set[str] = set()
        focus_bits: list[str] = []
        for obj in ranked:
            object_type = str(obj.get("object_type") or "").strip()
            object_text = str(
                obj.get("entity")
                or obj.get("canonical_value")
                or obj.get("canonical_key")
                or obj.get("object_text")
                or ""
            ).strip()
            if not object_type or not object_text:
                continue
            signature = f"{object_type}:{object_text.lower()}"
            if signature in seen:
                continue
            seen.add(signature)
            focus_bits.append(f"{object_type}={object_text}")
            if len(focus_bits) >= limit:
                break
        return focus_bits


__all__ = ["MemoryWritebackService"]
