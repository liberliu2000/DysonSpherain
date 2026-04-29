from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import sphere_cli as sphere_cli_package


CURRENT_FILE = Path(__file__).resolve()
_MEMORY_OS_TRACE_ENV = "SPHERE_MEMORY_OS_TRACE_ROOT"
_MEMORY_OS_CONSOLIDATE_ENV = "SPHERE_MEMORY_OS_ENABLE_REPLAY_CONSOLIDATE"


def _load_base_memory_writer_module() -> Any:
    for package_path in list(getattr(sphere_cli_package, "__path__", [])):
        candidate = Path(package_path) / "memory_writer.py"
        if not candidate.exists():
            continue
        if candidate.resolve() == CURRENT_FILE:
            continue
        module_name = "sphere_cli._base_memory_writer_overlay"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError("Unable to locate the base sphere_cli.memory_writer module for overlay extension.")


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


_BASE_MEMORY_WRITER = _load_base_memory_writer_module()


class MemoryWriter(_BASE_MEMORY_WRITER.MemoryWriter):
    def __init__(self, storage, config) -> None:
        super().__init__(storage, config)
        self._memory_os_prepare_context: dict[str, dict[str, Any]] = {}

    def prepare_chunks(self, node, source_kind: str | None = None, source_path: str | None = None) -> list[dict[str, Any]]:
        chunks = super().prepare_chunks(node, source_kind=source_kind, source_path=source_path)
        node_id = str(getattr(node, "id", "") or "")
        if node_id:
            self._memory_os_prepare_context[node_id] = {
                "source_kind": source_kind or str(getattr(node, "molecular_type", "") or "raw_content"),
                "source_path": source_path or "",
            }
        return chunks

    def extract_objects(self, node, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        consolidate_enabled = bool(_env_flag(_MEMORY_OS_CONSOLIDATE_ENV, default=True))
        objects = super().extract_objects(node, chunks) if consolidate_enabled else []
        self._emit_memory_os_trace(
            node=node,
            chunks=chunks,
            objects=objects,
            consolidate_enabled=consolidate_enabled,
        )
        return objects

    def _emit_memory_os_trace(
        self,
        *,
        node,
        chunks: list[dict[str, Any]],
        objects: list[dict[str, Any]],
        consolidate_enabled: bool,
    ) -> None:
        node_id = str(getattr(node, "id", "") or "")
        prepare_context = self._memory_os_prepare_context.get(node_id, {})
        source_kind = str(prepare_context.get("source_kind") or getattr(node, "molecular_type", "") or "")
        source_path = str(prepare_context.get("source_path") or "")
        grain_counts: dict[str, int] = {}
        for chunk in chunks:
            grain = str(chunk.get("grain") or "unknown")
            grain_counts[grain] = grain_counts.get(grain, 0) + 1
        object_type_counts: dict[str, int] = {}
        for obj in objects:
            object_type = str(obj.get("object_type") or "unknown")
            object_type_counts[object_type] = object_type_counts.get(object_type, 0) + 1
        entity_tags = str(getattr(node, "entity_tags", "") or "")
        task_type_tag = str(getattr(node, "task_type_tag", "") or "")
        time_bucket = str(getattr(node, "time_bucket", "") or "")
        typed_state_labels = ["episode"]
        if object_type_counts:
            typed_state_labels.append("belief")
        if entity_tags:
            typed_state_labels.append("entity")
        if task_type_tag:
            typed_state_labels.append("task")
        if time_bucket or source_kind in {"conversation_turn", "dialog_session", "text"}:
            typed_state_labels.append("constraint")
        candidate_chunks = self._iter_object_candidate_chunks(chunks) if consolidate_enabled else []
        payload = {
            "operator_family": "memory_os",
            "operator_stage": "ingest",
            "node_id": node_id,
            "source_kind": source_kind,
            "source_path": source_path,
            "typed_state_labels": list(dict.fromkeys(typed_state_labels)),
            "bind_signals": {
                "entity_tags": entity_tags,
                "task_type_tag": task_type_tag,
                "time_bucket": time_bucket,
            },
            "write": {
                "memory_version": None,
                "representation_count": 0,
                "edge_count": 0,
                "chunk_count": len(chunks),
                "grain_counts": grain_counts,
            },
            "consolidate": {
                "consolidate_operator": "structured_object_extraction",
                "consolidate_enabled": consolidate_enabled,
                "operator_off_via_env": not consolidate_enabled,
                "candidate_chunk_count": len(candidate_chunks),
                "selected_chunk_count": len({str(obj.get("source_chunk_id") or "") for obj in objects if obj.get("source_chunk_id")}),
                "selected_parent_chunk_ids": sorted(
                    {str(obj.get("source_chunk_id") or "") for obj in objects if obj.get("source_chunk_id")}
                ),
                "generated_proxy_kinds": sorted(object_type_counts),
                "generated_representation_count": len(objects),
                "object_type_counts": object_type_counts,
            },
        }
        _append_trace_event("writeback_operator_trace.jsonl", payload)
