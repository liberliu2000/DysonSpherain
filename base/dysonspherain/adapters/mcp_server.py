from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from dysonspherain.context_pack.builder import apply_sections, build_pack, build_pack_from_candidates, build_pack_from_memory_ids
from dysonspherain.context_pack.renderers import render_context_pack
from dysonspherain.context_pack.token_budgeter import fit_context_pack
from dysonspherain.memory_os.memory_intent import classify_memory_intent
from dysonspherain.memory_os.observation_store import get_observations, resume_context, search_observations, timeline
from dysonspherain.memory_os.project_state import ProjectStateRequest, get_project_state
from dysonspherain.memory_os.recall_service import RecallRequest, recall
from dysonspherain.memory_os.write_service import WriteMemoryRequest, write_memory
from dysonspherain.product import (
    benchmark_compare,
    benchmark_record,
    create_context_pack as product_context_pack,
    doctor as product_doctor,
    get_capsule,
    mark_contradicted,
    mark_deprecated,
    mark_reverted,
    mark_superseded,
    remember as product_remember,
    retrieve as product_retrieve,
    runtime_event as product_runtime_event,
    search as product_search,
    update_capsule,
)
from dysonspherain.token_economy.evaluator import evaluate


SERVER_INFO = {"name": "dyson-memory", "version": "0.1.0"}
PROTOCOL_VERSION = "2024-11-05"


def mcp_sdk_available() -> bool:
    try:
        return importlib.util.find_spec("mcp.server.fastmcp") is not None
    except ModuleNotFoundError:
        return False


def transport_metadata() -> dict[str, Any]:
    return {
        "transport": "stdio",
        "preferred_implementation": "mcp_sdk",
        "transport_implementation": "mcp_sdk" if mcp_sdk_available() else "jsonrpc_fallback",
        "mcp_sdk_available": mcp_sdk_available(),
        "fallback_reason": "" if mcp_sdk_available() else "python package `mcp` is not installed",
    }

TOOLS = [
    "dyson_memory_intent",
    "dyson_recall",
    "dyson_context_pack",
    "dyson_write_memory",
    "dyson_project_state",
    "dyson_token_economy_eval",
    "dyson_search_memory",
    "dyson_timeline",
    "dyson_get_observations",
    "dyson_resume_context",
    "dyson_product_write",
    "dyson_product_search",
    "dyson_product_retrieve",
    "dyson_product_wake",
    "dyson_product_inspect",
    "dyson_product_update_validity",
    "dyson_product_context_pack",
    "dyson_runtime_before_task",
    "dyson_runtime_on_error",
    "dyson_runtime_after_task",
    "dyson_runtime_pre_compact",
    "dyson_benchmark_record",
    "dyson_benchmark_compare",
    "dyson_health_doctor",
]

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "dyson_memory_intent": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "task_type": {"type": "string", "enum": ["coding", "benchmark", "paper", "debug", "planning", "unknown"], "default": "unknown"},
        },
        "required": ["prompt"],
    },
    "dyson_recall": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "cwd": {"type": "string"},
            "task_type": {"type": "string", "enum": ["coding", "benchmark", "paper", "debug", "planning", "unknown"]},
            "token_budget": {"type": "integer", "default": 1600},
            "include_files": {"type": "boolean", "default": True},
            "include_benchmarks": {"type": "boolean", "default": True},
            "include_prior_prompts": {"type": "boolean", "default": True},
            "freshness": {"type": "string", "enum": ["auto", "recent", "stable"], "default": "auto"},
            "project": {"type": "string", "default": "DysonSpherain"},
        },
        "required": ["query"],
    },
    "dyson_context_pack": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "default": ""},
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "memory_ids": {"type": "array", "items": {"type": "string"}, "default": []},
            "candidates": {"type": "array", "items": {"type": "object"}, "default": []},
            "ranked_items": {"type": "array", "items": {"type": "object"}, "default": []},
            "memory_objects": {"type": "array", "items": {"type": "object"}, "default": []},
            "candidate_type": {"type": "string", "default": "candidate"},
            "token_budget": {"type": "integer", "default": 1600},
            "include_files": {"type": "boolean", "default": True},
            "include_benchmarks": {"type": "boolean", "default": True},
            "include_prior_prompts": {"type": "boolean", "default": True},
            "freshness": {"type": "string", "enum": ["auto", "recent", "stable"], "default": "auto"},
            "format": {"type": "string", "enum": ["markdown", "json"], "default": "markdown"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "summary",
                        "core_evidence",
                        "prior_decisions",
                        "known_failures",
                        "benchmark_state",
                        "relevant_files",
                        "warnings",
                        "next_actions",
                        "recommended_next_actions",
                        "token_economy",
                    ],
                },
                "default": [],
            },
        },
    },
    "dyson_write_memory": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "session_id": {"type": "string"},
            "task_goal": {"type": "string"},
            "summary": {"type": "string"},
            "files_changed": {"type": "array", "items": {"type": "string"}},
            "commands_run": {"type": "array", "items": {"type": "string"}},
            "tests_run": {"type": "array", "items": {"type": "string"}},
            "benchmark_results": {"type": "array", "items": {"type": "string"}},
            "failures": {"type": "array", "items": {"type": "string"}},
            "next_actions": {"type": "array", "items": {"type": "string"}},
            "source": {"type": "string", "enum": ["claude_code", "claude_code_post_compact", "claude_code_post_tool_use", "claude_code_stop", "codex", "manual", "doctor", "integration_report"]},
            "project": {"type": "string", "default": "DysonSpherain"},
        },
        "required": ["summary"],
    },
    "dyson_project_state": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "include_recent_benchmarks": {"type": "boolean", "default": True},
            "include_open_tasks": {"type": "boolean", "default": True},
            "token_budget": {"type": "integer", "default": 1200},
            "project": {"type": "string", "default": "DysonSpherain"},
        },
        "required": ["cwd"],
    },
    "dyson_token_economy_eval": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "candidate_context": {"type": "string"},
            "baseline_context_tokens": {"type": "integer", "default": 0},
            "token_budget": {"type": "integer", "default": 1600},
            "task_type": {"type": "string", "enum": ["coding", "benchmark", "paper", "debug", "planning", "unknown"]},
        },
        "required": ["query", "candidate_context"],
    },
    "dyson_search_memory": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "default": ""},
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "limit": {"type": "integer", "default": 10},
            "kind": {"type": "string"},
            "include_archived": {"type": "boolean", "default": False},
        },
    },
    "dyson_timeline": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "observation_id": {"type": "string"},
            "session_id": {"type": "string"},
            "limit": {"type": "integer", "default": 20},
        },
    },
    "dyson_get_observations": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "observation_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["observation_ids"],
    },
    "dyson_resume_context": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "session_id": {"type": "string"},
            "lookback_hours": {"type": "integer", "default": 24},
            "limit": {"type": "integer", "default": 12},
            "token_budget": {"type": "integer", "default": 1200},
            "format": {"type": "string", "enum": ["markdown", "json"], "default": "markdown"},
        },
    },
    "dyson_product_write": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "text": {"type": "string"},
            "evidence_type": {"type": "string", "default": "note"},
            "source_type": {"type": "string", "default": "mcp"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "session_id": {"type": "string"},
            "task_id": {"type": "string"},
            "agent_id": {"type": "string"},
            "validity_state": {"type": "string", "default": "active"},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
            "file_refs": {"type": "array", "items": {"type": "string"}, "default": []},
            "command_refs": {"type": "array", "items": {"type": "string"}, "default": []},
            "artifact_refs": {"type": "array", "items": {"type": "string"}, "default": []},
            "benchmark_refs": {"type": "array", "items": {"type": "string"}, "default": []},
            "metadata": {"type": "object", "default": {}},
        },
        "required": ["text"],
    },
    "dyson_product_search": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "query": {"type": "string", "default": ""},
            "limit": {"type": "integer", "default": 10},
            "task_type": {"type": "string"},
            "include_invalid": {"type": "boolean", "default": False},
            "gold_ids": {"type": "array", "items": {"type": "string"}, "default": []},
        },
    },
    "dyson_product_retrieve": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
            "show_audit": {"type": "boolean", "default": True},
            "context_pack": {"type": "boolean", "default": False},
            "max_tokens": {"type": "integer", "default": 2000},
            "task_type": {"type": "string"},
            "format": {"type": "string", "enum": ["markdown", "json", "yaml", "text"], "default": "markdown"},
            "sections": {"type": "array", "items": {"type": "string"}, "default": []},
            "section_budget": {"type": "object", "default": {}},
            "agent_role": {"type": "string", "default": "coder"},
            "include_raw_quotes": {"type": "boolean", "default": False},
            "include_artifact_refs": {"type": "boolean", "default": True},
            "include_debug_trace": {"type": "boolean", "default": False},
        },
        "required": ["query"],
    },
    "dyson_product_wake": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "task": {"type": "string", "default": ""},
            "max_tokens": {"type": "integer", "default": 4000},
            "agent_role": {"type": "string", "default": "coder"},
            "task_type": {"type": "string"},
            "format": {"type": "string", "enum": ["markdown", "json", "yaml", "text"], "default": "markdown"},
            "sections": {"type": "array", "items": {"type": "string"}, "default": []},
            "section_budget": {"type": "object", "default": {}},
            "include_raw_quotes": {"type": "boolean", "default": False},
            "include_artifact_refs": {"type": "boolean", "default": True},
            "include_debug_trace": {"type": "boolean", "default": False},
        },
    },
    "dyson_product_inspect": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "capsule_id": {"type": "string"},
        },
        "required": ["capsule_id"],
    },
    "dyson_product_update_validity": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "capsule_id": {"type": "string"},
            "validity_state": {"type": "string", "enum": ["active", "superseded", "deprecated", "contradicted", "reverted", "unknown"]},
            "by_capsule_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["capsule_id", "validity_state"],
    },
    "dyson_product_context_pack": {
        "type": "object",
        "properties": {
            "cwd": {"type": "string"},
            "project": {"type": "string", "default": "DysonSpherain"},
            "query": {"type": "string", "default": ""},
            "max_tokens": {"type": "integer", "default": 2000},
            "agent_role": {"type": "string", "default": "coder"},
            "task_type": {"type": "string"},
            "format": {"type": "string", "enum": ["markdown", "json", "yaml", "text"], "default": "markdown"},
            "sections": {"type": "array", "items": {"type": "string"}, "default": []},
            "section_budget": {"type": "object", "default": {}},
            "include_raw_quotes": {"type": "boolean", "default": False},
            "include_artifact_refs": {"type": "boolean", "default": True},
            "include_debug_trace": {"type": "boolean", "default": False},
        },
    },
    "dyson_runtime_before_task": {
        "type": "object",
        "properties": {"cwd": {"type": "string"}, "project": {"type": "string", "default": "DysonSpherain"}, "task": {"type": "string"}, "max_tokens": {"type": "integer", "default": 4000}},
        "required": ["task"],
    },
    "dyson_runtime_on_error": {
        "type": "object",
        "properties": {"cwd": {"type": "string"}, "project": {"type": "string", "default": "DysonSpherain"}, "error": {"type": "string"}, "max_tokens": {"type": "integer", "default": 3000}},
        "required": ["error"],
    },
    "dyson_runtime_after_task": {
        "type": "object",
        "properties": {"cwd": {"type": "string"}, "project": {"type": "string", "default": "DysonSpherain"}, "summary": {"type": "string"}, "task_id": {"type": "string"}, "changed_files": {"type": "array", "items": {"type": "string"}, "default": []}, "max_tokens": {"type": "integer", "default": 3000}},
        "required": ["summary"],
    },
    "dyson_runtime_pre_compact": {
        "type": "object",
        "properties": {"cwd": {"type": "string"}, "project": {"type": "string", "default": "DysonSpherain"}, "session_id": {"type": "string"}, "max_tokens": {"type": "integer", "default": 3000}},
    },
    "dyson_benchmark_record": {
        "type": "object",
        "properties": {"cwd": {"type": "string"}, "project": {"type": "string", "default": "DysonSpherain"}, "artifact": {"type": "string"}, "benchmark": {"type": "string"}, "status": {"type": "string", "default": "success"}},
        "required": ["artifact"],
    },
    "dyson_benchmark_compare": {
        "type": "object",
        "properties": {"cwd": {"type": "string"}, "project": {"type": "string", "default": "DysonSpherain"}, "current": {"type": "string"}, "baseline": {"type": "string"}},
        "required": ["current", "baseline"],
    },
    "dyson_health_doctor": {
        "type": "object",
        "properties": {"cwd": {"type": "string"}, "project": {"type": "string", "default": "DysonSpherain"}},
    },
}


def smoke_payload() -> dict[str, Any]:
    return {"status": "ok", "serverInfo": SERVER_INFO, "protocolVersion": PROTOCOL_VERSION, "tools": TOOLS, **transport_metadata()}


def tool_descriptor(name: str) -> dict[str, Any]:
    title = name.replace("_", " ").title()
    return {
        "name": name,
        "title": title,
        "description": f"DysonSpherain memory tool {name}",
        "inputSchema": TOOL_SCHEMAS[name],
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _allowed_roots() -> list[Path]:
    project_root = Path(os.environ.get("DYSON_PROJECT_ROOT") or os.getcwd()).expanduser().resolve()
    roots = [project_root, Path(os.environ.get("DYSON_HOME") or project_root / ".dyson").expanduser().resolve(), Path(os.getcwd()).resolve()]
    for raw in str(os.environ.get("DYSON_ALLOWED_PATHS") or "").split(os.pathsep):
        if raw.strip():
            roots.append(Path(raw).expanduser().resolve())
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


@contextmanager
def temporary_allowed_roots(paths: list[Path] | tuple[Path, ...]):
    previous = os.environ.get("DYSON_ALLOWED_PATHS")
    additions = [str(Path(path).expanduser().resolve()) for path in paths]
    current = [item for item in str(previous or "").split(os.pathsep) if item]
    os.environ["DYSON_ALLOWED_PATHS"] = os.pathsep.join([*current, *additions])
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DYSON_ALLOWED_PATHS", None)
        else:
            os.environ["DYSON_ALLOWED_PATHS"] = previous


def _resolve_allowed_path(value: str | None) -> Path:
    path = Path(value or os.getcwd()).expanduser().resolve()
    roots = _allowed_roots()
    if any(path == root or _is_relative_to(path, root) for root in roots):
        return path
    raise ValueError(
        "path_outside_allowed_roots:"
        + json.dumps({"path": str(path), "allowed_roots": [str(root) for root in roots]}, ensure_ascii=False, sort_keys=True)
    )


def _cwd(arguments: dict[str, Any]) -> Path:
    return _resolve_allowed_path(str(arguments.get("cwd") or os.getcwd()))


def _str_list(value: Any) -> list[str]:
    return [str(item) for item in value or []]


def _section_budget(value: Any) -> dict[str, int]:
    return {str(key): int(item) for key, item in dict(value or {}).items()}


def _artifact_path(cwd: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return _resolve_allowed_path(str(path))


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "dyson_memory_intent":
        cwd = str(_resolve_allowed_path(str(arguments.get("cwd") or os.getcwd())))
        return classify_memory_intent(
            str(arguments.get("prompt") or ""),
            cwd=cwd,
            project=str(arguments.get("project") or "DysonSpherain"),
            task_type=str(arguments.get("task_type") or "unknown"),
        ).to_dict()
    if name == "dyson_recall":
        if arguments.get("cwd") is not None:
            arguments = {**arguments, "cwd": str(_resolve_allowed_path(str(arguments.get("cwd"))))}
        request = RecallRequest(**{key: value for key, value in arguments.items() if key in RecallRequest.__dataclass_fields__})
        return recall(request).to_dict()
    if name == "dyson_context_pack":
        cwd = str(_resolve_allowed_path(str(arguments.get("cwd") or os.getcwd())))
        memory_ids = [str(item) for item in arguments.get("memory_ids") or []]
        supplied_candidates: list[dict[str, Any]] = []
        candidate_type = str(arguments.get("candidate_type") or "candidate")
        for key, default_type in (("candidates", "candidate"), ("ranked_items", "ranked_item"), ("memory_objects", "memory_object")):
            values = arguments.get(key) or []
            if values:
                candidate_type = default_type if candidate_type == "candidate" else candidate_type
            supplied_candidates.extend(item for item in values if isinstance(item, dict))
        sections = [str(item) for item in arguments.get("sections") or []] or None
        if supplied_candidates:
            pack, budget = build_pack_from_candidates(
                project=str(arguments.get("project") or "DysonSpherain"),
                candidates=supplied_candidates,
                candidate_type=candidate_type,
                token_budget=int(arguments.get("token_budget") or 1600),
                sections=sections,
                include_files=bool(arguments.get("include_files", True)),
                include_benchmarks=bool(arguments.get("include_benchmarks", True)),
                include_prior_prompts=bool(arguments.get("include_prior_prompts", True)),
                freshness=str(arguments.get("freshness") or "auto"),
            )
        elif memory_ids:
            pack, budget = build_pack_from_memory_ids(
                base_dir=Path(cwd).resolve(),
                project=str(arguments.get("project") or "DysonSpherain"),
                memory_ids=memory_ids,
                token_budget=int(arguments.get("token_budget") or 1600),
                sections=sections,
            )
        else:
            pack, budget = build_pack(
                base_dir=Path(cwd).resolve(),
                project=str(arguments.get("project") or "DysonSpherain"),
                query=str(arguments.get("query") or ""),
                token_budget=int(arguments.get("token_budget") or 1600),
            )
            pack = apply_sections(pack, sections)
            budget_result = fit_context_pack(pack, int(arguments.get("token_budget") or 1600))
            pack = budget_result.pack
            pack.token_economy = budget_result.to_dict()
            budget = budget_result.to_dict()
        rendered = render_context_pack(pack, str(arguments.get("format") or "markdown"))
        return {"status": "ok", "rendered_context": rendered, "token_estimate": {"estimated_tokens": budget["estimated_tokens_after"], "over_budget": budget["over_budget"]}}
    if name == "dyson_write_memory":
        fields = WriteMemoryRequest.__dataclass_fields__
        payload = {key: arguments.get(key) for key in fields}
        if payload.get("cwd") is None:
            payload["cwd"] = os.getcwd()
        payload["cwd"] = str(_resolve_allowed_path(str(payload["cwd"])))
        for list_key in ("files_changed", "commands_run", "tests_run", "benchmark_results", "failures", "next_actions"):
            if payload.get(list_key) is None:
                payload[list_key] = []
        for str_key in ("session_id", "task_goal", "summary"):
            if payload.get(str_key) is None:
                payload[str_key] = ""
        payload["source"] = payload.get("source") or "manual"
        payload["project"] = payload.get("project") or "DysonSpherain"
        return write_memory(WriteMemoryRequest(**payload)).to_dict()
    if name == "dyson_project_state":
        if arguments.get("cwd") is not None:
            arguments = {**arguments, "cwd": str(_resolve_allowed_path(str(arguments.get("cwd"))))}
        return get_project_state(ProjectStateRequest(**{key: value for key, value in arguments.items() if key in ProjectStateRequest.__dataclass_fields__}))
    if name == "dyson_token_economy_eval":
        return evaluate(
            query=str(arguments.get("query") or ""),
            candidate_context=str(arguments.get("candidate_context") or ""),
            baseline_context_tokens=int(arguments.get("baseline_context_tokens") or 0),
            token_budget=int(arguments.get("token_budget") or 1600),
            task_type=str(arguments.get("task_type") or "unknown"),
        ).to_dict()
    if name == "dyson_search_memory":
        cwd = _resolve_allowed_path(str(arguments.get("cwd") or os.getcwd()))
        return search_observations(
            cwd,
            project=str(arguments.get("project") or "DysonSpherain"),
            query=str(arguments.get("query") or ""),
            limit=int(arguments.get("limit") or 10),
            kind=str(arguments.get("kind")) if arguments.get("kind") else None,
            include_archived=bool(arguments.get("include_archived", False)),
        )
    if name == "dyson_timeline":
        cwd = _resolve_allowed_path(str(arguments.get("cwd") or os.getcwd()))
        return timeline(
            cwd,
            project=str(arguments.get("project") or "DysonSpherain"),
            observation_id=str(arguments.get("observation_id")) if arguments.get("observation_id") else None,
            session_id=str(arguments.get("session_id")) if arguments.get("session_id") else None,
            limit=int(arguments.get("limit") or 20),
        )
    if name == "dyson_get_observations":
        cwd = _resolve_allowed_path(str(arguments.get("cwd") or os.getcwd()))
        return get_observations(
            cwd,
            project=str(arguments.get("project") or "DysonSpherain"),
            observation_ids=[str(item) for item in arguments.get("observation_ids") or []],
        )
    if name == "dyson_resume_context":
        cwd = _resolve_allowed_path(str(arguments.get("cwd") or os.getcwd()))
        payload = resume_context(
            cwd,
            project=str(arguments.get("project") or "DysonSpherain"),
            session_id=str(arguments.get("session_id")) if arguments.get("session_id") else None,
            lookback_hours=int(arguments.get("lookback_hours") or 24),
            limit=int(arguments.get("limit") or 12),
            token_budget=int(arguments.get("token_budget") or 1200),
        )
        if str(arguments.get("format") or "markdown") == "markdown":
            return {
                "status": payload.get("status"),
                "project": payload.get("project"),
                "session_id": payload.get("session_id"),
                "rendered_context": payload.get("rendered_context"),
                "token_estimate": payload.get("token_estimate"),
            }
        return payload
    if name == "dyson_product_write":
        cwd = _cwd(arguments)
        return product_remember(
            cwd,
            project_id=str(arguments.get("project") or "DysonSpherain"),
            text=str(arguments.get("text") or ""),
            evidence_type=str(arguments.get("evidence_type") or "note"),
            source_type=str(arguments.get("source_type") or "mcp"),
            title=str(arguments.get("title")) if arguments.get("title") else None,
            summary=str(arguments.get("summary")) if arguments.get("summary") else None,
            session_id=str(arguments.get("session_id")) if arguments.get("session_id") else None,
            task_id=str(arguments.get("task_id")) if arguments.get("task_id") else None,
            agent_id=str(arguments.get("agent_id")) if arguments.get("agent_id") else None,
            validity_state=str(arguments.get("validity_state") or "active"),
            tags=_str_list(arguments.get("tags")),
            file_refs=_str_list(arguments.get("file_refs")),
            command_refs=_str_list(arguments.get("command_refs")),
            artifact_refs=_str_list(arguments.get("artifact_refs")),
            benchmark_refs=_str_list(arguments.get("benchmark_refs")),
            metadata=dict(arguments.get("metadata") or {}),
        )
    if name == "dyson_product_search":
        cwd = _cwd(arguments)
        return product_search(
            cwd,
            project_id=str(arguments.get("project") or "DysonSpherain"),
            query=str(arguments.get("query") or ""),
            limit=int(arguments.get("limit") or 10),
            task_type=str(arguments.get("task_type")) if arguments.get("task_type") else None,
            include_invalid=bool(arguments.get("include_invalid", False)),
            gold_ids=_str_list(arguments.get("gold_ids")),
        )
    if name == "dyson_product_retrieve":
        cwd = _cwd(arguments)
        return product_retrieve(
            cwd,
            project_id=str(arguments.get("project") or "DysonSpherain"),
            query=str(arguments.get("query") or ""),
            limit=int(arguments.get("limit") or 10),
            show_audit=bool(arguments.get("show_audit", True)),
            context_pack=bool(arguments.get("context_pack", False)),
            max_tokens=int(arguments.get("max_tokens") or 2000),
            task_type=str(arguments.get("task_type")) if arguments.get("task_type") else None,
            context_format=str(arguments.get("format") or "markdown"),
            sections=_str_list(arguments.get("sections")),
            section_budget=_section_budget(arguments.get("section_budget")),
            agent_role=str(arguments.get("agent_role") or "coder"),
            include_raw_quotes=bool(arguments.get("include_raw_quotes", False)),
            include_artifact_refs=bool(arguments.get("include_artifact_refs", True)),
            include_debug_trace=bool(arguments.get("include_debug_trace", False)),
        )
    if name in {"dyson_product_wake", "dyson_product_context_pack"}:
        cwd = _cwd(arguments)
        return product_context_pack(
            cwd,
            project_id=str(arguments.get("project") or "DysonSpherain"),
            query=str(arguments.get("task") or arguments.get("query") or ""),
            max_tokens=int(arguments.get("max_tokens") or 2000),
            section_budget=_section_budget(arguments.get("section_budget")),
            sections=_str_list(arguments.get("sections")),
            agent_role=str(arguments.get("agent_role") or "coder"),
            task_type=str(arguments.get("task_type")) if arguments.get("task_type") else None,
            include_raw_quotes=bool(arguments.get("include_raw_quotes", False)),
            include_artifact_refs=bool(arguments.get("include_artifact_refs", True)),
            include_debug_trace=bool(arguments.get("include_debug_trace", False)),
            fmt=str(arguments.get("format") or "markdown"),
        )
    if name == "dyson_product_inspect":
        cwd = _cwd(arguments)
        return {"status": "ok", "capsule": get_capsule(cwd, str(arguments.get("capsule_id") or ""), project_id=str(arguments.get("project") or "DysonSpherain"))}
    if name == "dyson_product_update_validity":
        cwd = _cwd(arguments)
        state = str(arguments.get("validity_state") or "")
        capsule_id = str(arguments.get("capsule_id") or "")
        by = str(arguments.get("by_capsule_id")) if arguments.get("by_capsule_id") else None
        reason = str(arguments.get("reason")) if arguments.get("reason") else None
        if state == "superseded" and by:
            return mark_superseded(cwd, capsule_id, by, reason)
        if state == "contradicted" and by:
            return mark_contradicted(cwd, capsule_id, by, reason)
        if state == "deprecated":
            return mark_deprecated(cwd, capsule_id, reason)
        if state == "reverted" and by:
            return mark_reverted(cwd, capsule_id, by, reason)
        return update_capsule(cwd, capsule_id, project_id=str(arguments.get("project") or "DysonSpherain"), updates={"validity_state": state})
    if name in {"dyson_runtime_before_task", "dyson_runtime_on_error", "dyson_runtime_after_task", "dyson_runtime_pre_compact"}:
        cwd = _cwd(arguments)
        event_map = {
            "dyson_runtime_before_task": ("before_task", {"task": str(arguments.get("task") or "")}),
            "dyson_runtime_on_error": ("on_error", {"error": str(arguments.get("error") or "")}),
            "dyson_runtime_after_task": ("after_task", {"summary": str(arguments.get("summary") or ""), "task_id": arguments.get("task_id"), "changed_files": _str_list(arguments.get("changed_files"))}),
            "dyson_runtime_pre_compact": ("pre_compact", {"session_id": arguments.get("session_id"), "task": "prepare safe compaction context"}),
        }
        event_type, payload = event_map[name]
        return product_runtime_event(cwd, project_id=str(arguments.get("project") or "DysonSpherain"), event_type=event_type, payload=payload, max_tokens=int(arguments.get("max_tokens") or 3000))
    if name == "dyson_benchmark_record":
        cwd = _cwd(arguments)
        return benchmark_record(
            cwd,
            project_id=str(arguments.get("project") or "DysonSpherain"),
            artifact=_artifact_path(cwd, str(arguments.get("artifact") or "")),
            benchmark=str(arguments.get("benchmark")) if arguments.get("benchmark") else None,
            status=str(arguments.get("status") or "success"),
        )
    if name == "dyson_benchmark_compare":
        cwd = _cwd(arguments)
        return benchmark_compare(
            cwd,
            project_id=str(arguments.get("project") or "DysonSpherain"),
            current=_artifact_path(cwd, str(arguments.get("current") or "")),
            baseline=_artifact_path(cwd, str(arguments.get("baseline") or "")),
        )
    if name == "dyson_health_doctor":
        cwd = _cwd(arguments)
        return product_doctor(cwd, project_id=str(arguments.get("project") or "DysonSpherain"))
    return {"status": "error", "error": f"unknown_tool:{name}"}


def _jsonrpc_success(request_id: Any, result: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": {} if result is None else result}


def _jsonrpc_error(request_id: Any, code: int, message: str, data: Any | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def handle_jsonrpc_request(request: Any) -> dict[str, Any] | None:
    if not isinstance(request, dict):
        return _jsonrpc_error(None, -32600, "Invalid Request", "request must be a JSON object")
    request_id = request.get("id")
    if request.get("jsonrpc") != "2.0":
        return _jsonrpc_error(request_id, -32600, "Invalid Request", "jsonrpc must be 2.0")
    method = request.get("method")
    if not isinstance(method, str):
        return _jsonrpc_error(request_id, -32600, "Invalid Request", "method must be a string")
    is_notification = "id" not in request
    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": SERVER_INFO,
            "capabilities": {"tools": {"listChanged": False}},
            "implementation": transport_metadata(),
            "instructions": "Use DysonSpherain memory tools for recall, context packing, writeback, project state, and token economy diagnostics.",
        }
        return None if is_notification else _jsonrpc_success(request_id, result)
    if method == "notifications/initialized":
        return None if is_notification else _jsonrpc_success(request_id, {})
    if method == "tools/list":
        return None if is_notification else _jsonrpc_success(request_id, {"tools": [tool_descriptor(name) for name in TOOLS]})
    if method == "tools/call":
        params = request.get("params")
        if not isinstance(params, dict):
            return _jsonrpc_error(request_id, -32602, "Invalid params", "tools/call params must be an object")
        name = str(params.get("name") or "")
        if name not in TOOLS:
            return _jsonrpc_error(request_id, -32602, "Invalid params", f"unknown_tool:{name}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _jsonrpc_error(request_id, -32602, "Invalid params", "tools/call arguments must be an object")
        try:
            payload = call_tool(name, arguments)
        except Exception as exc:
            return _jsonrpc_error(request_id, -32000, "Tool execution failed", str(exc))
        result = {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, sort_keys=True)}]}
        return None if is_notification else _jsonrpc_success(request_id, result)
    return _jsonrpc_error(request_id, -32601, "Method not found", method)


def _run_mcp_sdk_server() -> bool:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except Exception:
        return False

    server = FastMCP(SERVER_INFO["name"])

    @server.tool(name="dyson_memory_intent", description="Decide whether a prompt should call DysonSpherain memory and which tools to prefer.")
    def dyson_memory_intent_tool(
        prompt: str,
        cwd: str | None = None,
        project: str = "DysonSpherain",
        task_type: str = "unknown",
    ) -> dict[str, Any]:
        return call_tool(
            "dyson_memory_intent",
            {
                "prompt": prompt,
                "cwd": cwd,
                "project": project,
                "task_type": task_type,
            },
        )

    @server.tool(name="dyson_recall", description="Recall DysonSpherain project memory for a query.")
    def dyson_recall_tool(
        query: str,
        cwd: str | None = None,
        task_type: str = "unknown",
        token_budget: int = 1600,
        include_files: bool = True,
        include_benchmarks: bool = True,
        include_prior_prompts: bool = True,
        freshness: str = "auto",
        project: str = "DysonSpherain",
    ) -> dict[str, Any]:
        return call_tool(
            "dyson_recall",
            {
                "query": query,
                "cwd": cwd,
                "task_type": task_type,
                "token_budget": token_budget,
                "include_files": include_files,
                "include_benchmarks": include_benchmarks,
                "include_prior_prompts": include_prior_prompts,
                "freshness": freshness,
                "project": project,
            },
        )

    @server.tool(name="dyson_context_pack", description="Pack DysonSpherain memory records or supplied candidates into a compact context.")
    def dyson_context_pack_tool(
        query: str = "",
        cwd: str | None = None,
        project: str = "DysonSpherain",
        memory_ids: list[str] | None = None,
        candidates: list[dict[str, Any]] | None = None,
        ranked_items: list[dict[str, Any]] | None = None,
        memory_objects: list[dict[str, Any]] | None = None,
        candidate_type: str = "candidate",
        token_budget: int = 1600,
        include_files: bool = True,
        include_benchmarks: bool = True,
        include_prior_prompts: bool = True,
        freshness: str = "auto",
        format: str = "markdown",
        sections: list[str] | None = None,
    ) -> dict[str, Any]:
        return call_tool(
            "dyson_context_pack",
            {
                "query": query,
                "cwd": cwd,
                "project": project,
                "memory_ids": memory_ids or [],
                "candidates": candidates or [],
                "ranked_items": ranked_items or [],
                "memory_objects": memory_objects or [],
                "candidate_type": candidate_type,
                "token_budget": token_budget,
                "include_files": include_files,
                "include_benchmarks": include_benchmarks,
                "include_prior_prompts": include_prior_prompts,
                "freshness": freshness,
                "format": format,
                "sections": sections or [],
            },
        )

    @server.tool(name="dyson_write_memory", description="Write a sanitized, deduplicated task summary into DysonSpherain memory.")
    def dyson_write_memory_tool(
        summary: str,
        cwd: str | None = None,
        session_id: str = "",
        task_goal: str = "",
        files_changed: list[str] | None = None,
        commands_run: list[str] | None = None,
        tests_run: list[str] | None = None,
        benchmark_results: list[str] | None = None,
        failures: list[str] | None = None,
        next_actions: list[str] | None = None,
        source: str = "manual",
        project: str = "DysonSpherain",
    ) -> dict[str, Any]:
        return call_tool(
            "dyson_write_memory",
            {
                "cwd": cwd,
                "session_id": session_id,
                "task_goal": task_goal,
                "summary": summary,
                "files_changed": files_changed or [],
                "commands_run": commands_run or [],
                "tests_run": tests_run or [],
                "benchmark_results": benchmark_results or [],
                "failures": failures or [],
                "next_actions": next_actions or [],
                "source": source,
                "project": project,
            },
        )

    @server.tool(name="dyson_project_state", description="Return a budgeted DysonSpherain project-state summary.")
    def dyson_project_state_tool(
        cwd: str,
        include_recent_benchmarks: bool = True,
        include_open_tasks: bool = True,
        token_budget: int = 1200,
        project: str = "DysonSpherain",
    ) -> dict[str, Any]:
        return call_tool(
            "dyson_project_state",
            {
                "cwd": cwd,
                "include_recent_benchmarks": include_recent_benchmarks,
                "include_open_tasks": include_open_tasks,
                "token_budget": token_budget,
                "project": project,
            },
        )

    @server.tool(name="dyson_token_economy_eval", description="Evaluate whether a candidate context should be injected.")
    def dyson_token_economy_eval_tool(
        query: str,
        candidate_context: str,
        baseline_context_tokens: int = 0,
        token_budget: int = 1600,
        task_type: str = "unknown",
    ) -> dict[str, Any]:
        return call_tool(
            "dyson_token_economy_eval",
            {
                "query": query,
                "candidate_context": candidate_context,
                "baseline_context_tokens": baseline_context_tokens,
                "token_budget": token_budget,
                "task_type": task_type,
            },
        )

    @server.tool(name="dyson_search_memory", description="Search compact DysonSpherain observations before fetching details.")
    def dyson_search_memory_tool(
        query: str = "",
        cwd: str | None = None,
        project: str = "DysonSpherain",
        limit: int = 10,
        kind: str | None = None,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        return call_tool(
            "dyson_search_memory",
            {
                "query": query,
                "cwd": cwd,
                "project": project,
                "limit": limit,
                "kind": kind,
                "include_archived": include_archived,
            },
        )

    @server.tool(name="dyson_timeline", description="Return observation timeline around an observation or session.")
    def dyson_timeline_tool(
        cwd: str | None = None,
        project: str = "DysonSpherain",
        observation_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        return call_tool(
            "dyson_timeline",
            {
                "cwd": cwd,
                "project": project,
                "observation_id": observation_id,
                "session_id": session_id,
                "limit": limit,
            },
        )

    @server.tool(name="dyson_get_observations", description="Fetch full observation details by stable IDs.")
    def dyson_get_observations_tool(
        observation_ids: list[str],
        cwd: str | None = None,
        project: str = "DysonSpherain",
    ) -> dict[str, Any]:
        return call_tool(
            "dyson_get_observations",
            {
                "cwd": cwd,
                "project": project,
                "observation_ids": observation_ids,
            },
        )

    @server.tool(name="dyson_resume_context", description="Return a compact continuation packet for the latest or selected session.")
    def dyson_resume_context_tool(
        cwd: str | None = None,
        project: str = "DysonSpherain",
        session_id: str | None = None,
        lookback_hours: int = 24,
        limit: int = 12,
        token_budget: int = 1200,
        format: str = "markdown",
    ) -> dict[str, Any]:
        return call_tool(
            "dyson_resume_context",
            {
                "cwd": cwd,
                "project": project,
                "session_id": session_id,
                "lookback_hours": lookback_hours,
                "limit": limit,
                "token_budget": token_budget,
                "format": format,
            },
        )

    @server.tool(name="dyson_product_write", description="Write a product evidence capsule.")
    def dyson_product_write_tool(text: str, cwd: str | None = None, project: str = "DysonSpherain", evidence_type: str = "note", source_type: str = "mcp") -> dict[str, Any]:
        return call_tool("dyson_product_write", {"cwd": cwd, "project": project, "text": text, "evidence_type": evidence_type, "source_type": source_type})

    @server.tool(name="dyson_product_search", description="Search product evidence capsules with admission trace support.")
    def dyson_product_search_tool(query: str = "", cwd: str | None = None, project: str = "DysonSpherain", limit: int = 10, task_type: str | None = None) -> dict[str, Any]:
        return call_tool("dyson_product_search", {"cwd": cwd, "project": project, "query": query, "limit": limit, "task_type": task_type})

    @server.tool(name="dyson_product_retrieve", description="Retrieve product evidence capsules and optionally build a context pack.")
    def dyson_product_retrieve_tool(query: str, cwd: str | None = None, project: str = "DysonSpherain", limit: int = 10, show_audit: bool = True, context_pack: bool = False, max_tokens: int = 2000) -> dict[str, Any]:
        return call_tool("dyson_product_retrieve", {"cwd": cwd, "project": project, "query": query, "limit": limit, "show_audit": show_audit, "context_pack": context_pack, "max_tokens": max_tokens})

    @server.tool(name="dyson_product_wake", description="Build a product context pack for task wakeup.")
    def dyson_product_wake_tool(task: str = "", cwd: str | None = None, project: str = "DysonSpherain", max_tokens: int = 4000, format: str = "markdown") -> dict[str, Any]:
        return call_tool("dyson_product_wake", {"cwd": cwd, "project": project, "task": task, "max_tokens": max_tokens, "format": format})

    @server.tool(name="dyson_product_inspect", description="Inspect a product evidence capsule by id.")
    def dyson_product_inspect_tool(capsule_id: str, cwd: str | None = None, project: str = "DysonSpherain") -> dict[str, Any]:
        return call_tool("dyson_product_inspect", {"cwd": cwd, "project": project, "capsule_id": capsule_id})

    @server.tool(name="dyson_product_update_validity", description="Update product capsule validity state.")
    def dyson_product_update_validity_tool(capsule_id: str, validity_state: str, cwd: str | None = None, project: str = "DysonSpherain", by_capsule_id: str | None = None, reason: str | None = None) -> dict[str, Any]:
        return call_tool("dyson_product_update_validity", {"cwd": cwd, "project": project, "capsule_id": capsule_id, "validity_state": validity_state, "by_capsule_id": by_capsule_id, "reason": reason})

    @server.tool(name="dyson_product_context_pack", description="Build a product context pack.")
    def dyson_product_context_pack_tool(query: str = "", cwd: str | None = None, project: str = "DysonSpherain", max_tokens: int = 2000, format: str = "markdown") -> dict[str, Any]:
        return call_tool("dyson_product_context_pack", {"cwd": cwd, "project": project, "query": query, "max_tokens": max_tokens, "format": format})

    @server.tool(name="dyson_runtime_before_task", description="Record before-task runtime event and return context.")
    def dyson_runtime_before_task_tool(task: str, cwd: str | None = None, project: str = "DysonSpherain", max_tokens: int = 4000) -> dict[str, Any]:
        return call_tool("dyson_runtime_before_task", {"cwd": cwd, "project": project, "task": task, "max_tokens": max_tokens})

    @server.tool(name="dyson_runtime_on_error", description="Record runtime error and return context.")
    def dyson_runtime_on_error_tool(error: str, cwd: str | None = None, project: str = "DysonSpherain", max_tokens: int = 3000) -> dict[str, Any]:
        return call_tool("dyson_runtime_on_error", {"cwd": cwd, "project": project, "error": error, "max_tokens": max_tokens})

    @server.tool(name="dyson_runtime_after_task", description="Record after-task runtime summary.")
    def dyson_runtime_after_task_tool(summary: str, cwd: str | None = None, project: str = "DysonSpherain", max_tokens: int = 3000) -> dict[str, Any]:
        return call_tool("dyson_runtime_after_task", {"cwd": cwd, "project": project, "summary": summary, "max_tokens": max_tokens})

    @server.tool(name="dyson_runtime_pre_compact", description="Prepare context before compaction.")
    def dyson_runtime_pre_compact_tool(cwd: str | None = None, project: str = "DysonSpherain", session_id: str | None = None, max_tokens: int = 3000) -> dict[str, Any]:
        return call_tool("dyson_runtime_pre_compact", {"cwd": cwd, "project": project, "session_id": session_id, "max_tokens": max_tokens})

    @server.tool(name="dyson_benchmark_record", description="Record benchmark artifact into product benchmark lab.")
    def dyson_benchmark_record_tool(artifact: str, cwd: str | None = None, project: str = "DysonSpherain", benchmark: str | None = None, status: str = "success") -> dict[str, Any]:
        return call_tool("dyson_benchmark_record", {"cwd": cwd, "project": project, "artifact": artifact, "benchmark": benchmark, "status": status})

    @server.tool(name="dyson_benchmark_compare", description="Compare benchmark artifacts.")
    def dyson_benchmark_compare_tool(current: str, baseline: str, cwd: str | None = None, project: str = "DysonSpherain") -> dict[str, Any]:
        return call_tool("dyson_benchmark_compare", {"cwd": cwd, "project": project, "current": current, "baseline": baseline})

    @server.tool(name="dyson_health_doctor", description="Run product memory health doctor.")
    def dyson_health_doctor_tool(cwd: str | None = None, project: str = "DysonSpherain") -> dict[str, Any]:
        return call_tool("dyson_health_doctor", {"cwd": cwd, "project": project})

    server.run(transport="stdio")
    return True


def _jsonrpc_loop() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = handle_jsonrpc_request(request)
        except Exception as exc:
            response = _jsonrpc_error(None, -32700, "Parse error", str(exc))
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-jsonrpc-fallback", action="store_true", help="Use the built-in JSON-RPC fallback even when the MCP SDK is installed.")
    parser.add_argument("--call-tool")
    parser.add_argument("--arguments", default="{}")
    args = parser.parse_args(argv)
    if args.smoke:
        print(json.dumps(smoke_payload(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.call_tool:
        print(json.dumps(call_tool(args.call_tool, json.loads(args.arguments)), ensure_ascii=False, indent=2, sort_keys=True))
        return
    if not args.force_jsonrpc_fallback and _run_mcp_sdk_server():
        return
    _jsonrpc_loop()


if __name__ == "__main__":
    main()
