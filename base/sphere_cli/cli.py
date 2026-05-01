from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from dysonspherain.memory_runtime.ledger import append_event_payload, replay_events
from dysonspherain.memory_runtime.runtime import append_from_file, compact_safe, graph_state, recall_runtime
from dysonspherain.memory_runtime.scheduler import enqueue_maintenance_jobs, run_maintenance_job, run_scheduler_daemon, run_scheduler_once, schedule_memory_maintenance

from .activation_engine import ActivationEngine
from .background_tasks import BackgroundTaskRunner
from .config import AppConfig
from .context_compiler import compile_context_packet
from .context_assembler import ContextAssembler
from .code_index import build_code_index, relevant_files as code_relevant_files, search_symbol as code_search_symbol
from .creative_reflection_engine import CreativeReflectionEngine
from .evidence_pipeline import EvidencePipeline
from .execution_ledger import (
    create_execution_run,
    get_execution_run,
    load_execution_runs,
    record_postrun,
    render_ledger_list,
    render_resume_packet,
)
from .experiment_registry import (
    compare_runs,
    ingest_artifacts,
    latest_run,
    load_registry,
    markdown_compare_table,
    registry_path,
    registry_summary_path,
    regression_explanation,
    resolve_run,
    write_compare_report,
    write_regression_report,
)
from .memory_lifecycle import append_lifecycle_action, detect_conflicts, load_conflicts, write_conflict_report
from .project_state import (
    archive_memory,
    get_memory,
    list_memories,
    load_project_state,
    render_project_state_markdown,
    save_project_state,
    search_memories,
    update_project_state_from_registry,
    update_memory,
    write_agent_run_summary,
    write_constraint,
    write_conversation_summary,
    write_memory,
)
from .ingestion import FileIngestor
from .memory_auditor import MemoryAuditor
from .memory_manager import SphereMemoryManager
from .memory_writer import MemoryWriter
from .models import ArtifactRecord, MemoryNode, MemoryObject, OpenLoopItem, now_iso
from .path_router import PathRouter
from .real_task_eval import RealTaskEvaluator
from .reranker import RetrievalReranker
from .runtime import UnifiedMemoryRuntime
from .storage import Storage
from .vector_store import VectorStore
from .utils import stable_content_hash
from .workspace import WorkspaceContext, compose_scope, normalize_scope_order

app = typer.Typer(help="Sphere Memory CLI")
remember_app = typer.Typer(help="Write and manage memories", invoke_without_command=True)
memory_app = typer.Typer(help="Search, trace, audit, visualize")
memory_state_app = typer.Typer(help="Project state commands")
memory_conflicts_app = typer.Typer(help="Memory conflict commands")
memory_lifecycle_app = typer.Typer(help="Memory lifecycle commands")
creative_app = typer.Typer(help="Creative reflection commands")
ingest_app = typer.Typer(help="Ingest markdown/code/log/pdf files")
maint_app = typer.Typer(help="Background maintenance tasks")
eval_app = typer.Typer(help="Real-task evaluation commands")
config_app = typer.Typer(help="Persistent local configuration")
benchmark_app = typer.Typer(help="Benchmark smoke and artifact commands")
runs_app = typer.Typer(help="Artifact-backed benchmark run registry")
agent_app = typer.Typer(help="Agent run utilities")
agent_ledger_app = typer.Typer(help="Execution ledger commands")
code_app = typer.Typer(help="Code intelligence index commands")
adapters_app = typer.Typer(help="Install and inspect external agent adapters")
runtime_app = typer.Typer(help="Product runtime event commands")
benchmark_lab_app = typer.Typer(help="Product benchmark lab commands")
product_index_app = typer.Typer(help="Product index maintenance commands")
import_app = typer.Typer(help="Product import commands")

app.add_typer(remember_app, name="remember")
app.add_typer(memory_app, name="memory")
memory_app.add_typer(memory_state_app, name="state")
memory_app.add_typer(memory_conflicts_app, name="conflicts")
memory_app.add_typer(memory_lifecycle_app, name="lifecycle")
app.add_typer(creative_app, name="creative")
app.add_typer(ingest_app, name="ingest")
app.add_typer(maint_app, name="maint")
app.add_typer(eval_app, name="eval")
app.add_typer(config_app, name="config")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(runs_app, name="runs")
app.add_typer(agent_app, name="agent")
agent_app.add_typer(agent_ledger_app, name="ledger")
app.add_typer(code_app, name="code")
app.add_typer(adapters_app, name="adapters")
app.add_typer(runtime_app, name="runtime")
app.add_typer(benchmark_lab_app, name="benchmark-lab")
app.add_typer(product_index_app, name="index")
app.add_typer(import_app, name="import")

console = Console()
_RUNTIME_OVERRIDES: dict[str, object] = {}


def _active_project(explicit: Optional[str] = None) -> str:
    return explicit or str(_RUNTIME_OVERRIDES.get("project_name") or "default")


def build_runtime() -> UnifiedMemoryRuntime:
    overrides = _load_local_config_overrides()
    overrides.update(dict(_RUNTIME_OVERRIDES))
    return UnifiedMemoryRuntime.from_base_dir(Path.cwd(), config_overrides=overrides or None)


def _local_config_path() -> Path:
    return Path.cwd() / ".dysonspherain_config.json"


def _load_local_config_overrides() -> dict[str, object]:
    path = _local_config_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): v for k, v in payload.items()}


def _save_local_config_overrides(payload: dict[str, object]) -> None:
    path = _local_config_path()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _coerce_config_value(key: str, raw: str) -> object:
    bool_keys = {
        "embedding_fail_fast",
        "enable_benchmark_route_tuning",
        "enable_lightweight_edge_writeback",
        "enable_retrieval_cache",
        "enable_context_compressor",
    }
    int_keys = {"retrieval_topk_coarse", "retrieval_topk_fine", "creative_beam_width", "creative_max_hops"}
    float_keys = {"scope_priority_weight", "creative_novelty_weight", "creative_support_weight"}
    if key in bool_keys:
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    if key in int_keys:
        return int(raw)
    if key in float_keys:
        return float(raw)
    if key == "retrieval_scope_order":
        return normalize_scope_order(raw)
    return raw


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _json_object_option(value: Optional[str]) -> dict[str, int]:
    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise typer.BadParameter("Expected a JSON object.")
    return {str(key): int(item) for key, item in payload.items()}


def build_services() -> tuple:
    runtime = build_runtime()
    services = runtime.services
    config = services.config
    storage = services.storage
    vector_store = services.vector_store
    manager = services.manager
    activation = services.activation
    router = services.router
    assembler = services.assembler
    writer = services.writer
    auditor = services.auditor
    creative = services.creative
    reranker = services.reranker
    ingestor = services.ingestor
    background = services.background
    return config, storage, vector_store, manager, activation, router, assembler, writer, auditor, creative, reranker, ingestor, background


@app.callback()
def main(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Active workspace name"),
    project: Optional[str] = typer.Option(None, "--project", help="Active project name"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Active session id"),
    scope: Optional[str] = typer.Option(None, "--scope", help="Explicit scope, for example global or project:alpha"),
    scope_order: str = typer.Option("project,session,global", "--scope-order", help="Recall priority order"),
    mode: str = typer.Option("balanced", "--mode", help="Mode alias: fast, balanced, or deep"),
) -> None:
    context = WorkspaceContext.from_values(
        workspace=workspace,
        project=project,
        session_id=session_id,
        scope=scope,
        scope_order=normalize_scope_order(scope_order),
        mode=mode,
    )
    _RUNTIME_OVERRIDES.clear()
    if workspace:
        _RUNTIME_OVERRIDES["workspace_name"] = context.workspace
    if project:
        _RUNTIME_OVERRIDES["project_name"] = context.project
    if session_id:
        _RUNTIME_OVERRIDES["session_id"] = context.session_id
    if scope:
        _RUNTIME_OVERRIDES["scope"] = context.scope
    if scope_order:
        _RUNTIME_OVERRIDES["retrieval_scope_order"] = context.scope_order
    if mode:
        _RUNTIME_OVERRIDES["mode"] = context.mode


@app.command()
def init(project: Optional[str] = typer.Option(None, "--project", help="Initialize product evidence store for a project")) -> None:
    config, storage, vector_store, *_ = build_services()
    print(f"[green]Initialized metadata DB:[/green] {storage.config.db_path}")
    print(f"[green]Initialized vector DB:[/green] {config.vector_dir}")
    if project:
        from dysonspherain.product import init_product_store

        product = init_product_store(Path.cwd(), project_id=project)
        print(f"[green]Initialized product memory DB:[/green] {product['db_path']}")
    console.print_json(json.dumps(vector_store.info(), ensure_ascii=False, indent=2))


@remember_app.callback(invoke_without_command=True)
def remember_direct(
    ctx: typer.Context,
    project: str = typer.Option("DysonSpherain", "--project", help="Project id"),
    evidence_type: str = typer.Option("note", "--type", help="Evidence type"),
    text: Optional[str] = typer.Option(None, "--text", help="Raw memory text"),
    task: Optional[str] = typer.Option(None, "--task", help="Task id"),
    status: str = typer.Option("active", "--status", help="Validity state"),
    tags: list[str] = typer.Option(None, "--tags", help="Tag values"),
    title: Optional[str] = typer.Option(None, "--title", help="Short title"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session id"),
    no_index: bool = typer.Option(False, "--no-index", help="Do not extract sparse terms"),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if not text:
        raise typer.BadParameter("Provide --text or use `remember add`.")
    from dysonspherain.product import remember

    payload = remember(
        Path.cwd(),
        project_id=project,
        text=text,
        evidence_type=evidence_type,
        source_type="manual",
        title=title,
        task_id=task,
        validity_state=status,
        tags=tags or [],
        session_id=session_id,
        no_index=no_index,
    )
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@app.command()
def status() -> None:
    config, storage, vector_store, *_ = build_services()
    with storage.connect() as conn:
        representation_count = int(conn.execute("SELECT COUNT(*) AS c FROM memory_representations").fetchone()["c"])
        delta_count = int(conn.execute("SELECT COUNT(*) AS c FROM memory_deltas").fetchone()["c"])
        retrieval_cache_count = int(conn.execute("SELECT COUNT(*) AS c FROM retrieval_cache").fetchone()["c"])
        completion_cache_count = int(conn.execute("SELECT COUNT(*) AS c FROM completion_cache").fetchone()["c"])
        snapshot_count = int(conn.execute("SELECT COUNT(*) AS c FROM profile_snapshots").fetchone()["c"])
        artifact_count = int(conn.execute("SELECT COUNT(*) AS c FROM artifact_registry").fetchone()["c"])
        open_loop_count = int(conn.execute("SELECT COUNT(*) AS c FROM open_loops").fetchone()["c"])
        edge_count = int(conn.execute("SELECT COUNT(*) AS c FROM memory_edges").fetchone()["c"])
    payload = {
        "metadata_db": str(storage.config.db_path),
        "vector_db": str(config.vector_dir),
        "vector_info": vector_store.info(),
        "chunk_count": storage.count_chunks(),
        "object_count": len(storage.fetch_objects()),
        "node_count": len(storage.fetch_nodes()),
        "edge_count": edge_count,
        "representation_count": representation_count,
        "delta_count": delta_count,
        "tracked_files": len(storage.fetch_ingest_files()),
        "artifact_count": artifact_count,
        "open_loop_count": open_loop_count,
        "workspace_config": {
            "workspace_name": config.workspace_name,
            "project_name": config.project_name,
            "session_id": config.session_id,
            "scope": config.scope,
            "retrieval_scope_order": list(config.retrieval_scope_order),
            "mode": config.mode,
        },
        "embedding_guard": {
            "embedding_fail_fast": config.embedding_fail_fast,
            "embedding_provider": vector_store.info().get("embedding_provider"),
            "embedding_model": vector_store.info().get("embedding_model"),
            "fallback_in_use": vector_store.info().get("fallback_in_use"),
            "primary_embedding_load_error": vector_store.info().get("primary_embedding_load_error"),
        },
        "retrieval_config": {
            "enable_task_router": config.enable_task_router,
            "enable_object_shortcut": config.enable_object_shortcut,
            "enable_temporal_prefilter": config.enable_temporal_prefilter,
            "retrieval_topk_coarse": config.retrieval_topk_coarse,
            "retrieval_topk_fine": config.retrieval_topk_fine,
            "enable_seed_clustering": config.enable_seed_clustering,
            "enable_semantic_dedup": config.enable_semantic_dedup,
            "enable_conditional_rerank": config.enable_conditional_rerank,
            "enable_lightweight_edge_writeback": config.enable_lightweight_edge_writeback,
            "enable_scope_priority": config.enable_scope_priority,
            "scope_priority_weight": config.scope_priority_weight,
        },
        "compression_config": {
            "enable_ingest_compression": config.enable_ingest_compression,
            "enable_content_hash_dedup": config.enable_content_hash_dedup,
            "enable_delta_memory_writer": config.enable_delta_memory_writer,
            "enable_structured_compression": config.enable_structured_compression,
            "enable_multires_summaries": config.enable_multires_summaries,
            "enable_retrieval_proxy_index": config.enable_retrieval_proxy_index,
            "enable_context_compressor": config.enable_context_compressor,
        },
        "cache_config": {
            "enable_retrieval_cache": config.enable_retrieval_cache,
            "enable_completion_cache": config.enable_completion_cache,
            "enable_profile_snapshot_cache": config.enable_profile_snapshot_cache,
            "enable_prism_path_cache": config.enable_prism_path_cache,
            "retrieval_cache_ttl_seconds": config.retrieval_cache_ttl_seconds,
            "completion_cache_ttl_seconds": config.completion_cache_ttl_seconds,
            "profile_snapshot_ttl_seconds": config.profile_snapshot_ttl_seconds,
            "retrieval_cache_entries": retrieval_cache_count,
            "completion_cache_entries": completion_cache_count,
            "profile_snapshot_entries": snapshot_count,
        },
        "creative_config": {
            "creative_mode": config.creative_mode_name,
            "creative_beam_width": config.creative_beam_width,
            "creative_max_hops": config.creative_max_hops,
            "creative_neighbors_per_hop": config.creative_neighbors_per_hop,
            "creative_enable_analogy": config.creative_enable_analogy,
            "creative_enable_contrast": config.creative_enable_contrast,
            "creative_enable_transfer": config.creative_enable_transfer,
            "creative_enable_temporal": config.creative_enable_temporal,
            "creative_enable_composition": config.creative_enable_composition,
            "creative_novelty_weight": config.creative_novelty_weight,
            "creative_support_weight": config.creative_support_weight,
            "creative_diversity_weight": config.creative_diversity_weight,
            "creative_conflict_penalty": config.creative_conflict_penalty,
            "creative_reflection_gain": config.creative_reflection_gain,
            "creative_max_output_paths": config.creative_max_output_paths,
        },
    }
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))


@remember_app.command("add")
def remember_add(
    shell: int = typer.Option(..., min=0, max=4),
    sector: str = typer.Option(...),
    zone: str = typer.Option(...),
    cell: str = typer.Option(...),
    molecular_type: str = typer.Option(..., "--molecular-type"),
    summary: str = typer.Option(...),
    content: str = typer.Option("", help="Raw content or detailed note"),
    content_ref: str = typer.Option("", help="Raw file pointer"),
    importance: float = typer.Option(0.5, min=0.0, max=1.0),
    creative_score: float = typer.Option(0.3, min=0.0, max=1.0),
    stability_score: float = typer.Option(0.5, min=0.0, max=1.0),
    compression_level: str = typer.Option("medium"),
    stage: str = typer.Option("staging"),
    tags: str = typer.Option(""),
    scope: Optional[str] = typer.Option(None, help="Optional explicit scope override"),
    project: Optional[str] = typer.Option(None, help="Optional project override"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Optional session override"),
    source_type: str = typer.Option("memory_note", help="Source type / provenance tag"),
    verification_status: str = typer.Option("unverified", help="Verification status"),
) -> None:
    runtime = build_runtime()
    active_scope = compose_scope(scope=scope, project=project or runtime.services.config.project_name, session_id=session_id or runtime.services.config.session_id)
    node = MemoryNode(
        shell=shell,
        sector=sector,
        zone=zone,
        cell=cell,
        molecular_type=molecular_type,
        summary=summary,
        raw_content=content or None,
        content_ref=content_ref or None,
        scope=active_scope,
        workspace=runtime.services.config.workspace_name,
        project=project or runtime.services.config.project_name,
        session_id=session_id or runtime.services.config.session_id,
        source_type=source_type,
        source_ref=content_ref or None,
        verification_status=verification_status,
        importance=importance,
        creative_score=creative_score,
        stability_score=stability_score,
        compression_level=compression_level,
        stage=stage,
        tags=tags or None,
    )
    report = runtime.writeback_memory(node, source_kind=molecular_type, source_path=content_ref or None)
    print(f"[green]Added memory node[/green] {node.id}")
    print(
        f"Created {report['chunk_count']} chunks, {report['object_count']} objects, "
        f"{report['neighbor_count']} neighbor links and {report['edge_count']} relation edges"
    )


@remember_app.command("promote")
def remember_promote(node_id: str = typer.Option(..., "--node-id")) -> None:
    _, _, _, manager, *_ = build_services()
    manager.promote_node(node_id)
    print(f"[green]Promoted[/green] {node_id} to long_term")


@ingest_app.command("path")
def ingest_path(
    target: str = typer.Argument(..., help="File or directory to ingest"),
    shell: int = typer.Option(4, min=0, max=4),
    sector: str = typer.Option("raw"),
    zone: str = typer.Option(...),
    recursive: bool = typer.Option(True, help="Recurse into subdirectories"),
    stage: str = typer.Option("staging"),
    tags: str = typer.Option(""),
) -> None:
    _, _, _, _, _, _, _, _, _, _, _, ingestor, _ = build_services()
    results = ingestor.ingest_path(target, shell=shell, sector=sector, zone=zone, recursive=recursive, stage=stage, tags=tags)
    _render_ingest_results(results, f"Ingested Files: {target}")


@ingest_app.command("sync")
def ingest_sync(
    target: str = typer.Argument(..., help="File or directory to incrementally sync"),
    shell: int = typer.Option(4, min=0, max=4),
    sector: str = typer.Option("raw"),
    zone: str = typer.Option(...),
    recursive: bool = typer.Option(True),
    stage: str = typer.Option("staging"),
    tags: str = typer.Option(""),
) -> None:
    _, _, _, _, _, _, _, _, _, _, _, ingestor, _ = build_services()
    results = ingestor.sync_path(target, shell=shell, sector=sector, zone=zone, recursive=recursive, stage=stage, tags=tags)
    _render_ingest_results(results, f"Incremental Sync: {target}")


@ingest_app.command("watch")
def ingest_watch(
    target: str = typer.Argument(...),
    shell: int = typer.Option(4, min=0, max=4),
    sector: str = typer.Option("raw"),
    zone: str = typer.Option(...),
    recursive: bool = typer.Option(True),
    stage: str = typer.Option("staging"),
    tags: str = typer.Option(""),
    poll_seconds: float = typer.Option(3.0),
    max_rounds: int = typer.Option(3, help="0 means forever; CLI default here is 3 for safety"),
) -> None:
    _, _, _, _, _, _, _, _, _, _, _, ingestor, _ = build_services()
    for idx, results in enumerate(
        ingestor.watch_path(
            target,
            shell=shell,
            sector=sector,
            zone=zone,
            recursive=recursive,
            stage=stage,
            tags=tags,
            poll_seconds=poll_seconds,
            max_rounds=max_rounds,
        ),
        start=1,
    ):
        _render_ingest_results(results, f"Watch Round {idx}: {target}")


@memory_app.command("find")
def memory_find(
    query: str,
    top_k: int = typer.Option(8),
    task_type: str = typer.Option("trace"),
    rerank_mode: str = typer.Option("rule", help="Reserved for node rerank compatibility; evidence ranking is now evidence-first"),
    explain: bool = typer.Option(False, help="Print retrieval diagnostics JSON"),
) -> None:
    runtime = build_runtime()
    evidence = runtime.retrieve_evidence(query, task_type=task_type, top_k=top_k)
    completion = runtime.complete_with_objects(
        query=query,
        evidence=evidence,
        support_top_k=min(4, top_k),
        object_top_k=min(4, top_k),
    )

    table = Table(title=f"Memory Results: {query}")
    table.add_column("chunk_id")
    table.add_column("node_id")
    table.add_column("address")
    table.add_column("evidence")
    table.add_column("dense")
    table.add_column("lex")
    table.add_column("time")
    table.add_column("pref")
    table.add_column("grain")
    table.add_column("text")
    for chunk in completion.core_evidence:
        address = f"{chunk.get('shell')}/{chunk.get('sector')}/{chunk.get('zone')}/{chunk.get('cell')}"
        table.add_row(
            str(chunk.get("chunk_id", "")),
            str(chunk.get("node_id", "")),
            address,
            str(chunk.get("evidence_score", "")),
            str(chunk.get("dense_score", "")),
            str(chunk.get("query_lexical", "")),
            str(chunk.get("time_score", "")),
            str(chunk.get("preference_score", "")),
            str(chunk.get("grain", "")),
            (chunk.get("text") or "")[:90],
        )
    console.print(table)

    if completion.evidence_objects:
        obj_table = Table(title="Evidence Objects")
        obj_table.add_column("object_id")
        obj_table.add_column("type")
        obj_table.add_column("source_node")
        obj_table.add_column("score")
        obj_table.add_column("text")
        for obj in completion.evidence_objects:
            obj_table.add_row(
                str(obj.get("object_id", "")),
                str(obj.get("object_type", "")),
                str(obj.get("source_node_id", "")),
                str(obj.get("object_score", "")),
                (obj.get("object_text") or "")[:90],
            )
        console.print(obj_table)
    if explain:
        console.print_json(
            json.dumps(
                {
                    "route": evidence.query_route,
                    "retrieval_diagnostics": evidence.diagnostics,
                    "completion_diagnostics": completion.diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


@memory_app.command("remember")
def memory_remember(
    content: str = typer.Option(..., "--content", help="Memory content"),
    memory_type: str = typer.Option("fact", "--type", help="Memory type, for example decision/task/fact"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    title: Optional[str] = typer.Option(None, "--title", help="Short title"),
    source: str = typer.Option("cli", "--source", help="Source/provenance"),
    status: str = typer.Option("current", "--status", help="Memory status"),
) -> None:
    payload = write_memory(
        Path.cwd(),
        memory_type=memory_type,
        project=project,
        content=content,
        source=source,
        metadata={"title": title or content[:80], "status": status},
    )
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument("", help="Search terms"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    memory_type: Optional[str] = typer.Option(None, "--type", help="Optional memory type filter"),
    include_archived: bool = typer.Option(False, "--include-archived", help="Include archived memories"),
    json_output: bool = typer.Option(False, "--json", help="Render JSON instead of a table"),
) -> None:
    rows = search_memories(
        Path.cwd(),
        project,
        query,
        include_archived=include_archived,
        memory_type=memory_type,
    )
    if json_output:
        console.print_json(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    table = Table(title=f"Project Memories: {project}")
    table.add_column("memory_id")
    table.add_column("type")
    table.add_column("status")
    table.add_column("title")
    table.add_column("content")
    for row in rows:
        table.add_row(
            str(row.get("memory_id") or ""),
            str(row.get("memory_type") or ""),
            str(row.get("status") or ""),
            str(row.get("title") or "")[:60],
            str(row.get("content") or "")[:90],
        )
    console.print(table)


@memory_app.command("obs-search")
def memory_obs_search(
    query: str = typer.Argument("", help="Observation search terms"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    limit: int = typer.Option(10, "--limit", min=1, max=100),
    include_archived: bool = typer.Option(False, "--include-archived"),
) -> None:
    from dysonspherain.memory_os.observation_store import search_observations

    console.print_json(
        json.dumps(
            search_observations(Path.cwd(), project=project, query=query, limit=limit, include_archived=include_archived),
            ensure_ascii=False,
            indent=2,
        )
    )


@memory_app.command("obs-get")
def memory_obs_get(
    observation_ids: list[str] = typer.Argument(..., help="Observation ids"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    from dysonspherain.memory_os.observation_store import get_observations

    console.print_json(json.dumps(get_observations(Path.cwd(), project=project, observation_ids=observation_ids), ensure_ascii=False, indent=2))


@memory_app.command("obs-timeline")
def memory_obs_timeline(
    observation_id: Optional[str] = typer.Option(None, "--observation-id"),
    session_id: Optional[str] = typer.Option(None, "--session-id"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
) -> None:
    from dysonspherain.memory_os.observation_store import timeline

    console.print_json(
        json.dumps(
            timeline(Path.cwd(), project=project, observation_id=observation_id, session_id=session_id, limit=limit),
            ensure_ascii=False,
            indent=2,
        )
    )


@memory_app.command("resume-context")
def memory_resume_context(
    session_id: Optional[str] = typer.Option(None, "--session-id"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    lookback_hours: int = typer.Option(24, "--lookback-hours", min=1),
    token_budget: int = typer.Option(1200, "--token-budget", min=100),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    from dysonspherain.memory_os.observation_store import resume_context

    payload = resume_context(Path.cwd(), project=project, session_id=session_id, lookback_hours=lookback_hours, token_budget=token_budget)
    if json_output:
        console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(payload.get("rendered_context") or "")


@memory_app.command("append")
def memory_runtime_append(
    file: Optional[Path] = typer.Option(None, "--file", help="JSON event file"),
    event_type: str = typer.Option("user_instruction_received", "--event-type", help="MemoryEvent type"),
    content: Optional[str] = typer.Option(None, "--content", help="Inline event content"),
    source: str = typer.Option("manual", "--source", help="Event source"),
    actor: str = typer.Option("user", "--actor", help="Event actor"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Session id"),
) -> None:
    if file is not None:
        payload = append_from_file(Path.cwd(), file, source=source, project=project)
    else:
        if not content:
            raise typer.BadParameter("Provide --file or --content.")
        payload = append_event_payload(
            Path.cwd(),
            event_type=event_type,
            payload={"content": content, "summary": content[:240]},
            source=source,
            actor=actor,
            project=project,
            session_id=session_id,
        ).to_dict()
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))


@memory_app.command("recall")
def memory_runtime_recall(
    query: str = typer.Argument(..., help="Recall query"),
    budget: int = typer.Option(1200, "--budget", min=100, help="Context token budget"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    trace: bool = typer.Option(False, "--trace", help="Include router/compiler trace"),
    explain: bool = typer.Option(False, "--explain", help="Render JSON explanation instead of context markdown"),
) -> None:
    payload = recall_runtime(Path.cwd(), query, project=project, budget=budget, trace=trace or explain)
    if explain:
        console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(payload.get("rendered_context") or "")


@memory_app.command("graph")
def memory_runtime_graph(
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    json_output: bool = typer.Option(True, "--json/--table", help="Render full graph JSON"),
) -> None:
    payload = graph_state(Path.cwd(), project=project)
    if json_output:
        console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    table = Table(title=f"Task Situation Graph: {project}")
    table.add_column("node_id")
    table.add_column("type")
    table.add_column("status")
    table.add_column("title")
    for node in payload.get("nodes", []):
        table.add_row(str(node.get("node_id")), str(node.get("node_type")), str(node.get("status")), str(node.get("title"))[:80])
    console.print(table)


@memory_app.command("replay")
def memory_runtime_replay(
    from_file: Optional[Path] = typer.Option(None, "--from", help="Replay a specific ledger JSONL file"),
    project: Optional[str] = typer.Option(None, "--project", help="Project filter"),
) -> None:
    events = replay_events(Path.cwd(), project=project, from_path=from_file)
    console.print_json(json.dumps({"status": "ok", "event_count": len(events), "events": [event.to_dict() for event in events]}, ensure_ascii=False, indent=2))


@memory_app.command("compact")
def memory_runtime_compact(
    safe: bool = typer.Option(False, "--safe", help="Run safe compaction/maintenance only"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    if not safe:
        raise typer.BadParameter("Only --safe compaction is available in the first runtime implementation.")
    console.print_json(json.dumps(compact_safe(Path.cwd(), project=project), ensure_ascii=False, indent=2))


@memory_app.command("scheduler")
def memory_runtime_scheduler(
    trigger: str = typer.Option("session_ended", "--trigger", help="Scheduler trigger"),
    event_ids: str = typer.Option("", "--event-ids", help="Comma-separated event ids"),
    run_once: bool = typer.Option(False, "--run-once", help="Run scheduled jobs immediately"),
    enqueue: bool = typer.Option(False, "--enqueue", help="Persist jobs into the scheduler queue"),
    drain_queue: bool = typer.Option(False, "--drain-queue", help="Run pending queued jobs"),
    daemon: bool = typer.Option(False, "--daemon", help="Continuously drain pending queued jobs"),
    interval_seconds: float = typer.Option(5.0, "--interval-seconds", help="Daemon polling interval"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    if daemon:
        console.print_json(json.dumps(run_scheduler_daemon(Path.cwd(), project=project, interval_seconds=interval_seconds), ensure_ascii=False, indent=2))
        return
    if drain_queue:
        console.print_json(json.dumps(run_scheduler_once(Path.cwd(), project=project), ensure_ascii=False, indent=2))
        return
    jobs = enqueue_maintenance_jobs(Path.cwd(), trigger, _split_csv(event_ids), project=project) if enqueue else schedule_memory_maintenance(trigger, _split_csv(event_ids))
    if not run_once:
        console.print_json(json.dumps({"status": "ok", "jobs": [job.to_dict() for job in jobs]}, ensure_ascii=False, indent=2))
        return
    results = [run_maintenance_job(Path.cwd(), job, project=project).to_dict() for job in jobs]
    console.print_json(json.dumps({"status": "ok", "jobs": [job.to_dict() for job in jobs], "results": results}, ensure_ascii=False, indent=2))


@memory_app.command("inspect")
def memory_inspect(
    memory_id: str = typer.Argument(..., help="Project memory id"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    payload = get_memory(Path.cwd(), project, memory_id)
    if payload is None:
        raise typer.Exit(1)
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))


@memory_app.command("update")
def memory_update(
    memory_id: str = typer.Argument(..., help="Project memory id"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    content: Optional[str] = typer.Option(None, "--content", help="Replacement content"),
    title: Optional[str] = typer.Option(None, "--title", help="Replacement title"),
    status: Optional[str] = typer.Option(None, "--status", help="Replacement status"),
) -> None:
    patch = {key: value for key, value in {"content": content, "title": title, "status": status}.items() if value is not None}
    if not patch:
        raise typer.BadParameter("Provide --content, --title, or --status.")
    try:
        payload = update_memory(Path.cwd(), project, memory_id, patch)
    except KeyError:
        raise typer.Exit(1) from None
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))


@memory_app.command("raw-find")
def memory_raw_find(query: str, top_k: int = typer.Option(6)) -> None:
    config, storage, vector_store, *_ = build_services()
    results = vector_store.search(query, top_k=top_k)
    table = Table(title=f"Raw Vector Retrieval: {query}")
    table.add_column("chunk_id")
    table.add_column("node_id")
    table.add_column("similarity")
    table.add_column("text")
    for item in results:
        meta = item.get("metadata") or {}
        table.add_row(item["chunk_id"], str(meta.get("node_id", "")), str(item["similarity"]), item["document"][:110])
    console.print(table)
    console.print_json(json.dumps(vector_store.info(), ensure_ascii=False, indent=2))
    print(f"[cyan]SQLite metadata DB:[/cyan] {storage.config.db_path}")


@memory_app.command("list")
def memory_list(
    limit: int = typer.Option(20, min=1),
    project: Optional[str] = typer.Option(None),
    scope: Optional[str] = typer.Option(None),
    molecular_type: Optional[str] = typer.Option(None, "--molecular-type"),
) -> None:
    _, storage, _, *_ = build_services()
    clauses: list[str] = []
    params: list[object] = []
    if project:
        clauses.append("project = ?")
        params.append(project)
    if scope:
        clauses.append("scope = ?")
        params.append(scope)
    if molecular_type:
        clauses.append("molecular_type = ?")
        params.append(molecular_type)
    where = " AND ".join(clauses)
    rows = storage.fetch_nodes(where, tuple(params))
    rows.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
    table = Table(title="Memory Nodes")
    table.add_column("id")
    table.add_column("scope")
    table.add_column("project")
    table.add_column("type")
    table.add_column("status")
    table.add_column("summary")
    for row in rows[:limit]:
        table.add_row(
            str(row.get("id") or ""),
            str(row.get("scope") or "global"),
            str(row.get("project") or ""),
            str(row.get("molecular_type") or ""),
            str(row.get("verification_status") or ""),
            (str(row.get("summary") or ""))[:90],
        )
    console.print(table)


@memory_app.command("explain")
def memory_explain(
    node_id: Optional[str] = typer.Option(None),
    object_id: Optional[str] = typer.Option(None),
    chunk_id: Optional[str] = typer.Option(None),
    last_recall: bool = typer.Option(False, help="Show the last stored recall trace"),
) -> None:
    if last_recall:
        latest_packet = Path.cwd() / "data" / "projections" / "latest_context_packet.json"
        latest_audit = Path.cwd() / "data" / "projections" / "latest_recall_audit.json"
        if latest_packet.exists():
            payload = {
                "packet": json.loads(latest_packet.read_text(encoding="utf-8")),
                "audit": json.loads(latest_audit.read_text(encoding="utf-8")) if latest_audit.exists() else {},
            }
            console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        _, storage, _, *_ = build_services()
        payload = storage.get_runtime_state("last_recall_trace")
        console.print_json(payload or "{}")
        return
    _, storage, _, *_ = build_services()
    if node_id:
        payload = {
            "node": storage.fetch_node_by_id(node_id),
            "chunks": storage.fetch_chunks_for_node(node_id),
            "objects": storage.fetch_objects("source_node_id = ?", (node_id,)),
        }
        console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if object_id:
        payload = storage.fetch_objects_by_ids([object_id])
        console.print_json(json.dumps(payload[0] if payload else {}, ensure_ascii=False, indent=2))
        return
    if chunk_id:
        payload = storage.hydrate_chunk_with_node_metadata(chunk_id)
        console.print_json(json.dumps(payload or {}, ensure_ascii=False, indent=2))
        return
    raise typer.BadParameter("Provide --node-id, --object-id, --chunk-id, or --last-recall.")


@memory_app.command("debug-context")
def memory_debug_context(last: bool = typer.Option(True, "--last/--no-last", help="Show the last stored recall trace")) -> None:
    _, storage, _, *_ = build_services()
    if last:
        console.print_json(storage.get_runtime_state("last_recall_trace") or "{}")
        return
    print(compile_context_packet(Path.cwd(), task="Debug current memory context.", project=_active_project(), mode="debug", max_tokens=4000))


def _collect_selection_matches(payload: object, memory_id: str, path: str = "$") -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    if isinstance(payload, dict):
        candidate_ids = {
            str(payload.get(key))
            for key in ("id", "node_id", "chunk_id", "object_id", "memory_id", "source_id")
            if payload.get(key) is not None
        }
        if memory_id in candidate_ids:
            matches.append({"path": path, "payload": payload})
        for key, value in payload.items():
            matches.extend(_collect_selection_matches(value, memory_id, f"{path}.{key}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            matches.extend(_collect_selection_matches(value, memory_id, f"{path}[{index}]"))
    return matches


@memory_app.command("why-selected")
def memory_why_selected(
    memory_id: str = typer.Argument(..., help="Memory, node, chunk, object, or source id"),
    last: bool = typer.Option(True, "--last/--no-last", help="Inspect the last stored recall trace"),
) -> None:
    if not last:
        raise typer.BadParameter("Only --last is currently supported.")
    _, storage, _, *_ = build_services()
    raw = storage.get_runtime_state("last_recall_trace") or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"raw": raw}
    matches = _collect_selection_matches(payload, memory_id)
    console.print_json(json.dumps({"memory_id": memory_id, "match_count": len(matches), "matches": matches}, ensure_ascii=False, indent=2))


@memory_app.command("trace")
def memory_trace(
    task: str = typer.Option(...),
    task_type: str = typer.Option("design"),
    temperature: float = typer.Option(0.5, min=0.0, max=1.0),
    max_tokens: int = typer.Option(1800),
    rerank_mode: str = typer.Option("rule", help="Reserved for node rerank compatibility; evidence ranking is now evidence-first"),
) -> None:
    runtime = build_runtime()
    storage = runtime.services.storage
    result = runtime.run_query(
        query=task,
        task_type=task_type,
        temperature=temperature,
        max_tokens=max_tokens,
        evidence_top_k=8,
        support_top_k=4,
        object_top_k=4,
        cognitive_top_k=4,
    )
    evidence = result["evidence"]
    completion = result["completion"]
    cognitive = result["cognitive"]
    bundle = result["bundle"]

    storage.log_activation(
        {
            "task_id": f"task_{abs(hash(task + now_iso()))}",
            "task_type": task_type,
            "main_nodes": [n["id"] for n in completion.evidence_nodes],
            "reflected_nodes": [n["id"] for n in cognitive.relevant_experience],
            "refracted_nodes": [n["id"] for n in cognitive.creative_reflections],
            "final_used_nodes": [
                x["id"]
                for x in (
                    bundle.core_evidence
                    + bundle.evidence_objects
                    + bundle.supporting_context
                    + bundle.relevant_experience
                    + bundle.creative_reflections
                )
                if x.get("id")
            ],
            "token_cost_input": bundle.debug["estimated_input_tokens"],
            "token_cost_output": 0,
            "quality_feedback": None,
            "created_at": now_iso(),
        }
    )
    console.print_json(json.dumps(asdict(bundle), ensure_ascii=False, indent=2))


@creative_app.command("reflect")
def creative_reflect(
    task: str = typer.Option(...),
    task_type: str = typer.Option("creative"),
    temperature: float = typer.Option(0.7, min=0.0, max=1.0),
) -> None:
    runtime = build_runtime()
    creative = runtime.services.creative
    evidence = runtime.retrieve_evidence(task, task_type=task_type, top_k=6)
    completion = runtime.complete_with_objects(task, evidence, support_top_k=2, object_top_k=2)
    cognitive = runtime.augment_cognitively(task, task_type=task_type, completion=completion, cognitive_top_k=8)
    reflected_nodes = cognitive.relevant_experience
    refracted_nodes = cognitive.creative_reflections
    alternative_paths = cognitive.alternative_paths

    table = Table(title=f"Creative Reflection: {task}")
    table.add_column("bucket")
    table.add_column("id")
    table.add_column("score")
    table.add_column("summary")

    for node in creative.annotate(reflected_nodes, "Relevant Experience"):
        table.add_row(node["bucket"], node["id"], str(node.get("reflection_score")), (node.get("summary") or "")[:90])
    for node in creative.annotate(refracted_nodes, "Creative Reflection"):
        table.add_row(
            node["bucket"],
            node["id"],
            str(node.get("reflection_score") or node.get("refraction_score")),
            (node.get("summary") or "")[:90],
        )
    console.print(table)
    if alternative_paths:
        alt_table = Table(title="Alternative Paths")
        alt_table.add_column("beam")
        alt_table.add_column("role")
        alt_table.add_column("score")
        alt_table.add_column("hops")
        alt_table.add_column("summary")
        for path in alternative_paths:
            alt_table.add_row(
                str(path.get("beam_type", "")),
                str(path.get("path_role", "")),
                str(path.get("score", "")),
                str(path.get("hop_count", "")),
                (path.get("summary") or "")[:100],
            )
        console.print(alt_table)


@memory_app.command("audit")
def memory_audit(last: bool = typer.Option(False, "--last", help="Show latest memory-runtime recall audit")) -> None:
    if last:
        latest_audit = Path.cwd() / "data" / "projections" / "latest_recall_audit.json"
        if latest_audit.exists():
            console.print_json(latest_audit.read_text(encoding="utf-8"))
            return
        console.print_json(json.dumps({"status": "empty", "error": "latest_recall_audit_not_found"}, ensure_ascii=False))
        return
    _, _, _, manager, _, _, _, _, auditor, _, _, _, _ = build_services()
    report = auditor.audit()
    console.print_json(json.dumps(report, ensure_ascii=False, indent=2))

    table = Table(title="Shell Capacity")
    table.add_column("shell")
    table.add_column("name")
    table.add_column("count")
    table.add_column("max_items_hint")
    table.add_column("utilization")
    for row in manager.capacity_report():
        table.add_row(str(row["shell"]), row["name"], str(row["count"]), str(row["max_items_hint"]), str(row["utilization"]))
    console.print(table)


@memory_app.command("map")
def memory_map(zone: str = typer.Option(..., "--zone"), out: Optional[str] = typer.Option(None, "--out")) -> None:
    _, storage, _, *_ = build_services()
    nodes = storage.export_zone_nodes(zone)
    export_path = Path(out) if out else storage.config.export_dir / f"{zone}_map.json"
    payload = [
        {
            "id": n["id"],
            "shell": n["shell"],
            "sector": n["sector"],
            "zone": n["zone"],
            "cell": n["cell"],
            "theta": n["theta"],
            "phi": n["phi"],
            "summary": n["summary"],
        }
        for n in nodes
    ]
    export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[green]Exported map[/green] -> {export_path}")


@memory_state_app.command("show")
def memory_state_show(
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    json_output: bool = typer.Option(False, "--json", help="Render JSON instead of Markdown"),
) -> None:
    state = load_project_state(Path.cwd(), project)
    if json_output:
        console.print_json(json.dumps(asdict(state), ensure_ascii=False, indent=2))
        return
    print(render_project_state_markdown(state))


@memory_state_app.command("update")
def memory_state_update(
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    from_latest_memories: bool = typer.Option(False, "--from-latest-memories", help="Update from registry/latest artifacts"),
) -> None:
    state = update_project_state_from_registry(Path.cwd(), project) if from_latest_memories else load_project_state(Path.cwd(), project)
    path = save_project_state(Path.cwd(), state)
    print(f"[green]Saved project state:[/green] {path}")


@memory_state_app.command("set-goal")
def memory_state_set_goal(
    goal: str = typer.Option(..., "--goal", help="Current project goal"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    state = load_project_state(Path.cwd(), project)
    state.current_goal = goal
    path = save_project_state(Path.cwd(), state)
    print(f"[green]Saved project state:[/green] {path}")


@memory_state_app.command("add-constraint")
def memory_state_add_constraint(
    constraint: str = typer.Option(..., "--constraint", help="Constraint to pin"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    payload = write_constraint(Path.cwd(), project, constraint, source="cli")
    print(f"[green]Wrote constraint memory:[/green] {payload['memory_id']}")


@memory_app.command("summarize-session")
def memory_summarize_session(
    session_log: Path = typer.Argument(..., help="Session log path"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    text = session_log.read_text(encoding="utf-8", errors="replace")
    summary = "\n".join(line.strip() for line in text.splitlines() if line.strip())[:4000]
    payload = write_conversation_summary(
        Path.cwd(),
        project,
        summary,
        source=str(session_log),
        metadata={"bytes": session_log.stat().st_size},
    )
    print(f"[green]Wrote conversation summary:[/green] {payload['memory_id']}")


@memory_app.command("context")
def memory_context(
    task: str = typer.Argument(..., help="Task objective"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    mode: str = typer.Option("codex", "--mode", help="Context mode"),
    max_tokens: int = typer.Option(8000, "--max-tokens", help="Approximate token budget"),
) -> None:
    print(compile_context_packet(Path.cwd(), task=task, project=project, mode=mode, max_tokens=max_tokens))


@memory_conflicts_app.command("list")
def memory_conflicts_list(
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    refresh: bool = typer.Option(True, "--refresh/--no-refresh", help="Refresh conflicts from registry before listing"),
) -> None:
    conflicts = detect_conflicts(Path.cwd(), project) if refresh else load_conflicts(Path.cwd())
    if refresh:
        write_conflict_report(Path.cwd(), conflicts)
    table = Table(title=f"Memory Conflicts: {project}")
    table.add_column("id")
    table.add_column("dataset")
    table.add_column("type")
    table.add_column("winner")
    table.add_column("reason")
    for conflict in conflicts:
        table.add_row(
            conflict.conflict_id,
            conflict.dataset,
            conflict.conflict_type,
            str(conflict.recommended_winner or ""),
            conflict.reason[:90],
        )
    console.print(table)


@memory_conflicts_app.command("show")
def memory_conflicts_show(conflict_id: str = typer.Argument(..., help="Conflict id")) -> None:
    conflicts = load_conflicts(Path.cwd())
    for conflict in conflicts:
        if conflict.conflict_id == conflict_id:
            console.print_json(json.dumps(asdict(conflict), ensure_ascii=False, indent=2))
            return
    raise typer.Exit(1)


@memory_conflicts_app.command("resolve")
def memory_conflicts_resolve(
    conflict_id: str = typer.Argument(..., help="Conflict id"),
    winner: str = typer.Option(..., "--winner", help="Winning memory/run id"),
) -> None:
    row = append_lifecycle_action(Path.cwd(), "resolve_conflict", {"conflict_id": conflict_id, "winner": winner})
    print(f"[green]Recorded conflict resolution:[/green] {row['action_id']}")


@memory_conflicts_app.command("mark-disputed")
def memory_conflicts_mark_disputed(memory_id: str = typer.Argument(..., help="Memory/run id")) -> None:
    row = append_lifecycle_action(Path.cwd(), "mark_disputed", {"memory_id": memory_id})
    print(f"[green]Marked disputed:[/green] {row['action_id']}")


@memory_lifecycle_app.command("review")
def memory_lifecycle_review(project: str = typer.Option("DysonSpherain", "--project", help="Project name")) -> None:
    conflicts = detect_conflicts(Path.cwd(), project)
    path = write_conflict_report(Path.cwd(), conflicts)
    print(f"[green]Saved lifecycle conflict review:[/green] {path}")


@memory_app.command("supersede")
def memory_supersede(old_memory_id: str, new_memory_id: str) -> None:
    row = append_lifecycle_action(Path.cwd(), "supersede", {"old_memory_id": old_memory_id, "new_memory_id": new_memory_id})
    print(f"[green]Recorded supersession:[/green] {row['action_id']}")


@memory_app.command("pin")
def memory_pin(memory_id: str) -> None:
    row = append_lifecycle_action(Path.cwd(), "pin", {"memory_id": memory_id})
    print(f"[green]Pinned memory:[/green] {row['action_id']}")


@memory_app.command("archive")
def memory_archive(
    memory_id: str,
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    try:
        payload = archive_memory(Path.cwd(), project, memory_id)
    except KeyError:
        payload = None
    row = append_lifecycle_action(Path.cwd(), "archive", {"memory_id": memory_id})
    if payload is None:
        print(f"[green]Recorded archive action:[/green] {row['action_id']}")
        return
    print(f"[green]Archived memory:[/green] {memory_id}")


@memory_app.command("merge")
def memory_merge(memory_id_a: str, memory_id_b: str) -> None:
    row = append_lifecycle_action(Path.cwd(), "merge", {"memory_id_a": memory_id_a, "memory_id_b": memory_id_b})
    print(f"[green]Recorded merge:[/green] {row['action_id']}")


@agent_app.command("postrun")
def agent_postrun(
    run_log: Path = typer.Argument(..., help="Agent run log path"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Existing execution run id to update"),
    task: Optional[str] = typer.Option(None, "--task", help="Task description for a new or updated ledger run"),
    status: str = typer.Option("completed", "--status", help="Ledger status to record"),
    next_action: Optional[str] = typer.Option(None, "--next-action", help="Next safe action for resume packets"),
    changed_files: Optional[str] = typer.Option(None, "--changed-files", help="Comma-separated changed files"),
    artifacts: Optional[str] = typer.Option(None, "--artifacts", help="Comma-separated artifact paths"),
    tests_run: Optional[str] = typer.Option(None, "--tests-run", help="Comma-separated test commands"),
    benchmarks_run: Optional[str] = typer.Option(None, "--benchmarks-run", help="Comma-separated benchmark commands"),
    errors: Optional[str] = typer.Option(None, "--errors", help="Comma-separated error summaries"),
) -> None:
    text = run_log.read_text(encoding="utf-8", errors="replace")
    summary = "\n".join(line.strip() for line in text.splitlines() if line.strip())[:4000]
    payload = write_agent_run_summary(
        Path.cwd(),
        project,
        summary,
        source=str(run_log),
        metadata={"bytes": run_log.stat().st_size},
    )
    ledger_run = record_postrun(
        Path.cwd(),
        project=project,
        summary=summary,
        source=str(run_log),
        run_id=run_id,
        task=task,
        status=status,
        next_action=next_action,
        changed_files=_split_csv(changed_files),
        artifacts=_split_csv(artifacts),
        tests_run=_split_csv(tests_run),
        benchmarks_run=_split_csv(benchmarks_run),
        errors=_split_csv(errors),
    )
    print(f"[green]Wrote agent run summary:[/green] {payload['memory_id']}")
    print(f"[green]Updated execution ledger:[/green] {ledger_run.run_id}")


@agent_app.command("preflight")
def agent_preflight(
    task: str = typer.Argument(..., help="Task objective"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    max_tokens: int = typer.Option(8000, "--max-tokens", help="Approximate token budget"),
) -> None:
    print(compile_context_packet(Path.cwd(), task=task, project=project, mode="codex", max_tokens=max_tokens))


@agent_app.command("status")
def agent_status(project: str = typer.Option("DysonSpherain", "--project", help="Project name")) -> None:
    print(compile_context_packet(Path.cwd(), task="Report current agent/project status.", project=project, mode="project", max_tokens=3000))


@agent_ledger_app.command("start")
def agent_ledger_start(
    task: str = typer.Argument(..., help="Task objective"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    run = create_execution_run(Path.cwd(), project=project, task=task, status="running")
    print(f"[green]Started execution run:[/green] {run.run_id}")


@agent_ledger_app.command("list")
def agent_ledger_list(
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    json_output: bool = typer.Option(False, "--json", help="Render JSON instead of Markdown"),
) -> None:
    runs = load_execution_runs(Path.cwd(), project)
    if json_output:
        console.print_json(json.dumps([asdict(run) for run in runs], ensure_ascii=False, indent=2))
        return
    print(render_ledger_list(runs))


@agent_ledger_app.command("show")
def agent_ledger_show(
    run_id: str = typer.Argument(..., help="Execution run id"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    run = get_execution_run(Path.cwd(), project, run_id)
    if run is None:
        raise typer.Exit(1)
    console.print_json(json.dumps(asdict(run), ensure_ascii=False, indent=2))


@agent_ledger_app.command("resume-packet")
def agent_ledger_resume_packet(
    run_id: str = typer.Argument(..., help="Execution run id"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    try:
        print(render_resume_packet(Path.cwd(), project, run_id))
    except KeyError:
        raise typer.Exit(1) from None


@app.command("recall")
def dyson_recall_command(
    query: str = typer.Option(..., "--query", help="Query/prompt to recall context for"),
    cwd: str = typer.Option(".", "--cwd", help="Project root"),
    budget: int = typer.Option(1600, "--budget", help="Context token budget"),
    format: str = typer.Option("json", "--format", help="json or markdown"),
    task_type: str = typer.Option("unknown", "--task-type"),
) -> None:
    from dysonspherain.memory_os.recall_service import RecallRequest, recall

    result = recall(RecallRequest(query=query, cwd=cwd, token_budget=budget, task_type=task_type))
    if format == "markdown":
        print(result.rendered_context)
    else:
        console.print_json(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


@app.command("context-pack")
def dyson_context_pack_command(
    query: str = typer.Option(..., "--query", help="Query/prompt to pack context for"),
    cwd: str = typer.Option(".", "--cwd", help="Project root"),
    budget: int = typer.Option(1600, "--budget", help="Context token budget"),
    format: str = typer.Option("markdown", "--format", help="markdown or json"),
    memory_ids: str = typer.Option("", "--memory-ids", help="Comma-separated memory ids to pack explicitly"),
    sections: str = typer.Option("", "--sections", help="Comma-separated context sections to include"),
) -> None:
    from dysonspherain.adapters.mcp_server import call_tool

    result = call_tool(
        "dyson_context_pack",
        {
            "query": query,
            "cwd": cwd,
            "token_budget": budget,
            "format": format,
            "memory_ids": _split_csv(memory_ids),
            "sections": _split_csv(sections),
        },
    )
    if format == "markdown":
        print(result["rendered_context"])
    else:
        console.print_json(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("write-memory")
def dyson_write_memory_command(summary_file: Path = typer.Option(..., "--summary-file", help="JSON summary file to write back")) -> None:
    from dysonspherain.memory_os.write_service import WriteMemoryRequest, write_memory

    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    for list_key in ("files_changed", "commands_run", "tests_run", "benchmark_results", "failures", "next_actions"):
        payload.setdefault(list_key, [])
    payload.setdefault("cwd", str(Path.cwd()))
    payload.setdefault("session_id", "")
    payload.setdefault("task_goal", "")
    payload.setdefault("summary", "")
    payload.setdefault("source", "manual")
    result = write_memory(WriteMemoryRequest(**{key: payload.get(key) for key in WriteMemoryRequest.__dataclass_fields__}))
    console.print_json(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


@app.command("project-state")
def dyson_project_state_command(
    cwd: str = typer.Option(".", "--cwd", help="Project root"),
    budget: int = typer.Option(1200, "--budget", help="Token budget"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    from dysonspherain.memory_os.project_state import ProjectStateRequest, get_project_state

    result = get_project_state(ProjectStateRequest(cwd=cwd, token_budget=budget, project=project))
    console.print_json(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("token-economy-eval")
def dyson_token_economy_eval_command(
    query: str = typer.Option(..., "--query", help="Current query"),
    context_file: Path = typer.Option(..., "--context-file", help="Candidate context file"),
    baseline_context_tokens: int = typer.Option(0, "--baseline-context-tokens"),
    budget: int = typer.Option(1600, "--budget"),
    task_type: str = typer.Option("unknown", "--task-type"),
) -> None:
    from dysonspherain.token_economy.evaluator import evaluate

    result = evaluate(
        query=query,
        candidate_context=context_file.read_text(encoding="utf-8"),
        baseline_context_tokens=baseline_context_tokens,
        token_budget=budget,
        task_type=task_type,
    )
    console.print_json(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


@app.command("record")
def product_record_command(
    source: str = typer.Option("manual", "--source", help="shell, error, code-diff, benchmark, manual"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project id"),
    text: Optional[str] = typer.Option(None, "--text", help="Inline raw text"),
    file: Optional[Path] = typer.Option(None, "--file", help="Raw source file"),
    command: Optional[str] = typer.Option(None, "--command", help="Shell command"),
    capture_output: bool = typer.Option(False, "--capture-output", help="Run command and capture stdout/stderr"),
    artifact: Optional[Path] = typer.Option(None, "--artifact", help="Artifact path"),
    session_id: Optional[str] = typer.Option(None, "--session-id"),
    task: Optional[str] = typer.Option(None, "--task"),
    allow: list[str] = typer.Option(None, "--allow", help="Allowlist glob; may be repeated"),
    deny: list[str] = typer.Option(None, "--deny", help="Denylist glob; may be repeated"),
) -> None:
    from dysonspherain.product import record_source

    payload = record_source(
        Path.cwd(),
        project_id=project,
        source=source,
        text=text,
        file=file,
        command=command,
        capture_output=capture_output,
        artifact=artifact,
        session_id=session_id,
        task_id=task,
        allowlist=allow or [],
        denylist=deny or [],
    )
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@app.command("search")
def product_search_command(
    query: str = typer.Argument("", help="Search query"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project id"),
    limit: int = typer.Option(10, "--limit", min=1, max=100),
    include_invalid: bool = typer.Option(False, "--include-invalid"),
) -> None:
    from dysonspherain.product import search

    console.print_json(json.dumps(search(Path.cwd(), project_id=project, query=query, limit=limit, include_invalid=include_invalid), ensure_ascii=False, indent=2, sort_keys=True))


@app.command("retrieve")
def product_retrieve_command(
    query: str = typer.Argument(..., help="Retrieval query"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project id"),
    limit: int = typer.Option(10, "--limit", min=1, max=100),
    show_audit: bool = typer.Option(False, "--show-audit", help="Include candidate admission trace"),
    context_pack: bool = typer.Option(False, "--context-pack", help="Generate context pack"),
    max_tokens: int = typer.Option(2000, "--max-tokens", min=100),
    task_type: Optional[str] = typer.Option(None, "--task-type", help="Optional route task type"),
    context_format: str = typer.Option("markdown", "--format", help="Context pack format: markdown/json/yaml/text"),
    sections: Optional[str] = typer.Option(None, "--sections", help="Comma-separated context sections"),
    section_budget: Optional[str] = typer.Option(None, "--section-budget", help="JSON object of section token budgets"),
    agent_role: str = typer.Option("coder", "--agent-role"),
    include_raw_quotes: bool = typer.Option(False, "--include-raw-quotes"),
    include_debug_trace: bool = typer.Option(False, "--include-debug-trace"),
) -> None:
    from dysonspherain.product import retrieve

    console.print_json(
        json.dumps(
            retrieve(
                Path.cwd(),
                project_id=project,
                query=query,
                limit=limit,
                show_audit=show_audit,
                context_pack=context_pack,
                max_tokens=max_tokens,
                task_type=task_type,
                context_format=context_format,
                sections=_split_csv(sections),
                section_budget=_json_object_option(section_budget),
                agent_role=agent_role,
                include_raw_quotes=include_raw_quotes,
                include_debug_trace=include_debug_trace,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@app.command("wake")
def product_wake_command(
    task: str = typer.Option("", "--task", help="Task or mission query"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project id"),
    max_tokens: int = typer.Option(8000, "--max-tokens", min=100),
    format: str = typer.Option("markdown", "--format", help="markdown/json/yaml/text"),
    agent_role: str = typer.Option("coder", "--agent-role"),
    task_type: Optional[str] = typer.Option(None, "--task-type", help="Optional route task type"),
    sections: Optional[str] = typer.Option(None, "--sections", help="Comma-separated context sections"),
    section_budget: Optional[str] = typer.Option(None, "--section-budget", help="JSON object of section token budgets"),
    include_raw_quotes: bool = typer.Option(False, "--include-raw-quotes"),
    include_debug_trace: bool = typer.Option(False, "--include-debug-trace"),
) -> None:
    from dysonspherain.product import create_context_pack

    payload = create_context_pack(
        Path.cwd(),
        project_id=project,
        query=task,
        max_tokens=max_tokens,
        agent_role=agent_role,
        task_type=task_type,
        sections=_split_csv(sections),
        section_budget=_json_object_option(section_budget),
        include_raw_quotes=include_raw_quotes,
        include_debug_trace=include_debug_trace,
        fmt=format,
    )
    if format == "json":
        console.print_json(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(payload["rendered"])


@app.command("inspect")
def product_inspect_command(
    capsule_id: str = typer.Argument(..., help="Capsule id"),
    project: Optional[str] = typer.Option(None, "--project", help="Project id"),
) -> None:
    from dysonspherain.product import get_capsule

    console.print_json(json.dumps(get_capsule(Path.cwd(), capsule_id, project_id=project), ensure_ascii=False, indent=2, sort_keys=True))


@app.command("forget")
def product_forget_command(
    capsule_id: Optional[str] = typer.Option(None, "--capsule-id", help="Capsule id"),
    project: Optional[str] = typer.Option(None, "--project", help="Project id"),
    hard: bool = typer.Option(False, "--hard", help="Delete instead of tombstone"),
    before: Optional[str] = typer.Option(None, "--before", help="Forget active capsules before this ISO timestamp"),
    keep_last: Optional[int] = typer.Option(None, "--keep-last", help="Retention mode: keep newest N active capsules"),
) -> None:
    from dysonspherain.product import apply_retention, forget_before, forget_capsule

    active_project = project or "DysonSpherain"
    if before:
        payload = forget_before(Path.cwd(), project_id=active_project, before=before, hard=hard)
    elif keep_last is not None:
        payload = apply_retention(Path.cwd(), project_id=active_project, keep_last=keep_last, hard=hard)
    elif capsule_id:
        payload = forget_capsule(Path.cwd(), capsule_id=capsule_id, project_id=project, hard=hard)
    else:
        raise typer.BadParameter("Provide --capsule-id, --before, or --keep-last.")
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@app.command("export")
def product_export_command(
    project: str = typer.Option("DysonSpherain", "--project", help="Project id"),
    format: str = typer.Option("json", "--format", help="json or markdown"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output path"),
) -> None:
    from dysonspherain.product import export_project

    console.print_json(json.dumps(export_project(Path.cwd(), project_id=project, fmt=format, output=output), ensure_ascii=False, indent=2, sort_keys=True))


@runtime_app.command("before-task")
def runtime_before_task_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    task: str = typer.Option(..., "--task"),
    max_tokens: int = typer.Option(6000, "--max-tokens", min=100),
    agent_role: str = typer.Option("coder", "--agent-role"),
) -> None:
    from dysonspherain.product import runtime_event

    console.print_json(json.dumps(runtime_event(Path.cwd(), project_id=project, event_type="before_task", payload={"task": task, "agent_role": agent_role}, max_tokens=max_tokens), ensure_ascii=False, indent=2, sort_keys=True))


@runtime_app.command("on-error")
def runtime_on_error_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    error_file: Path = typer.Option(..., "--error-file"),
    max_tokens: int = typer.Option(4000, "--max-tokens", min=100),
) -> None:
    from dysonspherain.product import runtime_event

    payload = {"error": error_file.read_text(encoding="utf-8", errors="replace"), "error_file": str(error_file)}
    console.print_json(json.dumps(runtime_event(Path.cwd(), project_id=project, event_type="on_error", payload=payload, max_tokens=max_tokens), ensure_ascii=False, indent=2, sort_keys=True))


@runtime_app.command("during-task")
def runtime_during_task_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    summary: str = typer.Option(..., "--summary"),
    task_id: Optional[str] = typer.Option(None, "--task-id"),
    max_tokens: int = typer.Option(3000, "--max-tokens", min=100),
) -> None:
    from dysonspherain.product import runtime_event

    payload = {"summary": summary, "task_id": task_id}
    console.print_json(json.dumps(runtime_event(Path.cwd(), project_id=project, event_type="during_task", payload=payload, max_tokens=max_tokens), ensure_ascii=False, indent=2, sort_keys=True))


@runtime_app.command("after-task")
def runtime_after_task_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    summary_file: Optional[Path] = typer.Option(None, "--summary-file"),
    changed_files: list[str] = typer.Option(None, "--changed-files"),
    summary: Optional[str] = typer.Option(None, "--summary"),
    max_tokens: int = typer.Option(3000, "--max-tokens", min=100),
) -> None:
    from dysonspherain.product import runtime_event

    text = summary or (summary_file.read_text(encoding="utf-8", errors="replace") if summary_file else "after task checkpoint")
    payload = {"summary": text, "summary_file": str(summary_file) if summary_file else None, "changed_files": changed_files or []}
    console.print_json(json.dumps(runtime_event(Path.cwd(), project_id=project, event_type="after_task", payload=payload, max_tokens=max_tokens), ensure_ascii=False, indent=2, sort_keys=True))


@runtime_app.command("pre-compact")
def runtime_pre_compact_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    session_id: Optional[str] = typer.Option(None, "--session-id"),
    max_tokens: int = typer.Option(3000, "--max-tokens", min=100),
) -> None:
    from dysonspherain.product import runtime_event

    console.print_json(json.dumps(runtime_event(Path.cwd(), project_id=project, event_type="pre_compact", payload={"session_id": session_id, "task": "prepare safe compaction context"}, max_tokens=max_tokens), ensure_ascii=False, indent=2, sort_keys=True))


@runtime_app.command("before-benchmark")
def runtime_before_benchmark_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    benchmark: str = typer.Option(..., "--benchmark"),
    profile: Optional[str] = typer.Option(None, "--profile"),
    max_tokens: int = typer.Option(4000, "--max-tokens", min=100),
) -> None:
    from dysonspherain.product import runtime_event

    payload = {"task": f"prepare benchmark {benchmark}", "benchmark": benchmark, "profile": profile}
    console.print_json(json.dumps(runtime_event(Path.cwd(), project_id=project, event_type="before_benchmark", payload=payload, max_tokens=max_tokens), ensure_ascii=False, indent=2, sort_keys=True))


@runtime_app.command("after-benchmark")
def runtime_after_benchmark_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    metrics: Path = typer.Option(..., "--metrics"),
    artifacts: Optional[Path] = typer.Option(None, "--artifacts"),
    max_tokens: int = typer.Option(3000, "--max-tokens", min=100),
) -> None:
    from dysonspherain.product import benchmark_record, runtime_event

    run = benchmark_record(Path.cwd(), project_id=project, artifact=metrics)
    payload = {"task": "benchmark completed", "metrics": str(metrics), "artifacts": str(artifacts) if artifacts else None, "run": run}
    console.print_json(json.dumps(runtime_event(Path.cwd(), project_id=project, event_type="after_benchmark", payload=payload, max_tokens=max_tokens), ensure_ascii=False, indent=2, sort_keys=True))


@runtime_app.command("before-commit")
def runtime_before_commit_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    summary: str = typer.Option("prepare commit", "--summary"),
    changed_files: list[str] = typer.Option(None, "--changed-files"),
    max_tokens: int = typer.Option(3000, "--max-tokens", min=100),
) -> None:
    from dysonspherain.product import runtime_event

    payload = {"summary": summary, "changed_files": changed_files or []}
    console.print_json(json.dumps(runtime_event(Path.cwd(), project_id=project, event_type="before_commit", payload=payload, max_tokens=max_tokens), ensure_ascii=False, indent=2, sort_keys=True))


@runtime_app.command("after-commit")
def runtime_after_commit_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    commit: Optional[str] = typer.Option(None, "--commit"),
    summary: str = typer.Option("commit completed", "--summary"),
    max_tokens: int = typer.Option(3000, "--max-tokens", min=100),
) -> None:
    from dysonspherain.product import runtime_event

    payload = {"summary": summary, "commit": commit}
    console.print_json(json.dumps(runtime_event(Path.cwd(), project_id=project, event_type="after_commit", payload=payload, max_tokens=max_tokens), ensure_ascii=False, indent=2, sort_keys=True))


@runtime_app.command("manual-checkpoint")
def runtime_manual_checkpoint_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    summary: str = typer.Option(..., "--summary"),
    task_id: Optional[str] = typer.Option(None, "--task-id"),
    max_tokens: int = typer.Option(3000, "--max-tokens", min=100),
) -> None:
    from dysonspherain.product import runtime_event

    payload = {"summary": summary, "task_id": task_id}
    console.print_json(json.dumps(runtime_event(Path.cwd(), project_id=project, event_type="manual_checkpoint", payload=payload, max_tokens=max_tokens), ensure_ascii=False, indent=2, sort_keys=True))


@benchmark_lab_app.command("record")
def benchmark_lab_record_command(
    artifact: Path = typer.Option(..., "--artifact", help="metrics.json or artifact directory"),
    project: str = typer.Option("DysonSpherain", "--project"),
    benchmark: Optional[str] = typer.Option(None, "--benchmark"),
    status: str = typer.Option("success", "--status"),
) -> None:
    from dysonspherain.product import benchmark_record

    console.print_json(json.dumps(benchmark_record(Path.cwd(), project_id=project, artifact=artifact, benchmark=benchmark, status=status), ensure_ascii=False, indent=2, sort_keys=True))


@benchmark_lab_app.command("compare")
def benchmark_lab_compare_command(
    current: Path = typer.Option(..., "--current"),
    baseline: Path = typer.Option(..., "--baseline"),
    project: str = typer.Option("DysonSpherain", "--project"),
) -> None:
    from dysonspherain.product import benchmark_compare

    console.print_json(json.dumps(benchmark_compare(Path.cwd(), project_id=project, current=current, baseline=baseline), ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("rebuild")
def product_index_rebuild_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    include_archived: bool = typer.Option(False, "--include-archived", help="Also rebuild archived capsule embeddings"),
    backend: Optional[str] = typer.Option(None, "--backend", help="Override embedding backend for this rebuild"),
    model: Optional[str] = typer.Option(None, "--model", help="Optional embedding model name"),
) -> None:
    from dysonspherain.product import doctor, init_product_store, rebuild_product_embeddings, rebuild_product_vector_index

    init_product_store(Path.cwd(), project_id=project)
    rebuild = rebuild_product_embeddings(Path.cwd(), project_id=project, include_archived=include_archived, backend=backend, model=model)
    vector_rebuild = rebuild_product_vector_index(Path.cwd(), project_id=project)
    console.print_json(json.dumps({"status": "ok", "action": "rebuild", "embedding_rebuild": rebuild, "vector_rebuild": vector_rebuild, "doctor": doctor(Path.cwd(), project_id=project)}, ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("verify")
def product_index_verify_command(project: str = typer.Option("DysonSpherain", "--project")) -> None:
    from dysonspherain.product import doctor

    console.print_json(json.dumps(doctor(Path.cwd(), project_id=project), ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("repair")
def product_index_repair_command(project: str = typer.Option("DysonSpherain", "--project")) -> None:
    from dysonspherain.product import doctor, init_product_store, maintenance_suggestions, rebuild_product_embeddings

    init_product_store(Path.cwd(), project_id=project)
    rebuild = rebuild_product_embeddings(Path.cwd(), project_id=project)
    suggestions = maintenance_suggestions(Path.cwd(), project_id=project)
    console.print_json(json.dumps({"status": "ok", "action": "repair", "embedding_rebuild": rebuild, "maintenance": suggestions, "doctor": doctor(Path.cwd(), project_id=project)}, ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("maintenance")
def product_index_maintenance_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    apply: Optional[str] = typer.Option(None, "--apply", help="Apply a maintenance suggestion id"),
    dismiss: Optional[str] = typer.Option(None, "--dismiss", help="Dismiss a maintenance suggestion id"),
    canonical_id: Optional[str] = typer.Option(None, "--canonical-id", help="Canonical capsule id for duplicate merge apply"),
    reason: Optional[str] = typer.Option(None, "--reason", help="Dismiss reason"),
    limit: int = typer.Option(100, "--limit", min=1, max=500),
) -> None:
    from dysonspherain.product import apply_maintenance_suggestion, dismiss_maintenance_suggestion, maintenance_suggestions

    if apply and dismiss:
        raise typer.BadParameter("Use only one of --apply or --dismiss.")
    if apply:
        payload = apply_maintenance_suggestion(Path.cwd(), project_id=project, suggestion_id=apply, canonical_id=canonical_id)
    elif dismiss:
        payload = dismiss_maintenance_suggestion(Path.cwd(), project_id=project, suggestion_id=dismiss, reason=reason)
    else:
        payload = maintenance_suggestions(Path.cwd(), project_id=project, limit=limit)
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("embedding-backends")
def product_index_embedding_backends_command(project: str = typer.Option("DysonSpherain", "--project")) -> None:
    from dysonspherain.product import product_embedding_backends

    console.print_json(json.dumps(product_embedding_backends(Path.cwd()), ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("vector-backends")
def product_index_vector_backends_command(project: str = typer.Option("DysonSpherain", "--project")) -> None:
    from dysonspherain.product import product_vector_backends

    console.print_json(json.dumps(product_vector_backends(Path.cwd()), ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("configure-vector")
def product_index_configure_vector_command(
    backend: str = typer.Argument(..., help="sqlite_inline or chroma"),
    path: Optional[Path] = typer.Option(None, "--path", help="Chroma index directory"),
    collection: str = typer.Option("product_capsules", "--collection", help="Product vector collection name"),
    allow_unavailable: bool = typer.Option(False, "--allow-unavailable", help="Write config even when optional backend dependency is missing"),
) -> None:
    from dysonspherain.product import configure_product_vector_backend

    console.print_json(json.dumps(configure_product_vector_backend(Path.cwd(), backend=backend, path=path, collection=collection, allow_unavailable=allow_unavailable), ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("rebuild-vector")
def product_index_rebuild_vector_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    backend: Optional[str] = typer.Option(None, "--backend", help="Override vector backend before rebuilding"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum capsules to index"),
) -> None:
    from dysonspherain.product import rebuild_product_vector_index

    console.print_json(json.dumps(rebuild_product_vector_index(Path.cwd(), project_id=project, backend=backend, limit=limit), ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("configure-embedding")
def product_index_configure_embedding_command(
    backend: str = typer.Argument(..., help="local_hash_embedding or sentence_transformers"),
    model: Optional[str] = typer.Option(None, "--model", help="Optional model name for semantic backends"),
    allow_unavailable: bool = typer.Option(False, "--allow-unavailable", help="Write config even when optional backend dependency is missing"),
) -> None:
    from dysonspherain.product import configure_embedding_backend

    console.print_json(json.dumps(configure_embedding_backend(Path.cwd(), backend=backend, model=model, allow_unavailable=allow_unavailable), ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("configure-encryption")
def product_index_configure_encryption_command(
    provider: str = typer.Argument(..., help="external_or_os_managed or sqlcipher"),
    key_env: str = typer.Option("DYSON_MEMORY_SQLCIPHER_KEY", "--key-env", help="Environment variable containing SQLCipher key"),
    scope: str = typer.Option("product_sqlite", "--scope", help="Encryption scope description"),
    allow_unavailable: bool = typer.Option(False, "--allow-unavailable", help="Write marker even when SQLCipher/key is unavailable"),
) -> None:
    from dysonspherain.product import configure_encryption

    console.print_json(json.dumps(configure_encryption(Path.cwd(), provider=provider, key_env=key_env, scope=scope, allow_unavailable=allow_unavailable), ensure_ascii=False, indent=2, sort_keys=True))


@product_index_app.command("migrate-sqlcipher")
def product_index_migrate_sqlcipher_command(
    key_env: str = typer.Option("DYSON_MEMORY_SQLCIPHER_KEY", "--key-env", help="Environment variable containing SQLCipher key"),
    output: Optional[Path] = typer.Option(None, "--output", help="Encrypted DB output path"),
    replace: bool = typer.Option(False, "--replace", help="Replace the active DB and keep a plaintext backup"),
) -> None:
    from dysonspherain.product import migrate_product_db_to_sqlcipher

    console.print_json(json.dumps(migrate_product_db_to_sqlcipher(Path.cwd(), key_env=key_env, output=output, replace=replace), ensure_ascii=False, indent=2, sort_keys=True))


def _import_text_files(paths: list[Path], *, project: str, source: str, session_id: Optional[str]) -> dict[str, object]:
    from dysonspherain.product import remember

    ids: list[str] = []
    for path in paths:
        if path.is_file():
            payload = remember(
                Path.cwd(),
                project_id=project,
                text=path.read_text(encoding="utf-8", errors="replace"),
                evidence_type="artifact",
                source_type=source,
                title=path.name,
                session_id=session_id,
                artifact_refs=[str(path)],
                metadata={"import_path": str(path), "import_source": source},
            )
            ids.append(payload["capsule_id"])
    return {"status": "ok", "imported_count": len(ids), "capsule_ids": ids}


@import_app.command("markdown")
def product_import_markdown(path: Path, project: str = typer.Option("DysonSpherain", "--project"), session_id: Optional[str] = typer.Option(None, "--session-id")) -> None:
    console.print_json(json.dumps(_import_text_files([path], project=project, source="markdown_import", session_id=session_id), ensure_ascii=False, indent=2, sort_keys=True))


@import_app.command("transcript")
def product_import_transcript(path: Path, project: str = typer.Option("DysonSpherain", "--project"), session_id: Optional[str] = typer.Option(None, "--session-id")) -> None:
    console.print_json(json.dumps(_import_text_files([path], project=project, source="transcript_import", session_id=session_id), ensure_ascii=False, indent=2, sort_keys=True))


@import_app.command("directory")
def product_import_directory(path: Path, project: str = typer.Option("DysonSpherain", "--project"), session_id: Optional[str] = typer.Option(None, "--session-id")) -> None:
    files = [item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in {".md", ".txt", ".json", ".jsonl", ".log"}]
    console.print_json(json.dumps(_import_text_files(files, project=project, source="directory_import", session_id=session_id), ensure_ascii=False, indent=2, sort_keys=True))


@app.command("ui")
def product_ui_command(
    project: str = typer.Option("DysonSpherain", "--project"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(37777, "--port"),
) -> None:
    from dysonspherain.adapters.daemon import run_server

    run_server(Path.cwd(), host=host, port=port, project=project)


@app.command("evaluate-token-economy")
def dyson_evaluate_token_economy_command(
    input: Optional[Path] = typer.Option(None, "--input", help="Queries JSONL"),
    benchmark_artifact_root: Optional[Path] = typer.Option(None, "--benchmark-artifact-root", help="Benchmark artifact root containing */metrics.json"),
    memory_db: Optional[Path] = typer.Option(None, "--memory-db", help="Memory DB path or runtime base directory for live context assembly"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name for memory-store baselines"),
    output: Path = typer.Option(Path("artifacts/token_economy"), "--output", help="Output artifact directory"),
    modes: str = typer.Option("conservative", "--modes"),
    baseline_types: str = typer.Option("full_history", "--baseline-types"),
    context_token_budget: str = typer.Option("1600", "--context-token-budget"),
    max_samples: int = typer.Option(0, "--max-samples"),
    tokenizer_model: str = typer.Option("cl100k_base", "--tokenizer-model"),
    tokenizer_strategy: str = typer.Option("auto", "--tokenizer-strategy"),
    tokenizer_calibration: Optional[Path] = typer.Option(None, "--tokenizer-calibration"),
    recent_k: int = typer.Option(20, "--recent-k"),
    allow_evidence_truncation: bool = typer.Option(False, "--allow-evidence-truncation", help="Allow truncating payload evidence when no live memory DB is provided"),
    smoke: bool = typer.Option(False, "--smoke"),
) -> None:
    from dysonspherain.evaluation.token_economy import main as token_economy_main

    argv = [
        "--output",
        str(output),
        "--modes",
        modes,
        "--baseline-types",
        baseline_types,
        "--context-token-budget",
        context_token_budget,
        "--max-samples",
        str(max_samples),
        "--tokenizer-model",
        tokenizer_model,
        "--tokenizer-strategy",
        tokenizer_strategy,
        "--recent-k",
        str(recent_k),
        "--project",
        project,
    ]
    if tokenizer_calibration:
        argv.extend(["--tokenizer-calibration", str(tokenizer_calibration)])
    if input:
        argv.extend(["--input", str(input)])
    if benchmark_artifact_root:
        argv.extend(["--benchmark-artifact-root", str(benchmark_artifact_root)])
    if memory_db:
        argv.extend(["--memory-db", str(memory_db)])
    if allow_evidence_truncation:
        argv.append("--allow-evidence-truncation")
    if smoke:
        argv.append("--smoke")
    token_economy_main(argv)


@app.command("evaluate-token-economy-smoke")
def dyson_evaluate_token_economy_smoke_command(
    samples: int = typer.Option(20, "--samples", min=1),
    output: Path = typer.Option(Path("artifacts/token_economy_smoke"), "--output"),
) -> None:
    from dysonspherain.evaluation.token_economy import main as token_economy_main

    token_economy_main(
        [
            "--smoke",
            "--output",
            str(output),
            "--max-samples",
            str(samples),
            "--modes",
            "off,conservative,exploratory,minimal",
            "--baseline-types",
            "full_history,naive_recent,manual_summary",
            "--context-token-budget",
            "800,1200,1600,2400",
        ]
    )


@app.command("calibrate-tokenizer")
def dyson_calibrate_tokenizer_command(
    input: Path = typer.Option(..., "--input", help="JSONL calibration samples"),
    output: Path = typer.Option(Path("artifacts/tokenizer_calibration.json"), "--output"),
    strategy: str = typer.Option("mixed_content_heuristic", "--strategy"),
) -> None:
    from dysonspherain.token_economy.tokenizer_calibration import calibrate

    console.print_json(json.dumps(calibrate(input, output, strategy=strategy), ensure_ascii=False, indent=2, sort_keys=True))


@app.command("token-economy-final-report")
def dyson_token_economy_final_report_command(
    token_economy_dir: Path = typer.Option(..., "--token-economy-dir", help="Directory containing token economy summary artifacts"),
    output: Optional[Path] = typer.Option(None, "--output", help="Final report path"),
    benchmark_rerun_status: str = typer.Option("", "--benchmark-rerun-status", help="Short status note for benchmark reruns"),
) -> None:
    from dysonspherain.token_economy.final_report import write_final_report

    path = write_final_report(token_economy_dir, output, benchmark_rerun_status=benchmark_rerun_status)
    print(f"[green]Saved token economy final report:[/green] {path}")


@adapters_app.command("install-codex-mcp")
def adapters_install_codex_mcp(project: str = typer.Option(".", "--project", help="Project root")) -> None:
    from dysonspherain.adapters.codex.generate_config import install_agents_policy, install_codex_mcp

    root = Path(project).resolve()
    results = [install_codex_mcp(root).to_dict(), install_agents_policy(root).to_dict()]
    console.print_json(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


@adapters_app.command("install-claude-hooks")
def adapters_install_claude_hooks(project: str = typer.Option(".", "--project", help="Project root")) -> None:
    from dysonspherain.adapters.codex.generate_config import install_claude_hooks

    result = install_claude_hooks(Path(project).resolve())
    console.print_json(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


@adapters_app.command("install-agents-policy")
def adapters_install_agents_policy(project: str = typer.Option(".", "--project", help="Project root")) -> None:
    from dysonspherain.adapters.codex.generate_config import install_agents_policy

    result = install_agents_policy(Path(project).resolve())
    console.print_json(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


@adapters_app.command("install-plugin-manifests")
def adapters_install_plugin_manifests(project: str = typer.Option(".", "--project", help="Project root")) -> None:
    from dysonspherain.adapters.codex.generate_config import install_plugin_manifests

    result = install_plugin_manifests(Path(project).resolve())
    console.print_json(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


@adapters_app.command("doctor")
def adapters_doctor(project: str = typer.Option(".", "--project", help="Project root")) -> None:
    from dysonspherain.adapters.codex.generate_config import doctor

    console.print_json(json.dumps(doctor(Path(project).resolve()), ensure_ascii=False, indent=2, sort_keys=True))


@adapters_app.command("write-integration-report")
def adapters_write_integration_report(
    project: str = typer.Option(".", "--project", help="Project root"),
    tests_run: str = typer.Option("", "--tests-run", help="Semicolon-separated test/smoke commands to record"),
) -> None:
    from dysonspherain.adapters.integration_report import write_memory_agent_integration_report

    tests = [item.strip() for item in tests_run.split(";") if item.strip()]
    path = write_memory_agent_integration_report(Path(project).resolve(), tests_run=tests or None)
    print(f"[green]Saved integration report:[/green] {path}")


@adapters_app.command("daemon")
def adapters_daemon(
    project_root: str = typer.Option(".", "--project-root", help="Project root"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(37777, "--port"),
) -> None:
    from dysonspherain.adapters.daemon import run_server

    run_server(Path(project_root).resolve(), host=host, port=port, project=project)


@adapters_app.command("install-supervisor")
def adapters_install_supervisor(
    project: str = typer.Option(".", "--project", help="Project root"),
    service: str = typer.Option("all", "--service", help="memory-daemon, memory-scheduler, or all"),
    platform_name: Optional[str] = typer.Option(None, "--platform", help="launchd or systemd; defaults to current OS"),
    python: Optional[str] = typer.Option(None, "--python", help="Python executable for the service"),
    project_name: str = typer.Option("DysonSpherain", "--project-name", help="Runtime project name"),
    host: str = typer.Option("127.0.0.1", "--host", help="Daemon host"),
    port: int = typer.Option(37777, "--port", help="Daemon port"),
    interval_seconds: float = typer.Option(5.0, "--interval-seconds", help="Scheduler daemon polling interval"),
    activate: bool = typer.Option(False, "--activate", help="Load/enable the service after writing it"),
) -> None:
    from dysonspherain.adapters.supervisor import install_all, install_supervisor

    root = Path(project).resolve()
    kwargs = {
        "platform_name": platform_name,
        "python": python,
        "host": host,
        "port": port,
        "project_name": project_name,
        "interval_seconds": interval_seconds,
        "activate": activate,
    }
    if service == "all":
        payload = install_all(root, **kwargs)
    else:
        payload = install_supervisor(root, service=service, **kwargs).to_dict()
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@adapters_app.command("uninstall-supervisor")
def adapters_uninstall_supervisor(
    service: str = typer.Option("all", "--service", help="memory-daemon, memory-scheduler, or all"),
    platform_name: Optional[str] = typer.Option(None, "--platform", help="launchd or systemd; defaults to current OS"),
    project_name: str = typer.Option("DysonSpherain", "--project-name", help="Runtime project name"),
    deactivate: bool = typer.Option(False, "--deactivate", help="Stop/disable the service before deleting config"),
) -> None:
    from dysonspherain.adapters.supervisor import uninstall_all, uninstall_supervisor

    kwargs = {"platform_name": platform_name, "project_name": project_name, "deactivate": deactivate}
    if service == "all":
        payload = uninstall_all(**kwargs)
    else:
        payload = uninstall_supervisor(service=service, **kwargs).to_dict()
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@adapters_app.command("supervisor-status")
def adapters_supervisor_status(
    service: str = typer.Option("all", "--service", help="memory-daemon, memory-scheduler, or all"),
    platform_name: Optional[str] = typer.Option(None, "--platform", help="launchd or systemd; defaults to current OS"),
    project_name: str = typer.Option("DysonSpherain", "--project-name", help="Runtime project name"),
) -> None:
    from dysonspherain.adapters.supervisor import status_all, supervisor_status

    kwargs = {"platform_name": platform_name, "project_name": project_name}
    if service == "all":
        payload = status_all(**kwargs)
    else:
        payload = supervisor_status(service=service, **kwargs)
    console.print_json(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@code_app.command("index")
def code_index_command(
    target: Path = typer.Argument(Path("."), help="Repository or Python file to index"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    payload = build_code_index(Path.cwd(), target, project=project)
    print(
        f"[green]Indexed code:[/green] files={payload['file_count']} "
        f"parse_errors={payload['parse_error_count']}"
    )


@code_app.command("search-symbol")
def code_search_symbol_command(
    query: str = typer.Argument(..., help="Symbol substring"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
) -> None:
    console.print_json(json.dumps(code_search_symbol(Path.cwd(), query, project=project), ensure_ascii=False, indent=2))


@code_app.command("relevant-files")
def code_relevant_files_command(
    query: str = typer.Argument(..., help="Natural language or keyword query"),
    project: str = typer.Option("DysonSpherain", "--project", help="Project name"),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum files"),
) -> None:
    console.print_json(json.dumps(code_relevant_files(Path.cwd(), query, project=project, limit=limit), ensure_ascii=False, indent=2))


@app.command("legacy-recall")
def legacy_recall(
    query: str,
    top_k: int = typer.Option(8),
    task_type: str = typer.Option("trace"),
    explain: bool = typer.Option(False),
) -> None:
    memory_find(query=query, top_k=top_k, task_type=task_type, explain=explain)


@app.command("memory-list")
def memory_list_alias(
    limit: int = typer.Option(20, min=1),
    project: Optional[str] = typer.Option(None),
    scope: Optional[str] = typer.Option(None),
    molecular_type: Optional[str] = typer.Option(None, "--molecular-type"),
) -> None:
    memory_list(limit=limit, project=project, scope=scope, molecular_type=molecular_type)


@app.command("artifact-add")
def artifact_add(
    path: str = typer.Argument(..., help="Local file path to register"),
    artifact_type: str = typer.Option("file"),
    title: Optional[str] = typer.Option(None),
    summary: str = typer.Option(""),
    tags: str = typer.Option(""),
    project: Optional[str] = typer.Option(None),
    session_id: Optional[str] = typer.Option(None, "--session-id"),
    scope: Optional[str] = typer.Option(None),
    source_ref: Optional[str] = typer.Option(None),
) -> None:
    runtime = build_runtime()
    storage = runtime.services.storage
    vector_store = runtime.services.vector_store
    config = runtime.services.config
    artifact_path = str(Path(path).expanduser().resolve())
    tag_values = _csv_values(tags)
    existing = storage.fetch_artifact_by_path(artifact_path)
    record = ArtifactRecord(
        path=artifact_path,
        artifact_type=artifact_type,
        scope=compose_scope(scope=scope, project=project or config.project_name, session_id=session_id or config.session_id),
        workspace=config.workspace_name,
        project=project or config.project_name,
        session_id=session_id or config.session_id,
        title=title or Path(artifact_path).name,
        summary=summary or None,
        tags_json=json.dumps(tag_values, ensure_ascii=False),
        source_ref=source_ref or artifact_path,
    )
    if existing:
        record.artifact_id = str(existing.get("artifact_id") or record.artifact_id)
        record.created_at = str(existing.get("created_at") or record.created_at)
    storage.insert_artifact(record.to_dict())
    object_text, source_unit_text = _artifact_object_text(
        title=record.title,
        summary=record.summary,
        path=artifact_path,
        artifact_type=artifact_type,
        tags=tag_values,
    )
    artifact_object = MemoryObject(
        object_type="artifact",
        object_text=object_text,
        source_chunk_id="",
        source_node_id="",
        scope=record.scope,
        workspace=record.workspace,
        project=record.project,
        session_id=record.session_id,
        subject="artifact",
        predicate="registered",
        entity=(record.title or Path(artifact_path).name).lower(),
        attribute="artifact_type",
        new_value=artifact_type,
        canonical_key=f"artifact:{artifact_path.lower()}",
        source_unit_text=source_unit_text,
        confidence=0.9,
        status="active",
        source_type="artifact_registry",
        source_ref=artifact_path,
        verification_status="registered",
        metadata_json=json.dumps({"artifact_id": record.artifact_id, "title": record.title, "summary": record.summary, "tags": tag_values}, ensure_ascii=False),
        object_id=f"obj_art_{record.artifact_id.split('_', 1)[-1]}",
    )
    storage.insert_objects([artifact_object.to_dict()])
    vector_store.upsert_objects([artifact_object.to_dict()])
    console.print_json(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))


@app.command("open-loop-add")
def open_loop_add(
    title: str = typer.Argument(...),
    details: str = typer.Option(""),
    priority: str = typer.Option("normal"),
    tags: str = typer.Option(""),
    project: Optional[str] = typer.Option(None),
    session_id: Optional[str] = typer.Option(None, "--session-id"),
    scope: Optional[str] = typer.Option(None),
) -> None:
    runtime = build_runtime()
    storage = runtime.services.storage
    vector_store = runtime.services.vector_store
    config = runtime.services.config
    tag_values = _csv_values(tags)
    item = OpenLoopItem(
        title=title,
        details=details or None,
        priority=priority,
        scope=compose_scope(scope=scope, project=project or config.project_name, session_id=session_id or config.session_id),
        workspace=config.workspace_name,
        project=project or config.project_name,
        session_id=session_id or config.session_id,
        tags_json=json.dumps(tag_values, ensure_ascii=False),
    )
    storage.insert_open_loop(item.to_dict())
    object_text, source_unit_text = _open_loop_object_text(
        title=item.title,
        details=item.details,
        status=item.status,
        priority=item.priority,
        tags=tag_values,
        blocked_reason=item.blocked_reason,
    )
    loop_object = MemoryObject(
        object_type="open_loop",
        object_text=object_text,
        source_chunk_id="",
        source_node_id="",
        scope=item.scope,
        workspace=item.workspace,
        project=item.project,
        session_id=item.session_id,
        subject="task",
        predicate="pending",
        entity=title.lower(),
        attribute="open_loop",
        canonical_key=f"open_loop:{item.loop_id}",
        source_unit_text=source_unit_text,
        confidence=0.88,
        status=item.status,
        source_type="open_loop_registry",
        source_ref=item.loop_id,
        verification_status="registered",
        metadata_json=json.dumps({"details": details, "priority": priority, "tags": tag_values}, ensure_ascii=False),
        object_id=f"obj_loop_{item.loop_id.split('_', 1)[-1]}",
    )
    storage.insert_objects([loop_object.to_dict()])
    vector_store.upsert_objects([loop_object.to_dict()])
    console.print_json(json.dumps(item.to_dict(), ensure_ascii=False, indent=2))


@app.command("open-loop-list")
def open_loop_list(
    status: str = typer.Option("open"),
    project: Optional[str] = typer.Option(None),
    include_all: bool = typer.Option(False, help="Ignore status filter"),
) -> None:
    _, storage, _, *_ = build_services()
    clauses: list[str] = []
    params: list[object] = []
    if not include_all:
        clauses.append("status = ?")
        params.append(status)
    if project:
        clauses.append("project = ?")
        params.append(project)
    rows = storage.fetch_open_loops(" AND ".join(clauses), tuple(params))
    table = Table(title="Open Loops")
    table.add_column("loop_id")
    table.add_column("status")
    table.add_column("scope")
    table.add_column("project")
    table.add_column("priority")
    table.add_column("title")
    for row in rows:
        table.add_row(
            str(row.get("loop_id") or ""),
            str(row.get("status") or ""),
            str(row.get("scope") or "global"),
            str(row.get("project") or ""),
            str(row.get("priority") or ""),
            (str(row.get("title") or ""))[:90],
        )
    console.print(table)


@app.command("open-loop-update")
def open_loop_update(
    loop_id: str = typer.Argument(...),
    status: str = typer.Option(..., help="open, closed, deferred, blocked"),
    blocked_reason: str = typer.Option(""),
) -> None:
    runtime = build_runtime()
    storage = runtime.services.storage
    storage.update_open_loop_status(loop_id, status, blocked_reason=blocked_reason or None)
    loop_rows = storage.fetch_open_loops("loop_id = ?", (loop_id,))
    loop_row = dict(loop_rows[0]) if loop_rows else {}
    objects = storage.fetch_objects("object_type = ? AND canonical_key = ?", ("open_loop", f"open_loop:{loop_id}"))
    if objects:
        obj = dict(objects[0])
        obj["status"] = status
        obj["updated_at"] = now_iso()
        object_text, source_unit_text = _open_loop_object_text(
            title=str(loop_row.get("title") or obj.get("object_text") or loop_id),
            details=str(loop_row.get("details") or ""),
            status=status,
            priority=str(loop_row.get("priority") or ""),
            tags=_json_list_values(loop_row.get("tags_json")),
            blocked_reason=blocked_reason or str(loop_row.get("blocked_reason") or ""),
        )
        metadata_payload = {
            "details": loop_row.get("details"),
            "priority": loop_row.get("priority"),
            "blocked_reason": blocked_reason or loop_row.get("blocked_reason"),
            "tags": _json_list_values(loop_row.get("tags_json")),
        }
        obj["object_text"] = object_text
        obj["source_unit_text"] = source_unit_text
        obj["metadata_json"] = json.dumps(metadata_payload, ensure_ascii=False)
        storage.insert_objects([obj])
        runtime.services.vector_store.upsert_objects([obj])
    print(f"[green]Updated open loop[/green] {loop_id} -> {status}")


@app.command("workspace-list")
def workspace_list() -> None:
    _, storage, _, *_ = build_services()
    rows = storage.list_workspace_inventory()
    table = Table(title="Workspace Inventory")
    table.add_column("workspace")
    table.add_column("project")
    table.add_column("scope")
    table.add_column("session")
    table.add_column("source")
    table.add_column("count")
    for row in rows:
        table.add_row(
            str(row.get("workspace") or ""),
            str(row.get("project") or ""),
            str(row.get("scope") or "global"),
            str(row.get("session_id") or ""),
            str(row.get("source") or ""),
            str(row.get("item_count") or 0),
        )
    console.print(table)


@maint_app.command("reembed")
def maint_reembed(batch_size: int = typer.Option(200)) -> None:
    *_, background = build_services()
    report = background.reembed_all(batch_size=batch_size)
    console.print_json(json.dumps(report, ensure_ascii=False, indent=2))


@maint_app.command("compress")
def maint_compress(
    cold_days: int = typer.Option(30),
    access_threshold: int = typer.Option(1),
    limit: int = typer.Option(200),
) -> None:
    *_, background = build_services()
    report = background.compress_cold_data(cold_days=cold_days, access_threshold=access_threshold, limit=limit)
    console.print_json(json.dumps(report, ensure_ascii=False, indent=2))


@maint_app.command("decay-edges")
def maint_decay_edges(
    factor: float = typer.Option(0.97),
    floor: float = typer.Option(0.05),
) -> None:
    *_, background = build_services()
    report = background.decay_edges(factor=factor, floor=floor)
    console.print_json(json.dumps(report, ensure_ascii=False, indent=2))


@maint_app.command("split-zones")
def maint_split_zones(
    threshold: int = typer.Option(40),
    group_size: int = typer.Option(20),
    apply_changes: bool = typer.Option(True),
) -> None:
    *_, background = build_services()
    report = background.split_large_zones(threshold=threshold, group_size=group_size, apply_changes=apply_changes)
    console.print_json(json.dumps(report, ensure_ascii=False, indent=2))


@maint_app.command("rebuild-representations")
def maint_rebuild_representations(
    node_id: Optional[str] = typer.Option(None, help="Optional single node id to rebuild"),
    limit: int = typer.Option(200, min=1, help="Maximum number of nodes to rebuild when node_id is omitted"),
) -> None:
    runtime = build_runtime()
    storage = runtime.services.storage
    ingest_rows = storage.fetch_ingest_files()
    source_path_by_node = {str(row.get("node_id")): str(row.get("source_path")) for row in ingest_rows if row.get("node_id")}
    nodes = [storage.fetch_node_by_id(node_id)] if node_id else storage.fetch_nodes()
    nodes = [node for node in nodes if node]
    if not node_id:
        nodes = nodes[:limit]

    rewritten = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, object]] = []
    for row in nodes:
        if not ((row.get("raw_content") or "").strip() or (row.get("summary") or "").strip()):
            skipped += 1
            continue
        try:
            node = MemoryNode(
                shell=int(row["shell"]),
                sector=str(row["sector"]),
                zone=str(row["zone"]),
                cell=str(row["cell"]),
                molecular_type=str(row["molecular_type"]),
                summary=str(row.get("summary") or ""),
                content_ref=str(row.get("content_ref") or "") or None,
                raw_content=str(row.get("raw_content") or "") or None,
                theta=row.get("theta"),
                phi=row.get("phi"),
                importance=float(row.get("importance") or 0.0),
                creative_score=float(row.get("creative_score") or 0.0),
                stability_score=float(row.get("stability_score") or 0.0),
                access_count=int(row.get("access_count") or 0),
                compression_level=str(row.get("compression_level") or "medium"),
                stage=str(row.get("stage") or "long_term"),
                tags=str(row.get("tags") or "") or None,
                id=str(row["id"]),
                created_at=str(row["created_at"]),
                last_accessed_at=str(row["last_accessed_at"]),
            )
            runtime.writeback_memory(
                node=node,
                source_kind=str(row.get("molecular_type") or "raw_content"),
                source_path=source_path_by_node.get(node.id) or (str(row.get("content_ref") or "") or None),
                replace_node_id=node.id,
            )
            rewritten += 1
        except Exception as exc:
            failed += 1
            failures.append({"node_id": row.get("id"), "error": str(exc)})
    report = {
        "selected": len(nodes),
        "rewritten": rewritten,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
    }
    console.print_json(json.dumps(report, ensure_ascii=False, indent=2))


@eval_app.command("run")
def eval_run(
    dataset: str = typer.Option("evaluation/real_tasks_sample.json", help="Path to a real-task evaluation dataset"),
    out: Optional[str] = typer.Option(None, help="Optional output report path"),
) -> None:
    evaluator = RealTaskEvaluator(Path(dataset), Path(out) if out else None)
    report = evaluator.run()
    console.print_json(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"[green]Saved evaluation report[/green] -> {out or Path(report['run_root']) / 'real_task_eval_report.json'}")


@app.command("ask")
def ask(
    query: str = typer.Argument(..., help="Question to answer from local memory"),
    task_type: str = typer.Option("qa", "--task-type"),
    answer_mode: str = typer.Option("local", "--answer-mode", help="local is deterministic and citation-first"),
    include_creative: bool = typer.Option(False, "--include-creative", help="Allow creative sidecar evidence in the answer"),
    evidence_top_k: int = typer.Option(8, "--evidence-top-k"),
    max_tokens: int = typer.Option(1800, "--max-tokens"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    runtime = build_runtime()
    result = runtime.answer(
        query=query,
        task_type=task_type,
        max_tokens=max_tokens,
        evidence_top_k=evidence_top_k,
        answer_mode=answer_mode,
        include_creative=include_creative,
    )
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result["answer"])
    meta = {
        "abstained": result.get("abstained"),
        "confidence": result.get("confidence"),
        "citations": result.get("citations"),
        "creative_used": result.get("creative_used"),
    }
    console.print_json(json.dumps(meta, ensure_ascii=False, indent=2))


@app.command("doctor")
def doctor(json_output: bool = typer.Option(False, "--json")) -> None:
    report: dict[str, object] = {
        "ok": True,
        "checks": {},
        "recommendations": [],
    }
    checks = report["checks"]  # type: ignore[assignment]
    recommendations = report["recommendations"]  # type: ignore[assignment]
    try:
        runtime = build_runtime()
        config = runtime.services.config
        vector_info = runtime.services.vector_store.info()
        checks["metadata_db"] = {"ok": config.db_path.exists(), "path": str(config.db_path)}
        checks["vector_store"] = {"ok": True, **vector_info}
        checks["embedding_guard"] = {
            "ok": not (config.embedding_fail_fast and vector_info.get("fallback_in_use")),
            "embedding_fail_fast": config.embedding_fail_fast,
            "fallback_in_use": vector_info.get("fallback_in_use"),
            "provider": vector_info.get("embedding_provider"),
            "model": vector_info.get("embedding_model"),
        }
        if vector_info.get("fallback_in_use"):
            recommendations.append(
                "SentenceTransformer is unavailable; benchmark runs should set SPHERE_EMBEDDING_FAIL_FAST=1 and install/cache the requested model."
            )
        checks["edge_writeback"] = {
            "ok": bool(config.enable_lightweight_edge_writeback),
            "enabled": bool(config.enable_lightweight_edge_writeback),
            "edge_count": runtime.services.storage.count_edges(),
        }
        checks["tests_present"] = {"ok": (Path.cwd() / "tests").exists(), "path": str(Path.cwd() / "tests")}
    except Exception as exc:
        report["ok"] = False
        checks["runtime"] = {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
    if json_output:
        console.print_json(json.dumps(report, ensure_ascii=False, indent=2))
        return
    console.print_json(json.dumps(report, ensure_ascii=False, indent=2))


@runs_app.command("ingest")
def runs_ingest(
    path: str = typer.Argument(..., help="Benchmark result directory or metrics JSON file"),
    project: Optional[str] = typer.Option(None, "--project", help="Project name for the ingested runs"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    base_dir = Path.cwd()
    runs = ingest_artifacts(Path(path), base_dir=base_dir, project=_active_project(project))
    payload = {
        "ingested_count": len(runs),
        "registry": str(registry_path(base_dir)),
        "summary": str(registry_summary_path(base_dir)),
        "runs": [run.run_id for run in runs],
    }
    if json_output:
        console.print_json(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"[green]Ingested[/green] {len(runs)} benchmark run artifact(s)")
    print(f"[green]Registry[/green] {payload['registry']}")
    print(f"[green]Summary[/green] {payload['summary']}")


@runs_app.command("list")
def runs_list(
    project: Optional[str] = typer.Option(None, "--project", help="Filter by project"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Filter by dataset"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    active_project = _active_project(project) if project else None
    runs = load_registry(Path.cwd())
    if active_project:
        runs = [run for run in runs if run.project == active_project]
    if dataset:
        runs = [run for run in runs if run.dataset.lower() == dataset.lower()]
    if json_output:
        console.print_json(json.dumps([run.__dict__ for run in runs], ensure_ascii=False, indent=2))
        return
    table = Table(title="Benchmark Runs")
    for column in ("run_id", "project", "dataset", "type", "questions", "elapsed_s", "fallback", "comparable"):
        table.add_column(column)
    for run in runs:
        table.add_row(
            run.run_id,
            run.project,
            run.dataset,
            run.run_type,
            str(run.question_count or ""),
            "" if run.elapsed_seconds is None else f"{run.elapsed_seconds:.3f}",
            str(run.fallback_in_use),
            str(run.comparable),
        )
    console.print(table)


@runs_app.command("latest")
def runs_latest(
    dataset: str = typer.Option(..., "--dataset", help="Dataset name, for example LongMemEval or CloneMem"),
    project: Optional[str] = typer.Option(None, "--project", help="Project name"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    run = latest_run(load_registry(Path.cwd()), project=_active_project(project), dataset=dataset)
    if run is None:
        raise typer.BadParameter(f"No run found for dataset {dataset!r}.")
    if json_output:
        console.print_json(json.dumps(run.__dict__, ensure_ascii=False, indent=2))
        return
    console.print_json(json.dumps(run.__dict__, ensure_ascii=False, indent=2))


@runs_app.command("compare")
def runs_compare(
    a: str = typer.Option(..., "--a", help="Run id or unique prefix for baseline run"),
    b: str = typer.Option(..., "--b", help="Run id or unique prefix for comparison run"),
    out: Optional[str] = typer.Option(None, "--out", help="Optional markdown report path"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    runs = load_registry(Path.cwd())
    comparison = compare_runs(resolve_run(runs, a), resolve_run(runs, b))
    report_path = write_compare_report(Path.cwd(), comparison, Path(out) if out else None)
    if json_output:
        console.print_json(json.dumps({**comparison, "report_path": str(report_path)}, ensure_ascii=False, indent=2))
        return
    console.print(markdown_compare_table(comparison))
    print(f"[green]Saved[/green] {report_path}")


@runs_app.command("explain-regression")
def runs_explain_regression(
    dataset: str = typer.Option(..., "--dataset", help="Dataset name to inspect"),
    project: Optional[str] = typer.Option(None, "--project", help="Project name"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON"),
    out: Optional[str] = typer.Option(None, "--out", help="Optional markdown report path"),
) -> None:
    explanation = regression_explanation(load_registry(Path.cwd()), project=_active_project(project), dataset=dataset)
    report_path = write_regression_report(Path.cwd(), explanation, Path(out) if out else None)
    if json_output:
        console.print_json(json.dumps({**explanation, "report_path": str(report_path)}, ensure_ascii=False, indent=2))
        return
    console.print_json(json.dumps(explanation, ensure_ascii=False, indent=2))
    print(f"[green]Saved[/green] {report_path}")


@config_app.command("get")
def config_get() -> None:
    console.print_json(json.dumps(_load_local_config_overrides(), ensure_ascii=False, indent=2))


@config_app.command("set")
def config_set(key: str = typer.Argument(...), value: str = typer.Argument(...)) -> None:
    allowed = set(AppConfig.__dataclass_fields__.keys())
    if key not in allowed:
        raise typer.BadParameter(f"Unknown config key {key!r}.")
    payload = _load_local_config_overrides()
    payload[key] = _coerce_config_value(key, value)
    _save_local_config_overrides(payload)
    print(f"[green]Saved[/green] {key} = {payload[key]!r} in {_local_config_path()}")


@config_app.command("profile")
def config_profile(name: str = typer.Argument(..., help="fast, balanced, deep, paper, or benchmark")) -> None:
    name = name.strip().lower()
    profiles: dict[str, dict[str, object]] = {
        "fast": {"mode": "fast", "creative_mode": "off", "enable_benchmark_route_tuning": False},
        "balanced": {"mode": "balanced", "creative_mode": "off"},
        "deep": {"mode": "deep", "creative_mode": "conservative"},
        "paper": {
            "mode": "deep",
            "creative_mode": "off",
            "embedding_fail_fast": True,
            "enable_benchmark_route_tuning": False,
            "enable_lightweight_edge_writeback": True,
        },
        "benchmark": {
            "mode": "deep",
            "embedding_fail_fast": True,
            "enable_benchmark_route_tuning": True,
            "creative_mode": "off",
            "enable_lightweight_edge_writeback": True,
        },
    }
    if name not in profiles:
        raise typer.BadParameter("Use one of: fast, balanced, deep, paper, benchmark.")
    payload = _load_local_config_overrides()
    payload.update(profiles[name])
    _save_local_config_overrides(payload)
    print(f"[green]Activated config profile:[/green] {name}")


@memory_app.command("export")
def memory_export(
    out: Optional[str] = typer.Option(None, "--out", help="Output JSON path"),
    format: str = typer.Option("json", "--format", help="json or jsonl"),
) -> None:
    _, storage, _, *_ = build_services()
    with storage.connect() as conn:
        nodes = [dict(r) for r in conn.execute("SELECT * FROM memory_nodes").fetchall()]
        chunks = [dict(r) for r in conn.execute("SELECT * FROM memory_chunks").fetchall()]
        objects = [dict(r) for r in conn.execute("SELECT * FROM memory_objects").fetchall()]
        edges = [dict(r) for r in conn.execute("SELECT * FROM memory_edges").fetchall()]
    export_path = Path(out) if out else storage.config.export_dir / f"memory_export_{stable_content_hash(now_iso())[:8]}.{format}"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "jsonl":
        with export_path.open("w", encoding="utf-8") as fh:
            for kind, rows in (("node", nodes), ("chunk", chunks), ("object", objects), ("edge", edges)):
                for row in rows:
                    fh.write(json.dumps({"kind": kind, "payload": row}, ensure_ascii=False) + "\n")
    else:
        export_path.write_text(
            json.dumps({"nodes": nodes, "chunks": chunks, "objects": objects, "edges": edges}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[green]Exported memory[/green] -> {export_path}")


@memory_app.command("obs-export")
def memory_obs_export(
    out: str = typer.Option("artifacts/memory_os/observations_export.json", "--out"),
    project: str = typer.Option("DysonSpherain", "--project"),
) -> None:
    from dysonspherain.memory_os.observation_store import export_observations

    path = export_observations(Path.cwd(), project, Path(out))
    print(f"[green]Exported observations[/green] -> {path}")


@memory_app.command("obs-delete")
def memory_obs_delete(
    observation_id: str = typer.Argument(...),
    project: str = typer.Option("DysonSpherain", "--project"),
    hard: bool = typer.Option(False, "--hard"),
) -> None:
    from dysonspherain.memory_os.observation_store import delete_observation

    console.print_json(json.dumps(delete_observation(Path.cwd(), project, observation_id, hard=hard), ensure_ascii=False, indent=2))


@memory_app.command("obs-retention")
def memory_obs_retention(
    keep_last: int = typer.Option(200, "--keep-last", min=1),
    project: str = typer.Option("DysonSpherain", "--project"),
) -> None:
    from dysonspherain.memory_os.observation_store import apply_retention

    console.print_json(json.dumps(apply_retention(Path.cwd(), project, keep_last=keep_last), ensure_ascii=False, indent=2))


@memory_app.command("backup")
def memory_backup(out: Optional[str] = typer.Option(None, "--out", help="Output backup directory")) -> None:
    config, storage, vector_store, *_ = build_services()
    backup_root = Path(out) if out else config.export_dir / f"backup_{stable_content_hash(now_iso())[:8]}"
    backup_root.mkdir(parents=True, exist_ok=True)
    vector_store.close()
    if config.db_path.exists():
        shutil.copy2(config.db_path, backup_root / "memory.db")
    if config.vector_dir.exists():
        target = backup_root / "vector_db"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(config.vector_dir, target)
    (_local_config_path()).exists() and shutil.copy2(_local_config_path(), backup_root / ".dysonspherain_config.json")
    manifest = {
        "created_at": now_iso(),
        "db_path": str(config.db_path),
        "vector_dir": str(config.vector_dir),
        "embedding": vector_store.info() if hasattr(vector_store, "info") else {},
    }
    (backup_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[green]Created backup[/green] -> {backup_root}")


@memory_app.command("forget")
def memory_forget(
    query: str = typer.Argument(..., help="Case-insensitive text to match in summary/raw content/object text"),
    confirm: bool = typer.Option(False, "--confirm", help="Actually delete matching memories"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    _, storage, vector_store, *_ = build_services()
    pattern = f"%{query.lower()}%"
    with storage.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            """
            SELECT DISTINCT id, summary
            FROM memory_nodes
            WHERE lower(coalesce(summary, '') || ' ' || coalesce(raw_content, '')) LIKE ?
            LIMIT ?
            """,
            (pattern, int(limit)),
        ).fetchall()]
    if not confirm:
        console.print_json(json.dumps({"dry_run": True, "matches": rows}, ensure_ascii=False, indent=2))
        print("[yellow]Re-run with --confirm to delete these memories.[/yellow]")
        return
    deleted = []
    for row in rows:
        node_id = str(row["id"])
        old_chunk_ids = storage.delete_chunks_for_node(node_id)
        old_object_ids = storage.delete_objects_for_node(node_id)
        old_representation_ids = storage.delete_representations_for_parent_ids([node_id] + old_chunk_ids)
        storage.delete_chunk_neighbors(old_chunk_ids)
        vector_store.delete_chunks(old_chunk_ids)
        vector_store.delete_objects(old_object_ids)
        vector_store.delete_representations(old_representation_ids)
        storage.delete_node(node_id)
        deleted.append(node_id)
    console.print_json(json.dumps({"deleted": deleted, "count": len(deleted)}, ensure_ascii=False, indent=2))


@benchmark_app.command("smoke-all")
def benchmark_smoke_all(
    out: Optional[str] = typer.Option(None, "--out", help="Output artifact JSON path"),
    record_token_economy: bool = typer.Option(False, "--record-token-economy", help="Record diagnostic token economy artifacts"),
    token_economy_output: Optional[str] = typer.Option(None, "--token-economy-output", help="Token economy artifact directory"),
    tokenizer_model: str = typer.Option("cl100k_base", "--tokenizer-model"),
    token_economy_baseline_types: str = typer.Option("full_history,naive_recent", "--token-economy-baseline-types"),
    token_economy_modes: str = typer.Option("conservative", "--token-economy-modes"),
    context_token_budget: str = typer.Option("800,1600", "--context-token-budget"),
) -> None:
    with tempfile.TemporaryDirectory(prefix="dysonspherain_smoke_") as tmp:
        runtime = UnifiedMemoryRuntime.from_base_dir(Path(tmp), config_overrides={
            "vector_backend": "json",
            "embedding_fail_fast": True,
            "enable_benchmark_route_tuning": True,
            "creative_mode": "off",
            "enable_lightweight_edge_writeback": True,
        })
        cases = [
            ("temporal", "2026-04-20 decision: use temporal edge reranking for wrong-time memory drift.", "What fixed wrong-time memory drift?"),
            ("crowding", "2026-04-21 decision: use competition-aware inhibition to suppress local candidate crowding.", "What suppresses local candidate crowding?"),
            ("creative", "Creative mode must stay as a bounded sidecar and must not contaminate factual evidence.", "How should creative mode be used?"),
        ]
        writes = []
        answers = []
        for cell, content, query in cases:
            node = MemoryNode(
                shell=1,
                sector="project",
                zone="benchmark_smoke",
                cell=cell,
                molecular_type="decision",
                summary=content,
                raw_content=content,
                importance=0.7,
                creative_score=0.4,
                stability_score=0.8,
                verification_status="verified",
            )
            writes.append(runtime.writeback_memory(node))
            answers.append(runtime.answer(query, evidence_top_k=4, answer_mode="local"))
        artifact = {
            "created_at": now_iso(),
            "kind": "benchmark_smoke_all",
            "code_package_hash": stable_content_hash(str(Path.cwd().resolve())),
            "embedding": runtime.services.vector_store.info(),
            "write_results": writes,
            "answers": answers,
            "storage_counts": {
                "nodes": len(runtime.services.storage.fetch_nodes()),
                "chunks": runtime.services.storage.count_chunks(),
                "objects": len(runtime.services.storage.fetch_objects()),
                "edges": runtime.services.storage.count_edges(),
            },
            "valid_for_full_benchmark_claims": False,
            "note": "Smoke artifact only. Full LongMemEval/LoCoMo/KnowMe/CloneMem datasets must be mounted and run separately.",
        }
        if record_token_economy:
            from dysonspherain.evaluation.token_economy import _sample_from_payload, _split_csv, _split_int_csv
            from dysonspherain.token_economy.report import write_report
            from dysonspherain.utils.token_counter import TokenCounter

            counter = TokenCounter(tokenizer_model)
            payloads = [
                {
                    "sample_id": cell,
                    "query": query,
                    "history": content * 20,
                    "retrieved_context": str(answer.get("answer") or ""),
                    "retrieval_quality": {"recall_at_10": 1.0 if not answer.get("abstained") else 0.0},
                    "candidate_count": len(answer.get("citations") or []),
                }
                for (cell, content, query), answer in zip(cases, answers)
            ]
            samples = []
            for mode_name in _split_csv(token_economy_modes):
                for baseline_type in _split_csv(token_economy_baseline_types):
                    for budget in _split_int_csv(context_token_budget):
                        for index, payload in enumerate(payloads):
                            samples.append(
                                _sample_from_payload(
                                    payload,
                                    index=index,
                                    mode=mode_name,
                                    baseline_type=baseline_type,
                                    budget=budget,
                                    recent_k=20,
                                    counter=counter,
                                    allow_evidence_truncation=False,
                                )
                            )
            te_dir = Path(token_economy_output) if token_economy_output else Path.cwd() / "artifacts" / "token_economy_smoke"
            token_summary = write_report(samples, te_dir)
            artifact["token_economy"] = {
                "summary": token_summary,
                "artifact_dir": str(te_dir),
                "diagnostic_only": True,
            }
    artifact_path = Path(out) if out else Path.cwd() / "data" / "exports" / "benchmark_smoke_all.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print_json(json.dumps(artifact["storage_counts"], ensure_ascii=False, indent=2))
    print(f"[green]Saved smoke artifact[/green] -> {artifact_path}")


def _render_ingest_results(results: list, title: str) -> None:
    table = Table(title=title)
    table.add_column("path")
    table.add_column("node_id")
    table.add_column("status")
    table.add_column("kind")
    table.add_column("cell")
    table.add_column("chunks")
    for r in results:
        table.add_row(r.path, r.node_id, r.status, r.detected_kind, r.cell, str(r.chunk_count))
    console.print(table)
    print(f"[green]Processed:[/green] {len(results)} files")


def _join_searchable_parts(parts: list[str | None]) -> str:
    return " ".join(part.strip() for part in parts if str(part or "").strip())


def _artifact_object_text(
    *,
    title: str | None,
    summary: str | None,
    path: str,
    artifact_type: str,
    tags: list[str],
) -> tuple[str, str]:
    basename = Path(path).name if path else ""
    object_text = _join_searchable_parts([title, summary, basename, artifact_type])
    source_unit_text = _join_searchable_parts([title, summary, basename, path, artifact_type, " ".join(tags)])
    return object_text or basename or path, source_unit_text or object_text or basename or path


def _open_loop_object_text(
    *,
    title: str,
    details: str | None,
    status: str,
    priority: str,
    tags: list[str],
    blocked_reason: str | None,
) -> tuple[str, str]:
    object_text = _join_searchable_parts([title, details])
    source_unit_text = _join_searchable_parts(
        [
            title,
            details,
            f"status {status}" if status else "",
            f"priority {priority}" if priority else "",
            blocked_reason,
            " ".join(tags),
        ]
    )
    return object_text or title, source_unit_text or object_text or title


def _csv_values(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part and part.strip()]


def _json_list_values(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    try:
        payload = json.loads(str(raw or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]


if __name__ == "__main__":
    app()
