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

from .activation_engine import ActivationEngine
from .background_tasks import BackgroundTaskRunner
from .config import AppConfig
from .context_assembler import ContextAssembler
from .creative_reflection_engine import CreativeReflectionEngine
from .evidence_pipeline import EvidencePipeline
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
remember_app = typer.Typer(help="Write and manage memories")
memory_app = typer.Typer(help="Search, trace, audit, visualize")
creative_app = typer.Typer(help="Creative reflection commands")
ingest_app = typer.Typer(help="Ingest markdown/code/log/pdf files")
maint_app = typer.Typer(help="Background maintenance tasks")
eval_app = typer.Typer(help="Real-task evaluation commands")
config_app = typer.Typer(help="Persistent local configuration")
benchmark_app = typer.Typer(help="Benchmark smoke and artifact commands")

app.add_typer(remember_app, name="remember")
app.add_typer(memory_app, name="memory")
app.add_typer(creative_app, name="creative")
app.add_typer(ingest_app, name="ingest")
app.add_typer(maint_app, name="maint")
app.add_typer(eval_app, name="eval")
app.add_typer(config_app, name="config")
app.add_typer(benchmark_app, name="benchmark")

console = Console()
_RUNTIME_OVERRIDES: dict[str, object] = {}


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
def init() -> None:
    config, storage, vector_store, *_ = build_services()
    print(f"[green]Initialized metadata DB:[/green] {storage.config.db_path}")
    print(f"[green]Initialized vector DB:[/green] {config.vector_dir}")
    console.print_json(json.dumps(vector_store.info(), ensure_ascii=False, indent=2))


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
    _, storage, _, *_ = build_services()
    if last_recall:
        payload = storage.get_runtime_state("last_recall_trace")
        console.print_json(payload or "{}")
        return
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
def memory_audit() -> None:
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


@app.command("recall")
def recall(
    query: str,
    top_k: int = typer.Option(8),
    task_type: str = typer.Option("trace"),
    explain: bool = typer.Option(False),
) -> None:
    memory_find(query=query, top_k=top_k, task_type=task_type, explain=explain)


@app.command("ask")
def ask(
    query: str,
    task_type: str = typer.Option("qa"),
    temperature: float = typer.Option(0.3, min=0.0, max=1.0),
    max_tokens: int = typer.Option(1400),
) -> None:
    memory_trace(task=query, task_type=task_type, temperature=temperature, max_tokens=max_tokens)


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
            "enable_benchmark_route_tuning": False,
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
def benchmark_smoke_all(out: Optional[str] = typer.Option(None, "--out", help="Output artifact JSON path")) -> None:
    with tempfile.TemporaryDirectory(prefix="dysonspherain_smoke_") as tmp:
        runtime = UnifiedMemoryRuntime.from_base_dir(Path(tmp), config_overrides={
            "vector_backend": "json",
            "embedding_fail_fast": False,
            "enable_benchmark_route_tuning": False,
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
