from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

from .config import AppConfig
from .utils import tokenize


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_nodes (
    id TEXT PRIMARY KEY,
    shell INTEGER NOT NULL,
    sector TEXT NOT NULL,
    zone TEXT NOT NULL,
    cell TEXT NOT NULL,
    molecular_type TEXT NOT NULL,
    content_hash TEXT,
    normalized_hash TEXT,
    theta REAL,
    phi REAL,
    summary TEXT,
    content_ref TEXT,
    raw_content TEXT,
    scope TEXT DEFAULT 'global',
    workspace TEXT,
    project TEXT,
    session_id TEXT,
    source_type TEXT DEFAULT 'memory_note',
    source_ref TEXT,
    extraction_method TEXT DEFAULT 'manual',
    confidence REAL DEFAULT 0.0,
    verification_status TEXT DEFAULT 'unverified',
    metadata_json TEXT,
    seen_count INTEGER DEFAULT 1,
    last_seen_at TEXT,
    base_node_id TEXT,
    delta_summary TEXT,
    changed_fields_json TEXT,
    retrieval_summary TEXT,
    structured_summary TEXT,
    retrieval_signature TEXT,
    time_bucket TEXT,
    entity_tags TEXT,
    task_type_tag TEXT,
    importance REAL DEFAULT 0.0,
    creative_score REAL DEFAULT 0.0,
    stability_score REAL DEFAULT 0.0,
    access_count INTEGER DEFAULT 0,
    compression_level TEXT,
    stage TEXT DEFAULT 'long_term',
    tags TEXT,
    created_at TEXT,
    updated_at TEXT,
    last_accessed_at TEXT
);

CREATE TABLE IF NOT EXISTS memory_chunks (
    chunk_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    grain TEXT DEFAULT 'micro',
    text TEXT NOT NULL,
    content_hash TEXT,
    scope TEXT DEFAULT 'global',
    workspace TEXT,
    project TEXT,
    session_id TEXT,
    retrieval_summary TEXT,
    structured_summary TEXT,
    retrieval_signature TEXT,
    time_bucket TEXT,
    entity_tags TEXT,
    task_type_tag TEXT,
    token_estimate INTEGER DEFAULT 0,
    source_kind TEXT DEFAULT 'raw_content',
    source_path TEXT,
    source_type TEXT DEFAULT 'memory_note',
    source_ref TEXT,
    created_at TEXT,
    updated_at TEXT,
    vector_synced_at TEXT,
    FOREIGN KEY(node_id) REFERENCES memory_nodes(id)
);

CREATE TABLE IF NOT EXISTS memory_objects (
    object_id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    scope TEXT DEFAULT 'global',
    workspace TEXT,
    project TEXT,
    subject TEXT,
    predicate TEXT,
    object_text TEXT NOT NULL,
    polarity REAL,
    entity TEXT,
    attribute TEXT,
    old_value TEXT,
    new_value TEXT,
    event_text TEXT,
    canonical_key TEXT,
    temporal_marker TEXT,
    sequence_index INTEGER,
    source_unit_text TEXT,
    content_hash TEXT,
    confidence REAL DEFAULT 0.0,
    source_chunk_id TEXT,
    source_node_id TEXT,
    session_id TEXT,
    status TEXT DEFAULT 'active',
    turn_index INTEGER,
    timestamp TEXT,
    snapshot_key TEXT,
    merge_policy TEXT,
    source_type TEXT DEFAULT 'memory_extraction',
    source_ref TEXT,
    extraction_method TEXT DEFAULT 'heuristic',
    verification_status TEXT DEFAULT 'unverified',
    related_artifact_ids_json TEXT,
    metadata_json TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS chunk_neighbors (
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    neighbor_chunk_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS memory_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    semantic_weight REAL DEFAULT 0.0,
    task_weight REAL DEFAULT 0.0,
    temporal_weight REAL DEFAULT 0.0,
    causal_weight REAL DEFAULT 0.0,
    creative_weight REAL DEFAULT 0.0,
    structural_weight REAL DEFAULT 0.0,
    last_activated_at TEXT
);

CREATE TABLE IF NOT EXISTS zone_index (
    zone_id TEXT PRIMARY KEY,
    shell INTEGER NOT NULL,
    sector TEXT NOT NULL,
    zone TEXT NOT NULL,
    zone_summary TEXT,
    centroid_theta REAL,
    centroid_phi REAL,
    item_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS task_routes (
    route_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    preferred_shells TEXT,
    preferred_sectors TEXT,
    creative_temperature REAL DEFAULT 0.0,
    compression_policy TEXT
);

CREATE TABLE IF NOT EXISTS activation_logs (
    task_id TEXT PRIMARY KEY,
    task_type TEXT,
    main_nodes TEXT,
    reflected_nodes TEXT,
    refracted_nodes TEXT,
    final_used_nodes TEXT,
    token_cost_input INTEGER,
    token_cost_output INTEGER,
    quality_feedback REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS ingest_files (
    source_path TEXT PRIMARY KEY,
    file_hash TEXT,
    size_bytes INTEGER,
    modified_at REAL,
    node_id TEXT,
    last_ingested_at TEXT,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS vector_sync_state (
    collection_name TEXT NOT NULL,
    item_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    synced_at TEXT,
    PRIMARY KEY(collection_name, item_id)
);

CREATE TABLE IF NOT EXISTS runtime_state (
    state_key TEXT PRIMARY KEY,
    state_value TEXT
);

CREATE TABLE IF NOT EXISTS memory_deltas (
    delta_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    base_node_id TEXT NOT NULL,
    object_type TEXT,
    changed_fields_json TEXT,
    delta_summary TEXT,
    effective_time TEXT,
    merge_policy TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS memory_representations (
    representation_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL,
    parent_type TEXT NOT NULL,
    proxy_kind TEXT NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    scope TEXT DEFAULT 'global',
    workspace TEXT,
    project TEXT,
    session_id TEXT,
    time_bucket TEXT,
    entity_tags TEXT,
    task_type_tag TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_cache (
    query_fingerprint TEXT PRIMARY KEY,
    normalized_query TEXT NOT NULL,
    task_type TEXT NOT NULL,
    route_type TEXT,
    memory_version INTEGER DEFAULT 0,
    payload_json TEXT NOT NULL,
    created_at TEXT,
    hit_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS completion_cache (
    cache_key TEXT PRIMARY KEY,
    evidence_signature TEXT NOT NULL,
    task_type TEXT NOT NULL,
    memory_version INTEGER DEFAULT 0,
    payload_json TEXT NOT NULL,
    created_at TEXT,
    hit_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS profile_snapshots (
    snapshot_type TEXT NOT NULL,
    snapshot_key TEXT NOT NULL,
    memory_version INTEGER DEFAULT 0,
    payload_json TEXT NOT NULL,
    source_object_ids TEXT,
    updated_at TEXT,
    PRIMARY KEY(snapshot_type, snapshot_key)
);

CREATE TABLE IF NOT EXISTS artifact_registry (
    artifact_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    scope TEXT DEFAULT 'global',
    workspace TEXT,
    project TEXT,
    session_id TEXT,
    title TEXT,
    summary TEXT,
    tags_json TEXT,
    source_type TEXT DEFAULT 'file',
    source_ref TEXT,
    related_memory_ids_json TEXT,
    related_object_ids_json TEXT,
    related_goal_ids_json TEXT,
    metadata_json TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS open_loops (
    loop_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    details TEXT,
    status TEXT DEFAULT 'open',
    scope TEXT DEFAULT 'global',
    workspace TEXT,
    project TEXT,
    session_id TEXT,
    priority TEXT DEFAULT 'normal',
    tags_json TEXT,
    blocked_reason TEXT,
    source_type TEXT DEFAULT 'manual',
    source_ref TEXT,
    related_memory_ids_json TEXT,
    related_artifact_ids_json TEXT,
    metadata_json TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_chunks_node_id ON memory_chunks(node_id);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_content_hash ON memory_chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_time_bucket ON memory_chunks(time_bucket);
CREATE INDEX IF NOT EXISTS idx_memory_objects_source_node_id ON memory_objects(source_node_id);
CREATE INDEX IF NOT EXISTS idx_memory_objects_source_chunk_id ON memory_objects(source_chunk_id);
CREATE INDEX IF NOT EXISTS idx_memory_objects_canonical_key ON memory_objects(canonical_key);
CREATE INDEX IF NOT EXISTS idx_memory_objects_content_hash ON memory_objects(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_normalized_hash ON memory_nodes(normalized_hash);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_base_node_id ON memory_nodes(base_node_id);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_scope_project ON memory_nodes(scope, project, session_id);
CREATE INDEX IF NOT EXISTS idx_chunk_neighbors_chunk_id ON chunk_neighbors(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_neighbors_neighbor_chunk_id ON chunk_neighbors(neighbor_chunk_id);
CREATE INDEX IF NOT EXISTS idx_vector_sync_state_collection_hash ON vector_sync_state(collection_name, content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_representations_parent ON memory_representations(parent_type, parent_id);
CREATE INDEX IF NOT EXISTS idx_memory_representations_proxy_kind ON memory_representations(proxy_kind, time_bucket);
CREATE INDEX IF NOT EXISTS idx_profile_snapshots_version ON profile_snapshots(snapshot_type, memory_version);
CREATE INDEX IF NOT EXISTS idx_artifact_registry_path ON artifact_registry(path);
CREATE INDEX IF NOT EXISTS idx_artifact_registry_scope_project ON artifact_registry(scope, project, session_id);
CREATE INDEX IF NOT EXISTS idx_open_loops_status_project ON open_loops(status, project, session_id);
""" 


class Storage:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._persistent_conn: sqlite3.Connection | None = None
        self._stats: dict[str, dict[str, float | int]] = {}

    def _record_stat(self, operation: str, elapsed_ms: float, rows: int | None = None) -> None:
        bucket = self._stats.setdefault(operation, {"calls": 0, "total_ms": 0.0, "rows": 0})
        bucket["calls"] = int(bucket["calls"]) + 1
        bucket["total_ms"] = float(bucket["total_ms"]) + float(elapsed_ms)
        if rows is not None:
            bucket["rows"] = int(bucket["rows"]) + int(rows)

    def snapshot_stats(self, reset: bool = False) -> dict[str, Any]:
        ops = {
            name: {
                "calls": int(values.get("calls", 0)),
                "rows": int(values.get("rows", 0)),
                "total_ms": round(float(values.get("total_ms", 0.0)), 2),
            }
            for name, values in self._stats.items()
        }
        snapshot = {
            "total_ms": round(sum(float(values["total_ms"]) for values in ops.values()), 2),
            "calls": sum(int(values["calls"]) for values in ops.values()),
            "rows": sum(int(values["rows"]) for values in ops.values()),
            "ops": ops,
        }
        if reset:
            self._stats = {}
        return snapshot

    def open_persistent(self) -> None:
        """Open a persistent connection for batch operations."""
        if self._persistent_conn is None:
            self._persistent_conn = sqlite3.connect(self.config.db_path, timeout=30.0)
            self._persistent_conn.row_factory = sqlite3.Row
            self._persistent_conn.execute("PRAGMA journal_mode=WAL")
            self._persistent_conn.execute("PRAGMA busy_timeout=30000")
            self._persistent_conn.execute("PRAGMA foreign_keys=ON")

    def close_persistent(self) -> None:
        """Close the persistent connection."""
        if self._persistent_conn is not None:
            self._persistent_conn.commit()
            self._persistent_conn.close()
            self._persistent_conn = None

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self._persistent_conn is not None:
            yield self._persistent_conn
            self._persistent_conn.commit()
            return
        conn = sqlite3.connect(self.config.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        self.config.ensure_dirs()
        started = perf_counter()
        with self.connect() as conn:
            self._apply_schema_statements(conn, include_indexes=False)
            self._ensure_column(conn, "memory_nodes", "content_hash", "TEXT")
            self._ensure_column(conn, "memory_nodes", "normalized_hash", "TEXT")
            self._ensure_column(conn, "memory_nodes", "scope", "TEXT DEFAULT 'global'")
            self._ensure_column(conn, "memory_nodes", "workspace", "TEXT")
            self._ensure_column(conn, "memory_nodes", "project", "TEXT")
            self._ensure_column(conn, "memory_nodes", "session_id", "TEXT")
            self._ensure_column(conn, "memory_nodes", "source_type", "TEXT DEFAULT 'memory_note'")
            self._ensure_column(conn, "memory_nodes", "source_ref", "TEXT")
            self._ensure_column(conn, "memory_nodes", "extraction_method", "TEXT DEFAULT 'manual'")
            self._ensure_column(conn, "memory_nodes", "confidence", "REAL DEFAULT 0.0")
            self._ensure_column(conn, "memory_nodes", "verification_status", "TEXT DEFAULT 'unverified'")
            self._ensure_column(conn, "memory_nodes", "metadata_json", "TEXT")
            self._ensure_column(conn, "memory_nodes", "seen_count", "INTEGER DEFAULT 1")
            self._ensure_column(conn, "memory_nodes", "last_seen_at", "TEXT")
            self._ensure_column(conn, "memory_nodes", "base_node_id", "TEXT")
            self._ensure_column(conn, "memory_nodes", "delta_summary", "TEXT")
            self._ensure_column(conn, "memory_nodes", "changed_fields_json", "TEXT")
            self._ensure_column(conn, "memory_nodes", "retrieval_summary", "TEXT")
            self._ensure_column(conn, "memory_nodes", "structured_summary", "TEXT")
            self._ensure_column(conn, "memory_nodes", "retrieval_signature", "TEXT")
            self._ensure_column(conn, "memory_nodes", "time_bucket", "TEXT")
            self._ensure_column(conn, "memory_nodes", "entity_tags", "TEXT")
            self._ensure_column(conn, "memory_nodes", "task_type_tag", "TEXT")
            self._ensure_column(conn, "memory_nodes", "updated_at", "TEXT")
            self._ensure_column(conn, "memory_chunks", "source_path", "TEXT")
            self._ensure_column(conn, "memory_chunks", "grain", "TEXT DEFAULT 'micro'")
            self._ensure_column(conn, "memory_chunks", "content_hash", "TEXT")
            self._ensure_column(conn, "memory_chunks", "scope", "TEXT DEFAULT 'global'")
            self._ensure_column(conn, "memory_chunks", "workspace", "TEXT")
            self._ensure_column(conn, "memory_chunks", "project", "TEXT")
            self._ensure_column(conn, "memory_chunks", "session_id", "TEXT")
            self._ensure_column(conn, "memory_chunks", "retrieval_summary", "TEXT")
            self._ensure_column(conn, "memory_chunks", "structured_summary", "TEXT")
            self._ensure_column(conn, "memory_chunks", "retrieval_signature", "TEXT")
            self._ensure_column(conn, "memory_chunks", "time_bucket", "TEXT")
            self._ensure_column(conn, "memory_chunks", "entity_tags", "TEXT")
            self._ensure_column(conn, "memory_chunks", "task_type_tag", "TEXT")
            self._ensure_column(conn, "memory_chunks", "source_type", "TEXT DEFAULT 'memory_note'")
            self._ensure_column(conn, "memory_chunks", "source_ref", "TEXT")
            self._ensure_column(conn, "memory_chunks", "updated_at", "TEXT")
            self._ensure_column(conn, "memory_objects", "canonical_key", "TEXT")
            self._ensure_column(conn, "memory_objects", "temporal_marker", "TEXT")
            self._ensure_column(conn, "memory_objects", "sequence_index", "INTEGER")
            self._ensure_column(conn, "memory_objects", "source_unit_text", "TEXT")
            self._ensure_column(conn, "memory_objects", "content_hash", "TEXT")
            self._ensure_column(conn, "memory_objects", "scope", "TEXT DEFAULT 'global'")
            self._ensure_column(conn, "memory_objects", "workspace", "TEXT")
            self._ensure_column(conn, "memory_objects", "project", "TEXT")
            self._ensure_column(conn, "memory_objects", "snapshot_key", "TEXT")
            self._ensure_column(conn, "memory_objects", "merge_policy", "TEXT")
            self._ensure_column(conn, "memory_objects", "status", "TEXT DEFAULT 'active'")
            self._ensure_column(conn, "memory_objects", "source_type", "TEXT DEFAULT 'memory_extraction'")
            self._ensure_column(conn, "memory_objects", "source_ref", "TEXT")
            self._ensure_column(conn, "memory_objects", "extraction_method", "TEXT DEFAULT 'heuristic'")
            self._ensure_column(conn, "memory_objects", "verification_status", "TEXT DEFAULT 'unverified'")
            self._ensure_column(conn, "memory_objects", "related_artifact_ids_json", "TEXT")
            self._ensure_column(conn, "memory_objects", "metadata_json", "TEXT")
            self._ensure_column(conn, "memory_objects", "updated_at", "TEXT")
            self._ensure_column(conn, "memory_representations", "scope", "TEXT DEFAULT 'global'")
            self._ensure_column(conn, "memory_representations", "workspace", "TEXT")
            self._ensure_column(conn, "memory_representations", "project", "TEXT")
            self._ensure_column(conn, "memory_representations", "session_id", "TEXT")
            self._ensure_column(conn, "memory_representations", "time_bucket", "TEXT")
            self._ensure_column(conn, "memory_representations", "entity_tags", "TEXT")
            self._ensure_column(conn, "memory_representations", "task_type_tag", "TEXT")
            self._ensure_fts_tables(conn)
            self._apply_schema_statements(conn, include_indexes=True)
            conn.execute(
                """
                INSERT INTO runtime_state (state_key, state_value)
                VALUES ('memory_version', '0')
                ON CONFLICT(state_key) DO NOTHING
                """
            )
        self._record_stat("init_db", (perf_counter() - started) * 1000.0)

    def _ensure_column(self, conn: sqlite3.Connection, table_name: str, column_name: str, column_type: str) -> None:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        if column_name not in cols:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    @staticmethod
    def _apply_schema_statements(conn: sqlite3.Connection, *, include_indexes: bool) -> None:
        for raw_statement in SCHEMA_SQL.split(";"):
            statement = raw_statement.strip()
            if not statement:
                continue
            is_index = statement.upper().startswith("CREATE INDEX")
            if is_index != include_indexes:
                continue
            conn.execute(statement)

    @staticmethod
    def _node_payload(node: dict[str, Any]) -> dict[str, Any]:
        payload = dict(node)
        payload.setdefault("content_hash", None)
        payload.setdefault("normalized_hash", None)
        payload.setdefault("theta", None)
        payload.setdefault("phi", None)
        payload.setdefault("content_ref", None)
        payload.setdefault("raw_content", None)
        payload.setdefault("scope", "global")
        payload.setdefault("workspace", None)
        payload.setdefault("project", None)
        payload.setdefault("session_id", None)
        payload.setdefault("source_type", "memory_note")
        payload.setdefault("source_ref", None)
        payload.setdefault("extraction_method", "manual")
        payload.setdefault("confidence", 0.0)
        payload.setdefault("verification_status", "unverified")
        payload.setdefault("metadata_json", None)
        payload.setdefault("seen_count", 1)
        payload.setdefault("last_seen_at", payload.get("last_accessed_at") or payload.get("created_at"))
        payload.setdefault("base_node_id", None)
        payload.setdefault("delta_summary", None)
        payload.setdefault("changed_fields_json", None)
        payload.setdefault("retrieval_summary", None)
        payload.setdefault("structured_summary", None)
        payload.setdefault("retrieval_signature", None)
        payload.setdefault("time_bucket", None)
        payload.setdefault("entity_tags", None)
        payload.setdefault("task_type_tag", None)
        payload.setdefault("importance", 0.0)
        payload.setdefault("creative_score", 0.0)
        payload.setdefault("stability_score", 0.0)
        payload.setdefault("access_count", 0)
        payload.setdefault("compression_level", "medium")
        payload.setdefault("stage", "long_term")
        payload.setdefault("tags", None)
        payload.setdefault("created_at", None)
        payload.setdefault("updated_at", payload.get("created_at"))
        payload.setdefault("last_accessed_at", payload.get("created_at"))
        return payload

    @staticmethod
    def _chunk_payload(chunk: dict[str, Any]) -> dict[str, Any]:
        payload = dict(chunk)
        payload.setdefault("grain", "micro")
        payload.setdefault("content_hash", None)
        payload.setdefault("scope", "global")
        payload.setdefault("workspace", None)
        payload.setdefault("project", None)
        payload.setdefault("session_id", None)
        payload.setdefault("retrieval_summary", None)
        payload.setdefault("structured_summary", None)
        payload.setdefault("retrieval_signature", None)
        payload.setdefault("time_bucket", None)
        payload.setdefault("entity_tags", None)
        payload.setdefault("task_type_tag", None)
        payload.setdefault("token_estimate", 0)
        payload.setdefault("source_kind", "raw_content")
        payload.setdefault("source_path", None)
        payload.setdefault("source_type", "memory_note")
        payload.setdefault("source_ref", None)
        payload.setdefault("created_at", None)
        payload.setdefault("updated_at", payload.get("created_at"))
        payload.setdefault("vector_synced_at", None)
        return payload

    @staticmethod
    def _object_payload(obj: dict[str, Any]) -> dict[str, Any]:
        payload = dict(obj)
        payload.setdefault("scope", "global")
        payload.setdefault("workspace", None)
        payload.setdefault("project", None)
        payload.setdefault("subject", None)
        payload.setdefault("predicate", None)
        payload.setdefault("polarity", None)
        payload.setdefault("entity", None)
        payload.setdefault("attribute", None)
        payload.setdefault("old_value", None)
        payload.setdefault("new_value", None)
        payload.setdefault("event_text", None)
        payload.setdefault("canonical_key", None)
        payload.setdefault("temporal_marker", None)
        payload.setdefault("sequence_index", None)
        payload.setdefault("source_unit_text", None)
        payload.setdefault("content_hash", None)
        payload.setdefault("confidence", 0.0)
        payload.setdefault("source_chunk_id", None)
        payload.setdefault("source_node_id", None)
        payload.setdefault("session_id", None)
        payload.setdefault("status", "active")
        payload.setdefault("turn_index", None)
        payload.setdefault("timestamp", None)
        payload.setdefault("snapshot_key", None)
        payload.setdefault("merge_policy", None)
        payload.setdefault("source_type", "memory_extraction")
        payload.setdefault("source_ref", None)
        payload.setdefault("extraction_method", "heuristic")
        payload.setdefault("verification_status", "unverified")
        payload.setdefault("related_artifact_ids_json", None)
        payload.setdefault("metadata_json", None)
        payload.setdefault("created_at", None)
        payload.setdefault("updated_at", payload.get("created_at"))
        return payload

    @staticmethod
    def _artifact_payload(artifact: dict[str, Any]) -> dict[str, Any]:
        payload = dict(artifact)
        payload.setdefault("scope", "global")
        payload.setdefault("workspace", None)
        payload.setdefault("project", None)
        payload.setdefault("session_id", None)
        payload.setdefault("title", None)
        payload.setdefault("summary", None)
        payload.setdefault("tags_json", None)
        payload.setdefault("source_type", "file")
        payload.setdefault("source_ref", None)
        payload.setdefault("related_memory_ids_json", None)
        payload.setdefault("related_object_ids_json", None)
        payload.setdefault("related_goal_ids_json", None)
        payload.setdefault("metadata_json", None)
        payload.setdefault("created_at", None)
        payload.setdefault("updated_at", payload.get("created_at"))
        return payload

    @staticmethod
    def _open_loop_payload(loop: dict[str, Any]) -> dict[str, Any]:
        payload = dict(loop)
        payload.setdefault("details", None)
        payload.setdefault("status", "open")
        payload.setdefault("scope", "global")
        payload.setdefault("workspace", None)
        payload.setdefault("project", None)
        payload.setdefault("session_id", None)
        payload.setdefault("priority", "normal")
        payload.setdefault("tags_json", None)
        payload.setdefault("blocked_reason", None)
        payload.setdefault("source_type", "manual")
        payload.setdefault("source_ref", None)
        payload.setdefault("related_memory_ids_json", None)
        payload.setdefault("related_artifact_ids_json", None)
        payload.setdefault("metadata_json", None)
        payload.setdefault("created_at", None)
        payload.setdefault("updated_at", payload.get("created_at"))
        return payload

    def insert_node(self, node: dict[str, Any]) -> None:
        started = perf_counter()
        payload = self._node_payload(node)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_nodes (
                    id, shell, sector, zone, cell, molecular_type,
                    content_hash, normalized_hash, theta, phi, summary, content_ref, raw_content,
                    scope, workspace, project, session_id, source_type, source_ref, extraction_method, confidence, verification_status, metadata_json,
                    seen_count, last_seen_at, base_node_id, delta_summary, changed_fields_json,
                    retrieval_summary, structured_summary, retrieval_signature, time_bucket, entity_tags, task_type_tag,
                    importance, creative_score, stability_score,
                    access_count, compression_level, stage, tags,
                    created_at, updated_at, last_accessed_at
                ) VALUES (
                    :id, :shell, :sector, :zone, :cell, :molecular_type,
                    :content_hash, :normalized_hash, :theta, :phi, :summary, :content_ref, :raw_content,
                    :scope, :workspace, :project, :session_id, :source_type, :source_ref, :extraction_method, :confidence, :verification_status, :metadata_json,
                    :seen_count, :last_seen_at, :base_node_id, :delta_summary, :changed_fields_json,
                    :retrieval_summary, :structured_summary, :retrieval_signature, :time_bucket, :entity_tags, :task_type_tag,
                    :importance, :creative_score, :stability_score,
                    :access_count, :compression_level, :stage, :tags,
                    :created_at, :updated_at, :last_accessed_at
                )
                ON CONFLICT(id) DO UPDATE SET
                    shell=excluded.shell,
                    sector=excluded.sector,
                    zone=excluded.zone,
                    cell=excluded.cell,
                    molecular_type=excluded.molecular_type,
                    content_hash=excluded.content_hash,
                    normalized_hash=excluded.normalized_hash,
                    theta=excluded.theta,
                    phi=excluded.phi,
                    summary=excluded.summary,
                    content_ref=excluded.content_ref,
                    raw_content=excluded.raw_content,
                    scope=excluded.scope,
                    workspace=excluded.workspace,
                    project=excluded.project,
                    session_id=excluded.session_id,
                    source_type=excluded.source_type,
                    source_ref=excluded.source_ref,
                    extraction_method=excluded.extraction_method,
                    confidence=excluded.confidence,
                    verification_status=excluded.verification_status,
                    metadata_json=excluded.metadata_json,
                    seen_count=excluded.seen_count,
                    last_seen_at=excluded.last_seen_at,
                    base_node_id=excluded.base_node_id,
                    delta_summary=excluded.delta_summary,
                    changed_fields_json=excluded.changed_fields_json,
                    retrieval_summary=excluded.retrieval_summary,
                    structured_summary=excluded.structured_summary,
                    retrieval_signature=excluded.retrieval_signature,
                    time_bucket=excluded.time_bucket,
                    entity_tags=excluded.entity_tags,
                    task_type_tag=excluded.task_type_tag,
                    importance=excluded.importance,
                    creative_score=excluded.creative_score,
                    stability_score=excluded.stability_score,
                    access_count=excluded.access_count,
                    compression_level=excluded.compression_level,
                    stage=excluded.stage,
                    tags=excluded.tags,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    last_accessed_at=excluded.last_accessed_at
                """,
                payload,
            )
        self._record_stat("insert_node", (perf_counter() - started) * 1000.0, rows=1)

    def insert_nodes(self, nodes: list[dict[str, Any]]) -> None:
        if not nodes:
            return
        started = perf_counter()
        payload = [self._node_payload(node) for node in nodes]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO memory_nodes (
                    id, shell, sector, zone, cell, molecular_type,
                    content_hash, normalized_hash, theta, phi, summary, content_ref, raw_content,
                    scope, workspace, project, session_id, source_type, source_ref, extraction_method, confidence, verification_status, metadata_json,
                    seen_count, last_seen_at, base_node_id, delta_summary, changed_fields_json,
                    retrieval_summary, structured_summary, retrieval_signature, time_bucket, entity_tags, task_type_tag,
                    importance, creative_score, stability_score,
                    access_count, compression_level, stage, tags,
                    created_at, updated_at, last_accessed_at
                ) VALUES (
                    :id, :shell, :sector, :zone, :cell, :molecular_type,
                    :content_hash, :normalized_hash, :theta, :phi, :summary, :content_ref, :raw_content,
                    :scope, :workspace, :project, :session_id, :source_type, :source_ref, :extraction_method, :confidence, :verification_status, :metadata_json,
                    :seen_count, :last_seen_at, :base_node_id, :delta_summary, :changed_fields_json,
                    :retrieval_summary, :structured_summary, :retrieval_signature, :time_bucket, :entity_tags, :task_type_tag,
                    :importance, :creative_score, :stability_score,
                    :access_count, :compression_level, :stage, :tags,
                    :created_at, :updated_at, :last_accessed_at
                )
                ON CONFLICT(id) DO UPDATE SET
                    shell=excluded.shell,
                    sector=excluded.sector,
                    zone=excluded.zone,
                    cell=excluded.cell,
                    molecular_type=excluded.molecular_type,
                    content_hash=excluded.content_hash,
                    normalized_hash=excluded.normalized_hash,
                    theta=excluded.theta,
                    phi=excluded.phi,
                    summary=excluded.summary,
                    content_ref=excluded.content_ref,
                    raw_content=excluded.raw_content,
                    scope=excluded.scope,
                    workspace=excluded.workspace,
                    project=excluded.project,
                    session_id=excluded.session_id,
                    source_type=excluded.source_type,
                    source_ref=excluded.source_ref,
                    extraction_method=excluded.extraction_method,
                    confidence=excluded.confidence,
                    verification_status=excluded.verification_status,
                    metadata_json=excluded.metadata_json,
                    seen_count=excluded.seen_count,
                    last_seen_at=excluded.last_seen_at,
                    base_node_id=excluded.base_node_id,
                    delta_summary=excluded.delta_summary,
                    changed_fields_json=excluded.changed_fields_json,
                    retrieval_summary=excluded.retrieval_summary,
                    structured_summary=excluded.structured_summary,
                    retrieval_signature=excluded.retrieval_signature,
                    time_bucket=excluded.time_bucket,
                    entity_tags=excluded.entity_tags,
                    task_type_tag=excluded.task_type_tag,
                    importance=excluded.importance,
                    creative_score=excluded.creative_score,
                    stability_score=excluded.stability_score,
                    access_count=excluded.access_count,
                    compression_level=excluded.compression_level,
                    stage=excluded.stage,
                    tags=excluded.tags,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    last_accessed_at=excluded.last_accessed_at
                """,
                payload,
            )
        self._record_stat("insert_nodes", (perf_counter() - started) * 1000.0, rows=len(nodes))

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        started = perf_counter()
        payload = [self._chunk_payload(chunk) for chunk in chunks]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO memory_chunks (
                    chunk_id, node_id, chunk_index, grain, text, content_hash,
                    scope, workspace, project, session_id,
                    retrieval_summary, structured_summary, retrieval_signature, time_bucket, entity_tags, task_type_tag,
                    token_estimate,
                    source_kind, source_path, source_type, source_ref, created_at, updated_at, vector_synced_at
                ) VALUES (
                    :chunk_id, :node_id, :chunk_index, :grain, :text, :content_hash,
                    :scope, :workspace, :project, :session_id,
                    :retrieval_summary, :structured_summary, :retrieval_signature, :time_bucket, :entity_tags, :task_type_tag,
                    :token_estimate,
                    :source_kind, :source_path, :source_type, :source_ref, :created_at, :updated_at, :vector_synced_at
                )
                """,
                payload,
            )
            self._upsert_chunk_fts(conn, payload)
        self._record_stat("insert_chunks", (perf_counter() - started) * 1000.0, rows=len(chunks))

    def insert_objects(self, objects: list[dict[str, Any]]) -> None:
        if not objects:
            return
        started = perf_counter()
        payload = [self._object_payload(obj) for obj in objects]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO memory_objects (
                    object_id, object_type, scope, workspace, project, subject, predicate, object_text, polarity,
                    entity, attribute, old_value, new_value, event_text, canonical_key,
                    temporal_marker, sequence_index, source_unit_text, content_hash, confidence,
                    source_chunk_id, source_node_id, session_id, status, turn_index, timestamp, snapshot_key, merge_policy,
                    source_type, source_ref, extraction_method, verification_status, related_artifact_ids_json, metadata_json,
                    created_at, updated_at
                ) VALUES (
                    :object_id, :object_type, :scope, :workspace, :project, :subject, :predicate, :object_text, :polarity,
                    :entity, :attribute, :old_value, :new_value, :event_text, :canonical_key,
                    :temporal_marker, :sequence_index, :source_unit_text, :content_hash, :confidence,
                    :source_chunk_id, :source_node_id, :session_id, :status, :turn_index, :timestamp, :snapshot_key, :merge_policy,
                    :source_type, :source_ref, :extraction_method, :verification_status, :related_artifact_ids_json, :metadata_json,
                    :created_at, :updated_at
                )
                """,
                payload,
            )
            self._upsert_object_fts(conn, payload)
        self._record_stat("insert_objects", (perf_counter() - started) * 1000.0, rows=len(objects))

    def insert_chunk_neighbors(self, neighbors: list[dict[str, Any]]) -> None:
        if not neighbors:
            return
        started = perf_counter()
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_neighbors (
                    id, chunk_id, neighbor_chunk_id, relation_type, weight
                ) VALUES (
                    :id, :chunk_id, :neighbor_chunk_id, :relation_type, :weight
                )
                """,
                neighbors,
            )
        self._record_stat("insert_chunk_neighbors", (perf_counter() - started) * 1000.0, rows=len(neighbors))

    def delete_chunks_for_node(self, node_id: str) -> list[str]:
        rows = self.fetch_chunks_for_node(node_id)
        chunk_ids = [r['chunk_id'] for r in rows]
        with self.connect() as conn:
            conn.execute('DELETE FROM memory_chunks WHERE node_id = ?', (node_id,))
            self._delete_chunk_fts(conn, chunk_ids)
        return chunk_ids

    def delete_objects_for_node(self, node_id: str) -> list[str]:
        rows = self.fetch_objects('source_node_id = ?', (node_id,))
        object_ids = [r['object_id'] for r in rows]
        with self.connect() as conn:
            conn.execute('DELETE FROM memory_objects WHERE source_node_id = ?', (node_id,))
            self._delete_object_fts(conn, object_ids)
        return object_ids

    def delete_chunk_neighbors(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        placeholders = ','.join(['?'] * len(chunk_ids))
        with self.connect() as conn:
            conn.execute(
                f'DELETE FROM chunk_neighbors WHERE chunk_id IN ({placeholders}) OR neighbor_chunk_id IN ({placeholders})',
                (*chunk_ids, *chunk_ids),
            )

    def delete_node(self, node_id: str) -> None:
        with self.connect() as conn:
            conn.execute('DELETE FROM memory_nodes WHERE id = ?', (node_id,))
            conn.execute('DELETE FROM memory_objects WHERE source_node_id = ?', (node_id,))
            conn.execute('DELETE FROM memory_edges WHERE source_id = ? OR target_id = ?', (node_id, node_id))
            conn.execute('DELETE FROM memory_deltas WHERE node_id = ? OR base_node_id = ?', (node_id, node_id))
            conn.execute('DELETE FROM memory_representations WHERE parent_id = ? OR parent_id IN (SELECT chunk_id FROM memory_chunks WHERE node_id = ?)', (node_id, node_id))
            try:
                conn.execute('DELETE FROM memory_chunks_fts WHERE node_id = ?', (node_id,))
                conn.execute('DELETE FROM memory_objects_fts WHERE source_node_id = ?', (node_id,))
            except sqlite3.OperationalError:
                pass

    def insert_edge(self, edge: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_edges (
                    id, source_id, target_id, semantic_weight, task_weight,
                    temporal_weight, causal_weight, creative_weight,
                    structural_weight, last_activated_at
                ) VALUES (
                    :id, :source_id, :target_id, :semantic_weight, :task_weight,
                    :temporal_weight, :causal_weight, :creative_weight,
                    :structural_weight, :last_activated_at
                )
                """,
                edge,
            )

    def upsert_zone_index(
        self,
        zone_id: str,
        shell: int,
        sector: str,
        zone: str,
        zone_summary: str,
        centroid_theta: float | None,
        centroid_phi: float | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO zone_index (
                    zone_id, shell, sector, zone, zone_summary,
                    centroid_theta, centroid_phi, item_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(zone_id) DO UPDATE SET
                    zone_summary=excluded.zone_summary,
                    centroid_theta=excluded.centroid_theta,
                    centroid_phi=excluded.centroid_phi,
                    item_count=item_count+1
                """,
                (zone_id, shell, sector, zone, zone_summary, centroid_theta, centroid_phi),
            )

    def upsert_ingest_file(
        self,
        source_path: str,
        file_hash: str,
        size_bytes: int,
        modified_at: float,
        node_id: str,
        last_ingested_at: str,
        status: str = 'active',
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ingest_files (
                    source_path, file_hash, size_bytes, modified_at, node_id, last_ingested_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    file_hash=excluded.file_hash,
                    size_bytes=excluded.size_bytes,
                    modified_at=excluded.modified_at,
                    node_id=excluded.node_id,
                    last_ingested_at=excluded.last_ingested_at,
                    status=excluded.status
                """,
                (source_path, file_hash, size_bytes, modified_at, node_id, last_ingested_at, status),
            )

    def fetch_ingest_file(self, source_path: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM ingest_files WHERE source_path = ?', (source_path,)).fetchone()
        return dict(row) if row else None

    def fetch_ingest_files(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute('SELECT * FROM ingest_files ORDER BY last_ingested_at DESC').fetchall()
        return [dict(r) for r in rows]

    def fetch_nodes(self, where: str = '', params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        query = 'SELECT * FROM memory_nodes'
        if where:
            query += f' WHERE {where}'
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        self._record_stat("fetch_nodes", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(r) for r in rows]

    def fetch_node_by_id(self, node_id: str) -> dict[str, Any] | None:
        rows = self.fetch_nodes('id = ?', (node_id,))
        return rows[0] if rows else None

    def fetch_nodes_by_ids(self, node_ids: list[str]) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        unique_node_ids = list(dict.fromkeys(node_ids))
        placeholders = ','.join(['?'] * len(unique_node_ids))
        query = f'SELECT * FROM memory_nodes WHERE id IN ({placeholders})'
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(query, tuple(unique_node_ids)).fetchall()
        self._record_stat("fetch_nodes_by_ids", (perf_counter() - started) * 1000.0, rows=len(rows))
        by_id = {row['id']: dict(row) for row in rows}
        return [by_id[node_id] for node_id in node_ids if node_id in by_id]

    def get_nodes_by_ids(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        rows = self.fetch_nodes_by_ids(node_ids)
        return {str(row["id"]): dict(row) for row in rows if row.get("id")}

    def fetch_nodes_by_normalized_hash(self, normalized_hash: str) -> list[dict[str, Any]]:
        if not normalized_hash:
            return []
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM memory_nodes
                WHERE normalized_hash = ?
                ORDER BY created_at DESC, id ASC
                """,
                (normalized_hash,),
            ).fetchall()
        self._record_stat("fetch_nodes_by_normalized_hash", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def fetch_recent_nodes(
        self,
        limit: int = 24,
        *,
        sector: str | None = None,
        zone: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if sector:
            clauses.append("sector = ?")
            params.append(sector)
        if zone:
            clauses.append("zone = ?")
            params.append(zone)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM memory_nodes
                {where}
                ORDER BY created_at DESC, last_seen_at DESC, last_accessed_at DESC
                LIMIT ?
                """,
                (*params, max(1, int(limit))),
            ).fetchall()
        self._record_stat("fetch_recent_nodes", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def update_node_seen(self, node_id: str, timestamp: str, *, salience_boost: float = 0.0) -> None:
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE memory_nodes
                SET seen_count = COALESCE(seen_count, 1) + 1,
                    last_seen_at = ?,
                    access_count = access_count + 1,
                    importance = importance + ?,
                    last_accessed_at = ?
                WHERE id = ?
                """,
                (timestamp, float(salience_boost), timestamp, node_id),
            )
        self._record_stat("update_node_seen", (perf_counter() - started) * 1000.0, rows=1)

    def get_memory_version(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT state_value FROM runtime_state WHERE state_key = 'memory_version'"
            ).fetchone()
        return int(row["state_value"]) if row and row["state_value"] is not None else 0

    def bump_memory_version(self) -> int:
        started = perf_counter()
        with self.connect() as conn:
            current = conn.execute(
                "SELECT state_value FROM runtime_state WHERE state_key = 'memory_version'"
            ).fetchone()
            next_version = int(current["state_value"]) + 1 if current and current["state_value"] is not None else 1
            conn.execute(
                """
                INSERT INTO runtime_state (state_key, state_value)
                VALUES ('memory_version', ?)
                ON CONFLICT(state_key) DO UPDATE SET state_value = excluded.state_value
                """,
                (str(next_version),),
            )
        self._record_stat("bump_memory_version", (perf_counter() - started) * 1000.0, rows=1)
        return next_version

    def set_runtime_state(self, key: str, value: str) -> None:
        if not key:
            return
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_state (state_key, state_value)
                VALUES (?, ?)
                ON CONFLICT(state_key) DO UPDATE SET state_value = excluded.state_value
                """,
                (key, value),
            )
        self._record_stat("set_runtime_state", (perf_counter() - started) * 1000.0, rows=1)

    def get_runtime_state(self, key: str) -> str | None:
        if not key:
            return None
        started = perf_counter()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT state_value FROM runtime_state WHERE state_key = ?",
                (key,),
            ).fetchone()
        self._record_stat("get_runtime_state", (perf_counter() - started) * 1000.0, rows=1 if row else 0)
        return str(row["state_value"]) if row and row["state_value"] is not None else None

    def fetch_chunks_for_node(self, node_id: str) -> list[dict[str, Any]]:
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute('SELECT * FROM memory_chunks WHERE node_id = ? ORDER BY chunk_index ASC', (node_id,)).fetchall()
        self._record_stat("fetch_chunks_for_node", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(r) for r in rows]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        unique_chunk_ids = list(dict.fromkeys(chunk_ids))
        placeholders = ','.join(['?'] * len(unique_chunk_ids))
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f'SELECT * FROM memory_chunks WHERE chunk_id IN ({placeholders})',
                tuple(unique_chunk_ids),
            ).fetchall()
        self._record_stat("fetch_chunks_by_ids", (perf_counter() - started) * 1000.0, rows=len(rows))
        by_id = {row['chunk_id']: dict(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def fetch_all_chunks(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute('SELECT * FROM memory_chunks ORDER BY created_at ASC, chunk_index ASC').fetchall()
        return [dict(r) for r in rows]

    def fetch_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        started = perf_counter()
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM memory_chunks WHERE chunk_id = ?', (chunk_id,)).fetchone()
        self._record_stat("fetch_chunk_by_id", (perf_counter() - started) * 1000.0, rows=1 if row else 0)
        return dict(row) if row else None

    def hydrate_chunk_with_node_metadata(self, chunk_id: str) -> dict[str, Any] | None:
        started = perf_counter()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT c.chunk_id, c.node_id, c.chunk_index, c.grain, c.text, c.content_hash,
                       c.scope, c.workspace, c.project, c.session_id,
                       c.retrieval_summary, c.structured_summary, c.retrieval_signature, c.time_bucket, c.entity_tags, c.task_type_tag,
                       c.token_estimate, c.source_kind, c.source_path, c.source_type, c.source_ref, c.created_at, c.updated_at,
                       n.shell, n.sector, n.zone, n.cell, n.scope AS node_scope, n.workspace AS node_workspace, n.project AS node_project, n.session_id AS node_session_id
                FROM memory_chunks c
                JOIN memory_nodes n ON n.id = c.node_id
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        self._record_stat("hydrate_chunk_with_node_metadata", (perf_counter() - started) * 1000.0, rows=1 if row else 0)
        return dict(row) if row else None

    def hydrate_chunks_with_node_metadata(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        unique_chunk_ids = list(dict.fromkeys(chunk_ids))
        placeholders = ",".join(["?"] * len(unique_chunk_ids))
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.chunk_id, c.node_id, c.chunk_index, c.grain, c.text, c.content_hash,
                       c.scope, c.workspace, c.project, c.session_id,
                       c.retrieval_summary, c.structured_summary, c.retrieval_signature, c.time_bucket, c.entity_tags, c.task_type_tag,
                       c.token_estimate, c.source_kind, c.source_path, c.source_type, c.source_ref, c.created_at, c.updated_at,
                       n.shell, n.sector, n.zone, n.cell, n.scope AS node_scope, n.workspace AS node_workspace, n.project AS node_project, n.session_id AS node_session_id
                FROM memory_chunks c
                JOIN memory_nodes n ON n.id = c.node_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                tuple(unique_chunk_ids),
            ).fetchall()
        self._record_stat("hydrate_chunks_with_node_metadata", (perf_counter() - started) * 1000.0, rows=len(rows))
        by_id = {row["chunk_id"]: dict(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def fetch_chunks_with_node_metadata_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        unique_chunk_ids = list(dict.fromkeys(chunk_ids))
        placeholders = ",".join(["?"] * len(unique_chunk_ids))
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.chunk_id, c.node_id, c.chunk_index, c.grain, c.text, c.content_hash,
                       c.scope, c.workspace, c.project, c.session_id,
                       c.retrieval_summary, c.structured_summary, c.retrieval_signature, c.time_bucket, c.entity_tags, c.task_type_tag,
                       c.token_estimate, c.source_kind, c.source_path, c.source_type, c.source_ref, c.created_at, c.updated_at,
                       n.shell, n.sector, n.zone, n.cell, n.scope AS node_scope, n.workspace AS node_workspace, n.project AS node_project, n.session_id AS node_session_id
                FROM memory_chunks c
                JOIN memory_nodes n ON n.id = c.node_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                tuple(unique_chunk_ids),
            ).fetchall()
        self._record_stat("fetch_chunks_with_node_metadata_by_ids", (perf_counter() - started) * 1000.0, rows=len(rows))
        by_id = {str(row["chunk_id"]): dict(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def fetch_objects(self, where: str = '', params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        query = 'SELECT * FROM memory_objects'
        if where:
            query += f' WHERE {where}'
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        self._record_stat("fetch_objects", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(r) for r in rows]

    def fetch_objects_by_ids(self, object_ids: list[str]) -> list[dict[str, Any]]:
        if not object_ids:
            return []
        unique_object_ids = list(dict.fromkeys(object_ids))
        placeholders = ','.join(['?'] * len(unique_object_ids))
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f'SELECT * FROM memory_objects WHERE object_id IN ({placeholders})',
                tuple(unique_object_ids),
            ).fetchall()
        self._record_stat("fetch_objects_by_ids", (perf_counter() - started) * 1000.0, rows=len(rows))
        by_id = {row['object_id']: dict(row) for row in rows}
        return [by_id[object_id] for object_id in object_ids if object_id in by_id]

    def fetch_objects_for_nodes(
        self,
        node_ids: list[str],
        object_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        filtered_node_ids = list(dict.fromkeys(node_id for node_id in node_ids if node_id))
        if not filtered_node_ids:
            return []
        node_placeholders = ",".join(["?"] * len(filtered_node_ids))
        params: list[Any] = list(filtered_node_ids)
        where = f"source_node_id IN ({node_placeholders})"
        if object_types:
            filtered_types = list(dict.fromkeys(obj_type for obj_type in object_types if obj_type))
            if filtered_types:
                type_placeholders = ",".join(["?"] * len(filtered_types))
                where += f" AND object_type IN ({type_placeholders})"
                params.extend(filtered_types)
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM memory_objects
                WHERE {where}
                ORDER BY source_node_id ASC, sequence_index ASC, turn_index ASC, timestamp DESC, object_id ASC
                """,
                tuple(params),
            ).fetchall()
        self._record_stat("fetch_objects_for_nodes", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def insert_artifact(self, artifact: dict[str, Any]) -> None:
        if not artifact:
            return
        payload = self._artifact_payload(artifact)
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifact_registry (
                    artifact_id, path, artifact_type, scope, workspace, project, session_id,
                    title, summary, tags_json, source_type, source_ref,
                    related_memory_ids_json, related_object_ids_json, related_goal_ids_json, metadata_json,
                    created_at, updated_at
                ) VALUES (
                    :artifact_id, :path, :artifact_type, :scope, :workspace, :project, :session_id,
                    :title, :summary, :tags_json, :source_type, :source_ref,
                    :related_memory_ids_json, :related_object_ids_json, :related_goal_ids_json, :metadata_json,
                    :created_at, :updated_at
                )
                """,
                payload,
            )
        self._record_stat("insert_artifact", (perf_counter() - started) * 1000.0, rows=1)

    def fetch_artifacts(self, where: str = "", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        query = "SELECT * FROM artifact_registry"
        if where:
            query += f" WHERE {where}"
        query += " ORDER BY updated_at DESC, created_at DESC, artifact_id ASC"
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        self._record_stat("fetch_artifacts", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def fetch_artifact_by_path(self, path: str) -> dict[str, Any] | None:
        rows = self.fetch_artifacts("path = ?", (path,))
        return rows[0] if rows else None

    def insert_open_loop(self, loop: dict[str, Any]) -> None:
        if not loop:
            return
        payload = self._open_loop_payload(loop)
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO open_loops (
                    loop_id, title, details, status, scope, workspace, project, session_id,
                    priority, tags_json, blocked_reason, source_type, source_ref,
                    related_memory_ids_json, related_artifact_ids_json, metadata_json,
                    created_at, updated_at
                ) VALUES (
                    :loop_id, :title, :details, :status, :scope, :workspace, :project, :session_id,
                    :priority, :tags_json, :blocked_reason, :source_type, :source_ref,
                    :related_memory_ids_json, :related_artifact_ids_json, :metadata_json,
                    :created_at, :updated_at
                )
                """,
                payload,
            )
        self._record_stat("insert_open_loop", (perf_counter() - started) * 1000.0, rows=1)

    def fetch_open_loops(self, where: str = "", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        query = "SELECT * FROM open_loops"
        if where:
            query += f" WHERE {where}"
        query += " ORDER BY updated_at DESC, created_at DESC, loop_id ASC"
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        self._record_stat("fetch_open_loops", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def update_open_loop_status(self, loop_id: str, status: str, *, blocked_reason: str | None = None) -> None:
        if not loop_id:
            return
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE open_loops
                SET status = ?, blocked_reason = ?, updated_at = CURRENT_TIMESTAMP
                WHERE loop_id = ?
                """,
                (status, blocked_reason, loop_id),
            )
        self._record_stat("update_open_loop_status", (perf_counter() - started) * 1000.0, rows=1)

    def list_workspace_inventory(self) -> list[dict[str, Any]]:
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT workspace, project, scope, session_id, 'memory_node' AS source, COUNT(*) AS item_count
                FROM memory_nodes
                GROUP BY workspace, project, scope, session_id
                UNION ALL
                SELECT workspace, project, scope, session_id, 'artifact' AS source, COUNT(*) AS item_count
                FROM artifact_registry
                GROUP BY workspace, project, scope, session_id
                UNION ALL
                SELECT workspace, project, scope, session_id, 'open_loop' AS source, COUNT(*) AS item_count
                FROM open_loops
                GROUP BY workspace, project, scope, session_id
                ORDER BY workspace ASC, project ASC, scope ASC, session_id ASC, source ASC
                """
            ).fetchall()
        self._record_stat("list_workspace_inventory", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def insert_memory_delta(self, delta: dict[str, Any]) -> None:
        if not delta:
            return
        started = perf_counter()
        payload = dict(delta)
        payload.setdefault("object_type", None)
        payload.setdefault("changed_fields_json", None)
        payload.setdefault("delta_summary", None)
        payload.setdefault("effective_time", None)
        payload.setdefault("merge_policy", None)
        payload.setdefault("created_at", None)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_deltas (
                    delta_id, node_id, base_node_id, object_type, changed_fields_json,
                    delta_summary, effective_time, merge_policy, created_at
                ) VALUES (
                    :delta_id, :node_id, :base_node_id, :object_type, :changed_fields_json,
                    :delta_summary, :effective_time, :merge_policy, :created_at
                )
                """,
                payload,
            )
        self._record_stat("insert_memory_delta", (perf_counter() - started) * 1000.0, rows=1)

    def fetch_deltas_for_base(self, base_node_id: str) -> list[dict[str, Any]]:
        if not base_node_id:
            return []
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM memory_deltas
                WHERE base_node_id = ?
                ORDER BY created_at DESC, delta_id ASC
                """,
                (base_node_id,),
            ).fetchall()
        self._record_stat("fetch_deltas_for_base", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def delete_deltas_for_node(self, node_id: str) -> None:
        if not node_id:
            return
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM memory_deltas WHERE node_id = ? OR base_node_id = ?",
                (node_id, node_id),
            )
        self._record_stat("delete_deltas_for_node", (perf_counter() - started) * 1000.0, rows=1)

    def insert_representations(self, representations: list[dict[str, Any]]) -> None:
        if not representations:
            return
        started = perf_counter()
        payload = [dict(item) for item in representations]
        for item in payload:
            item.setdefault("scope", "global")
            item.setdefault("workspace", None)
            item.setdefault("project", None)
            item.setdefault("session_id", None)
            item.setdefault("time_bucket", None)
            item.setdefault("entity_tags", None)
            item.setdefault("task_type_tag", None)
            item.setdefault("created_at", None)
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO memory_representations (
                    representation_id, parent_id, parent_type, proxy_kind, text, content_hash,
                    scope, workspace, project, session_id,
                    time_bucket, entity_tags, task_type_tag, created_at
                ) VALUES (
                    :representation_id, :parent_id, :parent_type, :proxy_kind, :text, :content_hash,
                    :scope, :workspace, :project, :session_id,
                    :time_bucket, :entity_tags, :task_type_tag, :created_at
                )
                """,
                payload,
            )
        self._record_stat("insert_representations", (perf_counter() - started) * 1000.0, rows=len(representations))

    def fetch_representations(
        self,
        parent_ids: list[str],
        *,
        parent_type: str | None = None,
        proxy_kinds: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        filtered_parent_ids = list(dict.fromkeys(parent_id for parent_id in parent_ids if parent_id))
        if not filtered_parent_ids:
            return []
        parent_placeholders = ",".join(["?"] * len(filtered_parent_ids))
        clauses = [f"parent_id IN ({parent_placeholders})"]
        params: list[Any] = list(filtered_parent_ids)
        if parent_type:
            clauses.append("parent_type = ?")
            params.append(parent_type)
        if proxy_kinds:
            filtered_kinds = list(dict.fromkeys(kind for kind in proxy_kinds if kind))
            if filtered_kinds:
                placeholders = ",".join(["?"] * len(filtered_kinds))
                clauses.append(f"proxy_kind IN ({placeholders})")
                params.extend(filtered_kinds)
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM memory_representations
                WHERE {' AND '.join(clauses)}
                ORDER BY parent_id ASC, proxy_kind ASC, created_at DESC
                """,
                tuple(params),
            ).fetchall()
        self._record_stat("fetch_representations", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def delete_representations_for_parent_ids(self, parent_ids: list[str], *, parent_type: str | None = None) -> list[str]:
        filtered_parent_ids = list(dict.fromkeys(parent_id for parent_id in parent_ids if parent_id))
        if not filtered_parent_ids:
            return []
        parent_placeholders = ",".join(["?"] * len(filtered_parent_ids))
        clauses = [f"parent_id IN ({parent_placeholders})"]
        params: list[Any] = list(filtered_parent_ids)
        if parent_type:
            clauses.append("parent_type = ?")
            params.append(parent_type)
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT representation_id FROM memory_representations WHERE {' AND '.join(clauses)}",
                tuple(params),
            ).fetchall()
            conn.execute(
                f"DELETE FROM memory_representations WHERE {' AND '.join(clauses)}",
                tuple(params),
            )
        self._record_stat("delete_representations_for_parent_ids", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [str(row["representation_id"]) for row in rows]

    def fetch_profile_snapshot(
        self,
        snapshot_type: str,
        snapshot_key: str = "current",
        *,
        memory_version: int | None = None,
    ) -> dict[str, Any] | None:
        started = perf_counter()
        params: list[Any] = [snapshot_type, snapshot_key]
        sql = """
            SELECT *
            FROM profile_snapshots
            WHERE snapshot_type = ? AND snapshot_key = ?
        """
        if memory_version is not None:
            sql += " AND memory_version = ?"
            params.append(int(memory_version))
        with self.connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        self._record_stat("fetch_profile_snapshot", (perf_counter() - started) * 1000.0, rows=1 if row else 0)
        if not row:
            return None
        payload = dict(row)
        payload["payload"] = json.loads(str(payload.get("payload_json") or "{}"))
        payload["source_object_ids"] = json.loads(str(payload.get("source_object_ids") or "[]"))
        return payload

    def upsert_profile_snapshot(
        self,
        *,
        snapshot_type: str,
        snapshot_key: str = "current",
        memory_version: int,
        payload_json: str,
        source_object_ids: list[str] | None = None,
        updated_at: str | None = None,
    ) -> None:
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO profile_snapshots (
                    snapshot_type, snapshot_key, memory_version, payload_json, source_object_ids, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_type, snapshot_key) DO UPDATE SET
                    memory_version = excluded.memory_version,
                    payload_json = excluded.payload_json,
                    source_object_ids = excluded.source_object_ids,
                    updated_at = excluded.updated_at
                """,
                (
                    snapshot_type,
                    snapshot_key,
                    int(memory_version),
                    payload_json,
                    json.dumps(list(source_object_ids or []), ensure_ascii=False),
                    updated_at,
                ),
            )
        self._record_stat("upsert_profile_snapshot", (perf_counter() - started) * 1000.0, rows=1)

    def fetch_cache_entry(
        self,
        table_name: str,
        primary_key_name: str,
        key: str,
        *,
        memory_version: int | None = None,
    ) -> dict[str, Any] | None:
        if not key:
            return None
        params: list[Any] = [key]
        sql = f"SELECT * FROM {table_name} WHERE {primary_key_name} = ?"
        if memory_version is not None:
            sql += " AND memory_version = ?"
            params.append(int(memory_version))
        started = perf_counter()
        with self.connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
            if row is not None:
                conn.execute(
                    f"UPDATE {table_name} SET hit_count = COALESCE(hit_count, 0) + 1 WHERE {primary_key_name} = ?",
                    (key,),
                )
        self._record_stat(f"fetch_{table_name}", (perf_counter() - started) * 1000.0, rows=1 if row else 0)
        return dict(row) if row else None

    def upsert_retrieval_cache(
        self,
        *,
        query_fingerprint: str,
        normalized_query: str,
        task_type: str,
        route_type: str,
        memory_version: int,
        payload_json: str,
        created_at: str | None,
    ) -> None:
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_cache (
                    query_fingerprint, normalized_query, task_type, route_type, memory_version, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(query_fingerprint) DO UPDATE SET
                    normalized_query = excluded.normalized_query,
                    task_type = excluded.task_type,
                    route_type = excluded.route_type,
                    memory_version = excluded.memory_version,
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                (
                    query_fingerprint,
                    normalized_query,
                    task_type,
                    route_type,
                    int(memory_version),
                    payload_json,
                    created_at,
                ),
            )
        self._record_stat("upsert_retrieval_cache", (perf_counter() - started) * 1000.0, rows=1)

    def upsert_completion_cache(
        self,
        *,
        cache_key: str,
        evidence_signature: str,
        task_type: str,
        memory_version: int,
        payload_json: str,
        created_at: str | None,
    ) -> None:
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO completion_cache (
                    cache_key, evidence_signature, task_type, memory_version, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    evidence_signature = excluded.evidence_signature,
                    task_type = excluded.task_type,
                    memory_version = excluded.memory_version,
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                (
                    cache_key,
                    evidence_signature,
                    task_type,
                    int(memory_version),
                    payload_json,
                    created_at,
                ),
            )
        self._record_stat("upsert_completion_cache", (perf_counter() - started) * 1000.0, rows=1)

    def fetch_memory_deltas(
        self,
        *,
        node_id: str | None = None,
        base_node_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if node_id:
            clauses.append("node_id = ?")
            params.append(node_id)
        if base_node_id:
            clauses.append("base_node_id = ?")
            params.append(base_node_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM memory_deltas
                {where}
                ORDER BY created_at DESC, delta_id ASC
                """,
                tuple(params),
            ).fetchall()
        self._record_stat("fetch_memory_deltas", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def delete_memory_deltas_for_node(self, node_id: str) -> None:
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM memory_deltas WHERE node_id = ? OR base_node_id = ?",
                (node_id, node_id),
            )
        self._record_stat("delete_memory_deltas_for_node", (perf_counter() - started) * 1000.0)

    def get_retrieval_cache(self, query_fingerprint: str, memory_version: int) -> dict[str, Any] | None:
        if not query_fingerprint:
            return None
        started = perf_counter()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM retrieval_cache
                WHERE query_fingerprint = ? AND memory_version = ?
                """,
                (query_fingerprint, int(memory_version)),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE retrieval_cache SET hit_count = hit_count + 1 WHERE query_fingerprint = ?",
                    (query_fingerprint,),
                )
        self._record_stat("get_retrieval_cache", (perf_counter() - started) * 1000.0, rows=1 if row else 0)
        if not row:
            return None
        payload = dict(row)
        payload["payload"] = json.loads(str(payload.get("payload_json") or "{}"))
        return payload

    def put_retrieval_cache(
        self,
        *,
        query_fingerprint: str,
        normalized_query: str,
        task_type: str,
        route_type: str,
        memory_version: int,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_cache (
                    query_fingerprint, normalized_query, task_type, route_type,
                    memory_version, payload_json, created_at, hit_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(query_fingerprint) DO UPDATE SET
                    normalized_query = excluded.normalized_query,
                    task_type = excluded.task_type,
                    route_type = excluded.route_type,
                    memory_version = excluded.memory_version,
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                (
                    query_fingerprint,
                    normalized_query,
                    task_type,
                    route_type,
                    int(memory_version),
                    json.dumps(payload, ensure_ascii=False),
                    created_at,
                ),
            )
        self._record_stat("put_retrieval_cache", (perf_counter() - started) * 1000.0, rows=1)

    def get_completion_cache(self, cache_key: str, memory_version: int) -> dict[str, Any] | None:
        if not cache_key:
            return None
        started = perf_counter()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM completion_cache
                WHERE cache_key = ? AND memory_version = ?
                """,
                (cache_key, int(memory_version)),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE completion_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (cache_key,),
                )
        self._record_stat("get_completion_cache", (perf_counter() - started) * 1000.0, rows=1 if row else 0)
        if not row:
            return None
        payload = dict(row)
        payload["payload"] = json.loads(str(payload.get("payload_json") or "{}"))
        return payload

    def put_completion_cache(
        self,
        *,
        cache_key: str,
        evidence_signature: str,
        task_type: str,
        memory_version: int,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        started = perf_counter()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO completion_cache (
                    cache_key, evidence_signature, task_type,
                    memory_version, payload_json, created_at, hit_count
                ) VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    evidence_signature = excluded.evidence_signature,
                    task_type = excluded.task_type,
                    memory_version = excluded.memory_version,
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                (
                    cache_key,
                    evidence_signature,
                    task_type,
                    int(memory_version),
                    json.dumps(payload, ensure_ascii=False),
                    created_at,
                ),
            )
        self._record_stat("put_completion_cache", (perf_counter() - started) * 1000.0, rows=1)

    def fetch_neighbor_links(self, chunk_ids: list[str], limit: int | None = None) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ','.join(['?'] * len(chunk_ids))
        sql = f'SELECT * FROM chunk_neighbors WHERE chunk_id IN ({placeholders}) ORDER BY weight DESC'
        params: tuple[Any, ...] = tuple(chunk_ids)
        if limit is not None and limit > 0:
            sql += ' LIMIT ?'
            params = (*params, limit)
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        self._record_stat("fetch_neighbor_links", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(r) for r in rows]

    def get_local_chunk_neighbors(
        self,
        chunk_ids: list[str],
        *,
        limit_per_chunk: int = 8,
        min_weight: float = 0.0,
    ) -> dict[str, list[dict[str, Any]]]:
        filtered_chunk_ids = list(dict.fromkeys(chunk_id for chunk_id in chunk_ids if chunk_id))
        if not filtered_chunk_ids:
            return {}
        placeholders = ",".join(["?"] * len(filtered_chunk_ids))
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM chunk_neighbors
                WHERE chunk_id IN ({placeholders}) AND weight >= ?
                ORDER BY chunk_id ASC, weight DESC, neighbor_chunk_id ASC
                """,
                (*filtered_chunk_ids, float(min_weight)),
            ).fetchall()
        self._record_stat("get_local_chunk_neighbors", (perf_counter() - started) * 1000.0, rows=len(rows))
        grouped: dict[str, list[dict[str, Any]]] = {chunk_id: [] for chunk_id in filtered_chunk_ids}
        for row in rows:
            key = str(row["chunk_id"])
            items = grouped.setdefault(key, [])
            if len(items) >= max(1, int(limit_per_chunk)):
                continue
            items.append(dict(row))
        return grouped

    def fetch_neighbor_counts(self, chunk_ids: list[str]) -> dict[str, int]:
        if not chunk_ids:
            return {}
        unique_chunk_ids = list(dict.fromkeys(chunk_id for chunk_id in chunk_ids if chunk_id))
        if not unique_chunk_ids:
            return {}
        placeholders = ','.join(['?'] * len(unique_chunk_ids))
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f'SELECT chunk_id, COUNT(*) AS neighbor_count FROM chunk_neighbors WHERE chunk_id IN ({placeholders}) GROUP BY chunk_id',
                tuple(unique_chunk_ids),
            ).fetchall()
        self._record_stat("fetch_neighbor_counts", (perf_counter() - started) * 1000.0, rows=len(rows))
        return {str(row['chunk_id']): int(row['neighbor_count']) for row in rows}

    def search_chunks_fts(self, query: str, limit: int) -> list[dict[str, Any]]:
        match_query = self._build_fts_match_query(query)
        if not match_query or limit <= 0:
            return []
        try:
            started = perf_counter()
            with self.connect() as conn:
                rows = conn.execute(
                    """
                    SELECT c.chunk_id, c.node_id, c.chunk_index, c.grain, c.text, c.created_at,
                           c.scope, c.workspace, c.project, c.session_id, c.source_type, c.source_ref,
                           c.retrieval_summary, c.structured_summary, c.retrieval_signature, c.time_bucket, c.entity_tags, c.task_type_tag,
                           n.shell, n.sector, n.zone, n.cell, n.summary, n.content_ref, n.access_count,
                           n.scope AS node_scope, n.workspace AS node_workspace, n.project AS node_project, n.session_id AS node_session_id,
                           bm25(memory_chunks_fts, 1.2, 0.35, 0.08, 0.08) AS bm25_score
                    FROM memory_chunks_fts
                    JOIN memory_chunks c ON c.chunk_id = memory_chunks_fts.chunk_id
                    JOIN memory_nodes n ON n.id = c.node_id
                    WHERE memory_chunks_fts MATCH ?
                    ORDER BY bm25_score
                    LIMIT ?
                    """,
                    (match_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        self._record_stat("search_chunks_fts", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def search_objects_fts(self, query: str, limit: int, object_types: list[str] | None = None) -> list[dict[str, Any]]:
        match_query = self._build_fts_match_query(query)
        if not match_query or limit <= 0:
            return []
        where = ''
        params: list[Any] = [match_query]
        if object_types:
            placeholders = ','.join(['?'] * len(object_types))
            where = f' AND o.object_type IN ({placeholders})'
            params.extend(object_types)
        params.append(limit)
        sql = f"""
            SELECT o.*,
                   bm25(memory_objects_fts, 0.2, 0.12, 0.65, 0.65, 1.1, 1.35, 0.32) AS bm25_score
            FROM memory_objects_fts
            JOIN memory_objects o ON o.object_id = memory_objects_fts.object_id
            WHERE memory_objects_fts MATCH ?{where}
            ORDER BY bm25_score
            LIMIT ?
        """
        try:
            started = perf_counter()
            with self.connect() as conn:
                rows = conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.OperationalError:
            return []
        self._record_stat("search_objects_fts", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def fetch_vector_sync_state(self, collection_name: str, item_ids: list[str]) -> dict[str, str]:
        if not item_ids:
            return {}
        unique_item_ids = list(dict.fromkeys(item_ids))
        batch_size = 900
        started = perf_counter()
        rows: list[sqlite3.Row] = []
        with self.connect() as conn:
            for offset in range(0, len(unique_item_ids), batch_size):
                batch = unique_item_ids[offset : offset + batch_size]
                placeholders = ",".join(["?"] * len(batch))
                rows.extend(
                    conn.execute(
                        f"""
                        SELECT item_id, content_hash
                        FROM vector_sync_state
                        WHERE collection_name = ? AND item_id IN ({placeholders})
                        """,
                        (collection_name, *batch),
                    ).fetchall()
                )
        self._record_stat("fetch_vector_sync_state", (perf_counter() - started) * 1000.0, rows=len(rows))
        return {str(row["item_id"]): str(row["content_hash"]) for row in rows}

    def upsert_vector_sync_state(self, collection_name: str, item_states: list[tuple[str, str]]) -> None:
        if not item_states:
            return
        started = perf_counter()
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO vector_sync_state (collection_name, item_id, content_hash, synced_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(collection_name, item_id) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    synced_at=CURRENT_TIMESTAMP
                """,
                [(collection_name, item_id, content_hash) for item_id, content_hash in item_states if item_id and content_hash],
            )
        self._record_stat("upsert_vector_sync_state", (perf_counter() - started) * 1000.0, rows=len(item_states))

    def delete_vector_sync_state(self, collection_name: str, item_ids: list[str]) -> None:
        if not item_ids:
            return
        unique_item_ids = list(dict.fromkeys(item_ids))
        batch_size = 900
        started = perf_counter()
        with self.connect() as conn:
            for offset in range(0, len(unique_item_ids), batch_size):
                batch = unique_item_ids[offset : offset + batch_size]
                placeholders = ",".join(["?"] * len(batch))
                conn.execute(
                    f"DELETE FROM vector_sync_state WHERE collection_name = ? AND item_id IN ({placeholders})",
                    (collection_name, *batch),
                )
        self._record_stat("delete_vector_sync_state", (perf_counter() - started) * 1000.0, rows=len(unique_item_ids))

    def count_vector_sync_state(self, collection_name: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM vector_sync_state WHERE collection_name = ?",
                (collection_name,),
            ).fetchone()
        return int(row["c"] if row else 0)

    def clear_vector_sync_state(self, collection_name: str) -> None:
        started = perf_counter()
        with self.connect() as conn:
            conn.execute("DELETE FROM vector_sync_state WHERE collection_name = ?", (collection_name,))
        self._record_stat("clear_vector_sync_state", (perf_counter() - started) * 1000.0)

    def mark_chunks_vector_synced(self, chunk_ids: list[str], timestamp: str) -> None:
        if not chunk_ids:
            return
        placeholders = ','.join(['?'] * len(chunk_ids))
        with self.connect() as conn:
            conn.execute(f'UPDATE memory_chunks SET vector_synced_at = ? WHERE chunk_id IN ({placeholders})', (timestamp, *chunk_ids))

    def count_chunks(self) -> int:
        with self.connect() as conn:
            row = conn.execute('SELECT COUNT(*) AS c FROM memory_chunks').fetchone()
        return int(row['c'] if row else 0)

    def count_edges(self) -> int:
        with self.connect() as conn:
            row = conn.execute('SELECT COUNT(*) AS c FROM memory_edges').fetchone()
        return int(row["c"] if row else 0)

    def fetch_edges(self, where: str = '', params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        query = 'SELECT * FROM memory_edges'
        if where:
            query += f' WHERE {where}'
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        self._record_stat("fetch_edges", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(r) for r in rows]

    def fetch_edges_for_nodes(self, node_ids: list[str]) -> list[dict[str, Any]]:
        filtered_node_ids = list(dict.fromkeys(node_id for node_id in node_ids if node_id))
        if not filtered_node_ids:
            return []
        placeholders = ",".join(["?"] * len(filtered_node_ids))
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM memory_edges
                WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                ORDER BY creative_weight DESC, structural_weight DESC, semantic_weight DESC, temporal_weight DESC
                """,
                tuple(filtered_node_ids + filtered_node_ids),
            ).fetchall()
        self._record_stat("fetch_edges_for_nodes", (perf_counter() - started) * 1000.0, rows=len(rows))
        return [dict(row) for row in rows]

    def get_top_neighbor_edges(
        self,
        node_ids: list[str],
        edge_modes: list[str] | None = None,
        limit_per_node: int = 8,
        min_weight: float = 0.0,
    ) -> dict[str, list[dict[str, Any]]]:
        filtered_node_ids = list(dict.fromkeys(node_id for node_id in node_ids if node_id))
        if not filtered_node_ids:
            return {}
        placeholders = ",".join(["?"] * len(filtered_node_ids))
        started = perf_counter()
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM memory_edges
                WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
                ORDER BY creative_weight DESC, structural_weight DESC, semantic_weight DESC, temporal_weight DESC, task_weight DESC
                """,
                tuple(filtered_node_ids + filtered_node_ids),
            ).fetchall()
        self._record_stat("get_top_neighbor_edges", (perf_counter() - started) * 1000.0, rows=len(rows))

        requested_modes = list(dict.fromkeys(edge_modes or ["semantic"]))
        grouped: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in filtered_node_ids}
        for row in rows:
            edge = dict(row)
            for pivot_node_id in filtered_node_ids:
                if pivot_node_id != str(edge.get("source_id") or "") and pivot_node_id != str(edge.get("target_id") or ""):
                    continue
                neighbor_node_id = str(edge.get("target_id") if pivot_node_id == str(edge.get("source_id") or "") else edge.get("source_id"))
                edge["pivot_node_id"] = pivot_node_id
                edge["neighbor_node_id"] = neighbor_node_id
                edge["mode_scores"] = {mode: round(self._edge_mode_score(edge, mode), 4) for mode in requested_modes}
                edge["best_mode_score"] = max(edge["mode_scores"].values()) if edge["mode_scores"] else 0.0
                if edge["best_mode_score"] < float(min_weight):
                    continue
                bucket = grouped.setdefault(pivot_node_id, [])
                if len(bucket) >= max(1, int(limit_per_node)) and edge["best_mode_score"] <= float(bucket[-1].get("best_mode_score") or 0.0):
                    continue
                bucket.append(dict(edge))
                bucket.sort(key=lambda item: float(item.get("best_mode_score") or 0.0), reverse=True)
                if len(bucket) > max(1, int(limit_per_node)):
                    del bucket[max(1, int(limit_per_node)) :]
        return grouped

    @staticmethod
    def _edge_mode_score(edge: dict[str, Any], mode: str) -> float:
        semantic = float(edge.get("semantic_weight") or 0.0)
        task = float(edge.get("task_weight") or 0.0)
        temporal = float(edge.get("temporal_weight") or 0.0)
        causal = float(edge.get("causal_weight") or 0.0)
        creative = float(edge.get("creative_weight") or 0.0)
        structural = float(edge.get("structural_weight") or 0.0)
        if mode == "semantic":
            return semantic * 0.58 + task * 0.14 + structural * 0.16 + causal * 0.12
        if mode == "analogy":
            return creative * 0.45 + structural * 0.3 + semantic * 0.15 + task * 0.1
        if mode == "contrast":
            return structural * 0.3 + creative * 0.2 + semantic * 0.15 + task * 0.1
        if mode == "transfer":
            return structural * 0.32 + task * 0.24 + creative * 0.18 + semantic * 0.14 + causal * 0.12
        if mode == "temporal":
            return temporal * 0.48 + causal * 0.27 + semantic * 0.1 + structural * 0.1 + task * 0.05
        if mode == "composition":
            return structural * 0.32 + semantic * 0.18 + task * 0.18 + creative * 0.14 + temporal * 0.08 + causal * 0.1
        return semantic + task + temporal + causal + creative + structural

    def update_node_access(self, node_id: str, timestamp: str) -> None:
        started = perf_counter()
        with self.connect() as conn:
            conn.execute('UPDATE memory_nodes SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?', (timestamp, node_id))
        self._record_stat("update_node_access", (perf_counter() - started) * 1000.0, rows=1)

    def update_node_stage(self, node_id: str, stage: str) -> None:
        with self.connect() as conn:
            conn.execute('UPDATE memory_nodes SET stage = ? WHERE id = ?', (stage, node_id))

    def update_node_compression(self, node_id: str, summary: str, raw_content: str | None, compression_level: str) -> None:
        with self.connect() as conn:
            conn.execute(
                'UPDATE memory_nodes SET summary = ?, raw_content = ?, compression_level = ? WHERE id = ?',
                (summary, raw_content, compression_level, node_id),
            )

    def bulk_update_zone(self, node_ids: list[str], new_zone: str) -> None:
        if not node_ids:
            return
        placeholders = ','.join(['?'] * len(node_ids))
        with self.connect() as conn:
            conn.execute(f'UPDATE memory_nodes SET zone = ? WHERE id IN ({placeholders})', (new_zone, *node_ids))
            conn.execute(f'UPDATE memory_chunks SET source_path = source_path WHERE node_id IN ({placeholders})', tuple(node_ids))

    def decay_edges(self, factor: float, floor: float) -> int:
        edges = self.fetch_edges()
        with self.connect() as conn:
            for edge in edges:
                conn.execute(
                    """
                    UPDATE memory_edges
                    SET semantic_weight = MAX(?, semantic_weight * ?),
                        task_weight = MAX(?, task_weight * ?),
                        temporal_weight = MAX(?, temporal_weight * ?),
                        causal_weight = MAX(?, causal_weight * ?),
                        creative_weight = MAX(?, creative_weight * ?),
                        structural_weight = MAX(?, structural_weight * ?)
                    WHERE id = ?
                    """,
                    (
                        floor, factor,
                        floor, factor,
                        floor, factor,
                        floor, factor,
                        floor, factor,
                        floor, factor,
                        edge['id'],
                    ),
                )
        return len(edges)

    def zone_counts(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                'SELECT zone, COUNT(*) AS count, MIN(shell) AS shell, MIN(sector) AS sector FROM memory_nodes GROUP BY zone ORDER BY count DESC'
            ).fetchall()
        return [dict(r) for r in rows]

    def rebuild_zone_index(self) -> None:
        with self.connect() as conn:
            conn.execute('DELETE FROM zone_index')
            rows = conn.execute(
                """
                SELECT shell, sector, zone, COUNT(*) AS item_count, AVG(theta) AS centroid_theta, AVG(phi) AS centroid_phi,
                       SUBSTR(GROUP_CONCAT(summary, ' | '), 1, 500) AS zone_summary
                FROM memory_nodes
                GROUP BY shell, sector, zone
                """
            ).fetchall()
            for row in rows:
                zone_id = f"{row['shell']}:{row['sector']}:{row['zone']}"
                conn.execute(
                    'INSERT INTO zone_index (zone_id, shell, sector, zone, zone_summary, centroid_theta, centroid_phi, item_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        zone_id,
                        row['shell'],
                        row['sector'],
                        row['zone'],
                        row['zone_summary'],
                        row['centroid_theta'],
                        row['centroid_phi'],
                        row['item_count'],
                    ),
                )

    def log_activation(self, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO activation_logs (
                    task_id, task_type, main_nodes, reflected_nodes, refracted_nodes,
                    final_used_nodes, token_cost_input, token_cost_output,
                    quality_feedback, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload['task_id'],
                    payload['task_type'],
                    json.dumps(payload['main_nodes'], ensure_ascii=False),
                    json.dumps(payload['reflected_nodes'], ensure_ascii=False),
                    json.dumps(payload['refracted_nodes'], ensure_ascii=False),
                    json.dumps(payload['final_used_nodes'], ensure_ascii=False),
                    payload['token_cost_input'],
                    payload['token_cost_output'],
                    payload.get('quality_feedback'),
                    payload['created_at'],
                ),
            )

    def export_zone_nodes(self, zone: str) -> list[dict[str, Any]]:
        return self.fetch_nodes('zone = ?', (zone,))

    def raw_sqlite_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_fts_tables(self, conn: sqlite3.Connection) -> None:
        chunk_required = ["chunk_id", "node_id", "grain", "text", "summary", "cell", "zone"]
        object_required = [
            "object_id",
            "source_node_id",
            "source_chunk_id",
            "object_type",
            "entity",
            "attribute",
            "old_value",
            "new_value",
            "canonical_key",
            "object_text",
            "source_unit_text",
        ]

        # SQLite expects chained tokenizers in `porter unicode61` order.
        # The previous `unicode61 porter` form fails on the Python sqlite build
        # used by the benchmark environment, which silently disabled all FTS.
        chunk_create_sql = """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts USING fts5(
                chunk_id UNINDEXED,
                node_id UNINDEXED,
                grain UNINDEXED,
                text,
                summary,
                cell,
                zone,
                tokenize='porter unicode61'
            )
        """
        object_create_sql = """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_objects_fts USING fts5(
                object_id UNINDEXED,
                source_node_id UNINDEXED,
                source_chunk_id UNINDEXED,
                object_type UNINDEXED,
                entity,
                attribute,
                old_value,
                new_value,
                canonical_key,
                object_text,
                source_unit_text,
                tokenize='porter unicode61'
            )
        """
        fallback_chunk_sql = chunk_create_sql.replace("porter unicode61", "unicode61")
        fallback_object_sql = object_create_sql.replace("porter unicode61", "unicode61")
        try:
            self._ensure_fts_table(
                conn,
                table_name="memory_chunks_fts",
                create_sql=chunk_create_sql,
                required_columns=chunk_required,
            )
            self._ensure_fts_table(
                conn,
                table_name="memory_objects_fts",
                create_sql=object_create_sql,
                required_columns=object_required,
            )
        except sqlite3.OperationalError:
            try:
                self._ensure_fts_table(
                    conn,
                    table_name="memory_chunks_fts",
                    create_sql=fallback_chunk_sql,
                    required_columns=chunk_required,
                )
                self._ensure_fts_table(
                    conn,
                    table_name="memory_objects_fts",
                    create_sql=fallback_object_sql,
                    required_columns=object_required,
                )
            except sqlite3.OperationalError:
                return

        chunk_count = int(conn.execute('SELECT COUNT(*) AS c FROM memory_chunks').fetchone()['c'])
        object_count = int(conn.execute('SELECT COUNT(*) AS c FROM memory_objects').fetchone()['c'])
        chunk_fts_count = int(conn.execute('SELECT COUNT(*) AS c FROM memory_chunks_fts').fetchone()['c'])
        object_fts_count = int(conn.execute('SELECT COUNT(*) AS c FROM memory_objects_fts').fetchone()['c'])
        if chunk_count != chunk_fts_count:
            self._rebuild_chunk_fts(conn)
        if object_count != object_fts_count:
            self._rebuild_object_fts(conn)

    def _rebuild_chunk_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute('DELETE FROM memory_chunks_fts')
        conn.execute(
            """
            INSERT INTO memory_chunks_fts (chunk_id, node_id, grain, text, summary, cell, zone)
            SELECT c.chunk_id,
                   c.node_id,
                   c.grain,
                   c.text,
                   COALESCE(c.retrieval_summary, n.summary, ''),
                   COALESCE(n.cell, ''),
                   COALESCE(n.zone, '')
            FROM memory_chunks c
            JOIN memory_nodes n ON n.id = c.node_id
            """
        )

    def _rebuild_object_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute('DELETE FROM memory_objects_fts')
        conn.execute(
            """
            INSERT INTO memory_objects_fts (
                object_id, source_node_id, source_chunk_id, object_type,
                entity, attribute, old_value, new_value, canonical_key, object_text, source_unit_text
            )
            SELECT object_id,
                   COALESCE(source_node_id, ''),
                   COALESCE(source_chunk_id, ''),
                   object_type,
                   COALESCE(entity, ''),
                   COALESCE(attribute, ''),
                   COALESCE(old_value, ''),
                   COALESCE(new_value, ''),
                   COALESCE(canonical_key, ''),
                   object_text,
                   COALESCE(source_unit_text, '')
            FROM memory_objects
            """
        )

    def _upsert_chunk_fts(self, conn: sqlite3.Connection, chunks: list[dict[str, Any]]) -> None:
        try:
            chunk_ids = [str(chunk['chunk_id']) for chunk in chunks]
            self._delete_chunk_fts(conn, chunk_ids)
            conn.executemany(
                """
                INSERT INTO memory_chunks_fts (chunk_id, node_id, grain, text, summary, cell, zone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(chunk['chunk_id']),
                        str(chunk['node_id']),
                        str(chunk.get('grain') or 'micro'),
                        str(chunk.get('text') or ''),
                        str(chunk.get('retrieval_summary') or chunk.get('summary') or ''),
                        str(chunk.get('cell') or ''),
                        str(chunk.get('zone') or ''),
                    )
                    for chunk in chunks
                ],
            )
        except sqlite3.OperationalError:
            return

    def _upsert_object_fts(self, conn: sqlite3.Connection, objects: list[dict[str, Any]]) -> None:
        try:
            object_ids = [str(obj['object_id']) for obj in objects]
            self._delete_object_fts(conn, object_ids)
            conn.executemany(
                """
                INSERT INTO memory_objects_fts (
                    object_id, source_node_id, source_chunk_id, object_type,
                    entity, attribute, old_value, new_value, canonical_key, object_text, source_unit_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(obj['object_id']),
                        str(obj.get('source_node_id') or ''),
                        str(obj.get('source_chunk_id') or ''),
                        str(obj.get('object_type') or ''),
                        str(obj.get('entity') or ''),
                        str(obj.get('attribute') or ''),
                        str(obj.get('old_value') or ''),
                        str(obj.get('new_value') or ''),
                        str(obj.get('canonical_key') or ''),
                        str(obj.get('object_text') or ''),
                        str(obj.get('source_unit_text') or ''),
                    )
                    for obj in objects
                ],
            )
        except sqlite3.OperationalError:
            return

    def _delete_chunk_fts(self, conn: sqlite3.Connection, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        try:
            placeholders = ','.join(['?'] * len(chunk_ids))
            conn.execute(f'DELETE FROM memory_chunks_fts WHERE chunk_id IN ({placeholders})', tuple(chunk_ids))
        except sqlite3.OperationalError:
            return

    def _delete_object_fts(self, conn: sqlite3.Connection, object_ids: list[str]) -> None:
        if not object_ids:
            return
        try:
            placeholders = ','.join(['?'] * len(object_ids))
            conn.execute(f'DELETE FROM memory_objects_fts WHERE object_id IN ({placeholders})', tuple(object_ids))
        except sqlite3.OperationalError:
            return

    def _ensure_fts_table(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        create_sql: str,
        required_columns: list[str],
    ) -> None:
        try:
            existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        except sqlite3.OperationalError:
            existing_columns = set()
        if existing_columns and not set(required_columns).issubset(existing_columns):
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(create_sql)

    def _build_fts_match_query(self, query: str) -> str:
        tokens = []
        seen: set[str] = set()
        for token in tokenize(query):
            if len(token) <= 1 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= 16:
                break
        if not tokens:
            return ''
        return ' OR '.join(f'"{token}"' for token in tokens)
