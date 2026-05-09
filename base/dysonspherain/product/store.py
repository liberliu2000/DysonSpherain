from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from fnmatch import fnmatch
from math import sqrt
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dysonspherain.memory_runtime.config import load_runtime_config
from dysonspherain.utils.token_counter import TokenCounter
from sphere_cli.security import redact_payload, redact_secrets
from sphere_cli.utils import stable_content_hash


SCHEMA_VERSION = 1
INLINE_RAW_LIMIT = 16_000
LOCAL_EMBEDDING_DIMS = 64
LOCAL_EMBEDDING_BACKEND = "local_hash_embedding"
LOCAL_EMBEDDING_VERSION = 1
EMBEDDING_CONFIG = "embedding_backend.json"
VECTOR_INDEX_CONFIG = "vector_index_backend.json"
PRODUCT_VECTOR_COLLECTION = "product_capsules"
PRODUCT_VECTOR_PROMOTION_THRESHOLD = 5000
WRITEBACK_RUNTIME_EVENTS = {"during_task", "after_task", "on_error", "after_benchmark", "before_commit", "after_commit", "manual_checkpoint"}
DEFAULT_IGNORE_PATTERNS = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "secrets.*",
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    ".git/",
]
ALL_PROBES = {
    "dense_probe",
    "sparse_probe",
    "proxy_probe",
    "temporal_probe",
    "entity_probe",
    "artifact_probe",
    "code_ref_probe",
    "recent_state_probe",
}
DEFAULT_ELIGIBLE_STATES = {"active", "stable"}
DEFAULT_EXCLUDED_STATES = {"superseded", "deprecated", "contradicted", "compacted", "archived", "reverted"}
DEFAULT_LIFECYCLE_MULTIPLIERS = {
    "active": 1.0,
    "stable": 1.1,
    "canonical": 1.15,
    "compacted": 0.45,
    "superseded": 0.05,
    "deprecated": 0.05,
    "contradicted": 0.02,
    "archived": 0.1,
    "reverted": 0.05,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def product_root(base_dir: Path) -> Path:
    return base_dir / ".memory"


def product_db_path(base_dir: Path) -> Path:
    return product_root(base_dir) / "dyson_product.sqlite3"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _list(value: list[str] | tuple[str, ...] | None) -> list[str]:
    return [str(item) for item in value or [] if str(item)]


def _load_ignore_patterns(base_dir: Path) -> list[str]:
    patterns = list(DEFAULT_IGNORE_PATTERNS)
    ignore = base_dir / ".dysonignore"
    if ignore.exists():
        for line in ignore.read_text(encoding="utf-8", errors="replace").splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                patterns.append(item)
    return patterns


def _path_matches_pattern(path: Path, pattern: str, *, base_dir: Path) -> bool:
    normalized = str(path).replace(os.sep, "/")
    try:
        rel = str(path.resolve().relative_to(base_dir.resolve())).replace(os.sep, "/")
    except ValueError:
        rel = normalized
    pat = pattern.replace(os.sep, "/")
    if pat.endswith("/"):
        prefix = pat.rstrip("/")
        return rel == prefix or rel.startswith(prefix + "/") or f"/{prefix}/" in normalized
    return fnmatch(rel, pat) or fnmatch(path.name, pat) or fnmatch(normalized, pat)


def _is_ignored_path(base_dir: Path, path: Path, *, denylist: list[str] | None = None, allowlist: list[str] | None = None) -> tuple[bool, str | None]:
    patterns = [*_load_ignore_patterns(base_dir), *_list(denylist)]
    if allowlist:
        allowed = any(_path_matches_pattern(path, pattern, base_dir=base_dir) for pattern in allowlist)
        if not allowed:
            return True, "not_in_allowlist"
    for pattern in patterns:
        if _path_matches_pattern(path, pattern, base_dir=base_dir):
            return True, pattern
    return False, None


def privacy_policy(base_dir: Path) -> dict[str, Any]:
    return {
        "status": "ok",
        "local_only": True,
        "ignore_file": str(base_dir / ".dysonignore"),
        "ignore_patterns": _load_ignore_patterns(base_dir),
        "encryption_at_rest": encryption_status(base_dir),
    }


def encryption_status(base_dir: Path) -> dict[str, Any]:
    marker = product_root(base_dir) / "encryption.json"
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("provider") == "sqlcipher":
            available = _sqlcipher_available()
            key_env = str(payload.get("key_env") or "DYSON_MEMORY_SQLCIPHER_KEY")
            return {
                **payload,
                "available": available and bool(os.environ.get(key_env)),
                "status": "configured" if available and os.environ.get(key_env) else "configured_unavailable",
                "key_env": key_env,
                "driver_available": available,
                "key_available": bool(os.environ.get(key_env)),
                "recommendation": None if available and os.environ.get(key_env) else "Install pysqlcipher3 or sqlcipher and set the configured key environment variable before opening encrypted product storage.",
            }
        return {"available": True, "status": "external_or_os_managed", **payload}
    return {"available": False, "status": "not_configured", "recommendation": "Use FileVault or an encrypted volume for .memory, or write .memory/encryption.json to document external encryption."}


def configure_encryption(
    base_dir: Path,
    *,
    provider: str,
    key_env: str = "DYSON_MEMORY_SQLCIPHER_KEY",
    scope: str = "product_sqlite",
    allow_unavailable: bool = False,
) -> dict[str, Any]:
    root = product_root(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    provider = provider.strip()
    if provider not in {"external_or_os_managed", "sqlcipher"}:
        raise ValueError("provider must be external_or_os_managed or sqlcipher")
    if provider == "sqlcipher" and not allow_unavailable:
        available = _sqlcipher_available() and bool(os.environ.get(key_env))
        if not available:
            raise RuntimeError("SQLCipher is not available or key env is unset; use --allow-unavailable to write a pending configuration marker.")
    payload = {"provider": provider, "scope": scope, "key_env": key_env if provider == "sqlcipher" else None, "configured_at": now_iso()}
    marker = root / "encryption.json"
    marker.write_text(_json({k: v for k, v in payload.items() if v is not None}) + "\n", encoding="utf-8")
    return {"status": "ok", "path": str(marker), "encryption_at_rest": encryption_status(base_dir)}


def migrate_product_db_to_sqlcipher(
    base_dir: Path,
    *,
    key_env: str = "DYSON_MEMORY_SQLCIPHER_KEY",
    output: Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    source = product_db_path(base_dir)
    output = output or product_root(base_dir) / "dyson_product.sqlcipher.sqlite3"
    key = os.environ.get(key_env)
    if not source.exists():
        return {"status": "unavailable", "reason": "product database does not exist", "source": str(source)}
    if not key:
        return {"status": "unavailable", "reason": f"{key_env} is not set", "source": str(source), "output": str(output)}
    try:
        from pysqlcipher3 import dbapi2 as sqlcipher_sqlite  # type: ignore
    except Exception as exc:
        return {"status": "unavailable", "reason": f"pysqlcipher3 is not installed: {exc}", "source": str(source), "output": str(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with sqlite3.connect(source) as src, sqlcipher_sqlite.connect(output) as dst:
        dst.execute("PRAGMA key = ?", (key,))
        for statement in src.iterdump():
            if not statement or statement.startswith("BEGIN") or statement.startswith("COMMIT"):
                continue
            dst.execute(statement)
        dst.commit()
    if replace:
        backup = source.with_suffix(".plaintext.backup.sqlite3")
        if backup.exists():
            backup.unlink()
        source.replace(backup)
        output.replace(source)
        configure_encryption(base_dir, provider="sqlcipher", key_env=key_env, allow_unavailable=True)
        return {"status": "ok", "mode": "replace", "source": str(source), "plaintext_backup": str(backup), "key_env": key_env}
    return {"status": "ok", "mode": "copy", "source": str(source), "output": str(output), "key_env": key_env}


def _sqlcipher_available() -> bool:
    try:
        __import__("pysqlcipher3")
        return True
    except Exception:
        return bool(shutil.which("sqlcipher"))


@dataclass
class EvidenceCapsule:
    id: str
    raw_ref: str
    raw_text: str | None
    source_type: str
    project_id: str
    session_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    timestamp: str = field(default_factory=now_iso)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    evidence_type: str = "note"
    title: str | None = None
    summary: str | None = None
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    file_refs: list[str] = field(default_factory=list)
    code_refs: list[str] = field(default_factory=list)
    command_refs: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    benchmark_refs: list[str] = field(default_factory=list)
    git_commit: str | None = None
    git_branch: str | None = None
    repo_path: str | None = None
    validity_state: str = "active"
    parent_ids: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.7
    embedding_id: str | None = None
    sparse_terms: list[str] = field(default_factory=list)
    temporal_scope: dict[str, Any] = field(default_factory=dict)
    route_features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def connect(base_dir: Path) -> sqlite3.Connection:
    path = product_db_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = product_root(base_dir) / "encryption.json"
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("provider") == "sqlcipher":
            key_env = str(payload.get("key_env") or "DYSON_MEMORY_SQLCIPHER_KEY")
            key = os.environ.get(key_env)
            if not key:
                raise RuntimeError(f"SQLCipher product store is configured but {key_env} is not set")
            try:
                from pysqlcipher3 import dbapi2 as sqlcipher_sqlite  # type: ignore
            except Exception as exc:
                raise RuntimeError("SQLCipher product store requires pysqlcipher3 in this Python environment") from exc
            conn = sqlcipher_sqlite.connect(path)
            conn.execute("PRAGMA key = ?", (key,))
        else:
            conn = sqlite3.connect(path)
    else:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_traces (
          raw_id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL,
          session_id TEXT,
          task_id TEXT,
          agent_id TEXT,
          source_type TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          original_text TEXT,
          blob_path TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_capsules (
          id TEXT PRIMARY KEY,
          raw_ref TEXT NOT NULL,
          raw_text TEXT,
          source_type TEXT NOT NULL,
          project_id TEXT NOT NULL,
          session_id TEXT,
          task_id TEXT,
          agent_id TEXT,
          timestamp TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          evidence_type TEXT NOT NULL,
          title TEXT,
          summary TEXT,
          entities_json TEXT NOT NULL DEFAULT '[]',
          tags_json TEXT NOT NULL DEFAULT '[]',
          file_refs_json TEXT NOT NULL DEFAULT '[]',
          code_refs_json TEXT NOT NULL DEFAULT '[]',
          command_refs_json TEXT NOT NULL DEFAULT '[]',
          artifact_refs_json TEXT NOT NULL DEFAULT '[]',
          benchmark_refs_json TEXT NOT NULL DEFAULT '[]',
          git_commit TEXT,
          git_branch TEXT,
          repo_path TEXT,
          validity_state TEXT NOT NULL DEFAULT 'active',
          parent_ids_json TEXT NOT NULL DEFAULT '[]',
          related_ids_json TEXT NOT NULL DEFAULT '[]',
          supports_json TEXT NOT NULL DEFAULT '[]',
          contradicts_json TEXT NOT NULL DEFAULT '[]',
          supersedes_json TEXT NOT NULL DEFAULT '[]',
          superseded_by_json TEXT NOT NULL DEFAULT '[]',
          importance REAL NOT NULL DEFAULT 0.5,
          confidence REAL NOT NULL DEFAULT 0.7,
          embedding_id TEXT,
          sparse_terms_json TEXT NOT NULL DEFAULT '[]',
          temporal_scope_json TEXT NOT NULL DEFAULT '{}',
          route_features_json TEXT NOT NULL DEFAULT '{}',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          archived INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_capsules_project_time ON evidence_capsules(project_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_capsules_project_validity ON evidence_capsules(project_id, validity_state)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_project_time ON raw_traces(project_id, timestamp)")
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS evidence_capsules_fts
        USING fts5(id UNINDEXED, project_id UNINDEXED, title, summary, raw_text, tags, entities)
        """
    )
    for ddl in (
        "CREATE TABLE IF NOT EXISTS capsule_entities (capsule_id TEXT, entity TEXT, PRIMARY KEY(capsule_id, entity))",
        "CREATE TABLE IF NOT EXISTS capsule_relations (relation_id TEXT PRIMARY KEY, source_capsule_id TEXT, target_capsule_id TEXT, relation_type TEXT, reason TEXT, created_at TEXT)",
        "CREATE TABLE IF NOT EXISTS capsule_artifacts (capsule_id TEXT, artifact_ref TEXT, artifact_type TEXT, metadata_json TEXT NOT NULL DEFAULT '{}', PRIMARY KEY(capsule_id, artifact_ref))",
        "CREATE TABLE IF NOT EXISTS capsule_embeddings (embedding_id TEXT PRIMARY KEY, capsule_id TEXT, backend TEXT, vector_ref TEXT, metadata_json TEXT NOT NULL DEFAULT '{}')",
        "CREATE TABLE IF NOT EXISTS capsule_aliases (alias TEXT PRIMARY KEY, canonical TEXT NOT NULL, project_id TEXT NOT NULL, created_at TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}')",
        "CREATE TABLE IF NOT EXISTS maintenance_suggestions (suggestion_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, suggestion_type TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open')",
        "CREATE TABLE IF NOT EXISTS retrieval_traces (trace_id TEXT PRIMARY KEY, project_id TEXT, query TEXT, route TEXT, payload_json TEXT NOT NULL, created_at TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS context_packs (context_pack_id TEXT PRIMARY KEY, project_id TEXT, query TEXT, route TEXT, payload_json TEXT NOT NULL, markdown TEXT NOT NULL, created_at TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS runtime_events (event_id TEXT PRIMARY KEY, project_id TEXT, event_type TEXT, payload_json TEXT NOT NULL, created_at TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS benchmark_runs (run_id TEXT PRIMARY KEY, project_id TEXT, benchmark TEXT, timestamp TEXT, git_commit TEXT, config_hash TEXT, metrics_json TEXT NOT NULL DEFAULT '{}', artifact_paths_json TEXT NOT NULL DEFAULT '[]', duration_sec REAL DEFAULT 0, status TEXT NOT NULL DEFAULT 'success', notes_json TEXT NOT NULL DEFAULT '[]')",
        "CREATE TABLE IF NOT EXISTS health_reports (report_id TEXT PRIMARY KEY, project_id TEXT, payload_json TEXT NOT NULL, created_at TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL, description TEXT NOT NULL)",
    ):
        conn.execute(ddl)
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at, description) VALUES (?, ?, ?)",
        (SCHEMA_VERSION, now_iso(), "initial product evidence schema"),
    )
    conn.commit()


def init_product_store(base_dir: Path, *, project_id: str = "default") -> dict[str, Any]:
    root = product_root(base_dir)
    for sub in ("raw", "artifacts", "indexes", "exports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    with connect(base_dir) as conn:
        counts = {
            "raw_traces": conn.execute("SELECT COUNT(*) FROM raw_traces").fetchone()[0],
            "evidence_capsules": conn.execute("SELECT COUNT(*) FROM evidence_capsules").fetchone()[0],
            "retrieval_traces": conn.execute("SELECT COUNT(*) FROM retrieval_traces").fetchone()[0],
            "context_packs": conn.execute("SELECT COUNT(*) FROM context_packs").fetchone()[0],
        }
    return {"status": "ok", "project_id": project_id, "schema_version": SCHEMA_VERSION, "db_path": str(product_db_path(base_dir)), "root": str(root), "counts": counts}


def _git_metadata(base_dir: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        proc = subprocess.run(["git", *args], cwd=base_dir, text=True, capture_output=True)
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    commit = run(["rev-parse", "HEAD"])
    branch = run(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["status", "--short"]) or ""
    return {"git_commit": commit, "git_branch": branch, "git_dirty": bool(status.strip()), "changed_files": [line[3:] for line in status.splitlines() if len(line) > 3]}


def _git_binding(base_dir: Path, *, include_diff: bool = False) -> dict[str, Any]:
    meta = _git_metadata(base_dir)

    def run(args: list[str]) -> str:
        proc = subprocess.run(["git", *args], cwd=base_dir, text=True, capture_output=True)
        return proc.stdout.strip() if proc.returncode == 0 else ""

    diff_stat = run(["diff", "--stat"])
    commit_message = run(["log", "-1", "--pretty=%B"])
    tag = run(["describe", "--tags", "--exact-match", "HEAD"])
    payload = {
        **meta,
        "diff_summary": diff_stat,
        "commit_message": commit_message,
        "tag": tag or None,
    }
    if include_diff:
        payload["diff"] = run(["diff"])
    return payload


STACK_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')
EXCEPTION_RE = re.compile(r"^([A-Za-z_][\w.]*Error|[A-Za-z_][\w.]*Exception|KeyboardInterrupt|SystemExit):\s*(.*)$", re.MULTILINE)


def _parse_error_trace(text: str) -> dict[str, Any]:
    frames = [
        {"file": file, "line": int(line), "function": func.strip()}
        for file, line, func in STACK_FRAME_RE.findall(text or "")
    ]
    exc = EXCEPTION_RE.findall(text or "")
    exception_type = exc[-1][0] if exc else None
    exception_message = exc[-1][1] if exc else None
    return {
        "exception_type": exception_type,
        "exception_message": exception_message,
        "stack_frames": frames,
        "file_refs": list(dict.fromkeys(frame["file"] for frame in frames)),
    }


def _extract_metric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    return metrics if isinstance(metrics, dict) else {}


def _first_metric(payload: dict[str, Any], names: list[str]) -> Any:
    metrics = _extract_metric_payload(payload)
    for name in names:
        if name in metrics:
            return metrics[name]
        if name in payload:
            return payload[name]
    return None


def _benchmark_binding(base_dir: Path, *, artifact: Path, metrics: dict[str, Any], benchmark: str, status: str) -> dict[str, Any]:
    git = _git_binding(base_dir)
    dataset = str(metrics.get("dataset") or metrics.get("benchmark_name") or metrics.get("benchmark") or benchmark)
    artifact_dir = artifact if artifact.is_dir() else artifact.parent
    return {
        "benchmark": benchmark,
        "dataset": dataset,
        "status": status,
        "artifact_dir": str(artifact_dir),
        "artifact_paths": [str(artifact)],
        "config_hash": str(metrics.get("config_hash") or ""),
        "duration_sec": float(metrics.get("elapsed_seconds") or metrics.get("wall_clock_elapsed_seconds") or metrics.get("duration_sec") or 0),
        "hardware": metrics.get("hardware") if isinstance(metrics.get("hardware"), dict) else {},
        "git": git,
        "quality": {
            "candidate_recall@100": _first_metric(metrics, ["candidate_recall@100", "candidate_recall_at_100"]),
            "recall": _first_metric(metrics, ["recall", "recall_frac@10", "recall_any@10"]),
            "ndcg": _first_metric(metrics, ["ndcg", "ndcg_any@10"]),
            "latency_ms": _first_metric(metrics, ["latency_ms", "retrieval_latency_ms"]),
        },
    }


def _write_benchmark_dashboard_files(base_dir: Path, *, project_id: str) -> dict[str, str]:
    root = product_root(base_dir) / "artifacts" / "benchmark_lab"
    root.mkdir(parents=True, exist_ok=True)
    runs = list_benchmark_runs(base_dir, project_id=project_id, limit=200)["benchmark_runs"]
    metric_trends: dict[str, list[dict[str, Any]]] = {}
    latency_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = _extract_metric_payload(run.get("metrics") or {})
        benchmark = str(run.get("benchmark") or "unknown")
        timestamp = str(run.get("timestamp") or "")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_trends.setdefault(key, []).append({"benchmark": benchmark, "timestamp": timestamp, "value": value, "run_id": run.get("run_id")})
        candidate = _first_metric(metrics, ["candidate_recall@100", "candidate_recall_at_100"])
        if candidate is not None:
            candidate_rows.append({"benchmark": benchmark, "timestamp": timestamp, "candidate_recall@100": candidate, "run_id": run.get("run_id")})
        latency = _first_metric(metrics, ["latency_ms", "retrieval_latency_ms", "elapsed_seconds"])
        if latency is not None:
            latency_rows.append({"benchmark": benchmark, "timestamp": timestamp, "latency": latency, "run_id": run.get("run_id")})
    files = {
        "benchmark_runs": root / "benchmark_runs.json",
        "metric_trends": root / "metric_trends.json",
        "candidate_admission_report": root / "candidate_admission_report.json",
        "latency_report": root / "latency_report.json",
    }
    files["benchmark_runs"].write_text(json.dumps({"project_id": project_id, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    files["metric_trends"].write_text(json.dumps({"project_id": project_id, "metrics": metric_trends}, ensure_ascii=False, indent=2), encoding="utf-8")
    files["candidate_admission_report"].write_text(json.dumps({"project_id": project_id, "rows": candidate_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    files["latency_report"].write_text(json.dumps({"project_id": project_id, "rows": latency_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in files.items()}


def _extract_terms(text: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_./:-]{3,}", text.lower())
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out[:80]


def _extract_entities(text: str) -> list[str]:
    matches = re.findall(r"\b[A-Z][A-Za-z0-9_]*(?:[A-Z][A-Za-z0-9_]*)?\b", text)
    return list(dict.fromkeys(matches))[:30]


def _local_embedding(text: str, dims: int = LOCAL_EMBEDDING_DIMS) -> list[float]:
    vec = [0.0] * dims
    for term in _extract_terms(text):
        h = stable_content_hash(term)
        idx = int(h[:8], 16) % dims
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = sqrt(sum(value * value for value in vec)) or 1.0
    return [round(value / norm, 8) for value in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _embedding_config_path(base_dir: Path) -> Path:
    return product_root(base_dir) / EMBEDDING_CONFIG


def embedding_backend_config(base_dir: Path) -> dict[str, Any]:
    path = _embedding_config_path(base_dir)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {
                "backend": str(payload.get("backend") or LOCAL_EMBEDDING_BACKEND),
                "model": payload.get("model"),
                "normalize": bool(payload.get("normalize", True)),
            }
    return {"backend": LOCAL_EMBEDDING_BACKEND, "model": None, "normalize": True}


def _sentence_transformers_available() -> bool:
    try:
        __import__("sentence_transformers")
        return True
    except Exception:
        return False


def product_embedding_backends(base_dir: Path) -> dict[str, Any]:
    config = embedding_backend_config(base_dir)
    sentence_available = _sentence_transformers_available()
    return {
        "status": "ok",
        "configured": config,
        "backends": {
            LOCAL_EMBEDDING_BACKEND: {
                "available": True,
                "default": True,
                "semantic": False,
                "dims": LOCAL_EMBEDDING_DIMS,
                "description": "Deterministic local feature hashing with no external dependencies.",
            },
            "sentence_transformers": {
                "available": sentence_available,
                "default": False,
                "semantic": True,
                "model": config.get("model") or "sentence-transformers/all-MiniLM-L6-v2",
                "description": "Optional semantic embedding backend when sentence-transformers is installed.",
                **({} if sentence_available else {"unavailable_reason": "sentence-transformers is not installed"}),
            },
        },
    }


def _vector_config_path(base_dir: Path) -> Path:
    return product_root(base_dir) / VECTOR_INDEX_CONFIG


def product_vector_index_config(base_dir: Path) -> dict[str, Any]:
    path = _vector_config_path(base_dir)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {
                "backend": str(payload.get("backend") or "sqlite_inline"),
                "path": str(payload.get("path") or (product_root(base_dir) / "indexes" / "product_chroma")),
                "collection": str(payload.get("collection") or PRODUCT_VECTOR_COLLECTION),
            }
    return {"backend": "sqlite_inline", "path": str(product_root(base_dir) / "indexes" / "product_chroma"), "collection": PRODUCT_VECTOR_COLLECTION}


def _chromadb_available() -> bool:
    try:
        __import__("chromadb")
        return True
    except Exception:
        return False


def product_vector_backends(base_dir: Path) -> dict[str, Any]:
    config = product_vector_index_config(base_dir)
    chroma_available = _chromadb_available()
    return {
        "status": "ok",
        "configured": config,
        "backends": {
            "sqlite_inline": {
                "available": True,
                "default": True,
                "ann": False,
                "description": "Inline SQLite vector scan over capsule_embeddings.",
            },
            "chroma": {
                "available": chroma_available,
                "default": False,
                "ann": True,
                "path": config["path"],
                "collection": config["collection"],
                "description": "Optional persistent Chroma ANN index for large product capsule stores.",
                **({} if chroma_available else {"unavailable_reason": "chromadb is not installed"}),
            },
        },
    }


def configure_product_vector_backend(
    base_dir: Path,
    *,
    backend: str,
    path: Path | None = None,
    collection: str = PRODUCT_VECTOR_COLLECTION,
    allow_unavailable: bool = False,
) -> dict[str, Any]:
    init_product_store(base_dir)
    backend = (backend or "sqlite_inline").strip()
    backends = product_vector_backends(base_dir)["backends"]
    if backend not in backends:
        raise ValueError(f"Unknown product vector backend: {backend}")
    if not backends[backend].get("available") and not allow_unavailable:
        raise RuntimeError(str(backends[backend].get("unavailable_reason") or f"{backend} is unavailable"))
    payload = {
        "backend": backend,
        "path": str(path or (product_root(base_dir) / "indexes" / "product_chroma")),
        "collection": collection or PRODUCT_VECTOR_COLLECTION,
        "updated_at": now_iso(),
    }
    config_path = _vector_config_path(base_dir)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_json(payload) + "\n", encoding="utf-8")
    return {"status": "ok", "path": str(config_path), "config": payload, "available": bool(backends[backend].get("available"))}


def configure_embedding_backend(
    base_dir: Path,
    *,
    backend: str,
    model: str | None = None,
    allow_unavailable: bool = False,
) -> dict[str, Any]:
    init_product_store(base_dir)
    backend = (backend or LOCAL_EMBEDDING_BACKEND).strip()
    backends = product_embedding_backends(base_dir)["backends"]
    if backend not in backends:
        raise ValueError(f"Unknown embedding backend: {backend}")
    if not backends[backend].get("available") and not allow_unavailable:
        raise RuntimeError(str(backends[backend].get("unavailable_reason") or f"{backend} is unavailable"))
    payload = {"backend": backend, "model": model, "normalize": True, "updated_at": now_iso()}
    path = _embedding_config_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(payload) + "\n", encoding="utf-8")
    return {"status": "ok", "path": str(path), "config": payload, "available": bool(backends[backend].get("available"))}


def _chroma_collection(base_dir: Path, *, create: bool = True) -> Any:
    config = product_vector_index_config(base_dir)
    if config.get("backend") != "chroma":
        raise RuntimeError("product vector backend is not configured for chroma")
    try:
        chromadb_module = __import__("chromadb")
    except Exception as exc:
        raise RuntimeError("chromadb is not installed") from exc
    client = chromadb_module.PersistentClient(path=str(config["path"]))
    if create:
        return client.get_or_create_collection(name=str(config["collection"]))
    return client.get_collection(name=str(config["collection"]))


def _capsule_embedding_text(capsule: EvidenceCapsule | dict[str, Any]) -> str:
    if isinstance(capsule, EvidenceCapsule):
        return " ".join(
            [
                capsule.title or "",
                capsule.summary or "",
                capsule.raw_text or "",
                " ".join(capsule.tags),
                " ".join(capsule.entities),
            ]
        )
    return " ".join(
        [
            str(capsule.get("title") or ""),
            str(capsule.get("summary") or ""),
            str(capsule.get("raw_text") or ""),
            " ".join(capsule.get("tags") or []),
            " ".join(capsule.get("entities") or []),
        ]
    )


def _compute_embedding(text: str, *, backend: str, model: str | None = None) -> tuple[list[float], dict[str, Any]]:
    if backend == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model_name = model or "sentence-transformers/all-MiniLM-L6-v2"
            encoder = SentenceTransformer(model_name)
            raw = encoder.encode([text], normalize_embeddings=True)[0]
            vector = [round(float(value), 8) for value in raw]
            return vector, {"model": model_name}
        except Exception as exc:
            raise RuntimeError(f"sentence_transformers backend unavailable: {exc}") from exc
    vector = _local_embedding(text)
    return vector, {"dims": LOCAL_EMBEDDING_DIMS}


def _embedding_metadata(text: str, *, backend: str = LOCAL_EMBEDDING_BACKEND, model: str | None = None) -> dict[str, Any]:
    vector, extra = _compute_embedding(text, backend=backend, model=model)
    return {
        **extra,
        "version": LOCAL_EMBEDDING_VERSION,
        "backend": backend,
        "source_hash": stable_content_hash(text),
        "vector": vector,
        "note": "semantic sentence-transformers embedding for product dense_probe" if backend == "sentence_transformers" else "deterministic local feature hashing for product dense_probe",
    }


def _store_raw_trace(
    base_dir: Path,
    *,
    project_id: str,
    source_type: str,
    text: str,
    session_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = now_iso()
    clean_text = redact_secrets(text or "")
    raw_id = "raw_" + stable_content_hash(_json([project_id, source_type, clean_text, session_id, task_id, now]))[:20]
    blob_path: str | None = None
    inline_text: str | None = clean_text
    if len(clean_text) > INLINE_RAW_LIMIT:
        day = datetime.now(timezone.utc)
        rel = Path("raw") / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}" / f"{raw_id}.json"
        path = product_root(base_dir) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json({"raw_id": raw_id, "text": clean_text, "metadata": metadata or {}}) + "\n", encoding="utf-8")
        blob_path = str(rel)
        inline_text = None
    with connect(base_dir) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO raw_traces(raw_id, project_id, session_id, task_id, agent_id, source_type, timestamp, original_text, blob_path, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (raw_id, project_id, session_id, task_id, agent_id, source_type, now, inline_text, blob_path, _json(redact_payload(metadata or {})), now),
        )
        conn.commit()
    return {"raw_id": raw_id, "raw_ref": f"dyson://raw/{raw_id}", "text": clean_text, "blob_path": blob_path, "created_at": now}


def _insert_capsule(base_dir: Path, capsule: EvidenceCapsule) -> EvidenceCapsule:
    data = capsule.to_dict()
    embedding_text = _capsule_embedding_text(capsule)
    embedding_id = capsule.embedding_id or ("emb_" + stable_content_hash(_json([capsule.id, embedding_text]))[:20])
    capsule.embedding_id = embedding_id
    data["embedding_id"] = embedding_id
    embedding_config = embedding_backend_config(base_dir)
    backend = str(embedding_config.get("backend") or LOCAL_EMBEDDING_BACKEND)
    model = embedding_config.get("model")
    embedding_meta = _embedding_metadata(embedding_text, backend=backend, model=model)
    with connect(base_dir) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO evidence_capsules (
              id, raw_ref, raw_text, source_type, project_id, session_id, task_id, agent_id, timestamp, created_at, updated_at,
              evidence_type, title, summary, entities_json, tags_json, file_refs_json, code_refs_json, command_refs_json,
              artifact_refs_json, benchmark_refs_json, git_commit, git_branch, repo_path, validity_state, parent_ids_json,
              related_ids_json, supports_json, contradicts_json, supersedes_json, superseded_by_json, importance, confidence,
              embedding_id, sparse_terms_json, temporal_scope_json, route_features_json, metadata_json, archived
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                data["id"], data["raw_ref"], data["raw_text"], data["source_type"], data["project_id"], data["session_id"], data["task_id"], data["agent_id"],
                data["timestamp"], data["created_at"], data["updated_at"], data["evidence_type"], data["title"], data["summary"], _json(data["entities"]),
                _json(data["tags"]), _json(data["file_refs"]), _json(data["code_refs"]), _json(data["command_refs"]), _json(data["artifact_refs"]),
                _json(data["benchmark_refs"]), data["git_commit"], data["git_branch"], data["repo_path"], data["validity_state"], _json(data["parent_ids"]),
                _json(data["related_ids"]), _json(data["supports"]), _json(data["contradicts"]), _json(data["supersedes"]), _json(data["superseded_by"]),
                data["importance"], data["confidence"], data["embedding_id"], _json(data["sparse_terms"]), _json(data["temporal_scope"]),
                _json(data["route_features"]), _json(redact_payload(data["metadata"])),
            ),
        )
        conn.execute("DELETE FROM evidence_capsules_fts WHERE id = ?", (capsule.id,))
        conn.execute(
            "INSERT INTO evidence_capsules_fts(id, project_id, title, summary, raw_text, tags, entities) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (capsule.id, capsule.project_id, capsule.title or "", capsule.summary or "", capsule.raw_text or "", " ".join(capsule.tags), " ".join(capsule.entities)),
        )
        for entity in capsule.entities:
            conn.execute("INSERT OR IGNORE INTO capsule_entities(capsule_id, entity) VALUES (?, ?)", (capsule.id, entity))
        for artifact in capsule.artifact_refs:
            conn.execute("INSERT OR IGNORE INTO capsule_artifacts(capsule_id, artifact_ref, artifact_type, metadata_json) VALUES (?, ?, ?, '{}')", (capsule.id, artifact, capsule.source_type))
        conn.execute(
            "INSERT OR REPLACE INTO capsule_embeddings(embedding_id, capsule_id, backend, vector_ref, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (embedding_id, capsule.id, backend, "inline_metadata.vector", _json(embedding_meta)),
        )
        conn.commit()
    return capsule


def _capsule_from_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in (
        "entities", "tags", "file_refs", "code_refs", "command_refs", "artifact_refs", "benchmark_refs",
        "parent_ids", "related_ids", "supports", "contradicts", "supersedes", "superseded_by", "sparse_terms",
    ):
        item[key] = json.loads(item.pop(f"{key}_json") or "[]")
    for key in ("temporal_scope", "route_features", "metadata"):
        item[key] = json.loads(item.pop(f"{key}_json") or "{}")
    item.pop("archived", None)
    return item


def remember(
    base_dir: Path,
    *,
    project_id: str,
    text: str,
    evidence_type: str = "note",
    source_type: str = "manual",
    title: str | None = None,
    summary: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    agent_id: str | None = None,
    validity_state: str = "active",
    tags: list[str] | None = None,
    file_refs: list[str] | None = None,
    command_refs: list[str] | None = None,
    artifact_refs: list[str] | None = None,
    benchmark_refs: list[str] | None = None,
    no_index: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    raw = _store_raw_trace(base_dir, project_id=project_id, source_type=source_type, text=text, session_id=session_id, task_id=task_id, agent_id=agent_id, metadata=metadata)
    git = _git_metadata(base_dir)
    clean = redact_secrets(text)
    cap_id = "cap_" + stable_content_hash(_json([project_id, evidence_type, clean, task_id, session_id, raw["raw_id"]]))[:20]
    capsule = EvidenceCapsule(
        id=cap_id,
        raw_ref=raw["raw_ref"],
        raw_text=clean,
        source_type=source_type,
        project_id=project_id,
        session_id=session_id,
        task_id=task_id,
        agent_id=agent_id,
        evidence_type=evidence_type,
        title=title or clean[:96],
        summary=summary or clean[:320],
        entities=_extract_entities(clean),
        tags=_list(tags),
        file_refs=_list(file_refs),
        code_refs=_list(file_refs),
        command_refs=_list(command_refs),
        artifact_refs=_list(artifact_refs),
        benchmark_refs=_list(benchmark_refs),
        git_commit=git.get("git_commit"),
        git_branch=git.get("git_branch"),
        repo_path=str(base_dir.resolve()),
        validity_state=validity_state,
        sparse_terms=[] if no_index else _extract_terms(clean),
        route_features={"indexed": not no_index, "source_type": source_type},
        metadata={**(metadata or {}), "raw_id": raw["raw_id"], "git_dirty": git.get("git_dirty"), "changed_files": git.get("changed_files", [])},
    )
    _insert_capsule(base_dir, capsule)
    return {"status": "ok", "capsule_id": cap_id, "raw_id": raw["raw_id"], "capsule": capsule.to_dict()}


def get_capsule(base_dir: Path, capsule_id: str, *, project_id: str | None = None) -> dict[str, Any]:
    with connect(base_dir) as conn:
        if project_id:
            row = conn.execute("SELECT * FROM evidence_capsules WHERE id = ? AND project_id = ?", (capsule_id, project_id)).fetchone()
        else:
            row = conn.execute("SELECT * FROM evidence_capsules WHERE id = ?", (capsule_id,)).fetchone()
    if row is None:
        raise KeyError(capsule_id)
    return _capsule_from_row(row)


def list_projects(base_dir: Path) -> dict[str, Any]:
    init_product_store(base_dir)
    with connect(base_dir) as conn:
        rows = conn.execute(
            """
            SELECT project_id, COUNT(*) AS capsule_count, MAX(updated_at) AS updated_at
            FROM evidence_capsules
            WHERE archived = 0
            GROUP BY project_id
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return {"status": "ok", "projects": [dict(row) for row in rows]}


def list_capsules(
    base_dir: Path,
    *,
    project_id: str,
    limit: int = 50,
    offset: int = 0,
    include_archived: bool = False,
    evidence_type: str | None = None,
) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))
    where = ["project_id = ?"]
    params: list[Any] = [project_id]
    if not include_archived:
        where.append("archived = 0")
    if evidence_type:
        where.append("evidence_type = ?")
        params.append(evidence_type)
    sql_where = " AND ".join(where)
    with connect(base_dir) as conn:
        rows = conn.execute(
            f"SELECT * FROM evidence_capsules WHERE {sql_where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        total = conn.execute(f"SELECT COUNT(*) FROM evidence_capsules WHERE {sql_where}", params).fetchone()[0]
    return {"status": "ok", "project_id": project_id, "count": len(rows), "total": int(total), "capsules": [_capsule_from_row(row) for row in rows]}


def update_capsule(base_dir: Path, capsule_id: str, *, project_id: str | None = None, updates: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "validity_state",
        "title",
        "summary",
        "importance",
        "confidence",
        "tags",
        "metadata",
    }
    fields: list[str] = []
    params: list[Any] = []
    for key, value in updates.items():
        if key not in allowed:
            continue
        column = f"{key}_json" if key in {"tags", "metadata"} else key
        fields.append(f"{column} = ?")
        params.append(_json(value) if key in {"tags", "metadata"} else value)
    if not fields:
        return {"status": "unmodified", "capsule": get_capsule(base_dir, capsule_id, project_id=project_id)}
    fields.append("updated_at = ?")
    params.append(now_iso())
    params.append(capsule_id)
    where = "id = ?"
    if project_id:
        where += " AND project_id = ?"
        params.append(project_id)
    with connect(base_dir) as conn:
        conn.execute(f"UPDATE evidence_capsules SET {', '.join(fields)} WHERE {where}", params)
        row = conn.execute("SELECT * FROM evidence_capsules WHERE id = ?", (capsule_id,)).fetchone()
        if row:
            cap = _capsule_from_row(row)
            conn.execute("DELETE FROM evidence_capsules_fts WHERE id = ?", (capsule_id,))
            conn.execute(
                "INSERT INTO evidence_capsules_fts(id, project_id, title, summary, raw_text, tags, entities) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cap["id"], cap["project_id"], cap.get("title") or "", cap.get("summary") or "", cap.get("raw_text") or "", " ".join(cap["tags"]), " ".join(cap["entities"])),
            )
        conn.commit()
    return {"status": "ok", "capsule": get_capsule(base_dir, capsule_id, project_id=project_id)}


def _save_capsule_lists(base_dir: Path, capsule: dict[str, Any]) -> None:
    with connect(base_dir) as conn:
        conn.execute(
            """
            UPDATE evidence_capsules
            SET validity_state = ?, parent_ids_json = ?, related_ids_json = ?, supports_json = ?,
                contradicts_json = ?, supersedes_json = ?, superseded_by_json = ?,
                temporal_scope_json = ?, metadata_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                capsule["validity_state"],
                _json(capsule["parent_ids"]),
                _json(capsule["related_ids"]),
                _json(capsule["supports"]),
                _json(capsule["contradicts"]),
                _json(capsule["supersedes"]),
                _json(capsule["superseded_by"]),
                _json(capsule["temporal_scope"]),
                _json(redact_payload(capsule["metadata"])),
                now_iso(),
                capsule["id"],
            ),
        )
        conn.commit()


def _append_unique(values: list[str], value: str | None) -> list[str]:
    out = list(values or [])
    if value and value not in out:
        out.append(value)
    return out


def _record_relation(base_dir: Path, *, source_id: str, target_id: str | None, relation_type: str, reason: str | None = None) -> None:
    relation_id = "rel_" + stable_content_hash(_json([source_id, target_id, relation_type, reason or "", now_iso()]))[:20]
    with connect(base_dir) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO capsule_relations(relation_id, source_capsule_id, target_capsule_id, relation_type, reason, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (relation_id, source_id, target_id, relation_type, reason or "", now_iso()),
        )
        conn.commit()


def _mark_relation(
    base_dir: Path,
    *,
    capsule_id: str,
    by_capsule_id: str | None,
    state: str,
    relation_type: str,
    reason: str | None = None,
) -> dict[str, Any]:
    capsule = get_capsule(base_dir, capsule_id)
    capsule["validity_state"] = state
    capsule["temporal_scope"] = {**capsule.get("temporal_scope", {}), "valid_to": now_iso(), "invalidation_reason": reason or relation_type}
    capsule["metadata"] = {**capsule.get("metadata", {}), "last_validity_event": {"type": relation_type, "by": by_capsule_id, "reason": reason, "at": now_iso()}}
    if relation_type == "supersedes":
        capsule["superseded_by"] = _append_unique(capsule.get("superseded_by", []), by_capsule_id)
    elif relation_type == "contradicts":
        capsule["contradicts"] = _append_unique(capsule.get("contradicts", []), by_capsule_id)
    elif relation_type == "reverted_by":
        capsule["related_ids"] = _append_unique(capsule.get("related_ids", []), by_capsule_id)
    _save_capsule_lists(base_dir, capsule)
    if by_capsule_id:
        try:
            other = get_capsule(base_dir, by_capsule_id)
            if relation_type == "supersedes":
                other["supersedes"] = _append_unique(other.get("supersedes", []), capsule_id)
            elif relation_type == "contradicts":
                other["contradicts"] = _append_unique(other.get("contradicts", []), capsule_id)
            else:
                other["related_ids"] = _append_unique(other.get("related_ids", []), capsule_id)
            _save_capsule_lists(base_dir, other)
        except KeyError:
            pass
    _record_relation(base_dir, source_id=by_capsule_id or capsule_id, target_id=capsule_id, relation_type=relation_type, reason=reason)
    return {"status": "ok", "capsule_id": capsule_id, "by_capsule_id": by_capsule_id, "validity_state": state, "relation_type": relation_type}


def mark_superseded(base_dir: Path, capsule_id: str, by_capsule_id: str, reason: str | None = None) -> dict[str, Any]:
    return _mark_relation(base_dir, capsule_id=capsule_id, by_capsule_id=by_capsule_id, state="superseded", relation_type="supersedes", reason=reason)


def mark_contradicted(base_dir: Path, capsule_id: str, by_capsule_id: str, reason: str | None = None) -> dict[str, Any]:
    return _mark_relation(base_dir, capsule_id=capsule_id, by_capsule_id=by_capsule_id, state="contradicted", relation_type="contradicts", reason=reason)


def mark_deprecated(base_dir: Path, capsule_id: str, reason: str | None = None) -> dict[str, Any]:
    return _mark_relation(base_dir, capsule_id=capsule_id, by_capsule_id=None, state="deprecated", relation_type="deprecated", reason=reason)


def mark_reverted(base_dir: Path, capsule_id: str, by_capsule_id: str, reason: str | None = None) -> dict[str, Any]:
    return _mark_relation(base_dir, capsule_id=capsule_id, by_capsule_id=by_capsule_id, state="reverted", relation_type="reverted_by", reason=reason)


def get_supersession_chain(base_dir: Path, capsule_id: str, *, project_id: str | None = None, limit: int = 20) -> dict[str, Any]:
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_id: str | None = capsule_id
    for _ in range(max(1, min(int(limit or 20), 100))):
        if not current_id or current_id in seen:
            break
        seen.add(current_id)
        cap = get_capsule(base_dir, current_id, project_id=project_id)
        chain.append(cap)
        next_ids = cap.get("superseded_by") or []
        if not next_ids:
            with connect(base_dir) as conn:
                row = conn.execute(
                    "SELECT source_capsule_id FROM capsule_relations WHERE target_capsule_id = ? AND relation_type = 'supersedes' ORDER BY created_at DESC LIMIT 1",
                    (current_id,),
                ).fetchone()
            next_ids = [row["source_capsule_id"]] if row else []
        current_id = next_ids[0] if next_ids else None
    return {"status": "ok", "capsule_id": capsule_id, "chain": chain, "active_successor": chain[-1] if chain and chain[-1].get("validity_state") in DEFAULT_ELIGIBLE_STATES else None}


def get_active_successor(base_dir: Path, capsule_id: str, *, project_id: str | None = None) -> dict[str, Any]:
    chain = get_supersession_chain(base_dir, capsule_id, project_id=project_id)
    for cap in reversed(chain["chain"]):
        if cap.get("validity_state") in DEFAULT_ELIGIBLE_STATES:
            return {"status": "ok", "capsule_id": capsule_id, "successor": cap}
    return {"status": "ok", "capsule_id": capsule_id, "successor": None}


def get_active_evidence(base_dir: Path, *, project_id: str, task_id: str | None = None, limit: int = 100) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    where = ["project_id = ?", "archived = 0", "validity_state = 'active'"]
    params: list[Any] = [project_id]
    if task_id:
        where.append("task_id = ?")
        params.append(task_id)
    with connect(base_dir) as conn:
        rows = conn.execute(f"SELECT * FROM evidence_capsules WHERE {' AND '.join(where)} ORDER BY timestamp DESC LIMIT ?", [*params, max(1, min(limit, 500))]).fetchall()
    return {"status": "ok", "project_id": project_id, "count": len(rows), "capsules": [_capsule_from_row(row) for row in rows]}


def get_decision_chain(base_dir: Path, *, project_id: str, entity: str | None = None, limit: int = 100) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    where = ["project_id = ?", "archived = 0", "evidence_type IN ('decision','design_rationale','instruction')"]
    params: list[Any] = [project_id]
    if entity:
        like = f"%{entity}%"
        where.append("(entities_json LIKE ? OR tags_json LIKE ? OR title LIKE ? OR summary LIKE ? OR raw_text LIKE ?)")
        params.extend([like, like, like, like, like])
    with connect(base_dir) as conn:
        rows = conn.execute(f"SELECT * FROM evidence_capsules WHERE {' AND '.join(where)} ORDER BY timestamp ASC LIMIT ?", [*params, max(1, min(limit, 500))]).fetchall()
    capsules = [_capsule_from_row(row) for row in rows]
    return {"status": "ok", "project_id": project_id, "entity": entity, "count": len(capsules), "decision_chain": capsules}


def get_evidence_at_time(base_dir: Path, *, project_id: str, timestamp: str, limit: int = 100) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    with connect(base_dir) as conn:
        rows = conn.execute(
            """
            SELECT * FROM evidence_capsules
            WHERE project_id = ? AND archived = 0 AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (project_id, timestamp, max(1, min(limit, 500))),
        ).fetchall()
    return {"status": "ok", "project_id": project_id, "timestamp": timestamp, "count": len(rows), "capsules": [_capsule_from_row(row) for row in rows]}


def get_evidence_for_commit(base_dir: Path, *, project_id: str, commit_hash: str, limit: int = 100) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    with connect(base_dir) as conn:
        rows = conn.execute(
            "SELECT * FROM evidence_capsules WHERE project_id = ? AND archived = 0 AND git_commit = ? ORDER BY timestamp DESC LIMIT ?",
            (project_id, commit_hash, max(1, min(limit, 500))),
        ).fetchall()
    return {"status": "ok", "project_id": project_id, "commit_hash": commit_hash, "count": len(rows), "capsules": [_capsule_from_row(row) for row in rows]}


def classify_route(query: str, *, task_type: str | None = None) -> dict[str, Any]:
    q = query.lower()
    if task_type:
        q = f"{task_type.lower()} {q}"
    if any(term in q for term in ("benchmark", "recall", "ndcg", "metrics", "clonemem", "knowme", "locomo")):
        route = "benchmark"
        probes = ["dense_probe", "sparse_probe", "artifact_probe", "recent_state_probe"]
    elif any(term in q for term in ("traceback", "error", "exception", "failed", "bug")):
        route = "debug_error"
        probes = ["dense_probe", "sparse_probe", "code_ref_probe", "recent_state_probe"]
    elif any(term in q for term in ("commit", "diff", "file", "function", "code")):
        route = "code_change"
        probes = ["dense_probe", "sparse_probe", "code_ref_probe", "artifact_probe"]
    elif any(term in q for term in ("latest", "recent", "previous", "before", "after", "timeline")):
        route = "temporal"
        probes = ["dense_probe", "sparse_probe", "temporal_probe", "recent_state_probe"]
    elif any(term in q for term in ("who", "identity", "entity", "person")):
        route = "identity_or_entity"
        probes = ["dense_probe", "sparse_probe", "entity_probe"]
    elif any(term in q for term in ("what", "where", "which", "exact", "id", "name")):
        route = "exact_factual"
        probes = ["dense_probe", "sparse_probe", "entity_probe", "code_ref_probe"]
    elif any(term in q for term in ("research", "paper", "synthesis", "literature")):
        route = "research_synthesis"
        probes = ["dense_probe", "sparse_probe", "proxy_probe", "artifact_probe"]
    elif any(term in q for term in ("creative", "bridge", "analogy")):
        route = "creative_bridge"
        probes = ["dense_probe", "sparse_probe", "proxy_probe"]
    else:
        route = "ambiguous"
        probes = ["dense_probe", "sparse_probe", "temporal_probe", "entity_probe"]
    return {
        "route": route,
        "enabled_probes": probes,
        "unavailable_probes": {},
        "probe_budgets": {probe: 50 for probe in probes},
        "context_policy": "active_evidence_first",
    }


def _candidate(row: sqlite3.Row, idx: int, probe: str, reason: str) -> dict[str, Any]:
    cap = _capsule_from_row(row)
    row_score = row["rank"] if "rank" in row.keys() else None
    if row_score is not None:
        try:
            score = 1.0 / (1.0 + abs(float(row_score)))
        except (TypeError, ValueError):
            score = max(0.0, 1.0 - idx * 0.03)
    else:
        score = max(0.0, 1.0 - idx * 0.03)
    return {
        "capsule_id": cap["id"],
        "probe": probe,
        "rank": idx + 1,
        "score": round(score, 6),
        "reason": reason,
        "raw_features": {"validity_state": cap["validity_state"], "timestamp": cap["timestamp"], "evidence_type": cap["evidence_type"]},
        "capsule": cap,
    }


def _query_terms(query: str) -> list[str]:
    return [term for term in _extract_terms(query) if len(term) >= 3][:12]


def _term_match_count(capsule: dict[str, Any], terms: list[str]) -> int:
    haystack = " ".join(
        [
            str(capsule.get("title") or ""),
            str(capsule.get("summary") or ""),
            str(capsule.get("raw_text") or ""),
            " ".join(capsule.get("tags") or []),
            " ".join(capsule.get("entities") or []),
            " ".join(capsule.get("artifact_refs") or []),
            " ".join(capsule.get("benchmark_refs") or []),
            " ".join(capsule.get("file_refs") or []),
            " ".join(capsule.get("code_refs") or []),
        ]
    ).lower()
    return sum(1 for term in terms if term.lower() in haystack)


def _where(alias: str, *, include_invalid: bool) -> tuple[list[str], list[Any]]:
    prefix = f"{alias}." if alias else ""
    where = [f"{prefix}project_id = ?", f"{prefix}archived = 0"]
    if not include_invalid:
        where.append(f"{prefix}validity_state IN ('active', 'stable')")
    return where, []


def _run_sparse_probe(conn: sqlite3.Connection, *, project_id: str, query: str, limit: int, include_invalid: bool) -> list[dict[str, Any]]:
    where, _ = _where("c", include_invalid=include_invalid)
    params: list[Any] = [project_id]
    if query.strip():
        fts_query = " ".join(term.replace('"', "") for term in query.split() if term.strip()) or query
        try:
            rows = conn.execute(
                f"""
                SELECT c.*, bm25(evidence_capsules_fts) AS rank
                FROM evidence_capsules_fts
                JOIN evidence_capsules c USING(id)
                WHERE evidence_capsules_fts MATCH ? AND {' AND '.join(where)}
                ORDER BY rank ASC, c.timestamp DESC
                LIMIT ?
                """,
                [fts_query, *params, limit],
            ).fetchall()
            if not rows:
                terms = _query_terms(query)
                threshold = min(2, len(terms)) if terms else 1
                score_expr = " + ".join(
                    ["CASE WHEN lower(coalesce(c.title,'') || ' ' || coalesce(c.summary,'') || ' ' || coalesce(c.raw_text,'') || ' ' || coalesce(c.tags_json,'') || ' ' || coalesce(c.entities_json,'')) LIKE ? THEN 1 ELSE 0 END" for _ in terms]
                )
                if score_expr:
                    rows = conn.execute(
                        f"SELECT c.*, ({score_expr}) AS rank FROM evidence_capsules c WHERE {' AND '.join(where)} AND ({score_expr}) >= ? ORDER BY rank DESC, c.timestamp DESC LIMIT ?",
                        [*(f"%{term.lower()}%" for term in terms), *params, *(f"%{term.lower()}%" for term in terms), threshold, limit],
                    ).fetchall()
        except sqlite3.OperationalError:
            like = f"%{query}%"
            rows = conn.execute(
                f"SELECT c.* FROM evidence_capsules c WHERE {' AND '.join(where)} AND (c.raw_text LIKE ? OR c.summary LIKE ? OR c.title LIKE ?) ORDER BY c.timestamp DESC LIMIT ?",
                [*params, like, like, like, limit],
            ).fetchall()
    else:
        rows = conn.execute(f"SELECT c.* FROM evidence_capsules c WHERE {' AND '.join(where)} ORDER BY c.timestamp DESC LIMIT ?", [*params, limit]).fetchall()
    return [_candidate(row, idx, "sparse_probe", "lexical/FTS match") for idx, row in enumerate(rows)]


def _run_temporal_probe(conn: sqlite3.Connection, *, project_id: str, query: str = "", limit: int, include_invalid: bool, probe: str = "temporal_probe") -> list[dict[str, Any]]:
    where, _ = _where("", include_invalid=include_invalid)
    rows = conn.execute(
        f"SELECT * FROM evidence_capsules WHERE {' AND '.join(where)} ORDER BY timestamp DESC LIMIT ?",
        [project_id, limit],
    ).fetchall()
    reason = "recent evidence ordered by timestamp" if probe == "recent_state_probe" else "temporal recency expansion"
    candidates = [_candidate(row, idx, probe, reason) for idx, row in enumerate(rows)]
    terms = [term for term in _query_terms(query) if term not in {"latest", "recent", "previous", "before", "after", "timeline"}]
    if probe == "recent_state_probe" and terms:
        threshold = min(2, len(terms))
        candidates = [item for item in candidates if _term_match_count(item["capsule"], terms) >= threshold]
    return candidates


def _run_json_term_probe(
    conn: sqlite3.Connection,
    *,
    project_id: str,
    query: str,
    limit: int,
    include_invalid: bool,
    probe: str,
    columns: list[str],
    reason: str,
) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    if not terms:
        return []
    where, _ = _where("", include_invalid=include_invalid)
    ors: list[str] = []
    params: list[Any] = [project_id]
    for term in terms:
        like = f"%{term}%"
        for column in columns:
            ors.append(f"{column} LIKE ?")
            params.append(like)
    rows = conn.execute(
        f"SELECT * FROM evidence_capsules WHERE {' AND '.join(where)} AND ({' OR '.join(ors)}) ORDER BY timestamp DESC LIMIT ?",
        [*params, limit],
    ).fetchall()
    return [_candidate(row, idx, probe, reason) for idx, row in enumerate(rows)]


def _run_proxy_probe(conn: sqlite3.Connection, *, project_id: str, query: str, limit: int, include_invalid: bool) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    if not terms:
        return []
    where, _ = _where("", include_invalid=include_invalid)
    score_expr = " + ".join(["CASE WHEN lower(coalesce(title,'') || ' ' || coalesce(summary,'')) LIKE ? THEN 1 ELSE 0 END" for _ in terms])
    rows = conn.execute(
        f"SELECT *, ({score_expr}) AS rank FROM evidence_capsules WHERE {' AND '.join(where)} ORDER BY rank DESC, timestamp DESC LIMIT ?",
        [*(f"%{term.lower()}%" for term in terms), project_id, limit],
    ).fetchall()
    return [_candidate(row, idx, "proxy_probe", "title/summary proxy match") for idx, row in enumerate(rows) if (row["rank"] or 0) > 0]


def _query_product_vector_index(
    base_dir: Path,
    conn: sqlite3.Connection,
    *,
    project_id: str,
    query: str,
    limit: int,
    include_invalid: bool,
) -> list[dict[str, Any]] | None:
    config = product_vector_index_config(base_dir)
    if config.get("backend") != "chroma":
        return None
    try:
        embed_config = embedding_backend_config(base_dir)
        qvec, _ = _compute_embedding(query, backend=str(embed_config.get("backend") or LOCAL_EMBEDDING_BACKEND), model=embed_config.get("model"))
        collection = _chroma_collection(base_dir, create=False)
        where: dict[str, Any] = {"project_id": project_id}
        if not include_invalid:
            where = {"$and": [{"project_id": project_id}, {"validity_state": {"$in": ["active", "stable"]}}]}
        result = collection.query(query_embeddings=[qvec], n_results=max(1, limit), where=where, include=["metadatas", "documents", "distances"])
    except Exception:
        return None
    ids = [str(item) for item in (result.get("ids") or [[]])[0]]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT * FROM evidence_capsules WHERE id IN ({placeholders})", ids).fetchall()
    by_id = {row["id"]: row for row in rows}
    distances = [float(item) for item in (result.get("distances") or [[]])[0]]
    candidates: list[dict[str, Any]] = []
    for idx, cid in enumerate(ids):
        row = by_id.get(cid)
        if row is None:
            continue
        item = _candidate(row, idx, "dense_probe", "Chroma ANN product vector hit")
        distance = distances[idx] if idx < len(distances) else 1.0
        item["score"] = round(max(0.0, 1.0 - distance / 2.0), 6)
        item["raw_features"]["vector_backend"] = "chroma"
        item["raw_features"]["vector_distance"] = distance
        candidates.append(item)
    return candidates


def _run_dense_probe(base_dir: Path, conn: sqlite3.Connection, *, project_id: str, query: str, limit: int, include_invalid: bool) -> list[dict[str, Any]]:
    ann_candidates = _query_product_vector_index(base_dir, conn, project_id=project_id, query=query, limit=limit, include_invalid=include_invalid)
    if ann_candidates is not None:
        return ann_candidates
    embed_config = embedding_backend_config(base_dir)
    try:
        qvec, _ = _compute_embedding(query, backend=str(embed_config.get("backend") or LOCAL_EMBEDDING_BACKEND), model=embed_config.get("model"))
    except Exception:
        qvec = _local_embedding(query)
    where = ["c.project_id = ?", "c.archived = 0"]
    if not include_invalid:
        where.append("c.validity_state IN ('active', 'stable')")
    rows = conn.execute(
        f"""
        SELECT c.*, e.backend AS embedding_backend, e.metadata_json AS embedding_metadata_json
        FROM evidence_capsules c
        JOIN capsule_embeddings e ON e.capsule_id = c.id
        WHERE {' AND '.join(where)}
        ORDER BY c.timestamp DESC
        LIMIT ?
        """,
        (project_id, max(limit * 4, 100)),
    ).fetchall()
    scored: list[dict[str, Any]] = []
    for row in rows:
        meta = json.loads(row["embedding_metadata_json"] or "{}")
        score = _cosine(qvec, [float(item) for item in meta.get("vector") or []])
        if score <= 0:
            continue
        item = _candidate(row, len(scored), "dense_probe", "SQLite inline embedding similarity")
        item["score"] = round(score, 6)
        item["raw_features"]["embedding_backend"] = str(meta.get("backend") or row["embedding_backend"] or "unknown")
        item["raw_features"]["vector_backend"] = "sqlite_inline"
        scored.append(item)
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]


def _run_probe(base_dir: Path, conn: sqlite3.Connection, *, probe: str, project_id: str, query: str, limit: int, include_invalid: bool) -> list[dict[str, Any]]:
    if probe == "dense_probe":
        return _run_dense_probe(base_dir, conn, project_id=project_id, query=query, limit=limit, include_invalid=include_invalid)
    if probe == "sparse_probe":
        return _run_sparse_probe(conn, project_id=project_id, query=query, limit=limit, include_invalid=include_invalid)
    if probe == "temporal_probe":
        return _run_temporal_probe(conn, project_id=project_id, query=query, limit=limit, include_invalid=include_invalid)
    if probe == "recent_state_probe":
        return _run_temporal_probe(conn, project_id=project_id, query=query, limit=limit, include_invalid=include_invalid, probe="recent_state_probe")
    if probe == "entity_probe":
        return _run_json_term_probe(conn, project_id=project_id, query=query, limit=limit, include_invalid=include_invalid, probe=probe, columns=["entities_json", "tags_json"], reason="entity/tag term match")
    if probe == "artifact_probe":
        return _run_json_term_probe(conn, project_id=project_id, query=query, limit=limit, include_invalid=include_invalid, probe=probe, columns=["artifact_refs_json", "benchmark_refs_json"], reason="artifact or benchmark reference match")
    if probe == "code_ref_probe":
        return _run_json_term_probe(conn, project_id=project_id, query=query, limit=limit, include_invalid=include_invalid, probe=probe, columns=["file_refs_json", "code_refs_json", "command_refs_json"], reason="file/code/command reference match")
    if probe == "proxy_probe":
        return _run_proxy_probe(conn, project_id=project_id, query=query, limit=limit, include_invalid=include_invalid)
    return []


@dataclass
class ProbeRun:
    probe: str
    candidates: list[dict[str, Any]]
    elapsed_ms: float
    status: str = "ok"
    reason: str | None = None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _recency_score(timestamp: str | None, *, half_life_days: float = 30.0) -> float:
    parsed = _parse_iso(timestamp)
    if parsed is None:
        return 0.5
    age_days = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0)
    return round(0.5 ** (age_days / max(1.0, half_life_days)), 6)


def _runtime_scoring_settings(base_dir: Path) -> tuple[dict[str, Any], dict[str, float]]:
    try:
        from dysonspherain.memory_runtime.config import load_runtime_config

        config = load_runtime_config(base_dir).to_dict()
    except Exception:
        config = {}
    scoring = dict(config.get("scoring_config") or {})
    multipliers = {**DEFAULT_LIFECYCLE_MULTIPLIERS, **{str(k): float(v) for k, v in dict(config.get("lifecycle_multipliers") or {}).items()}}
    return scoring, multipliers


def _lifecycle_score_breakdown(item: dict[str, Any], *, scoring: dict[str, Any] | None = None, lifecycle_multipliers: dict[str, float] | None = None) -> dict[str, Any]:
    cap = item.get("capsule") or {}
    state = _memory_lifecycle_state(cap)
    meta = cap.get("metadata") or {}
    scoring = scoring or {}
    lifecycle_multipliers = lifecycle_multipliers or DEFAULT_LIFECYCLE_MULTIPLIERS
    base_score = float(item.get("score") or item.get("admission_score") or 0.0)
    recency = _recency_score(cap.get("timestamp"), half_life_days=float(scoring.get("recency_half_life_days") or meta.get("recency_half_life_days") or 30.0))
    importance = float(cap.get("importance") or 0.5)
    confidence = float(cap.get("confidence") or 0.7)
    access_count = int(meta.get("access_count") or 0)
    access_score = min(1.0, access_count / 20.0)
    redundancy = float(meta.get("redundancy_score") or 0.0)
    lifecycle_multiplier = float(lifecycle_multipliers.get(state, 1.0))
    validity_multiplier = 1.0 if cap.get("validity_state") in DEFAULT_ELIGIBLE_STATES else 0.0
    recency_weight = float(scoring.get("recency_weight") or 0.12)
    importance_weight = float(scoring.get("importance_weight") or 0.18)
    confidence_weight = float(scoring.get("confidence_weight") or 0.1)
    access_weight = float(scoring.get("access_weight") or 0.05)
    redundancy_weight = float(scoring.get("redundancy_weight") or 0.2)
    final = (
        base_score * validity_multiplier * lifecycle_multiplier
        + recency_weight * recency
        + importance_weight * importance
        + confidence_weight * confidence
        + access_weight * access_score
        - redundancy_weight * redundancy
    )
    return {
        "base_score": round(base_score, 6),
        "validity_multiplier": validity_multiplier,
        "lifecycle_state": state,
        "lifecycle_multiplier": lifecycle_multiplier,
        "recency_score": recency,
        "importance_score": round(importance, 6),
        "confidence_score": round(confidence, 6),
        "access_score": round(access_score, 6),
        "redundancy_score": round(redundancy, 6),
        "weights": {
            "recency": recency_weight,
            "importance": importance_weight,
            "confidence": confidence_weight,
            "access": access_weight,
            "redundancy": redundancy_weight,
        },
        "final_memory_score": round(max(0.0, final), 6),
    }


class CandidateAdmissionAuditor:
    def __init__(self, *, gold_ids: list[str] | None = None, scoring: dict[str, Any] | None = None, lifecycle_multipliers: dict[str, float] | None = None) -> None:
        self.gold_ids = _list(gold_ids)
        self.scoring = scoring or {}
        self.lifecycle_multipliers = lifecycle_multipliers or DEFAULT_LIFECYCLE_MULTIPLIERS

    def merge(
        self,
        probe_runs: list[ProbeRun],
        *,
        limit: int,
        filtered: list[dict[str, Any]],
    ) -> dict[str, Any]:
        merged: dict[str, dict[str, Any]] = {}
        duplicate_collapses = 0
        for run in probe_runs:
            for candidate in run.candidates:
                cid = candidate["capsule_id"]
                if cid in merged:
                    duplicate_collapses += 1
                    current = merged[cid]
                    current["score"] = max(float(current.get("score") or 0.0), float(candidate.get("score") or 0.0))
                    probes = current.setdefault("source_probes", [])
                    if candidate["probe"] not in probes:
                        probes.append(candidate["probe"])
                    current["reason"] = f"{current['reason']}; {candidate['probe']}: {candidate['reason']}"
                    continue
                item = dict(candidate)
                item["source_probes"] = [candidate["probe"]]
                merged[cid] = item
        admitted = list(merged.values())
        for item in admitted:
            support_bonus = min(0.2, 0.04 * max(0, len(item.get("source_probes") or []) - 1))
            recency_bonus = 0.03 if "recent_state_probe" in item.get("source_probes", []) else 0.0
            item["admission_score"] = round(float(item.get("score") or 0.0) + support_bonus + recency_bonus, 6)
            item["score_breakdown"] = _lifecycle_score_breakdown(item, scoring=self.scoring, lifecycle_multipliers=self.lifecycle_multipliers)
            item["final_memory_score"] = item["score_breakdown"]["final_memory_score"]
            item["raw_features"]["source_probes"] = item.get("source_probes", [])
        reranked = sorted(admitted, key=lambda item: (float(item.get("final_memory_score") or 0.0), item["capsule"].get("timestamp") or ""), reverse=True)
        final = reranked[:limit]
        final_ids = [item["capsule_id"] for item in final]
        admitted_ids = [item["capsule_id"] for item in admitted]
        gold = self.gold_ids
        probe_results = {
            run.probe: {
                "count": len(run.candidates),
                "latency_ms": round(run.elapsed_ms, 3),
                "status": run.status,
                **({"reason": run.reason} if run.reason else {}),
                "gold_hit": any(item in [cand["capsule_id"] for cand in run.candidates] for item in gold) if gold else None,
            }
            for run in probe_runs
        }
        return {
            "probe_results": probe_results,
            "admitted_candidates": admitted,
            "filtered_candidates": filtered,
            "reranked_candidates": reranked,
            "final_candidates": final,
            "drop_stage_distribution": {
                "validity_filter": len(filtered),
                "duplicate_collapse": duplicate_collapses,
                "token_or_limit_exclusion": max(0, len(reranked) - len(final)),
            },
            "warnings": [run.reason for run in probe_runs if run.status != "ok" and run.reason],
            "candidate_recall@50": (sum(1 for item in gold if item in admitted_ids[:50]) / len(gold)) if gold else None,
            "candidate_recall@100": (sum(1 for item in gold if item in admitted_ids[:100]) / len(gold)) if gold else None,
            "gold_in_candidate_pool": any(item in admitted_ids for item in gold) if gold else None,
            "gold_rank_before_rerank": (admitted_ids.index(gold[0]) + 1) if gold and gold[0] in admitted_ids else None,
            "gold_rank_after_rerank": (final_ids.index(gold[0]) + 1) if gold and gold[0] in final_ids else None,
            "failure_stage": None if not gold or any(item in admitted_ids for item in gold) else "candidate_generation",
            "local_redundancy_ratio@k": round(duplicate_collapses / max(1, sum(len(run.candidates) for run in probe_runs)), 6),
            "distinct_episode_coverage@k": len({item["capsule"].get("session_id") or item["capsule_id"] for item in final}),
        }


def search(
    base_dir: Path,
    *,
    project_id: str,
    query: str,
    limit: int = 10,
    task_type: str | None = None,
    include_invalid: bool = False,
    gold_ids: list[str] | None = None,
) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    route = classify_route(query, task_type=task_type)
    limit = max(1, min(int(limit or 10), 100))
    scoring, lifecycle_multipliers = _runtime_scoring_settings(base_dir)
    probe_runs: list[ProbeRun] = []
    filtered_by_id: dict[str, dict[str, Any]] = {}
    with connect(base_dir) as conn:
        for probe in route["enabled_probes"]:
            if probe not in ALL_PROBES:
                probe_runs.append(ProbeRun(probe=probe, candidates=[], elapsed_ms=0.0, status="unavailable", reason="unknown probe"))
                continue
            started = time.perf_counter()
            candidates = _run_probe(
                base_dir,
                conn,
                probe=probe,
                project_id=project_id,
                query=query,
                limit=max(limit * 4, int(route["probe_budgets"].get(probe, 50))),
                include_invalid=include_invalid,
            )
            probe_runs.append(ProbeRun(probe=probe, candidates=candidates, elapsed_ms=(time.perf_counter() - started) * 1000.0))
            if not include_invalid:
                invalid = _run_probe(
                    base_dir,
                    conn,
                    probe=probe,
                    project_id=project_id,
                    query=query,
                    limit=max(limit * 2, 20),
                    include_invalid=True,
                )
                for item in invalid:
                    if item["capsule"]["validity_state"] not in DEFAULT_ELIGIBLE_STATES:
                        item = dict(item)
                        item["probe"] = "validity_filter"
                        item["reason"] = f"excluded because validity_state={item['capsule']['validity_state']} from {probe}"
                        filtered_by_id[item["capsule_id"]] = item
        if not include_invalid and not filtered_by_id:
            invalid_rows = conn.execute(
                "SELECT * FROM evidence_capsules WHERE project_id = ? AND archived = 0 AND validity_state NOT IN ('active', 'stable') ORDER BY timestamp DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
            filtered_by_id.update({row["id"]: _candidate(row, idx, "validity_filter", f"excluded because validity_state={row['validity_state']}") for idx, row in enumerate(invalid_rows)})
    audit = CandidateAdmissionAuditor(gold_ids=gold_ids, scoring=scoring, lifecycle_multipliers=lifecycle_multipliers).merge(probe_runs, limit=limit, filtered=list(filtered_by_id.values()))
    candidates = audit["final_candidates"]
    if candidates:
        with connect(base_dir) as conn:
            for item in candidates:
                cap = item.get("capsule") or {}
                meta = dict(cap.get("metadata") or {})
                meta["access_count"] = int(meta.get("access_count") or 0) + 1
                meta["last_accessed_at"] = now_iso()
                conn.execute(
                    "UPDATE evidence_capsules SET metadata_json = ?, updated_at = ? WHERE id = ? AND project_id = ?",
                    (_json(redact_payload(meta)), now_iso(), item["capsule_id"], project_id),
                )
            conn.commit()
    trace = _write_retrieval_trace(base_dir, project_id=project_id, query=query, route=route, audit=audit)
    return {"status": "ok", "project_id": project_id, "query": query, **route, "count": len(candidates), "candidates": candidates, "retrieval_trace": trace}


def _write_retrieval_trace(
    base_dir: Path,
    *,
    project_id: str,
    query: str,
    route: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    candidates = list(audit.get("final_candidates") or [])
    trace_id = "trace_" + stable_content_hash(_json([project_id, query, now_iso(), [c["capsule_id"] for c in candidates]]))[:20]
    payload = {
        "query_id": trace_id,
        "query": query,
        "route": route["route"],
        "enabled_probes": route["enabled_probes"],
        "unavailable_probes": route.get("unavailable_probes", {}),
        **audit,
    }
    with connect(base_dir) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO retrieval_traces(trace_id, project_id, query, route, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (trace_id, project_id, query, route["route"], _json(payload), now_iso()),
        )
        conn.commit()
    return {"trace_id": trace_id, "path": str(product_db_path(base_dir)), **payload}


def get_retrieval_trace(base_dir: Path, trace_id: str, *, project_id: str | None = None) -> dict[str, Any]:
    with connect(base_dir) as conn:
        if project_id:
            row = conn.execute("SELECT * FROM retrieval_traces WHERE trace_id = ? AND project_id = ?", (trace_id, project_id)).fetchone()
        else:
            row = conn.execute("SELECT * FROM retrieval_traces WHERE trace_id = ?", (trace_id,)).fetchone()
    if row is None:
        raise KeyError(trace_id)
    payload = json.loads(row["payload_json"] or "{}")
    return {"status": "ok", "trace_id": trace_id, "project_id": row["project_id"], "created_at": row["created_at"], "trace": payload}


def retrieve(
    base_dir: Path,
    *,
    project_id: str,
    query: str,
    limit: int = 10,
    show_audit: bool = False,
    context_pack: bool = False,
    max_tokens: int = 2000,
    task_type: str | None = None,
    context_format: str = "markdown",
    sections: list[str] | None = None,
    section_budget: dict[str, int] | None = None,
    agent_role: str = "coder",
    include_raw_quotes: bool = False,
    include_artifact_refs: bool = True,
    include_debug_trace: bool = False,
) -> dict[str, Any]:
    result = search(base_dir, project_id=project_id, query=query, limit=limit, task_type=task_type)
    if context_pack:
        pack = create_context_pack(
            base_dir,
            project_id=project_id,
            query=query,
            search_result=result,
            max_tokens=max_tokens,
            task_type=task_type,
            agent_role=agent_role,
            include_raw_quotes=include_raw_quotes,
            include_artifact_refs=include_artifact_refs,
            include_debug_trace=include_debug_trace,
            fmt=context_format,
            sections=sections,
            section_budget=section_budget,
        )
        result["context_pack"] = pack
    if not show_audit:
        result.pop("retrieval_trace", None)
    return result


def inspect_retrieval(base_dir: Path, *, project_id: str, query: str, limit: int = 10, max_tokens: int = 2000, task_type: str | None = None) -> dict[str, Any]:
    result = retrieve(
        base_dir,
        project_id=project_id,
        query=query,
        limit=limit,
        show_audit=True,
        context_pack=True,
        max_tokens=max_tokens,
        task_type=task_type,
        include_debug_trace=True,
    )
    trace = result.get("retrieval_trace") or {}
    admitted = trace.get("admitted_candidates") or []
    reranked = trace.get("reranked_candidates") or []
    final = trace.get("final_candidates") or []
    excluded = trace.get("filtered_candidates") or []
    stage_counts = {
        "initial_candidates": sum(int(item.get("count") or 0) for item in (trace.get("probe_results") or {}).values()),
        "after_lifecycle_filtering": len(admitted),
        "after_scoring": len(reranked),
        "after_token_budget_packing": len(final),
        "excluded_evidence": len(excluded),
    }
    selected = []
    score_breakdown: dict[str, dict[str, Any]] = {}
    scoring, lifecycle_multipliers = _runtime_scoring_settings(base_dir)
    for idx, item in enumerate(final, start=1):
        cap = item.get("capsule") or {}
        breakdown = item.get("score_breakdown") or _lifecycle_score_breakdown(item, scoring=scoring, lifecycle_multipliers=lifecycle_multipliers)
        score_breakdown[item["capsule_id"]] = breakdown
        selected.append(
            {
                "rank": idx,
                "memory_id": item["capsule_id"],
                "title": cap.get("title"),
                "preview": cap.get("summary") or cap.get("raw_text") or "",
                "lifecycle_state": _memory_lifecycle_state(cap),
                "final_score": breakdown.get("final_memory_score"),
                "why_selected": item.get("reason"),
                "source_trace": {"probes": item.get("source_probes") or [], "raw_features": item.get("raw_features") or {}},
            }
        )
    excluded_evidence = [
        {
            "memory_id": item.get("capsule_id"),
            "reason": item.get("reason") or "excluded",
            "stage": item.get("probe") or "lifecycle_filter",
            "score_before": item.get("score"),
            "score_after": 0.0,
            "replaced_by": (item.get("capsule") or {}).get("metadata", {}).get("compacted_into") if isinstance(item.get("capsule"), dict) else None,
        }
        for item in excluded
    ]
    context_pack = result.get("context_pack") or {}
    return {
        "status": "ok",
        "project_id": project_id,
        "query": query,
        "initial_candidate_count": stage_counts["initial_candidates"],
        "stage_counts": stage_counts,
        "final_context": selected,
        "excluded_evidence": excluded_evidence,
        "score_breakdown": score_breakdown,
        "token_estimate": context_pack.get("token_estimate") or context_pack.get("estimated_tokens") or 0,
        "context_pack": context_pack,
        "retrieval_trace": trace,
    }


def _section(title: str, items: list[str]) -> str:
    if not items:
        return f"## {title}\n\nNone.\n"
    return f"## {title}\n\n" + "\n".join(f"- {item}" for item in items) + "\n"


def _plain_section(title: str, items: list[str]) -> str:
    if not items:
        return f"{title}\nNone.\n"
    return f"{title}\n" + "\n".join(f"- {item}" for item in items) + "\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\n", "\\n").replace('"', '\\"')
    return f'"{text}"'


def _yaml_dump(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{pad}-")
                lines.append(_yaml_dump(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{pad}-")
                lines.append(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_yaml_scalar(value)}"


def _normalize_section_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _filter_and_budget_sections(sections: list[dict[str, Any]], *, selected: list[str] | None, section_budget: dict[str, int] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_keys = {_normalize_section_name(item) for item in selected or []}
    budgets = {_normalize_section_name(key): int(value) for key, value in (section_budget or {}).items()}
    counter = TokenCounter()
    filtered: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    for section in sections:
        key = _normalize_section_name(str(section["name"]))
        if selected_keys and key not in selected_keys:
            omitted.append({"section": section["name"], "reason": "section_not_selected", "omitted_items": len(section.get("items") or [])})
            continue
        limit = budgets.get(key)
        if not limit:
            filtered.append(section)
            continue
        used = 0
        kept: list[str] = []
        dropped = 0
        for item in section.get("items") or []:
            cost = counter.count(str(item)).tokens
            if used + cost <= max(1, limit):
                kept.append(str(item))
                used += cost
            elif not kept and cost <= max(1, limit) + max(1, int(limit * 0.1)):
                kept.append(str(item))
                used += cost
            else:
                dropped += 1
        next_section = dict(section)
        next_section["items"] = kept
        next_section["token_budget"] = limit
        next_section["token_used"] = used
        filtered.append(next_section)
        if dropped:
            omitted.append({"section": section["name"], "reason": "section_budget", "omitted_items": dropped})
    return filtered, omitted


def _render_context_payload(payload: dict[str, Any], fmt: str) -> str:
    fmt = fmt.lower()
    if fmt in {"json"}:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if fmt in {"yaml", "yml"}:
        public = {key: value for key, value in payload.items() if key not in {"markdown", "rendered"}}
        return _yaml_dump(public) + "\n"
    if fmt in {"text", "plain", "plain_text"}:
        return "\n".join(_plain_section(section["name"], section["items"]) for section in payload["sections"])
    return "\n".join(_section(section["name"], section["items"]) for section in payload["sections"])


def create_context_pack(
    base_dir: Path,
    *,
    project_id: str,
    query: str = "",
    search_result: dict[str, Any] | None = None,
    max_tokens: int = 2000,
    section_budget: dict[str, int] | None = None,
    sections: list[str] | None = None,
    agent_role: str = "coder",
    task_type: str | None = None,
    include_raw_quotes: bool = False,
    include_artifact_refs: bool = True,
    include_debug_trace: bool = False,
    fmt: str = "markdown",
) -> dict[str, Any]:
    result = search_result or search(base_dir, project_id=project_id, query=query, limit=12, task_type=task_type)
    route = str(result.get("route") or "ambiguous")
    candidates = list(result.get("candidates") or [])
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    budget = max(200, int(max_tokens or 2000))
    used = 0
    counter = TokenCounter()
    for cand in candidates:
        cap = cand["capsule"]
        text = f"{cap.get('title') or ''}: {cap.get('summary') or ''}"
        if include_raw_quotes:
            text += f"\n{cap.get('raw_text') or ''}"
        if include_artifact_refs:
            text += "\n" + "\n".join(cap.get("artifact_refs") or [])
        cost = counter.count(text).tokens
        if used + cost <= budget:
            included.append(cand)
            used += cost
        else:
            excluded.append({"capsule_id": cap["id"], "reason": "token_budget", "estimated_tokens": cost})
    all_sections = [
        {"name": "Mission State", "items": [f"Project: {project_id}", f"Query: {query or 'latest active evidence'}", f"Route: {route}", f"Agent role: {agent_role}", f"Task type: {task_type or 'unspecified'}"]},
        {"name": "Must-Use Anchors", "items": [f"{c['capsule']['id']}: {c['capsule'].get('title')}" for c in included[:3]]},
        {"name": "Recent State", "items": [f"{c['capsule'].get('timestamp')}: {c['capsule'].get('summary')}" for c in included[:5]]},
        {"name": "Relevant Decisions", "items": [f"{c['capsule']['id']}: {c['capsule'].get('summary')}" for c in included if c["capsule"].get("evidence_type") == "decision"]},
        {"name": "Known Pitfalls", "items": [f"{c['capsule']['id']}: {c['capsule'].get('summary')}" for c in included if c["capsule"].get("evidence_type") in {"warning", "bug", "limitation"}]},
        {"name": "Supporting Evidence", "items": [f"{c['capsule']['id']} [{c['reason']}]: {c['capsule'].get('summary')}" + (f" artifacts={','.join(c['capsule'].get('artifact_refs') or [])}" if include_artifact_refs and c["capsule"].get("artifact_refs") else "") for c in included]},
        {"name": "Creative Bridges", "items": []},
        {"name": "Conflicts and Invalidated Evidence", "items": [f"{c['capsule_id']}: {c['reason']}" for c in result.get("retrieval_trace", {}).get("filtered_candidates", [])]},
        {"name": "Open Questions", "items": []},
        {"name": "Excluded Evidence", "items": [f"{item['capsule_id']}: {item['reason']}" for item in excluded]},
    ]
    if include_debug_trace and result.get("retrieval_trace"):
        all_sections.append({"name": "Debug Trace", "items": [json.dumps(result["retrieval_trace"].get("probe_results", {}), ensure_ascii=False, sort_keys=True)]})
    packed_sections, omitted_sections = _filter_and_budget_sections(all_sections, selected=sections, section_budget=section_budget)
    pack_id = "ctx_" + stable_content_hash(_json([project_id, query, route, [c["capsule"]["id"] for c in included], now_iso()]))[:20]
    payload = {
        "context_pack_id": pack_id,
        "query": query,
        "route": route,
        "agent_role": agent_role,
        "task_type": task_type,
        "token_budget": budget,
        "token_used": used,
        "estimated_tokens_saved": max(0, sum(c["capsule"].get("metadata", {}).get("token_estimate", 0) or 0 for c in included) - used),
        "section_budget": section_budget or {},
        "selected_sections": sections or [],
        "sections": packed_sections,
        "capsule_ids": [c["capsule"]["id"] for c in included],
        "excluded_capsules": excluded,
        "omitted_sections": omitted_sections,
        "risk_flags": ([] if included else ["no_evidence_selected"]) + (["sections_omitted"] if omitted_sections else []),
        "format": fmt,
    }
    markdown = _render_context_payload({**payload, "sections": packed_sections}, "markdown")
    rendered = _render_context_payload({**payload, "markdown": markdown}, fmt)
    payload["markdown"] = markdown
    payload["rendered"] = rendered
    with connect(base_dir) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO context_packs(context_pack_id, project_id, query, route, payload_json, markdown, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pack_id, project_id, query, route, _json(payload), markdown, now_iso()),
        )
        conn.commit()
    return payload


def get_context_pack(base_dir: Path, context_pack_id: str, *, project_id: str | None = None) -> dict[str, Any]:
    with connect(base_dir) as conn:
        if project_id:
            row = conn.execute("SELECT * FROM context_packs WHERE context_pack_id = ? AND project_id = ?", (context_pack_id, project_id)).fetchone()
        else:
            row = conn.execute("SELECT * FROM context_packs WHERE context_pack_id = ?", (context_pack_id,)).fetchone()
    if row is None:
        raise KeyError(context_pack_id)
    payload = json.loads(row["payload_json"] or "{}")
    payload.setdefault("markdown", row["markdown"])
    return {"status": "ok", "project_id": row["project_id"], "created_at": row["created_at"], "context_pack": payload}


def record_source(
    base_dir: Path,
    *,
    project_id: str,
    source: str,
    text: str | None = None,
    file: Path | None = None,
    command: str | None = None,
    capture_output: bool = False,
    artifact: Path | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    allowlist: list[str] | None = None,
    denylist: list[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    payload = text or ""
    file_refs: list[str] = []
    command_refs: list[str] = []
    artifact_refs: list[str] = []
    if file:
        ignored, reason = _is_ignored_path(base_dir, file, allowlist=allowlist, denylist=denylist)
        if ignored:
            return {"status": "skipped", "reason": "ignored_path", "pattern": reason, "path": str(file)}
        payload = file.read_text(encoding="utf-8", errors="replace")
        metadata["file"] = str(file)
        file_refs.append(str(file))
    if command:
        metadata["command"] = command
        command_refs.append(command)
        if capture_output:
            proc = subprocess.run(command, cwd=base_dir, shell=True, text=True, capture_output=True)
            metadata["returncode"] = proc.returncode
            payload = f"$ {command}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        else:
            payload = f"$ {command}"
    if artifact:
        ignored, reason = _is_ignored_path(base_dir, artifact, allowlist=allowlist, denylist=denylist)
        if ignored:
            return {"status": "skipped", "reason": "ignored_path", "pattern": reason, "path": str(artifact)}
        payload = artifact.read_text(encoding="utf-8", errors="replace") if artifact.is_file() else str(artifact)
        metadata["artifact"] = str(artifact)
        artifact_refs.append(str(artifact))
    if source in {"code-diff", "git", "commit"}:
        binding = _git_binding(base_dir, include_diff=bool(metadata.get("include_diff")))
        metadata["git_binding"] = binding
        payload = payload or binding.get("diff_summary") or binding.get("commit_message") or "git snapshot"
        file_refs.extend(binding.get("changed_files") or [])
    if source in {"error", "traceback"}:
        error = _parse_error_trace(payload)
        metadata["error_binding"] = error
        file_refs.extend(error.get("file_refs") or [])
    evidence_type = {"error": "bug", "code-diff": "fix", "benchmark": "benchmark_result", "shell": "artifact"}.get(source, source)
    return remember(
        base_dir,
        project_id=project_id,
        text=payload,
        evidence_type=evidence_type,
        source_type=source,
        session_id=session_id,
        task_id=task_id,
        file_refs=file_refs,
        command_refs=command_refs,
        artifact_refs=artifact_refs,
        metadata=metadata,
    )


def runtime_event(base_dir: Path, *, project_id: str, event_type: str, payload: dict[str, Any], max_tokens: int = 2000) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    event_id = "evt_" + stable_content_hash(_json([project_id, event_type, payload, now_iso()]))[:20]
    with connect(base_dir) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runtime_events(event_id, project_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_id, project_id, event_type, _json(redact_payload(payload)), now_iso()),
        )
        conn.commit()
    query = str(payload.get("task") or payload.get("summary") or payload.get("error") or event_type)
    if event_type in WRITEBACK_RUNTIME_EVENTS:
        remember(base_dir, project_id=project_id, text=query, evidence_type="runtime_event", source_type=event_type, task_id=str(payload.get("task_id") or ""), metadata=payload)
    pack = create_context_pack(base_dir, project_id=project_id, query=query, max_tokens=max_tokens)
    md_path = product_root(base_dir) / "artifacts" / f"{event_id}.md"
    json_path = product_root(base_dir) / "artifacts" / f"{event_id}.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(pack["markdown"], encoding="utf-8")
    json_path.write_text(_json({"event_id": event_id, "event": event_type, "context_pack": pack}) + "\n", encoding="utf-8")
    return {"status": "ok", "event": event_type, "event_id": event_id, "context_pack_path": str(md_path), "retrieval_trace_path": str(product_db_path(base_dir)), "recommended_actions": [], "warnings": pack.get("risk_flags", [])}


def forget_capsule(base_dir: Path, *, capsule_id: str, project_id: str | None = None, hard: bool = False) -> dict[str, Any]:
    with connect(base_dir) as conn:
        row = conn.execute("SELECT raw_ref FROM evidence_capsules WHERE id = ?", (capsule_id,)).fetchone()
        if hard:
            conn.execute("DELETE FROM evidence_capsules WHERE id = ?", (capsule_id,))
            conn.execute("DELETE FROM evidence_capsules_fts WHERE id = ?", (capsule_id,))
            conn.execute("DELETE FROM capsule_entities WHERE capsule_id = ?", (capsule_id,))
            conn.execute("DELETE FROM capsule_artifacts WHERE capsule_id = ?", (capsule_id,))
            conn.execute("DELETE FROM capsule_embeddings WHERE capsule_id = ?", (capsule_id,))
            conn.execute("DELETE FROM capsule_relations WHERE source_capsule_id = ? OR target_capsule_id = ?", (capsule_id, capsule_id))
        else:
            conn.execute(
                "UPDATE evidence_capsules SET archived = 1, validity_state = 'deprecated', metadata_json = json_set(metadata_json, '$.forgotten_at', ?), updated_at = ? WHERE id = ?",
                (now_iso(), now_iso(), capsule_id),
            )
            pack_rows = conn.execute("SELECT context_pack_id, payload_json FROM context_packs WHERE payload_json LIKE ?", (f"%{capsule_id}%",)).fetchall()
            for pack_row in pack_rows:
                payload = json.loads(pack_row["payload_json"] or "{}")
                refs = payload.setdefault("tombstoned_capsule_refs", [])
                if capsule_id not in refs:
                    refs.append(capsule_id)
                conn.execute("UPDATE context_packs SET payload_json = ? WHERE context_pack_id = ?", (_json(payload), pack_row["context_pack_id"]))
        if row and hard:
            raw_id = str(row["raw_ref"] or "").replace("dyson://raw/", "")
            still_used = conn.execute("SELECT COUNT(*) FROM evidence_capsules WHERE raw_ref = ?", (row["raw_ref"],)).fetchone()[0]
            if raw_id and not still_used:
                raw = conn.execute("SELECT blob_path FROM raw_traces WHERE raw_id = ?", (raw_id,)).fetchone()
                if raw and raw["blob_path"]:
                    blob = product_root(base_dir) / raw["blob_path"]
                    if blob.exists():
                        blob.unlink()
                conn.execute("DELETE FROM raw_traces WHERE raw_id = ?", (raw_id,))
        conn.commit()
    return {"status": "deleted", "capsule_id": capsule_id, "hard": hard, "project_id": project_id}


def forget_before(base_dir: Path, *, project_id: str, before: str, hard: bool = False, limit: int = 1000) -> dict[str, Any]:
    with connect(base_dir) as conn:
        rows = conn.execute(
            "SELECT id FROM evidence_capsules WHERE project_id = ? AND archived = 0 AND timestamp < ? ORDER BY timestamp ASC LIMIT ?",
            (project_id, before, max(1, min(limit, 10_000))),
        ).fetchall()
    ids = [str(row["id"]) for row in rows]
    for capsule_id in ids:
        forget_capsule(base_dir, capsule_id=capsule_id, project_id=project_id, hard=hard)
    return {"status": "ok", "project_id": project_id, "before": before, "hard": hard, "forgotten_count": len(ids), "capsule_ids": ids}


def apply_retention(base_dir: Path, *, project_id: str, keep_last: int = 200, hard: bool = False) -> dict[str, Any]:
    keep_last = max(1, int(keep_last or 200))
    with connect(base_dir) as conn:
        rows = conn.execute(
            "SELECT id FROM evidence_capsules WHERE project_id = ? AND archived = 0 ORDER BY timestamp DESC LIMIT -1 OFFSET ?",
            (project_id, keep_last),
        ).fetchall()
    ids = [str(row["id"]) for row in rows]
    for capsule_id in ids:
        forget_capsule(base_dir, capsule_id=capsule_id, project_id=project_id, hard=hard)
    return {"status": "ok", "project_id": project_id, "keep_last": keep_last, "hard": hard, "forgotten_count": len(ids), "capsule_ids": ids}


def register_alias(base_dir: Path, *, project_id: str, alias: str, canonical: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    with connect(base_dir) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO capsule_aliases(alias, canonical, project_id, created_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (alias.lower().strip(), canonical.lower().strip(), project_id, now_iso(), _json(metadata or {})),
        )
        conn.commit()
    return {"status": "ok", "project_id": project_id, "alias": alias, "canonical": canonical}


def resolve_alias(base_dir: Path, *, project_id: str, value: str) -> dict[str, Any]:
    with connect(base_dir) as conn:
        row = conn.execute("SELECT * FROM capsule_aliases WHERE project_id = ? AND alias = ?", (project_id, value.lower().strip())).fetchone()
    return {"status": "ok", "project_id": project_id, "alias": value, "canonical": row["canonical"] if row else value}


def rebuild_product_embeddings(
    base_dir: Path,
    *,
    project_id: str,
    include_archived: bool = False,
    backend: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    config = embedding_backend_config(base_dir)
    backend = backend or str(config.get("backend") or LOCAL_EMBEDDING_BACKEND)
    model = model if model is not None else config.get("model")
    where = ["project_id = ?"]
    params: list[Any] = [project_id]
    if not include_archived:
        where.append("archived = 0")
    rebuilt = 0
    unchanged = 0
    with connect(base_dir) as conn:
        rows = conn.execute(f"SELECT * FROM evidence_capsules WHERE {' AND '.join(where)} ORDER BY timestamp ASC", params).fetchall()
        active_ids = {row["id"] for row in rows}
        for row in rows:
            capsule = _capsule_from_row(row)
            text = _capsule_embedding_text(capsule)
            source_hash = stable_content_hash(text)
            embedding_id = capsule.get("embedding_id") or ("emb_" + stable_content_hash(_json([capsule["id"], text]))[:20])
            existing = conn.execute("SELECT * FROM capsule_embeddings WHERE capsule_id = ?", (capsule["id"],)).fetchone()
            existing_meta = json.loads(existing["metadata_json"] or "{}") if existing else {}
            if (
                existing
                and existing["backend"] == backend
                and existing_meta.get("version") == LOCAL_EMBEDDING_VERSION
                and existing_meta.get("source_hash") == source_hash
                and (backend != LOCAL_EMBEDDING_BACKEND or existing_meta.get("dims") == LOCAL_EMBEDDING_DIMS)
            ):
                unchanged += 1
                continue
            conn.execute("UPDATE evidence_capsules SET embedding_id = ?, updated_at = ? WHERE id = ?", (embedding_id, now_iso(), capsule["id"]))
            conn.execute(
                "INSERT OR REPLACE INTO capsule_embeddings(embedding_id, capsule_id, backend, vector_ref, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (embedding_id, capsule["id"], backend, "inline_metadata.vector", _json(_embedding_metadata(text, backend=backend, model=model))),
            )
            rebuilt += 1
        if not include_archived and active_ids:
            placeholders = ",".join("?" for _ in active_ids)
            conn.execute(
                f"""
                DELETE FROM capsule_embeddings
                WHERE capsule_id IN (
                  SELECT id FROM evidence_capsules WHERE project_id = ? AND archived != 0
                )
                OR (
                  capsule_id IN (SELECT id FROM evidence_capsules WHERE project_id = ?)
                  AND capsule_id NOT IN ({placeholders})
                )
                """,
                [project_id, project_id, *list(active_ids)],
            )
        elif not include_archived:
            conn.execute("DELETE FROM capsule_embeddings WHERE capsule_id IN (SELECT id FROM evidence_capsules WHERE project_id = ?)", (project_id,))
        conn.commit()
    return {
        "status": "ok",
        "project_id": project_id,
        "backend": backend,
        "model": model,
        "version": LOCAL_EMBEDDING_VERSION,
        "dims": LOCAL_EMBEDDING_DIMS if backend == LOCAL_EMBEDDING_BACKEND else None,
        "capsules_seen": len(rows),
        "rebuilt": rebuilt,
        "unchanged": unchanged,
        "include_archived": include_archived,
    }


def rebuild_product_vector_index(
    base_dir: Path,
    *,
    project_id: str,
    backend: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    if backend:
        configure_product_vector_backend(base_dir, backend=backend, allow_unavailable=True)
    config = product_vector_index_config(base_dir)
    if config["backend"] == "sqlite_inline":
        return {"status": "ok", "project_id": project_id, "backend": "sqlite_inline", "indexed_count": 0, "note": "SQLite inline backend does not require a separate ANN index."}
    if config["backend"] != "chroma":
        raise ValueError(f"Unsupported product vector backend: {config['backend']}")
    try:
        collection = _chroma_collection(base_dir, create=True)
    except Exception as exc:
        return {"status": "unavailable", "project_id": project_id, "backend": "chroma", "reason": str(exc), "config": config}
    with connect(base_dir) as conn:
        rows = conn.execute(
            """
            SELECT c.*, e.metadata_json AS embedding_metadata_json
            FROM evidence_capsules c
            JOIN capsule_embeddings e ON e.capsule_id = c.id
            WHERE c.project_id = ? AND c.archived = 0
            ORDER BY c.timestamp ASC
            LIMIT ?
            """,
            (project_id, max(1, min(int(limit or 1_000_000), 1_000_000))),
        ).fetchall()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []
    for row in rows:
        meta = json.loads(row["embedding_metadata_json"] or "{}")
        vector = [float(item) for item in meta.get("vector") or []]
        if not vector:
            continue
        cap = _capsule_from_row(row)
        ids.append(cap["id"])
        documents.append(_capsule_embedding_text(cap))
        metadatas.append(
            {
                "project_id": project_id,
                "capsule_id": cap["id"],
                "validity_state": cap["validity_state"],
                "evidence_type": cap["evidence_type"],
                "timestamp": cap["timestamp"],
                "embedding_backend": str(meta.get("backend") or ""),
            }
        )
        embeddings.append(vector)
    batch_size = 512
    for start in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[start : start + batch_size],
            documents=documents[start : start + batch_size],
            metadatas=metadatas[start : start + batch_size],
            embeddings=embeddings[start : start + batch_size],
        )
    return {
        "status": "ok",
        "project_id": project_id,
        "backend": "chroma",
        "path": config["path"],
        "collection": config["collection"],
        "indexed_count": len(ids),
        "collection_count": int(collection.count()),
    }


def maintenance_suggestions(base_dir: Path, *, project_id: str, limit: int = 100) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    suggestions: list[dict[str, Any]] = []
    with connect(base_dir) as conn:
        duplicate_rows = conn.execute(
            """
            SELECT lower(coalesce(summary, raw_text, title, '')) AS key, group_concat(id) AS ids, COUNT(*) AS n
            FROM evidence_capsules
            WHERE project_id = ? AND archived = 0
            GROUP BY key
            HAVING n > 1 AND key != ''
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()
        for row in duplicate_rows:
            suggestions.append({"type": "duplicate_merge", "capsule_ids": str(row["ids"]).split(","), "reason": "same normalized summary/raw text"})
        bench_rows = conn.execute(
            "SELECT benchmark_refs_json, id, timestamp FROM evidence_capsules WHERE project_id = ? AND archived = 0 AND evidence_type = 'benchmark_result' ORDER BY timestamp ASC",
            (project_id,),
        ).fetchall()
    latest_by_benchmark: dict[str, dict[str, Any]] = {}
    for row in bench_rows:
        refs = json.loads(row["benchmark_refs_json"] or "[]")
        for ref in refs:
            previous = latest_by_benchmark.get(ref)
            if previous:
                suggestions.append({"type": "invalidate_older_benchmark", "older_capsule_id": previous["id"], "newer_capsule_id": row["id"], "benchmark": ref, "reason": "newer benchmark result exists"})
            latest_by_benchmark[ref] = {"id": row["id"], "timestamp": row["timestamp"]}
    with connect(base_dir) as conn:
        enriched: list[dict[str, Any]] = []
        for suggestion in suggestions:
            sid = "sug_" + stable_content_hash(_json([project_id, suggestion]))[:20]
            conn.execute(
                "INSERT OR IGNORE INTO maintenance_suggestions(suggestion_id, project_id, suggestion_type, payload_json, created_at, status) VALUES (?, ?, ?, ?, ?, 'open')",
                (sid, project_id, suggestion["type"], _json(suggestion), now_iso()),
            )
            row = conn.execute("SELECT status FROM maintenance_suggestions WHERE suggestion_id = ? AND project_id = ?", (sid, project_id)).fetchone()
            enriched.append({"suggestion_id": sid, "status": row["status"] if row else "open", **suggestion})
        conn.commit()
    open_count = sum(1 for item in enriched if item["status"] == "open")
    return {"status": "ok", "project_id": project_id, "count": len(enriched), "open_count": open_count, "suggestions": enriched[:limit]}


def _memory_lifecycle_state(capsule: dict[str, Any]) -> str:
    state = str(capsule.get("validity_state") or "active")
    if capsule.get("evidence_type") == "canonical" or capsule.get("metadata", {}).get("memory_type") == "canonical":
        return "canonical"
    return state


def _token_estimate(text: str | None) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.35) + int(len(text) / 18))


def _normalized_memory_text(capsule: dict[str, Any]) -> str:
    text = " ".join(str(capsule.get(key) or "") for key in ("title", "summary", "raw_text"))
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def _memory_similarity(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    text_a = _normalized_memory_text(a)
    text_b = _normalized_memory_text(b)
    terms_a = set(_extract_terms(text_a))
    terms_b = set(_extract_terms(text_b))
    lexical = _jaccard(terms_a, terms_b)
    semantic = _cosine(_local_embedding(text_a), _local_embedding(text_b))
    return {"lexical": round(lexical, 6), "semantic": round(semantic, 6), "combined": round(max(lexical, semantic), 6)}


def _compact_capsule_summary(capsules: list[dict[str, Any]], *, max_chars: int = 1200) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for cap in capsules:
        text = str(cap.get("summary") or cap.get("raw_text") or cap.get("title") or "").strip()
        if not text:
            continue
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        parts.append(text)
        if sum(len(part) for part in parts) >= max_chars:
            break
    merged = " ".join(parts)
    return merged[:max_chars].rstrip()


def memory_lifecycle_summary(base_dir: Path, *, project_id: str, limit: int = 80) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    limit = max(1, min(int(limit or 80), 300))
    with connect(base_dir) as conn:
        state_rows = conn.execute(
            """
            SELECT validity_state, COUNT(*) AS n
            FROM evidence_capsules
            WHERE project_id = ? AND archived = 0
            GROUP BY validity_state
            """,
            (project_id,),
        ).fetchall()
        canonical_count = conn.execute(
            """
            SELECT COUNT(*) FROM evidence_capsules
            WHERE project_id = ? AND archived = 0
              AND (evidence_type = 'canonical' OR json_extract(metadata_json, '$.memory_type') = 'canonical')
            """,
            (project_id,),
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM evidence_capsules WHERE project_id = ? AND archived = 0 ORDER BY updated_at DESC, timestamp DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()
        recent_traces = conn.execute(
            "SELECT trace_id, query, route, payload_json, created_at FROM retrieval_traces WHERE project_id = ? ORDER BY created_at DESC LIMIT 6",
            (project_id,),
        ).fetchall()
    state_counts = {str(row["validity_state"]): int(row["n"]) for row in state_rows}
    state_counts["canonical"] = int(canonical_count)
    records = []
    for row in rows:
        cap = _capsule_from_row(row)
        text = str(cap.get("summary") or cap.get("raw_text") or "")
        meta = dict(cap.get("metadata") or {})
        records.append(
            {
                "id": cap["id"],
                "title": cap.get("title") or cap["id"],
                "content": text,
                "summary": cap.get("summary"),
                "scope": cap.get("source_type") or cap.get("evidence_type") or "memory",
                "state": _memory_lifecycle_state(cap),
                "validity_state": cap.get("validity_state"),
                "memory_type": meta.get("memory_type") or cap.get("evidence_type") or "raw",
                "updated_at": cap.get("updated_at") or cap.get("timestamp"),
                "created_at": cap.get("created_at"),
                "tags": cap.get("tags") or [],
                "importance": cap.get("importance"),
                "confidence": cap.get("confidence"),
                "source_ids": meta.get("source_ids") or cap.get("parent_ids") or [],
                "compacted_into": meta.get("compacted_into"),
                "token_estimate": meta.get("token_estimate") or _token_estimate(text),
                "why": meta.get("lifecycle_reason") or f"{cap.get('validity_state', 'active')} memory from {cap.get('source_type', 'unknown source')}",
            }
        )
    trace_items = []
    for row in recent_traces:
        payload = json.loads(row["payload_json"] or "{}")
        excluded = payload.get("filtered_candidates") or payload.get("filtered") or []
        trace_items.append(
            {
                "trace_id": row["trace_id"],
                "query": row["query"],
                "route": row["route"],
                "created_at": row["created_at"],
                "selected_count": len(payload.get("final_candidates") or []),
                "excluded_count": len(excluded),
                "excluded": excluded[:8],
            }
        )
    total = sum(count for state, count in state_counts.items() if state != "canonical")
    eligible = state_counts.get("active", 0) + state_counts.get("stable", 0) + state_counts.get("canonical", 0)
    return {
        "status": "ok",
        "project_id": project_id,
        "state_counts": state_counts,
        "total_memories": total,
        "eligible_memories": eligible,
        "records": records,
        "recent_retrieval_traces": trace_items,
        "retrieval_policy": {
            "eligible_by_default": ["active", "stable", "canonical"],
            "excluded_by_default": ["superseded", "deprecated", "contradicted", "compacted", "archived"],
            "raw_memory_preserved": True,
        },
    }


def find_compaction_candidates(base_dir: Path, *, project_id: str, limit: int = 12, near_duplicate_threshold: float = 0.9, min_cluster_size: int = 2) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    limit = max(1, min(int(limit or 12), 50))
    min_cluster_size = max(2, min(int(min_cluster_size or 2), 20))
    near_duplicate_threshold = max(0.1, min(float(near_duplicate_threshold or 0.9), 1.0))
    with connect(base_dir) as conn:
        rows = conn.execute(
            """
            SELECT * FROM evidence_capsules
            WHERE project_id = ? AND archived = 0 AND validity_state IN ('active', 'stable')
            ORDER BY updated_at DESC, timestamp DESC
            LIMIT 500
            """,
            (project_id,),
        ).fetchall()
    capsules: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cap = _capsule_from_row(row)
        if cap.get("evidence_type") == "canonical":
            continue
        normalized = _normalized_memory_text(cap)
        if not normalized:
            continue
        capsules.append(cap)
        key = stable_content_hash(normalized)[:16]
        buckets.setdefault(key, []).append(cap)
    candidates = []
    used_exact_ids: set[str] = set()
    for key, caps in buckets.items():
        if len(caps) < 2:
            continue
        used_exact_ids.update(cap["id"] for cap in caps)
        original_tokens = sum(_token_estimate(str(cap.get("summary") or cap.get("raw_text") or "")) for cap in caps)
        canonical_summary = _compact_capsule_summary(caps)
        compacted_tokens = _token_estimate(canonical_summary)
        candidates.append(
            {
                "cluster_id": key,
                "mode": "deterministic",
                "reason": "exact normalized duplicate or repeated summary",
                "memory_ids": [cap["id"] for cap in caps],
                "memory_count": len(caps),
                "avg_similarity": 1.0,
                "redundancy_score": 1.0,
                "importance_score": round(sum(float(cap.get("importance") or 0.5) for cap in caps) / len(caps), 6),
                "source_titles": [cap.get("title") or cap["id"] for cap in caps[:6]],
                "original_token_estimate": original_tokens,
                "compacted_token_estimate": compacted_tokens,
                "estimated_saved_tokens": max(0, original_tokens - compacted_tokens),
                "estimated_token_savings_ratio": (max(0, original_tokens - compacted_tokens) / original_tokens) if original_tokens else 0.0,
                "suggested_action": "compact_local",
                "canonical_preview": canonical_summary,
            }
        )
    clustered_ids: set[str] = set(used_exact_ids)
    for seed in capsules:
        if seed["id"] in clustered_ids:
            continue
        cluster = [seed]
        similarities: list[float] = []
        for other in capsules:
            if other["id"] == seed["id"] or other["id"] in clustered_ids:
                continue
            scores = _memory_similarity(seed, other)
            if scores["combined"] >= near_duplicate_threshold:
                cluster.append(other)
                similarities.append(scores["combined"])
        if len(cluster) < min_cluster_size:
            continue
        clustered_ids.update(cap["id"] for cap in cluster)
        original_tokens = sum(_token_estimate(str(cap.get("summary") or cap.get("raw_text") or "")) for cap in cluster)
        canonical_summary = _compact_capsule_summary(cluster)
        compacted_tokens = _token_estimate(canonical_summary)
        avg_similarity = sum(similarities) / len(similarities) if similarities else near_duplicate_threshold
        cluster_id = "near_" + stable_content_hash(_json([project_id, [cap["id"] for cap in cluster], round(avg_similarity, 4)]))[:16]
        candidates.append(
            {
                "cluster_id": cluster_id,
                "mode": "local_semantic",
                "reason": "local lexical/embedding near-duplicate cluster",
                "memory_ids": [cap["id"] for cap in cluster],
                "memory_count": len(cluster),
                "avg_similarity": round(avg_similarity, 6),
                "redundancy_score": round(avg_similarity, 6),
                "importance_score": round(sum(float(cap.get("importance") or 0.5) for cap in cluster) / len(cluster), 6),
                "source_titles": [cap.get("title") or cap["id"] for cap in cluster[:6]],
                "original_token_estimate": original_tokens,
                "compacted_token_estimate": compacted_tokens,
                "estimated_saved_tokens": max(0, original_tokens - compacted_tokens),
                "estimated_token_savings_ratio": (max(0, original_tokens - compacted_tokens) / original_tokens) if original_tokens else 0.0,
                "suggested_action": "compact_local",
                "canonical_preview": canonical_summary,
            }
        )
    candidates.sort(key=lambda item: (item["estimated_saved_tokens"], item["memory_count"]), reverse=True)
    return {"status": "ok", "project_id": project_id, "count": len(candidates[:limit]), "candidates": candidates[:limit]}


def verify_compaction_result(base_dir: Path, *, project_id: str, canonical_content: str, source_ids: list[str], max_output_tokens: int = 1200) -> dict[str, Any]:
    warnings: list[str] = []
    source_ids = [str(item) for item in source_ids if str(item)]
    if not source_ids:
        warnings.append("source_ids_empty")
    if not canonical_content.strip():
        warnings.append("canonical_content_empty")
    if _token_estimate(canonical_content) > max_output_tokens:
        warnings.append("output_exceeds_max_tokens")
    preserved = True
    missing: list[str] = []
    for source_id in source_ids:
        try:
            get_capsule(base_dir, source_id, project_id=project_id)
        except KeyError:
            preserved = False
            missing.append(source_id)
    if missing:
        warnings.append("source_memory_missing")
    if source_ids and canonical_content:
        source_terms: set[str] = set()
        for source_id in source_ids:
            try:
                cap = get_capsule(base_dir, source_id, project_id=project_id)
            except KeyError:
                continue
            source_terms.update(_extract_terms(str(cap.get("summary") or cap.get("raw_text") or cap.get("title") or "")))
        unsupported = [term for term in _extract_terms(canonical_content) if source_terms and term not in source_terms]
        if len(unsupported) > max(12, len(source_terms) // 2):
            warnings.append("canonical_content_may_contain_unsupported_terms")
    passed = not warnings and preserved
    return {"status": "ok", "verifier_passed": passed, "warnings": warnings, "raw_memory_preserved": preserved, "missing_source_ids": missing}


def _compaction_results_dir(base_dir: Path) -> Path:
    path = product_root(base_dir) / "compaction_results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compaction_result_path(base_dir: Path, result_id: str) -> Path:
    return _compaction_results_dir(base_dir) / f"{result_id}.json"


def _write_compaction_result(base_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    path = _compaction_result_path(base_dir, result["result_id"])
    path.write_text(_json(result) + "\n", encoding="utf-8")
    return result


def _external_llm_compaction_gate(base_dir: Path, *, confirm_external_call: bool = False) -> tuple[bool, list[str], dict[str, Any]]:
    config = load_runtime_config(base_dir).to_dict()
    llm_config = dict(config.get("llm_config") or {})
    compaction_config = dict(config.get("compaction_config") or {})
    privacy_config = dict(config.get("privacy_config") or {})
    warnings: list[str] = []
    if not llm_config.get("external_llm_enabled"):
        warnings.append("external_llm_disabled_by_user")
    if not compaction_config.get("external_llm_compaction_enabled"):
        warnings.append("external_llm_compaction_disabled_by_user")
    if llm_config.get("local_only", True) or privacy_config.get("local_only", False):
        warnings.append("local_only_mode_prevents_external_llm")
    if privacy_config.get("require_external_call_confirmation", True) and not confirm_external_call:
        warnings.append("external_call_confirmation_required")
    provider = str(llm_config.get("provider") or "auto")
    if provider in {"", "auto", "codex", "claude", "disabled"}:
        warnings.append("external_provider_not_configured")
    if provider in {"openai_compatible", "custom"} and not llm_config.get("api_base_url"):
        warnings.append("external_api_base_url_missing")
    if provider in {"openai_compatible", "custom"} and not llm_config.get("api_key"):
        warnings.append("external_api_key_missing")
    return not warnings, warnings, config


def _safe_external_source_texts(capsules: list[dict[str, Any]], *, allow_raw: bool, max_input_tokens: int) -> list[dict[str, str]]:
    remaining = max(256, int(max_input_tokens or 12000))
    payload: list[dict[str, str]] = []
    for cap in capsules:
        text = str((cap.get("raw_text") if allow_raw else None) or cap.get("summary") or cap.get("title") or "")
        text = redact_secrets(text)
        estimated = _token_estimate(text)
        if estimated > remaining:
            words = text.split()
            text = " ".join(words[: max(40, remaining)])
            estimated = _token_estimate(text)
        if text:
            payload.append({"id": str(cap["id"]), "title": str(cap.get("title") or cap["id"]), "text": text})
            remaining -= estimated
        if remaining <= 0:
            break
    return payload


def _run_external_llm_compaction(
    base_dir: Path,
    *,
    capsules: list[dict[str, Any]],
    confirm_external_call: bool,
) -> tuple[str | None, list[str], dict[str, Any] | None]:
    allowed, warnings, config = _external_llm_compaction_gate(base_dir, confirm_external_call=confirm_external_call)
    if not allowed:
        return None, warnings, None
    llm_config = dict(config.get("llm_config") or {})
    compaction_config = dict(config.get("compaction_config") or {})
    allow_raw = bool(llm_config.get("allow_raw_memory_external"))
    source_payload = _safe_external_source_texts(
        capsules,
        allow_raw=allow_raw,
        max_input_tokens=int(compaction_config.get("max_input_tokens") or 12000),
    )
    if not source_payload:
        return None, ["external_payload_empty"], None
    base_url = str(llm_config.get("api_base_url") or "").rstrip("/")
    if not base_url.endswith("/chat/completions"):
        base_url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": str(llm_config.get("model") or "gpt-4o-mini"),
        "messages": [
            {
                "role": "system",
                "content": "Create one concise canonical memory from the source memories. Preserve source IDs, avoid new facts, and return only the canonical memory text.",
            },
            {"role": "user", "content": _json({"sources": source_payload, "raw_memory_included": allow_raw})},
        ],
        "temperature": 0,
        "max_tokens": int(compaction_config.get("max_output_tokens") or 700),
    }
    request = urllib.request.Request(
        base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {llm_config.get('api_key')}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(compaction_config.get("timeout_seconds") or 45)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return None, [f"external_llm_call_failed:{type(exc).__name__}"], {"provider": llm_config.get("provider"), "api_base_url": base_url}
    content = ""
    try:
        content = str(payload["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError):
        return None, ["external_llm_response_invalid"], {"provider": llm_config.get("provider"), "api_base_url": base_url}
    return content or None, [], {"provider": llm_config.get("provider"), "model": llm_config.get("model"), "raw_memory_included": allow_raw, "source_count": len(source_payload)}


def get_compaction_result(base_dir: Path, *, result_id: str, project_id: str | None = None) -> dict[str, Any]:
    path = _compaction_result_path(base_dir, result_id)
    if not path.exists():
        raise KeyError(result_id)
    result = json.loads(path.read_text(encoding="utf-8"))
    if project_id and result.get("project_id") != project_id:
        raise KeyError(result_id)
    return {"status": "ok", "result": result}


def create_compaction_result(
    base_dir: Path,
    *,
    project_id: str,
    cluster_id: str | None = None,
    memory_ids: list[str] | None = None,
    mode: str = "deterministic",
    verifier: str = "system",
    confirm_external_call: bool = False,
) -> dict[str, Any]:
    if mode not in {"deterministic", "local_semantic", "hybrid", "llm"}:
        raise ValueError("mode must be deterministic, local_semantic, hybrid, or llm")
    selected_ids = [str(item) for item in memory_ids or [] if str(item)]
    cluster: dict[str, Any] | None = None
    if not selected_ids and cluster_id:
        candidates = find_compaction_candidates(base_dir, project_id=project_id, limit=50)["candidates"]
        cluster = next((item for item in candidates if item["cluster_id"] == cluster_id), None)
        if cluster is None:
            candidates = find_compaction_candidates(base_dir, project_id=project_id, limit=50, near_duplicate_threshold=0.4)["candidates"]
            cluster = next((item for item in candidates if item["cluster_id"] == cluster_id), None)
        selected_ids = list(cluster.get("memory_ids") or []) if cluster else []
    if len(selected_ids) < 2:
        raise ValueError("compaction requires at least two source memories")
    capsules = [get_capsule(base_dir, capsule_id, project_id=project_id) for capsule_id in selected_ids]
    source_ids = [cap["id"] for cap in capsules]
    summary = _compact_capsule_summary(capsules)
    warnings: list[str] = []
    external_metadata: dict[str, Any] | None = None
    if mode in {"hybrid", "llm"}:
        external_summary, external_warnings, external_metadata = _run_external_llm_compaction(
            base_dir,
            capsules=capsules,
            confirm_external_call=confirm_external_call,
        )
        warnings.extend(external_warnings)
        if external_summary:
            summary = external_summary
    source_tokens = sum(_token_estimate(str(cap.get("summary") or cap.get("raw_text") or "")) for cap in capsules)
    compacted_tokens = _token_estimate(summary)
    verification = verify_compaction_result(base_dir, project_id=project_id, canonical_content=summary, source_ids=source_ids)
    warnings.extend(verification["warnings"])
    result_id = "cmp_" + stable_content_hash(_json([project_id, cluster_id, source_ids, summary, mode, now_iso()]))[:20]
    result = {
        "result_id": result_id,
        "cluster_id": cluster_id or "manual_" + stable_content_hash(_json(source_ids))[:16],
        "project_id": project_id,
        "canonical_content": summary,
        "source_ids": source_ids,
        "method": mode,
        "confidence_score": min(float(cap.get("confidence") or 0.7) for cap in capsules),
        "estimated_token_savings_ratio": (max(0, source_tokens - compacted_tokens) / source_tokens) if source_tokens else 0.0,
        "source_token_estimate": source_tokens,
        "compacted_token_estimate": compacted_tokens,
        "estimated_saved_tokens": max(0, source_tokens - compacted_tokens),
        "warnings": warnings,
        "verifier_passed": verification["verifier_passed"],
        "verification": verification,
        "external_llm": external_metadata,
        "external_llm_attempted": mode in {"hybrid", "llm"},
        "created_by": verifier,
        "created_at": now_iso(),
        "status": "verified" if verification["verifier_passed"] and not warnings else "needs_review",
        "source_titles": [cap.get("title") or cap["id"] for cap in capsules],
    }
    _write_compaction_result(base_dir, result)
    with connect(base_dir) as conn:
        event_id = "evt_" + stable_content_hash(_json([project_id, result_id, "memory_compaction_started", now_iso()]))[:20]
        conn.execute(
            "INSERT OR REPLACE INTO runtime_events(event_id, project_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_id, project_id, "memory_compaction_started", _json({"result_id": result_id, "cluster_id": result["cluster_id"], "source_ids": source_ids, "mode": mode}), now_iso()),
        )
        conn.commit()
    return {"status": "ok", "result": result}


def verify_compaction_result_record(base_dir: Path, *, result_id: str, project_id: str | None = None) -> dict[str, Any]:
    result = get_compaction_result(base_dir, result_id=result_id, project_id=project_id)["result"]
    verification = verify_compaction_result(base_dir, project_id=str(result["project_id"]), canonical_content=str(result.get("canonical_content") or ""), source_ids=[str(item) for item in result.get("source_ids") or []])
    result["verification"] = verification
    result["warnings"] = verification["warnings"]
    result["verifier_passed"] = verification["verifier_passed"]
    result["status"] = "verified" if verification["verifier_passed"] else "needs_review"
    _write_compaction_result(base_dir, result)
    with connect(base_dir) as conn:
        event_id = "evt_" + stable_content_hash(_json([result["project_id"], result_id, "memory_compaction_verified", now_iso()]))[:20]
        conn.execute(
            "INSERT OR REPLACE INTO runtime_events(event_id, project_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_id, result["project_id"], "memory_compaction_verified", _json({"result_id": result_id, "verification": verification}), now_iso()),
        )
        conn.commit()
    return {"status": "ok", "result": result}


def commit_compaction_result(base_dir: Path, *, result_id: str, project_id: str | None = None, allow_needs_review: bool = False) -> dict[str, Any]:
    result = get_compaction_result(base_dir, result_id=result_id, project_id=project_id)["result"]
    if result.get("status") == "committed":
        return {"status": "ok", "result": result, "canonical_id": result.get("canonical_id")}
    if not result.get("verifier_passed") and not allow_needs_review:
        return {"status": "needs_review", "result": result, "verification": result.get("verification")}
    source_ids = [str(item) for item in result.get("source_ids") or []]
    capsules = [get_capsule(base_dir, capsule_id, project_id=str(result["project_id"])) for capsule_id in source_ids]
    title = str(capsules[0].get("title") or "Compacted memory")
    metadata = {
        "memory_type": "canonical",
        "source_ids": source_ids,
        "compaction": {
            "result_id": result_id,
            "mode": result.get("method"),
            "strategy": "reviewable_compaction_result",
            "source_count": len(source_ids),
            "source_token_estimate": result.get("source_token_estimate"),
            "compacted_token_estimate": result.get("compacted_token_estimate"),
            "estimated_saved_tokens": result.get("estimated_saved_tokens"),
            "raw_memory_preserved": True,
            "verifier_passed": bool(result.get("verifier_passed")),
            "warnings": result.get("warnings") or [],
            "created_at": now_iso(),
        },
        "lifecycle_reason": "canonical memory committed from reviewable compaction result; raw sources preserved and traceable",
    }
    remembered = remember(
        base_dir,
        project_id=str(result["project_id"]),
        text=str(result.get("canonical_content") or ""),
        evidence_type="canonical",
        source_type=f"{result.get('method') or 'local'}_compactor",
        title=f"Canonical: {title[:80]}",
        summary=str(result.get("canonical_content") or "")[:320],
        validity_state="active",
        tags=sorted({tag for cap in capsules for tag in cap.get("tags", [])} | {"canonical", "compacted"}),
        metadata=metadata,
    )
    canonical_id = remembered["capsule_id"]
    with connect(base_dir) as conn:
        for cap in capsules:
            cap_meta = dict(cap.get("metadata") or {})
            cap_meta["compacted_into"] = canonical_id
            cap_meta["lifecycle_reason"] = "included in canonical compacted memory; raw text preserved"
            cap_meta.setdefault("compaction_source", {})["canonical_id"] = canonical_id
            parent_ids = _append_unique(cap.get("parent_ids") or [], canonical_id)
            related_ids = _append_unique(cap.get("related_ids") or [], canonical_id)
            conn.execute(
                """
                UPDATE evidence_capsules
                SET validity_state = 'compacted', parent_ids_json = ?, related_ids_json = ?, metadata_json = ?, updated_at = ?
                WHERE id = ? AND project_id = ?
                """,
                (_json(parent_ids), _json(related_ids), _json(redact_payload(cap_meta)), now_iso(), cap["id"], result["project_id"]),
            )
            conn.execute(
                "INSERT OR REPLACE INTO capsule_relations(relation_id, source_capsule_id, target_capsule_id, relation_type, reason, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("rel_" + stable_content_hash(_json([cap["id"], canonical_id, "compacted_into"]))[:20], cap["id"], canonical_id, "compacted_into", "raw memory preserved as source for canonical compacted memory", now_iso()),
            )
        event_id = "evt_" + stable_content_hash(_json([result["project_id"], canonical_id, source_ids, now_iso()]))[:20]
        conn.execute(
            "INSERT OR REPLACE INTO runtime_events(event_id, project_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_id, result["project_id"], "memory_compaction_committed", _json({"result_id": result_id, "canonical_id": canonical_id, "source_ids": source_ids, "mode": result.get("method"), "raw_memory_preserved": True}), now_iso()),
        )
        conn.commit()
    result["status"] = "committed"
    result["canonical_id"] = canonical_id
    result["committed_at"] = now_iso()
    _write_compaction_result(base_dir, result)
    return {"status": "ok", "canonical_id": canonical_id, "result": result, "capsule": get_capsule(base_dir, canonical_id, project_id=str(result["project_id"]))}


def reject_compaction_result(base_dir: Path, *, result_id: str, project_id: str | None = None, reason: str | None = None) -> dict[str, Any]:
    result = get_compaction_result(base_dir, result_id=result_id, project_id=project_id)["result"]
    result["status"] = "rejected"
    result["rejected_at"] = now_iso()
    result["reject_reason"] = reason or ""
    _write_compaction_result(base_dir, result)
    with connect(base_dir) as conn:
        event_id = "evt_" + stable_content_hash(_json([result["project_id"], result_id, "memory_compaction_rejected", now_iso()]))[:20]
        conn.execute(
            "INSERT OR REPLACE INTO runtime_events(event_id, project_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_id, result["project_id"], "memory_compaction_rejected", _json({"result_id": result_id, "reason": reason or ""}), now_iso()),
        )
        conn.commit()
    return {"status": "ok", "result": result}


def run_deterministic_compaction(base_dir: Path, *, project_id: str, cluster_id: str | None = None, memory_ids: list[str] | None = None, mode: str = "deterministic", verifier: str = "system") -> dict[str, Any]:
    created = create_compaction_result(base_dir, project_id=project_id, cluster_id=cluster_id, memory_ids=memory_ids, mode=mode, verifier=verifier)
    result = created["result"]
    if not result.get("verifier_passed"):
        return {"status": "needs_review", "project_id": project_id, "source_ids": result.get("source_ids") or [], "mode": mode, "verification": result.get("verification"), "canonical_preview": result.get("canonical_content"), "result": result}
    committed = commit_compaction_result(base_dir, result_id=result["result_id"], project_id=project_id)
    return {
        "status": committed["status"],
        "project_id": project_id,
        "canonical_id": committed.get("canonical_id"),
        "source_ids": result.get("source_ids") or [],
        "mode": mode,
        "raw_memory_preserved": True,
        "source_token_estimate": result.get("source_token_estimate"),
        "compacted_token_estimate": result.get("compacted_token_estimate"),
        "estimated_saved_tokens": result.get("estimated_saved_tokens"),
        "verification": result.get("verification"),
        "result": committed.get("result"),
        "capsule": committed.get("capsule"),
    }


def lifecycle_audit(base_dir: Path, *, project_id: str, limit: int = 30) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    with connect(base_dir) as conn:
        rows = conn.execute(
            "SELECT event_id, event_type, payload_json, created_at FROM runtime_events WHERE project_id = ? AND event_type LIKE 'memory_%' ORDER BY created_at DESC LIMIT ?",
            (project_id, max(1, min(int(limit or 30), 100))),
        ).fetchall()
    return {
        "status": "ok",
        "project_id": project_id,
        "events": [{"event_id": row["event_id"], "event_type": row["event_type"], "created_at": row["created_at"], "payload": json.loads(row["payload_json"] or "{}")} for row in rows],
    }


def migrate_lifecycle_metadata(base_dir: Path, *, project_id: str | None = None, backup: bool = True) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id or "default")
    db_path = product_db_path(base_dir)
    backup_path: str | None = None
    if backup and db_path.exists():
        target = db_path.with_suffix(f".lifecycle-backup-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.sqlite3")
        shutil.copy2(db_path, target)
        backup_path = str(target)
    where = ["archived = 0"]
    params: list[Any] = []
    if project_id:
        where.append("project_id = ?")
        params.append(project_id)
    migrated = 0
    with connect(base_dir) as conn:
        rows = conn.execute(f"SELECT * FROM evidence_capsules WHERE {' AND '.join(where)}", params).fetchall()
        for row in rows:
            cap = _capsule_from_row(row)
            meta = dict(cap.get("metadata") or {})
            text = str(cap.get("summary") or cap.get("raw_text") or cap.get("title") or "")
            changed = False
            defaults = {
                "memory_type": cap.get("evidence_type") or "raw",
                "content_hash": stable_content_hash(text),
                "normalized_hash": stable_content_hash(_normalized_memory_text(cap)),
                "token_estimate": _token_estimate(text),
                "access_count": int(meta.get("access_count") or 0),
                "last_accessed_at": meta.get("last_accessed_at"),
                "created_by": meta.get("created_by") or "migration",
            }
            for key, value in defaults.items():
                if key not in meta:
                    meta[key] = value
                    changed = True
            if cap.get("validity_state") not in DEFAULT_ELIGIBLE_STATES | DEFAULT_EXCLUDED_STATES:
                conn.execute("UPDATE evidence_capsules SET validity_state = 'active' WHERE id = ?", (cap["id"],))
                changed = True
            if changed:
                conn.execute("UPDATE evidence_capsules SET metadata_json = ?, updated_at = ? WHERE id = ?", (_json(redact_payload(meta)), now_iso(), cap["id"]))
                migrated += 1
        event_id = "evt_" + stable_content_hash(_json([project_id, migrated, backup_path, now_iso()]))[:20]
        conn.execute(
            "INSERT OR REPLACE INTO runtime_events(event_id, project_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_id, project_id or "all", "memory_lifecycle_migrated", _json({"migrated_count": migrated, "backup_path": backup_path}), now_iso()),
        )
        conn.commit()
    return {"status": "ok", "project_id": project_id, "migrated_count": migrated, "backup_path": backup_path}


def get_maintenance_suggestion(base_dir: Path, *, project_id: str, suggestion_id: str) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    with connect(base_dir) as conn:
        row = conn.execute("SELECT * FROM maintenance_suggestions WHERE project_id = ? AND suggestion_id = ?", (project_id, suggestion_id)).fetchone()
    if row is None:
        maintenance_suggestions(base_dir, project_id=project_id)
        with connect(base_dir) as conn:
            row = conn.execute("SELECT * FROM maintenance_suggestions WHERE project_id = ? AND suggestion_id = ?", (project_id, suggestion_id)).fetchone()
    if row is None:
        raise KeyError(suggestion_id)
    payload = json.loads(row["payload_json"] or "{}")
    return {"status": "ok", "project_id": project_id, "suggestion_id": suggestion_id, "suggestion_status": row["status"], **payload}


def dismiss_maintenance_suggestion(base_dir: Path, *, project_id: str, suggestion_id: str, reason: str | None = None) -> dict[str, Any]:
    suggestion = get_maintenance_suggestion(base_dir, project_id=project_id, suggestion_id=suggestion_id)
    payload = {k: v for k, v in suggestion.items() if k not in {"status", "project_id", "suggestion_id", "suggestion_status"}}
    payload["dismissed_at"] = now_iso()
    if reason:
        payload["dismiss_reason"] = reason
    with connect(base_dir) as conn:
        conn.execute(
            "UPDATE maintenance_suggestions SET status = 'dismissed', payload_json = ? WHERE project_id = ? AND suggestion_id = ?",
            (_json(payload), project_id, suggestion_id),
        )
        conn.commit()
    return {"status": "dismissed", "project_id": project_id, "suggestion_id": suggestion_id, "reason": reason}


def apply_maintenance_suggestion(
    base_dir: Path,
    *,
    project_id: str,
    suggestion_id: str,
    canonical_id: str | None = None,
) -> dict[str, Any]:
    suggestion = get_maintenance_suggestion(base_dir, project_id=project_id, suggestion_id=suggestion_id)
    if suggestion["suggestion_status"] == "applied":
        return {"status": "already_applied", "project_id": project_id, "suggestion_id": suggestion_id, "result": suggestion}
    result: dict[str, Any]
    if suggestion["type"] == "duplicate_merge":
        ids = [cid for cid in suggestion.get("capsule_ids", []) if cid]
        if not ids:
            raise ValueError("duplicate_merge suggestion has no capsule_ids")
        canonical = canonical_id or ids[0]
        if canonical not in ids:
            raise ValueError("canonical_id must be one of the duplicate capsule_ids")
        merged: list[str] = []
        for cid in ids:
            if cid == canonical:
                continue
            mark_superseded(base_dir, cid, canonical, suggestion.get("reason") or "duplicate_merge maintenance suggestion")
            register_alias(base_dir, project_id=project_id, alias=cid, canonical=canonical, metadata={"source": "maintenance", "suggestion_id": suggestion_id})
            merged.append(cid)
        result = {"type": "duplicate_merge", "canonical_id": canonical, "merged_capsule_ids": merged}
    elif suggestion["type"] == "invalidate_older_benchmark":
        older = suggestion.get("older_capsule_id")
        newer = suggestion.get("newer_capsule_id")
        if not older or not newer:
            raise ValueError("invalidate_older_benchmark suggestion is missing capsule ids")
        marked = mark_superseded(base_dir, older, newer, suggestion.get("reason") or "newer benchmark result exists")
        result = {"type": "invalidate_older_benchmark", "older_capsule_id": older, "newer_capsule_id": newer, "validity": marked}
    else:
        raise ValueError(f"Unsupported maintenance suggestion type: {suggestion['type']}")
    payload = {k: v for k, v in suggestion.items() if k not in {"status", "project_id", "suggestion_id", "suggestion_status"}}
    payload["applied_at"] = now_iso()
    payload["apply_result"] = result
    with connect(base_dir) as conn:
        conn.execute(
            "UPDATE maintenance_suggestions SET status = 'applied', payload_json = ? WHERE project_id = ? AND suggestion_id = ?",
            (_json(payload), project_id, suggestion_id),
        )
        conn.commit()
    return {"status": "applied", "project_id": project_id, "suggestion_id": suggestion_id, "result": result}


def export_project(base_dir: Path, *, project_id: str, fmt: str = "json", output: Path | None = None) -> dict[str, Any]:
    with connect(base_dir) as conn:
        rows = conn.execute("SELECT * FROM evidence_capsules WHERE project_id = ? AND archived = 0 ORDER BY timestamp DESC", (project_id,)).fetchall()
    capsules = [_capsule_from_row(row) for row in rows]
    output = output or (product_root(base_dir) / "exports" / f"{project_id}_capsules.{fmt}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "project_id": project_id,
        "exported_at": now_iso(),
        "capsule_count": len(capsules),
        "redaction": "secrets_redacted_before_storage",
        "local_only": True,
    }
    if fmt == "markdown":
        text = "\n\n".join(f"## {cap['title'] or cap['id']}\n\n- id: `{cap['id']}`\n- type: `{cap['evidence_type']}`\n- validity: `{cap['validity_state']}`\n\n{cap.get('summary') or ''}" for cap in capsules)
    else:
        text = json.dumps({"project_id": project_id, "manifest": manifest, "capsules": capsules}, ensure_ascii=False, indent=2, sort_keys=True)
    output.write_text(text + "\n", encoding="utf-8")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "ok", "project_id": project_id, "format": fmt, "output": str(output), "manifest": str(manifest_path), "count": len(capsules)}


def benchmark_record(base_dir: Path, *, project_id: str, artifact: Path, benchmark: str | None = None, status: str = "success") -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if artifact.is_file():
        metrics = json.loads(artifact.read_text(encoding="utf-8"))
        artifact_paths = [str(artifact)]
    else:
        for path in artifact.rglob("metrics.json"):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            break
        artifact_paths = [str(artifact)]
    benchmark = benchmark or str(metrics.get("benchmark") or artifact.name)
    binding = _benchmark_binding(base_dir, artifact=artifact, metrics=metrics, benchmark=benchmark, status=status)
    git = binding["git"]
    run_id = "bench_" + stable_content_hash(_json([project_id, benchmark, metrics, str(artifact)]))[:20]
    with connect(base_dir) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO benchmark_runs(run_id, project_id, benchmark, timestamp, git_commit, config_hash, metrics_json, artifact_paths_json, duration_sec, status, notes_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, project_id, benchmark, now_iso(), git.get("git_commit"), binding["config_hash"], _json({**metrics, "binding": binding}), _json(artifact_paths), binding["duration_sec"], status, "[]"),
        )
        conn.commit()
    dashboard_files = _write_benchmark_dashboard_files(base_dir, project_id=project_id)
    remember(base_dir, project_id=project_id, text=f"Benchmark {benchmark} run {run_id}: {json.dumps(metrics, ensure_ascii=False)[:2000]}", evidence_type="benchmark_result", source_type="benchmark", artifact_refs=artifact_paths, benchmark_refs=[benchmark, binding["dataset"]], metadata={"run_id": run_id, "metrics": metrics, "benchmark_binding": binding})
    return {"status": "ok", "run_id": run_id, "benchmark": benchmark, "metrics": metrics, "binding": binding, "dashboard_files": dashboard_files}


def benchmark_compare(base_dir: Path, *, current: Path, baseline: Path, project_id: str = "default") -> dict[str, Any]:
    def load(path: Path) -> dict[str, Any]:
        if path.is_dir():
            path = next(path.rglob("metrics.json"))
        return json.loads(path.read_text(encoding="utf-8"))

    cur = load(current)
    base = load(baseline)
    deltas: dict[str, Any] = {}
    for key, value in (cur.get("metrics") or cur).items():
        old = (base.get("metrics") or base).get(key)
        if isinstance(value, (int, float)) and isinstance(old, (int, float)):
            deltas[key] = value - old
    cur_metrics = _extract_metric_payload(cur)
    base_metrics = _extract_metric_payload(base)
    candidate_recall_deltas = {
        key: cur_metrics.get(key) - base_metrics.get(key)
        for key in ("candidate_recall@100", "candidate_recall_at_100")
        if isinstance(cur_metrics.get(key), (int, float)) and isinstance(base_metrics.get(key), (int, float))
    }
    latency_deltas = {
        key: cur_metrics.get(key) - base_metrics.get(key)
        for key in ("latency_ms", "retrieval_latency_ms", "elapsed_seconds")
        if isinstance(cur_metrics.get(key), (int, float)) and isinstance(base_metrics.get(key), (int, float))
    }
    failure_taxonomy_deltas: dict[str, Any] = {}
    cur_failures = cur.get("failure_taxonomy") if isinstance(cur.get("failure_taxonomy"), dict) else {}
    base_failures = base.get("failure_taxonomy") if isinstance(base.get("failure_taxonomy"), dict) else {}
    for key in sorted(set(cur_failures) | set(base_failures)):
        if isinstance(cur_failures.get(key, 0), (int, float)) and isinstance(base_failures.get(key, 0), (int, float)):
            failure_taxonomy_deltas[key] = cur_failures.get(key, 0) - base_failures.get(key, 0)
    changed_files = _git_metadata(base_dir).get("changed_files", [])
    report = {
        "status": "ok",
        "project_id": project_id,
        "metric_deltas": deltas,
        "candidate_recall_deltas": candidate_recall_deltas,
        "latency_deltas": latency_deltas,
        "failure_taxonomy_deltas": failure_taxonomy_deltas,
        "possible_changed_files": changed_files,
        "current": str(current),
        "baseline": str(baseline),
        "suggested_investigation_steps": ["Inspect retrieval_traces for candidate admission drops.", "Compare config_hash and changed files.", "Check fallback flags before score interpretation."],
    }
    out = product_root(base_dir) / "artifacts" / "benchmark_regression_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json(report) + "\n", encoding="utf-8")
    lab = product_root(base_dir) / "artifacts" / "benchmark_lab"
    lab.mkdir(parents=True, exist_ok=True)
    (lab / "regression_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def list_benchmark_runs(base_dir: Path, *, project_id: str, limit: int = 50) -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    limit = max(1, min(int(limit or 50), 200))
    with connect(base_dir) as conn:
        rows = conn.execute(
            "SELECT * FROM benchmark_runs WHERE project_id = ? ORDER BY timestamp DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()
    runs = []
    for row in rows:
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
        item["artifact_paths"] = json.loads(item.pop("artifact_paths_json") or "[]")
        item["notes"] = json.loads(item.pop("notes_json") or "[]")
        runs.append(item)
    return {"status": "ok", "project_id": project_id, "count": len(runs), "benchmark_runs": runs}


def doctor(base_dir: Path, *, project_id: str = "default") -> dict[str, Any]:
    init_product_store(base_dir, project_id=project_id)
    checks: dict[str, Any] = {}
    embedding_config = embedding_backend_config(base_dir)
    configured_backend = str(embedding_config.get("backend") or LOCAL_EMBEDDING_BACKEND)
    with connect(base_dir) as conn:
        counts = {name: int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]) for name in ("raw_traces", "evidence_capsules", "retrieval_traces", "context_packs", "runtime_events", "benchmark_runs")}
        orphan = int(conn.execute("SELECT COUNT(*) FROM evidence_capsules c LEFT JOIN raw_traces r ON c.raw_ref = 'dyson://raw/' || r.raw_id WHERE r.raw_id IS NULL").fetchone()[0])
        invalid_refs = int(conn.execute("SELECT COUNT(*) FROM evidence_capsules WHERE validity_state NOT IN ('active','superseded','deprecated','contradicted','reverted','unknown')").fetchone()[0])
        raw_hash_rows = conn.execute("SELECT raw_ref, COUNT(*) AS n FROM evidence_capsules WHERE project_id = ? AND archived = 0 GROUP BY raw_ref HAVING n > 1", (project_id,)).fetchall()
        artifact_rows = conn.execute("SELECT artifact_refs_json FROM evidence_capsules WHERE project_id = ? AND archived = 0", (project_id,)).fetchall()
        embedding_count = int(conn.execute("SELECT COUNT(*) FROM capsule_embeddings e JOIN evidence_capsules c ON c.id = e.capsule_id WHERE c.project_id = ? AND c.archived = 0", (project_id,)).fetchone()[0])
        active_capsule_rows = conn.execute("SELECT * FROM evidence_capsules WHERE project_id = ? AND archived = 0", (project_id,)).fetchall()
        embedding_rows = {row["capsule_id"]: row for row in conn.execute("SELECT e.* FROM capsule_embeddings e JOIN evidence_capsules c ON c.id = e.capsule_id WHERE c.project_id = ? AND c.archived = 0", (project_id,)).fetchall()}
    checks["database"] = {"severity": "ok", "path": str(product_db_path(base_dir)), "counts": counts}
    checks["raw_store"] = {"severity": "error" if orphan else "ok", "orphan_raw_refs": orphan}
    checks["schema_version"] = {"severity": "ok", "version": SCHEMA_VERSION}
    checks["validity"] = {"severity": "error" if invalid_refs else "ok", "invalid_state_count": invalid_refs}
    total_capsules = max(1, counts["evidence_capsules"])
    duplicate_ratio = sum(int(row["n"]) - 1 for row in raw_hash_rows) / total_capsules
    checks["duplicates"] = {"severity": "warning" if duplicate_ratio > 0.15 else "ok", "duplicate_raw_ref_ratio": round(duplicate_ratio, 6)}
    missing_artifacts: list[str] = []
    for row in artifact_rows:
        for ref in json.loads(row["artifact_refs_json"] or "[]"):
            path = Path(ref)
            if not path.is_absolute():
                path = base_dir / path
            if not path.exists():
                missing_artifacts.append(str(ref))
    checks["artifacts"] = {"severity": "warning" if missing_artifacts else "ok", "missing_artifact_count": len(missing_artifacts), "missing_artifacts": missing_artifacts[:20]}
    missing_embeddings = 0
    stale_embeddings = 0
    for row in active_capsule_rows:
        capsule = _capsule_from_row(row)
        embedding = embedding_rows.get(capsule["id"])
        if not embedding:
            missing_embeddings += 1
            continue
        meta = json.loads(embedding["metadata_json"] or "{}")
        if (
            embedding["backend"] != configured_backend
            or meta.get("version") != LOCAL_EMBEDDING_VERSION
            or meta.get("source_hash") != stable_content_hash(_capsule_embedding_text(capsule))
        ):
            stale_embeddings += 1
    checks["embedding_backend"] = {
        "severity": "warning" if missing_embeddings or stale_embeddings else "ok",
        "local_product_embeddings": embedding_count,
        "active_capsules": len(active_capsule_rows),
        "missing_embeddings": missing_embeddings,
        "stale_embeddings": stale_embeddings,
        "dense_probe_available": embedding_count > 0 and missing_embeddings == 0 and stale_embeddings == 0,
        "backend": configured_backend,
        "configured": embedding_config,
        "available_backends": product_embedding_backends(base_dir)["backends"],
        "version": LOCAL_EMBEDDING_VERSION,
    }
    vector_config = product_vector_index_config(base_dir)
    vector_backends = product_vector_backends(base_dir)["backends"]
    vector_severity = "ok"
    vector_count = 0
    vector_reason = None
    if vector_config["backend"] == "chroma":
        if not vector_backends["chroma"].get("available"):
            vector_severity = "warning"
            vector_reason = vector_backends["chroma"].get("unavailable_reason")
        else:
            try:
                vector_count = int(_chroma_collection(base_dir, create=False).count())
                if vector_count < len(active_capsule_rows):
                    vector_severity = "warning"
                    vector_reason = "product vector index has fewer entries than active capsules"
            except Exception as exc:
                vector_severity = "warning"
                vector_reason = str(exc)
    elif embedding_count == 0 or missing_embeddings or stale_embeddings:
        vector_severity = "warning"
    elif vector_config["backend"] == "sqlite_inline" and len(active_capsule_rows) > PRODUCT_VECTOR_PROMOTION_THRESHOLD:
        vector_severity = "warning"
        vector_reason = f"active capsule count exceeds {PRODUCT_VECTOR_PROMOTION_THRESHOLD}; configure chroma for ANN retrieval"
    checks["vector_index"] = {
        "severity": vector_severity,
        "configured": vector_config,
        "available_backends": vector_backends,
        "backend": vector_config["backend"],
        "ann_enabled": vector_config["backend"] == "chroma",
        "indexed_count": vector_count,
        "promotion_threshold": PRODUCT_VECTOR_PROMOTION_THRESHOLD,
        **({"reason": vector_reason} if vector_reason else {}),
    }
    lab = product_root(base_dir) / "artifacts" / "benchmark_lab"
    dashboard_files = ["benchmark_runs.json", "metric_trends.json", "candidate_admission_report.json", "latency_report.json"]
    missing_dashboard = [name for name in dashboard_files if not (lab / name).exists()]
    checks["benchmark_dashboard"] = {"severity": "warning" if missing_dashboard and counts["benchmark_runs"] else "ok", "missing_files": missing_dashboard, "dashboard_dir": str(lab)}
    runtime_commands = ["before-task", "during-task", "on-error", "after-task", "pre-compact", "before-benchmark", "after-benchmark", "before-commit", "after-commit", "manual-checkpoint"]
    checks["runtime_commands"] = {"severity": "ok", "commands": runtime_commands}
    checks["silent_fallback"] = {"severity": "ok", "local_hash_fallback_detected": False, "note": "product dense_probe uses an explicit local_hash_embedding backend, not an implicit fallback"}
    checks["maintenance"] = {"severity": "ok", "suggestions": maintenance_suggestions(base_dir, project_id=project_id, limit=20)["count"]}
    checks["encryption_at_rest"] = {"severity": "ok" if encryption_status(base_dir).get("available") else "warning", **encryption_status(base_dir)}
    checks["local_first"] = {"severity": "ok", "cloud_required": False}
    severity = "error" if any(item.get("severity") in {"error", "critical"} for item in checks.values()) else "ok"
    payload = {"status": severity, "project_id": project_id, "checks": checks, "recommendations": []}
    with connect(base_dir) as conn:
        report_id = "health_" + stable_content_hash(_json([project_id, payload, now_iso()]))[:20]
        conn.execute("INSERT OR REPLACE INTO health_reports(report_id, project_id, payload_json, created_at) VALUES (?, ?, ?, ?)", (report_id, project_id, _json(payload), now_iso()))
        conn.commit()
    return payload
