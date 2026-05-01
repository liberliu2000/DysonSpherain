from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import stable_content_hash


RUN_TYPE_ORDER = {
    "full": 5,
    "targeted_subset": 4,
    "partial": 3,
    "smoke": 2,
    "debug": 1,
    "unknown": 0,
}

PRIMARY_METRIC_KEYS = (
    "recall_any@10",
    "recall_frac@10",
    "ndcg_any@10",
    "candidate_recall@100",
    "final_recall@10",
    "final_ndcg@10",
)

MAX_DISCOVERED_METRICS_BYTES = 25 * 1024 * 1024
MAX_SIDECAR_BYTES = 2 * 1024 * 1024
SIDECAR_NAMES = (
    "oracle",
    "candidate",
    "failure",
    "integrity",
    "diagnostic",
    "manifest",
    "log",
    "report",
)


@dataclass
class BenchmarkRun:
    run_id: str
    project: str
    dataset: str
    run_type: str
    timestamp: str
    artifact_dir: str
    metrics: dict[str, Any]
    question_count: int | None = None
    sample_count: int | None = None
    elapsed_seconds: float | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    fallback_in_use: bool | None = None
    fallback_reason: str | None = None
    config_hash: str | None = None
    dataset_version: str | None = None
    code_commit: str | None = None
    command: str | None = None
    comparable: bool = True
    comparability_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def registry_dir(base_dir: Path) -> Path:
    return base_dir / "artifacts" / "registry"


def registry_path(base_dir: Path) -> Path:
    return registry_dir(base_dir) / "benchmark_runs.jsonl"


def registry_summary_path(base_dir: Path) -> Path:
    return base_dir / "reports" / "artifact_registry_summary.md"


def registry_report_dir(base_dir: Path) -> Path:
    return base_dir / "reports" / "registry"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_read_json(path: Path, max_bytes: int = MAX_SIDECAR_BYTES) -> dict[str, Any]:
    try:
        if path.stat().st_size > max_bytes:
            return {}
        return _read_json(path)
    except Exception:
        return {}


def _now_from_file(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _first(payload: dict[str, Any], paths: list[tuple[str, ...]]) -> Any:
    for path in paths:
        cur: Any = payload
        ok = True
        for part in path:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_dataset(payload: dict[str, Any], path: Path) -> str:
    value = payload.get("benchmark") or payload.get("benchmark_name")
    if value:
        return str(value)
    known = {"longmemeval", "locomo", "knowme", "clonemem", "convomem"}
    for part in reversed(path.parts):
        normalized = part.strip().lower()
        if normalized in known:
            return normalized
    return "unknown"


def _infer_run_type(payload: dict[str, Any], path: Path) -> str:
    explicit = str(payload.get("run_type") or "").strip().lower()
    if explicit in RUN_TYPE_ORDER:
        return explicit
    text = " ".join(path.parts).lower()
    if "smoke" in text:
        return "smoke"
    if "targeted" in text or "subset" in text:
        return "targeted_subset"
    if "debug" in text:
        return "debug"
    if payload.get("failed_shards") or payload.get("status") in {"failed", "timeout", "runtime_timeout"}:
        return "partial"
    question_count = _coerce_int(payload.get("total_question_count") or payload.get("question_count"))
    if question_count and question_count > 100:
        return "full"
    return "unknown"


def _extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    summary = payload.get("candidate_recall_summary") if isinstance(payload.get("candidate_recall_summary"), dict) else {}
    extracted: dict[str, Any] = {}
    for key in PRIMARY_METRIC_KEYS:
        value = summary.get(key) if key in summary else metrics.get(key)
        if value is None and isinstance(metrics.get("session"), dict):
            value = metrics["session"].get(key)
        if value is None and isinstance(metrics.get("segment"), dict):
            value = metrics["segment"].get(key)
        if value is None and isinstance(payload.get("candidate_recall_report"), dict):
            value = payload["candidate_recall_report"].get(key)
        if value is not None:
            extracted[key] = value
    return extracted


def _hash_jsonish(value: Any) -> str | None:
    if value in (None, "", {}, []):
        return None
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return stable_content_hash(text)


def _current_code_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def _config_hash(payload: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    explicit = payload.get("config_hash") or manifest.get("config_hash")
    if explicit:
        return str(explicit)
    runtime_hash = _hash_jsonish(payload.get("runtime_config") or manifest.get("runtime_config"))
    if runtime_hash:
        return runtime_hash
    manifest_config = {
        "benchmark": payload.get("benchmark") or manifest.get("benchmark"),
        "mode": payload.get("mode"),
        "run_type": payload.get("run_type"),
        "ablation": payload.get("ablation"),
        "shard_strategy": manifest.get("shard_strategy"),
        "workers": manifest.get("workers"),
        "chunks": len(manifest.get("chunks") or []) if isinstance(manifest.get("chunks"), list) else None,
        "vector_info": payload.get("vector_info"),
        "embedding_info": payload.get("embedding_info"),
        "quality_guardrail_status": payload.get("quality_guardrail_status") or payload.get("quality_status"),
    }
    return _hash_jsonish({key: value for key, value in manifest_config.items() if value not in (None, "", {}, [])})


def _dataset_version(payload: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    explicit = payload.get("dataset_version") or manifest.get("dataset_version")
    if explicit:
        return str(explicit)
    dataset_shape = {
        "benchmark": payload.get("benchmark") or payload.get("benchmark_name") or manifest.get("benchmark"),
        "data_root": manifest.get("data_root"),
        "sample_shards": manifest.get("sample_shards"),
        "source_files": payload.get("source_files"),
        "question_count": payload.get("total_question_count") or payload.get("question_count"),
        "sample_count": payload.get("sample_count") or manifest.get("sample_count"),
    }
    return _hash_jsonish({key: value for key, value in dataset_shape.items() if value not in (None, "", {}, [])})


def _route_policy_hash(payload: dict[str, Any]) -> str | None:
    route_policy = (
        payload.get("route_policy_config")
        or payload.get("route_policy")
        or _first(payload, [("metadata", "route_policy_config"), ("metadata", "route_policy"), ("config", "route_policy")])
    )
    return _hash_jsonish(route_policy)


def _collect_sidecar_artifacts(metrics_path: Path) -> dict[str, Any]:
    roots = [metrics_path.parent, metrics_path.parent / "reports", metrics_path.parent / "reports" / "diagnostics"]
    sidecars: list[dict[str, Any]] = []
    failure_taxonomy: dict[str, Any] = {}
    oracle_summary: dict[str, Any] = {}
    integrity_summary: dict[str, Any] = {}
    notes: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path in seen or not path.is_file() or path == metrics_path:
                continue
            seen.add(path)
            name = path.name.lower()
            if not any(token in name for token in SIDECAR_NAMES):
                continue
            if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".log"}:
                continue
            rel = str(path.relative_to(metrics_path.parent)) if path.is_relative_to(metrics_path.parent) else str(path)
            sidecars.append({"path": rel, "bytes": path.stat().st_size})
            if path.suffix.lower() == ".json":
                payload = _safe_read_json(path)
                if not payload:
                    continue
                if "oracle" in name:
                    oracle_summary = {key: value for key, value in payload.items() if key != "rows"}
                if "integrity" in name:
                    integrity_summary = {key: value for key, value in payload.items() if key != "rows"}
                if "failure" in name:
                    failure_taxonomy = payload.get("failure_type_distribution") or payload.get("failure_summary") or payload
            elif path.suffix.lower() in {".md", ".txt", ".log"} and path.stat().st_size <= MAX_SIDECAR_BYTES:
                notes.append(rel)
    return {
        "sidecar_artifacts": sidecars[:200],
        "sidecar_artifact_count": len(sidecars),
        "failure_taxonomy": failure_taxonomy,
        "oracle_summary": oracle_summary,
        "integrity_summary": integrity_summary,
        "notes": notes[:50],
    }


def _find_nearby_manifest(metrics_path: Path) -> dict[str, Any]:
    candidates = [
        metrics_path.parent / "run_manifest.json",
        metrics_path.parent.parent / "run_manifest.json",
        metrics_path.parent.parent.parent / "run_manifest.json",
        metrics_path.parent / "chunk_manifest.json",
    ]
    for path in candidates:
        if path.exists():
            payload = _safe_read_json(path, max_bytes=8 * 1024 * 1024)
            if payload:
                payload["_manifest_path"] = str(path)
                return payload
    return {}


def _manifest_sample_count(manifest: dict[str, Any]) -> int | None:
    direct = _coerce_int(manifest.get("sample_count"))
    if direct is not None:
        return direct
    sample_ids: set[str] = set()
    chunks = manifest.get("chunks") if isinstance(manifest.get("chunks"), list) else []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for sample_id in list(chunk.get("sample_ids") or []):
            if str(sample_id or "").strip():
                sample_ids.add(str(sample_id))
        allowlist = chunk.get("sample_id_allowlist")
        if allowlist:
            path = Path(str(allowlist))
            try:
                if path.exists():
                    sample_ids.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            except OSError:
                pass
    return len(sample_ids) if sample_ids else None


def _manifest_command(manifest: dict[str, Any]) -> str | None:
    command = manifest.get("command")
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    if command:
        return str(command)
    chunks = manifest.get("chunks") if isinstance(manifest.get("chunks"), list) else []
    commands = [
        " ".join(str(part) for part in chunk.get("command"))
        for chunk in chunks
        if isinstance(chunk, dict) and isinstance(chunk.get("command"), list)
    ]
    if commands:
        return f"chunked_subprocess x{len(commands)}; first={commands[0]}"
    return None


def _extract_embedding(payload: dict[str, Any]) -> tuple[str | None, str | None, bool | None, str | None]:
    embedding_info = payload.get("embedding_info") if isinstance(payload.get("embedding_info"), dict) else {}
    provider = payload.get("embedding_provider") or embedding_info.get("embedding_provider") or embedding_info.get("provider")
    model = payload.get("embedding_model") or embedding_info.get("embedding_model") or embedding_info.get("model")
    fallback = payload.get("fallback_in_use")
    if fallback is None:
        fallback = embedding_info.get("fallback_in_use")
    reason = payload.get("fallback_reason") or embedding_info.get("fallback_reason")
    return (
        str(provider) if provider is not None else None,
        str(model) if model is not None else None,
        bool(fallback) if fallback is not None else None,
        str(reason) if reason is not None else None,
    )


def _comparability_warnings(run: BenchmarkRun) -> list[str]:
    warnings: list[str] = []
    if run.fallback_in_use:
        warnings.append("fallback_in_use")
    if run.run_type in {"partial", "debug", "unknown"}:
        warnings.append(f"non_formal_run_type:{run.run_type}")
    if not run.question_count:
        warnings.append("missing_question_count")
    if not run.embedding_provider:
        warnings.append("missing_embedding_provider")
    if not run.embedding_model:
        warnings.append("missing_embedding_model")
    return warnings


def run_from_metrics(path: Path, project: str) -> BenchmarkRun:
    payload = _read_json(path)
    dataset = _infer_dataset(payload, path)
    run_type = _infer_run_type(payload, path)
    provider, model, fallback, fallback_reason = _extract_embedding(payload)
    elapsed = _coerce_float(payload.get("wall_clock_elapsed_seconds") or payload.get("elapsed_seconds"))
    sidecars = _collect_sidecar_artifacts(path)
    manifest = _find_nearby_manifest(path)
    route_hash = _route_policy_hash(payload)
    config_hash = _config_hash(payload, manifest)
    dataset_version = _dataset_version(payload, manifest)
    manifest_sample_count = _manifest_sample_count(manifest)
    manifest_command = _manifest_command(manifest)
    code_commit = payload.get("code_commit") or payload.get("git_sha") or manifest.get("code_commit") or manifest.get("git_sha") or _current_code_commit()
    metadata = {
        "source_metrics_path": str(path),
        "manifest_path": manifest.get("_manifest_path"),
        "quality_guardrail_status": payload.get("quality_guardrail_status") or payload.get("quality_status"),
        "successful_shards": payload.get("successful_shards") or manifest.get("successful_chunks"),
        "failed_shards": payload.get("failed_shards") or manifest.get("failed_chunks"),
        "shard_count": payload.get("shard_count")
        or (len(manifest.get("chunks")) if isinstance(manifest.get("chunks"), list) else None),
        "workers": manifest.get("workers"),
        "shard_strategy": manifest.get("shard_strategy"),
        "speedup_estimate": payload.get("speedup_estimate"),
        "route_policy_hash": route_hash,
        "runtime_config_hash": _hash_jsonish(payload.get("runtime_config") or manifest.get("runtime_config")),
        **{key: value for key, value in sidecars.items() if value not in (None, {}, [], 0)},
    }
    seed = {
        "project": project,
        "dataset": dataset,
        "artifact": str(path.resolve()),
        "mtime": path.stat().st_mtime,
    }
    run = BenchmarkRun(
        run_id=f"{dataset}-{stable_content_hash(json.dumps(seed, sort_keys=True))[:12]}",
        project=project,
        dataset=dataset,
        run_type=run_type,
        timestamp=str(payload.get("timestamp") or _now_from_file(path)),
        artifact_dir=str(path.parent),
        metrics=_extract_metrics(payload),
        question_count=_coerce_int(payload.get("total_question_count") or payload.get("question_count")),
        sample_count=_coerce_int(payload.get("sample_count")) or manifest_sample_count,
        elapsed_seconds=elapsed,
        embedding_provider=provider,
        embedding_model=model,
        fallback_in_use=fallback,
        fallback_reason=fallback_reason,
        config_hash=config_hash,
        dataset_version=dataset_version,
        code_commit=code_commit,
        command=payload.get("command") or manifest_command,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )
    run.comparability_warnings = _comparability_warnings(run)
    run.comparable = not any(w in {"fallback_in_use", "missing_question_count"} for w in run.comparability_warnings)
    return run


def discover_metric_files(path: Path, max_discovered_metrics_bytes: int = MAX_DISCOVERED_METRICS_BYTES) -> list[Path]:
    if path.is_file():
        return [path] if path.name in {"metrics.json", "merged_metrics.json"} else []
    files = list(path.rglob("merged_metrics.json"))
    files.extend(p for p in path.rglob("metrics.json") if not p.parent.name.startswith("chunk_"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for file in sorted(files):
        resolved = file.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if file.name == "metrics.json" and max_discovered_metrics_bytes > 0 and file.stat().st_size > max_discovered_metrics_bytes:
            continue
        unique.append(file)
    return unique


def load_registry(base_dir: Path) -> list[BenchmarkRun]:
    path = registry_path(base_dir)
    if not path.exists():
        return []
    runs: list[BenchmarkRun] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        runs.append(BenchmarkRun(**payload))
    return runs


def save_registry(base_dir: Path, runs: list[BenchmarkRun]) -> None:
    path = registry_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {run.run_id: run for run in runs}
    lines = [json.dumps(asdict(run), ensure_ascii=False, sort_keys=True) for run in sorted(by_id.values(), key=lambda r: (r.project, r.dataset, r.timestamp, r.run_id))]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def ingest_artifacts(path: Path, base_dir: Path, project: str) -> list[BenchmarkRun]:
    existing = load_registry(base_dir)
    by_id = {run.run_id: run for run in existing}
    ingested: list[BenchmarkRun] = []
    for metrics_path in discover_metric_files(path):
        run = run_from_metrics(metrics_path, project)
        by_id[run.run_id] = run
        ingested.append(run)
    save_registry(base_dir, list(by_id.values()))
    write_registry_summary(base_dir, list(by_id.values()))
    return ingested


def _rank_latest(run: BenchmarkRun) -> tuple[int, int, str]:
    fallback_score = 0 if run.fallback_in_use else 1
    return (RUN_TYPE_ORDER.get(run.run_type, 0), fallback_score, run.timestamp)


def latest_run(runs: list[BenchmarkRun], project: str, dataset: str) -> BenchmarkRun | None:
    matches = [run for run in runs if run.project == project and run.dataset.lower() == dataset.lower()]
    if not matches:
        return None
    return sorted(matches, key=_rank_latest, reverse=True)[0]


def compare_runs(a: BenchmarkRun, b: BenchmarkRun) -> dict[str, Any]:
    warnings: list[str] = []
    for field_name in ("dataset", "run_type", "question_count", "embedding_provider", "embedding_model", "fallback_in_use", "config_hash", "dataset_version", "code_commit"):
        if getattr(a, field_name) != getattr(b, field_name):
            warnings.append(f"different_{field_name}")
    if a.sample_count != b.sample_count:
        if not (a.question_count and a.question_count == b.question_count and a.run_type == b.run_type == "full"):
            warnings.append("different_sample_count")
        else:
            warnings.append("different_sample_count_definition")
    if (a.metadata or {}).get("route_policy_hash") != (b.metadata or {}).get("route_policy_hash"):
        warnings.append("different_route_policy_config")
    metric_rows: list[dict[str, Any]] = []
    keys = sorted(set(a.metrics) | set(b.metrics))
    for key in keys:
        av = _coerce_float(a.metrics.get(key))
        bv = _coerce_float(b.metrics.get(key))
        metric_rows.append({"metric": key, "a": av, "b": bv, "delta": None if av is None or bv is None else bv - av})
    return {"a": asdict(a), "b": asdict(b), "warnings": warnings, "metrics": metric_rows}


def regression_explanation(runs: list[BenchmarkRun], project: str, dataset: str) -> dict[str, Any]:
    matches = [run for run in runs if run.project == project and run.dataset.lower() == dataset.lower()]
    matches = sorted(matches, key=lambda run: run.timestamp)
    if len(matches) < 2:
        return {"dataset": dataset, "project": project, "status": "insufficient_runs", "runs": [asdict(run) for run in matches]}
    comparison = compare_runs(matches[-2], matches[-1])
    regressions = [row for row in comparison["metrics"] if row.get("delta") is not None and row["delta"] < 0]
    return {
        "dataset": dataset,
        "project": project,
        "status": "regression_detected" if regressions else "no_metric_regression_detected",
        "previous_run_id": matches[-2].run_id,
        "latest_run_id": matches[-1].run_id,
        "comparability_warnings": comparison["warnings"],
        "regressions": regressions,
        "comparison": comparison,
    }


def markdown_compare_table(comparison: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark Run Compare",
        "",
        f"- A: `{comparison['a']['run_id']}`",
        f"- B: `{comparison['b']['run_id']}`",
        f"- Warnings: {', '.join(comparison['warnings']) if comparison['warnings'] else 'none'}",
        "",
        "| metric | a | b | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["metrics"]:
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.6f}"

        lines.append(f"| {row['metric']} | {fmt(row['a'])} | {fmt(row['b'])} | {fmt(row['delta'])} |")
    return "\n".join(lines) + "\n"


def compare_report_path(base_dir: Path, a: BenchmarkRun, b: BenchmarkRun) -> Path:
    return registry_report_dir(base_dir) / f"compare_{a.run_id}_vs_{b.run_id}.md"


def regression_report_path(base_dir: Path, dataset: str) -> Path:
    safe_dataset = "".join(ch.lower() if ch.isalnum() else "_" for ch in dataset).strip("_") or "dataset"
    return registry_report_dir(base_dir) / f"regression_{safe_dataset}.md"


def write_compare_report(base_dir: Path, comparison: dict[str, Any], out: Path | None = None) -> Path:
    path = out or compare_report_path(base_dir, BenchmarkRun(**comparison["a"]), BenchmarkRun(**comparison["b"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_compare_table(comparison), encoding="utf-8")
    return path


def markdown_regression_report(explanation: dict[str, Any]) -> str:
    lines = [
        f"# Regression Explanation: {explanation.get('dataset')}",
        "",
        f"- Status: `{explanation.get('status')}`",
        f"- Previous: `{explanation.get('previous_run_id', 'n/a')}`",
        f"- Latest: `{explanation.get('latest_run_id', 'n/a')}`",
        f"- Warnings: {', '.join(explanation.get('comparability_warnings') or []) or 'none'}",
        "",
        "| metric | previous | latest | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in explanation.get("regressions") or []:
        lines.append(f"| {row['metric']} | {row['a']:.6f} | {row['b']:.6f} | {row['delta']:.6f} |")
    if not explanation.get("regressions"):
        lines.append("| n/a | n/a | n/a | n/a |")
    return "\n".join(lines) + "\n"


def write_regression_report(base_dir: Path, explanation: dict[str, Any], out: Path | None = None) -> Path:
    path = out or regression_report_path(base_dir, str(explanation.get("dataset") or "dataset"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_regression_report(explanation), encoding="utf-8")
    return path


def write_registry_summary(base_dir: Path, runs: list[BenchmarkRun]) -> Path:
    path = registry_summary_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Artifact Registry Summary",
        "",
        f"Generated from artifact files at {datetime.now(timezone.utc).isoformat()}.",
        "",
        "| project | dataset | run_id | run_type | questions | fallback | comparable | artifact_dir |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for run in sorted(runs, key=lambda item: (item.project, item.dataset, item.timestamp, item.run_id)):
        lines.append(
            "| "
            + " | ".join(
                [
                    run.project,
                    run.dataset,
                    run.run_id,
                    run.run_type,
                    str(run.question_count or ""),
                    str(run.fallback_in_use),
                    str(run.comparable),
                    run.artifact_dir,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def resolve_run(runs: list[BenchmarkRun], run_id: str) -> BenchmarkRun:
    for run in runs:
        if run.run_id == run_id:
            return run
    matches = [run for run in runs if run.run_id.startswith(run_id)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Unknown run id {run_id!r}.")
    raise KeyError(f"Ambiguous run id {run_id!r}: {', '.join(run.run_id for run in matches)}")
