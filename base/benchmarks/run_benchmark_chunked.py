from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = Path(__file__).resolve().parent
MERGE_SCRIPT = BENCHMARKS_DIR / "merge_benchmark_results.py"
BENCHMARK_SCRIPTS = {
    "longmemeval": BENCHMARKS_DIR / "longmemeval_benchmark.py",
    "locomo": BENCHMARKS_DIR / "locomo_benchmark.py",
    "knowme": BENCHMARKS_DIR / "knowme_benchmark.py",
    "clonemem": BENCHMARKS_DIR / "clonemem_benchmark.py",
}

KNOWME_OFFICIAL_FORMAL_ENV = {
    "SPHERE_DISABLE_LOCAL_HASH_FALLBACK": "1",
    "SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING": "1",
    "SPHERE_ORACLE_RETRIEVAL_MODE": "direct_index",
    "SPHERE_KNOWME_LIGHTWEIGHT_DIAGNOSTICS": "1",
    "SPHERE_KNOWME_POOL_LIMIT": "100",
    "SPHERE_DENSE_TOP_K": "100",
    "SPHERE_LEXICAL_TOP_K": "100",
    "SPHERE_ENTITY_TOP_K": "80",
    "SPHERE_TEMPORAL_TOP_K": "50",
    "SPHERE_EXACT_PHRASE_TOP_K": "50",
    "SPHERE_PROFILE_SIDE_INDEX_TOP_K": "50",
    "SPHERE_QUERY_DECOMPOSITION_TOP_K": "50",
    "SPHERE_MAX_TOTAL_NEIGHBOR_CANDIDATES": "50",
    "SPHERE_PARENT_TOP_K": "10",
    "SPHERE_PARENT_EXPAND_SEGMENTS": "4",
}


def benchmark_profile_env(args: argparse.Namespace) -> dict[str, str]:
    if args.benchmark == "knowme" and args.knowme_profile == "official_formal":
        return dict(KNOWME_OFFICIAL_FORMAL_ENV)
    return {}


def benchmark_cache_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.benchmark == "knowme" and args.knowme_profile == "official_formal":
        return args.out / "cache_workspace", args.out / "cache_embedding"
    workspace_env_name = f"SPHERE_{args.benchmark.upper()}_CACHE_ROOT"
    embed_env_name = f"SPHERE_{args.benchmark.upper()}_EMBED_CACHE_ROOT"
    return (
        Path(
            os.environ.get(workspace_env_name)
            or ROOT / "benchmarks" / ".cache" / "workspace_cache" / args.benchmark
        ),
        Path(
            os.environ.get(embed_env_name)
            or ROOT / "benchmarks" / ".cache" / "embedding_cache" / args.benchmark
        ),
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def default_python() -> str:
    return sys.executable


def resolve_data_arg(benchmark: str, data_root: Path) -> Path:
    if benchmark == "longmemeval":
        for candidate in [data_root / "longmemeval_s_cleaned.json", data_root]:
            if candidate.exists() and candidate.is_file():
                return candidate
    if benchmark == "locomo":
        for candidate in [data_root / "locomo10.json", data_root / "locomo" / "locomo10.json", data_root]:
            if candidate.exists() and candidate.is_file():
                return candidate
    if benchmark == "knowme":
        for candidate in [data_root / "knowme", data_root / "KnowmeBench", data_root]:
            if candidate.exists():
                return candidate
    if benchmark == "clonemem":
        for candidate in [data_root / "clonemem", data_root / "releases", data_root]:
            if candidate.exists():
                return candidate
    return data_root / benchmark if (data_root / benchmark).exists() else data_root


def load_longmemeval_items(path: Path, max_questions: int) -> list[Any]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Expected LongMemEval list at {path}")
    return items[:max_questions] if max_questions > 0 else items


def split_even(items: list[Any], chunks: int) -> list[list[Any]]:
    if chunks <= 0:
        raise ValueError("--chunks must be positive")
    if not items:
        return [[] for _ in range(chunks)]
    chunk_size = math.ceil(len(items) / chunks)
    split = [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]
    while len(split) < chunks:
        split.append([])
    return split[:chunks]


def split_round_robin(items: list[str], chunks: int) -> list[list[str]]:
    if chunks <= 0:
        raise ValueError("--chunks must be positive")
    split: list[list[str]] = [[] for _ in range(chunks)]
    for index, item in enumerate(sorted(items)):
        split[index % chunks].append(item)
    return split


def _stable_bucket(value: str, chunks: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % chunks


def detect_clonemem_sample_id(sample_path: Path) -> str:
    match = re.search(r"_benchmark_(en|zh)$", sample_path.stem)
    if match:
        suffix = f"_benchmark_{match.group(1)}"
        return sample_path.stem[: -len(suffix)]
    return sample_path.stem


def discover_clonemem_sample_ids(data_root: Path, context_len: str, language: str) -> list[str]:
    data_arg = resolve_data_arg("clonemem", data_root)
    sample_ids: set[str] = set()
    for context_dir in sorted(path for path in data_arg.iterdir() if path.is_dir()):
        if context_len != "all" and context_dir.name != context_len:
            continue
        pattern = "*_benchmark_*.json" if language == "all" else f"*_benchmark_{language}.json"
        for sample_path in sorted(context_dir.glob(pattern)):
            sample_ids.add(detect_clonemem_sample_id(sample_path))
    return sorted(sample_ids)


def validate_metrics(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(payload.get("question_count") or len(payload.get("results") or []) or 0) >= 0


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def chunk_sample_allowlist_path(args: argparse.Namespace, chunk_index: int, chunk_dir: Path) -> Path | None:
    if args.shard_strategy != "sample":
        return None
    path = chunk_dir / "sample_id_allowlist.txt"
    return path if path.exists() else None


def sample_allowlist_is_empty(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    return not any(line.strip() for line in path.read_text(encoding="utf-8").splitlines())


def benchmark_command(args: argparse.Namespace, chunk_index: int, metrics_path: Path, chunk_dir: Path) -> list[str]:
    benchmark = args.benchmark
    script = BENCHMARK_SCRIPTS[benchmark]
    data_arg = resolve_data_arg(benchmark, args.data_root)
    command = [args.python_exe, str(script), str(data_arg)]
    if benchmark == "longmemeval":
        command = [args.python_exe, str(script), str(chunk_dir / "input.json")]
    command.extend(["--mode", args.mode, "--top-k", str(args.top_k), "--rerank-mode", args.rerank_mode])
    if benchmark in {"longmemeval", "locomo"}:
        command.extend(["--granularity", args.granularity])
    if benchmark == "clonemem":
        command.extend(["--context-len", args.context_len, "--language", args.language])
    if benchmark in {"locomo", "knowme", "clonemem"} and args.shard_strategy != "sample":
        command.extend(["--shard-index", str(chunk_index), "--shard-count", str(args.chunks)])
    sample_allowlist = chunk_sample_allowlist_path(args, chunk_index, chunk_dir)
    if sample_allowlist is not None:
        command.extend(["--sample-id-allowlist", str(sample_allowlist)])
    if benchmark in {"locomo", "knowme", "clonemem"}:
        if args.max_questions > 0:
            command.extend(["--max-questions", str(args.max_questions)])
        if args.resume:
            command.append("--resume-existing")
    command.extend(["--out", str(metrics_path)])
    return command


def chunk_env(args: argparse.Namespace, chunk_index: int, attempt: int, chunk_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("SPHERE_CREATIVE_MODE", "off")
    env.setdefault("SPHERE_EMBEDDING_FAIL_FAST", "1")
    env.setdefault("SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING", "1")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["SPHERE_WORKSPACE_NAME"] = f"{args.benchmark}_chunk_{chunk_index:02d}_attempt_{attempt:02d}"
    cache_root = chunk_dir / f"cache_attempt_{attempt:02d}"
    workspace_env_name = f"SPHERE_{args.benchmark.upper()}_CACHE_ROOT"
    embed_env_name = f"SPHERE_{args.benchmark.upper()}_EMBED_CACHE_ROOT"
    workspace_root, embed_root = benchmark_cache_roots(args)
    env[workspace_env_name] = str(workspace_root)
    env[f"SPHERE_{args.benchmark.upper()}_CHUNK_CACHE_ROOT"] = str(cache_root)
    env[embed_env_name] = str(embed_root)
    env["SPHERE_BENCHMARK_SHARED_WORKSPACE_CACHE"] = "1"
    env.update(benchmark_profile_env(args))
    return env


def prepare_longmemeval_inputs(args: argparse.Namespace, run_dir: Path) -> None:
    if args.benchmark != "longmemeval":
        return
    data_arg = resolve_data_arg("longmemeval", args.data_root)
    chunks = split_even(load_longmemeval_items(data_arg, args.max_questions), args.chunks)
    for index, items in enumerate(chunks):
        chunk_dir = run_dir / args.benchmark / f"chunk_{index:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        (chunk_dir / "input.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_sample_shard_allowlists(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    if args.shard_strategy != "sample":
        return {}
    if args.benchmark != "clonemem":
        raise ValueError("--shard-strategy=sample is currently supported for CloneMem only")
    sample_ids = discover_clonemem_sample_ids(args.data_root, args.context_len, args.language)
    if not sample_ids:
        raise FileNotFoundError(f"No CloneMem sample ids found under {args.data_root}")
    if args.sample_shard_assignment == "hash":
        split: list[list[str]] = [[] for _ in range(args.chunks)]
        for sample_id in sample_ids:
            split[_stable_bucket(f"clonemem:{sample_id}", args.chunks)].append(sample_id)
        split = [sorted(items) for items in split]
    else:
        split = split_round_robin(sample_ids, args.chunks)
    manifest: dict[str, Any] = {
        "shard_strategy": "sample",
        "sample_shard_assignment": args.sample_shard_assignment,
        "total_sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "chunks": {},
    }
    for index, chunk_sample_ids in enumerate(split):
        chunk_dir = run_dir / args.benchmark / f"chunk_{index:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        allowlist_path = chunk_dir / "sample_id_allowlist.txt"
        allowlist_path.write_text("\n".join(chunk_sample_ids) + ("\n" if chunk_sample_ids else ""), encoding="utf-8")
        manifest["chunks"][f"chunk_{index:02d}"] = {
            "sample_count": len(chunk_sample_ids),
            "sample_ids": chunk_sample_ids,
            "sample_id_allowlist": str(allowlist_path),
        }
    return manifest


def run_chunk(args: argparse.Namespace, run_dir: Path, manifest_path: Path, manifest: dict[str, Any], chunk_index: int) -> dict[str, Any]:
    chunk_name = f"chunk_{chunk_index:02d}"
    chunk_dir = run_dir / args.benchmark / chunk_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = chunk_dir / "metrics.json"
    stdout_path = chunk_dir / "stdout.txt"
    stderr_path = chunk_dir / "stderr.txt"
    chunk_manifest_path = chunk_dir / "chunk_manifest.json"
    if args.resume and not args.force and validate_metrics(metrics_path):
        result = {
            "chunk": chunk_name,
            "status": "skipped_existing",
            "metrics": str(metrics_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "retry_count": 0,
            "benchmark_profile": {
                "knowme_profile": args.knowme_profile if args.benchmark == "knowme" else None,
                "env": benchmark_profile_env(args),
            },
        }
        write_manifest(chunk_manifest_path, result)
        return result
    sample_allowlist = chunk_sample_allowlist_path(args, chunk_index, chunk_dir)
    if sample_allowlist_is_empty(sample_allowlist):
        metrics_payload = {
            "benchmark": args.benchmark,
            "mode": args.mode,
            "ablation": os.environ.get("SPHERE_BENCHMARK_ABLATION", "") or None,
            "run_type": os.environ.get("SPHERE_BENCHMARK_ABLATION", "") or None,
            "question_count": 0,
            "total_question_count": 0,
            "results": [],
            "metrics": {},
            "elapsed_seconds": 0.0,
            "empty_sample_allowlist": True,
            "sample_id_allowlist": str(sample_allowlist),
        }
        metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        stdout_path.write_text("empty sample allowlist; skipped benchmark subprocess\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        result = {
            "chunk": chunk_name,
            "status": "completed",
            "started_at": now_iso(),
            "finished_at": now_iso(),
            "elapsed_seconds": 0.0,
            "returncode": 0,
            "retry_count": 0,
            "command": [],
            "metrics": str(metrics_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "empty_sample_allowlist": True,
        }
        write_manifest(chunk_manifest_path, result)
        return result
    attempts = int(args.max_chunk_retries) + 1
    last_result: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        command = benchmark_command(args, chunk_index, metrics_path, chunk_dir)
        started = time.monotonic()
        started_at = now_iso()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=chunk_env(args, chunk_index, attempt, chunk_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        elapsed = time.monotonic() - started
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        ok = completed.returncode == 0 and validate_metrics(metrics_path)
        last_result = {
            "chunk": chunk_name,
            "status": "completed" if ok else "failed_attempt",
            "started_at": started_at,
            "finished_at": now_iso(),
            "elapsed_seconds": round(elapsed, 3),
            "returncode": completed.returncode,
            "retry_count": attempt - 1,
            "command": command,
            "metrics": str(metrics_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "benchmark_profile": {
                "knowme_profile": args.knowme_profile if args.benchmark == "knowme" else None,
                "env": benchmark_profile_env(args),
            },
        }
        write_manifest(chunk_manifest_path, last_result)
        if ok:
            return last_result
    last_result["status"] = "failed"
    return last_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark in chunked parallel mode.")
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARK_SCRIPTS))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-chunk-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--python-exe", default=default_python())
    parser.add_argument("--mode", choices=["vector", "bm25", "evidence", "activation", "hybrid"], default="evidence")
    parser.add_argument("--granularity", choices=["session", "dialog", "turn"], default="session")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rerank-mode", choices=["rule", "hybrid", "cross_encoder"], default="rule")
    parser.add_argument("--context-len", choices=["100k", "500k", "all"], default="all")
    parser.add_argument("--language", choices=["en", "zh", "all"], default="all")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument(
        "--knowme-profile",
        choices=["default", "official_formal"],
        default="default",
        help="KnowMe runtime profile. official_formal pins the capped env used by formal artifacts.",
    )
    parser.add_argument(
        "--shard-strategy",
        choices=["auto", "question", "sample"],
        default="auto",
        help="Chunk assignment strategy. auto uses sample sharding for CloneMem and question sharding otherwise.",
    )
    parser.add_argument(
        "--sample-shard-assignment",
        choices=["round_robin", "hash"],
        default="round_robin",
        help="Deterministic sample assignment used by --shard-strategy=sample.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.benchmark = str(args.benchmark).lower()
    args.data_root = args.data_root.resolve()
    args.out = args.out.resolve()
    args.python_exe = shutil.which(str(args.python_exe)) or str(args.python_exe)
    if args.shard_strategy == "auto":
        args.shard_strategy = "sample" if args.benchmark == "clonemem" else "question"
    workers = args.workers or min(max(1, (os.cpu_count() or 2) // 2), args.chunks)
    workers = max(1, min(workers, args.chunks))
    run_dir = args.out
    manifest_path = run_dir / "run_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_longmemeval_inputs(args, run_dir)
    sample_shard_manifest = prepare_sample_shard_allowlists(args, run_dir)
    workspace_root, embed_root = benchmark_cache_roots(args)
    manifest = {
        "benchmark": args.benchmark,
        "data_root": str(args.data_root),
        "out": str(run_dir),
        "benchmark_profile": {
            "knowme_profile": args.knowme_profile if args.benchmark == "knowme" else None,
            "env": benchmark_profile_env(args),
        },
        "shard_strategy": args.shard_strategy,
        "sample_shards": sample_shard_manifest,
        "chunks": [
            {
                "chunk": f"chunk_{index:02d}",
                "status": "pending",
                **dict((sample_shard_manifest.get("chunks") or {}).get(f"chunk_{index:02d}") or {}),
            }
            for index in range(args.chunks)
        ],
        "workers": workers,
        "max_chunk_retries": args.max_chunk_retries,
        "resume": bool(args.resume),
        "force": bool(args.force),
        "shared_workspace_cache_root": str(workspace_root),
        "shared_embedding_cache_root": str(embed_root),
        "started_at": now_iso(),
        "status": "running",
    }
    write_manifest(manifest_path, manifest)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    if not args.merge_only:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(run_chunk, args, run_dir, manifest_path, manifest, index): index
                for index in range(args.chunks)
            }
            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                for chunk in manifest["chunks"]:
                    if chunk["chunk"] == result["chunk"]:
                        chunk.update(result)
                        break
                write_manifest(manifest_path, manifest)
    else:
        for index in range(args.chunks):
            chunk_name = f"chunk_{index:02d}"
            metrics_path = run_dir / args.benchmark / chunk_name / "metrics.json"
            results.append({"chunk": chunk_name, "status": "completed" if validate_metrics(metrics_path) else "failed", "metrics": str(metrics_path)})
    successful_metrics = [Path(result["metrics"]) for result in results if str(result.get("status")) in {"completed", "skipped_existing"} and validate_metrics(Path(result["metrics"]))]
    failed = [result for result in results if str(result.get("status")) not in {"completed", "skipped_existing"}]
    wall = time.monotonic() - started
    manifest["status"] = "completed" if not failed else "failed"
    manifest["finished_at"] = now_iso()
    manifest["wall_clock_elapsed_seconds"] = round(wall, 3)
    manifest["successful_chunks"] = len(successful_metrics)
    manifest["failed_chunks"] = len(failed)
    write_manifest(manifest_path, manifest)
    if successful_metrics:
        merged_path = run_dir / args.benchmark / "merged_metrics.json"
        command = [args.python_exe, str(MERGE_SCRIPT), *[str(path) for path in successful_metrics], "--out", str(merged_path), "--run-manifest", str(manifest_path)]
        subprocess.run(command, cwd=ROOT, check=True)
        manifest["merged_metrics"] = str(merged_path)
        write_manifest(manifest_path, manifest)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
