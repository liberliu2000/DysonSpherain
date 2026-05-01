from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "benchmarks" / "longmemeval_benchmark.py"
MERGE_SCRIPT = ROOT / "benchmarks" / "merge_longmemeval_results.py"
DEFAULT_DATA_FILE = ROOT / "data" / "benchmarks" / "longmemeval_s_cleaned.json"
DEFAULT_RUNS_ROOT = ROOT / "benchmarks" / "runs"


def default_python_exe() -> Path:
    venv_python = ROOT / ".venv312" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LongMemEval in chunked parallel mode and merge results.")
    parser.add_argument("data_file", nargs="?", type=Path, default=DEFAULT_DATA_FILE, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--mode", choices=["vector", "bm25", "evidence", "activation", "hybrid"], default="evidence")
    parser.add_argument("--granularity", choices=["session", "turn"], default="session")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rerank-mode", choices=["rule", "hybrid", "cross_encoder"], default="rule")
    parser.add_argument("--task-type", default="qa")
    parser.add_argument("--shell", type=int, default=2)
    parser.add_argument("--sector", default="knowledge")
    parser.add_argument("--zone", default="longmemeval")
    parser.add_argument("--chunk-pool", type=int, default=400)
    parser.add_argument("--object-top-k", type=int, default=4)
    parser.add_argument("--support-top-k", type=int, default=4)
    parser.add_argument("--cognitive-top-k", type=int, default=0)
    parser.add_argument("--cross-encoder", action="store_true", help="Enable cross-encoder reranking in the benchmark runner.")
    parser.add_argument("--chunks", type=int, default=5, help="Number of input chunks to create.")
    parser.add_argument("--workers", type=int, default=5, help="Maximum number of chunk processes to run in parallel.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N questions before splitting.")
    parser.add_argument("--max-chunk-retries", type=int, default=1, help="Retry a failed chunk process this many times.")
    parser.add_argument("--python-exe", type=Path, default=default_python_exe(), help="Python executable used for child benchmark processes.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="Parent directory for benchmark run folders.")
    parser.add_argument("--run-name", default=None, help="Optional run directory name. Defaults to a timestamped name.")
    parser.add_argument("--out", type=Path, default=None, help="Optional merged JSON output path. Defaults inside the run directory.")
    parser.add_argument("--tail-lines", type=int, default=60, help="How many log lines to print when a chunk fails.")
    parser.add_argument("--force", action="store_true", help="Re-run chunks even if a valid output JSON already exists.")
    parser.add_argument("--online", action="store_true", help="Allow model code to contact Hugging Face instead of forcing offline mode.")
    return parser.parse_args()


@dataclass
class ChunkSpec:
    index: int
    question_count: int
    input_path: Path
    output_path: Path

    @property
    def chunk_id(self) -> str:
        return f"{self.index:02d}"


class RunState:
    def __init__(self, meta_path: Path, payload: dict[str, Any]) -> None:
        self.meta_path = meta_path
        self.payload = payload
        self._lock = threading.Lock()
        self.flush()

    def update_chunk(self, chunk_id: str, **updates: Any) -> None:
        with self._lock:
            for chunk in self.payload.get("chunks", []):
                if chunk.get("chunk") == chunk_id:
                    chunk.update(updates)
                    break
            self.flush()

    def update_root(self, **updates: Any) -> None:
        with self._lock:
            self.payload.update(updates)
            self.flush()

    def flush(self) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions(data_file: Path, limit: int) -> list[dict[str, Any]]:
    items = json.loads(data_file.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Expected a top-level list in {data_file}")
    if limit > 0:
        return items[:limit]
    return items


def split_questions(items: list[dict[str, Any]], chunks: int) -> list[list[dict[str, Any]]]:
    if chunks <= 0:
        raise ValueError("--chunks must be positive")
    if not items:
        raise ValueError("No questions available to split")
    chunk_size = math.ceil(len(items) / chunks)
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def build_run_dir(args: argparse.Namespace, question_count: int) -> Path:
    if args.run_name:
        return args.runs_root / args.run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"longmemeval_{args.mode}_{question_count}_chunked"
    return args.runs_root / f"{stamp}_{suffix}"


def rerank_tag(mode: str, rerank_mode: str) -> str:
    return f"_{rerank_mode}" if mode == "hybrid" else ""


def merged_output_path(args: argparse.Namespace, run_dir: Path, question_count: int) -> Path:
    if args.out is not None:
        return args.out
    filename = f"results_longmemeval_{args.mode}{rerank_tag(args.mode, args.rerank_mode)}_{question_count}_merged.json"
    return run_dir / filename


def build_chunk_specs(run_dir: Path, chunked_items: list[list[dict[str, Any]]], mode: str) -> list[ChunkSpec]:
    specs: list[ChunkSpec] = []
    for index, chunk_items in enumerate(chunked_items, start=1):
        input_path = run_dir / f"longmemeval_chunk_{index:02d}_input.json"
        input_path.write_text(json.dumps(chunk_items, ensure_ascii=False, indent=2), encoding="utf-8")
        output_path = run_dir / f"longmemeval_{mode}_chunk_{index:02d}.json"
        specs.append(
            ChunkSpec(
                index=index,
                question_count=len(chunk_items),
                input_path=input_path,
                output_path=output_path,
            )
        )
    return specs


def validate_chunk_output(path: Path, expected_questions: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(payload.get("question_count", -1)) == expected_questions and len(payload.get("results", [])) == expected_questions


def tail_text(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def build_chunk_command(args: argparse.Namespace, chunk: ChunkSpec, output_path: Path) -> list[str]:
    command = [
        str(args.python_exe),
        str(BENCHMARK_SCRIPT),
        str(chunk.input_path),
        "--mode",
        args.mode,
        "--granularity",
        args.granularity,
        "--top-k",
        str(args.top_k),
        "--rerank-mode",
        args.rerank_mode,
        "--task-type",
        args.task_type,
        "--shell",
        str(args.shell),
        "--sector",
        args.sector,
        "--zone",
        args.zone,
        "--chunk-pool",
        str(args.chunk_pool),
        "--object-top-k",
        str(args.object_top_k),
        "--support-top-k",
        str(args.support_top_k),
        "--cognitive-top-k",
        str(args.cognitive_top_k),
        "--out",
        str(output_path),
    ]
    if args.cross_encoder:
        command.append("--cross-encoder")
    return command


def chunk_env(base_env: dict[str, str], run_dir: Path, chunk: ChunkSpec, attempt: int, online: bool) -> dict[str, str]:
    env = dict(base_env)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if not online:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["SPHERE_LONGMEMEVAL_CACHE_ROOT"] = str(run_dir / f"cache_chunk_{chunk.chunk_id}_workspace_attempt_{attempt:02d}")
    env["SPHERE_LONGMEMEVAL_EMBED_CACHE_ROOT"] = str(run_dir / f"cache_chunk_{chunk.chunk_id}_embed_attempt_{attempt:02d}")
    return env


def run_chunk(args: argparse.Namespace, run_dir: Path, chunk: ChunkSpec, state: RunState) -> Path:
    if not args.force and validate_chunk_output(chunk.output_path, chunk.question_count):
        print(f"[chunk {chunk.chunk_id}] reusing existing output", flush=True)
        state.update_chunk(chunk.chunk_id, status="completed", attempts=0, reused_existing_output=True, out=str(chunk.output_path))
        return chunk.output_path

    base_env = os.environ.copy()
    attempts_total = args.max_chunk_retries + 1
    for attempt in range(1, attempts_total + 1):
        log_path = run_dir / f"longmemeval_{args.mode}_chunk_{chunk.chunk_id}_attempt_{attempt:02d}.log"
        state.update_chunk(
            chunk.chunk_id,
            status="running",
            attempts=attempt,
            reused_existing_output=False,
            log=str(log_path),
            out=str(chunk.output_path),
        )
        print(f"[chunk {chunk.chunk_id}] attempt {attempt}/{attempts_total} starting", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                build_chunk_command(args, chunk, chunk.output_path),
                cwd=ROOT,
                env=chunk_env(base_env, run_dir, chunk, attempt, args.online),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode == 0 and validate_chunk_output(chunk.output_path, chunk.question_count):
            print(f"[chunk {chunk.chunk_id}] completed", flush=True)
            state.update_chunk(chunk.chunk_id, status="completed", returncode=0, log=str(log_path))
            return chunk.output_path
        failure_tail = tail_text(log_path, args.tail_lines)
        state.update_chunk(
            chunk.chunk_id,
            status="failed_attempt",
            returncode=completed.returncode,
            log=str(log_path),
            last_failure_tail=failure_tail,
        )
        print(f"[chunk {chunk.chunk_id}] attempt {attempt} failed", flush=True)
        if attempt == attempts_total:
            raise RuntimeError(
                f"Chunk {chunk.chunk_id} failed after {attempts_total} attempts.\n"
                f"Log: {log_path}\n"
                f"{failure_tail}"
            )
    raise RuntimeError(f"Chunk {chunk.chunk_id} exited retry loop unexpectedly")


def merge_results(args: argparse.Namespace, outputs: list[Path], merged_path: Path) -> None:
    command = [str(args.python_exe), str(MERGE_SCRIPT), *[str(path) for path in outputs], "--out", str(merged_path)]
    subprocess.run(command, cwd=ROOT, check=True)


def summarize_merged(merged_path: Path) -> dict[str, Any]:
    payload = json.loads(merged_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {}).get("session", {})
    return {
        "question_count": int(payload.get("question_count", 0)),
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
        "recall_any@5": float(metrics.get("recall_any@5", 0.0)),
        "recall_any@10": float(metrics.get("recall_any@10", 0.0)),
        "ndcg_any@10": float(metrics.get("ndcg_any@10", 0.0)),
    }


def main() -> None:
    args = parse_args()
    args.runs_root = args.runs_root if args.runs_root.is_absolute() else (ROOT / args.runs_root)
    if args.out is not None and not args.out.is_absolute():
        args.out = ROOT / args.out
    data_file = args.data_file if args.data_file.is_absolute() else (ROOT / args.data_file)
    python_exe_raw = str(args.python_exe)
    resolved_python = str(args.python_exe) if Path(args.python_exe).exists() else shutil.which(python_exe_raw)
    if resolved_python is None:
        raise FileNotFoundError(f"Python executable not found: {python_exe_raw}")
    args.python_exe = Path(resolved_python)
    if not data_file.exists():
        raise FileNotFoundError(f"LongMemEval data file not found: {data_file}")

    questions = load_questions(data_file, args.limit)
    chunked_items = split_questions(questions, args.chunks)
    run_dir = build_run_dir(args, len(questions))
    run_dir.mkdir(parents=True, exist_ok=True)
    merged_path = merged_output_path(args, run_dir, len(questions))
    chunk_specs = build_chunk_specs(run_dir, chunked_items, args.mode)

    meta = {
        "data_file": str(data_file),
        "question_count": len(questions),
        "chunks": [
            {
                "chunk": spec.chunk_id,
                "question_count": spec.question_count,
                "input": str(spec.input_path),
                "out": str(spec.output_path),
                "status": "pending",
            }
            for spec in chunk_specs
        ],
        "config": {
            "mode": args.mode,
            "granularity": args.granularity,
            "top_k": args.top_k,
            "rerank_mode": args.rerank_mode,
            "task_type": args.task_type,
            "shell": args.shell,
            "sector": args.sector,
            "zone": args.zone,
            "chunk_pool": args.chunk_pool,
            "object_top_k": args.object_top_k,
            "support_top_k": args.support_top_k,
            "cognitive_top_k": args.cognitive_top_k,
            "cross_encoder": args.cross_encoder,
            "creative_mode_env": os.environ.get("SPHERE_CREATIVE_MODE", "off"),
            "creative_beam_width_env": os.environ.get("SPHERE_CREATIVE_BEAM_WIDTH"),
            "creative_max_hops_env": os.environ.get("SPHERE_CREATIVE_MAX_HOPS"),
            "creative_neighbors_per_hop_env": os.environ.get("SPHERE_CREATIVE_NEIGHBORS_PER_HOP"),
            "creative_max_output_paths_env": os.environ.get("SPHERE_CREATIVE_MAX_OUTPUT_PATHS"),
            "workers": args.workers,
            "max_chunk_retries": args.max_chunk_retries,
            "online": args.online,
            "python_exe": str(args.python_exe),
        },
        "run_dir": str(run_dir),
        "merged_out": str(merged_path),
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    state = RunState(run_dir / "run_meta.json", meta)

    print(f"Run dir: {run_dir}", flush=True)
    print(f"Questions: {len(questions)} split across {len(chunk_specs)} chunk(s)", flush=True)
    print(f"Workers: {min(args.workers, len(chunk_specs))}", flush=True)

    completed_outputs: dict[str, Path] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(chunk_specs))) as executor:
        future_map = {executor.submit(run_chunk, args, run_dir, spec, state): spec for spec in chunk_specs}
        for future in as_completed(future_map):
            spec = future_map[future]
            try:
                completed_outputs[spec.chunk_id] = future.result()
            except Exception as exc:
                failures.append(str(exc))
                state.update_chunk(spec.chunk_id, status="failed")

    if failures:
        state.update_root(status="failed", finished_at=datetime.now().isoformat(timespec="seconds"), failures=failures)
        for failure in failures:
            print(failure, flush=True)
        raise SystemExit(1)

    ordered_outputs = [completed_outputs[spec.chunk_id] for spec in chunk_specs]
    merge_results(args, ordered_outputs, merged_path)
    summary = summarize_merged(merged_path)
    state.update_root(
        status="completed",
        finished_at=datetime.now().isoformat(timespec="seconds"),
        summary=summary,
        merged_out=str(merged_path),
    )

    print("Chunked LongMemEval complete", flush=True)
    print(f"  Merged:     {merged_path}", flush=True)
    print(f"  Questions:  {summary['question_count']}", flush=True)
    print(f"  Recall@5:   {summary['recall_any@5']:.4f}", flush=True)
    print(f"  Recall@10:  {summary['recall_any@10']:.4f}", flush=True)
    print(f"  NDCG@10:    {summary['ndcg_any@10']:.4f}", flush=True)
    print(f"  Elapsed:    {summary['elapsed_seconds']:.1f}s (sum of chunk elapsed_seconds)", flush=True)


if __name__ == "__main__":
    main()
