#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT.parent / "BenchmarkResult"
CHUNKED = ROOT / "base" / "benchmarks" / "run_benchmark_chunked.py"
SWEEPS = (
    ("candidate", 50),
    ("candidate", 100),
    ("candidate", 200),
    ("candidate", 500),
    ("rerank", 20),
    ("rerank", 50),
    ("rerank", 100),
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def default_python() -> str:
    venv_python = ROOT / ".venv312" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def sweep_name(sweep_type: str, budget: int) -> str:
    prefix = "candidate_top" if sweep_type == "candidate" else "rerank_top"
    return f"{prefix}{budget}"


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_command(args: argparse.Namespace, sweep_type: str, budget: int, out_dir: Path) -> list[str]:
    top_k = budget if sweep_type == "candidate" else max(50, budget)
    command = [
        sys.executable,
        str(CHUNKED),
        "--benchmark",
        args.benchmark,
        "--data-root",
        str(args.data_root),
        "--out",
        str(out_dir),
        "--chunks",
        str(args.chunks),
        "--workers",
        str(args.workers),
        "--python-exe",
        str(args.python_exe),
        "--mode",
        "evidence",
        "--top-k",
        str(top_k),
        "--rerank-mode",
        "rule",
        "--max-questions",
        str(args.max_questions),
        "--force",
    ]
    if args.benchmark == "clonemem":
        command.extend(["--context-len", args.context_len, "--language", args.language])
    if args.benchmark in {"longmemeval", "locomo"}:
        command.extend(["--granularity", args.granularity])
    return command


def base_payload(args: argparse.Namespace, sweep_type: str, budget: int, command: list[str]) -> dict[str, Any]:
    name = sweep_name(sweep_type, budget)
    run_scope = "smoke" if args.mode == "metadata-smoke" else args.run_scope
    return {
        "schema": "dysonspherain.efficiency_budget_sweep.v1",
        "benchmark": args.benchmark,
        "sweep_name": name,
        "sweep_type": sweep_type,
        "budget": budget,
        "run_type": "efficiency_sweep",
        "run_scope": run_scope,
        "mode": "evidence",
        "rerank_mode": "rule",
        "question_count": 0,
        "total_question_count": 0,
        "elapsed_seconds": 0.0,
        "wall_clock_elapsed_seconds": 0.0,
        "fallback_in_use": False,
        "formal_eligible": bool(args.formal_eligible) and args.mode == "execute",
        "formal_use_warning": "Metadata smoke verifies the sweep artifact contract only; do not use as formal efficiency evidence.",
        "config_hash": stable_hash(
            {
                "benchmark": args.benchmark,
                "sweep_type": sweep_type,
                "budget": budget,
                "mode": "evidence",
                "rerank_mode": "rule",
                "run_scope": run_scope,
            }
        ),
        "dataset_version": args.dataset_version,
        "command": " ".join(shlex.quote(part) for part in command),
        "created_at": now_iso(),
        "metrics": {},
    }


def write_metadata_smoke(args: argparse.Namespace, sweep_type: str, budget: int, sweep_dir: Path) -> dict[str, Any]:
    command = build_command(args, sweep_type, budget, sweep_dir)
    payload = base_payload(args, sweep_type, budget, command)
    payload.update(
        {
            "status": "available",
            "smoke_only": True,
            "metadata_contract": {
                "required_fields": [
                    "sweep_type",
                    "budget",
                    "benchmark",
                    "question_count",
                    "config_hash",
                    "dataset_version",
                    "fallback_in_use",
                    "command",
                ],
                "validated": True,
            },
        }
    )
    metrics_path = sweep_dir / args.benchmark / "metrics.json"
    write_json(metrics_path, payload)
    write_json(
        sweep_dir / "run_manifest.json",
        {
            "schema": "dysonspherain.efficiency_budget_sweep_manifest.v1",
            "status": "metadata_smoke_completed",
            "sweep_name": payload["sweep_name"],
            "benchmark": args.benchmark,
            "metrics": str(metrics_path),
            "command": command,
            "created_at": payload["created_at"],
        },
    )
    return {"sweep": payload["sweep_name"], "status": "metadata_smoke_completed", "metrics": str(metrics_path)}


def run_execute(args: argparse.Namespace, sweep_type: str, budget: int, sweep_dir: Path) -> dict[str, Any]:
    command = build_command(args, sweep_type, budget, sweep_dir)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, text=True, check=False)
    elapsed = round(time.monotonic() - started, 3)
    merged = sweep_dir / args.benchmark / "merged_metrics.json"
    status = "completed" if completed.returncode == 0 and merged.exists() else "failed"
    if merged.exists():
        source_payload = json.loads(merged.read_text(encoding="utf-8"))
        payload = {**source_payload, **base_payload(args, sweep_type, budget, command)}
        payload["status"] = "available" if status == "completed" else status
        payload["question_count"] = source_payload.get("question_count") or source_payload.get("total_question_count") or 0
        payload["total_question_count"] = source_payload.get("total_question_count") or source_payload.get("question_count") or 0
        payload["elapsed_seconds"] = source_payload.get("elapsed_seconds") or elapsed
        payload["wall_clock_elapsed_seconds"] = source_payload.get("wall_clock_elapsed_seconds") or elapsed
        write_json(merged, payload)
    return {"sweep": sweep_name(sweep_type, budget), "status": status, "metrics": str(merged), "elapsed_seconds": elapsed}


def write_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Efficiency Budget Sweep Smoke",
        "",
        "| sweep | status | metrics | elapsed_s |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row.get('sweep')} | {row.get('status')} | {row.get('metrics')} | {row.get('elapsed_seconds', '')} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or smoke-test dedicated Phase 13 efficiency budget sweeps.")
    parser.add_argument("--mode", choices=["metadata-smoke", "execute"], default="metadata-smoke")
    parser.add_argument("--benchmark", choices=["longmemeval", "locomo", "knowme", "clonemem"], default="longmemeval")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "benchmark")
    parser.add_argument("--out-root", type=Path, default=RESULT_ROOT / "20260429_phase13_efficiency_sweep_smoke_v1")
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python-exe", default=default_python())
    parser.add_argument("--context-len", choices=["100k", "500k", "all"], default="100k")
    parser.add_argument("--language", choices=["en", "zh", "all"], default="en")
    parser.add_argument("--granularity", choices=["session", "dialog", "turn"], default="session")
    parser.add_argument("--dataset-version", default="metadata-smoke")
    parser.add_argument("--run-scope", choices=["smoke", "matched_medium", "full"], default="smoke")
    parser.add_argument("--formal-eligible", action="store_true", help="Mark executed artifacts as eligible for formal efficiency evidence.")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "phase13_efficiency_sweep_smoke.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for sweep_type, budget in SWEEPS:
        name = sweep_name(sweep_type, budget)
        sweep_dir = args.out_root / name
        if args.mode == "metadata-smoke":
            rows.append(write_metadata_smoke(args, sweep_type, budget, sweep_dir))
        else:
            rows.append(run_execute(args, sweep_type, budget, sweep_dir))
    write_json(args.out_root / "sweep_summary.json", {"schema": "dysonspherain.efficiency_budget_sweep_summary.v1", "rows": rows})
    write_report(rows, args.report)
    print(json.dumps({"rows": len(rows), "out_root": str(args.out_root), "report": str(args.report)}, ensure_ascii=False))
    return 0 if all(row.get("status") in {"metadata_smoke_completed", "completed"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
