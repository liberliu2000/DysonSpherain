from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BENCHMARKS = {
    "longmemeval": "longmemeval_benchmark.py",
    "locomo": "locomo_benchmark.py",
    "knowme": "knowme_benchmark.py",
    "clonemem": "clonemem_benchmark.py",
    "convomem": "convomem_benchmark.py",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run available DysonSpherain benchmark adapters with manifest artifacts.")
    parser.add_argument("--data-root", required=True, help="Directory containing benchmark datasets.")
    parser.add_argument("--out", default="benchmark_runs/current", help="Output run directory.")
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS), help="Comma-separated benchmark names.")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow local-hash fallback. Default is fail-fast.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = [name.strip().lower() for name in args.benchmarks.split(",") if name.strip()]
    env = os.environ.copy()
    env.setdefault("SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING", "0")
    env.setdefault("SPHERE_ENABLE_LIGHTWEIGHT_EDGE_WRITEBACK", "1")
    if not args.allow_fallback:
        env["SPHERE_EMBEDDING_FAIL_FAST"] = "1"

    manifest = {
        "created_at": now_iso(),
        "data_root": str(Path(args.data_root).resolve()),
        "out": str(out),
        "allow_fallback": bool(args.allow_fallback),
        "benchmark_route_tuning": env.get("SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING"),
        "runs": [],
    }

    exit_code = 0
    for name in selected:
        script = BENCHMARKS.get(name)
        if not script:
            manifest["runs"].append({"benchmark": name, "status": "unknown_benchmark"})
            exit_code = 2
            continue
        script_path = root / script
        bench_out = out / name
        bench_out.mkdir(parents=True, exist_ok=True)
        data_root = Path(args.data_root).resolve()
        if name == "longmemeval":
            data_arg = data_root / "longmemeval_s_cleaned.json"
        elif name == "locomo":
            data_arg = data_root / "locomo10.json"
        else:
            data_arg = data_root / name
            if not data_arg.exists():
                data_arg = data_root
        if not data_arg.exists():
            manifest["runs"].append({
                "benchmark": name,
                "status": "missing_dataset",
                "expected_path": str(data_arg),
            })
            exit_code = 1
            continue
        cmd = [sys.executable, str(script_path), str(data_arg), "--out", str(bench_out / "metrics.json")]
        started = now_iso()
        try:
            completed = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except Exception as exc:
            manifest["runs"].append({"benchmark": name, "status": "failed_to_start", "error": str(exc), "started_at": started})
            exit_code = 1
            continue
        (bench_out / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (bench_out / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        run = {
            "benchmark": name,
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "started_at": started,
            "finished_at": now_iso(),
            "command": cmd,
            "stdout": str(bench_out / "stdout.txt"),
            "stderr": str(bench_out / "stderr.txt"),
        }
        manifest["runs"].append(run)
        if completed.returncode != 0:
            exit_code = 1
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
