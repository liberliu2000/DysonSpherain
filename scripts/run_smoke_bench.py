from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "base"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from dysonspherain.memory_runtime.config import load_runtime_config


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _fallback_state() -> dict[str, Any]:
    explicit = str(os.environ.get("ALLOW_EMBEDDING_FALLBACK") or "").strip().lower() in {"1", "true", "yes", "on"}
    return {
        "fallback_in_use": explicit,
        "fallback_reason": "ALLOW_EMBEDDING_FALLBACK is enabled" if explicit else "",
    }


RUNNER_SCRIPTS = {
    "longmemeval": ROOT / "base" / "benchmarks" / "longmemeval_benchmark.py",
    "locomo": ROOT / "base" / "benchmarks" / "locomo_benchmark.py",
    "knowme": ROOT / "base" / "benchmarks" / "knowme_benchmark.py",
    "clonemem": ROOT / "base" / "benchmarks" / "clonemem_benchmark.py",
    "convomem": ROOT / "base" / "benchmarks" / "convomem_benchmark.py",
}


def _runner_command(dataset: str, n: int, *, output_dir: Path, data_path: Path | None, runner_script: Path | None, extra_args: list[str]) -> tuple[list[str], Path]:
    script = runner_script or RUNNER_SCRIPTS[dataset]
    metrics_path = output_dir / f"{dataset}_runner_smoke_metrics.json"
    command = [sys.executable, str(script)]
    if data_path is not None:
        command.append(str(data_path))
    command.extend(["--limit", str(max(1, n)), "--out", str(metrics_path)])
    command.extend(extra_args)
    return command, metrics_path


def run_real_runner(dataset: str, n: int, *, output_dir: Path, data_path: Path | None, runner_script: Path | None, extra_args: list[str]) -> dict[str, Any]:
    command, metrics_path = _runner_command(dataset, n, output_dir=output_dir, data_path=data_path, runner_script=runner_script, extra_args=extra_args)
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "command": command,
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def build_report(dataset: str, n: int, *, output_dir: Path, allow_fallback: bool, runner_result: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_runtime_config(ROOT).to_dict()
    fallback = _fallback_state()
    packet = _load_json(ROOT / "data" / "projections" / "latest_context_packet.json")
    audit = _load_json(ROOT / "data" / "projections" / "latest_recall_audit.json")
    index = _load_json(ROOT / "data" / "indexes" / "index_freshness.json")
    report = {
        "status": "ok",
        "dataset": dataset,
        "n": n,
        "embedding_backend": config.get("embedding_backend"),
        "fallback_in_use": fallback["fallback_in_use"],
        "fallback_reason": fallback["fallback_reason"],
        "fallback_allowed": allow_fallback,
        "index_freshness": index or {"status": "unknown", "reason": "index_freshness_report_missing"},
        "context_packet_trace": {
            "packet_id": packet.get("packet_id"),
            "intent": (packet.get("intent") or {}).get("intent_type"),
            "budget_tokens": packet.get("budget_tokens"),
            "used_tokens": packet.get("used_tokens"),
            "compiler_trace": packet.get("compiler_trace") or {},
            "audit": audit,
        },
        "runner": runner_result or {"status": "not_run", "reason": "pass --run-runner to execute the benchmark runner smoke"},
        "command": " ".join(sys.argv),
    }
    if fallback["fallback_in_use"] and not allow_fallback:
        report["status"] = "error"
        report["error"] = "silent_fallback_blocked"
    if runner_result and runner_result.get("status") != "ok":
        report["status"] = "error"
        report["error"] = "runner_smoke_failed"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{dataset}_smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {dataset} Smoke Report",
        "",
        f"- status: `{report['status']}`",
        f"- n: `{n}`",
        f"- embedding_backend: `{report['embedding_backend']}`",
        f"- fallback_in_use: `{report['fallback_in_use']}`",
        f"- index_freshness: `{report['index_freshness'].get('status')}`",
        f"- context_packet: `{report['context_packet_trace'].get('packet_id')}`",
    ]
    if report.get("error"):
        lines.append(f"- error: `{report['error']}`")
    (output_dir / f"{dataset}_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["longmemeval", "locomo", "knowme", "clonemem", "convomem"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output-dir", default="reports/smoke")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--run-pytest-smoke", action="store_true", help="Run focused smoke tests before writing report.")
    parser.add_argument("--run-runner", action="store_true", help="Run the real benchmark runner with --limit N and attach its metrics artifact.")
    parser.add_argument("--data-path", type=Path, default=None, help="Dataset file or directory passed to the benchmark runner.")
    parser.add_argument("--runner-script", type=Path, default=None, help="Override benchmark runner path, mainly for tests.")
    parser.add_argument("--runner-arg", action="append", default=[], help="Extra argument passed through to the benchmark runner. Repeatable.")
    args = parser.parse_args(argv)
    if args.run_pytest_smoke:
        subprocess.run([sys.executable, "-m", "pytest", "tests/memory_runtime", "-q"], cwd=ROOT, check=True)
    output_dir = Path(args.output_dir)
    runner_result = None
    if args.run_runner:
        output_dir.mkdir(parents=True, exist_ok=True)
        runner_result = run_real_runner(args.dataset, args.n, output_dir=output_dir, data_path=args.data_path, runner_script=args.runner_script, extra_args=list(args.runner_arg or []))
    report = build_report(args.dataset, args.n, output_dir=output_dir, allow_fallback=args.allow_fallback, runner_result=runner_result)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if report.get("status") == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
