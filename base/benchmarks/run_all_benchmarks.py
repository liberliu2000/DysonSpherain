from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARKS = {
    "longmemeval": "longmemeval_benchmark.py",
    "locomo": "locomo_benchmark.py",
    "knowme": "knowme_benchmark.py",
    "clonemem": "clonemem_benchmark.py",
    "convomem": "convomem_benchmark.py",
}

BENCHMARK_RUNTIME_RED_LINES_SECONDS = {
    "longmemeval": {"warning": 10 * 60, "fail": 10 * 60, "expected_queries": 500},
    "locomo": {"warning": 45 * 60, "fail": 45 * 60, "expected_queries": 1986},
    "knowme": {"warning": 35 * 60, "fail": 35 * 60, "expected_queries": 1010},
    "clonemem": {"warning": 75 * 60, "fail": 90 * 60, "expected_queries": 2374},
}

SUITE_RUNTIME_RED_LINES_SECONDS = {"warning": int(2.5 * 60 * 60), "fail": int(3.5 * 60 * 60)}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def resolve_data_arg(benchmark: str, data_root: Path) -> Path:
    candidates: list[Path]
    if benchmark == "longmemeval":
        candidates = [
            data_root / "longmemeval_s_cleaned.json",
            data_root / "longmemeval_hf_test" / "longmemeval_s_cleaned.json",
        ]
    elif benchmark == "locomo":
        candidates = [
            data_root / "locomo10.json",
            data_root / "locomo" / "locomo10.json",
        ]
    else:
        candidates = [data_root / benchmark, data_root]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_with_timeout(cmd: list[str], env: dict[str, str], timeout_seconds: float | None) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        stdout = _coerce_output(stdout) or _coerce_output(exc.stdout)
        stderr = _coerce_output(stderr) or _coerce_output(exc.stderr)
    elapsed = time.monotonic() - started
    return {
        "returncode": 124 if timed_out else int(process.returncode or 0),
        "stdout": _coerce_output(stdout),
        "stderr": _coerce_output(stderr),
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "timeout_seconds": timeout_seconds,
    }


def _runtime_redline_for(benchmark: str) -> dict[str, int] | None:
    redline = BENCHMARK_RUNTIME_RED_LINES_SECONDS.get(benchmark)
    return dict(redline) if redline else None


def _format_minutes(seconds: float | int | None) -> float | None:
    if seconds is None:
        return None
    return round(float(seconds) / 60.0, 3)


def _record_token_economy_artifacts(
    *,
    manifest: dict[str, Any],
    out: Path,
    token_economy_output: str | None,
    tokenizer_model: str,
    baseline_types: str,
    modes: str,
    context_token_budget: str,
    recent_k: int,
    low_saving_threshold: float,
    quality_drop_threshold: float,
    evidence_bloat_threshold: float,
    metadata_bloat_threshold: float,
) -> dict[str, Any]:
    from token_economy_support import record_token_economy_for_manifest

    return record_token_economy_for_manifest(
        manifest=manifest,
        out=out,
        token_economy_output=token_economy_output,
        tokenizer_model=tokenizer_model,
        baseline_types=baseline_types,
        modes=modes,
        context_token_budget=context_token_budget,
        recent_k=recent_k,
        low_saving_threshold=low_saving_threshold,
        quality_drop_threshold=quality_drop_threshold,
        evidence_bloat_threshold=evidence_bloat_threshold,
        metadata_bloat_threshold=metadata_bloat_threshold,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run available DysonSpherain benchmark adapters with manifest artifacts.")
    parser.add_argument("--data-root", required=True, help="Directory containing benchmark datasets.")
    parser.add_argument("--out", default="benchmark_runs/current", help="Output run directory.")
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS), help="Comma-separated benchmark names.")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow local-hash fallback. Default is fail-fast.")
    parser.add_argument("--parallel-benchmarks", action="store_true", help="Allow benchmark-level parallelism metadata; chunk workers remain bounded by --workers.")
    parser.add_argument("--chunked", action="store_true", help="Run supported benchmarks via run_benchmark_chunked.py.")
    parser.add_argument("--chunks", type=int, default=4, help="Shard count for --chunked benchmarks.")
    parser.add_argument("--workers", type=int, default=0, help="Total worker budget. Defaults to a conservative half-CPU budget.")
    parser.add_argument("--per-benchmark-workers", type=int, default=0, help="Worker count passed to each chunked benchmark.")
    parser.add_argument("--serial", action="store_true", help="Force benchmark-level serial execution.")
    parser.add_argument("--resume", action="store_true", help="Resume existing successful chunks in --chunked mode.")
    parser.add_argument("--force", action="store_true", help="Force rerun existing chunks in --chunked mode.")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing chunk outputs in --chunked mode.")
    parser.add_argument(
        "--knowme-profile",
        choices=["default", "official_formal"],
        default="default",
        help="KnowMe runtime profile passed to run_benchmark_chunked.py.",
    )
    parser.add_argument(
        "--disable-runtime-redlines",
        action="store_true",
        help="Disable benchmark runtime warning/fail red lines. Defaults to enabled.",
    )
    parser.add_argument("--record-token-economy", action="store_true", help="Record diagnostic token economy artifacts after benchmark runs.")
    parser.add_argument("--token-economy-output", default=None, help="Token economy output directory.")
    parser.add_argument("--tokenizer-model", default="cl100k_base", help="Tokenizer model for token economy counting.")
    parser.add_argument("--token-economy-baseline-types", default="full_history,naive_recent,oracle_minimal")
    parser.add_argument("--token-economy-modes", default="conservative,exploratory")
    parser.add_argument("--context-token-budget", default="2000,4000,8000")
    parser.add_argument("--recent-k", type=int, default=20)
    parser.add_argument("--low-saving-threshold", type=float, default=0.2)
    parser.add_argument("--quality-drop-threshold", type=float, default=0.05)
    parser.add_argument("--evidence-bloat-threshold", type=float, default=0.85)
    parser.add_argument("--metadata-bloat-threshold", type=float, default=0.25)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = [name.strip().lower() for name in args.benchmarks.split(",") if name.strip()]
    env = os.environ.copy()
    env.setdefault("SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING", "1")
    env.setdefault("SPHERE_ENABLE_LIGHTWEIGHT_EDGE_WRITEBACK", "1")
    env.setdefault("SPHERE_CREATIVE_MODE", "off")
    if not args.allow_fallback:
        env["SPHERE_EMBEDDING_FAIL_FAST"] = "1"

    manifest = {
        "created_at": now_iso(),
        "data_root": str(Path(args.data_root).resolve()),
        "out": str(out),
        "allow_fallback": bool(args.allow_fallback),
        "benchmark_route_tuning": env.get("SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING"),
        "chunked": bool(args.chunked),
        "parallel_benchmarks": bool(args.parallel_benchmarks and not args.serial),
        "chunks": int(args.chunks),
        "workers": int(args.workers or 0),
        "per_benchmark_workers": int(args.per_benchmark_workers or 0),
        "knowme_profile": args.knowme_profile,
        "runtime_redlines_enabled": not bool(args.disable_runtime_redlines),
        "runtime_redlines_seconds": {
            "benchmarks": BENCHMARK_RUNTIME_RED_LINES_SECONDS,
            "suite": SUITE_RUNTIME_RED_LINES_SECONDS,
        },
        "token_economy_requested": bool(args.record_token_economy),
        "runs": [],
    }

    exit_code = 0
    suite_started_monotonic = time.monotonic()
    suite_warning_emitted = False
    for name in selected:
        suite_elapsed_before = time.monotonic() - suite_started_monotonic
        if not args.disable_runtime_redlines and suite_elapsed_before >= SUITE_RUNTIME_RED_LINES_SECONDS["fail"]:
            manifest["runs"].append({
                "benchmark": name,
                "status": "skipped_suite_runtime_timeout",
                "suite_elapsed_seconds": round(suite_elapsed_before, 3),
                "suite_fail_seconds": SUITE_RUNTIME_RED_LINES_SECONDS["fail"],
            })
            exit_code = 1
            break
        script = BENCHMARKS.get(name)
        if not script:
            manifest["runs"].append({"benchmark": name, "status": "unknown_benchmark"})
            exit_code = 2
            continue
        script_path = root / script
        bench_out = out / name
        bench_out.mkdir(parents=True, exist_ok=True)
        data_root = Path(args.data_root).resolve()
        data_arg = resolve_data_arg(name, data_root)
        if not data_arg.exists():
            manifest["runs"].append({
                "benchmark": name,
                "status": "missing_dataset",
                "expected_path": str(data_arg),
            })
            exit_code = 1
            continue
        chunked_supported = {"longmemeval", "locomo", "knowme", "clonemem"}
        if args.chunked and name in chunked_supported:
            chunk_workers = int(args.per_benchmark_workers or args.workers or 0)
            if chunk_workers <= 0:
                chunk_workers = min(max(1, (os.cpu_count() or 2) // 2), int(args.chunks))
            if args.parallel_benchmarks and not args.serial and args.workers:
                chunk_workers = max(1, min(chunk_workers, int(args.workers)))
            cmd = [
                sys.executable,
                str(root / "run_benchmark_chunked.py"),
                "--benchmark",
                name,
                "--data-root",
                str(Path(args.data_root).resolve()),
                "--out",
                str(bench_out),
                "--chunks",
                str(args.chunks),
                "--workers",
                str(chunk_workers),
            ]
            if args.resume:
                cmd.append("--resume")
            if args.force:
                cmd.append("--force")
            if args.merge_only:
                cmd.append("--merge-only")
            if name == "knowme":
                cmd.extend(["--knowme-profile", args.knowme_profile])
        else:
            cmd = [sys.executable, str(script_path), str(data_arg), "--out", str(bench_out / "metrics.json")]
        started = now_iso()
        try:
            redline = None if args.disable_runtime_redlines else _runtime_redline_for(name)
            suite_remaining_fail = None
            if not args.disable_runtime_redlines:
                suite_remaining_fail = max(1.0, SUITE_RUNTIME_RED_LINES_SECONDS["fail"] - suite_elapsed_before)
            timeout_candidates = [
                float(redline["fail"]) for redline in [redline] if redline and redline.get("fail")
            ]
            if suite_remaining_fail is not None:
                timeout_candidates.append(float(suite_remaining_fail))
            timeout_seconds = min(timeout_candidates) if timeout_candidates else None
            completed = _run_with_timeout(cmd, env=env, timeout_seconds=timeout_seconds)
        except Exception as exc:
            manifest["runs"].append({"benchmark": name, "status": "failed_to_start", "error": str(exc), "started_at": started})
            exit_code = 1
            continue
        (bench_out / "stdout.txt").write_text(completed["stdout"], encoding="utf-8")
        (bench_out / "stderr.txt").write_text(completed["stderr"], encoding="utf-8")
        elapsed_seconds = float(completed["elapsed_seconds"])
        redline = None if args.disable_runtime_redlines else _runtime_redline_for(name)
        runtime_redline = {
            "enabled": not bool(args.disable_runtime_redlines),
            "benchmark_warning_seconds": redline.get("warning") if redline else None,
            "benchmark_fail_seconds": redline.get("fail") if redline else None,
            "benchmark_warning_minutes": _format_minutes(redline.get("warning") if redline else None),
            "benchmark_fail_minutes": _format_minutes(redline.get("fail") if redline else None),
            "expected_queries": redline.get("expected_queries") if redline else None,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "elapsed_minutes": _format_minutes(elapsed_seconds),
            "warning_exceeded": bool(redline and elapsed_seconds > float(redline["warning"])),
            "fail_exceeded": bool(redline and elapsed_seconds > float(redline["fail"])),
            "timeout_seconds": completed["timeout_seconds"],
            "timed_out": bool(completed["timed_out"]),
        }
        suite_elapsed_after = time.monotonic() - suite_started_monotonic
        suite_redline = {
            "elapsed_seconds": round(suite_elapsed_after, 3),
            "elapsed_minutes": _format_minutes(suite_elapsed_after),
            "warning_seconds": SUITE_RUNTIME_RED_LINES_SECONDS["warning"],
            "fail_seconds": SUITE_RUNTIME_RED_LINES_SECONDS["fail"],
            "warning_minutes": _format_minutes(SUITE_RUNTIME_RED_LINES_SECONDS["warning"]),
            "fail_minutes": _format_minutes(SUITE_RUNTIME_RED_LINES_SECONDS["fail"]),
            "warning_exceeded": bool(
                not args.disable_runtime_redlines
                and suite_elapsed_after > SUITE_RUNTIME_RED_LINES_SECONDS["warning"]
            ),
            "fail_exceeded": bool(
                not args.disable_runtime_redlines
                and suite_elapsed_after > SUITE_RUNTIME_RED_LINES_SECONDS["fail"]
            ),
        }
        if suite_redline["warning_exceeded"]:
            suite_warning_emitted = True
        status = "ok" if completed["returncode"] == 0 else "failed"
        if runtime_redline["timed_out"] or runtime_redline["fail_exceeded"] or suite_redline["fail_exceeded"]:
            status = "runtime_timeout"
        run = {
            "benchmark": name,
            "status": status,
            "returncode": completed["returncode"],
            "started_at": started,
            "finished_at": now_iso(),
            "command": cmd,
            "stdout": str(bench_out / "stdout.txt"),
            "stderr": str(bench_out / "stderr.txt"),
            "chunked": bool(args.chunked and name in chunked_supported),
            "shard_count": int(args.chunks) if args.chunked else None,
            "worker_count": int(args.per_benchmark_workers or args.workers or 0) if args.chunked else None,
            "runtime_redline": runtime_redline,
            "suite_runtime_redline": suite_redline,
        }
        chunk_manifest = bench_out / "run_manifest.json"
        if chunk_manifest.exists():
            try:
                chunk_payload = json.loads(chunk_manifest.read_text(encoding="utf-8"))
                run["chunk_manifest"] = str(chunk_manifest)
                run["benchmark_profile"] = chunk_payload.get("benchmark_profile")
                run["wall_clock_elapsed_seconds"] = chunk_payload.get("wall_clock_elapsed_seconds")
                run["successful_chunks"] = chunk_payload.get("successful_chunks")
                run["failed_chunks"] = chunk_payload.get("failed_chunks")
                run["merged_metrics"] = chunk_payload.get("merged_metrics")
                if chunk_payload.get("merged_metrics"):
                    merged_path = Path(str(chunk_payload.get("merged_metrics")))
                    if merged_path.exists():
                        merged_payload = json.loads(merged_path.read_text(encoding="utf-8"))
                        run["serial_equivalent_elapsed_seconds"] = merged_payload.get("serial_elapsed_seconds_sum")
                        run["speedup_estimate"] = merged_payload.get("speedup_estimate")
            except Exception as exc:
                run["chunk_manifest_error"] = str(exc)
        manifest["runs"].append(run)
        if runtime_redline["warning_exceeded"] and completed["returncode"] == 0:
            print(
                f"[runtime-warning] {name} elapsed {runtime_redline['elapsed_minutes']} min "
                f"> warning {runtime_redline['benchmark_warning_minutes']} min",
                file=sys.stderr,
            )
        if completed["returncode"] != 0 or status == "runtime_timeout":
            exit_code = 1
        if status == "runtime_timeout" or suite_redline["fail_exceeded"]:
            break
    suite_elapsed_total = time.monotonic() - suite_started_monotonic
    manifest["suite_runtime_redline"] = {
        "elapsed_seconds": round(suite_elapsed_total, 3),
        "elapsed_minutes": _format_minutes(suite_elapsed_total),
        "warning_seconds": SUITE_RUNTIME_RED_LINES_SECONDS["warning"],
        "fail_seconds": SUITE_RUNTIME_RED_LINES_SECONDS["fail"],
        "warning_minutes": _format_minutes(SUITE_RUNTIME_RED_LINES_SECONDS["warning"]),
        "fail_minutes": _format_minutes(SUITE_RUNTIME_RED_LINES_SECONDS["fail"]),
        "warning_exceeded": bool(
            not args.disable_runtime_redlines
            and suite_elapsed_total > SUITE_RUNTIME_RED_LINES_SECONDS["warning"]
        ),
        "fail_exceeded": bool(
            not args.disable_runtime_redlines
            and suite_elapsed_total > SUITE_RUNTIME_RED_LINES_SECONDS["fail"]
        ),
        "warning_emitted": suite_warning_emitted,
    }
    serial_equivalent = 0.0
    for run in manifest["runs"]:
        serial_equivalent += float(run.get("serial_equivalent_elapsed_seconds") or (run.get("runtime_redline") or {}).get("elapsed_seconds") or 0.0)
    manifest["serial_equivalent_elapsed_seconds"] = round(serial_equivalent, 3)
    manifest["wall_clock_elapsed_seconds"] = round(suite_elapsed_total, 3)
    manifest["speedup_estimate"] = round(serial_equivalent / suite_elapsed_total, 4) if suite_elapsed_total > 0 else 0.0
    if manifest["suite_runtime_redline"]["fail_exceeded"]:
        exit_code = 1
    if args.record_token_economy:
        manifest["token_economy"] = _record_token_economy_artifacts(
            manifest=manifest,
            out=out,
            token_economy_output=args.token_economy_output,
            tokenizer_model=args.tokenizer_model,
            baseline_types=args.token_economy_baseline_types,
            modes=args.token_economy_modes,
            context_token_budget=args.context_token_budget,
            recent_k=args.recent_k,
            low_saving_threshold=args.low_saving_threshold,
            quality_drop_threshold=args.quality_drop_threshold,
            evidence_bloat_threshold=args.evidence_bloat_threshold,
            metadata_bloat_threshold=args.metadata_bloat_threshold,
        )
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
