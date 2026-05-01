from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def add_token_economy_args(parser: Any) -> None:
    parser.add_argument("--record-token-economy", action="store_true", help="Record diagnostic token economy artifacts after the benchmark run.")
    parser.add_argument("--token-economy-output", type=Path, default=None, help="Token economy artifact directory. Defaults to the benchmark output directory.")
    parser.add_argument("--tokenizer-model", default="cl100k_base", help="Tokenizer model for token economy counting.")
    parser.add_argument("--token-economy-baseline-types", default="full_history,naive_recent,oracle_minimal")
    parser.add_argument("--token-economy-modes", default="conservative,exploratory")
    parser.add_argument("--context-token-budget", default="2000,4000,8000")
    parser.add_argument("--recent-k", type=int, default=20)
    parser.add_argument("--low-saving-threshold", type=float, default=0.2)
    parser.add_argument("--quality-drop-threshold", type=float, default=0.05)
    parser.add_argument("--evidence-bloat-threshold", type=float, default=0.85)
    parser.add_argument("--metadata-bloat-threshold", type=float, default=0.25)


def _build_samples(
    *,
    payloads: list[dict[str, Any]],
    tokenizer_model: str,
    baseline_types: str,
    modes: str,
    context_token_budget: str,
    recent_k: int,
) -> list[Any]:
    from dysonspherain.evaluation.token_economy import _sample_from_payload, _split_csv, _split_int_csv
    from dysonspherain.utils.token_counter import TokenCounter

    counter = TokenCounter(tokenizer_model)
    samples = []
    for mode_name in _split_csv(modes):
        for baseline_type in _split_csv(baseline_types):
            for budget in _split_int_csv(context_token_budget):
                for index, payload in enumerate(payloads):
                    samples.append(
                        _sample_from_payload(
                            payload,
                            index=index,
                            mode=mode_name,
                            baseline_type=baseline_type,
                            budget=budget,
                            recent_k=recent_k,
                            counter=counter,
                            allow_evidence_truncation=False,
                        )
                    )
    return samples


def record_token_economy_for_metrics(
    *,
    metrics_path: Path,
    output_dir: Path | None = None,
    tokenizer_model: str = "cl100k_base",
    baseline_types: str = "full_history,naive_recent,oracle_minimal",
    modes: str = "conservative,exploratory",
    context_token_budget: str = "2000,4000,8000",
    recent_k: int = 20,
    low_saving_threshold: float = 0.2,
    quality_drop_threshold: float = 0.05,
    evidence_bloat_threshold: float = 0.85,
    metadata_bloat_threshold: float = 0.25,
) -> dict[str, Any]:
    from dysonspherain.token_economy.artifact_inputs import payloads_from_benchmark_metrics
    from dysonspherain.token_economy.report import write_report

    metrics_path = Path(metrics_path)
    artifact_dir = Path(output_dir) if output_dir is not None else metrics_path.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payloads = payloads_from_benchmark_metrics(metrics_path) if metrics_path.exists() else []
    samples = _build_samples(
        payloads=payloads,
        tokenizer_model=tokenizer_model,
        baseline_types=baseline_types,
        modes=modes,
        context_token_budget=context_token_budget,
        recent_k=recent_k,
    )
    summary = write_report(
        samples,
        artifact_dir,
        filename_prefix="token_economy_",
        low_saving_threshold=low_saving_threshold,
        quality_drop_threshold=quality_drop_threshold,
        evidence_bloat_threshold=evidence_bloat_threshold,
        metadata_bloat_threshold=metadata_bloat_threshold,
    )
    result = {
        "artifact_dir": str(artifact_dir),
        "metrics_path": str(metrics_path),
        "summary": summary,
        "files": {
            "per_sample": str(artifact_dir / "token_economy_per_sample.jsonl"),
            "summary_json": str(artifact_dir / "token_economy_summary.json"),
            "summary_md": str(artifact_dir / "token_economy_summary.md"),
            "mode_comparison": str(artifact_dir / "token_economy_mode_comparison.csv"),
            "token_quality_tradeoff": str(artifact_dir / "token_economy_token_quality_tradeoff.csv"),
            "failure_cases": str(artifact_dir / "token_economy_failure_cases.json"),
        },
        "diagnostic_only": True,
        "thresholds": {
            "low_saving_threshold": low_saving_threshold,
            "quality_drop_threshold": quality_drop_threshold,
            "evidence_bloat_threshold": evidence_bloat_threshold,
            "metadata_bloat_threshold": metadata_bloat_threshold,
        },
    }
    (artifact_dir / "token_economy_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def record_token_economy_for_manifest(
    *,
    manifest: dict[str, Any],
    out: Path,
    token_economy_output: str | None,
    tokenizer_model: str,
    baseline_types: str,
    modes: str,
    context_token_budget: str,
    recent_k: int,
    low_saving_threshold: float = 0.2,
    quality_drop_threshold: float = 0.05,
    evidence_bloat_threshold: float = 0.85,
    metadata_bloat_threshold: float = 0.25,
) -> dict[str, Any]:
    from dysonspherain.token_economy.artifact_inputs import payloads_from_benchmark_metrics
    from dysonspherain.token_economy.report import write_report

    out = Path(out)
    payloads: list[dict[str, Any]] = []
    per_benchmark: dict[str, dict[str, Any]] = {}
    for run in manifest.get("runs") or []:
        bench = str(run.get("benchmark") or "")
        bench_dir = out / bench
        metrics_path = Path(str(run.get("merged_metrics") or bench_dir / "metrics.json"))
        if metrics_path.exists():
            bench_payloads = payloads_from_benchmark_metrics(metrics_path)
            payloads.extend(bench_payloads)
            per_benchmark[bench] = {"payloads": bench_payloads, "artifact_dir": str(bench_dir), "metrics_path": str(metrics_path)}
            continue
        stdout_text = (bench_dir / "stdout.txt").read_text(encoding="utf-8", errors="replace")[:4000] if (bench_dir / "stdout.txt").exists() else ""
        bench_payloads = [
            {
                "sample_id": bench or f"run_{len(payloads)}",
                "query": f"Token economy diagnostic for {bench} benchmark run.",
                "history": stdout_text,
                "retrieved_context": json.dumps({"benchmark": bench, "status": run.get("status"), "artifact": str(metrics_path)}, ensure_ascii=False, sort_keys=True),
                "metadata": json.dumps({"command": run.get("command"), "status": run.get("status")}, ensure_ascii=False),
                "candidate_count": 1,
                "final_context_item_count": 1,
            }
        ]
        payloads.extend(bench_payloads)
        per_benchmark[bench] = {"payloads": bench_payloads, "artifact_dir": str(bench_dir), "metrics_path": str(metrics_path), "fallback_payload": True}

    benchmark_reports: dict[str, Any] = {}
    for bench, payload in per_benchmark.items():
        bench_dir = Path(str(payload["artifact_dir"]))
        bench_dir.mkdir(parents=True, exist_ok=True)
        bench_samples = _build_samples(
            payloads=list(payload.get("payloads") or []),
            tokenizer_model=tokenizer_model,
            baseline_types=baseline_types,
            modes=modes,
            context_token_budget=context_token_budget,
            recent_k=recent_k,
        )
        bench_summary = write_report(
            bench_samples,
            bench_dir,
            filename_prefix="token_economy_",
            low_saving_threshold=low_saving_threshold,
            quality_drop_threshold=quality_drop_threshold,
            evidence_bloat_threshold=evidence_bloat_threshold,
            metadata_bloat_threshold=metadata_bloat_threshold,
        )
        benchmark_reports[bench] = {
            "artifact_dir": str(bench_dir),
            "metrics_path": payload.get("metrics_path"),
            "summary": bench_summary,
            "files": {
                "per_sample": str(bench_dir / "token_economy_per_sample.jsonl"),
                "summary_json": str(bench_dir / "token_economy_summary.json"),
                "summary_md": str(bench_dir / "token_economy_summary.md"),
                "mode_comparison": str(bench_dir / "token_economy_mode_comparison.csv"),
                "token_quality_tradeoff": str(bench_dir / "token_economy_token_quality_tradeoff.csv"),
                "failure_cases": str(bench_dir / "token_economy_failure_cases.json"),
            },
            "diagnostic_only": True,
            "thresholds": {
                "low_saving_threshold": low_saving_threshold,
                "quality_drop_threshold": quality_drop_threshold,
                "evidence_bloat_threshold": evidence_bloat_threshold,
                "metadata_bloat_threshold": metadata_bloat_threshold,
            },
        }

    te_dir = Path(token_economy_output) if token_economy_output else out / "token_economy"
    samples = _build_samples(
        payloads=payloads,
        tokenizer_model=tokenizer_model,
        baseline_types=baseline_types,
        modes=modes,
        context_token_budget=context_token_budget,
        recent_k=recent_k,
    )
    summary = write_report(
        samples,
        te_dir,
        low_saving_threshold=low_saving_threshold,
        quality_drop_threshold=quality_drop_threshold,
        evidence_bloat_threshold=evidence_bloat_threshold,
        metadata_bloat_threshold=metadata_bloat_threshold,
    )
    return {
        "artifact_dir": str(te_dir),
        "summary": summary,
        "benchmarks": benchmark_reports,
        "diagnostic_only": True,
        "thresholds": {
            "low_saving_threshold": low_saving_threshold,
            "quality_drop_threshold": quality_drop_threshold,
            "evidence_bloat_threshold": evidence_bloat_threshold,
            "metadata_bloat_threshold": metadata_bloat_threshold,
        },
    }
