from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import platform
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "benchmark_outputs"
REPORT_PATH = PROJECT_ROOT / "sphere_memory_cli_benchmark_report.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(values[int(k)], 2)
    return round(values[f] * (c - k) + values[c] * (k - f), 2)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    def fmt(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def powershell_text(command: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return out.strip() or None


def detect_environment() -> dict:
    ram_bytes = powershell_text(
        "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory"
    )
    cpu_name = powershell_text(
        "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
    )
    requirements = (
        (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if (REPO_ROOT / "requirements.txt").exists()
        else []
    )
    ram_gb = None
    if ram_bytes and ram_bytes.isdigit():
        ram_gb = round(int(ram_bytes) / (1024**3), 2)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "repo_root": str(REPO_ROOT),
        "git_available": (REPO_ROOT / ".git").exists(),
        "python_version": platform.python_version(),
        "python_executable": str(Path(__import__("sys").executable)),
        "platform": platform.platform(),
        "cpu": cpu_name or platform.processor(),
        "ram_gb": ram_gb,
        "requirements_txt": requirements,
        "sentence_transformers_installed": importlib.util.find_spec("sentence_transformers")
        is not None,
    }


def load_prism_smoke(name: str) -> dict:
    return read_json(REPO_ROOT / "evaluation" / "runs" / name / "prism_smoke_summary.json")


def corrected_round3_summary() -> tuple[dict, dict]:
    summary = load_prism_smoke("prism_smoke_round3")
    corrected = copy.deepcopy(summary)
    factual_debug = {}
    run_dir = REPO_ROOT / "evaluation" / "runs" / "prism_smoke_round3"
    for mode in ("off", "conservative", "exploratory"):
        report = read_json(run_dir / f"real_task_eval_prism_{mode}.json")
        factual_cases = []
        for case in report["cases"]:
            task_id = case.get("task_id", "")
            if not (task_id.startswith("exact_") or task_id.startswith("temporal_")):
                continue
            prism = ((case.get("diagnostics") or {}).get("prism_metrics") or {})
            factual_cases.append(
                {
                    "task_id": task_id,
                    "reason": prism.get("reason"),
                    "creative_reflections": len(
                        (case.get("diagnostics") or {}).get("creative_reflections") or []
                    ),
                    "alternative_paths": len(
                        (case.get("diagnostics") or {}).get("alternative_paths") or []
                    ),
                    "factual_creative_leakage": bool(prism.get("factual_creative_leakage")),
                    "primary_evidence_contamination": bool(
                        prism.get("primary_evidence_contamination")
                    ),
                }
            )
        factual_debug[mode] = factual_cases
        leakage_rate = 0.0
        contamination_rate = 0.0
        if factual_cases:
            leakage_rate = sum(
                1 for case in factual_cases if case["factual_creative_leakage"]
            ) / len(factual_cases)
            contamination_rate = sum(
                1 for case in factual_cases if case["primary_evidence_contamination"]
            ) / len(factual_cases)
        corrected["modes"][mode]["prism_quality"]["factual_creative_leakage_rate"] = round(
            leakage_rate, 4
        )
        corrected["modes"][mode]["prism_quality"][
            "primary_evidence_contamination_rate"
        ] = round(contamination_rate, 4)
        for row in corrected.get("quality_rows", []):
            if row.get("mode") == mode:
                row["factual_creative_leakage_rate"] = round(leakage_rate, 4)
                row["primary_evidence_contamination_rate"] = round(contamination_rate, 4)
    corrected["selection_note"] = (
        "Round3 is used as the reference smoke because it is the last comparable "
        "sentence-transformer run after the latency-control fixes. Factual contamination "
        "rates are recomputed from raw case diagnostics using the final factual-only scope."
    )
    return corrected, factual_debug


def summarize_longmemeval(run_file: Path, mode: str, run_name: str, note: str) -> dict:
    data = read_json(run_file)
    results = data["results"]
    total_ms = [item["stage_timing_ms"].get("question_total_ms", 0.0) for item in results]
    retrieval_ms = [item["stage_timing_ms"].get("retrieval_ms", 0.0) for item in results]
    completion_ms = [item["stage_timing_ms"].get("completion_ms", 0.0) for item in results]
    prism_ms = [
        item["stage_timing_ms"].get("cognitive_prism_total_ms", 0.0) for item in results
    ]
    reasons: dict[str, int] = {}
    enabled_cases = 0
    for item in results:
        prism = (((item.get("profiling") or {}).get("pipeline") or {}).get("cognitive") or {}).get(
            "prism"
        ) or {}
        if prism.get("enabled"):
            enabled_cases += 1
        reason = prism.get("reason")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    session_metrics = data["metrics"]["session"]
    return {
        "suite": "longmemeval",
        "run_name": run_name,
        "mode": mode,
        "question_count": len(results),
        "recall_at_5": session_metrics.get("recall_any@5"),
        "recall_at_10": session_metrics.get("recall_any@10"),
        "ndcg_at_10": session_metrics.get("ndcg_any@10"),
        "p50_question_total_ms": percentile(total_ms, 0.5),
        "p95_question_total_ms": percentile(total_ms, 0.95),
        "retrieval_p50_ms": percentile(retrieval_ms, 0.5),
        "completion_p50_ms": percentile(completion_ms, 0.5),
        "prism_p50_ms": percentile(prism_ms, 0.5),
        "enabled_cases": enabled_cases,
        "reason_distribution": reasons,
        "note": note,
    }
