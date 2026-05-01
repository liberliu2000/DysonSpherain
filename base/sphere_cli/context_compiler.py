from __future__ import annotations

from pathlib import Path

from .experiment_registry import latest_run, load_registry
from .memory_lifecycle import load_conflicts
from .project_state import load_project_state, render_project_state_markdown
from .security import redact_secrets


CONTEXT_MODES = {"general", "codex", "paper", "benchmark", "debug", "docs", "project"}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _trim_section(text: str, token_budget: int) -> str:
    char_budget = max(100, int(token_budget) * 4)
    if len(text) <= char_budget:
        return text
    return text[: char_budget - 32].rstrip() + "\n...[truncated]\n"


def _latest_runs_section(base_dir: Path, project: str) -> str:
    runs = load_registry(base_dir)
    lines = ["## Benchmark and Experiment Status", ""]
    for dataset in ("longmemeval", "locomo", "knowme", "clonemem", "convomem"):
        run = latest_run(runs, project=project, dataset=dataset)
        if run is None:
            lines.append(f"- `{dataset}`: missing")
            continue
        metrics = ", ".join(f"{key}={value}" for key, value in sorted(run.metrics.items())[:6])
        lines.append(
            f"- `{dataset}`: run_type=`{run.run_type}` q=`{run.question_count}` "
            f"fallback=`{run.fallback_in_use}` elapsed=`{run.elapsed_seconds}` metrics: {metrics}"
        )
    return "\n".join(lines) + "\n"


def _conflicts_section(base_dir: Path) -> str:
    conflicts = load_conflicts(base_dir)
    lines = ["## Conflict Warnings", ""]
    if not conflicts:
        lines.append("- none recorded")
    for conflict in conflicts[:20]:
        lines.append(
            f"- `{conflict.conflict_type}` `{conflict.dataset}` id=`{conflict.conflict_id}` "
            f"winner=`{conflict.recommended_winner}`: {conflict.reason}"
        )
    return "\n".join(lines) + "\n"


def _mode_title(mode: str) -> str:
    if mode == "paper":
        return "Paper Context Packet"
    if mode == "benchmark":
        return "Benchmark Context Packet"
    return "Runtime Context Packet"


def compile_context_packet(
    base_dir: Path,
    *,
    task: str,
    project: str,
    mode: str = "codex",
    max_tokens: int = 8000,
) -> str:
    if mode not in CONTEXT_MODES:
        raise ValueError(f"Unsupported context mode: {mode}")
    state = load_project_state(base_dir, project)
    state_md = render_project_state_markdown(state)
    benchmark_md = _latest_runs_section(base_dir, project)
    conflicts_md = _conflicts_section(base_dir)
    budget = {
        "project_state": int(max_tokens * 0.20),
        "benchmark": int(max_tokens * 0.18),
        "conflicts": int(max_tokens * 0.10),
        "task": int(max_tokens * 0.08),
    }
    lines = [
        f"# {_mode_title(mode)}",
        "",
        "## Task Objective",
        "",
        _trim_section(task, budget["task"]),
        "",
        "## Current Project State",
        "",
        _trim_section(state_md, budget["project_state"]),
        "",
    ]
    if mode == "paper":
        lines.extend(["## Validated Evidence", "", _trim_section(benchmark_md, budget["benchmark"]), ""])
        lines.extend(["## Limitations and Risks", "", _trim_section(conflicts_md, budget["conflicts"]), ""])
        lines.extend(["## Figure/Table Suggestions", "", "- Use artifact-backed benchmark tables only.", ""])
    elif mode == "benchmark":
        lines.extend([_trim_section(benchmark_md, budget["benchmark"]), ""])
        lines.extend([_trim_section(conflicts_md, budget["conflicts"]), ""])
        lines.extend(["## Suggested Next Diagnostics", "", "- Compare candidate_recall@100, oracle_recall@10, and failure buckets before promoting speed changes.", ""])
    else:
        lines.extend(["## Core Evidence", "", _trim_section(benchmark_md, budget["benchmark"]), ""])
        lines.extend(["## Decision Trace", "", "- Follow the standing iteration protocol in the unified implementation plan.", ""])
        lines.extend(["## Active Constraints", ""])
        lines.extend(f"- {item}" for item in state.constraints)
        lines.extend(["", "## Relevant Files and Artifacts", ""])
        lines.extend(f"- `{item}`" for item in state.relevant_artifacts)
        lines.extend(["", _trim_section(conflicts_md, budget["conflicts"]), ""])
        lines.extend(["## Suggested Codex Execution Plan", "", "- Diagnose from artifacts first; make the smallest reversible change; validate with comparable tests and benchmark slices.", ""])
        lines.extend(["## Acceptance Criteria", "", "- No Recall/NDCG/candidate_recall regression; no fallback contamination; diagnostics updated.", ""])
    packet = "\n".join(lines).strip() + "\n"
    if estimate_tokens(packet) > max_tokens:
        packet = _trim_section(packet, max_tokens)
    return redact_secrets(packet)
