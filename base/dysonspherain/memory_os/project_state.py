from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dysonspherain.context_pack.token_budgeter import fit_context_pack
from dysonspherain.utils.token_counter import TokenCounter
from sphere_cli.project_state import load_project_state


@dataclass(frozen=True)
class ProjectStateRequest:
    cwd: str
    include_recent_benchmarks: bool = True
    include_open_tasks: bool = True
    token_budget: int = 1200
    project: str = "DysonSpherain"


def _state_text(project_state: dict[str, Any]) -> str:
    return "\n".join(str(value) for value in project_state.values() if value)


def _trim_text(value: str, counter: TokenCounter, token_budget: int) -> str:
    text = str(value or "")
    if counter.count(text).tokens <= token_budget:
        return text
    if token_budget <= 0:
        return ""
    suffix = " ...[truncated]"
    lo = 0
    hi = len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip()
        if mid < len(text):
            candidate = candidate.rstrip() + suffix
        if counter.count(candidate).tokens <= token_budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _fit_project_state(project_state: dict[str, Any], token_budget: int) -> tuple[dict[str, Any], dict[str, Any]]:
    counter = TokenCounter()
    before = counter.count(_state_text(project_state)).tokens
    fitted = {
        "current_goal": str(project_state.get("current_goal") or ""),
        "recent_changes": list(project_state.get("recent_changes") or []),
        "latest_benchmarks": list(project_state.get("latest_benchmarks") or []),
        "known_regressions": list(project_state.get("known_regressions") or []),
        "open_tasks": list(project_state.get("open_tasks") or []),
        "recommended_focus": list(project_state.get("recommended_focus") or []),
    }
    dropped: list[str] = []

    # Drop lower-value bulk first. Constraints/focus and current goal are trimmed last.
    drop_order = ("recent_changes", "open_tasks", "latest_benchmarks", "known_regressions", "recommended_focus")
    made_progress = True
    while counter.count(_state_text(fitted)).tokens > token_budget and made_progress:
        made_progress = False
        for key in drop_order:
            values = fitted.get(key)
            if counter.count(_state_text(fitted)).tokens <= token_budget:
                break
            if isinstance(values, list) and values:
                values.pop()
                dropped.append(key)
                made_progress = True

    if counter.count(_state_text(fitted)).tokens > token_budget and fitted.get("current_goal"):
        remaining_for_goal = max(0, token_budget - counter.count(_state_text({**fitted, "current_goal": ""})).tokens)
        original = fitted["current_goal"]
        fitted["current_goal"] = _trim_text(str(original), counter, remaining_for_goal)
        if fitted["current_goal"] != original:
            dropped.append("current_goal_tail")

    if counter.count(_state_text(fitted)).tokens > token_budget:
        for key in ("recommended_focus", "known_regressions", "latest_benchmarks", "open_tasks", "recent_changes"):
            values = fitted.get(key)
            if not isinstance(values, list):
                continue
            for index, value in enumerate(list(values)):
                if counter.count(_state_text(fitted)).tokens <= token_budget:
                    break
                remaining = max(0, token_budget - counter.count(_state_text({**fitted, key: []})).tokens)
                values[index] = _trim_text(str(value), counter, remaining)
                dropped.append(f"{key}:tail")
            fitted[key] = [value for value in values if str(value).strip()]

    if counter.count(_state_text(fitted)).tokens > token_budget:
        for key in ("recent_changes", "open_tasks", "latest_benchmarks", "known_regressions", "recommended_focus"):
            if fitted.get(key):
                dropped.append(f"{key}:all")
                fitted[key] = []
            if counter.count(_state_text(fitted)).tokens <= token_budget:
                break

    if counter.count(_state_text(fitted)).tokens > token_budget:
        fitted["current_goal"] = ""
        dropped.append("current_goal")

    after = counter.count(_state_text(fitted)).tokens
    return fitted, {
        "estimated_tokens_before": before,
        "estimated_tokens": after,
        "budget": token_budget,
        "compression_ratio": (after / before) if before else 0.0,
        "dropped_items": dropped,
        "over_budget": after > token_budget,
    }


def get_project_state(request: ProjectStateRequest) -> dict[str, Any]:
    base_dir = Path(request.cwd or ".").resolve()
    state = load_project_state(base_dir, request.project)
    latest = []
    if request.include_recent_benchmarks:
        for benchmark, payload in sorted(state.latest_benchmark_status.items()):
            latest.append(f"{benchmark}: {payload.get('run_type', payload.get('status'))} q={payload.get('question_count')} fallback={payload.get('fallback_in_use')}")
    project_state = {
        "current_goal": state.current_goal or "",
        "recent_changes": list(state.recent_decisions[-8:]),
        "latest_benchmarks": latest,
        "known_regressions": list(state.known_blockers),
        "open_tasks": list(state.active_tasks) if request.include_open_tasks else [],
        "recommended_focus": list(state.constraints[:4]),
    }
    fitted_state, token_estimate = _fit_project_state(project_state, request.token_budget)
    return {
        "status": "ok",
        "project_state": fitted_state,
        "token_estimate": token_estimate,
    }
