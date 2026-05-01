from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dysonspherain.context_pack.builder import build_pack, build_pack_from_runtime_result
from dysonspherain.context_pack.renderers import render_markdown
from sphere_cli.runtime import UnifiedMemoryRuntime


@dataclass
class RecallRequest:
    query: str
    cwd: str | None = None
    task_type: str = "unknown"
    token_budget: int = 1600
    include_files: bool = True
    include_benchmarks: bool = True
    include_prior_prompts: bool = True
    freshness: str = "auto"
    project: str = "DysonSpherain"


@dataclass
class RecallResult:
    status: str
    context_pack: dict[str, Any]
    rendered_context: str
    token_estimate: dict[str, Any]
    trace: dict[str, Any]
    empty_result_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recall(request: RecallRequest) -> RecallResult:
    base_dir = Path(request.cwd or ".").resolve()
    trace: dict[str, Any]
    try:
        runtime = UnifiedMemoryRuntime.from_base_dir(
            base_dir,
            config_overrides={
                "project_name": request.project,
                "creative_mode": "off",
            },
        )
        run_result = runtime.run_query(
            request.query,
            task_type=request.task_type if request.task_type != "unknown" else "qa",
            max_tokens=request.token_budget,
            evidence_top_k=8,
            support_top_k=4,
            object_top_k=4,
            cognitive_top_k=0,
        )
        pack, budget = build_pack_from_runtime_result(
            project=request.project,
            query=request.query,
            run_result=run_result,
            token_budget=request.token_budget,
            include_files=request.include_files,
            include_benchmarks=request.include_benchmarks,
            include_prior_prompts=request.include_prior_prompts,
            freshness=request.freshness,
        )
        evidence = run_result.get("evidence")
        diagnostics = getattr(evidence, "diagnostics", {}) if evidence is not None else {}
        channel_stats = diagnostics.get("channel_stats") if isinstance(diagnostics, dict) else None
        retrieval_channels = sorted(channel_stats) if isinstance(channel_stats, dict) and channel_stats else []
        if not retrieval_channels and isinstance(diagnostics, dict):
            for key in ("retrieval_channels", "channels", "active_channels"):
                raw_channels = diagnostics.get(key)
                if isinstance(raw_channels, (list, tuple)):
                    retrieval_channels = sorted(str(channel) for channel in raw_channels if channel)
                    break
        if not retrieval_channels:
            retrieval_channels = ["runtime_pipeline"]
        trace = {
            "retrieval_channels": retrieval_channels,
            "channel_stats": channel_stats if isinstance(channel_stats, dict) else {},
            "candidate_count": len(getattr(evidence, "candidates", []) or []) if evidence is not None else 0,
            "selected_count": len(pack.core_evidence) + len(pack.prior_decisions) + len(pack.known_failures) + len(pack.benchmark_state),
            "freshness": request.freshness,
            "freshness_applied": request.freshness in {"recent", "stable"},
            "include_files": request.include_files,
            "include_benchmarks": request.include_benchmarks,
            "include_prior_prompts": request.include_prior_prompts,
            "pipeline": "UnifiedMemoryRuntime.run_query",
            "route": getattr(evidence, "query_route", {}) if evidence is not None else {},
        }
    except Exception as exc:
        pack, budget = build_pack(
            base_dir=base_dir,
            project=request.project,
            query=request.query,
            token_budget=request.token_budget,
            include_files=request.include_files,
            include_benchmarks=request.include_benchmarks,
        )
        evidence_count = len(pack.core_evidence) + len(pack.prior_decisions) + len(pack.known_failures) + len(pack.benchmark_state)
        trace = {
            "retrieval_channels": ["project_memory_lexical_fallback"],
            "candidate_count": evidence_count,
            "selected_count": evidence_count,
            "freshness": request.freshness,
            "freshness_applied": False,
            "include_files": request.include_files,
            "include_benchmarks": request.include_benchmarks,
            "include_prior_prompts": request.include_prior_prompts,
            "pipeline": "legacy_project_memory_fallback",
            "fallback_reason": str(exc),
        }
    rendered = render_markdown(pack)
    evidence_count = len(pack.core_evidence) + len(pack.prior_decisions) + len(pack.known_failures) + len(pack.benchmark_state)
    status = "ok" if evidence_count else "empty"
    return RecallResult(
        status=status,
        context_pack=pack.to_dict(),
        rendered_context=rendered,
        token_estimate={
            "budget": request.token_budget,
            "estimated_tokens": budget["estimated_tokens_after"],
            "compression_ratio": budget["compression_ratio"],
            "over_budget": budget["over_budget"],
        },
        trace=trace,
        empty_result_reason=None if evidence_count else "no_relevant_project_memory",
    )
