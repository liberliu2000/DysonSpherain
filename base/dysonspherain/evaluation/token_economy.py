from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dysonspherain.token_economy.metrics import RetrievalQuality, TokenEconomySample
from dysonspherain.evaluation.token_economy_modes import resolve_token_economy_mode
from dysonspherain.token_economy.artifact_inputs import payloads_from_benchmark_artifact_root
from dysonspherain.token_economy.prompt_parts import build_prompt_parts, join_prompt_parts
from dysonspherain.token_economy.report import write_report
from dysonspherain.utils.token_counter import TokenCounter
from sphere_cli.project_state import list_memories
from sphere_cli.runtime import UnifiedMemoryRuntime


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _split_int_csv(value: str) -> list[int]:
    return [int(item) for item in _split_csv(value)]


def _smoke_lines() -> list[str]:
    return [
        json.dumps(
            {
                "sample_id": "smoke",
                "query": "Fix current retrieval regression without changing default retrieval behavior.",
                "history": "Long project history with benchmark artifacts, prior decisions, known regression notes, and repeated prompts. " * 40,
                "retrieved_context": "Use artifact-backed diagnostics and keep token economy separate from retrieval quality.",
                "metadata": "source=smoke",
                "retrieval_quality": {"recall_at_10": 1.0, "ndcg_at_10": 1.0},
            }
        )
    ]


def _record_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("updated_at") or record.get("created_at") or "",
        record.get("memory_type") or "",
        record.get("title") or "",
        record.get("summary") or record.get("content") or "",
    ]
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata:
        parts.append(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return " | ".join(str(part) for part in parts if str(part).strip())


def _memory_store_history(memory_base_dir: Path | None, project: str, *, recent_k: int, baseline_type: str) -> tuple[str, dict[str, Any]]:
    if memory_base_dir is None:
        return "", {}
    records = list_memories(memory_base_dir, project)
    if baseline_type == "naive_recent":
        records = records[:recent_k]
    text = "\n".join(_record_text(record) for record in records if _record_text(record))
    if not text:
        return "", {"baseline_unavailable": f"{baseline_type}:memory_store_empty", "baseline_source": "memory_store"}
    return text, {"baseline_source": "memory_store", "baseline_memory_count": len(records), "baseline_project": project}


def _history_for_baseline(
    payload: dict[str, Any],
    baseline_type: str,
    recent_k: int,
    *,
    memory_base_dir: Path | None = None,
    project: str = "DysonSpherain",
) -> tuple[str, dict[str, Any]]:
    if baseline_type == "off":
        return "", {"baseline_source": "off"}
    if baseline_type == "manual_summary":
        summary = payload.get("manual_summary") or payload.get("summary")
        if summary:
            return str(summary), {"baseline_source": "payload_manual_summary"}
        history = str(payload.get("history") or payload.get("full_history") or payload.get("context") or "")
        return "\n".join(history.splitlines()[:8]) or history[:1000], {"baseline_source": "synthetic_manual_summary"}
    if baseline_type == "oracle_minimal":
        oracle = payload.get("oracle_context") or payload.get("gold_evidence")
        if oracle:
            return str(oracle), {}
        return "", {"baseline_unavailable": "oracle_minimal"}
    if memory_base_dir is not None:
        memory_text, memory_extra = _memory_store_history(memory_base_dir, project, recent_k=recent_k, baseline_type=baseline_type)
        if memory_text:
            return memory_text, memory_extra
    history = payload.get("history") or payload.get("full_history") or payload.get("context") or ""
    if baseline_type == "naive_recent":
        parts = str(history).splitlines()
        return "\n".join(parts[-recent_k:]) if parts else str(history)[-recent_k * 200 :], {"baseline_source": "payload"}
    return str(history), {"baseline_source": "payload"}


def _truncate_context(context: str, budget: int, counter: TokenCounter, allow_evidence_truncation: bool) -> tuple[str, bool]:
    if counter.count(context).tokens <= budget:
        return context, False
    if not allow_evidence_truncation:
        lines: list[str] = []
        for line in context.splitlines():
            candidate = "\n".join([*lines, line])
            if counter.count(candidate).tokens > budget:
                break
            lines.append(line)
        return "\n".join(lines), True
    char_budget = max(16, budget * 4)
    return context[:char_budget].rstrip() + "\n...[truncated]", True


def _runtime_base_dir_from_memory_db(memory_db: str | None) -> Path | None:
    if not memory_db:
        return None
    path = Path(memory_db).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"--memory-db does not exist: {path}")
    if path.is_dir():
        return path
    if path.name == "memory.db" and path.parent.name == "data":
        return path.parent.parent
    return path.parent


def _render_runtime_context(run_result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    bundle = run_result.get("bundle")
    evidence = run_result.get("evidence")
    completion = run_result.get("completion")
    sections: list[str] = []
    for label, values in (
        ("core_evidence", getattr(bundle, "core_evidence", []) if bundle is not None else []),
        ("supporting_context", getattr(bundle, "supporting_context", []) if bundle is not None else []),
        ("evidence_objects", getattr(bundle, "evidence_objects", []) if bundle is not None else []),
        ("relevant_experience", getattr(bundle, "relevant_experience", []) if bundle is not None else []),
        ("alternative_paths", getattr(bundle, "alternative_paths", []) if bundle is not None else []),
    ):
        if values:
            sections.append(f"## {label}")
            sections.append(json.dumps(values, ensure_ascii=False, sort_keys=True))
    if not sections and completion is not None:
        sections.append(json.dumps(getattr(completion, "core_evidence", []) or [], ensure_ascii=False, sort_keys=True))
    diagnostics = getattr(evidence, "diagnostics", {}) if evidence is not None else {}
    return "\n\n".join(sections), {
        "candidate_count": len(getattr(evidence, "candidates", []) or []) if evidence is not None else 0,
        "final_context_item_count": sum(len(getattr(bundle, name, []) or []) for name in ("core_evidence", "supporting_context", "evidence_objects", "relevant_experience", "alternative_paths")) if bundle is not None else 0,
        "retrieved_evidence_count": len(getattr(completion, "core_evidence", []) or []) if completion is not None else 0,
        "route": getattr(evidence, "query_route", {}) if evidence is not None else {},
        "diagnostics": diagnostics,
    }


def _assemble_runtime_context(
    *,
    query: str,
    mode: str,
    budget: int,
    memory_base_dir: Path | None,
) -> tuple[str, dict[str, Any]]:
    mode_config = resolve_token_economy_mode(mode)
    if mode_config.retrieval_disabled:
        return "", {"retrieval_mode": mode_config.name, "retrieval_disabled": True, "mode_status": mode_config.status}
    if not mode_config.available:
        return "", {"mode_unavailable": mode_config.name, "mode_status": mode_config.status, "unavailable_reason": mode_config.unavailable_reason}
    if memory_base_dir is None:
        return "", {"mode_unavailable": "memory_db_not_provided"}
    started = time.perf_counter()
    overrides = dict(mode_config.runtime_overrides or {})
    runtime = UnifiedMemoryRuntime.from_base_dir(memory_base_dir, config_overrides=overrides)
    run_result = runtime.run_query(
        query,
        task_type="qa",
        max_tokens=budget,
        evidence_top_k=8,
        support_top_k=4,
        object_top_k=4,
        cognitive_top_k=0 if str(mode).lower() != "exploratory" else 3,
    )
    context, extra = _render_runtime_context(run_result)
    extra["runtime_assembly_seconds"] = time.perf_counter() - started
    extra["runtime_base_dir"] = str(memory_base_dir)
    extra["runtime_mode"] = mode_config.name
    extra["runtime_mode_overrides"] = overrides
    return context, extra


def _quality(payload: dict[str, Any]) -> RetrievalQuality:
    raw = payload.get("retrieval_quality") if isinstance(payload.get("retrieval_quality"), dict) else {}
    return RetrievalQuality(
        recall_at_5=raw.get("recall_at_5") or payload.get("recall_at_5"),
        recall_at_10=raw.get("recall_at_10") or payload.get("recall_at_10"),
        ndcg_at_10=raw.get("ndcg_at_10") or payload.get("ndcg_at_10"),
        gold_rank=raw.get("gold_rank") or payload.get("gold_rank"),
        candidate_recall_at_100=raw.get("candidate_recall_at_100") or payload.get("candidate_recall_at_100"),
    )


def _local_compute_from_payload(payload: dict[str, Any], runtime_extra: dict[str, Any]) -> dict[str, Any]:
    local = payload.get("local_compute_economy") if isinstance(payload.get("local_compute_economy"), dict) else {}
    if local:
        return dict(local)
    diagnostics = runtime_extra.get("diagnostics") if isinstance(runtime_extra.get("diagnostics"), dict) else {}
    cache = diagnostics.get("cache") if isinstance(diagnostics.get("cache"), dict) else {}
    compression = diagnostics.get("compression") if isinstance(diagnostics.get("compression"), dict) else {}
    cache_metrics = payload.get("cache_metrics") if isinstance(payload.get("cache_metrics"), dict) else {}
    return {
        "embedding_cache_hit_count": int(cache_metrics.get("embedding_cache_hit_count") or 0),
        "embedding_cache_miss_count": int(cache_metrics.get("embedding_cache_miss_count") or 0),
        "embedding_cache_hit_ms_saved": float(cache_metrics.get("embedding_cache_hit_ms_saved") or 0.0),
        "retrieval_cache_hit_rate": float(cache.get("retrieval_hit") or cache_metrics.get("retrieval_hit") or 0.0),
        "profile_cache_hit_rate": float(cache.get("profile_hit") or cache_metrics.get("profile_hit") or 0.0),
        "estimated_local_runtime_saved_ms": float(cache_metrics.get("estimated_local_runtime_saved_ms") or cache_metrics.get("embedding_cache_hit_ms_saved") or 0.0),
        "dedup_hit_rate": float(compression.get("dedup_hit_rate") or 0.0),
    }


def _sample_from_payload(
    payload: dict[str, Any],
    *,
    index: int,
    mode: str,
    baseline_type: str,
    budget: int,
    recent_k: int,
    counter: TokenCounter,
    allow_evidence_truncation: bool,
    memory_base_dir: Path | None = None,
    project: str = "DysonSpherain",
) -> TokenEconomySample:
    started = time.perf_counter()
    query = str(payload.get("query") or payload.get("question") or "")
    baseline_context, baseline_extra = _history_for_baseline(payload, baseline_type, recent_k, memory_base_dir=memory_base_dir, project=project)
    runtime_context, runtime_extra = _assemble_runtime_context(query=query, mode=mode, budget=budget, memory_base_dir=memory_base_dir)
    retrieved = runtime_context or str(payload.get("retrieved_context") or payload.get("evidence") or "")
    decision = "inject" if retrieved else "skip"
    if mode == "off":
        decision = "skip"
    truncated = False
    if not runtime_context:
        retrieved, truncated = _truncate_context(retrieved, budget, counter, allow_evidence_truncation)
    metadata = str(payload.get("metadata") or "")
    if runtime_extra:
        metadata = "\n".join(value for value in [metadata, json.dumps(runtime_extra, ensure_ascii=False, sort_keys=True)] if value.strip())
    parts = build_prompt_parts(query=query, evidence=retrieved, metadata=metadata)
    final_prompt = join_prompt_parts(parts)
    raw = counter.count(baseline_context)
    final = counter.count(final_prompt)
    part_counts = {key: counter.count(value).tokens for key, value in parts.items()}
    retrieved_count = counter.count(retrieved)
    sample = TokenEconomySample(
        sample_id=str(payload.get("sample_id") or payload.get("id") or index),
        query=query,
        mode=mode,
        baseline_type=baseline_type,
        context_token_budget=budget,
        raw_history_tokens=raw.tokens,
        raw_history_chars=raw.chars,
        retrieved_context_tokens=retrieved_count.tokens,
        retrieved_context_chars=retrieved_count.chars,
        final_prompt_tokens=final.tokens,
        final_prompt_chars=final.chars,
        system_prompt_tokens=part_counts["system"],
        user_query_tokens=part_counts["query"],
        evidence_tokens=part_counts["evidence"],
        metadata_tokens=part_counts["metadata"],
        instruction_tokens=part_counts["instruction"],
        memory_header_tokens=part_counts["memory_header"],
        retrieved_evidence_count=int(payload.get("retrieved_evidence_count") or (1 if retrieved else 0)),
        candidate_count=int(payload.get("candidate_count") or 0),
        final_context_item_count=int(payload.get("final_context_item_count") or (1 if retrieved else 0)),
        latency_seconds=time.perf_counter() - started,
        tokenizer_name=final.tokenizer_name,
        fallback_tokenizer_used=final.fallback_used,
        retrieval_quality=_quality(payload),
        extra={
            "prompt_parts_best_effort": True,
            "context_truncated": truncated,
            "runtime_context_used": bool(runtime_context),
            "decision": decision,
            "adapter": "evaluation",
            "duplicate_token_ratio": 0.0,
            "quality_guard_status": "ok",
            "local_compute_economy": _local_compute_from_payload(payload, runtime_extra),
            **baseline_extra,
            **runtime_extra,
        },
    ).finalize()
    if runtime_extra.get("runtime_assembly_seconds"):
        sample.latency_seconds += float(runtime_extra["runtime_assembly_seconds"])
    return sample


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--benchmark-artifact-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-token-budget", default="1600")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--tokenizer-model", default="cl100k_base")
    parser.add_argument("--tokenizer-strategy", default="auto")
    parser.add_argument("--tokenizer-calibration")
    parser.add_argument("--modes", default="conservative")
    parser.add_argument("--baseline-types", default="full_history")
    parser.add_argument("--memory-db")
    parser.add_argument("--project", default="DysonSpherain")
    parser.add_argument("--recent-k", type=int, default=20)
    parser.add_argument("--low-saving-threshold", type=float, default=0.2)
    parser.add_argument("--quality-drop-threshold", type=float, default=0.05)
    parser.add_argument("--evidence-bloat-threshold", type=float, default=0.85)
    parser.add_argument("--metadata-bloat-threshold", type=float, default=0.25)
    parser.add_argument("--allow-evidence-truncation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    counter = TokenCounter(args.tokenizer_model, strategy=args.tokenizer_strategy, calibration_file=args.tokenizer_calibration)
    memory_base_dir = _runtime_base_dir_from_memory_db(args.memory_db)
    if args.smoke:
        lines = _smoke_lines()
    elif args.benchmark_artifact_root:
        lines = [json.dumps(item, ensure_ascii=False) for item in payloads_from_benchmark_artifact_root(Path(args.benchmark_artifact_root))]
    elif args.input:
        lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    else:
        raise SystemExit("--input is required unless --smoke is set")
    if args.max_samples:
        lines = lines[: args.max_samples]

    payloads = [json.loads(line) for line in lines if line.strip()]
    samples: list[TokenEconomySample] = []
    for mode in _split_csv(args.modes):
        for baseline_type in _split_csv(args.baseline_types):
            for budget in _split_int_csv(args.context_token_budget):
                for index, payload in enumerate(payloads):
                    samples.append(
                        _sample_from_payload(
                            payload,
                            index=index,
                            mode=mode,
                            baseline_type=baseline_type,
                            budget=budget,
                            recent_k=args.recent_k,
                            counter=counter,
                            allow_evidence_truncation=args.allow_evidence_truncation,
                            memory_base_dir=memory_base_dir,
                            project=args.project,
                        )
                    )
    summary = write_report(
        samples,
        Path(args.output),
        low_saving_threshold=args.low_saving_threshold,
        evidence_bloat_threshold=args.evidence_bloat_threshold,
        metadata_bloat_threshold=args.metadata_bloat_threshold,
        quality_drop_threshold=args.quality_drop_threshold,
    )
    print("Token Economy Summary")
    print("---------------------")
    print(f"Samples: {summary.get('sample_count', 0)}")
    print(f"Tokenizer: {summary.get('tokenizer_name', '')}")
    print(f"Fallback tokenizer used: {'yes' if summary.get('fallback_tokenizer_used_count', 0) else 'no'}")
    print(f"Modes: {', '.join(summary.get('modes') or [])}")
    print(f"Avg final prompt tokens: {summary.get('mean_final_prompt_tokens', 0):.1f}")
    print(f"Avg saved ratio: {summary.get('mean_saved_tokens_ratio', 0) * 100:.1f}%")
    print("Failure cases:")
    for key, value in (summary.get("failure_case_counts") or {}).items():
        print(f"  {key}: {value}")
    print(f"Reports written to: {Path(args.output)}")


if __name__ == "__main__":
    main()
