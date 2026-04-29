from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .models import MemoryNode, RealTaskCaseResult
from .runtime import UnifiedMemoryRuntime


class RealTaskEvaluator:
    def __init__(self, dataset_path: Path, out_path: Path | None = None) -> None:
        self.dataset_path = dataset_path.resolve()
        self.out_path = out_path
        self.dataset = json.loads(self.dataset_path.read_text(encoding="utf-8"))

    def run(self) -> dict[str, Any]:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = self.dataset_path.parent / "runs" / run_stamp
        workspace = run_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        runtime = UnifiedMemoryRuntime.from_base_dir(workspace)
        self._apply_setup(runtime)

        case_results: list[RealTaskCaseResult] = []
        for case in self.dataset.get("cases", []):
            case_results.append(self._run_case(runtime, case))

        payload = self._summarize(runtime, case_results, run_root)
        out_path = self.out_path or (run_root / "real_task_eval_report.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def _apply_setup(self, runtime: UnifiedMemoryRuntime) -> None:
        setup = self.dataset.get("setup", {})
        for memory in setup.get("memories", []):
            node = MemoryNode(
                shell=int(memory.get("shell", 2)),
                sector=str(memory.get("sector", "project")),
                zone=str(memory.get("zone", "eval")),
                cell=str(memory.get("cell", "memory")),
                molecular_type=str(memory.get("molecular_type", "note")),
                summary=str(memory["summary"]),
                raw_content=str(memory.get("content", "")) or None,
                content_ref=str(memory.get("content_ref", "")) or None,
                importance=float(memory.get("importance", 0.55)),
                creative_score=float(memory.get("creative_score", 0.2)),
                stability_score=float(memory.get("stability_score", 0.6)),
                stage=str(memory.get("stage", "long_term")),
                tags=str(memory.get("tags", "")) or None,
            )
            runtime.writeback_memory(
                node=node,
                source_kind=str(memory.get("source_kind", memory.get("molecular_type", "note"))),
                source_path=str(memory.get("content_ref", "")) or None,
            )

        for item in setup.get("files", []):
            path = self._resolve_path(str(item["path"]))
            runtime.services.ingestor.ingest_file(
                path,
                shell=int(item.get("shell", 4)),
                sector=str(item.get("sector", "raw")),
                zone=str(item.get("zone", "eval_files")),
                stage=str(item.get("stage", "long_term")),
                tags=str(item.get("tags", "")),
            )

    def _run_case(self, runtime: UnifiedMemoryRuntime, case: dict[str, Any]) -> RealTaskCaseResult:
        start = perf_counter()
        result = runtime.run_query(
            query=str(case["query"]),
            task_type=str(case.get("task_type", "qa")),
            temperature=float(case.get("temperature", 0.5)),
            max_tokens=int(case.get("max_tokens", 1200)),
            evidence_top_k=int(case.get("evidence_top_k", 6)),
            support_top_k=int(case.get("support_top_k", 3)),
            object_top_k=int(case.get("object_top_k", 4)),
            cognitive_top_k=int(case.get("cognitive_top_k", 3)),
        )
        latency_ms = (perf_counter() - start) * 1000.0

        completion = result["completion"]
        cognitive = result["cognitive"]
        bundle = result["bundle"]
        evidence = result["evidence"]

        core_text = "\n".join(item.get("text", "") for item in completion.core_evidence)
        support_text = "\n".join(item.get("text", "") for item in completion.supporting_context)
        object_text = "\n".join(item.get("object_text", "") for item in completion.evidence_objects)
        experience_text = "\n".join(item.get("summary", "") for item in cognitive.relevant_experience)

        errors: list[str] = []
        expected_core_terms = [str(term).lower() for term in case.get("expected_core_terms", [])]
        expected_any_terms = [str(term).lower() for term in case.get("expected_any_terms", [])]
        expected_latest_terms = [str(term).lower() for term in case.get("expected_latest_terms", [])]
        expected_object_types = [str(term) for term in case.get("expected_object_types", [])]
        expected_polarity = case.get("expected_polarity")
        require_raw = bool(case.get("require_raw_evidence", False))
        must_not_be_experience_only = bool(case.get("must_not_be_experience_only", True))

        core_hit = self._contains_all(core_text, expected_core_terms) if expected_core_terms else True
        top1_hit = self._contains_all((completion.core_evidence[0].get("text", "") if completion.core_evidence else ""), expected_core_terms) if expected_core_terms else True
        any_hit = self._contains_all("\n".join([core_text, support_text, object_text]), expected_any_terms) if expected_any_terms else True

        if not core_hit:
            errors.append("evidence_not_recalled")
        if core_hit and not top1_hit:
            errors.append("top1_precision_fail")
        if not any_hit:
            errors.append("evidence_set_incomplete")

        present_types = {str(item.get("object_type", "")) for item in completion.evidence_objects}
        if expected_object_types and not set(expected_object_types).issubset(present_types):
            errors.append("structured_completion_missing")

        if expected_latest_terms and not self._contains_all("\n".join([core_text, support_text, object_text]), expected_latest_terms):
            errors.append("temporal_chain_gap")

        if expected_polarity is not None and not self._polarity_matches(completion.evidence_objects, float(expected_polarity)):
            errors.append("preference_polarity_error")

        if must_not_be_experience_only and expected_any_terms:
            if (not any_hit) and self._contains_all(experience_text, expected_any_terms):
                errors.append("experience_overrides_fact")

        if require_raw and not bundle.raw_reference_pointers:
            errors.append("summary_drift")

        prism_metrics = self._prism_case_metrics(
            runtime=runtime,
            case=case,
            completion=completion,
            cognitive=cognitive,
            bundle=bundle,
        )
        diagnostics = {
            "query": case["query"],
            "category": str(case.get("category", case.get("task_type", "qa"))),
            "profile": asdict(evidence.profile),
            "query_route": dict(evidence.query_route or {}),
            "core_evidence": completion.core_evidence,
            "primary_evidence": bundle.primary_evidence,
            "evidence_objects": completion.evidence_objects,
            "supporting_context": completion.supporting_context,
            "relevant_experience": cognitive.relevant_experience,
            "creative_reflections": cognitive.creative_reflections,
            "assembled_creative_reflections": bundle.creative_reflections,
            "alternative_paths": cognitive.alternative_paths,
            "assembled_alternative_paths": bundle.alternative_paths,
            "raw_reference_pointers": bundle.raw_reference_pointers,
            "timings_ms": {
                "total_ms": round(latency_ms, 2),
                "retrieval_ms": round(float(evidence.timings_ms.get("total_ms", 0.0)), 2),
                "coarse_retrieval_ms": round(
                    float(evidence.timings_ms.get("dense_vector_ms", 0.0))
                    + float(evidence.timings_ms.get("sparse_fts_ms", 0.0))
                    + float(evidence.timings_ms.get("proxy_vector_ms", 0.0)),
                    2,
                ),
                "fine_retrieval_ms": round(
                    float(evidence.timings_ms.get("candidate_merge_ms", 0.0))
                    + float(evidence.timings_ms.get("rank_evidence_ms", 0.0))
                    + float(evidence.timings_ms.get("fetch_evidence_nodes_ms", 0.0)),
                    2,
                ),
                "rerank_ms": round(float(evidence.timings_ms.get("cross_encoder_rerank_ms", 0.0)), 2),
                "object_shortcut_ms": round(float(evidence.timings_ms.get("object_shortcut_ms", 0.0)), 2),
                "temporal_prefilter_ms": round(float(evidence.timings_ms.get("temporal_prefilter_ms", 0.0)), 2),
                "completion_ms": round(float(completion.timings_ms.get("total_ms", 0.0)), 2),
                "object_completion_ms": round(
                    float(completion.timings_ms.get("object_candidate_ms", 0.0))
                    + float(completion.timings_ms.get("object_rank_ms", 0.0))
                    + float(completion.timings_ms.get("temporal_completion_ms", 0.0))
                    + float(completion.timings_ms.get("personal_context_completion_ms", 0.0)),
                    2,
                ),
                "cognitive_ms": round(float(cognitive.timings_ms.get("total_ms", 0.0)), 2),
                "prism_total_ms": round(float(cognitive.timings_ms.get("prism_total_ms", 0.0)), 2),
                "assemble_ms": round(float(bundle.debug.get("assemble_ms", 0.0) or 0.0), 2),
            },
            "cache_metrics": {
                "retrieval_hit": float(evidence.diagnostics.get("cache", {}).get("retrieval_hit", 0.0) or 0.0),
                "completion_hit": float(completion.diagnostics.get("cache", {}).get("completion_hit", 0.0) or 0.0),
            },
            "routing_metrics": {
                "route_type": str((evidence.query_route or {}).get("route_type", "")),
                "object_shortcut_hit": float(1.0 if evidence.diagnostics.get("decisions", {}).get("object_shortcut") == "shortcut_sufficient" else 0.0),
                "temporal_prefilter_hit": float(0.0 if str(evidence.diagnostics.get("decisions", {}).get("temporal_prefilter", "")).startswith("disabled") else 1.0 if "prefilter_kept_" in str(evidence.diagnostics.get("decisions", {}).get("temporal_prefilter", "")) else 0.0),
            },
            "compression_metrics": {
                "seed_input": int(evidence.diagnostics.get("candidate_counts", {}).get("seed_input", 0) or 0),
                "seed_output": int(evidence.diagnostics.get("candidate_counts", {}).get("seed_output", 0) or 0),
                "dedup_hit_rate": float(evidence.diagnostics.get("compression", {}).get("dedup_hit_rate", 0.0) or 0.0),
                "context_tokens_before": int(bundle.debug.get("raw_input_tokens", 0) or 0),
                "context_tokens_after": int(bundle.debug.get("estimated_input_tokens", 0) or 0),
                "context_token_delta": int(bundle.debug.get("context_token_delta", 0) or 0),
            },
            "prism_metrics": prism_metrics,
            "expected_core_terms": expected_core_terms,
            "expected_any_terms": expected_any_terms,
            "expected_object_types": expected_object_types,
        }

        return RealTaskCaseResult(
            task_id=str(case["task_id"]),
            task_type=str(case.get("task_type", "qa")),
            latency_ms=round(latency_ms, 2),
            passed=not errors,
            errors=errors,
            diagnostics=diagnostics,
        )

    def _summarize(self, runtime: UnifiedMemoryRuntime, case_results: list[RealTaskCaseResult], run_root: Path) -> dict[str, Any]:
        total = len(case_results)
        passed = sum(1 for case in case_results if case.passed)
        error_counts: dict[str, int] = {}
        for case in case_results:
            for error in case.errors:
                error_counts[error] = error_counts.get(error, 0) + 1

        latencies = sorted(case.latency_ms for case in case_results)
        return {
            "dataset": self.dataset.get("name", self.dataset_path.stem),
            "dataset_path": str(self.dataset_path),
            "run_root": str(run_root),
            "workspace": str(run_root / "workspace"),
            "config": {
                "creative_mode": runtime.services.config.creative_mode_name,
                "creative_beam_width": runtime.services.config.creative_beam_width,
                "creative_max_hops": runtime.services.config.creative_max_hops,
                "creative_neighbors_per_hop": runtime.services.config.creative_neighbors_per_hop,
                "creative_max_output_paths": runtime.services.config.creative_max_output_paths,
            },
            "summary": {
                "real_task_quality": {
                    "total_cases": total,
                    "passed_cases": passed,
                    "pass_rate": round(passed / max(1, total), 4),
                    "pass_rate_by_task_type": self._pass_rate_by_task_type(case_results),
                    "error_counts": error_counts,
                },
                "evidence_quality": {
                    "evidence_recall_rate": self._error_free_rate(case_results, "evidence_not_recalled"),
                    "top1_precision_rate": self._error_free_rate(case_results, "top1_precision_fail"),
                    "evidence_set_completeness_rate": self._error_free_rate(case_results, "evidence_set_incomplete"),
                    "structured_completion_rate": self._error_free_rate(case_results, "structured_completion_missing"),
                    "temporal_consistency_rate": self._error_free_rate(case_results, "temporal_chain_gap"),
                    "preference_consistency_rate": self._error_free_rate(case_results, "preference_polarity_error"),
                    "grounding_safety_rate": self._grounding_safety_rate(case_results),
                },
                "engineering_quality": self._engineering_quality(runtime, case_results, latencies),
                "prism_quality": self._prism_summary(case_results),
            },
            "cases": [asdict(case) for case in case_results],
        }

    def _pass_rate_by_task_type(self, case_results: list[RealTaskCaseResult]) -> dict[str, float]:
        grouped: dict[str, list[RealTaskCaseResult]] = {}
        for case in case_results:
            grouped.setdefault(case.task_type, []).append(case)
        return {
            task_type: round(sum(1 for case in items if case.passed) / max(1, len(items)), 4)
            for task_type, items in grouped.items()
        }

    def _error_free_rate(self, case_results: list[RealTaskCaseResult], error_name: str) -> float:
        total = len(case_results)
        clean = sum(1 for case in case_results if error_name not in case.errors)
        return round(clean / max(1, total), 4)

    def _grounding_safety_rate(self, case_results: list[RealTaskCaseResult]) -> float:
        total = len(case_results)
        clean = sum(
            1
            for case in case_results
            if "experience_overrides_fact" not in case.errors and "summary_drift" not in case.errors
        )
        return round(clean / max(1, total), 4)

    def _engineering_quality(self, runtime: UnifiedMemoryRuntime, case_results: list[RealTaskCaseResult], latencies: list[float]) -> dict[str, Any]:
        config = runtime.services.config
        storage = runtime.services.storage
        vector_info = runtime.services.vector_store.info()
        chunks = storage.fetch_all_chunks()
        grain_counts: dict[str, int] = {}
        for chunk in chunks:
            grain = str(chunk.get("grain") or "micro")
            grain_counts[grain] = grain_counts.get(grain, 0) + 1
        stage_series = self._stage_latency_series(case_results)
        route_counts: dict[str, int] = {}
        retrieval_hits = 0.0
        completion_hits = 0.0
        object_shortcut_hits = 0.0
        temporal_prefilter_hits = 0.0
        context_token_delta = 0.0
        seed_input_total = 0.0
        seed_output_total = 0.0
        dedup_hit_total = 0.0
        for case in case_results:
            route_type = str(case.diagnostics.get("routing_metrics", {}).get("route_type", "") or "unknown")
            route_counts[route_type] = route_counts.get(route_type, 0) + 1
            retrieval_hits += float(case.diagnostics.get("cache_metrics", {}).get("retrieval_hit", 0.0) or 0.0)
            completion_hits += float(case.diagnostics.get("cache_metrics", {}).get("completion_hit", 0.0) or 0.0)
            object_shortcut_hits += float(case.diagnostics.get("routing_metrics", {}).get("object_shortcut_hit", 0.0) or 0.0)
            temporal_prefilter_hits += float(case.diagnostics.get("routing_metrics", {}).get("temporal_prefilter_hit", 0.0) or 0.0)
            context_token_delta += float(case.diagnostics.get("compression_metrics", {}).get("context_token_delta", 0.0) or 0.0)
            seed_input_total += float(case.diagnostics.get("compression_metrics", {}).get("seed_input", 0.0) or 0.0)
            seed_output_total += float(case.diagnostics.get("compression_metrics", {}).get("seed_output", 0.0) or 0.0)
            dedup_hit_total += float(case.diagnostics.get("compression_metrics", {}).get("dedup_hit_rate", 0.0) or 0.0)
        return {
            "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
            "p50_latency_ms": self._percentile(latencies, 0.5),
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "stage_latency_ms": {
                stage: {
                    "avg": round(sum(values) / max(1, len(values)), 2),
                    "p50": self._percentile(sorted(values), 0.5),
                    "p95": self._percentile(sorted(values), 0.95),
                }
                for stage, values in stage_series.items()
                if values
            },
            "metadata_db_bytes": config.db_path.stat().st_size if config.db_path.exists() else 0,
            "data_dir_bytes": self._dir_size(config.data_dir),
            "node_count": len(storage.fetch_nodes()),
            "chunk_count": storage.count_chunks(),
            "object_count": len(storage.fetch_objects()),
            "chunk_grain_counts": grain_counts,
            "tracked_file_count": len(storage.fetch_ingest_files()),
            "vector_info": vector_info,
            "route_distribution": route_counts,
            "cache_hit_rates": {
                "retrieval_cache_hit_rate": round(retrieval_hits / max(1, len(case_results)), 4),
                "completion_cache_hit_rate": round(completion_hits / max(1, len(case_results)), 4),
            },
            "shortcut_hit_rates": {
                "object_shortcut_hit_rate": round(object_shortcut_hits / max(1, len(case_results)), 4),
                "temporal_prefilter_hit_rate": round(temporal_prefilter_hits / max(1, len(case_results)), 4),
            },
            "compression": {
                "avg_context_token_delta": round(context_token_delta / max(1, len(case_results)), 2),
                "avg_seed_dedup_hit_rate": round(dedup_hit_total / max(1, len(case_results)), 4),
                "seed_reduction_ratio": round((seed_input_total - seed_output_total) / max(1.0, seed_input_total), 4) if seed_input_total else 0.0,
            },
        }

    def _percentile(self, values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        index = max(0, min(len(values) - 1, int(round((len(values) - 1) * quantile))))
        return round(values[index], 2)

    def _dir_size(self, path: Path) -> int:
        total = 0
        if not path.exists():
            return total
        for candidate in path.rglob("*"):
            if candidate.is_file():
                total += candidate.stat().st_size
        return total

    def _contains_all(self, text: str, terms: list[str]) -> bool:
        lowered = (text or "").lower()
        return all(term in lowered for term in terms)

    def _stage_latency_series(self, case_results: list[RealTaskCaseResult]) -> dict[str, list[float]]:
        series: dict[str, list[float]] = {
            "retrieval_ms": [],
            "coarse_retrieval_ms": [],
            "fine_retrieval_ms": [],
            "rerank_ms": [],
            "object_shortcut_ms": [],
            "temporal_prefilter_ms": [],
            "completion_ms": [],
            "object_completion_ms": [],
            "cognitive_ms": [],
            "prism_total_ms": [],
            "assemble_ms": [],
        }
        for case in case_results:
            timings = case.diagnostics.get("timings_ms", {})
            for key in list(series):
                value = float(timings.get(key, 0.0) or 0.0)
                series[key].append(value)
        return series

    def _prism_summary(self, case_results: list[RealTaskCaseResult]) -> dict[str, Any]:
        metrics = [dict(case.diagnostics.get("prism_metrics", {})) for case in case_results]
        if not metrics:
            return {}
        factual_metrics = [item for item in metrics if item.get("category") in {"qa_exact", "temporal"}]
        open_metrics = [item for item in metrics if item.get("category") == "open_creative"]
        path_metrics = [item for item in metrics if item.get("prism_enabled")]

        adjacency_hits = sum(int(item.get("cache", {}).get("adjacency_hits", 0)) for item in metrics)
        adjacency_total = sum(int(item.get("cache", {}).get("adjacency_total", 0)) for item in metrics)
        neighborhood_hits = sum(int(item.get("cache", {}).get("local_neighborhood_hits", 0)) for item in metrics)
        neighborhood_total = sum(int(item.get("cache", {}).get("local_neighborhood_total", 0)) for item in metrics)
        path_score_hits = sum(int(item.get("cache", {}).get("path_score_hits", 0)) for item in metrics)
        path_score_total = sum(int(item.get("cache", {}).get("path_score_total", 0)) for item in metrics)

        return {
            "prism_enabled_case_rate": round(sum(1 for item in metrics if item.get("prism_enabled")) / max(1, len(metrics)), 4),
            "creative_reflection_case_rate": round(sum(1 for item in metrics if int(item.get("creative_reflection_count", 0)) > 0) / max(1, len(metrics)), 4),
            "alternative_path_case_rate": round(sum(1 for item in metrics if int(item.get("alternative_path_count", 0)) > 0) / max(1, len(metrics)), 4),
            "avg_distinct_beam_types": round(sum(float(item.get("distinct_beam_types", 0.0)) for item in metrics) / max(1, len(metrics)), 2),
            "avg_distinct_signatures": round(sum(float(item.get("distinct_signatures", 0.0)) for item in metrics) / max(1, len(metrics)), 2),
            "open_task_creative_nonempty_rate": round(sum(1 for item in open_metrics if int(item.get("creative_reflection_count", 0)) > 0) / max(1, len(open_metrics)), 4),
            "factual_creative_leakage_rate": round(sum(1 for item in factual_metrics if item.get("factual_creative_leakage")) / max(1, len(factual_metrics)), 4),
            "primary_evidence_contamination_rate": round(sum(1 for item in metrics if item.get("primary_evidence_contamination")) / max(1, len(metrics)), 4),
            "path_bound_violation_rate": round(sum(1 for item in path_metrics if not item.get("path_bounds_ok", True)) / max(1, len(path_metrics)), 4),
            "avg_alternative_path_retention_rate": round(sum(float(item.get("assembled_alternative_retention_rate", 1.0)) for item in metrics) / max(1, len(metrics)), 4),
            "avg_selected_paths": round(sum(float(item.get("selected_paths", 0.0)) for item in metrics) / max(1, len(metrics)), 2),
            "avg_expanded_paths": round(sum(float(item.get("expanded_paths", 0.0)) for item in metrics) / max(1, len(metrics)), 2),
            "cache_hit_rates": {
                "adjacency_cache_hit_rate": round(adjacency_hits / adjacency_total, 4) if adjacency_total else 0.0,
                "local_neighborhood_cache_hit_rate": round(neighborhood_hits / neighborhood_total, 4) if neighborhood_total else 0.0,
                "path_score_cache_hit_rate": round(path_score_hits / path_score_total, 4) if path_score_total else 0.0,
            },
        }

    def _prism_case_metrics(
        self,
        *,
        runtime: UnifiedMemoryRuntime,
        case: dict[str, Any],
        completion: Any,
        cognitive: Any,
        bundle: Any,
    ) -> dict[str, Any]:
        config = runtime.services.config
        profile = completion.profile
        prism_diag = dict(cognitive.diagnostics.get("prism", {}) or {})
        cache = dict(prism_diag.get("cache", {}) or {})
        alternative_paths = list(cognitive.alternative_paths or [])
        creative_reflections = list(cognitive.creative_reflections or [])
        prism_enabled = bool(prism_diag.get("enabled", False))

        adjacency_hits = int(cache.get("adjacency_hits", 0)) + int(cache.get("beam_adjacency_hits", 0))
        adjacency_misses = int(cache.get("adjacency_misses", 0)) + int(cache.get("beam_adjacency_misses", 0))
        local_hits = int(cache.get("local_neighborhood_hits", 0))
        local_misses = int(cache.get("local_neighborhood_misses", 0))
        path_score_hits = int(cache.get("path_score_hits", 0))
        path_score_misses = int(cache.get("path_score_misses", 0))

        candidate_counts = dict(prism_diag.get("candidate_counts", {}) or {})
        seed_chunks = int(candidate_counts.get("seed_chunks", 0))
        operators = list(prism_diag.get("operators", []) or [])
        neighbors_per_hop = max(1, int(config.creative_neighbors_per_hop))
        beam_width = max(1, int(config.creative_beam_width))
        max_hops = max(1, int(config.creative_max_hops))
        expanded_upper_bound = seed_chunks * max(1, len(operators)) * neighbors_per_hop
        if max_hops > 1:
            expanded_upper_bound += (max_hops - 1) * beam_width * neighbors_per_hop
        selected_upper_bound = min(
            max(1, int(case.get("cognitive_top_k", 3))),
            max(1, int(config.creative_max_output_paths)),
        )
        max_hop_count = max((int(path.get("hop_count", 0) or 0) for path in alternative_paths), default=0)
        primary_evidence_contamination = any(
            any(key in item for key in ("beam_type", "path_id", "signature"))
            for item in bundle.primary_evidence
        )
        case_category = str(case.get("category", case.get("task_type", "qa")))
        factual_creative_leakage = bool(
            prism_enabled
            and case_category in {"qa_exact", "temporal"}
            and
            profile.needs_exact_evidence
            and (
                creative_reflections
                or any(str(path.get("path_role") or "") not in {"", "support"} for path in alternative_paths)
            )
        )

        return {
            "category": case_category,
            "prism_enabled": prism_enabled,
            "reason": str(prism_diag.get("reason", "")),
            "needs_exact_evidence": bool(profile.needs_exact_evidence),
            "creative_reflection_count": len(creative_reflections) if prism_enabled else 0,
            "assembled_creative_reflection_count": len(bundle.creative_reflections) if prism_enabled else 0,
            "alternative_path_count": len(alternative_paths) if prism_enabled else 0,
            "assembled_alternative_path_count": len(bundle.alternative_paths) if prism_enabled else 0,
            "assembled_alternative_retention_rate": round(len(bundle.alternative_paths) / max(1, len(alternative_paths)), 4) if prism_enabled and alternative_paths else 1.0,
            "distinct_beam_types": len({str(path.get("beam_type") or "") for path in alternative_paths if path.get("beam_type")}) if prism_enabled else 0,
            "distinct_signatures": len({str(path.get("signature") or "") for path in alternative_paths if path.get("signature")}) if prism_enabled else 0,
            "expanded_paths": int(candidate_counts.get("expanded_paths", 0)) if prism_enabled else 0,
            "selected_paths": int(candidate_counts.get("selected_paths", 0)) if prism_enabled else 0,
            "seed_chunks": seed_chunks,
            "operator_count": len(operators),
            "max_hop_count": max_hop_count,
            "path_bounds": {
                "expanded_upper_bound": expanded_upper_bound,
                "selected_upper_bound": selected_upper_bound,
                "max_hops": max_hops,
                "neighbors_per_hop": neighbors_per_hop,
                "beam_width": beam_width,
            },
            "path_bounds_ok": bool(
                int(candidate_counts.get("expanded_paths", 0)) <= expanded_upper_bound
                and int(candidate_counts.get("selected_paths", 0)) <= selected_upper_bound
                and max_hop_count <= max_hops
            ),
            "cache": {
                "adjacency_hits": adjacency_hits,
                "adjacency_total": adjacency_hits + adjacency_misses,
                "local_neighborhood_hits": local_hits,
                "local_neighborhood_total": local_hits + local_misses,
                "path_score_hits": path_score_hits,
                "path_score_total": path_score_hits + path_score_misses,
            },
            "primary_evidence_contamination": primary_evidence_contamination,
            "factual_creative_leakage": factual_creative_leakage,
        }

    def _polarity_matches(self, objects: list[dict[str, Any]], expected_polarity: float) -> bool:
        for item in objects:
            polarity = item.get("polarity")
            if polarity is None:
                continue
            if expected_polarity >= 0 and float(polarity) >= 0:
                return True
            if expected_polarity < 0 and float(polarity) < 0:
                return True
        return False

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = (self.dataset_path.parent / raw_path).resolve()
        if candidate.exists():
            return candidate
        candidate = (self.dataset_path.parent.parent / raw_path).resolve()
        if candidate.exists():
            return candidate
        return Path(raw_path).resolve()
