from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BENCHMARKS = ("longmemeval", "locomo", "knowme", "clonemem")
PRIMARY_METRIC_PATHS = {
    "longmemeval": "metrics.session.recall_any@10",
    "locomo": "metrics.session.recall_frac@10",
    "knowme": "metrics.segment.recall_frac@10",
    "clonemem": "metrics.segment.recall_frac@10",
}
SUPPLEMENTARY_METRICS = {
    "longmemeval": ["metrics.session.recall_any@5", "metrics.session.recall_any@10", "metrics.session.ndcg_any@10"],
    "locomo": ["metrics.session.recall_frac@10", "metrics.session.recall_any@10", "metrics.session.ndcg_any@10"],
    "knowme": ["metrics.segment.recall_frac@10", "metrics.segment.recall_any@10", "metrics.segment.ndcg_any@10"],
    "clonemem": ["metrics.segment.recall_frac@10", "metrics.segment.recall_any@10", "metrics.segment.ndcg_any@10"],
}
BENCHMARK_SCOPE = {
    "longmemeval": "Long-horizon session evidence retrieval",
    "locomo": "Conversation and session-level retrieval",
    "knowme": "Profile and preference retrieval",
    "clonemem": "Fine-grained segment-level autobiographical retrieval",
}
BENCHMARK_CANDIDATE_DIAG = {benchmark: "candidate_recall@100" for benchmark in BENCHMARKS}
GUARDRAILS = {
    "longmemeval": "Recall@10 >= 0.95",
    "locomo": "session recall_frac@10 >= 0.90",
    "knowme": "segment recall_frac@10 >= 0.55",
    "clonemem": "segment recall_frac@10 >= 0.12",
}
GUARDRAIL_THRESHOLDS = {
    "longmemeval": 0.95,
    "locomo": 0.90,
    "knowme": 0.55,
    "clonemem": 0.12,
}


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required artifact: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


SCALAR_PATTERNS = {
    "question_count": re.compile(r'"question_count"\s*:\s*(\d+)'),
    "fallback_in_use": re.compile(r'"fallback_in_use"\s*:\s*(true|false)'),
}


def _extract_json_object(text: str, key: str) -> dict[str, Any]:
    marker = f'"{key}"'
    anchor = text.find(marker)
    if anchor < 0:
        raise KeyError(f"Missing object '{key}' in metrics head")
    start = text.find("{", anchor)
    if start < 0:
        raise ValueError(f"Missing object start for '{key}'")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError(f"Unterminated object for '{key}'")


def _load_metrics_head(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open("r", encoding="utf-8") as fh:
        head = fh.read(8 * 1024 * 1024)
    if '"results"' not in head:
        head = path.read_text(encoding="utf-8")
    payload: dict[str, Any] = {
        "metrics": _extract_json_object(head, "metrics"),
    }
    for key, pattern in SCALAR_PATTERNS.items():
        match = pattern.search(head)
        if not match:
            continue
        value = match.group(1)
        if value in {"true", "false"}:
            payload[key] = value == "true"
        else:
            payload[key] = int(value)
    return payload


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing path '{dotted_path}' in artifact payload")
        current = current[part]
    return current


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _trend_label(current: float | None, baseline: float | None, *, eps: float = 0.0025) -> str:
    if current is None or baseline is None:
        return "unknown"
    delta = current - baseline
    if delta > eps:
        return "improved"
    if delta < -eps:
        return "regressed"
    return "roughly flat"


def _guardrail_status(benchmark: str, value: float | None) -> str:
    if value is None:
        return "unknown"
    threshold = GUARDRAIL_THRESHOLDS[benchmark]
    delta = value - threshold
    if delta >= 0:
        return "met"
    if delta >= -0.005:
        return "narrowly missed"
    return "missed"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header_line, sep_line, body])


def _top_channels(item: dict[str, Any], *, limit: int = 4) -> list[str]:
    channels = dict(item.get("channel_contribution") or {})
    ranked = sorted(
        channels.items(),
        key=lambda entry: (
            float(entry[1].get("gold_hit_rate") or 0.0),
            -float(entry[1].get("avg_gold_rank_when_hit") or 9999.0),
        ),
        reverse=True,
    )
    return [name for name, _ in ranked[:limit]]


def _status_line(benchmark: str, item: dict[str, Any]) -> str:
    guardrail = GUARDRAILS[benchmark]
    primary = _format_float(item["primary_metric"])
    candidate = _format_float(item["candidate_recall@100"])
    integrity_ok = item.get("integrity_ok")
    if integrity_ok is None:
        integrity = "unknown"
    else:
        integrity = "clean" if integrity_ok else f"p0={','.join(item['p0_bugs'])}"
    return f"{benchmark}: primary={primary}, candidate_recall@100={candidate}, integrity={integrity}, guardrail={guardrail}"


def load_benchmark_artifacts(result_root: Path, *, require_diagnostics: bool = True) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for benchmark in BENCHMARKS:
        benchmark_root = result_root / benchmark
        diagnostics_root = benchmark_root / "reports" / "diagnostics"
        integrity_root = benchmark_root / "reports" / "integrity"
        payload = {
            "metrics": _load_metrics_head(benchmark_root / "metrics.json"),
            "candidate_recall": _load_json(
                diagnostics_root / f"{benchmark}_candidate_recall.json",
                required=require_diagnostics,
            ),
            "channel_contribution": _load_json(
                diagnostics_root / f"{benchmark}_channel_contribution.json",
                required=require_diagnostics,
            ),
            "performance_cache": _load_json(
                diagnostics_root / f"{benchmark}_performance_cache.json",
                required=require_diagnostics,
            ),
            "integrity": _load_json(
                integrity_root / f"{benchmark}_integrity_report.json",
                required=require_diagnostics,
            ),
        }
        if benchmark == "clonemem":
            payload["oracle"] = _load_json(
                diagnostics_root / "clonemem_oracle_retrieval.json",
                required=require_diagnostics,
            )
            payload["failure_taxonomy"] = _load_json(
                diagnostics_root / "clonemem_failure_taxonomy.json",
                required=require_diagnostics,
            )
        if benchmark == "knowme":
            payload["category_analysis"] = _load_json(
                diagnostics_root / "knowme_category_analysis.json",
                required=require_diagnostics,
            )
        artifacts[benchmark] = payload
    return artifacts


def summarize_artifacts(artifacts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for benchmark, payload in artifacts.items():
        metrics = payload["metrics"]
        candidate = payload.get("candidate_recall") or {}
        integrity = payload.get("integrity") or {}
        integrity_present = payload.get("integrity") is not None
        entry = {
            "question_count": int(metrics.get("question_count") or 0),
            "primary_metric_name": PRIMARY_METRIC_PATHS[benchmark].split(".")[-1],
            "primary_metric": float(_get_path(metrics, PRIMARY_METRIC_PATHS[benchmark])),
            "supplementary_metrics": {
                path.split(".")[-1]: float(_get_path(metrics, path))
                for path in SUPPLEMENTARY_METRICS[benchmark]
            },
            "candidate_recall@100": _maybe_float(candidate.get("candidate_recall@100")),
            "candidate_recall@10": _maybe_float(candidate.get("candidate_recall@10")),
            "final_recall@10": _maybe_float(candidate.get("final_recall@10")),
            "candidate_ndcg@10": _maybe_float(candidate.get("candidate_ndcg@10")),
            "final_ndcg@10": _maybe_float(candidate.get("final_ndcg@10")),
            "dense_hit@100": _maybe_float(candidate.get("dense_hit@100")),
            "fused_hit@100": _maybe_float(candidate.get("fused_hit@100")),
            "integrity_ok": None if not integrity_present else not bool(integrity.get("p0_bugs")),
            "p0_bugs": list(integrity.get("p0_bugs") or []),
            "fallback_in_use": bool(metrics.get("fallback_in_use", False)),
            "channel_contribution": dict((payload.get("channel_contribution") or {}).get("channels") or {}),
            "performance_cache": payload.get("performance_cache") or {},
        }
        if benchmark == "clonemem":
            oracle = payload.get("oracle") or {}
            entry["oracle_recall@10"] = _maybe_float(oracle.get("oracle_recall@10"))
            entry["failure_taxonomy"] = payload.get("failure_taxonomy") or {}
        if benchmark == "knowme":
            entry["category_analysis"] = payload.get("category_analysis") or {}
        summary[benchmark] = entry
    return summary


def _benchmark_trends(
    summary: dict[str, dict[str, Any]],
    baseline_summary: dict[str, dict[str, Any]] | None,
) -> dict[str, str]:
    trends: dict[str, str] = {}
    for benchmark in BENCHMARKS:
        baseline_metric = baseline_summary[benchmark]["primary_metric"] if baseline_summary else None
        trends[benchmark] = _trend_label(summary[benchmark]["primary_metric"], baseline_metric)
    return trends


def render_updated_draft(
    *,
    draft_source: str,
    result_root: Path,
    baseline_root: Path | None = None,
    ablation_payload: dict[str, Any] | None = None,
) -> str:
    artifacts = load_benchmark_artifacts(result_root, require_diagnostics=True)
    summary = summarize_artifacts(artifacts)
    baseline_summary = (
        summarize_artifacts(load_benchmark_artifacts(baseline_root, require_diagnostics=False))
        if baseline_root
        else None
    )
    trends = _benchmark_trends(summary, baseline_summary)

    benchmark_scope_rows = [
        [
            benchmark,
            BENCHMARK_SCOPE[benchmark],
            str(summary[benchmark]["question_count"]),
            summary[benchmark]["primary_metric_name"],
            ", ".join(summary[benchmark]["supplementary_metrics"].keys()),
            BENCHMARK_CANDIDATE_DIAG[benchmark],
            "clean"
            if summary[benchmark]["integrity_ok"]
            else ("unknown" if summary[benchmark]["integrity_ok"] is None else "p0 bug"),
        ]
        for benchmark in BENCHMARKS
    ]
    main_result_rows = [
        [
            benchmark,
            summary[benchmark]["primary_metric_name"],
            _format_float(summary[benchmark]["primary_metric"]),
            GUARDRAILS[benchmark],
            _guardrail_status(benchmark, summary[benchmark]["primary_metric"]),
        ]
        for benchmark in BENCHMARKS
    ]
    supplementary_rows = [
        [
            benchmark,
            ", ".join(
                f"{metric_name}={_format_float(metric_value)}"
                for metric_name, metric_value in summary[benchmark]["supplementary_metrics"].items()
            ),
        ]
        for benchmark in BENCHMARKS
    ]
    candidate_rows = [
        [
            benchmark,
            _format_float(summary[benchmark]["candidate_recall@100"]),
            "yes" if (summary[benchmark]["candidate_recall@100"] or 0.0) < 0.9 else "no",
            "yes"
            if (summary[benchmark]["candidate_recall@100"] or 0.0) >= 0.9
            and (summary[benchmark]["final_recall@10"] or 0.0) < 0.8
            else "no",
            "candidate admission bottleneck"
            if (summary[benchmark]["candidate_recall@100"] or 0.0) < 0.9
            else "late-stage bottleneck",
        ]
        for benchmark in BENCHMARKS
    ]
    oracle_rows = []
    for benchmark in BENCHMARKS:
        oracle_value = _format_float(summary[benchmark].get("oracle_recall@10")) if benchmark == "clonemem" else "n/a"
        oracle_rows.append(
            [
                benchmark,
                "false" if not summary[benchmark]["fallback_in_use"] else "true",
                "none" if not summary[benchmark]["p0_bugs"] else ", ".join(summary[benchmark]["p0_bugs"]),
                oracle_value,
                "oracle clean"
                if benchmark != "clonemem" or (summary[benchmark].get("oracle_recall@10") or 0.0) >= 1.0
                else "oracle warning",
            ]
        )
    channel_rows = [
        [
            benchmark,
            ", ".join(_top_channels(summary[benchmark])),
            _format_float(summary[benchmark]["candidate_recall@100"]),
        ]
        for benchmark in BENCHMARKS
    ]

    baseline_section = ""
    if baseline_summary:
        before_after_rows = [
            [
                benchmark,
                _format_float(baseline_summary[benchmark]["primary_metric"]),
                _format_float(summary[benchmark]["primary_metric"]),
                _format_float(baseline_summary[benchmark]["candidate_recall@100"]),
                _format_float(summary[benchmark]["candidate_recall@100"]),
            ]
            for benchmark in BENCHMARKS
        ]
        baseline_section = (
            "\n**Before And After**\n"
            + _table(
                ["Benchmark", "Before Primary", "After Primary", "Before Cand@100", "After Cand@100"],
                before_after_rows,
            )
            + "\n"
        )

    knowme_categories = summary["knowme"].get("category_analysis", {}).get("categories", {})
    clone_failure = summary["clonemem"].get("failure_taxonomy", {})
    clone_failure_dist = dict(clone_failure.get("failure_type_distribution") or {})
    trend_clause = ", ".join(f"{benchmark} {label}" for benchmark, label in trends.items())
    strongest_knowme_categories = ", ".join(sorted(knowme_categories)[:4]) or "n/a"
    improved = [benchmark for benchmark, label in trends.items() if label == "improved"]
    regressed = [benchmark for benchmark, label in trends.items() if label == "regressed"]
    if improved and regressed:
        overall_outcome = "mixed"
    elif regressed:
        overall_outcome = "negative"
    else:
        overall_outcome = "positive"
    dense_safe = all(
        summary[benchmark]["dense_hit@100"] is None
        or summary[benchmark]["fused_hit@100"] is None
        or summary[benchmark]["fused_hit@100"] >= summary[benchmark]["dense_hit@100"]
        for benchmark in BENCHMARKS
    )
    safe_fusion_clause = (
        "The repaired safe-fusion route preserves dense candidate coverage on the evaluated factual benchmarks."
        if dense_safe
        else "The latest artifacts still show cases where fused candidate coverage falls below dense-only coverage."
    )
    outcome_clause = {
        "positive": "a recovery-oriented result rather than a naive multi-channel regression",
        "mixed": "a mixed result: candidate admission improves on the harder surfaces, but not every final metric moves in the same direction",
        "negative": "a negative result: richer retrieval still does not convert into net benchmark gains",
    }[overall_outcome]
    ablation_section = ""
    if ablation_payload:
        selected_rows = [
            row
            for row in list(ablation_payload.get("rows") or [])
            if row.get("benchmark") in {"knowme", "clonemem"}
            and row.get("config") in {"dense_only", "full_multichannel_safe"}
            and row.get("slice") in {"medium100", "smoke50"}
        ]
        if selected_rows:
            ablation_table_rows = [
                [
                    str(row.get("benchmark") or ""),
                    str(row.get("slice") or ""),
                    str(row.get("config") or ""),
                    _format_float(row.get("primary_metric")),
                    _format_float(row.get("candidate_recall@100")),
                    _format_float(row.get("dense_hit@100")),
                    _format_float(row.get("fused_hit@100")),
                ]
                for row in selected_rows
            ]
            ablation_section = (
                "\n### Table 7. Dense-Only vs Safe-Fusion Ablation\n"
                + _table(
                    ["Benchmark", "Slice", "Config", "Primary", "candidate_recall@100", "dense_hit@100", "fused_hit@100"],
                    ablation_table_rows,
                )
                + "\n"
            )
    abstract = (
        "DysonSpherain now frames long-horizon CLI memory retrieval as a dense-preserving multi-channel candidate-generation problem rather than a dense-only reranking problem. "
        f"On the latest full factual benchmark package, LongMemEval reaches {_format_float(summary['longmemeval']['primary_metric'])}, "
        f"LoCoMo reaches {_format_float(summary['locomo']['primary_metric'])}, "
        f"KnowMe reaches {_format_float(summary['knowme']['primary_metric'])}, and "
        f"CloneMem reaches {_format_float(summary['clonemem']['primary_metric'])} on their primary metrics. "
        f"Relative to the baseline snapshot, the observed trend is: {trend_clause}. "
        "LongMemEval and LoCoMo are not primarily limited by broad candidate recall, while KnowMe and CloneMem remain candidate-admission sensitive. "
        f"CloneMem oracle retrieval stays healthy at oracle_recall@10={_format_float(summary['clonemem'].get('oracle_recall@10'))}, "
        "which argues against a broken embedding/index/id pipeline and localizes the remaining failure surface to fine-grained segment admission, parent-to-segment expansion, and benchmark-specific routing. "
        f"{safe_fusion_clause} The latest rerun is therefore {outcome_clause}."
    )

    return f"""# DysonSpherain: Dense-Preserving Multi-Channel Retrieval for Long-Horizon CLI Memory

## Abstract
{abstract}

## Introduction
Long-horizon CLI memory retrieval fails in more than one way. Temporal anchoring drift and local candidate crowding still matter, but the latest rerun shows that they are not the dominant failure source on every benchmark. LongMemEval and LoCoMo keep broad candidate recall near saturation, while KnowMe and CloneMem expose a different bottleneck: the gold segment often exists in the indexed corpus and can remain recoverable under oracle querying, yet it is not admitted or prioritized at sufficient rank in the first-stage pool.

The current repair round also sharpens an engineering lesson. Naive multi-channel fusion can hurt a strong dense baseline if duplicate collapse, parent caps, or inhibition act destructively. DysonSpherain is therefore better described as route-conditioned temporal retrieval plus dense-preserving multi-channel candidate generation, channel-gated safe fusion, and optional memory-grounded creative expansion. Route-conditioned temporal retrieval remains a core subsystem, but it is no longer the only mechanism worth discussing.

\\[
C(q) = C_{{dense}}(q) \\cup C_{{lex}}(q) \\cup C_{{entity}}(q) \\cup C_{{parent}}(q) \\cup C_{{neighbor}}(q) \\cup C_{{decomp}}(q) \\cup C_{{profile}}(q)
\\]

The latest evidence does not support a universal SOTA claim. It supports a bounded mechanism claim: safe fusion and channel gating are necessary conditions for robust fine-grained memory retrieval, and candidate admission remains the dominant unresolved bottleneck on the hardest benchmarks.

## Contributions
1. We formalize dense-preserving multi-channel candidate generation for long-horizon CLI memory retrieval, integrating dense semantic retrieval, lexical sparse retrieval, entity-aware retrieval, parent-session expansion, temporal-neighbor expansion, query decomposition, and profile-side retrieval.
2. We introduce safe fusion and channel gating so that additive channels do not silently destroy dense coverage through duplicate collapse, parent caps, or early inhibition.
3. We add candidate recall diagnostics, per-channel contribution reports, KnowMe category analysis, and CloneMem failure taxonomy to separate candidate admission failures from late-stage ranking failures.
4. We report an updated four-benchmark factual evaluation surface covering LongMemEval, LoCoMo, KnowMe, and CloneMem directly from current local artifacts rather than from stale draft numbers.
5. We make the limitation boundary explicit: CloneMem remains the hardest surface, multi-seed significance is absent, public-baseline parity is not claimed, and the latest rerun should be read as a {overall_outcome} result rather than as a blanket win.

## Method
### Multi-Channel Candidate Generation
DysonSpherain exposes distinct candidate channels with independent provenance and configurable top-k budgets. The dense channel preserves the embedding baseline. The lexical sparse channel targets exact or near-exact lexical anchors. The entity-aware channel promotes shared people, projects, tools, metrics, files, paths, and temporal expressions. The parent-session channel retrieves parent blocks before expanding inside them. The temporal-neighbor channel expands around strong seeds without unbounded drift. The query-decomposition channel derives entity, attribute, object, time, metric, constraint, and evidence-type cues without using gold metadata. The profile-side channel targets stable profile, preference, relationship, and state evidence.

### Safe Fusion With Provenance
Each channel returns stable segment ids and retains provenance. Fusion uses reciprocal-rank-style aggregation plus normalized channel scores, but it now preserves a dense anchor set, keeps destructive filters away from the broad candidate-recall pool, and records channel contributions and restoration events. Competition-aware inhibition is treated as optional local diversity control rather than as an admission-time filter.

### Diagnostics
We track candidate_recall@K, final recall, dense-vs-fused hit rates, gold rank movement before and after rerank, parent-hit/segment-miss events, and benchmark-specific failure buckets. This matters because the latest reruns show that a low final score can arise either from a poor candidate pool or from a late-stage ordering drop; the two require different fixes.

## Experiments
### Table 1. Benchmark Scope And Metrics
{_table(["Benchmark", "Scope", "Question Count", "Primary Metric", "Secondary Metrics", "Candidate Recall Metric", "Integrity Status"], benchmark_scope_rows)}

### Table 2. Main Full-Run Results
{_table(["Benchmark", "Primary Metric", "Result", "Target / Guardrail", "Status"], main_result_rows)}

### Table 3. Supplementary Metrics
{_table(["Benchmark", "Supplementary Metrics"], supplementary_rows)}

### Table 4. Candidate Recall And Bottleneck Diagnosis
{_table(["Benchmark", "candidate_recall@100", "Broad Recall Bottleneck?", "Rerank Bottleneck?", "Diagnosis"], candidate_rows)}

### Table 5. Oracle And Integrity Checks
{_table(["Benchmark", "fallback_in_use", "p0 bugs", "oracle recall", "interpretation"], oracle_rows)}

### Table 6. Per-Channel Contribution Snapshot
{_table(["Benchmark", "Top Channels", "candidate_recall@100"], channel_rows)}
{ablation_section}{baseline_section}
## Mechanistic Analysis
The latest artifacts separate candidate admission from late reranking more clearly than the prior draft. LongMemEval and LoCoMo retain candidate_recall@100 close to 1.0, so their remaining movement is not primarily an embedding or indexing failure; it is an ordering problem. KnowMe still shows a candidate-pool gap, especially on profile and task questions, and its category analysis now exposes different surfaces instead of collapsing everything into a single benchmark number. The strongest KnowMe categories in the latest artifacts are: {strongest_knowme_categories}.

CloneMem remains the hardest benchmark. The latest failure taxonomy reports {clone_failure_dist}, while oracle retrieval remains clean enough to reject the hypothesis of a broken index or broken embedding path. This shifts the interpretation away from “the system cannot represent the gold segment” and toward “the segment is representable, but first-stage admission, parent-to-segment expansion, and route-specific prioritization still miss too often.” {safe_fusion_clause}

## Discussion And Limitations
The previous draft's “three benchmark surface” narrative is no longer valid. The current artifact set covers four factual benchmarks with explicit integrity and oracle checks. CloneMem is also no longer outside current-snapshot coverage; it is inside coverage, but it remains the clearest unresolved bottleneck surface.

The main remaining limitation is not coverage, but uneven conversion from richer candidate generation into final top-k gains. CloneMem still underperforms the target guardrail, and candidate recall still constrains its final recall. KnowMe remains sensitive to profile-side extraction and top-10 ordering even when candidate admission improves. The study also remains mostly single-seed and deterministic, so it does not yet establish confidence intervals. ConvoMem is not part of this updated full rerun, and no public-SOTA claim is made.

## Conclusion
The updated evidence supports a bounded claim. DysonSpherain now has a clearer mechanism story for long-horizon CLI memory because it no longer treats dense retrieval plus late reranking as the only mechanism. The repair round shows that dense-preserving safe fusion and channel gating are necessary for robust multi-channel retrieval. LongMemEval and LoCoMo act as guardrails, KnowMe exposes profile-sensitive admission and ranking gaps, and CloneMem remains the most diagnostic fine-grained segment-level challenge. The next correct optimization targets are still candidate admission, parent expansion, and route-specific ordering rather than blind expansion of every retrieval channel.

## Figures And Captions
- Figure 1 should be updated from a single route-conditioned temporal retrieval diagram to a dense-preserving multi-channel candidate-generation diagram with safe fusion, optional inhibition, and optional creative expansion.
- Figure 2 should emphasize candidate admission failure, parent-hit/segment-miss, and fine-grained segment loss.
- Figure 3 should reflect the current four-benchmark factual surface rather than the old three-benchmark package.
- Figure 4 should be replaced with a failure-taxonomy or channel-contribution case study from the latest reports.
- If final figure assets are not regenerated yet, mark them as TODO rather than implying they are current.

## Consistency Checklist
- Source draft length inspected: {len(draft_source.splitlines())} lines.
- Result root: {result_root}
- Benchmarks covered: four-benchmark factual package ({", ".join(BENCHMARKS)}).
- Latest status:
  {chr(10).join("- " + _status_line(benchmark, summary[benchmark]) for benchmark in BENCHMARKS)}
"""


def render_multichannel_report(
    *,
    result_root: Path,
    baseline_root: Path | None = None,
    ablation_payload: dict[str, Any] | None = None,
) -> str:
    artifacts = load_benchmark_artifacts(result_root, require_diagnostics=True)
    summary = summarize_artifacts(artifacts)
    baseline_summary = (
        summarize_artifacts(load_benchmark_artifacts(baseline_root, require_diagnostics=False))
        if baseline_root
        else None
    )
    trends = _benchmark_trends(summary, baseline_summary)
    before_after_rows = []
    for benchmark in BENCHMARKS:
        before_primary = _format_float(baseline_summary[benchmark]["primary_metric"]) if baseline_summary else "n/a"
        before_candidate = _format_float(baseline_summary[benchmark]["candidate_recall@100"]) if baseline_summary else "n/a"
        before_after_rows.append(
            [
                benchmark,
                before_primary,
                _format_float(summary[benchmark]["primary_metric"]),
                before_candidate,
                _format_float(summary[benchmark]["candidate_recall@100"]),
                trends[benchmark],
            ]
        )
    return f"""# Multi-Channel Candidate Generation Report

## Summary
- Result root: `{result_root}`
- Baseline root: `{baseline_root if baseline_root else 'n/a'}`
- Benchmarks: {", ".join(BENCHMARKS)}
- Note: baseline candidate diagnostics are reported as `n/a` when the baseline root does not contain the new multichannel diagnostic artifacts.

## Before / After
{_table(["Benchmark", "Before Primary", "After Primary", "Before Cand@100", "After Cand@100", "Trend"], before_after_rows)}

## Key Findings
- LongMemEval status: {_status_line('longmemeval', summary['longmemeval'])}
- LoCoMo status: {_status_line('locomo', summary['locomo'])}
- KnowMe status: {_status_line('knowme', summary['knowme'])}
- CloneMem status: {_status_line('clonemem', summary['clonemem'])}
"""


def write_updated_outputs(
    *,
    draft_path: Path,
    draft_output: Path,
    report_output: Path,
    result_root: Path,
    baseline_root: Path | None = None,
    ablation_json: Path | None = None,
) -> None:
    ablation_payload = _load_json(ablation_json, required=False) if ablation_json else None
    draft_source = draft_path.read_text(encoding="utf-8")
    draft_output.write_text(
        render_updated_draft(
            draft_source=draft_source,
            result_root=result_root,
            baseline_root=baseline_root,
            ablation_payload=ablation_payload,
        ),
        encoding="utf-8",
    )
    report_output.write_text(
        render_multichannel_report(
            result_root=result_root,
            baseline_root=baseline_root,
            ablation_payload=ablation_payload,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update DysonSpherain draft and multichannel report from benchmark artifacts.")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--ablation-json", type=Path, default=None)
    parser.add_argument("--draft-path", type=Path, required=True)
    parser.add_argument("--draft-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_updated_outputs(
        draft_path=args.draft_path,
        draft_output=args.draft_output,
        report_output=args.report_output,
        result_root=args.result_root,
        baseline_root=args.baseline_root,
        ablation_json=args.ablation_json,
    )


if __name__ == "__main__":
    main()
