from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sphere_cli.project_state import get_memory, list_memories
from sphere_cli.utils import lexical_score

from .schemas import ContextPack, EvidenceItem, RelevantFile
from .token_budgeter import fit_context_pack


def _text(record: dict[str, Any]) -> str:
    return str(record.get("summary") or record.get("content") or record.get("title") or "").strip()


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def _append_record(
    pack: ContextPack,
    record: dict[str, Any],
    *,
    confidence: float = 1.0,
    include_files: bool = True,
    include_benchmarks: bool = True,
) -> None:
    body = _text(record)
    if not body:
        return
    memory_type = str(record.get("memory_type") or "fact")
    if memory_type in {"decision", "constraint"}:
        pack.prior_decisions.append(body)
    elif memory_type == "failure":
        pack.known_failures.append(body)
    elif memory_type in {"benchmark", "experiment"}:
        pack.benchmark_state.append(body)
    else:
        pack.core_evidence.append(
            EvidenceItem(
                text=body,
                memory_id=str(record.get("memory_id") or record.get("id") or ""),
                timestamp=str(record.get("updated_at") or record.get("created_at") or ""),
                source=str(record.get("source") or record.get("source_type") or ""),
                confidence=round(float(confidence), 4),
                stale_possible=False,
                uncertain=confidence < 0.15,
            )
        )
    metadata = _metadata(record)
    if include_files:
        for path in metadata.get("files_changed") or metadata.get("relevant_files") or []:
            pack.relevant_files.append(RelevantFile(path=str(path), reason=f"Referenced by memory {record.get('memory_id')}", confidence=round(float(confidence), 4)))
    if include_benchmarks:
        for result in metadata.get("benchmark_results") or []:
            pack.benchmark_state.append(str(result))


def _item_text(item: dict[str, Any]) -> str:
    return str(
        item.get("text")
        or item.get("summary")
        or item.get("object_text")
        or item.get("event_text")
        or item.get("signature")
        or ""
    ).strip()


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("chunk_id") or item.get("object_id") or item.get("node_id") or item.get("path_id") or "")


def _is_prompt_like(value: str) -> bool:
    lowered = value.lower()
    markers = (
        "prior prompt",
        "previous prompt",
        "user prompt",
        "old prompt",
        "codex prompt",
        "claude prompt",
        "prompt:",
        "<user",
        "<assistant",
    )
    return any(marker in lowered for marker in markers)


def _apply_runtime_filters(
    pack: ContextPack,
    *,
    include_files: bool = True,
    include_benchmarks: bool = True,
    include_prior_prompts: bool = True,
    freshness: str = "auto",
) -> None:
    if not include_files:
        pack.relevant_files = []
    if not include_benchmarks:
        pack.benchmark_state = []
    if not include_prior_prompts:
        pack.core_evidence = [item for item in pack.core_evidence if not _is_prompt_like(f"{item.source}\n{item.text}")]
        for attr in ("prior_decisions", "known_failures", "warnings", "recommended_next_actions"):
            setattr(pack, attr, [value for value in getattr(pack, attr) if not _is_prompt_like(str(value))])
    normalized_freshness = str(freshness or "auto").lower()
    if normalized_freshness == "recent":
        pack.core_evidence.sort(key=lambda item: item.timestamp, reverse=True)
    elif normalized_freshness == "stable":
        pack.core_evidence = [item for item in pack.core_evidence if not item.uncertain]
        pack.core_evidence.sort(key=lambda item: (not item.stale_possible, item.confidence), reverse=True)


def _paths_from_item(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("path", "file_path", "filepath", "source_path"):
        value = item.get(key)
        if value:
            paths.append(str(value))
    for key in ("files", "file_paths", "relevant_files", "files_changed"):
        raw = item.get(key)
        if isinstance(raw, (list, tuple)):
            paths.extend(str(value) for value in raw if value)
        elif raw:
            paths.append(str(raw))
    return paths


def _candidate_metric_text(item: dict[str, Any]) -> str:
    metric_keys = (
        "benchmark",
        "metric",
        "recall",
        "recall_at_10",
        "ndcg",
        "ndcg_at_10",
        "gold_rank",
        "candidate_recall",
        "candidate_recall_at_100",
    )
    metrics = {key: item.get(key) for key in metric_keys if item.get(key) is not None}
    return json.dumps(metrics, ensure_ascii=False, sort_keys=True) if metrics else ""


def _candidate_confidence(item: dict[str, Any], default: float) -> float:
    raw = item.get("score")
    if raw is None:
        raw = item.get("confidence")
    if raw is None:
        raw = item.get("evidence_score")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _append_candidate_item(pack: ContextPack, item: dict[str, Any], *, source: str, confidence_default: float = 0.0) -> None:
    confidence = _candidate_confidence(item, confidence_default)
    if str(item.get("memory_type") or "").lower() in {"decision", "constraint", "failure", "benchmark", "experiment", "fact"}:
        _append_record(pack, item, confidence=confidence)
    else:
        _append_runtime_item(pack, item, source=str(item.get("source") or item.get("channel") or source), confidence_default=confidence)
    for path in _paths_from_item(item):
        pack.relevant_files.append(RelevantFile(path=path, reason=f"Referenced by {source}", confidence=round(confidence, 4)))
    metric_text = _candidate_metric_text(item)
    if metric_text:
        pack.benchmark_state.append(metric_text)


def _flatten_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        flattened.append(item)
        for nested_key in ("ranked_items", "candidates", "memory_objects"):
            nested = item.get(nested_key)
            if isinstance(nested, list):
                flattened.extend(nested_item for nested_item in nested if isinstance(nested_item, dict))
    return flattened


def _append_runtime_item(pack: ContextPack, item: dict[str, Any], *, source: str, confidence_default: float = 0.0) -> None:
    text = _item_text(item)
    if not text:
        return
    score = item.get("score")
    if score is None:
        score = item.get("evidence_score")
    try:
        confidence = float(score)
    except (TypeError, ValueError):
        confidence = confidence_default
    pack.core_evidence.append(
        EvidenceItem(
            text=text,
            memory_id=_item_id(item),
            timestamp=str(item.get("timestamp") or item.get("created_at") or item.get("updated_at") or item.get("time_bucket") or ""),
            source=source,
            confidence=round(confidence, 4),
            stale_possible=False,
            uncertain=confidence < 0.05,
        )
    )


def _dedupe_pack(pack: ContextPack) -> None:
    seen_texts: set[str] = set()
    deduped_evidence: list[EvidenceItem] = []
    for item in pack.core_evidence:
        key = " ".join(item.text.lower().split())
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped_evidence.append(item)
    pack.core_evidence = deduped_evidence
    for attr in ("prior_decisions", "known_failures", "benchmark_state", "warnings", "recommended_next_actions"):
        values = []
        seen: set[str] = set()
        for value in getattr(pack, attr):
            key = " ".join(str(value).lower().split())
            if key and key not in seen:
                seen.add(key)
                values.append(value)
        setattr(pack, attr, values)
    file_seen: set[str] = set()
    files: list[RelevantFile] = []
    for item in pack.relevant_files:
        key = item.path
        if key in file_seen:
            continue
        file_seen.add(key)
        files.append(item)
    pack.relevant_files = files


def apply_sections(pack: ContextPack, sections: list[str] | None) -> ContextPack:
    if not sections:
        return pack
    aliases = {"next_actions": "recommended_next_actions", "token_estimate": "token_economy"}
    allowed = {aliases.get(str(section), str(section)) for section in sections}
    setattr(pack, "_included_sections", allowed)
    for attr in (
        "core_evidence",
        "prior_decisions",
        "known_failures",
        "benchmark_state",
        "relevant_files",
        "warnings",
        "recommended_next_actions",
        "token_economy",
    ):
        if attr not in allowed:
            value = {} if attr == "token_economy" else []
            setattr(pack, attr, value)
    if "summary" not in allowed and allowed:
        pack.summary = ""
    return pack


def build_pack_from_memory_ids(
    *,
    base_dir: Path,
    project: str,
    memory_ids: list[str],
    token_budget: int,
    sections: list[str] | None = None,
) -> tuple[ContextPack, dict[str, Any]]:
    pack = ContextPack(summary=f"Packed {len(memory_ids)} requested memory record(s) for project {project}.")
    missing: list[str] = []
    for memory_id in memory_ids:
        record = get_memory(base_dir, project, memory_id)
        if record is None:
            missing.append(memory_id)
            continue
        _append_record(pack, record, confidence=1.0)
    if missing:
        pack.warnings.append(f"Missing memory ids: {', '.join(missing)}")
    _dedupe_pack(pack)
    pack = apply_sections(pack, sections)
    budget = fit_context_pack(pack, token_budget)
    budget.pack.token_economy = budget.to_dict()
    return budget.pack, budget.to_dict()


def build_pack_from_runtime_result(
    *,
    project: str,
    query: str,
    run_result: dict[str, Any],
    token_budget: int,
    sections: list[str] | None = None,
    include_files: bool = True,
    include_benchmarks: bool = True,
    include_prior_prompts: bool = True,
    freshness: str = "auto",
) -> tuple[ContextPack, dict[str, Any]]:
    pack = ContextPack(summary=f"Retrieved context via DysonSpherain runtime for project {project}.")
    bundle = run_result.get("bundle")
    evidence = run_result.get("evidence")
    completion = run_result.get("completion")
    cognitive = run_result.get("cognitive")
    if bundle is not None:
        for item in getattr(bundle, "core_evidence", []) or []:
            _append_runtime_item(pack, item, source="runtime_context.core_evidence")
        for item in getattr(bundle, "supporting_context", []) or []:
            _append_runtime_item(pack, item, source="runtime_context.supporting_context")
        for item in getattr(bundle, "evidence_objects", []) or []:
            _append_runtime_item(pack, item, source="runtime_context.evidence_object")
        for path in getattr(bundle, "raw_reference_pointers", []) or []:
            pack.relevant_files.append(RelevantFile(path=str(path), reason="Runtime raw reference pointer", confidence=0.0))
        for path in getattr(bundle, "alternative_paths", []) or []:
            if isinstance(path, dict):
                pack.recommended_next_actions.append(str(path.get("summary") or path.get("signature") or json.dumps(path, ensure_ascii=False, sort_keys=True))[:500])
    if not pack.core_evidence and completion is not None:
        for item in getattr(completion, "core_evidence", []) or []:
            _append_runtime_item(pack, item, source="runtime_completion.core_evidence")
    if cognitive is not None:
        for item in getattr(cognitive, "relevant_experience", []) or []:
            _append_runtime_item(pack, item, source="runtime_cognitive.relevant_experience")
    diagnostics = {}
    if evidence is not None:
        diagnostics = getattr(evidence, "diagnostics", {}) or {}
    if not pack.core_evidence:
        pack.warnings.append("Runtime retrieval returned no packed evidence.")
    if diagnostics:
        channel_stats = diagnostics.get("channel_stats") if isinstance(diagnostics, dict) else None
        if channel_stats:
            pack.warnings.append(f"channel_stats={json.dumps(channel_stats, ensure_ascii=False, sort_keys=True)[:800]}")
    _apply_runtime_filters(
        pack,
        include_files=include_files,
        include_benchmarks=include_benchmarks,
        include_prior_prompts=include_prior_prompts,
        freshness=freshness,
    )
    _dedupe_pack(pack)
    pack = apply_sections(pack, sections)
    budget = fit_context_pack(pack, token_budget)
    budget.pack.token_economy = budget.to_dict()
    return budget.pack, budget.to_dict()


def build_pack_from_candidates(
    *,
    project: str,
    candidates: list[dict[str, Any]],
    token_budget: int,
    sections: list[str] | None = None,
    candidate_type: str = "candidate",
    include_files: bool = True,
    include_benchmarks: bool = True,
    include_prior_prompts: bool = True,
    freshness: str = "auto",
) -> tuple[ContextPack, dict[str, Any]]:
    flat_candidates = _flatten_candidates(candidates)
    pack = ContextPack(summary=f"Compressed {len(flat_candidates)} supplied {candidate_type} record(s) for project {project}.")
    for item in flat_candidates:
        _append_candidate_item(pack, item, source=f"candidate.{candidate_type}")
    if not flat_candidates:
        pack.warnings.append("No candidate records were supplied.")
    _apply_runtime_filters(
        pack,
        include_files=include_files,
        include_benchmarks=include_benchmarks,
        include_prior_prompts=include_prior_prompts,
        freshness=freshness,
    )
    _dedupe_pack(pack)
    pack = apply_sections(pack, sections)
    budget = fit_context_pack(pack, token_budget)
    budget.pack.token_economy = budget.to_dict()
    return budget.pack, budget.to_dict()


def build_pack(
    *,
    base_dir: Path,
    project: str,
    query: str,
    token_budget: int,
    include_files: bool = True,
    include_benchmarks: bool = True,
) -> tuple[ContextPack, dict[str, Any]]:
    records = list_memories(base_dir, project)
    scored = sorted(
        ((lexical_score(query, _text(record)), record) for record in records),
        key=lambda item: (item[0], str(item[1].get("updated_at") or item[1].get("created_at") or "")),
        reverse=True,
    )
    selected = [(score, record) for score, record in scored if score > 0][:16] or scored[:8]
    pack = ContextPack()
    pack.summary = f"Retrieved {len(selected)} relevant memory record(s) for project {project}."
    for score, record in selected:
        _append_record(pack, record, confidence=float(score), include_files=include_files, include_benchmarks=include_benchmarks)
    if not selected:
        pack.warnings.append("No project memory records were available.")
    pack.recommended_next_actions.append("Inspect cited files/artifacts before changing retrieval or benchmark behavior.")
    _dedupe_pack(pack)
    budget = fit_context_pack(pack, token_budget)
    budget.pack.token_economy = budget.to_dict()
    return budget.pack, budget.to_dict()
