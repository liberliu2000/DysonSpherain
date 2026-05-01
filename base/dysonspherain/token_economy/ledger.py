from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

LEDGER_VERSION = "token-economy-ledger-v1"

Adapter = Literal["claude_hook", "codex_mcp", "cli", "daemon", "benchmark"]
Decision = Literal["inject", "skip", "inject_summary_only", "return_file_refs_only"]
Risk = Literal["low", "medium", "high"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def preview_query(query: str, limit: int = 160) -> str:
    return " ".join(str(query or "").split())[:limit]


def hash_query(query: str) -> str:
    return hashlib.sha256(str(query or "").encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TokenEconomyEvent:
    event_id: str
    timestamp: str
    project: str
    adapter: Adapter
    task_type: str
    mode: str
    query_hash: str
    query_preview: str
    decision: Decision
    risk: Risk
    reason: str
    baseline_type: str
    baseline_context_tokens: int
    candidate_context_tokens: int
    final_injected_tokens: int
    estimated_saved_tokens: int
    compression_ratio: float
    duplicate_token_ratio: float
    protected_evidence_tokens: int
    dropped_evidence_count: int
    fallback_tokenizer_used: bool
    tokenizer_name: str
    quality_guard_status: str
    source_files: list[str] = field(default_factory=list)
    ledger_version: str = LEDGER_VERSION
    local_compute_economy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_event_id(payload: dict[str, Any]) -> str:
    stable = {
        "timestamp": payload.get("timestamp"),
        "project": payload.get("project"),
        "adapter": payload.get("adapter"),
        "query_hash": payload.get("query_hash"),
        "decision": payload.get("decision"),
        "baseline": payload.get("baseline_context_tokens"),
        "candidate": payload.get("candidate_context_tokens"),
        "final": payload.get("final_injected_tokens"),
    }
    return "te_" + hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_token_economy_event(
    *,
    project: str,
    query: str,
    decision: str,
    adapter: str = "cli",
    task_type: str = "unknown",
    mode: str = "conservative",
    risk: str = "low",
    reason: str = "",
    baseline_type: str = "full_history",
    baseline_context_tokens: int = 0,
    candidate_context_tokens: int = 0,
    final_injected_tokens: int = 0,
    estimated_saved_tokens: int | None = None,
    duplicate_token_ratio: float = 0.0,
    protected_evidence_tokens: int = 0,
    dropped_evidence_count: int = 0,
    fallback_tokenizer_used: bool = False,
    tokenizer_name: str = "",
    quality_guard_status: str = "unknown",
    source_files: list[str] | None = None,
    local_compute_economy: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> TokenEconomyEvent:
    baseline = max(0, int(baseline_context_tokens or 0))
    candidate = max(0, int(candidate_context_tokens or 0))
    final = max(0, int(final_injected_tokens or 0))
    saved = max(0, baseline - final) if estimated_saved_tokens is None else max(0, int(estimated_saved_tokens or 0))
    compression = (final / baseline) if baseline else 0.0
    ts = timestamp or now_iso()
    payload = {
        "timestamp": ts,
        "project": str(project or "DysonSpherain"),
        "adapter": _coerce_adapter(adapter),
        "query_hash": hash_query(query),
        "decision": _coerce_decision(decision),
        "baseline_context_tokens": baseline,
        "candidate_context_tokens": candidate,
        "final_injected_tokens": final,
    }
    return TokenEconomyEvent(
        event_id=make_event_id(payload),
        timestamp=ts,
        project=payload["project"],
        adapter=payload["adapter"],
        task_type=str(task_type or "unknown"),
        mode=str(mode or "conservative"),
        query_hash=payload["query_hash"],
        query_preview=preview_query(query),
        decision=payload["decision"],
        risk=_coerce_risk(risk),
        reason=str(reason or ""),
        baseline_type=str(baseline_type or "full_history"),
        baseline_context_tokens=baseline,
        candidate_context_tokens=candidate,
        final_injected_tokens=final,
        estimated_saved_tokens=saved,
        compression_ratio=compression,
        duplicate_token_ratio=max(0.0, min(1.0, float(duplicate_token_ratio or 0.0))),
        protected_evidence_tokens=max(0, int(protected_evidence_tokens or 0)),
        dropped_evidence_count=max(0, int(dropped_evidence_count or 0)),
        fallback_tokenizer_used=bool(fallback_tokenizer_used),
        tokenizer_name=str(tokenizer_name or ""),
        quality_guard_status=str(quality_guard_status or "unknown"),
        source_files=[str(item) for item in (source_files or []) if str(item).strip()],
        local_compute_economy=dict(local_compute_economy or {}),
    )


def _coerce_adapter(value: str) -> Adapter:
    normalized = str(value or "cli").lower()
    aliases = {
        "claude_code_user_prompt_submit": "claude_hook",
        "claude_code": "claude_hook",
        "mcp": "codex_mcp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"claude_hook", "codex_mcp", "cli", "daemon", "benchmark"}:
        normalized = "cli"
    return normalized  # type: ignore[return-value]


def _coerce_decision(value: str) -> Decision:
    normalized = str(value or "skip").lower()
    if normalized == "summary_only":
        normalized = "inject_summary_only"
    if normalized == "file_refs_only":
        normalized = "return_file_refs_only"
    if normalized not in {"inject", "skip", "inject_summary_only", "return_file_refs_only"}:
        normalized = "skip"
    return normalized  # type: ignore[return-value]


def _coerce_risk(value: str) -> Risk:
    normalized = str(value or "low").lower()
    if normalized not in {"low", "medium", "high"}:
        normalized = "low"
    return normalized  # type: ignore[return-value]
