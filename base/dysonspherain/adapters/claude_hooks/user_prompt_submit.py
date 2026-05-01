from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dysonspherain.memory_os.memory_intent import classify_memory_intent
from dysonspherain.memory_os.observation_store import resume_context, search_observations, write_token_economy_event
from dysonspherain.memory_os.recall_service import RecallRequest, recall
from dysonspherain.token_economy.evaluator import evaluate
from dysonspherain.utils.token_counter import TokenCounter
from .runtime_ledger import append_hook_event


def should_skip_prompt(prompt: str) -> bool:
    return not classify_memory_intent(prompt).should_call_memory


def infer_task_type(prompt: str) -> str:
    lowered = prompt.lower()
    if "benchmark" in lowered:
        return "benchmark"
    if any(word in lowered for word in ("bug", "fix", "error", "regression", "debug")):
        return "debug"
    if any(word in lowered for word in ("paper", "draft", "review")):
        return "paper"
    if any(word in lowered for word in ("plan", "roadmap", "phase")):
        return "planning"
    return "coding"


def main() -> None:
    payload = json.load(sys.stdin)
    prompt = str(payload.get("prompt") or "")
    cwd = str(payload.get("cwd") or os.getcwd())
    task_type = infer_task_type(prompt)
    project = str(payload.get("project") or "DysonSpherain")
    intent = classify_memory_intent(prompt, cwd=cwd, project=project, task_type=task_type)
    ledger_result = append_hook_event(cwd, "UserPromptSubmit", {**payload, "project": project})
    if not intent.should_call_memory:
        print("{}")
        return
    recommended = set(intent.recommended_tools)
    context_parts: list[str] = []
    baseline_context_tokens = 0
    if "dyson_resume_context" in recommended:
        resume = resume_context(Path(cwd).resolve(), project=project, token_budget=int(intent.token_budget or 1200))
        if resume.get("status") == "ok":
            context_parts.append(str(resume.get("rendered_context") or ""))
            baseline_context_tokens += int((resume.get("token_estimate") or {}).get("estimated_tokens") or 0)
    if "dyson_search_memory" in recommended:
        search = search_observations(Path(cwd).resolve(), project=project, query=prompt, limit=5)
        observations = search.get("observations") or []
        if observations:
            lines = ["# DysonSpherain Memory Search", ""]
            for item in observations[:5]:
                lines.append(f"- {item.get('citation')}: {item.get('title')} - {item.get('snippet')}")
            context_parts.append("\n".join(lines) + "\n")
            baseline_context_tokens += sum(int(item.get("token_cost") or 0) for item in observations[:5])
    if "dyson_recall" in recommended or not context_parts:
        pack = recall(RecallRequest(query=prompt, cwd=cwd, token_budget=int(intent.token_budget or 1600), task_type=task_type))
        baseline_context_tokens += int((pack.context_pack.get("token_economy") or {}).get("estimated_tokens_before") or pack.token_estimate.get("estimated_tokens") or 0)
        context = pack.rendered_context
        context_parts.append(context)
    candidate_context = "\n\n".join(part.strip() for part in context_parts if part.strip())
    if not candidate_context.strip():
        print("{}")
        return
    mode = "debug" if os.environ.get("DYSON_TOKEN_ECONOMY_DEBUG") == "1" else "conservative"
    decision = evaluate(
        query=prompt,
        candidate_context=candidate_context,
        baseline_context_tokens=baseline_context_tokens,
        token_budget=int(intent.token_budget or 1600),
        task_type=task_type,
        mode=mode,
    )
    final_decision = decision.decision
    if decision.decision == "skip" and intent.reason == "cross_session_continuation":
        final_decision = "inject"
    if final_decision == "skip":
        print("{}")
        return
    context = candidate_context
    if final_decision == "return_file_refs_only":
        context = "\n".join(line for line in candidate_context.splitlines() if "base/" in line or ".py" in line or "dyson://observation/" in line) or candidate_context[:1200]
    elif final_decision == "inject_summary_only":
        context = candidate_context.split("\n\n", 1)[0]
    final_count = TokenCounter().count(context)
    compression_ratio = (final_count.tokens / int(intent.token_budget or 1600)) if final_count.tokens else 0.0
    write_token_economy_event(
        Path(cwd).resolve(),
        project=project,
        session_id=str(payload.get("session_id") or ""),
        prompt=prompt,
        decision=final_decision,
        injected_tokens=final_count.tokens,
        baseline_context_tokens=baseline_context_tokens,
        estimated_saved_tokens=decision.estimated_saved_tokens,
        budget_usage_ratio=compression_ratio,
        adapter="claude_hook",
        task_type=task_type,
        mode=mode,
        risk=decision.risk,
        reason=decision.reason,
        baseline_type="full_history",
        candidate_context_tokens=decision.estimated_tokens,
        final_injected_tokens=final_count.tokens,
        duplicate_token_ratio=decision.duplication_score,
        protected_evidence_tokens=0,
        dropped_evidence_count=1 if final_decision in {"inject_summary_only", "return_file_refs_only"} else 0,
        fallback_tokenizer_used=decision.fallback_tokenizer_used,
        tokenizer_name=decision.tokenizer_name,
        quality_guard_status=decision.quality_guard_status,
        source_files=list(decision.source_files),
        metadata={"intent_reason": intent.reason, "recommended_tools": intent.recommended_tools},
    )
    if os.environ.get("DYSON_INJECT_TOKEN_ECONOMY_NOTE") == "1" or os.environ.get("DYSON_TOKEN_ECONOMY_DEBUG") == "1":
        context = context.rstrip() + f"\n\n[dyson: memory injected, {final_count.tokens} tokens, estimated saved {decision.estimated_saved_tokens}]"
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "DysonSpherain recalled context:\n" + context}, "dysonLedger": ledger_result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
