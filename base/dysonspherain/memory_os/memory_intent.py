from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


CONTINUATION_PATTERNS = [
    r"\bcontinue\b",
    r"\bresume\b",
    r"\blast\s+(session|time|conversation|window)\b",
    r"\bprevious\s+(session|task|context|window)\b",
    r"继续",
    r"接着",
    r"上次",
    r"之前",
    r"上一轮",
    r"旧窗口",
    r"新窗口",
    r"上下文衔接",
]

BENCHMARK_PATTERNS = [
    r"\bbenchmark\b",
    r"\bregression\b",
    r"\bCloneMem\b",
    r"\bKnowMe\b",
    r"\bLongMemEval\b",
    r"\bLoCoMo\b",
    r"\bNDCG\b",
    r"\brecall\b",
    r"\bgold_rank\b",
    r"\bcandidate_recall\b",
    r"基准",
    r"回归",
    r"得分",
]

PRIOR_CONTEXT_PATTERNS = [
    r"\bphase\b",
    r"\bplan\b",
    r"\broadmap\b",
    r"\bprior\b",
    r"\bdecision\b",
    r"\barchitecture\b",
    r"根据.*(文档|报告|之前|上次)",
    r"之前的?(版本|决定|策略|结果)",
    r"任务顺序",
    r"还有哪些",
]

TOKEN_ECONOMY_PATTERNS = [
    r"\btoken\s+economy\b",
    r"\btoken\s+budget\b",
    r"\bcontext\s+compression\b",
    r"\bbudget\b",
    r"token",
    r"压缩",
    r"预算",
    r"节省",
]

WRITEBACK_PATTERNS = [
    r"\bremember\b",
    r"\bsave\b",
    r"\bstore\b",
    r"\brecord\b",
    r"记住",
    r"保存",
    r"写入记忆",
]

TRIVIAL_SHORT_PROMPTS = {
    "hi",
    "hello",
    "hey",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "你好",
    "好的",
    "收到",
    "谢谢",
}


@dataclass(frozen=True)
class MemoryIntent:
    status: str
    should_call_memory: bool
    reason: str
    recommended_tools: list[str]
    token_budget: int
    confidence: float
    prompt_length: int
    task_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matches(patterns: list[str], prompt: str) -> bool:
    return any(re.search(pattern, prompt, flags=re.IGNORECASE) for pattern in patterns)


def classify_memory_intent(
    prompt: str,
    *,
    cwd: str | None = None,
    project: str = "DysonSpherain",
    task_type: str = "unknown",
) -> MemoryIntent:
    del cwd, project
    text = str(prompt or "").strip()
    lowered = text.lower()
    length = len(text)
    effective_task_type = task_type if task_type and task_type != "unknown" else "unknown"

    if not text:
        return MemoryIntent("ok", False, "empty_prompt", [], 0, 1.0, 0, effective_task_type)
    if lowered in TRIVIAL_SHORT_PROMPTS:
        return MemoryIntent("ok", False, "low_value_short_prompt", [], 0, 0.95, length, effective_task_type)

    if _matches(CONTINUATION_PATTERNS, text):
        return MemoryIntent(
            "ok",
            True,
            "cross_session_continuation",
            ["dyson_resume_context", "dyson_search_memory"],
            1200,
            0.92,
            length,
            effective_task_type,
        )

    if _matches(BENCHMARK_PATTERNS, text):
        return MemoryIntent(
            "ok",
            True,
            "benchmark_or_regression",
            ["dyson_project_state", "dyson_recall", "dyson_context_pack"],
            1600,
            0.88,
            length,
            "benchmark" if effective_task_type == "unknown" else effective_task_type,
        )

    if _matches(TOKEN_ECONOMY_PATTERNS, text):
        return MemoryIntent(
            "ok",
            True,
            "token_economy",
            ["dyson_project_state", "dyson_recall", "dyson_token_economy_eval"],
            1400,
            0.84,
            length,
            effective_task_type,
        )

    if _matches(WRITEBACK_PATTERNS, text):
        return MemoryIntent("ok", True, "writeback", ["dyson_write_memory"], 800, 0.8, length, effective_task_type)

    if _matches(PRIOR_CONTEXT_PATTERNS, text):
        return MemoryIntent(
            "ok",
            True,
            "prior_decision_or_context",
            ["dyson_project_state", "dyson_recall"],
            1400,
            0.82,
            length,
            "planning" if effective_task_type == "unknown" else effective_task_type,
        )

    if length < 20:
        return MemoryIntent("ok", False, "low_value_short_prompt", [], 0, 0.72, length, effective_task_type)

    return MemoryIntent(
        "ok",
        True,
        "non_trivial_project_prompt",
        ["dyson_project_state", "dyson_recall"],
        1200,
        0.65,
        length,
        "coding" if effective_task_type == "unknown" else effective_task_type,
    )

