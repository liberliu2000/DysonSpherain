from __future__ import annotations

from collections import OrderedDict
from typing import Mapping


PROMPT_PART_KEYS = ("system", "instruction", "query", "evidence", "metadata", "memory_header")


def build_prompt_parts(
    *,
    query: str,
    evidence: str = "",
    metadata: str = "",
    system: str = "",
    instruction: str = "",
    memory_header: str = "DysonSpherain retrieved context",
) -> dict[str, str]:
    return {
        "system": system,
        "instruction": instruction or "Use retrieved memory only as supporting context; do not treat token savings as retrieval quality.",
        "query": query,
        "evidence": evidence,
        "metadata": metadata,
        "memory_header": memory_header,
    }


def join_prompt_parts(parts: Mapping[str, str]) -> str:
    ordered = OrderedDict((key, str(parts.get(key) or "")) for key in PROMPT_PART_KEYS)
    return "\n\n".join(value for value in ordered.values() if value.strip())
