from __future__ import annotations

import json
from typing import Iterable

from .schemas import ContextPack


def _bullet_lines(items: Iterable[str]) -> list[str]:
    return [f"- {item}" for item in items if str(item).strip()] or ["- none"]


def render_markdown(pack: ContextPack) -> str:
    lines = ["# DysonSpherain Retrieved Context", "", "## Summary"]
    included = getattr(pack, "_included_sections", None)

    def show(section: str) -> bool:
        return included is None or section in included

    if show("summary"):
        lines.extend(_bullet_lines([pack.summary] if pack.summary else []))
    else:
        lines = ["# DysonSpherain Retrieved Context"]
    if show("core_evidence"):
        lines.extend(["", "## Core Evidence"])
        lines.extend(_bullet_lines(item.text for item in pack.core_evidence))
    if show("prior_decisions"):
        lines.extend(["", "## Prior Decisions"])
        lines.extend(_bullet_lines(pack.prior_decisions))
    if show("known_failures") or show("warnings"):
        lines.extend(["", "## Known Failures / Warnings"])
        lines.extend(_bullet_lines([*(pack.known_failures if show("known_failures") else []), *(pack.warnings if show("warnings") else [])]))
    if show("benchmark_state"):
        lines.extend(["", "## Benchmark State"])
        lines.extend(_bullet_lines(pack.benchmark_state))
    if show("relevant_files"):
        lines.extend(["", "## Relevant Files"])
        lines.extend(_bullet_lines(f"`{item.path}`: {item.reason}" for item in pack.relevant_files))
    if show("recommended_next_actions"):
        lines.extend(["", "## Recommended Next Actions"])
        actions = [f"{idx}. {item}" for idx, item in enumerate(pack.recommended_next_actions, start=1)]
        lines.extend(actions or ["- none"])
    if show("token_economy"):
        lines.extend(["", "## Token Economy"])
        lines.extend(_bullet_lines(f"{key}: {value}" for key, value in sorted(pack.token_economy.items())))
    return "\n".join(lines).strip() + "\n"


def render_context_pack(pack: ContextPack, format: str = "markdown") -> str:
    if format == "json":
        return json.dumps(pack.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    return render_markdown(pack)
