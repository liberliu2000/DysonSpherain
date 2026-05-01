from __future__ import annotations

from pathlib import Path


def read_transcript(path: str | Path, *, max_chars: int = 12000) -> str:
    value = Path(path)
    if not value.exists():
        return ""
    return value.read_text(encoding="utf-8", errors="replace")[-max_chars:]
