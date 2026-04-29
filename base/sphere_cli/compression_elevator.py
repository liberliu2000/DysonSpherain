from __future__ import annotations

from typing import Any


class CompressionElevator:
    def compress_node(self, node: dict[str, Any], target_shell: int | None = None) -> dict[str, Any]:
        shell = int(node.get("shell", 2)) if target_shell is None else target_shell
        summary = node.get("summary") or ""
        raw = node.get("raw_content") or ""
        if shell == 0:
            text = self._truncate(summary, 80)
        elif shell == 1:
            text = self._truncate(summary, 160)
        elif shell == 2:
            text = self._truncate(summary or raw, 240)
        elif shell == 3:
            text = self._truncate(summary or raw, 320)
        else:
            text = self._truncate(raw or summary, 420)
        compact = dict(node)
        compact.setdefault("shell", shell)
        compact.setdefault("sector", "creative")
        compact.setdefault("zone", "sidecar")
        compact.setdefault("cell", str(node.get("id") or node.get("node_id") or "reflection"))
        compact.setdefault("molecular_type", "reflection")
        compact["compressed_text"] = text
        return compact

    def compress_text(self, text: str, target_ratio: float = 0.3, min_len: int = 120) -> str:
        normalized = " ".join((text or "").split())
        if not normalized:
            return ""
        target = max(min_len, int(len(normalized) * target_ratio))
        return self._truncate(normalized, target)

    def _truncate(self, text: str, max_len: int) -> str:
        text = " ".join(text.split())
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "…"
