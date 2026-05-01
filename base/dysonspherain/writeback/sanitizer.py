from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sphere_cli.security import SECRET_PATTERNS, redact_payload, redact_secrets


@dataclass(frozen=True)
class SanitizerResult:
    payload: dict[str, Any]
    has_redaction: bool
    redaction_count: int
    patterns_triggered: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_redaction": self.has_redaction,
            "redaction_count": self.redaction_count,
            "patterns_triggered": list(self.patterns_triggered),
        }


def sanitize_payload(payload: dict[str, Any]) -> SanitizerResult:
    raw = str(payload)
    redacted = redact_payload(payload)
    after = str(redacted)
    count = after.count("[REDACTED]")
    patterns = [getattr(pattern, "pattern", "secret") for pattern in SECRET_PATTERNS if pattern.search(raw)]
    for key in payload:
        if str(key).lower() in {"token", "password", "passwd", "pwd", "secret", "cookie"}:
            patterns.append(str(key))
    return SanitizerResult(payload=redacted, has_redaction=raw != after, redaction_count=count, patterns_triggered=sorted(set(patterns)))


@dataclass(frozen=True)
class SanitizedText:
    text: str
    has_redaction: bool
    redaction_count: int
    patterns_triggered: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "has_redaction": self.has_redaction,
            "redaction_count": self.redaction_count,
            "patterns_triggered": list(self.patterns_triggered),
        }


def sanitize_text(text: str) -> SanitizedText:
    redacted = redact_secrets(text)
    patterns = [getattr(pattern, "pattern", "secret") for pattern in SECRET_PATTERNS if pattern.search(text)]
    return SanitizedText(
        text=redacted,
        has_redaction=redacted != text,
        redaction_count=redacted.count("[REDACTED]"),
        patterns_triggered=patterns,
    )
