from __future__ import annotations

import re
from typing import Any


REDACTION = "[REDACTED]"

SECRET_PATTERNS = [
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._\-+/=]+"),
    re.compile(r"(?i)((?:api[_-]?key|access[_-]?token|secret[_-]?key|password|passwd|pwd)\s*[:=]\s*)[^\s\"']+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
]


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(lambda match: match.group(1) + REDACTION, redacted)
        else:
            redacted = pattern.sub(REDACTION, redacted)
    return redacted


def redact_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return redact_secrets(payload)
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(redact_payload(item) for item in payload)
    if isinstance(payload, dict):
        sanitized = {}
        for key, value in payload.items():
            if re.search(r"(?i)(api[_-]?key|access[_-]?token|\btoken\b|secret|password|passwd|pwd|authorization)", str(key)):
                sanitized[key] = REDACTION
            else:
                sanitized[key] = redact_payload(value)
        return sanitized
    return payload
