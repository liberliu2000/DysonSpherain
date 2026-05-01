from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Iterable


WORD_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")
COMPOUND_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:[./@:\-][A-Za-z0-9_]+)+")
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
ALNUM_BOUNDARY_RE = re.compile(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)")
SEPARATOR_SPLIT_RE = re.compile(r"[_\-./@:]+")


def _expand_token_variants(token: str) -> tuple[str, ...]:
    queue = [str(token or "").strip()]
    ordered: list[str] = []
    seen: set[str] = set()
    while queue:
        part = queue.pop(0).strip()
        if not part:
            continue
        lowered = part.lower()
        if lowered not in seen:
            seen.add(lowered)
            ordered.append(lowered)
        separator_parts = [piece for piece in SEPARATOR_SPLIT_RE.split(part) if piece]
        if len(separator_parts) > 1:
            queue.extend(separator_parts)
        camel_parts = [piece for piece in CAMEL_BOUNDARY_RE.sub(" ", part).split() if piece]
        if len(camel_parts) > 1:
            queue.extend(camel_parts)
        alnum_parts = [piece for piece in ALNUM_BOUNDARY_RE.sub(" ", part).split() if piece]
        if len(alnum_parts) > 1:
            queue.extend(alnum_parts)
    return tuple(ordered)


@lru_cache(maxsize=50000)
def _tokenize_cached(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    # Preserve compound surface forms such as `base/sphere_cli/config.py`,
    # `recall@10`, and `python3.12`, then expand them into subparts below.
    raw_tokens = list(COMPOUND_WORD_RE.findall(text))
    raw_tokens.extend(WORD_RE.findall(text))
    for raw in raw_tokens:
        for token in _expand_token_variants(raw):
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
    return tuple(ordered)


def tokenize(text: str) -> list[str]:
    return list(_tokenize_cached(text))


def token_tuple(text: str) -> tuple[str, ...]:
    return _tokenize_cached(text)


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def lexical_score(query: str, text: str) -> float:
    q_tokens = token_tuple(query)
    t_tokens = token_tuple(text)
    if not q_tokens or not t_tokens:
        return 0.0
    return 0.55 * jaccard_similarity(q_tokens, t_tokens) + 0.45 * cosine_counter(Counter(q_tokens), Counter(t_tokens))


def deterministic_angle(seed: str) -> tuple[float, float]:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    x = int(digest[:16], 16)
    y = int(digest[16:32], 16)
    theta = (x % 628319) / 100000.0
    phi = (y % 314159) / 100000.0
    return theta, phi


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_text_for_hash(text: str | None) -> str:
    if text is None:
        return ""
    # Conservative normalization only: normalize line endings and trim
    # inconsequential outer whitespace so equal content hashes stay lossless.
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def exact_content_hash(text: str | None) -> str:
    if text is None:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_content_hash(text: str | None) -> str:
    normalized = normalize_text_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
