from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable


@dataclass(frozen=True)
class TokenCountResult:
    tokens: int
    chars: int
    tokenizer_name: str
    fallback_used: bool


class TokenCounter:
    def __init__(self, model: str = "cl100k_base") -> None:
        self.model = model
        self._encoding = None
        self._fallback_used = False
        self._tokenizer_name = model
        try:
            import tiktoken  # type: ignore

            try:
                self._encoding = tiktoken.encoding_for_model(model)
                self._tokenizer_name = getattr(self._encoding, "name", model)
            except Exception:
                self._encoding = tiktoken.get_encoding(model)
                self._tokenizer_name = getattr(self._encoding, "name", model)
        except Exception:
            self._encoding = None
            self._fallback_used = True
            self._tokenizer_name = "char_ceil_len_div_4"

    def count(self, text: str | None) -> TokenCountResult:
        value = text or ""
        if not value:
            return TokenCountResult(tokens=0, chars=0, tokenizer_name=self._tokenizer_name, fallback_used=self._fallback_used)
        if self._encoding is not None:
            return TokenCountResult(
                tokens=len(self._encoding.encode(value)),
                chars=len(value),
                tokenizer_name=self._tokenizer_name,
                fallback_used=False,
            )
        return TokenCountResult(
            tokens=ceil(len(value) / 4),
            chars=len(value),
            tokenizer_name=self._tokenizer_name,
            fallback_used=True,
        )

    def count_many(self, texts: Iterable[str | None]) -> TokenCountResult:
        joined = "".join(text or "" for text in texts)
        return self.count(joined)
