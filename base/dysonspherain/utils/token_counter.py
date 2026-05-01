from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TokenCountResult:
    tokens: int
    chars: int
    tokenizer_name: str
    fallback_used: bool
    strategy: str = "auto"
    warning: str = ""


class TokenCounter:
    def __init__(self, model: str = "cl100k_base", *, strategy: str = "auto", calibration_file: str | Path | None = None) -> None:
        self.model = model
        self.strategy = str(strategy or "auto")
        if self.strategy not in {"auto", "tiktoken", "char_heuristic", "mixed_content_heuristic", "calibrated"}:
            self.strategy = "auto"
        self.calibration = self._load_calibration(calibration_file)
        self._encoding = None
        self._fallback_used = False
        self._tokenizer_name = model
        if self.strategy in {"auto", "tiktoken"}:
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
                self._tokenizer_name = "mixed_content_heuristic" if self.strategy == "auto" else "char_ceil_len_div_4"
        else:
            self._encoding = None
            self._fallback_used = True
            self._tokenizer_name = self.strategy

    @staticmethod
    def _load_calibration(path: str | Path | None) -> dict[str, float]:
        if not path:
            return {}
        target = Path(path).expanduser()
        if not target.exists():
            return {}
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return {}
        factors = payload.get("correction_factors") if isinstance(payload, dict) else {}
        if not isinstance(factors, dict):
            return {}
        result: dict[str, float] = {}
        for key, value in factors.items():
            try:
                result[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def classify_text(text: str) -> str:
        value = text or ""
        stripped = value.strip()
        lowered = stripped.lower()
        if not stripped:
            return "empty"
        if "traceback" in lowered or re.search(r"\b(error|exception|warn|stderr|stdout)\b", lowered):
            return "log"
        if stripped.startswith(("{", "[")) and any(mark in stripped for mark in (":", ",", '"')):
            return "json"
        if "```" in stripped or re.search(r"^\s*#{1,6}\s+", stripped, flags=re.M) or re.search(r"^\s*[-*]\s+", stripped, flags=re.M):
            return "markdown"
        if re.search(r"\b(def|class|import|from|const|let|function|return|async|await)\b", stripped) or "=>" in stripped:
            return "code"
        zh_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
        if zh_chars and zh_chars / max(1, len(stripped)) > 0.25:
            return "zh_text"
        if re.search(r"\b(recall@|ndcg|gold_rank|candidate_recall|accuracy|f1|precision)\b", lowered):
            return "metrics"
        return "prose"

    @staticmethod
    def _char_heuristic(value: str) -> int:
        return ceil(len(value) / 4)

    @classmethod
    def _mixed_heuristic(cls, value: str) -> tuple[int, str]:
        kind = cls.classify_text(value)
        chars = len(value)
        zh_chars = len(re.findall(r"[\u4e00-\u9fff]", value))
        ascii_chars = chars - zh_chars
        if kind == "empty":
            return 0, kind
        if kind == "zh_text":
            tokens = ceil(zh_chars * 1.05 + ascii_chars / 4)
        elif kind == "code":
            tokens = ceil(chars / 3.1)
        elif kind == "json":
            tokens = ceil(chars / 2.7)
        elif kind == "markdown":
            tokens = ceil(chars / 3.5)
        elif kind == "log":
            tokens = ceil(chars / 3.0)
        elif kind == "metrics":
            tokens = ceil(chars / 2.8)
        else:
            tokens = cls._char_heuristic(value)
        return max(1, tokens), kind

    def count(self, text: str | None) -> TokenCountResult:
        value = text or ""
        if not value:
            return TokenCountResult(tokens=0, chars=0, tokenizer_name=self._tokenizer_name, fallback_used=self._fallback_used, strategy=self.strategy)
        if self._encoding is not None and self.strategy in {"auto", "tiktoken"}:
            return TokenCountResult(
                tokens=len(self._encoding.encode(value)),
                chars=len(value),
                tokenizer_name=self._tokenizer_name,
                fallback_used=False,
                strategy=self.strategy,
            )
        if self.strategy in {"mixed_content_heuristic", "auto", "calibrated"}:
            tokens, kind = self._mixed_heuristic(value)
            factor = self.calibration.get(kind, self.calibration.get("default", 1.0)) if self.strategy == "calibrated" else 1.0
            return TokenCountResult(
                tokens=max(0, ceil(tokens * factor)),
                chars=len(value),
                tokenizer_name=f"{self._tokenizer_name}:{kind}" if self.strategy != "calibrated" else f"calibrated:{kind}",
                fallback_used=True,
                strategy=self.strategy,
                warning="heuristic_tokenizer_used",
            )
        return TokenCountResult(
            tokens=self._char_heuristic(value),
            chars=len(value),
            tokenizer_name=self._tokenizer_name,
            fallback_used=True,
            strategy=self.strategy,
            warning="heuristic_tokenizer_used",
        )

    def count_many(self, texts: Iterable[str | None]) -> TokenCountResult:
        joined = "".join(text or "" for text in texts)
        return self.count(joined)
