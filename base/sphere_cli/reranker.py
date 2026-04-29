from __future__ import annotations

from typing import Any, Protocol, Sequence, cast

from .config import AppConfig
from .utils import lexical_score, tokenize

class _CrossEncoderModel(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Any: ...


_CROSS_ENCODER_CACHE: dict[str, _CrossEncoderModel] = {}


class RuleReranker:
    def rank_evidence(self, query: str, candidates: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
        q_tokens = set(tokenize(query))
        reranked: list[dict[str, Any]] = []
        for rank, item in enumerate(candidates):
            summary = item.get("summary") or ""
            chunk = item.get("matched_chunk_text") or item.get("raw_content") or ""
            address = " ".join(str(item.get(k, "")) for k in ["sector", "zone", "cell"])
            text_blob = " ".join([summary, chunk, address]).strip()
            overlap = len(q_tokens & set(tokenize(text_blob))) / max(1, len(q_tokens))
            exact_phrase = 0.18 if query.lower() in text_blob.lower() else 0.0
            chunk_bonus = lexical_score(query, chunk) * 0.55
            summary_bonus = lexical_score(query, summary) * 0.25
            address_bonus = lexical_score(query, address) * 0.1
            vector = float(item.get("vector_score") or 0.0) * 0.45
            fusion = float(item.get("fusion_score") or 0.0) * 0.12
            score = overlap * 0.65 + exact_phrase + chunk_bonus + summary_bonus + address_bonus + vector + fusion
            enriched = dict(item)
            enriched["rerank_score"] = round(score, 4)
            enriched["rerank_method"] = "rule:evidence"
            enriched["pre_rerank_rank"] = rank + 1
            reranked.append(enriched)
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_k]

    def rank_cognitive(self, query: str, candidates: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
        q_tokens = set(tokenize(query))
        reranked: list[dict[str, Any]] = []
        for rank, item in enumerate(candidates):
            summary = item.get("summary") or ""
            chunk = item.get("matched_chunk_text") or item.get("raw_content") or ""
            address = " ".join(str(item.get(k, "")) for k in ["sector", "zone", "cell"])
            text_blob = " ".join([summary, chunk, address]).strip()
            overlap = len(q_tokens & set(tokenize(text_blob))) / max(1, len(q_tokens))
            exact_phrase = 0.15 if query.lower() in text_blob.lower() else 0.0
            chunk_bonus = lexical_score(query, chunk) * 0.45
            summary_bonus = lexical_score(query, summary) * 0.35
            address_bonus = lexical_score(query, address) * 0.2
            shell_bonus = 0.08 if int(item.get("shell", 9)) <= 2 else 0.0
            graph = float(item.get("activation_score") or 0.0) * 0.25
            vector = float(item.get("vector_score") or 0.0) * 0.35
            fusion = float(item.get("fusion_score") or 0.0) * 0.3
            score = overlap * 0.55 + exact_phrase + chunk_bonus + summary_bonus + address_bonus + shell_bonus + graph + vector + fusion
            enriched = dict(item)
            enriched["rerank_score"] = round(score, 4)
            enriched["rerank_method"] = "rule:cognitive"
            enriched["pre_rerank_rank"] = rank + 1
            reranked.append(enriched)
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_k]

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 8, channel: str = "evidence") -> list[dict[str, Any]]:
        if channel == "cognitive":
            return self.rank_cognitive(query, candidates, top_k=top_k)
        return self.rank_evidence(query, candidates, top_k=top_k)


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: _CrossEncoderModel | None = None
        self.is_available = False
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            model = _CROSS_ENCODER_CACHE.get(model_name)
            if model is None:
                model = cast(_CrossEncoderModel, CrossEncoder(model_name))
                _CROSS_ENCODER_CACHE[model_name] = model
            self._model = model
            self.is_available = True
        except Exception:
            self._model = None
            self.is_available = False

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
        if not self.is_available or self._model is None or not candidates:
            return []
        pairs: list[tuple[str, str]] = []
        for item in candidates:
            text = "\n".join(
                x for x in [item.get("summary") or "", item.get("matched_chunk_text") or item.get("raw_content") or ""] if x
            )
            pairs.append((query, text[:4000]))
        scores = [float(score) for score in self._model.predict(pairs)]
        reranked = []
        for item, score in zip(candidates, scores):
            enriched = dict(item)
            enriched["rerank_score"] = round(score, 4)
            enriched["rerank_method"] = f"cross_encoder:{self.model_name}"
            reranked.append(enriched)
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_k]


class RetrievalReranker:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.rule = RuleReranker()
        self.cross: CrossEncoderReranker | None = None

    def _get_cross(self) -> CrossEncoderReranker:
        if self.cross is None:
            self.cross = CrossEncoderReranker(self.config.cross_encoder_model_name)
        return self.cross

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 8,
        mode: str | None = None,
        channel: str = "evidence",
    ) -> list[dict[str, Any]]:
        mode = (mode or self.config.rerank_mode_default).lower()
        if mode == "cross_encoder":
            cross = self._get_cross()
            if cross.is_available:
                return cross.rerank(query, candidates, top_k=top_k)
        if mode == "hybrid":
            base = self.rule.rerank(query, candidates, top_k=max(top_k * 2, top_k), channel=channel)
            cross = self._get_cross()
            if cross.is_available:
                return cross.rerank(query, base, top_k=top_k)
            return base[:top_k]
        return self.rule.rerank(query, candidates, top_k=top_k, channel=channel)
