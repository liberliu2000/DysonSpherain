from __future__ import annotations

from dataclasses import asdict
from typing import Any


class EvidenceAnswerGenerator:
    """Deterministic answer generator for local/offline CLI use.

    This deliberately avoids pretending to be an LLM. It produces a concise
    answer from anchored evidence, cites memory ids/chunk ids, and abstains when
    evidence is too weak. LLM-backed generation can be layered on top later
    without changing the retrieval contract.
    """

    def generate(
        self,
        query: str,
        run_result: dict[str, Any],
        *,
        mode: str = "local",
        max_evidence: int = 5,
        include_creative: bool = False,
    ) -> dict[str, Any]:
        evidence = run_result.get("evidence")
        completion = run_result.get("completion")
        cognitive = run_result.get("cognitive")
        bundle = run_result.get("bundle")

        core = list(getattr(completion, "core_evidence", []) or [])
        objects = list(getattr(completion, "evidence_objects", []) or [])
        support = list(getattr(completion, "supporting_context", []) or [])
        creative = list(getattr(cognitive, "creative_reflections", []) or []) if include_creative else []
        ranked = self._dedupe_items(core + objects + support + creative)
        selected = ranked[: max(1, max_evidence)]

        if not selected:
            return {
                "answer": "I do not have enough grounded memory evidence to answer this query.",
                "abstained": True,
                "confidence": 0.0,
                "citations": [],
                "evidence_count": 0,
                "creative_used": False,
                "diagnostics": self._diagnostics(evidence, completion, bundle),
            }

        citations = [self._citation(item) for item in selected]
        bullets = []
        for item in selected:
            snippet = self._snippet(item)
            if snippet:
                bullets.append(f"- {snippet} [{self._citation(item)}]")
        answer = (
            f"Grounded answer for: {query}\n"
            + "\n".join(bullets)
        ).strip()
        confidence = min(0.95, 0.35 + 0.12 * len(selected))
        return {
            "answer": answer,
            "abstained": False,
            "confidence": round(confidence, 3),
            "citations": citations,
            "evidence_count": len(selected),
            "creative_used": bool(creative),
            "diagnostics": self._diagnostics(evidence, completion, bundle),
        }

    def _dedupe_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for item in items:
            key = (
                str(item.get("object_id") or item.get("chunk_id") or item.get("id") or item.get("node_id") or "")
                or self._snippet(item)[:120]
            )
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _snippet(self, item: dict[str, Any]) -> str:
        for key in ("object_text", "text", "summary", "document", "raw_content", "retrieval_summary", "structured_summary"):
            value = str(item.get(key) or "").strip()
            if value:
                return " ".join(value.split())[:320]
        return ""

    def _citation(self, item: dict[str, Any]) -> str:
        return str(
            item.get("object_id")
            or item.get("chunk_id")
            or item.get("id")
            or item.get("node_id")
            or "memory"
        )

    def _diagnostics(self, evidence: Any, completion: Any, bundle: Any) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        if evidence is not None:
            diagnostics["retrieval"] = getattr(evidence, "diagnostics", {})
            diagnostics["timings_ms"] = getattr(evidence, "timings_ms", {})
        if completion is not None:
            diagnostics["completion"] = getattr(completion, "diagnostics", {})
        if bundle is not None:
            diagnostics["bundle_debug"] = getattr(bundle, "debug", {})
        return diagnostics
