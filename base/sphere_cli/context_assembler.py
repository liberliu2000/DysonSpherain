from __future__ import annotations

import json
from typing import Any

from .compression_elevator import CompressionElevator
from .config import AppConfig
from .models import ActivationBundle


class ContextAssembler:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.compressor = CompressionElevator()

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def assemble(
        self,
        task: str,
        task_type: str,
        temperature: float,
        main_nodes: list[dict[str, Any]],
        reflected_nodes: list[dict[str, Any]],
        refracted_nodes: list[dict[str, Any]],
        max_tokens: int = 1800,
    ) -> ActivationBundle:
        core_budget = int(max_tokens * 0.55)
        object_budget = int(max_tokens * 0.1)
        support_budget = int(max_tokens * 0.1)
        experience_budget = int(max_tokens * 0.25)
        creative_budget = int(max_tokens * 0.15)
        alternative_budget = int(max_tokens * 0.08)
        raw_pointer_budget = max_tokens - core_budget - object_budget - support_budget - experience_budget - creative_budget - alternative_budget

        core_evidence = self._pack_nodes(main_nodes, core_budget)
        evidence_objects: list[dict[str, Any]] = []
        supporting_context: list[dict[str, Any]] = []
        relevant_experience = self._pack_nodes(reflected_nodes, experience_budget)
        creative_reflections = self._pack_nodes(refracted_nodes, creative_budget)
        alternative_paths: list[dict[str, Any]] = []
        raw_reference_pointers = self._raw_pointers(main_nodes + reflected_nodes + refracted_nodes, raw_pointer_budget)

        debug = {
            "core_budget": core_budget,
            "object_budget": object_budget,
            "support_budget": support_budget,
            "experience_budget": experience_budget,
            "creative_budget": creative_budget,
            "alternative_budget": alternative_budget,
            "raw_pointer_budget": raw_pointer_budget,
            "estimated_input_tokens": sum(
                self.estimate_tokens(json.dumps(x, ensure_ascii=False))
                for x in core_evidence + evidence_objects + supporting_context + relevant_experience + creative_reflections + alternative_paths
            ),
        }

        return ActivationBundle(
            task=task,
            task_type=task_type,
            temperature=temperature,
            primary_evidence=core_evidence,
            core_evidence=core_evidence,
            evidence_objects=evidence_objects,
            supporting_context=supporting_context,
            relevant_experience=relevant_experience,
            creative_reflections=creative_reflections,
            alternative_paths=alternative_paths,
            raw_reference_pointers=raw_reference_pointers,
            debug=debug,
        )

    def assemble_evidence_first(
        self,
        task: str,
        task_type: str,
        temperature: float,
        core_evidence: list[dict[str, Any]],
        evidence_objects: list[dict[str, Any]],
        supporting_context: list[dict[str, Any]],
        relevant_experience: list[dict[str, Any]],
        creative_reflections: list[dict[str, Any]],
        max_tokens: int = 1800,
    ) -> ActivationBundle:
        core_budget = int(max_tokens * 0.4)
        object_budget = int(max_tokens * 0.12)
        support_budget = int(max_tokens * 0.16)
        experience_budget = int(max_tokens * 0.16)
        creative_budget = int(max_tokens * 0.08)
        alternative_budget = int(max_tokens * 0.06)
        raw_pointer_budget = max_tokens - core_budget - object_budget - support_budget - experience_budget - creative_budget - alternative_budget

        packed_core = self._pack_chunks(core_evidence, core_budget, mode="primary")
        packed_objects = self._pack_objects(evidence_objects, object_budget)
        packed_support = self._pack_chunks(supporting_context, support_budget, mode="support")
        packed_experience = self._pack_nodes(relevant_experience, experience_budget)
        packed_creative = self._pack_creative_reflections(creative_reflections, creative_budget)
        packed_alternative: list[dict[str, Any]] = []
        raw_reference_pointers = self._raw_pointers(
            core_evidence + supporting_context + evidence_objects + relevant_experience + creative_reflections,
            raw_pointer_budget,
        )

        raw_input_tokens = sum(
            self.estimate_tokens(json.dumps(x, ensure_ascii=False))
            for x in core_evidence + evidence_objects + supporting_context + relevant_experience + creative_reflections
        )
        assembled_tokens = sum(
            self.estimate_tokens(json.dumps(x, ensure_ascii=False))
            for x in packed_core + packed_objects + packed_support + packed_experience + packed_creative + packed_alternative
        )
        debug = {
            "core_budget": core_budget,
            "object_budget": object_budget,
            "support_budget": support_budget,
            "experience_budget": experience_budget,
            "creative_budget": creative_budget,
            "alternative_budget": alternative_budget,
            "raw_pointer_budget": raw_pointer_budget,
            "estimated_input_tokens": assembled_tokens,
            "raw_input_tokens": raw_input_tokens,
            "context_token_delta": raw_input_tokens - assembled_tokens,
        }

        return ActivationBundle(
            task=task,
            task_type=task_type,
            temperature=temperature,
            primary_evidence=packed_core,
            core_evidence=packed_core,
            evidence_objects=packed_objects,
            supporting_context=packed_support,
            relevant_experience=packed_experience,
            creative_reflections=packed_creative,
            alternative_paths=packed_alternative,
            raw_reference_pointers=raw_reference_pointers,
            debug=debug,
        )

    def assemble_evidence_first_with_paths(
        self,
        task: str,
        task_type: str,
        temperature: float,
        core_evidence: list[dict[str, Any]],
        evidence_objects: list[dict[str, Any]],
        supporting_context: list[dict[str, Any]],
        relevant_experience: list[dict[str, Any]],
        creative_reflections: list[dict[str, Any]],
        alternative_paths: list[dict[str, Any]],
        max_tokens: int = 1800,
    ) -> ActivationBundle:
        bundle = self.assemble_evidence_first(
            task=task,
            task_type=task_type,
            temperature=temperature,
            core_evidence=core_evidence,
            evidence_objects=evidence_objects,
            supporting_context=supporting_context,
            relevant_experience=relevant_experience,
            creative_reflections=creative_reflections,
            max_tokens=max_tokens,
        )
        alternative_budget = int(max_tokens * 0.06)
        packed_alternative = self._pack_paths(alternative_paths, alternative_budget)
        debug = dict(bundle.debug)
        debug["alternative_budget"] = alternative_budget
        alternative_tokens = sum(
            self.estimate_tokens(json.dumps(x, ensure_ascii=False))
            for x in packed_alternative
        )
        debug["estimated_input_tokens"] = debug.get("estimated_input_tokens", 0) + alternative_tokens
        debug["context_token_delta"] = debug.get("raw_input_tokens", 0) - debug.get("estimated_input_tokens", 0)
        return ActivationBundle(
            task=bundle.task,
            task_type=bundle.task_type,
            temperature=bundle.temperature,
            primary_evidence=bundle.primary_evidence,
            core_evidence=bundle.core_evidence,
            evidence_objects=bundle.evidence_objects,
            supporting_context=bundle.supporting_context,
            relevant_experience=bundle.relevant_experience,
            creative_reflections=bundle.creative_reflections,
            alternative_paths=packed_alternative,
            raw_reference_pointers=bundle.raw_reference_pointers,
            debug=debug,
        )

    def _pack_nodes(self, nodes: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        packed = []
        used = 0
        for node in nodes:
            compact = self.compressor.compress_node(node)
            item = {
                "id": compact["id"],
                "address": {
                    "shell": compact["shell"],
                    "sector": compact["sector"],
                    "zone": compact["zone"],
                    "cell": compact["cell"],
                },
                "type": compact["molecular_type"],
                "summary": compact["compressed_text"],
                "content_ref": compact.get("content_ref"),
            }
            cost = self.estimate_tokens(json.dumps(item, ensure_ascii=False))
            if used + cost > budget:
                break
            packed.append(item)
            used += cost
        return packed

    def _pack_chunks(self, chunks: list[dict[str, Any]], budget: int, *, mode: str) -> list[dict[str, Any]]:
        packed = []
        used = 0
        for idx, chunk in enumerate(chunks):
            primary_mode = mode == "primary"
            raw_text = str(chunk.get("text") or "")
            retrieval_summary = str(chunk.get("retrieval_summary") or chunk.get("summary") or "")
            structured_summary = str(chunk.get("structured_summary") or retrieval_summary)
            if not self.config.enable_context_compressor:
                chosen_text = raw_text
            elif primary_mode:
                if idx < 2 or len(raw_text) <= 260:
                    chosen_text = raw_text
                else:
                    chosen_text = retrieval_summary or self.compressor.compress_text(raw_text, target_ratio=0.72, min_len=120)
            else:
                chosen_text = structured_summary or retrieval_summary or self.compressor.compress_text(raw_text, target_ratio=0.38, min_len=56)
            item = {
                "id": chunk.get("chunk_id"),
                "node_id": chunk.get("node_id"),
                "grain": chunk.get("grain", "micro"),
                "address": {
                    "shell": chunk.get("shell"),
                    "sector": chunk.get("sector"),
                    "zone": chunk.get("zone"),
                    "cell": chunk.get("cell"),
                },
                "score": chunk.get("evidence_score"),
                "text": chosen_text,
                "time_bucket": chunk.get("time_bucket"),
                "summary": retrieval_summary if primary_mode else structured_summary or retrieval_summary,
                "signature": chunk.get("retrieval_signature"),
            }
            cost = self.estimate_tokens(json.dumps(item, ensure_ascii=False))
            if used + cost > budget:
                break
            packed.append(item)
            used += cost
        return packed

    def _pack_objects(self, objects: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        packed = []
        used = 0
        for obj in objects:
            item = {
                "id": obj.get("object_id"),
                "type": obj.get("object_type"),
                "text": self.compressor.compress_text(obj.get("object_text") or "", target_ratio=0.65, min_len=40),
                "entity": obj.get("entity"),
                "polarity": obj.get("polarity"),
                "confidence": obj.get("confidence"),
                "old_value": obj.get("old_value"),
                "new_value": obj.get("new_value"),
                "source_chunk_id": obj.get("source_chunk_id"),
                "source_node_id": obj.get("source_node_id"),
            }
            cost = self.estimate_tokens(json.dumps(item, ensure_ascii=False))
            if used + cost > budget:
                break
            packed.append(item)
            used += cost
        return packed

    def _pack_creative_reflections(self, nodes: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        packed = []
        used = 0
        for node in nodes:
            summary = str(node.get("summary") or "")
            notes = list(node.get("reflection_notes") or [])
            item = {
                "path_id": node.get("path_id"),
                "beam_type": node.get("beam_type"),
                "path_role": node.get("path_role"),
                "why_relevant": notes[0] if notes else summary[:90],
                "core_idea": self.compressor.compress_text(summary, target_ratio=0.48, min_len=52) if self.config.enable_context_compressor else summary,
                "risk": node.get("conflict_risk"),
                "score": node.get("reflection_score") or node.get("refraction_score"),
            }
            cost = self.estimate_tokens(json.dumps(item, ensure_ascii=False))
            if used + cost > budget:
                break
            packed.append(item)
            used += cost
        return packed

    def _raw_pointers(self, nodes: list[dict[str, Any]], budget: int) -> list[str]:
        pointers: list[str] = []
        used = 0
        for node in nodes:
            pointer = (
                node.get("content_ref")
                or node.get("source_chunk_id")
                or node.get("chunk_id")
                or node.get("object_id")
                or f"memory://{node.get('id') or node.get('path_id') or node.get('node_id') or 'unknown'}"
            )
            cost = self.estimate_tokens(pointer)
            if used + cost > budget:
                break
            pointers.append(pointer)
            used += cost
        return pointers

    def _pack_paths(self, paths: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        packed = []
        used = 0
        for idx, path in enumerate(paths):
            expand = idx < 2
            summary = str(path.get("summary") or "")
            signature = str(path.get("signature") or "")
            score_breakdown = dict(path.get("score_breakdown") or {})
            item = {
                "path_id": path.get("path_id"),
                "beam_type": path.get("beam_type"),
                "path_role": path.get("path_role"),
                "hop_count": path.get("hop_count"),
                "score": path.get("score"),
                "signature": signature,
                "summary": (
                    self.compressor.compress_text(summary, target_ratio=0.58, min_len=64)
                    if expand and self.config.enable_context_compressor
                    else signature or self.compressor.compress_text(summary, target_ratio=0.32, min_len=28)
                ),
                "why_relevant": (list(path.get("reflection_notes") or [])[:1] or [summary[:88]])[0],
                "risk": score_breakdown.get("conflict_risk"),
            }
            if expand:
                item["node_ids"] = path.get("node_ids")
                item["chunk_ids"] = path.get("chunk_ids")
            cost = self.estimate_tokens(json.dumps(item, ensure_ascii=False))
            if used + cost > budget:
                if not packed and budget > 0:
                    compact_item = {
                        "path_id": path.get("path_id"),
                        "beam_type": path.get("beam_type"),
                        "hop_count": path.get("hop_count"),
                        "score": path.get("score"),
                        "signature": signature,
                        "summary": signature or self.compressor.compress_text(summary, target_ratio=0.25, min_len=24),
                    }
                    compact_cost = self.estimate_tokens(json.dumps(compact_item, ensure_ascii=False))
                    if compact_cost <= budget:
                        packed.append(compact_item)
                break
            packed.append(item)
            used += cost
        return packed
