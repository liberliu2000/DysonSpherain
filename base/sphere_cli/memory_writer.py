from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from .config import AppConfig
from .models import MemoryEdge, MemoryNode, MemoryObject, now_iso
from .storage import Storage
from .utils import lexical_score, stable_content_hash, tokenize


class MemoryWriter:
    FILE_PATH_PATTERN = re.compile(
        r"(?P<path>(?:[A-Za-z]:[\\/]|\.{1,2}[\\/]|~[\\/])?(?:[A-Za-z0-9._-]+[\\/])*[A-Za-z0-9][A-Za-z0-9._ -]*\.(?:md|markdown|txt|json|ya?ml|toml|ini|cfg|conf|py|js|ts|tsx|jsx|css|html|csv|tsv|sql|log|pdf|ipynb|sh|ps1|bat|java|go|rs|cpp|c|h|hpp|rb))",
        re.IGNORECASE,
    )
    ARTIFACT_HINT_PATTERN = re.compile(
        r"\b(?:artifact|file|document|doc|report|result|output|plan|notes?|checklist|script|module|config|log|readme|path|stored in|lives in|saved in|written to)\b",
        re.IGNORECASE,
    )
    PROJECT_SIGNAL_PATTERN = re.compile(
        r"\b(?:workspace|project|roadmap|rollout|milestone|deliverable|release|sprint|backlog|validation|checkpoint|upgrade|todo|blocker)\b",
        re.IGNORECASE,
    )
    WEEKDAY_PATTERN = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    MONTH_PATTERN = r"january|february|march|april|may|june|july|august|september|october|november|december"
    NUMBER_WORD_PATTERN = r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    PREFERENCE_PATTERNS = [
        (
            re.compile(
                r"\b(?:prefer|prefers|preferred|like|likes|love|loves|enjoy|enjoys|am into|i'm into|big fan of)\s+(?P<object>[^.;,\n]+)",
                re.IGNORECASE,
            ),
            1.0,
        ),
        (
            re.compile(
                r"\b(?:favorite|favourite)\s+(?:thing|food|drink|meal|snack|color|colour|movie|book|genre|team|artist|song|way)?\s*(?:is|are)?\s*(?P<object>[^.;,\n]+)",
                re.IGNORECASE,
            ),
            1.0,
        ),
        (
            re.compile(
                r"\b(?:dislike|dislikes|hate|hates|avoid|avoids|can't stand|cannot stand|don't like|do not like|prefer not to|not a fan of|no longer like)\s+(?P<object>[^.;,\n]+)",
                re.IGNORECASE,
            ),
            -1.0,
        ),
    ]
    COMPARATIVE_PREFERENCE_PATTERNS = [
        (
            re.compile(
                r"\b(?:prefer|prefers|preferred)\s+(?P<preferred>[^.;,\n]+?)\s+(?:over|to)\s+(?P<rejected>[^.;,\n]+)",
                re.IGNORECASE,
            ),
            1.0,
            -0.65,
        ),
        (
            re.compile(
                r"\b(?:like|likes|love|loves|enjoy|enjoys)\s+(?P<preferred>[^.;,\n]+?)\s+more than\s+(?P<rejected>[^.;,\n]+)",
                re.IGNORECASE,
            ),
            0.9,
            -0.45,
        ),
        (
            re.compile(
                r"\b(?:would rather|would prefer to)\s+(?P<preferred>[^.;,\n]+?)\s+than\s+(?P<rejected>[^.;,\n]+)",
                re.IGNORECASE,
            ),
            0.92,
            -0.52,
        ),
    ]
    HABIT_PREFERENCE_PATTERNS = [
        (
            re.compile(
                r"\b(?:usually|generally|typically|often|always|tend to)\s+(?:choose|use|eat|order|drink|pick|go with|stick with)\s+(?P<object>[^.;,\n]+)",
                re.IGNORECASE,
            ),
            0.75,
        ),
        (
            re.compile(
                r"\b(?:never|rarely|hardly ever)\s+(?:choose|use|eat|order|drink|pick|go with|stick with)\s+(?P<object>[^.;,\n]+)",
                re.IGNORECASE,
            ),
            -0.8,
        ),
    ]
    STATE_PATTERNS = [
        re.compile(
            r"\b(?P<entity>[A-Za-z0-9_\- /]{2,40}?)\s+(?:was\s+)?(?:changed|updated|moved|switched|renamed|rescheduled)\s+from\s+(?P<old>[^.;,\n]+?)\s+to\s+(?P<new>[^.;,\n]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<entity>[A-Za-z0-9_\- /]{2,40}?)\s+is now\s+(?P<new>[^.;,\n]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<entity>[A-Za-z0-9_\- /]{2,40}?)\s+(?:was|were|used to be|previously|formerly)\s+(?P<old>[^.;,\n]+?)(?:\s+but|\s+and)\s+(?:is|are)\s+(?:now|currently)\s+(?P<new>[^.;,\n]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:now|currently|latest(?:ly)?)\s+(?P<entity>[A-Za-z0-9_\- /]{2,40}?)\s+(?:is|are|uses?|works? at|lives? in|stays? at|resides in)\s+(?P<new>[^.;,\n]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<entity>[A-Za-z0-9_\- /]{2,40}?)\s+(?:moved|switched|updated|rescheduled)\s+to\s+(?P<new>[^.;,\n]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<entity>[A-Za-z0-9_\- /]{2,40}?)\s+(?:used to|previously|formerly|before)\s+(?P<old>[^.;,\n]+)",
            re.IGNORECASE,
        ),
    ]
    SUBJECT_PATTERN = re.compile(r"\b(?:i|i'm|i’ve|i've|my|me|mine|we|our|ours)\b", re.IGNORECASE)
    TEMPORAL_REFERENCE_PATTERNS = [
        (re.compile(r"\b(?P<time>yesterday|today|tomorrow|tonight|this morning|this afternoon|this evening)\b", re.IGNORECASE), 0.7),
        (re.compile(rf"\b(?P<time>last\s+(?:{WEEKDAY_PATTERN}|week|month|year|night|weekend))\b", re.IGNORECASE), 0.76),
        (re.compile(rf"\b(?P<time>next\s+(?:{WEEKDAY_PATTERN}|week|month|year|weekend))\b", re.IGNORECASE), 0.68),
        (re.compile(rf"\b(?P<time>(?:\d+|{NUMBER_WORD_PATTERN})\s+(?:day|days|week|weeks|month|months|year|years)\s+ago)\b", re.IGNORECASE), 0.84),
        (re.compile(rf"\b(?P<time>in\s+(?:{MONTH_PATTERN})(?:\s+\d{{4}})?)\b", re.IGNORECASE), 0.66),
        (re.compile(rf"\b(?P<time>during\s+(?:{MONTH_PATTERN}|the weekend|lunch|dinner|breakfast)(?:\s+\d{{4}})?)\b", re.IGNORECASE), 0.62),
        (re.compile(rf"\b(?P<time>on\s+(?:{WEEKDAY_PATTERN}|{MONTH_PATTERN}\s+\d{{1,2}}(?:,\s*\d{{4}})?|\d{{4}}-\d{{2}}-\d{{2}}))\b", re.IGNORECASE), 0.74),
        (re.compile(r"\b(?P<time>\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2})\b", re.IGNORECASE), 0.8),
        (re.compile(rf"\b(?P<time>(?:{MONTH_PATTERN})\s+\d{{1,2}}(?:,\s*\d{{4}})?|(?:{MONTH_PATTERN})\s+\d{{4}})\b", re.IGNORECASE), 0.72),
    ]
    PERSONAL_CONTEXT_PATTERNS = [
        ("problem", re.compile(r"\b(?:having trouble with|trouble with|issues? with|problem with|struggling with|having difficulty with)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE), 0.8),
        ("feeling", re.compile(r"\b(?:i've been|i have been|i'm|i am)\s+feeling\s+(?P<object>[^.;,\n]+?)(?:\s+lately|\s+recently|$)", re.IGNORECASE), 0.76),
        ("goal", re.compile(r"\b(?:i need to|i want to|i'm trying to|i am trying to|i plan to|i'm planning to|i am planning to|i'm considering|i am considering)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE), 0.72),
        ("resource", re.compile(r"\b(?:with|using)\s+my\s+(?P<object>(?:homegrown\s+)?[^.;,\n]+?(?:ingredients|garden|budget|schedule|phone|laptop|tools?))\b", re.IGNORECASE), 0.68),
    ]
    PERSONA_PATTERNS = [
        ("role", re.compile(r"\b(?:i am|i'm)\s+(?:a|an)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE), 0.74),
        ("identity", re.compile(r"\b(?:my role is|i work as|i serve as)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE), 0.72),
        ("trait", re.compile(r"\b(?:i am|i'm)\s+(?P<object>(?:careful|curious|frugal|introverted|extroverted|organized|messy|optimistic|pessimistic|patient|creative)[^.;,\n]*)", re.IGNORECASE), 0.66),
    ]
    RELATION_PATTERNS = [
        re.compile(
            r"\b(?P<object>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+is\s+(?:my|our)\s+(?P<relation>father|mother|daughter|son|wife|husband|friend|colleague|boss|manager|teacher|mentor)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:my|our)\s+(?P<relation>father|mother|daughter|son|wife|husband|friend|colleague|boss|manager|teacher|mentor)\s+(?P<object>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b",
            re.IGNORECASE,
        ),
    ]
    CONSTRAINT_PATTERNS = [
        ("budget", re.compile(r"\b(?:budget of|budget is|tight budget of|for under|under)\s+(?P<object>[$€£]?\d[\w\s.,-]*)", re.IGNORECASE), 0.8),
        ("deadline", re.compile(r"\b(?:by|before)\s+(?P<object>(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|next week|next month|[A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?))\b", re.IGNORECASE), 0.72),
        ("restriction", re.compile(r"\b(?:must|need to|cannot|can't|should avoid|do not want to)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE), 0.68),
    ]
    FACT_PATTERNS = [
        re.compile(r"\b(?P<subject>[A-Z][A-Za-z0-9_\- /]{1,40}?)\s+(?:is|was|are|were|has|have)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE),
    ]
    GOAL_PATTERNS = [
        ("goal", re.compile(r"\b(?:goal is to|goal:|aim is to|objective is to)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE), 0.76),
        ("goal", re.compile(r"\b(?:i need to|we need to|i want to|we want to|i plan to|we plan to|trying to)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE), 0.7),
        ("milestone", re.compile(r"\b(?:milestone|deliverable|priority|target)\s*(?:for\s+(?:this|the)\s+(?:sprint|phase|release))?\s*(?:is|:)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE), 0.72),
        ("goal", re.compile(r"\b(?:focus is to|focused on|aiming to|we are aiming to|plan is to)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE), 0.7),
    ]
    DECISION_PATTERNS = [
        re.compile(r"\b(?:decided to|chose to|opted to|we decided to|we chose to)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE),
        re.compile(r"\bdecision\s*:\s*(?P<object>[^.;,\n]+)", re.IGNORECASE),
    ]
    PROJECT_PATTERNS = [
        re.compile(r"\bproject\s+(?P<object>[A-Za-z0-9_\- /]{3,60})", re.IGNORECASE),
        re.compile(r"\bworking on\s+(?P<object>[^.;,\n]+)", re.IGNORECASE),
        re.compile(r"\b(?P<object>[A-Za-z0-9_\- /]{2,40})\s+(?:project|workspace|initiative|repo(?:sitory)?)\b", re.IGNORECASE),
        re.compile(r"\b(?:for|within|inside)\s+(?P<object>[A-Za-z0-9_\- /]{2,40})\s+(?:project|workspace|repo(?:sitory)?)\b", re.IGNORECASE),
    ]
    PATTERN_PATTERNS = [
        re.compile(r"\b(?:pattern|recurring pattern|trend)\s*:\s*(?P<object>[^.;,\n]+)", re.IGNORECASE),
        re.compile(r"\b(?:keeps happening|tends to happen|usually happens)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE),
    ]
    OPEN_LOOP_PATTERNS = [
        re.compile(r"\b(?:still need to|need to follow up on|follow up on|todo|to do|pending)\s*:?\s+(?P<object>[^.;,\n]+)", re.IGNORECASE),
        re.compile(r"\b(?:left unresolved|not finished|not yet done|outstanding)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE),
        re.compile(r"^\s*(?:[-*]\s*)?\[\s?\]\s*(?P<object>.+)$", re.IGNORECASE),
        re.compile(r"\b(?:action item|next step|remaining task|left to do|still open)\s*:?\s*(?P<object>[^.;,\n]+)", re.IGNORECASE),
        re.compile(r"\b(?:blocked on|waiting on|waiting for)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE),
        re.compile(r"\b(?:defer|deferred|postpone|postponed)\s+(?P<object>[^.;,\n]+)", re.IGNORECASE),
    ]
    SOLUTION_HINTS = ("fix", "fixed", "solution", "resolved", "resolution", "workaround", "cause", "because", "due to")
    PROBLEM_HINTS = ("error", "issue", "problem", "bug", "failure", "contention", "exception", "deadlock", "timeout")

    def __init__(self, storage: Storage, config: AppConfig) -> None:
        self.storage = storage
        self.config = config

    def prepare_chunks(self, node: MemoryNode, source_kind: str | None = None, source_path: str | None = None) -> list[dict[str, Any]]:
        source_text = (node.raw_content or "").strip() or node.summary.strip()
        if not source_text:
            return []
        source_kind = source_kind or (node.molecular_type or "raw_content")
        if source_kind == "markdown":
            pieces = self._chunk_markdown(source_text)
        elif source_kind == "code":
            pieces = self._chunk_by_lines(source_text, self.config.code_chunk_lines)
        elif source_kind == "log":
            pieces = self._chunk_by_lines(source_text, self.config.log_chunk_lines)
        elif source_kind == "pdf":
            pieces = self._chunk_by_chars(source_text, self.config.pdf_chunk_size, self.config.chunk_overlap)
        else:
            pieces = self._chunk_by_chars(source_text, self.config.chunk_size, self.config.chunk_overlap)

        micro_chunks: list[dict[str, Any]] = []
        next_index = 0
        for text in pieces:
            text = text.strip()
            if not text:
                continue
            micro_chunks.append(
                self._make_chunk(
                    node=node,
                    chunk_index=next_index,
                    grain="micro",
                    text=text,
                    source_kind=source_kind,
                    source_path=source_path,
                )
            )
            next_index += 1

        chunks: list[dict[str, Any]] = []
        chunks.extend(micro_chunks)

        local_windows = self._build_local_windows(micro_chunks, node, source_kind, source_path, start_index=next_index)
        if not self.config.embed_local_grain:
            for window in local_windows:
                window["skip_vector"] = True
        chunks.extend(local_windows)
        next_index += len(local_windows)

        macro_chunk = self._build_macro_chunk(node, source_kind, source_path, chunk_index=next_index)
        if macro_chunk is not None:
            chunks.append(macro_chunk)
        return chunks

    def _make_chunk(
        self,
        node: MemoryNode,
        chunk_index: int,
        grain: str,
        text: str,
        source_kind: str,
        source_path: str | None = None,
        member_chunk_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "chunk_id": f"{node.id}_chunk_{chunk_index:04d}",
            "node_id": node.id,
            "chunk_index": chunk_index,
            "grain": grain,
            "text": text,
            "content_hash": stable_content_hash(text),
            "scope": node.scope,
            "workspace": node.workspace,
            "project": node.project,
            "session_id": node.session_id,
            "token_estimate": max(1, len(text) // 4),
            "source_kind": source_kind,
            "source_path": source_path or node.content_ref or "",
            "source_type": node.source_type,
            "source_ref": node.source_ref or source_path or node.content_ref or "",
            "created_at": node.created_at,
            "updated_at": node.updated_at,
            "timestamp": node.created_at,
            "vector_synced_at": now_iso(),
            "shell": node.shell,
            "sector": node.sector,
            "zone": node.zone,
            "cell": node.cell,
            "summary": node.summary,
            "content_ref": node.content_ref,
            "access_count": int(node.access_count),
            "member_chunk_ids": member_chunk_ids or [],
        }

    def _build_local_windows(
        self,
        micro_chunks: list[dict[str, Any]],
        node: MemoryNode,
        source_kind: str,
        source_path: str | None,
        start_index: int,
    ) -> list[dict[str, Any]]:
        if len(micro_chunks) < 2:
            return []
        span = max(2, self.config.local_window_span)
        windows: list[dict[str, Any]] = []
        next_index = start_index
        for start in range(0, len(micro_chunks) - 1):
            members = micro_chunks[start : start + span]
            if len(members) < 2:
                continue
            text = "\n".join(member["text"] for member in members)
            windows.append(
                self._make_chunk(
                    node=node,
                    chunk_index=next_index,
                    grain="local",
                    text=text,
                    source_kind=source_kind,
                    source_path=source_path,
                    member_chunk_ids=[member["chunk_id"] for member in members],
                )
            )
            next_index += 1
        return windows

    def _build_macro_chunk(
        self,
        node: MemoryNode,
        source_kind: str,
        source_path: str | None,
        chunk_index: int,
    ) -> dict[str, Any] | None:
        text = (node.summary or "").strip()
        if not text:
            return None
        return self._make_chunk(
            node=node,
            chunk_index=chunk_index,
            grain="macro",
            text=text,
            source_kind=f"{source_kind}_summary",
            source_path=source_path,
        )

    def _chunk_markdown(self, text: str) -> list[str]:
        sections: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in text.splitlines():
            is_header = line.lstrip().startswith("#")
            if is_header and current:
                sections.append("\n".join(current))
                current = [line]
                current_len = len(line)
                continue
            current.append(line)
            current_len += len(line) + 1
            if current_len >= self.config.markdown_chunk_size:
                sections.append("\n".join(current))
                current = []
                current_len = 0
        if current:
            sections.append("\n".join(current))
        return sections

    def _chunk_by_lines(self, text: str, lines_per_chunk: int) -> list[str]:
        lines = text.splitlines()
        return ["\n".join(lines[i : i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]

    def _chunk_by_chars(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def build_chunk_neighbors(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        neighbors: list[dict[str, Any]] = []
        micro_chunks = [chunk for chunk in chunks if chunk.get("grain", "micro") == "micro"]
        local_chunks = [chunk for chunk in chunks if chunk.get("grain") == "local"]
        macro_chunks = [chunk for chunk in chunks if chunk.get("grain") == "macro"]

        for index, chunk in enumerate(micro_chunks):
            if index + 1 < len(micro_chunks):
                next_chunk = micro_chunks[index + 1]
                neighbors.append(
                    {
                        "id": f"nbr_{chunk['chunk_id']}_next",
                        "chunk_id": chunk["chunk_id"],
                        "neighbor_chunk_id": next_chunk["chunk_id"],
                        "relation_type": "next_chunk",
                        "weight": 1.0,
                    }
                )
                neighbors.append(
                    {
                        "id": f"nbr_{next_chunk['chunk_id']}_prev",
                        "chunk_id": next_chunk["chunk_id"],
                        "neighbor_chunk_id": chunk["chunk_id"],
                        "relation_type": "prev_chunk",
                        "weight": 1.0,
                    }
                )
            if index + 2 < len(micro_chunks):
                window_chunk = micro_chunks[index + 2]
                neighbors.append(
                    {
                        "id": f"nbr_{chunk['chunk_id']}_window_{index + 2}",
                        "chunk_id": chunk["chunk_id"],
                        "neighbor_chunk_id": window_chunk["chunk_id"],
                        "relation_type": "same_window",
                        "weight": 0.55,
                    }
                )
        for chunk in local_chunks:
            for member_id in chunk.get("member_chunk_ids", []):
                neighbors.append(
                    {
                        "id": f"nbr_{member_id}_local_{chunk['chunk_id']}",
                        "chunk_id": member_id,
                        "neighbor_chunk_id": chunk["chunk_id"],
                        "relation_type": "local_window",
                        "weight": 0.8,
                    }
                )
        for chunk in macro_chunks:
            for member in micro_chunks[:4]:
                neighbors.append(
                    {
                        "id": f"nbr_{member['chunk_id']}_macro_{chunk['chunk_id']}",
                        "chunk_id": member["chunk_id"],
                        "neighbor_chunk_id": chunk["chunk_id"],
                        "relation_type": "macro_context",
                        "weight": 0.45,
                    }
                )
        # Persist lightweight neighbor counts on the chunk payload itself so retrieval
        # can often avoid an extra SQLite aggregation for high-volume benchmarks.
        neighbor_counts: dict[str, int] = {}
        for link in neighbors:
            chunk_id = str(link.get("chunk_id") or "")
            if not chunk_id:
                continue
            neighbor_counts[chunk_id] = neighbor_counts.get(chunk_id, 0) + 1
        for chunk in chunks:
            chunk["neighbor_count"] = neighbor_counts.get(str(chunk.get("chunk_id") or ""), 0)
        return neighbors

    def extract_objects(self, node: MemoryNode, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        objects: list[MemoryObject] = []
        for chunk in self._iter_object_candidate_chunks(chunks):
            units = self._split_candidate_units(chunk["text"])
            for offset, unit in enumerate(units):
                objects.extend(self._extract_preference_objects(node, chunk, unit, offset))
                objects.extend(self._extract_temporal_objects(node, chunk, unit, offset))
                objects.extend(self._extract_personal_context_objects(node, chunk, unit, offset))
                objects.extend(self._extract_persona_objects(node, chunk, unit, offset))
                objects.extend(self._extract_relation_objects(node, chunk, unit, offset))
                objects.extend(self._extract_constraint_objects(node, chunk, unit, offset))
                objects.extend(self._extract_fact_objects(node, chunk, unit, offset))
                objects.extend(self._extract_goal_objects(node, chunk, unit, offset))
                objects.extend(self._extract_decision_objects(node, chunk, unit, offset))
                objects.extend(self._extract_project_objects(node, chunk, unit, offset))
                objects.extend(self._extract_artifact_objects(node, chunk, unit, offset))
                objects.extend(self._extract_pattern_objects(node, chunk, unit, offset))
                objects.extend(self._extract_open_loop_objects(node, chunk, unit, offset))
                solution = self._extract_solution_object(node, chunk, unit, offset)
                if solution is not None:
                    objects.append(solution)
        deduped: dict[tuple[str, str, str, str], MemoryObject] = {}
        for item in objects:
            key = (
                item.object_type,
                item.source_chunk_id,
                str(item.canonical_key or "").strip().lower(),
                str(item.content_hash or stable_content_hash(item.object_text)),
            )
            deduped.setdefault(key, item)
        augmented = self._augment_derived_objects(list(deduped.values()))
        for item in augmented:
            self._enrich_object_defaults(node, item)
        final_deduped: dict[tuple[str, str, str, str], MemoryObject] = {}
        for item in augmented:
            key = (
                item.object_type,
                item.source_chunk_id,
                str(item.canonical_key or "").strip().lower(),
                str(item.content_hash or stable_content_hash(item.object_text)),
            )
            final_deduped.setdefault(key, item)
        return [item.to_dict() for item in final_deduped.values()]

    def _iter_object_candidate_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for chunk in chunks:
            grain = str(chunk.get("grain") or "micro")
            if grain == "micro":
                candidates.append(chunk)
                continue
            if grain != "local":
                continue
            member_chunk_ids = list(chunk.get("member_chunk_ids") or [])
            if not member_chunk_ids:
                continue
            augmented = dict(chunk)
            augmented["source_chunk_id"] = member_chunk_ids[0]
            candidates.append(augmented)
        return candidates

    def _split_candidate_units(self, text: str) -> list[str]:
        raw_units = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
        return [unit.strip() for unit in raw_units if unit and unit.strip()]

    def _extract_preference_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        lowered = unit.lower()
        subject = "user" if self.SUBJECT_PATTERN.search(unit) else "unknown"
        seen_targets: set[tuple[str, float]] = set()

        for pattern, preferred_polarity, rejected_polarity in self.COMPARATIVE_PREFERENCE_PATTERNS:
            match = pattern.search(unit)
            if not match:
                continue
            preferred_target = self._clean_preference_target(match.group("preferred"))
            rejected_target = self._clean_preference_target(match.group("rejected"))
            if preferred_target:
                key = (self._normalize_preference_target(preferred_target), preferred_polarity)
                if key not in seen_targets:
                    results.append(
                        self._make_preference_object(
                            node=node,
                            chunk=chunk,
                            unit=unit,
                            subject=subject,
                            target=preferred_target,
                            polarity=preferred_polarity,
                            confidence=0.84,
                            offset=offset,
                        )
                    )
                    seen_targets.add(key)
            if rejected_target:
                key = (self._normalize_preference_target(rejected_target), rejected_polarity)
                if key not in seen_targets:
                    results.append(
                        self._make_preference_object(
                            node=node,
                            chunk=chunk,
                            unit=unit,
                            subject=subject,
                            target=rejected_target,
                            polarity=rejected_polarity,
                            confidence=0.72,
                            offset=offset,
                        )
                    )
                    seen_targets.add(key)

        for pattern, polarity in self.PREFERENCE_PATTERNS:
            match = pattern.search(unit)
            if not match:
                continue
            if polarity > 0 and self._is_negated_preference_match(unit, match.start()):
                continue
            object_text = self._clean_preference_target(match.group("object"))
            if not object_text:
                continue
            key = (self._normalize_preference_target(object_text), polarity)
            if key in seen_targets:
                continue
            results.append(
                self._make_preference_object(
                    node=node,
                    chunk=chunk,
                    unit=unit,
                    subject=subject,
                    target=object_text,
                    polarity=polarity,
                    confidence=0.78 if polarity > 0 else 0.8,
                    offset=offset,
                )
            )
            seen_targets.add(key)

        for pattern, polarity in self.HABIT_PREFERENCE_PATTERNS:
            match = pattern.search(unit)
            if not match:
                continue
            if polarity > 0 and self._is_negated_preference_match(unit, match.start()):
                continue
            object_text = self._clean_preference_target(match.group("object"))
            if not object_text:
                continue
            key = (self._normalize_preference_target(object_text), polarity)
            if key in seen_targets:
                continue
            results.append(
                self._make_preference_object(
                    node=node,
                    chunk=chunk,
                    unit=unit,
                    subject=subject,
                    target=object_text,
                    polarity=polarity,
                    confidence=0.68,
                    offset=offset,
                )
            )
            seen_targets.add(key)
        return results

    def _make_preference_object(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        subject: str,
        target: str,
        polarity: float,
        confidence: float,
        offset: int,
    ) -> MemoryObject:
        normalized_target = self._normalize_preference_target(target)
        verb = "prefers" if polarity > 0 else "avoids"
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        object_text = f"{subject} {verb} {target}"
        object_id = self._stable_object_id(
            object_type="preference",
            source_chunk_id=source_chunk_id,
            canonical_key=normalized_target,
            sequence_index=sequence_index,
            object_text=object_text,
        )
        return MemoryObject(
            object_type="preference",
            subject=subject,
            predicate=verb,
            object_text=object_text,
            polarity=polarity,
            entity=normalized_target,
            attribute="preference_target",
            canonical_key=normalized_target,
            source_unit_text=unit.strip(),
            sequence_index=sequence_index,
            content_hash=stable_content_hash(object_text),
            confidence=confidence,
            source_chunk_id=source_chunk_id,
            source_node_id=node.id,
            session_id=node.content_ref,
            turn_index=offset,
            timestamp=node.created_at,
            snapshot_key=normalized_target,
            merge_policy="latest_wins",
            object_id=object_id,
        )

    def _clean_preference_target(self, raw_target: str | None) -> str:
        if not raw_target:
            return ""
        cleaned = raw_target.strip(" -:;,.")
        cleaned = re.split(r"\b(?:because|since|but|although|though|if|when)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = re.sub(r"\b(?:the|a|an)\b\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if len(cleaned) >= 3 else ""

    def _normalize_preference_target(self, target: str) -> str:
        normalized = re.sub(r"\s+", " ", target).strip().lower()
        normalized = re.sub(r"^(?:the|a|an)\s+", "", normalized)
        return normalized

    def _is_negated_preference_match(self, unit: str, match_start: int) -> bool:
        prefix = unit[max(0, match_start - 20) : match_start].lower()
        return any(token in prefix for token in ("don't ", "do not ", "not ", "never ", "no longer ", "can't ", "cannot "))

    def _extract_temporal_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        for pattern in self.STATE_PATTERNS:
            match = pattern.search(unit)
            if not match:
                continue
            entity = self._clean_state_text(match.groupdict().get("entity") or "state")
            old_value = self._clean_state_text(match.groupdict().get("old")) or None
            new_value = self._clean_state_text(match.groupdict().get("new")) or None
            if not old_value and not new_value:
                continue
            temporal_marker = self._infer_temporal_marker(unit, old_value, new_value)
            attribute = self._infer_state_attribute(entity, old_value, new_value)
            canonical_key = self._state_canonical_key(entity, attribute)
            object_text = self._format_state_object_text(entity, old_value, new_value)
            results.append(
                MemoryObject(
                    object_type="state_update",
                    object_text=object_text,
                    entity=entity,
                    attribute=attribute,
                    old_value=old_value,
                    new_value=new_value,
                    event_text=unit.strip(),
                    canonical_key=canonical_key,
                    temporal_marker=temporal_marker,
                    sequence_index=sequence_index,
                    source_unit_text=unit.strip(),
                    content_hash=stable_content_hash(object_text),
                    confidence=0.74 if old_value and new_value else 0.64,
                    source_chunk_id=source_chunk_id,
                    source_node_id=node.id,
                    session_id=node.content_ref,
                    turn_index=offset,
                    timestamp=node.created_at,
                    snapshot_key=canonical_key,
                    merge_policy="base_plus_delta",
                    object_id=self._stable_object_id(
                        object_type="state_update",
                        source_chunk_id=source_chunk_id,
                        canonical_key=canonical_key,
                        sequence_index=sequence_index,
                        object_text=object_text,
                    ),
                )
            )
        results.extend(self._extract_temporal_reference_objects(node, chunk, unit, offset))
        return results

    def _extract_temporal_reference_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        seen: set[str] = set()
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        for pattern, confidence in self.TEMPORAL_REFERENCE_PATTERNS:
            for match in pattern.finditer(unit):
                time_phrase = self._clean_temporal_phrase(match.groupdict().get("time"))
                normalized_time = self._normalize_temporal_phrase(time_phrase)
                if not normalized_time or normalized_time in seen:
                    continue
                event_label = self._infer_temporal_event_label(unit, time_phrase)
                object_text = f"{event_label} @ {time_phrase}" if event_label else unit.strip()
                canonical_key = self._normalize_temporal_event_label(event_label) or normalized_time
                results.append(
                    MemoryObject(
                        object_type="temporal_reference",
                        object_text=object_text,
                        subject="user",
                        predicate="occurred_at",
                        entity=canonical_key,
                        attribute="event_time",
                        new_value=normalized_time,
                        event_text=unit.strip(),
                        canonical_key=canonical_key,
                        temporal_marker="point",
                        sequence_index=sequence_index,
                        source_unit_text=unit.strip(),
                        content_hash=stable_content_hash(object_text),
                        confidence=confidence,
                        source_chunk_id=source_chunk_id,
                        source_node_id=node.id,
                        session_id=node.content_ref,
                        turn_index=offset,
                        timestamp=node.created_at,
                        snapshot_key=canonical_key,
                        merge_policy="append",
                        object_id=self._stable_object_id(
                            object_type="temporal_reference",
                            source_chunk_id=source_chunk_id,
                            canonical_key=canonical_key,
                            sequence_index=sequence_index,
                            object_text=object_text,
                        ),
                    )
                )
                seen.add(normalized_time)
        return results

    def _extract_personal_context_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        subject = "user" if self.SUBJECT_PATTERN.search(unit) else "unknown"
        seen: set[tuple[str, str]] = set()
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        for attribute, pattern, confidence in self.PERSONAL_CONTEXT_PATTERNS:
            for match in pattern.finditer(unit):
                target = self._clean_context_target(match.groupdict().get("object"))
                normalized = self._normalize_context_target(target)
                if not normalized or (attribute, normalized) in seen:
                    continue
                verb = {
                    "problem": "has issue with",
                    "feeling": "feels",
                    "goal": "is considering",
                    "resource": "has",
                }.get(attribute, "has context")
                object_text = f"{subject} {verb} {target}"
                results.append(
                    MemoryObject(
                        object_type="personal_context",
                        object_text=object_text,
                        subject=subject,
                        predicate=verb,
                        entity=normalized,
                        attribute=attribute,
                        event_text=unit.strip(),
                        canonical_key=f"{attribute}:{normalized}",
                        sequence_index=sequence_index,
                        source_unit_text=unit.strip(),
                        content_hash=stable_content_hash(object_text),
                        confidence=confidence,
                        source_chunk_id=source_chunk_id,
                        source_node_id=node.id,
                        session_id=node.content_ref,
                        turn_index=offset,
                        timestamp=node.created_at,
                        snapshot_key=f"{attribute}:{normalized}",
                        merge_policy="latest_wins",
                        object_id=self._stable_object_id(
                            object_type="personal_context",
                            source_chunk_id=source_chunk_id,
                            canonical_key=f"{attribute}:{normalized}",
                            sequence_index=sequence_index,
                            object_text=object_text,
                        ),
                    )
                )
                seen.add((attribute, normalized))
        return results

    def _extract_persona_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        seen: set[tuple[str, str]] = set()
        for attribute, pattern, confidence in self.PERSONA_PATTERNS:
            for match in pattern.finditer(unit):
                target = self._clean_context_target(match.groupdict().get("object"))
                normalized = self._normalize_context_target(target)
                if not normalized or (attribute, normalized) in seen:
                    continue
                object_text = f"user {attribute} {target}"
                results.append(
                    MemoryObject(
                        object_type="persona",
                        object_text=object_text,
                        subject="user",
                        predicate=attribute,
                        entity=normalized,
                        attribute=attribute,
                        canonical_key=f"{attribute}:{normalized}",
                        source_unit_text=unit.strip(),
                        sequence_index=sequence_index,
                        content_hash=stable_content_hash(object_text),
                        confidence=confidence,
                        source_chunk_id=source_chunk_id,
                        source_node_id=node.id,
                        session_id=node.content_ref,
                        turn_index=offset,
                        timestamp=node.created_at,
                        snapshot_key="persona",
                        merge_policy="latest_wins",
                        object_id=self._stable_object_id(
                            object_type="persona",
                            source_chunk_id=source_chunk_id,
                            canonical_key=f"{attribute}:{normalized}",
                            sequence_index=sequence_index,
                            object_text=object_text,
                        ),
                    )
                )
                seen.add((attribute, normalized))
        return results

    def _extract_constraint_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        seen: set[tuple[str, str]] = set()
        for attribute, pattern, confidence in self.CONSTRAINT_PATTERNS:
            for match in pattern.finditer(unit):
                target = self._clean_context_target(match.groupdict().get("object"))
                normalized = self._normalize_context_target(target)
                if not normalized or (attribute, normalized) in seen:
                    continue
                object_text = f"constraint {attribute} {target}"
                results.append(
                    MemoryObject(
                        object_type="constraint",
                        object_text=object_text,
                        subject="user",
                        predicate="constrained_by",
                        entity=normalized,
                        attribute=attribute,
                        canonical_key=f"{attribute}:{normalized}",
                        source_unit_text=unit.strip(),
                        sequence_index=sequence_index,
                        content_hash=stable_content_hash(object_text),
                        confidence=confidence,
                        source_chunk_id=source_chunk_id,
                        source_node_id=node.id,
                        session_id=node.content_ref,
                        turn_index=offset,
                        timestamp=node.created_at,
                        snapshot_key="constraint",
                        merge_policy="append",
                        object_id=self._stable_object_id(
                            object_type="constraint",
                            source_chunk_id=source_chunk_id,
                            canonical_key=f"{attribute}:{normalized}",
                            sequence_index=sequence_index,
                            object_text=object_text,
                        ),
                    )
                )
                seen.add((attribute, normalized))
        return results

    def _extract_relation_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        seen: set[tuple[str, str]] = set()
        for pattern in self.RELATION_PATTERNS:
            for match in pattern.finditer(unit):
                relation = str(match.groupdict().get("relation") or "").strip().lower()
                target = self._clean_context_target(match.groupdict().get("object"))
                normalized = self._normalize_context_target(target)
                if not relation or not normalized or (relation, normalized) in seen:
                    continue
                object_text = f"user relation {relation} {target}"
                results.append(
                    MemoryObject(
                        object_type="relation",
                        object_text=object_text,
                        subject="user",
                        predicate="related_to",
                        entity=normalized,
                        attribute=relation,
                        canonical_key=f"relation:{relation}:{normalized}",
                        source_unit_text=unit.strip(),
                        sequence_index=sequence_index,
                        content_hash=stable_content_hash(object_text),
                        confidence=0.72,
                        source_chunk_id=source_chunk_id,
                        source_node_id=node.id,
                        session_id=node.content_ref,
                        turn_index=offset,
                        timestamp=node.created_at,
                        snapshot_key=f"relation:{relation}:{normalized}",
                        merge_policy="latest_wins",
                        object_id=self._stable_object_id(
                            object_type="relation",
                            source_chunk_id=source_chunk_id,
                            canonical_key=f"relation:{relation}:{normalized}",
                            sequence_index=sequence_index,
                            object_text=object_text,
                        ),
                    )
                )
                seen.add((relation, normalized))
        return results

    def _extract_fact_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        for pattern in self.FACT_PATTERNS:
            match = pattern.search(unit)
            if not match:
                continue
            subject = self._clean_context_target(match.groupdict().get("subject"))
            obj = self._clean_context_target(match.groupdict().get("object"))
            if not subject or not obj:
                continue
            normalized = self._normalize_context_target(subject)
            object_text = f"{subject} fact {obj}"
            results.append(
                MemoryObject(
                    object_type="fact",
                    object_text=object_text,
                    subject=subject,
                    predicate="fact",
                    entity=normalized,
                    attribute="fact",
                    canonical_key=f"fact:{normalized}",
                    source_unit_text=unit.strip(),
                    sequence_index=sequence_index,
                    content_hash=stable_content_hash(object_text),
                    confidence=0.58,
                    source_chunk_id=source_chunk_id,
                    source_node_id=node.id,
                    session_id=node.content_ref,
                    turn_index=offset,
                    timestamp=node.created_at,
                    merge_policy="append",
                    object_id=self._stable_object_id(
                        object_type="fact",
                        source_chunk_id=source_chunk_id,
                        canonical_key=f"fact:{normalized}",
                        sequence_index=sequence_index,
                        object_text=object_text,
                    ),
                )
            )
        return results

    def _extract_goal_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        seen: set[str] = set()
        for attribute, pattern, confidence in self.GOAL_PATTERNS:
            for match in pattern.finditer(unit):
                target = self._clean_context_target(match.groupdict().get("object"))
                normalized = self._normalize_context_target(target)
                if not normalized or normalized in seen:
                    continue
                results.append(self._make_generic_object(node, chunk, unit, offset, "goal", target, attribute, confidence))
                seen.add(normalized)
        return results

    def _extract_decision_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        seen: set[str] = set()
        for pattern in self.DECISION_PATTERNS:
            for match in pattern.finditer(unit):
                target = self._clean_context_target(match.groupdict().get("object"))
                normalized = self._normalize_context_target(target)
                if not normalized or normalized in seen:
                    continue
                results.append(self._make_generic_object(node, chunk, unit, offset, "decision", target, "decision", 0.74))
                seen.add(normalized)
        return results

    def _extract_project_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        seen: set[str] = set()
        task_like_unit = bool(re.search(r"\b(?:todo|to-do|blocked on|waiting on|waiting for|next step|action item|remaining task)\b", unit, re.IGNORECASE))
        if not task_like_unit:
            for pattern in self.PROJECT_PATTERNS:
                for match in pattern.finditer(unit):
                    target = self._clean_project_target(match.groupdict().get("object"))
                    normalized = self._normalize_context_target(target)
                    if not normalized or normalized in seen:
                        continue
                    results.append(self._make_generic_object(node, chunk, unit, offset, "project", target, "project", 0.62))
                    seen.add(normalized)
        project_fallback = self._project_target_from_context(node, unit)
        if project_fallback:
            normalized = self._normalize_context_target(project_fallback)
            if normalized and normalized not in seen:
                results.append(self._make_generic_object(node, chunk, unit, offset, "project", project_fallback, "project", 0.6))
                seen.add(normalized)
        return results

    def _extract_artifact_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        seen: set[str] = set()
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        has_artifact_hint = bool(self.ARTIFACT_HINT_PATTERN.search(unit))
        for match in self.FILE_PATH_PATTERN.finditer(unit):
            raw_target = match.groupdict().get("path")
            target = self._clean_artifact_target(raw_target)
            normalized_path = self._normalize_artifact_path(target)
            if not normalized_path or normalized_path in seen:
                continue
            artifact_type = self._infer_artifact_type(target, unit)
            artifact_title = self._artifact_title(target)
            object_text = " ".join(part for part in [artifact_title, artifact_type, target] if part).strip()
            confidence = 0.84 if has_artifact_hint else 0.72
            results.append(
                MemoryObject(
                    object_type="artifact",
                    object_text=object_text,
                    source_chunk_id=source_chunk_id,
                    source_node_id=node.id,
                    subject="artifact",
                    predicate="referenced",
                    entity=artifact_title.lower() if artifact_title else normalized_path,
                    attribute="artifact_type",
                    new_value=artifact_type,
                    canonical_key=f"artifact:{normalized_path}",
                    source_unit_text=unit.strip(),
                    sequence_index=sequence_index,
                    content_hash=stable_content_hash(object_text),
                    confidence=confidence,
                    session_id=node.session_id or node.content_ref,
                    turn_index=offset,
                    timestamp=node.created_at,
                    snapshot_key=f"artifact:{normalized_path}",
                    merge_policy="latest_wins",
                    source_ref=target,
                    metadata_json=json.dumps({"artifact_path": target, "artifact_title": artifact_title}, ensure_ascii=False),
                    object_id=self._stable_object_id(
                        object_type="artifact",
                        source_chunk_id=source_chunk_id,
                        canonical_key=f"artifact:{normalized_path}",
                        sequence_index=sequence_index,
                        object_text=object_text,
                    ),
                )
            )
            seen.add(normalized_path)
        return results

    def _extract_pattern_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        seen: set[str] = set()
        for pattern in self.PATTERN_PATTERNS:
            for match in pattern.finditer(unit):
                target = self._clean_context_target(match.groupdict().get("object"))
                normalized = self._normalize_context_target(target)
                if not normalized or normalized in seen:
                    continue
                results.append(self._make_generic_object(node, chunk, unit, offset, "pattern", target, "pattern", 0.58))
                seen.add(normalized)
        return results

    def _extract_open_loop_objects(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> list[MemoryObject]:
        results: list[MemoryObject] = []
        seen: set[str] = set()
        for pattern in self.OPEN_LOOP_PATTERNS:
            for match in pattern.finditer(unit):
                target = self._clean_context_target(match.groupdict().get("object"))
                normalized = self._normalize_context_target(target)
                if not normalized or normalized in seen:
                    continue
                item = self._make_generic_object(node, chunk, unit, offset, "open_loop", target, "pending", 0.7)
                item.status = self._infer_open_loop_status(unit)
                item.metadata_json = json.dumps(
                    {
                        "priority": self._infer_open_loop_priority(unit),
                        "status_inferred_from_text": item.status,
                    },
                    ensure_ascii=False,
                )
                item.snapshot_key = f"open_loop:{normalized}"
                item.merge_policy = "latest_wins"
                results.append(item)
                seen.add(normalized)
        return results

    def _make_generic_object(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
        object_type: str,
        target: str,
        attribute: str,
        confidence: float,
    ) -> MemoryObject:
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        normalized = self._normalize_context_target(target)
        canonical_key = f"{object_type}:{normalized}"
        return MemoryObject(
            object_type=object_type,
            object_text=target,
            subject="user",
            predicate=attribute,
            entity=normalized,
            attribute=attribute,
            canonical_key=canonical_key,
            source_unit_text=unit.strip(),
            sequence_index=sequence_index,
            content_hash=stable_content_hash(target),
            confidence=confidence,
            source_chunk_id=source_chunk_id,
            source_node_id=node.id,
            session_id=node.session_id or node.content_ref,
            turn_index=offset,
            timestamp=node.created_at,
            snapshot_key=canonical_key,
            merge_policy="latest_wins" if object_type in {"goal", "decision", "project"} else "append",
            object_id=self._stable_object_id(
                object_type=object_type,
                source_chunk_id=source_chunk_id,
                canonical_key=canonical_key,
                sequence_index=sequence_index,
                object_text=target,
            ),
        )

    def _enrich_object_defaults(self, node: MemoryNode, item: MemoryObject) -> None:
        item.scope = item.scope or node.scope
        item.workspace = item.workspace or node.workspace
        item.project = item.project or node.project
        if node.session_id:
            item.session_id = node.session_id
        else:
            item.session_id = item.session_id or node.content_ref
        item.source_type = item.source_type or node.source_type
        item.source_ref = item.source_ref or node.source_ref or node.content_ref
        item.extraction_method = item.extraction_method or "heuristic"
        item.verification_status = item.verification_status or node.verification_status or "unverified"
        item.updated_at = item.updated_at or node.updated_at or node.created_at

    def _augment_derived_objects(self, objects: list[MemoryObject]) -> list[MemoryObject]:
        augmented = list(objects)
        for item in list(objects):
            if item.entity:
                entity_text = str(item.entity)
                augmented.append(
                    MemoryObject(
                        object_type="entity",
                        object_text=f"entity {entity_text}",
                        subject=item.subject,
                        predicate="entity",
                        entity=entity_text,
                        attribute=item.attribute or item.object_type,
                        canonical_key=f"entity:{entity_text}",
                        source_unit_text=item.source_unit_text,
                        sequence_index=item.sequence_index,
                        content_hash=stable_content_hash(entity_text),
                        confidence=max(0.4, float(item.confidence) * 0.9),
                        source_chunk_id=item.source_chunk_id,
                        source_node_id=item.source_node_id,
                        session_id=item.session_id,
                        turn_index=item.turn_index,
                        timestamp=item.timestamp,
                        merge_policy="latest_wins",
                        object_id=self._stable_object_id(
                            object_type="entity",
                            source_chunk_id=item.source_chunk_id,
                            canonical_key=f"entity:{entity_text}",
                            sequence_index=int(item.sequence_index or 0),
                            object_text=entity_text,
                        ),
                    )
                )
            if item.object_type in {"temporal_reference", "state_update"}:
                event_text = self._clean_context_target(item.event_text or item.object_text)
                if event_text:
                    augmented.append(
                        MemoryObject(
                            object_type="event",
                            object_text=event_text,
                            subject=item.subject,
                            predicate="event",
                            entity=self._normalize_context_target(event_text),
                            attribute=item.attribute or "event",
                            canonical_key=f"event:{self._normalize_context_target(event_text)}",
                            temporal_marker=item.temporal_marker,
                            source_unit_text=item.source_unit_text,
                            sequence_index=item.sequence_index,
                            content_hash=stable_content_hash(event_text),
                            confidence=max(0.42, float(item.confidence) * 0.88),
                            source_chunk_id=item.source_chunk_id,
                            source_node_id=item.source_node_id,
                            session_id=item.session_id,
                            turn_index=item.turn_index,
                            timestamp=item.timestamp,
                            merge_policy="append",
                            object_id=self._stable_object_id(
                                object_type="event",
                                source_chunk_id=item.source_chunk_id,
                                canonical_key=f"event:{self._normalize_context_target(event_text)}",
                                sequence_index=int(item.sequence_index or 0),
                                object_text=event_text,
                            ),
                        )
                    )
            if item.object_type == "state_update":
                state_text = self._clean_context_target(item.new_value or item.object_text)
                normalized = self._normalize_context_target(state_text)
                if state_text and normalized:
                    augmented.append(
                        MemoryObject(
                            object_type="state",
                            object_text=state_text,
                            subject=item.subject,
                            predicate="state",
                            entity=normalized,
                            attribute=item.attribute or "state",
                            canonical_key=f"state:{item.canonical_key or normalized}",
                            temporal_marker=item.temporal_marker,
                            source_unit_text=item.source_unit_text,
                            sequence_index=item.sequence_index,
                            content_hash=stable_content_hash(state_text),
                            confidence=max(0.44, float(item.confidence) * 0.88),
                            source_chunk_id=item.source_chunk_id,
                            source_node_id=item.source_node_id,
                            session_id=item.session_id,
                            turn_index=item.turn_index,
                            timestamp=item.timestamp,
                            merge_policy="latest_wins",
                            object_id=self._stable_object_id(
                                object_type="state",
                                source_chunk_id=item.source_chunk_id,
                                canonical_key=f"state:{item.canonical_key or normalized}",
                                sequence_index=int(item.sequence_index or 0),
                                object_text=state_text,
                            ),
                        )
                    )
        return augmented

    def build_representations(
        self,
        node: MemoryNode,
        chunks: list[dict[str, Any]],
        objects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        object_index: dict[str, list[dict[str, Any]]] = {}
        for obj in objects:
            source_chunk_id = str(obj.get("source_chunk_id") or "")
            if source_chunk_id:
                object_index.setdefault(source_chunk_id, []).append(dict(obj))
        representations: list[dict[str, Any]] = []
        for chunk in chunks:
            related_objects = list(object_index.get(str(chunk.get("chunk_id") or ""), []))
            if chunk.get("grain") == "local":
                for member_id in chunk.get("member_chunk_ids") or []:
                    related_objects.extend(object_index.get(str(member_id), []))
            retrieval_summary = self._build_retrieval_summary(str(chunk.get("text") or ""), related_objects)
            structured_summary = self._build_structured_summary(str(chunk.get("text") or ""), related_objects)
            time_bucket = self._infer_time_bucket(str(chunk.get("text") or ""), related_objects, str(chunk.get("created_at") or node.created_at))
            entity_tags = self._collect_entity_tags(related_objects)
            task_type_tag = self._infer_task_type_tag(str(chunk.get("text") or ""), related_objects)
            retrieval_signature = self._build_retrieval_signature(
                related_objects=related_objects,
                time_bucket=time_bucket,
                task_type_tag=task_type_tag,
                entity_tags=entity_tags,
                text=str(chunk.get("text") or ""),
            )
            chunk["retrieval_summary"] = retrieval_summary
            chunk["structured_summary"] = structured_summary
            chunk["retrieval_signature"] = retrieval_signature
            chunk["time_bucket"] = time_bucket
            chunk["entity_tags"] = entity_tags
            chunk["task_type_tag"] = task_type_tag
            for proxy_kind, text in (
                ("summary", retrieval_summary),
                ("structured", structured_summary),
                ("signature", retrieval_signature),
            ):
                if not text:
                    continue
                chunk_id = str(chunk.get("chunk_id") or "")
                representation_seed = f"{chunk_id}|{proxy_kind}|{text}"
                representations.append(
                    {
                        "representation_id": f"repr_{stable_content_hash(representation_seed)[:16]}",
                        "parent_id": chunk_id,
                        "parent_type": "chunk",
                        "proxy_kind": proxy_kind,
                        "text": text,
                        "content_hash": stable_content_hash(text),
                        "scope": node.scope,
                        "workspace": node.workspace,
                        "project": node.project,
                        "session_id": node.session_id,
                        "time_bucket": time_bucket,
                        "entity_tags": entity_tags,
                        "task_type_tag": task_type_tag,
                        "created_at": str(chunk.get("created_at") or node.created_at),
                    }
                )

        node_text = (node.raw_content or "").strip() or node.summary
        node.retrieval_summary = self._build_retrieval_summary(node_text, objects)
        node.structured_summary = self._build_structured_summary(node_text, objects)
        node.time_bucket = self._infer_time_bucket(node_text, objects, node.created_at)
        node.entity_tags = self._collect_entity_tags(objects)
        node.task_type_tag = self._infer_task_type_tag(node_text, objects)
        node.retrieval_signature = self._build_retrieval_signature(
            related_objects=objects,
            time_bucket=node.time_bucket or "",
            task_type_tag=node.task_type_tag or "",
            entity_tags=node.entity_tags or "",
            text=node_text,
        )
        for proxy_kind, text in (
            ("summary", node.retrieval_summary),
            ("structured", node.structured_summary),
            ("signature", node.retrieval_signature),
        ):
            if not text:
                continue
            representation_seed = f"{node.id}|node|{proxy_kind}|{text}"
            representations.append(
                {
                    "representation_id": f"repr_{stable_content_hash(representation_seed)[:16]}",
                    "parent_id": node.id,
                    "parent_type": "node",
                    "proxy_kind": proxy_kind,
                    "text": text,
                    "content_hash": stable_content_hash(text),
                    "scope": node.scope,
                    "workspace": node.workspace,
                    "project": node.project,
                    "session_id": node.session_id,
                    "time_bucket": node.time_bucket,
                    "entity_tags": node.entity_tags,
                    "task_type_tag": node.task_type_tag,
                    "created_at": node.created_at,
                }
            )
        return representations

    def _build_retrieval_summary(self, text: str, related_objects: list[dict[str, Any]]) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        head = sentences[0][:220]
        time_part = self._first_matching_value(related_objects, "temporal_marker")
        entity_part = self._collect_entity_tags(related_objects)
        constraint_part = ", ".join(
            str(obj.get("entity") or obj.get("object_text") or "")
            for obj in related_objects
            if str(obj.get("object_type") or "") == "constraint"
        )[:120]
        result = head
        extras = []
        if time_part:
            extras.append(f"time={time_part}")
        if entity_part:
            extras.append(f"entities={entity_part}")
        if constraint_part:
            extras.append(f"constraints={constraint_part}")
        if extras:
            result = f"{head} [{' | '.join(extras)}]"
        return result[:320]

    def _build_structured_summary(self, text: str, related_objects: list[dict[str, Any]]) -> str:
        if not text and not related_objects:
            return ""
        parts = [
            f"types={','.join(sorted({str(obj.get('object_type') or '') for obj in related_objects if obj.get('object_type')}))}" if related_objects else "",
            f"entities={self._collect_entity_tags(related_objects)}" if related_objects else "",
            f"time={self._first_matching_value(related_objects, 'temporal_marker')}" if related_objects else "",
            f"focus={self._infer_task_type_tag(text, related_objects)}",
        ]
        compact = "; ".join(part for part in parts if part)
        return compact[:240]

    def _build_retrieval_signature(
        self,
        *,
        related_objects: list[dict[str, Any]],
        time_bucket: str,
        task_type_tag: str,
        entity_tags: str,
        text: str,
    ) -> str:
        object_types = ",".join(sorted({str(obj.get("object_type") or "") for obj in related_objects if obj.get("object_type")}))[:80]
        keywords = ",".join(tokenize(text)[:6])
        fields = [
            f"topic={keywords}" if keywords else "",
            f"time={time_bucket}" if time_bucket else "",
            f"task={task_type_tag}" if task_type_tag else "",
            f"entities={entity_tags}" if entity_tags else "",
            f"types={object_types}" if object_types else "",
        ]
        return " | ".join(field for field in fields if field)[:220]

    def _infer_time_bucket(self, text: str, related_objects: list[dict[str, Any]], fallback_time: str) -> str:
        for obj in related_objects:
            marker = str(obj.get("temporal_marker") or "").strip().lower()
            if marker:
                return marker
        lowered = text.lower()
        if "latest" in lowered or "currently" in lowered or "now" in lowered:
            return "latest"
        if "previously" in lowered or "used to" in lowered or "before" in lowered:
            return "previous"
        if fallback_time:
            return str(fallback_time)[:10]
        return "unknown"

    def _collect_entity_tags(self, related_objects: list[dict[str, Any]]) -> str:
        entities: list[str] = []
        for obj in related_objects:
            entity = str(obj.get("entity") or "").strip().lower()
            if entity and entity not in entities:
                entities.append(entity)
            if len(entities) >= 5:
                break
        return ",".join(entities)

    def _infer_task_type_tag(self, text: str, related_objects: list[dict[str, Any]]) -> str:
        object_types = {str(obj.get("object_type") or "") for obj in related_objects}
        lowered = text.lower()
        if {"preference", "persona", "personal_context", "relation"} & object_types:
            return "profile"
        if {"goal", "decision", "open_loop", "project", "pattern"} & object_types:
            return "planning"
        if {"state_update", "temporal_reference", "event"} & object_types:
            return "temporal"
        if {"solution_card", "constraint"} & object_types or any(token in lowered for token in self.PROBLEM_HINTS):
            return "debug"
        if {"fact", "state"} & object_types:
            return "fact"
        return "general"

    @staticmethod
    def _first_matching_value(related_objects: list[dict[str, Any]], key: str) -> str:
        for obj in related_objects:
            value = str(obj.get(key) or "").strip()
            if value:
                return value
        return ""

    def _extract_solution_object(
        self,
        node: MemoryNode,
        chunk: dict[str, Any],
        unit: str,
        offset: int,
    ) -> MemoryObject | None:
        lowered = unit.lower()
        if not any(hint in lowered for hint in self.PROBLEM_HINTS):
            return None
        if not any(hint in lowered for hint in self.SOLUTION_HINTS):
            return None
        source_chunk_id = self._source_chunk_id(chunk)
        sequence_index = self._sequence_index(chunk, offset)
        return MemoryObject(
            object_type="solution_card",
            object_text=unit.strip(),
            subject="system",
            predicate="resolves",
            event_text=unit.strip(),
            canonical_key=unit.strip().lower(),
            sequence_index=sequence_index,
            source_unit_text=unit.strip(),
            content_hash=stable_content_hash(unit.strip()),
            confidence=0.64,
            source_chunk_id=source_chunk_id,
            source_node_id=node.id,
            session_id=node.content_ref,
            turn_index=offset,
            timestamp=node.created_at,
            object_id=self._stable_object_id(
                object_type="solution_card",
                source_chunk_id=source_chunk_id,
                canonical_key=unit.strip().lower(),
                sequence_index=sequence_index,
                object_text=unit.strip(),
            ),
        )

    def _source_chunk_id(self, chunk: dict[str, Any]) -> str:
        return str(chunk.get("source_chunk_id") or chunk.get("chunk_id") or "")

    def _sequence_index(self, chunk: dict[str, Any], offset: int) -> int:
        return int(chunk.get("chunk_index") or 0) * 100 + int(offset)

    def _stable_object_id(
        self,
        object_type: str,
        source_chunk_id: str,
        canonical_key: str,
        sequence_index: int,
        object_text: str,
    ) -> str:
        seed = "|".join(
            [
                object_type,
                source_chunk_id,
                canonical_key,
                str(sequence_index),
                stable_content_hash(object_text),
            ]
        )
        return f"obj_{stable_content_hash(seed)[:16]}"

    def _clean_state_text(self, value: str | None) -> str:
        if not value:
            return ""
        cleaned = re.sub(r"\s+", " ", value).strip(" -:;,.")
        return cleaned

    def _infer_temporal_marker(self, unit: str, old_value: str | None, new_value: str | None) -> str:
        lowered = unit.lower()
        if old_value and new_value:
            return "change"
        if any(token in lowered for token in ("now", "currently", "latest")):
            return "latest"
        if any(token in lowered for token in ("previously", "formerly", "used to", "before", "prior")):
            return "previous"
        return "point"

    def _infer_state_attribute(self, entity: str, old_value: str | None, new_value: str | None) -> str:
        entity_text = entity.lower()
        values = " ".join(filter(None, [old_value or "", new_value or ""])).lower()
        if "time" in entity_text or "schedule" in entity_text or self._looks_temporal_value(old_value) or self._looks_temporal_value(new_value):
            return "time_state"
        if any(str(value or "").lower().startswith(prefix) for value in (old_value, new_value) for prefix in ("in ", "at ")):
            return "location_state"
        if any(token in entity_text or token in values for token in ("live", "location", "city", "address", "moved", "resides")):
            return "location_state"
        if any(token in entity_text or token in values for token in ("job", "role", "team", "works at", "company")):
            return "role_state"
        if any(token in entity_text or token in values for token in ("plan", "status", "stage", "version")):
            return "status_state"
        return "state"

    def _state_canonical_key(self, entity: str, attribute: str) -> str:
        entity_key = re.sub(r"\s+", " ", entity).strip().lower()
        return f"{attribute}:{entity_key}"

    def _format_state_object_text(self, entity: str, old_value: str | None, new_value: str | None) -> str:
        if old_value and new_value:
            return f"{entity} changed from {old_value} to {new_value}"
        if new_value:
            return f"{entity} is now {new_value}"
        if old_value:
            return f"{entity} was previously {old_value}"
        return entity

    def _clean_temporal_phrase(self, raw_time: str | None) -> str:
        if not raw_time:
            return ""
        cleaned = re.sub(r"\s+", " ", raw_time).strip(" -:;,.")
        return cleaned

    def _normalize_temporal_phrase(self, raw_time: str | None) -> str:
        cleaned = self._clean_temporal_phrase(raw_time)
        return cleaned.lower()

    def _infer_temporal_event_label(self, unit: str, time_phrase: str) -> str:
        if not time_phrase:
            return ""
        label = re.sub(re.escape(time_phrase), " ", unit, flags=re.IGNORECASE)
        label = re.sub(r"\b(?:on|in|at|during|around|about|from|to|last|next)\b", " ", label, flags=re.IGNORECASE)
        label = re.sub(r"\b(?:i|we|my|our|me)\b", " ", label, flags=re.IGNORECASE)
        label = re.sub(r"\s+", " ", label).strip(" -:;,.")
        return label if len(label) >= 4 else ""

    def _normalize_temporal_event_label(self, label: str | None) -> str:
        if not label:
            return ""
        normalized = re.sub(r"\s+", " ", label).strip().lower()
        return normalized

    def _looks_temporal_value(self, value: str | None) -> bool:
        if not value:
            return False
        lowered = value.lower()
        if re.search(r"\b(?:today|tomorrow|yesterday|ago|week|month|year)\b", lowered):
            return True
        if re.search(rf"\b(?:{self.WEEKDAY_PATTERN}|{self.MONTH_PATTERN})\b", lowered, flags=re.IGNORECASE):
            return True
        return bool(re.search(r"\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}", lowered))

    def _clean_context_target(self, raw_target: str | None) -> str:
        if not raw_target:
            return ""
        cleaned = raw_target.strip(" -:;,.")
        cleaned = re.split(r"\b(?:because|since|but|although|though|if|when)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if len(cleaned) >= 3 else ""

    def _normalize_context_target(self, target: str | None) -> str:
        if not target:
            return ""
        return re.sub(r"\s+", " ", target).strip().lower()

    def _clean_artifact_target(self, raw_target: str | None) -> str:
        if not raw_target:
            return ""
        cleaned = raw_target.strip(" \"'()[]{}<>-:;,.")
        cleaned = cleaned.replace("\\\\", "\\")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _clean_project_target(self, raw_target: str | None) -> str:
        cleaned = self._clean_context_target(raw_target)
        if not cleaned:
            return ""
        cleaned = re.sub(r"^\s*project\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.split(
            r"\b(?:workspace|goal(?:\s+is\s+to)?|objective|milestone|deliverable|rollout|validation|checkpoint|todo|blocker|blocked|plan(?:\s+is\s+to)?)\b",
            cleaned,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        cleaned = re.sub(r"\b(?:the|a|an)\b\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")
        return cleaned

    def _normalize_artifact_path(self, target: str | None) -> str:
        cleaned = self._clean_artifact_target(target)
        if not cleaned or "." not in cleaned:
            return ""
        normalized = cleaned.replace("\\", "/").strip().lower()
        normalized = re.sub(r"/+", "/", normalized)
        return normalized

    def _artifact_title(self, target: str) -> str:
        normalized = self._clean_artifact_target(target).replace("\\", "/")
        if not normalized:
            return ""
        return normalized.rsplit("/", 1)[-1]

    def _infer_artifact_type(self, target: str, unit: str) -> str:
        lowered_target = self._clean_artifact_target(target).lower()
        extension = lowered_target.rsplit(".", 1)[-1] if "." in lowered_target else ""
        mapping = {
            "md": "markdown",
            "markdown": "markdown",
            "txt": "text",
            "json": "json",
            "yaml": "yaml",
            "yml": "yaml",
            "toml": "config",
            "ini": "config",
            "cfg": "config",
            "conf": "config",
            "py": "code",
            "js": "code",
            "ts": "code",
            "tsx": "code",
            "jsx": "code",
            "java": "code",
            "go": "code",
            "rs": "code",
            "cpp": "code",
            "c": "code",
            "h": "code",
            "hpp": "code",
            "rb": "code",
            "sql": "query",
            "csv": "table",
            "tsv": "table",
            "log": "log",
            "pdf": "report",
            "ipynb": "notebook",
            "sh": "script",
            "ps1": "script",
            "bat": "script",
            "html": "markup",
            "css": "style",
        }
        if extension in mapping:
            return mapping[extension]
        lowered_unit = unit.lower()
        if "report" in lowered_unit or "summary" in lowered_unit:
            return "report"
        if "result" in lowered_unit or "output" in lowered_unit:
            return "result"
        if "config" in lowered_unit:
            return "config"
        return "file"

    def _project_target_from_context(self, node: MemoryNode, unit: str) -> str:
        if not node.project:
            return ""
        if not self.PROJECT_SIGNAL_PATTERN.search(unit):
            return ""
        return str(node.project)

    def _infer_open_loop_status(self, unit: str) -> str:
        lowered = unit.lower()
        if any(token in lowered for token in ("blocked on", "waiting on", "waiting for")):
            return "blocked"
        if any(token in lowered for token in ("defer ", "deferred ", "postpone", "postponed")):
            return "deferred"
        if re.search(r"\b(?:completed|finished|resolved|closed|checked off)\b", lowered) and not re.search(
            r"\b(?:not\s+finished|not\s+yet\s+done|unfinished|unresolved|still\s+open)\b",
            lowered,
        ):
            return "closed"
        return "open"

    def _infer_open_loop_priority(self, unit: str) -> str:
        lowered = unit.lower()
        if any(token in lowered for token in ("urgent", "critical", "p0", "high priority")):
            return "high"
        if any(token in lowered for token in ("low priority", "nice to have", "someday")):
            return "low"
        return "normal"

    def create_edges_for_new_node(self, node: MemoryNode, limit: int = 6) -> list[MemoryEdge]:
        existing_nodes = self.storage.fetch_nodes("id != ?", (node.id,))
        scored: list[tuple[float, dict[str, Any]]] = []
        for other in existing_nodes:
            semantic = lexical_score(node.summary, other.get("summary") or "")
            task_weight = 0.4 if node.zone == other["zone"] or node.cell == other["cell"] else 0.0
            temporal_weight = self._temporal_edge_weight(node, other)
            causal_weight = self._causal_edge_weight(node, other)
            structural_weight = 0.35 if self._structurally_similar(node, other) else 0.0
            creative_weight = 0.3 if self._weak_link_potential(node, other) else 0.0
            score = semantic + task_weight + temporal_weight + causal_weight + structural_weight + creative_weight
            if score > 0.25:
                scored.append((score, other))
        scored.sort(key=lambda x: x[0], reverse=True)
        edges = []
        for _, other in scored[:limit]:
            semantic = lexical_score(node.summary, other.get("summary") or "")
            edge = MemoryEdge(
                source_id=node.id,
                target_id=other["id"],
                semantic_weight=min(semantic, 1.0),
                task_weight=0.4 if node.zone == other["zone"] or node.cell == other["cell"] else 0.0,
                temporal_weight=self._temporal_edge_weight(node, other),
                causal_weight=self._causal_edge_weight(node, other),
                creative_weight=0.3 if self._weak_link_potential(node, other) else 0.0,
                structural_weight=0.35 if self._structurally_similar(node, other) else 0.0,
            )
            edges.append(edge)
        return edges

    @staticmethod
    def _parse_created_at(value: str | None) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _temporal_edge_weight(self, node: MemoryNode, other: dict[str, Any]) -> float:
        node_dt = self._parse_created_at(node.created_at)
        other_dt = self._parse_created_at(str(other.get("created_at") or ""))
        node_bucket = str(node.time_bucket or "").strip().lower()
        other_bucket = str(other.get("time_bucket") or "").strip().lower()
        weight = 0.0
        if node_dt is not None and other_dt is not None:
            day_gap = abs((node_dt.date() - other_dt.date()).days)
            if day_gap == 0:
                weight += 0.14
            elif day_gap == 1:
                weight += 0.08
            elif day_gap <= 3:
                weight += 0.03
        elif node.created_at[:10] == str(other.get("created_at", ""))[:10]:
            weight += 0.12
        if node_bucket and other_bucket:
            if node_bucket == other_bucket:
                weight += 0.06
            elif {node_bucket, other_bucket} <= {"latest", "previous", "point"}:
                weight += 0.02
        if node.sector == other.get("sector"):
            weight += 0.02
        return round(min(weight, 0.28), 4)

    def _causal_edge_weight(self, node: MemoryNode, other: dict[str, Any]) -> float:
        node_dt = self._parse_created_at(node.created_at)
        other_dt = self._parse_created_at(str(other.get("created_at") or ""))
        node_bucket = str(node.time_bucket or "").strip().lower()
        other_bucket = str(other.get("time_bucket") or "").strip().lower()
        weight = 0.0
        if node.sector == other.get("sector"):
            weight += 0.14
        if node.zone == other.get("zone") or node.cell == other.get("cell"):
            weight += 0.06
        if node_dt is not None and other_dt is not None:
            delta_hours = (node_dt - other_dt).total_seconds() / 3600.0
            if 0.0 < delta_hours <= 72.0:
                weight += 0.08
            elif -24.0 <= delta_hours <= 0.0:
                weight += 0.03
        if node_bucket and other_bucket:
            if node_bucket == "latest" and other_bucket in {"previous", "point"}:
                weight += 0.04
            elif node_bucket == "point" and other_bucket == "previous":
                weight += 0.03
        return round(min(weight, 0.3), 4)

    def _structurally_similar(self, node: MemoryNode, other: dict[str, Any]) -> bool:
        tags = set(tokenize(node.tags or ""))
        other_tags = set(tokenize(other.get("tags") or ""))
        if tags and other_tags and len(tags & other_tags) >= 1:
            return True
        keywords = ["cache", "memory", "token", "parallel", "sqlite", "graph", "scheduler"]
        text_a = (node.summary + " " + (node.raw_content or "")).lower()
        text_b = ((other.get("summary") or "") + " " + (other.get("raw_content") or "")).lower()
        return sum(1 for k in keywords if k in text_a and k in text_b) >= 2

    def _weak_link_potential(self, node: MemoryNode, other: dict[str, Any]) -> bool:
        same_zone = node.zone == other["zone"]
        same_sector = node.sector == other["sector"]
        lexical = lexical_score(node.summary, other.get("summary") or "")
        return (not same_zone) and lexical > 0.15 and (not same_sector or node.shell != other["shell"])
