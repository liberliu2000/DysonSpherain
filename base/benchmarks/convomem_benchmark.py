from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from longmemeval_benchmark import (
    KS,
    BenchmarkWorkspaceManager,
    SHARED_EMBEDDING_CACHE_DIR as DEFAULT_SHARED_EMBEDDING_CACHE_DIR,
    classify_bottleneck,
    evaluate_retrieval,
    flatten_numeric_metrics,
    materialize_ingest_profile_for_question,
    rank_evidence,
    rank_hybrid,
    rank_vector,
    summarize_numeric_metrics,
    summarize_question_profile,
)


CATEGORY_TO_TASK = {
    "user": "qa",
    "assistant_facts": "qa",
    "changing": "temporal_reasoning",
    "abstention": "qa",
    "preference": "preference_lookup",
    "implicit_connection": "temporal_reasoning",
}
PREFERENCE_TERMS = {
    "prefer",
    "preferred",
    "prefers",
    "favorite",
    "favourite",
    "like",
    "likes",
    "love",
    "loves",
    "enjoy",
    "enjoys",
    "dislike",
    "avoid",
    "avoids",
    "hate",
    "hates",
}
TEMPORAL_TERMS = {
    "before",
    "after",
    "earlier",
    "later",
    "latest",
    "current",
    "currently",
    "when",
    "timeline",
    "ago",
    "yesterday",
    "today",
    "tomorrow",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "day",
    "days",
}


def path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


WORKSPACE_CACHE_ROOT = path_from_env(
    "SPHERE_CONVOMEM_CACHE_ROOT",
    ROOT / "benchmarks" / ".cache" / "convomem_workspaces",
)
SHARED_EMBEDDING_CACHE_DIR = path_from_env(
    "SPHERE_CONVOMEM_EMBED_CACHE_ROOT",
    DEFAULT_SHARED_EMBEDDING_CACHE_DIR,
)
JSON_STREAM_CHUNK_SIZE = 1 << 20
CASE_GC_INTERVAL = 128


def default_data_root() -> Path:
    candidates = [
        ROOT / "data" / "benchmarks" / "convomem",
        ROOT.parent.parent / "tmp_benchmark_sources" / "ConvoMem",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def message_speaker(message: dict[str, Any]) -> str:
    return normalize_text(
        message.get("speaker")
        or message.get("role")
        or message.get("sender")
        or message.get("author")
        or "?"
    )


def message_body(message: dict[str, Any]) -> str:
    return normalize_text(
        message.get("text")
        or message.get("content")
        or message.get("message")
        or message.get("utterance")
        or ""
    )


def format_message(message: dict[str, Any]) -> str:
    speaker = message_speaker(message)
    text = message_body(message)
    return f"{speaker}: {text}" if text else f"{speaker}:"


def message_signature(message: dict[str, Any]) -> str:
    speaker = message_speaker(message).lower()
    text = message_body(message).lower()
    if not text:
        return ""
    return f"{speaker}::{text}"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def init_metric_bucket() -> dict[str, list[float]]:
    bucket: dict[str, list[float]] = {}
    for metric in ("recall_frac", "recall_any", "recall_all", "ndcg_any"):
        for k in KS:
            bucket[f"{metric}@{k}"] = []
    return bucket


def summarize_metric_bucket(bucket: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in bucket.items()
    }


def recall_fraction(ranked_ids: list[str], correct_ids: set[str], k: int) -> float:
    if not correct_ids:
        return 1.0
    return len(set(ranked_ids[:k]) & correct_ids) / len(correct_ids)


def infer_task_type(question: str, category_name: str) -> str:
    lowered = question.lower()
    if category_name == "preference" or any(term in lowered for term in PREFERENCE_TERMS):
        return "preference_lookup"
    if category_name in {"changing", "implicit_connection"} or any(
        term in lowered for term in TEMPORAL_TERMS
    ):
        return "temporal_reasoning"
    return CATEGORY_TO_TASK.get(category_name, "qa")


def discover_json_files(data_path: Path) -> list[Path]:
    if data_path.is_file():
        return [data_path]
    if not data_path.exists():
        raise FileNotFoundError(f"ConvoMem data path not found: {data_path}")

    all_json = sorted(path for path in data_path.rglob("*.json") if path.is_file())
    preferred = [path for path in all_json if "pre_mixed_testcases" in path.as_posix()]
    if preferred:
        return preferred
    batched = [path for path in all_json if path.name.startswith("batched_")]
    if batched:
        return batched
    return all_json


def iter_json_records(json_file: Path) -> Iterator[tuple[dict[str, Any], float]]:
    file_load_start = perf_counter()
    current_object_parts: list[str] = []
    object_start_idx: int | None = None
    root_kind: str | None = None
    root_completed = False
    object_depth = 0
    in_string = False
    escape_next = False
    emitted = 0

    with json_file.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(JSON_STREAM_CHUNK_SIZE)
            if not chunk:
                break

            idx = 0
            chunk_len = len(chunk)
            while idx < chunk_len:
                ch = chunk[idx]

                if object_depth > 0:
                    if in_string:
                        if escape_next:
                            escape_next = False
                        elif ch == "\\":
                            escape_next = True
                        elif ch == '"':
                            in_string = False
                    else:
                        if ch == '"':
                            in_string = True
                        elif ch == "{":
                            object_depth += 1
                        elif ch == "}":
                            object_depth -= 1
                            if object_depth == 0:
                                if object_start_idx is None:
                                    raise json.JSONDecodeError(
                                        f"Malformed streamed JSON in {json_file}",
                                        chunk,
                                        idx,
                                    )
                                current_object_parts.append(chunk[object_start_idx : idx + 1])
                                payload = json.loads("".join(current_object_parts))
                                emitted += 1
                                load_ms = (
                                    round((perf_counter() - file_load_start) * 1000.0, 2)
                                    if emitted == 1
                                    else 0.0
                                )
                                current_object_parts.clear()
                                object_start_idx = None
                                if isinstance(payload, dict):
                                    yield payload, load_ms
                                else:
                                    raise json.JSONDecodeError(
                                        f"Expected JSON object records in {json_file}",
                                        "",
                                        0,
                                    )
                                if root_kind == "object":
                                    root_completed = True
                    idx += 1
                    continue

                if root_completed:
                    if not ch.isspace():
                        raise json.JSONDecodeError(
                            f"Unexpected trailing content in {json_file}",
                            chunk,
                            idx,
                        )
                    idx += 1
                    continue

                if ch.isspace():
                    idx += 1
                    continue

                if root_kind is None:
                    if ch == "[":
                        root_kind = "array"
                        idx += 1
                        continue
                    if ch == "{":
                        root_kind = "object"
                        object_depth = 1
                        object_start_idx = idx
                        idx += 1
                        continue
                    raise json.JSONDecodeError(
                        f"Unsupported JSON root in {json_file}",
                        chunk,
                        idx,
                    )

                if root_kind == "array":
                    if ch == ",":
                        idx += 1
                        continue
                    if ch == "]":
                        root_completed = True
                        idx += 1
                        continue
                    if ch == "{":
                        object_depth = 1
                        object_start_idx = idx
                        idx += 1
                        continue
                    raise json.JSONDecodeError(
                        f"Expected JSON object entry in {json_file}",
                        chunk,
                        idx,
                    )

                raise json.JSONDecodeError(
                    f"Unexpected content after JSON object in {json_file}",
                    chunk,
                    idx,
                )

            if object_depth > 0 and object_start_idx is not None:
                current_object_parts.append(chunk[object_start_idx:])
                object_start_idx = 0

    if object_depth > 0:
        raise json.JSONDecodeError(
            f"Unterminated JSON object in {json_file}",
            "",
            0,
        )

    if root_kind is None:
        raise FileNotFoundError(f"No ConvoMem JSON payload found in {json_file}")


def iter_cases(data_path: Path, limit: int = 0) -> Iterator[tuple[dict[str, Any], float]]:
    yielded = 0
    for json_file in discover_json_files(data_path):
        for case_index, (item, file_load_ms) in enumerate(iter_json_records(json_file), start=1):
            yielded += 1
            case = dict(item)
            case["_source_file"] = str(json_file)
            case["_source_case_index"] = case_index
            yield case, file_load_ms
            if limit > 0 and yielded >= limit:
                return
    if yielded == 0:
        raise FileNotFoundError(f"No ConvoMem JSON cases found under {data_path}")


class ResultSpool:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0
        self._handle = path.open("w", encoding="utf-8")

    def append(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False))
        self._handle.write("\n")
        self.count += 1

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


def create_result_spool(out_file: Path | None) -> ResultSpool:
    spool_dir = out_file.parent if out_file is not None else None
    fd, name = tempfile.mkstemp(
        prefix="convomem_results_",
        suffix=".jsonl",
        dir=spool_dir,
    )
    os.close(fd)
    return ResultSpool(Path(name))


def unlink_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()


def iter_result_records(results_path: Path) -> Iterator[dict[str, Any]]:
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            yield json.loads(payload)


def indent_json_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def json_object_body(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    lines = serialized.splitlines()
    if len(lines) <= 2:
        return ""
    return "\n".join(lines[1:-1])


def write_streamed_payload(
    out_file: Path,
    payload: dict[str, Any],
    results_path: Path,
) -> tuple[float, float]:
    serialize_ms = 0.0
    write_ms = 0.0

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        write_start = perf_counter()
        handle.write("{\n")
        write_ms += (perf_counter() - write_start) * 1000.0

        serialize_start = perf_counter()
        body = json_object_body(payload)
        serialize_ms += (perf_counter() - serialize_start) * 1000.0
        if body:
            write_start = perf_counter()
            handle.write(body)
            handle.write(",\n")
            write_ms += (perf_counter() - write_start) * 1000.0

        write_start = perf_counter()
        handle.write('  "results": [')
        write_ms += (perf_counter() - write_start) * 1000.0

        first_record = True
        for record in iter_result_records(results_path):
            serialize_start = perf_counter()
            rendered_record = indent_json_block(
                json.dumps(record, ensure_ascii=False, indent=2),
                4,
            )
            serialize_ms += (perf_counter() - serialize_start) * 1000.0

            write_start = perf_counter()
            if first_record:
                handle.write("\n")
                first_record = False
            else:
                handle.write(",\n")
            handle.write(rendered_record)
            write_ms += (perf_counter() - write_start) * 1000.0

        write_start = perf_counter()
        if first_record:
            handle.write("]\n}\n")
        else:
            handle.write("\n  ]\n}\n")
        write_ms += (perf_counter() - write_start) * 1000.0

    return round(serialize_ms, 2), round(write_ms, 2)


def extract_conversations(
    case: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    conversations = case.get("conversations")
    if isinstance(conversations, list) and conversations:
        return [item for item in conversations if isinstance(item, dict)]
    if evidence_items:
        nested = evidence_items[0].get("conversations")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    return []


def build_corpus_from_case(
    case: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, set[str]], list[dict[str, Any]]]:
    conversations = extract_conversations(case, evidence_items)
    corpus_items: list[dict[str, Any]] = []
    signature_to_ids: dict[str, set[str]] = defaultdict(set)
    text_to_ids: dict[str, set[str]] = defaultdict(set)

    for conv_index, conversation in enumerate(conversations, start=1):
        conv_id = str(conversation.get("id") or f"conv_{conv_index:03d}")
        messages = conversation.get("messages") or []
        if not isinstance(messages, list):
            continue
        for message_index, raw_message in enumerate(messages, start=1):
            if not isinstance(raw_message, dict):
                continue
            text = message_body(raw_message)
            if not text:
                continue
            corpus_id = f"{conv_id}:m{message_index:04d}"
            corpus_item = {
                "corpus_id": corpus_id,
                "conversation_id": conv_id,
                "message_index": message_index,
                "timestamp": str(raw_message.get("timestamp") or raw_message.get("time") or ""),
                "text": format_message(raw_message),
            }
            corpus_items.append(corpus_item)

            signature = message_signature(raw_message)
            if signature:
                signature_to_ids[signature].add(corpus_id)
            text_to_ids[text.lower()].add(corpus_id)

    return corpus_items, signature_to_ids, text_to_ids, conversations


def derive_case_id(case: dict[str, Any], case_index: int) -> str:
    explicit = case.get("caseId") or case.get("case_id") or case.get("id")
    if explicit:
        return str(explicit)
    source_file = Path(str(case.get("_source_file") or f"case_{case_index:04d}.json")).stem
    source_index = safe_int(case.get("_source_case_index"), 1)
    return f"{source_file}_case{source_index:04d}"


def normalize_gold_message_ids(
    evidence_values: Any,
    signature_to_ids: dict[str, set[str]],
    text_to_ids: dict[str, set[str]],
) -> set[str]:
    if not isinstance(evidence_values, list):
        return set()
    gold_ids: set[str] = set()
    for value in evidence_values:
        if isinstance(value, dict):
            signature = message_signature(value)
            if signature and signature in signature_to_ids:
                gold_ids.update(signature_to_ids[signature])
                continue
            text = message_body(value).lower()
            if text and text in text_to_ids:
                gold_ids.update(text_to_ids[text])
            continue
        text = normalize_text(value).lower()
        if text and text in text_to_ids:
            gold_ids.update(text_to_ids[text])
    return gold_ids


def run_benchmark(
    data_path: Path,
    mode: str,
    top_k: int,
    limit: int,
    rerank_mode: str,
    shell: int,
    sector: str,
    zone: str,
    chunk_pool: int,
    out_file: Path | None,
    use_cross_encoder: bool = False,
    object_top_k: int = 4,
    support_top_k: int = 4,
    cognitive_top_k: int = 0,
) -> dict[str, Any]:
    load_data_ms = 0.0
    metrics_message = init_metric_bucket()
    timing_metrics: dict[str, list[float]] = defaultdict(list)
    profile_metrics: dict[str, list[float]] = defaultdict(list)
    per_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_context_size: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    vector_info: dict[str, Any] | None = None
    runtime_config: dict[str, Any] | None = None
    case_count = 0
    total_questions = 0
    start_time = datetime.now()
    totals_ms = {
        "total_ingest_ms": 0.0,
        "total_retrieval_ms": 0.0,
        "total_completion_ms": 0.0,
        "total_storage_ms": 0.0,
        "total_vector_ms": 0.0,
        "total_cache_reuse_saved_ms": 0.0,
        "total_embedding_cache_hit_ms_saved": 0.0,
    }
    workspace_manager = BenchmarkWorkspaceManager(WORKSPACE_CACHE_ROOT, SHARED_EMBEDDING_CACHE_DIR)
    result_spool = create_result_spool(out_file)

    cross_encoder_model = None
    try:
        try:
            if use_cross_encoder:
                try:
                    from sentence_transformers import CrossEncoder  # type: ignore

                    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    print("Cross-encoder reranking enabled")
                except Exception as exc:
                    print(f"Cross-encoder not available: {exc}")

            case_total_label = f"/{limit:03d}" if limit > 0 else ""
            for case_index, (case, case_load_ms) in enumerate(iter_cases(data_path, limit=limit), start=1):
                case_count = case_index
                load_data_ms += float(case_load_ms)
                evidence_items = list(case.get("evidenceItems") or case.get("evidence_items") or [])
                corpus_items, signature_to_ids, text_to_ids, conversations = build_corpus_from_case(
                    case,
                    evidence_items,
                )
                if not corpus_items or not evidence_items:
                    continue

                case_id = derive_case_id(case, case_index)
                context_size = safe_int(case.get("contextSize"), len(corpus_items))
                ordered_corpus_ids = [str(item["corpus_id"]) for item in corpus_items]
                corpus_by_node_id = {
                    f"bench_{item_index:04d}": item
                    for item_index, item in enumerate(corpus_items)
                }
                print(
                    f"[case {case_index:03d}{case_total_label}] {case_id}: "
                    f"context={context_size} docs={len(corpus_items)} qa={len(evidence_items)}"
                )

                current_workspace_manager = workspace_manager
                workspace = None
                pending_acquire: tuple[dict[str, Any], float] | None = None

                def acquire_case_workspace(manager: BenchmarkWorkspaceManager) -> tuple[Any, dict[str, Any], float]:
                    stage_start = perf_counter()
                    case_workspace, case_ingest_reuse = manager.acquire(
                        corpus_items=corpus_items,
                        granularity="message",
                        question_type="convomem",
                        shell=shell,
                        sector=sector,
                        zone=zone,
                        benchmark_name="convomem",
                        adapter_version="convomem_v1",
                    )
                    acquire_total_ms = round((perf_counter() - stage_start) * 1000.0, 2)
                    return case_workspace, case_ingest_reuse, acquire_total_ms

                for question_index, evidence_item in enumerate(evidence_items, start=1):
                    total_questions += 1
                    question = str(evidence_item.get("question") or "").strip()
                    answer = str(evidence_item.get("answer") or "")
                    category_name = normalize_text(
                        evidence_item.get("category") or evidence_item.get("evidence_type") or "unknown"
                    ).lower()
                    if not question:
                        continue
                    gold_message_ids = normalize_gold_message_ids(
                        evidence_item.get("message_evidences") or evidence_item.get("messageEvidences") or [],
                        signature_to_ids,
                        text_to_ids,
                    )
                    question_task_type = infer_task_type(question, category_name)
                    question_profile: dict[str, Any] = {}
                    retry_count = 0

                    while True:
                        question_start = perf_counter()
                        if workspace is None:
                            workspace, ingest_reuse, acquire_total_ms = acquire_case_workspace(
                                current_workspace_manager
                            )
                            pending_acquire = (ingest_reuse, acquire_total_ms)

                        if pending_acquire is not None:
                            ingest_reuse, acquire_total_ms = pending_acquire
                            pending_acquire = None
                        else:
                            ingest_reuse = {
                                "workspace_reused": True,
                                "workspace_reused_from_disk": False,
                                "cache_reuse_saved_ms": round(float(workspace.build_elapsed_ms), 2),
                                "ingest_lookup_ms": 0.0,
                            }
                            acquire_total_ms = 0.0

                        if ingest_reuse.get("workspace_reused"):
                            ingest_ms = round(float(ingest_reuse.get("ingest_lookup_ms", 0.0)), 2)
                            service_init_ms = round(max(0.0, acquire_total_ms - ingest_ms), 2)
                        else:
                            ingest_ms = round(float(workspace.build_elapsed_ms), 2)
                            service_init_ms = round(max(0.0, acquire_total_ms - ingest_ms), 2)

                        actual_ingest_profile = materialize_ingest_profile_for_question(
                            workspace.ingest_profile,
                            ingest_reuse,
                        )
                        if vector_info is None:
                            vector_info = workspace.vector_info
                        if runtime_config is None:
                            runtime_config = {
                                "embedding_model": workspace.config.embedding_model_name,
                                "embed_local_grain": bool(workspace.config.embed_local_grain),
                                "rerank_mode_default": workspace.config.rerank_mode_default,
                                "cross_encoder_model": workspace.config.cross_encoder_model_name,
                                "cross_encoder_requested": bool(use_cross_encoder),
                                "cross_encoder_loaded": bool(cross_encoder_model is not None),
                                "creative_mode": workspace.config.creative_mode_name,
                                "creative_beam_width": int(workspace.config.creative_beam_width),
                                "creative_max_hops": int(workspace.config.creative_max_hops),
                                "creative_neighbors_per_hop": int(workspace.config.creative_neighbors_per_hop),
                                "creative_max_output_paths": int(workspace.config.creative_max_output_paths),
                            }

                        workspace.pipeline.cross_encoder = cross_encoder_model
                        try:
                            if mode == "vector":
                                ranked_ids, ranked_items, stage_timing_ms = rank_vector(
                                    question,
                                    vector_store=workspace.vector_store,
                                    ordered_corpus_ids=ordered_corpus_ids,
                                    corpus_by_node_id=corpus_by_node_id,
                                    top_k=top_k,
                                    chunk_pool=chunk_pool,
                                )
                                question_profile = {
                                    "ingest": actual_ingest_profile,
                                    "ingest_cached": workspace.ingest_profile,
                                    "reuse": ingest_reuse,
                                }
                            elif mode == "evidence":
                                ranked_ids, ranked_items, stage_timing_ms, evidence_profile = rank_evidence(
                                    question,
                                    pipeline=workspace.pipeline,
                                    ordered_corpus_ids=ordered_corpus_ids,
                                    corpus_by_node_id=corpus_by_node_id,
                                    task_type=question_task_type,
                                    top_k=top_k,
                                    object_top_k=object_top_k,
                                    support_top_k=support_top_k,
                                    cognitive_top_k=cognitive_top_k,
                                    route_context={
                                        "benchmark": "convomem",
                                        "question_type": category_name,
                                        "task_name": category_name,
                                        "context_size": context_size,
                                        "conversation_count": len(conversations),
                                    },
                                )
                                question_profile = {
                                    "ingest": actual_ingest_profile,
                                    "ingest_cached": workspace.ingest_profile,
                                    "pipeline": evidence_profile,
                                    "reuse": ingest_reuse,
                                }
                            elif mode == "activation":
                                ranked_ids, ranked_items, stage_timing_ms = rank_hybrid(
                                    question,
                                    activation=workspace.activation,
                                    router=workspace.router,
                                    reranker=workspace.reranker,
                                    ordered_corpus_ids=ordered_corpus_ids,
                                    corpus_by_node_id=corpus_by_node_id,
                                    task_type=question_task_type,
                                    top_k=top_k,
                                    rerank_mode=rerank_mode,
                                    apply_rerank=False,
                                )
                                question_profile = {
                                    "ingest": actual_ingest_profile,
                                    "ingest_cached": workspace.ingest_profile,
                                    "reuse": ingest_reuse,
                                }
                            else:
                                ranked_ids, ranked_items, stage_timing_ms = rank_hybrid(
                                    question,
                                    activation=workspace.activation,
                                    router=workspace.router,
                                    reranker=workspace.reranker,
                                    ordered_corpus_ids=ordered_corpus_ids,
                                    corpus_by_node_id=corpus_by_node_id,
                                    task_type=question_task_type,
                                    top_k=top_k,
                                    rerank_mode=rerank_mode,
                                    apply_rerank=True,
                                )
                                question_profile = {
                                    "ingest": actual_ingest_profile,
                                    "ingest_cached": workspace.ingest_profile,
                                    "reuse": ingest_reuse,
                                }
                            if retry_count:
                                question_profile["retry_count"] = retry_count
                            break
                        except Exception as exc:
                            lowered = str(exc).lower()
                            retryable = (
                                "nothing found on disk" in lowered
                                or "creating hnsw segment reader" in lowered
                            )
                            if not retryable or retry_count >= 2:
                                raise
                            current_workspace_manager.release(workspace.signature, collect_garbage=False)
                            workspace = None
                            pending_acquire = None
                            sleep(0.25 * (retry_count + 1))
                            retry_count += 1
                            retry_cache_root = (
                                WORKSPACE_CACHE_ROOT
                                / "_question_retries"
                                / f"{case_id}_q{question_index:03d}_try{retry_count}"
                            )
                            current_workspace_manager = BenchmarkWorkspaceManager(
                                retry_cache_root,
                                SHARED_EMBEDDING_CACHE_DIR,
                            )
                            print(
                                f"[retry {retry_count}] rebuilding workspace for "
                                f"{case_id} q{question_index:03d} after vector store error"
                            )

                    question_total_ms = round((perf_counter() - question_start) * 1000.0, 2)
                    retrieval_ms = float(
                        stage_timing_ms.get(
                            "total_ms",
                            stage_timing_ms.get("vector_total_ms", stage_timing_ms.get("hybrid_total_ms", 0.0)),
                        )
                    )
                    completion_ms = float(stage_timing_ms.get("completion_total_ms", 0.0)) + float(
                        stage_timing_ms.get("cognitive_total_ms", 0.0)
                    )
                    question_summary = summarize_question_profile(
                        question_profile=question_profile,
                        retrieval_ms=retrieval_ms,
                        completion_ms=completion_ms,
                        question_total_ms=question_total_ms,
                        ingest_ms=ingest_ms,
                        ingest_reuse=ingest_reuse,
                    )
                    question_profile["summary"] = question_summary
                    stage_timing_ms = {
                        "service_init_ms": service_init_ms,
                        "ingest_ms": ingest_ms,
                        "cache_reuse_saved_ms": float(ingest_reuse.get("cache_reuse_saved_ms", 0.0)),
                        "retrieval_ms": retrieval_ms,
                        "completion_ms": completion_ms,
                        **stage_timing_ms,
                        "question_total_ms": question_total_ms,
                    }
                    ingest_profile = question_profile.get("ingest", {})
                    for key, value in ingest_profile.get("timings_ms", {}).items():
                        stage_timing_ms[f"ingest_{key}"] = float(value)
                    ingest_backend = ingest_profile.get("backend", {})
                    stage_timing_ms["ingest_storage_ms"] = float(
                        ingest_backend.get("storage", {}).get("total_ms", 0.0)
                    )
                    stage_timing_ms["ingest_vector_ms"] = float(
                        ingest_backend.get("vector", {}).get("total_ms", 0.0)
                    )
                    stage_timing_ms["embedding_cache_hit_ms_saved"] = float(
                        question_summary.get("embedding_cache_hit_ms_saved", 0.0)
                    )
                    for key, value in stage_timing_ms.items():
                        timing_metrics[key].append(float(value))
                    flatten_numeric_metrics(
                        "profile",
                        {key: value for key, value in question_profile.items() if key != "ingest_cached"},
                        profile_metrics,
                    )
                    totals_ms["total_ingest_ms"] += float(ingest_ms)
                    totals_ms["total_retrieval_ms"] += float(retrieval_ms)
                    totals_ms["total_completion_ms"] += float(completion_ms)
                    totals_ms["total_storage_ms"] += float(question_summary.get("total_storage_ms", 0.0))
                    totals_ms["total_vector_ms"] += float(question_summary.get("total_vector_ms", 0.0))
                    totals_ms["total_cache_reuse_saved_ms"] += float(
                        question_summary.get("cache_reuse_saved_ms", 0.0)
                    )
                    totals_ms["total_embedding_cache_hit_ms_saved"] += float(
                        question_summary.get("embedding_cache_hit_ms_saved", 0.0)
                    )

                    entry_metrics: dict[str, float] = {}
                    for k in KS:
                        message_frac = recall_fraction(ranked_ids, gold_message_ids, k)
                        message_ra, message_rl, message_nd = evaluate_retrieval(ranked_ids, gold_message_ids, k)
                        metrics_message[f"recall_frac@{k}"].append(message_frac)
                        metrics_message[f"recall_any@{k}"].append(message_ra)
                        metrics_message[f"recall_all@{k}"].append(message_rl)
                        metrics_message[f"ndcg_any@{k}"].append(message_nd)
                        entry_metrics[f"recall_frac@{k}"] = message_frac
                        entry_metrics[f"recall_any@{k}"] = message_ra
                        entry_metrics[f"ndcg_any@{k}"] = message_nd

                    per_category[category_name]["message_recall_frac@10"].append(entry_metrics["recall_frac@10"])
                    per_category[category_name]["message_ndcg_any@10"].append(entry_metrics["ndcg_any@10"])
                    per_context_size[str(context_size)]["message_recall_frac@10"].append(
                        entry_metrics["recall_frac@10"]
                    )
                    per_context_size[str(context_size)]["message_ndcg_any@10"].append(
                        entry_metrics["ndcg_any@10"]
                    )

                    result_spool.append(
                        {
                            "case_id": case_id,
                            "case_index": case_index,
                            "question_index": question_index,
                            "source_file": str(case.get("_source_file") or ""),
                            "context_size": context_size,
                            "conversation_count": len(conversations),
                            "category_name": category_name,
                            "task_type": question_task_type,
                            "question": question,
                            "answer": answer,
                            "gold_message_ids": sorted(gold_message_ids),
                            "metrics": entry_metrics,
                            "stage_timing_ms": stage_timing_ms,
                            "profiling": question_profile,
                            "retrieved_message_ids": ranked_ids[:top_k],
                            "ranked_items": ranked_items,
                        }
                    )

                    print(
                        f"  [q {question_index:03d}/{len(evidence_items):03d}] "
                        f"{category_name:18} "
                        f"R@10={entry_metrics['recall_frac@10']:.2f} "
                        f"NDCG@10={entry_metrics['ndcg_any@10']:.2f} "
                        f"Ingest={ingest_ms:.0f}ms ReuseSaved={float(ingest_reuse.get('cache_reuse_saved_ms', 0.0)):.0f}ms"
                    )
                if workspace is not None:
                    current_workspace_manager.release(workspace.signature, collect_garbage=False)
                if case_index % CASE_GC_INTERVAL == 0:
                    gc.collect()
        finally:
            workspace_manager.close_all()
            result_spool.close()

        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        message_summary = summarize_metric_bucket(metrics_message)
        per_category_summary = {
            category_name: {
                metric: (sum(values) / len(values) if values else 0.0)
                for metric, values in metric_map.items()
            }
            for category_name, metric_map in per_category.items()
        }
        per_context_size_summary = {
            context_size: {
                metric: (sum(values) / len(values) if values else 0.0)
                for metric, values in metric_map.items()
            }
            for context_size, metric_map in per_context_size.items()
        }
        timing_summary = {
            key: round(sum(values) / len(values), 2)
            for key, values in timing_metrics.items()
            if values
        }
        profile_summary = summarize_numeric_metrics(profile_metrics)
        question_total_sum = sum(timing_metrics.get("question_total_ms", []))
        bottlenecks: list[dict[str, Any]] = []
        for key, values in timing_metrics.items():
            if not values or key == "question_total_ms" or key.endswith("_saved_ms"):
                continue
            total_ms = float(sum(values))
            bottlenecks.append(
                {
                    "stage": key,
                    "avg_ms": round(total_ms / len(values), 2),
                    "total_ms": round(total_ms, 2),
                    "share_pct": round((total_ms / question_total_sum) * 100.0, 2) if question_total_sum else 0.0,
                    "classification": classify_bottleneck(key),
                }
            )
        bottlenecks.sort(key=lambda item: item["total_ms"], reverse=True)

        total_elapsed_ms = round(elapsed_seconds * 1000.0, 2)
        payload = {
            "data_path": str(data_path),
            "mode": mode,
            "rerank_mode": rerank_mode if mode == "hybrid" else None,
            "requested_rerank_mode": rerank_mode,
            "rerank_mode_active": mode == "hybrid",
            "case_count": case_count,
            "question_count": total_questions,
            "result_count": result_spool.count,
            "top_k": top_k,
            "shell": shell,
            "sector": sector,
            "zone": zone,
            "chunk_pool": chunk_pool,
            "object_top_k": object_top_k,
            "support_top_k": support_top_k,
            "cognitive_top_k": cognitive_top_k,
            "cross_encoder_requested": bool(use_cross_encoder),
            "cross_encoder_loaded": bool(cross_encoder_model is not None),
            "elapsed_seconds": elapsed_seconds,
            "benchmark_io_ms": {
                "load_data_ms": load_data_ms,
            },
            "totals_ms": {
                "total_elapsed_ms": total_elapsed_ms,
                "total_ingest_ms": round(totals_ms["total_ingest_ms"], 2),
                "total_retrieval_ms": round(totals_ms["total_retrieval_ms"], 2),
                "total_completion_ms": round(totals_ms["total_completion_ms"], 2),
                "total_storage_ms": round(totals_ms["total_storage_ms"], 2),
                "total_vector_ms": round(totals_ms["total_vector_ms"], 2),
                "total_io_ms": round(load_data_ms, 2),
            },
            "reuse_summary": {
                "total_cache_reuse_saved_ms": round(totals_ms["total_cache_reuse_saved_ms"], 2),
                "total_embedding_cache_hit_ms_saved": round(
                    totals_ms["total_embedding_cache_hit_ms_saved"],
                    2,
                ),
            },
            "vector_info": vector_info,
            "runtime_config": runtime_config or {},
            "metrics": {
                "message": message_summary,
            },
            "per_category": per_category_summary,
            "per_context_size": per_context_size_summary,
            "stage_timing_ms": timing_summary,
            "profiling_summary": profile_summary,
            "bottlenecks": bottlenecks[:12],
        }

        print("\nSummary")
        print(f"  Cases:           {payload['case_count']}")
        print(f"  Questions:       {payload['question_count']}")
        print(f"  Mode:            {mode}")
        print(f"  Message R@10:    {message_summary['recall_frac@10']:.4f}")
        print(f"  Message NDCG@10: {message_summary['ndcg_any@10']:.4f}")
        print(f"  Time:            {elapsed_seconds:.1f}s")
        if timing_summary:
            for key in sorted(timing_summary):
                print(f"  {key}: {timing_summary[key]:.2f}ms")
        if bottlenecks:
            print("  Top Bottlenecks:")
            for item in bottlenecks[:5]:
                print(
                    f"    - {item['stage']}: {item['avg_ms']:.2f}ms avg "
                    f"({item['share_pct']:.2f}%, {item['classification']})"
                )
        if vector_info:
            print(
                f"  Embedding:       {vector_info['embedding_model']} "
                f"({vector_info['embedding_provider']})"
            )

        if out_file is not None:
            payload["benchmark_io_ms"]["json_write_ms"] = 0.0
            serialize_ms, write_ms = write_streamed_payload(out_file, payload, result_spool.path)
            payload["benchmark_io_ms"]["json_serialize_ms"] = serialize_ms
            payload["benchmark_io_ms"]["json_write_ms"] = write_ms
            payload["totals_ms"]["total_io_ms"] = round(
                load_data_ms
                + float(payload["benchmark_io_ms"].get("json_serialize_ms", 0.0))
                + float(payload["benchmark_io_ms"].get("json_write_ms", 0.0)),
                2,
            )
            write_streamed_payload(out_file, payload, result_spool.path)
            print(f"  Saved:           {out_file}")
            return payload

        payload["results"] = list(iter_result_records(result_spool.path))
        return payload
    finally:
        unlink_if_exists(result_spool.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Sphere Memory CLI with ConvoMem.")
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        default=default_data_root(),
        help="Path to a ConvoMem directory or one pre_mixed_testcases JSON file.",
    )
    parser.add_argument("--mode", choices=["vector", "evidence", "activation", "hybrid"], default="evidence")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases")
    parser.add_argument("--rerank-mode", choices=["rule", "hybrid", "cross_encoder"], default="rule")
    parser.add_argument("--shell", type=int, default=2)
    parser.add_argument("--sector", default="knowledge")
    parser.add_argument("--zone", default="convomem")
    parser.add_argument("--chunk-pool", type=int, default=400)
    parser.add_argument("--cross-encoder", action="store_true", help="Enable cross-encoder reranking in evidence pipeline")
    parser.add_argument("--object-top-k", type=int, default=4, help="Number of structured evidence objects to keep for profiling/completion.")
    parser.add_argument("--support-top-k", type=int, default=4, help="Number of supporting context chunks to expand for profiling/completion.")
    parser.add_argument("--cognitive-top-k", type=int, default=0, help="Optional cognitive expansion budget for profiling.")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_file = args.out
    if out_file is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rerank_tag = f"_{args.rerank_mode}" if args.mode == "hybrid" else ""
        limit_tag = f"_limit{args.limit}" if args.limit else ""
        out_file = ROOT / "benchmarks" / f"results_convomem_{args.mode}{rerank_tag}_top{args.top_k}{limit_tag}_{stamp}.json"

    run_benchmark(
        data_path=args.data_path,
        mode=args.mode,
        top_k=args.top_k,
        limit=args.limit,
        rerank_mode=args.rerank_mode,
        shell=args.shell,
        sector=args.sector,
        zone=args.zone,
        chunk_pool=args.chunk_pool,
        out_file=out_file,
        use_cross_encoder=args.cross_encoder,
        object_top_k=args.object_top_k,
        support_top_k=args.support_top_k,
        cognitive_top_k=args.cognitive_top_k,
    )


if __name__ == "__main__":
    main()
