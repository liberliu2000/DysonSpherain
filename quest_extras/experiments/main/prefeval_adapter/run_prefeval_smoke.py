from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quest-local PrefEval smoke pass against the current DysonSpherain runtime."
    )
    parser.add_argument(
        "--dyson-root",
        type=Path,
        default=Path("/home/liber/Projects/DysonSpherain/sphere_memory_cli_next_main_code_20260417_164120"),
        help="External DysonSpherain code root to import the runtime from.",
    )
    parser.add_argument(
        "--prefeval-root",
        type=Path,
        default=Path("tmp/prefeval_source"),
        help="Quest-local PrefEval source snapshot root.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Explicit JSON file or directory to read PrefEval cases from. Defaults to the explicit topic file.",
    )
    parser.add_argument(
        "--topic",
        default="travel_restaurant",
        help="PrefEval topic file stem under benchmark_dataset/explicit_preference.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/experiment/prefeval-wave1-smoke"),
        help="Durable output root for predictions, traces, and summaries.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=5,
        help="Maximum number of explicit cases to run in this smoke pass.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based case start index inside the selected explicit dataset.",
    )
    parser.add_argument(
        "--inter-turns",
        type=int,
        default=0,
        help="Number of distractor conversation turns to inject from filtered_inter_turns.json.",
    )
    parser.add_argument(
        "--task-type",
        default="qa",
        help="DysonSpherain runtime task type for the query.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="DysonSpherain runtime temperature for the assembled bundle.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="DysonSpherain runtime max token budget for assembled context.",
    )
    parser.add_argument(
        "--evidence-top-k",
        type=int,
        default=8,
        help="DysonSpherain evidence_top_k for the query.",
    )
    parser.add_argument(
        "--support-top-k",
        type=int,
        default=4,
        help="DysonSpherain support_top_k for the query.",
    )
    parser.add_argument(
        "--object-top-k",
        type=int,
        default=4,
        help="DysonSpherain object_top_k for the query.",
    )
    parser.add_argument(
        "--cognitive-top-k",
        type=int,
        default=2,
        help="DysonSpherain cognitive_top_k for the query.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output root before running.",
    )
    parser.add_argument(
        "--preference-patch",
        choices=(
            "none",
            "dietary_v1",
            "dietary_v2",
            "route_v1",
            "dietary_v2_route_v1",
            "dietary_v3",
            "dietary_v3_route_v1",
        ),
        default="none",
        help="Optional quest-local runtime patch for dietary extraction gaps and/or preference-aware dining recommendation routing.",
    )
    return parser.parse_args()


def ensure_repo_import(repo_root: Path) -> None:
    resolved = repo_root.resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def load_explicit_cases(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    data_root = args.data_root
    if data_root is None:
        data_root = (
            args.prefeval_root
            / "benchmark_dataset"
            / "explicit_preference"
            / f"{args.topic}.json"
        )
    data_root = data_root.resolve()
    if data_root.is_dir():
        data_path = data_root / f"{args.topic}.json"
    else:
        data_path = data_root
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of explicit PrefEval cases at {data_path}")
    return data_path, payload


def load_inter_turn_pool(prefeval_root: Path) -> list[dict[str, Any]]:
    pool_path = prefeval_root / "benchmark_dataset" / "filtered_inter_turns.json"
    if not pool_path.exists():
        return []
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def infer_preference_polarity(preference_text: str) -> float:
    lowered = preference_text.lower()
    negative_cues = (
        "avoid",
        "avoids",
        "cannot",
        "can't",
        "won't",
        "will not",
        "dont",
        "don't",
        "do not",
        "dislike",
        "dislikes",
        "hate",
        "hates",
        "not a fan",
        "aversion",
        "averse",
        "uncomfortable",
        "allergy",
        "allergic",
        "intolerance",
        "strictly avoid",
        "not consume",
        "cannot consume",
        "refuse",
    )
    positive_cues = (
        "prefer",
        "prefers",
        "favorite",
        "favourite",
        "like",
        "likes",
        "love",
        "loves",
        "enjoy",
        "enjoys",
        "wants",
    )
    if any(cue in lowered for cue in negative_cues):
        return -1.0
    if any(cue in lowered for cue in positive_cues):
        return 1.0
    return 1.0


DINING_QUERY_TERMS = (
    "food",
    "foods",
    "dish",
    "dishes",
    "restaurant",
    "restaurants",
    "dining",
    "eat",
    "eating",
    "meal",
    "meals",
    "cuisine",
    "cuisines",
    "snack",
    "snacks",
    "spot",
    "spots",
)

DINING_RECOMMENDATION_CUES = (
    "must-try",
    "recommend",
    "recommended",
    "suggest",
    "what are some",
    "where should i eat",
    "where to eat",
    "dining options",
    "dining spots",
    "dining experience",
    "food experiences",
    "top-rated",
    "popular dining",
    "trendy dining",
    "should i try",
)

DINING_TEMPORAL_LOOKUP_CUES = (
    "when did",
    "when was",
    "what date",
    "what day",
    "which day",
    "how long",
    "timeline",
    "ago",
    "before ",
    "after ",
)


def looks_like_dining_recommendation_query(query: str) -> bool:
    lowered = query.lower()
    if not any(term in lowered for term in DINING_QUERY_TERMS):
        return False
    if any(cue in lowered for cue in DINING_TEMPORAL_LOOKUP_CUES):
        return False
    return any(cue in lowered for cue in DINING_RECOMMENDATION_CUES)


def apply_query_parser_route_patch(patch_name: str) -> dict[str, Any]:
    if patch_name not in {"route_v1", "dietary_v2_route_v1", "dietary_v3_route_v1"}:
        return {
            "name": patch_name,
            "applied": False,
            "heuristic": None,
        }

    from sphere_cli.evidence_pipeline import QueryParser  # type: ignore

    marker = "_prefeval_route_patch_name"
    if getattr(QueryParser, marker, None) == patch_name:
        return {
            "name": patch_name,
            "applied": False,
            "heuristic": "dining_recommendation_implies_preference",
            "already_present": True,
        }

    original_parse = getattr(QueryParser, "_prefeval_original_parse", QueryParser.parse)
    setattr(QueryParser, "_prefeval_original_parse", original_parse)

    def patched_parse(self: Any, query: str, task_type: str = "qa") -> Any:
        profile = original_parse(self, query, task_type)
        if profile.needs_preference_objects or not looks_like_dining_recommendation_query(query):
            return profile
        profile.needs_preference_objects = True
        profile.needs_personal_context_objects = True
        profile.preferred_object_types = list(
            dict.fromkeys([*profile.preferred_object_types, "preference", "personal_context"])
        )
        profile.granularity_bias = list(dict.fromkeys(["micro", "local", *profile.granularity_bias]))
        profile.lexical_priority = max(float(profile.lexical_priority), 0.75)
        profile.semantic_priority = max(float(profile.semantic_priority), 0.9)
        return profile

    QueryParser.parse = patched_parse
    setattr(QueryParser, marker, patch_name)
    return {
        "name": patch_name,
        "applied": True,
        "heuristic": "dining_recommendation_implies_preference",
    }


def short_preference_summary(preference_text: str) -> str:
    normalized = " ".join(preference_text.split())
    if len(normalized) <= 90:
        return normalized
    return normalized[:87] + "..."


def apply_memory_writer_preference_patch(patch_name: str) -> dict[str, Any]:
    if patch_name == "none":
        return {
            "name": patch_name,
            "applied": False,
            "added_preference_patterns": 0,
            "added_habit_patterns": 0,
            "route_query_parser_patch": False,
        }

    if patch_name not in {
        "dietary_v1",
        "dietary_v2",
        "route_v1",
        "dietary_v2_route_v1",
        "dietary_v3",
        "dietary_v3_route_v1",
    }:
        raise ValueError(f"Unsupported preference patch: {patch_name}")

    base_patch_name = {
        "dietary_v1": "dietary_v1",
        "dietary_v2": "dietary_v2",
        "dietary_v2_route_v1": "dietary_v2",
        "dietary_v3": "dietary_v3",
        "dietary_v3_route_v1": "dietary_v3",
    }.get(patch_name)

    memory_patch_report = {
        "name": base_patch_name or "none",
        "applied": False,
        "added_preference_patterns": 0,
        "added_habit_patterns": 0,
    }

    if base_patch_name is not None:
        from sphere_cli.memory_writer import MemoryWriter  # type: ignore

        marker = "_prefeval_preference_patch_name"
        if getattr(MemoryWriter, marker, None) == base_patch_name:
            memory_patch_report["already_present"] = True
        else:
            dietary_preference_patterns = [
                (
                    re.compile(
                        r"\b(?:must avoid|need to avoid|cannot consume|can't consume|can not consume|cannot eat|can't eat|can not eat|must not consume|must not eat|refuse to consume|refuse to eat)\s+(?P<object>[^.;,\n]+)",
                        re.IGNORECASE,
                    ),
                    -1.0,
                ),
                (
                    re.compile(
                        r"\b(?:allergic to|have(?: a)?(?: severe)? allergy to|intolerant to|have(?: a)?(?: severe)? intolerance to)\s+(?P<object>[^.;,\n]+)",
                        re.IGNORECASE,
                    ),
                    -1.0,
                ),
            ]
            dietary_habit_patterns = [
                (
                    re.compile(
                        r"\b(?:follow|am on|i(?:'m| am) on|keep to|maintain)\s+(?:a|an)?\s*(?P<object>(?:strict|mostly|plant-based|vegan|vegetarian|gluten-free|dairy-free|halal|kosher|low-sodium|low sodium)[^.;,\n]*\sdiet)\b",
                        re.IGNORECASE,
                    ),
                    0.78,
                ),
            ]
            if base_patch_name in {"dietary_v2", "dietary_v3"}:
                dietary_preference_patterns.extend(
                    [
                        (
                            re.compile(
                                r"\b(?:will not consume|won't consume|will not eat|won't eat)\s+(?P<object>[^.;,\n]+)",
                                re.IGNORECASE,
                            ),
                            -1.0,
                        ),
                        (
                            re.compile(
                                r"\b(?:have|with|suffer from)\s+(?:a|an)?(?:severe|strong|intense)?\s*(?P<object>[^.;,\n]+?)\s+(?:allergy|allergies|intolerance)\b",
                                re.IGNORECASE,
                            ),
                            -1.0,
                        ),
                        (
                            re.compile(
                                r"\b(?:aversion to|averse to)\s+(?P<object>[^.;,\n]+)",
                                re.IGNORECASE,
                            ),
                            -1.0,
                        ),
                    ]
                )
                dietary_habit_patterns.extend(
                    [
                        (
                            re.compile(
                                r"\b(?:adhere to|stick to)\s+(?:a|an)?\s*(?P<object>(?:strict|mostly|plant-based|vegan|vegetarian|gluten-free|dairy-free|halal|kosher|low-sodium|low sodium)[^.;,\n]*\sdiet)\b",
                                re.IGNORECASE,
                            ),
                            0.78,
                        ),
                    ]
                )
            if base_patch_name == "dietary_v3":
                dietary_preference_patterns.extend(
                    [
                        (
                            re.compile(
                                r"\b(?:do not consume|don't consume)\s+(?P<object>[^.;,\n]+)",
                                re.IGNORECASE,
                            ),
                            -1.0,
                        ),
                        (
                            re.compile(
                                r"\b(?:only eat at|only dine at)\s+(?P<object>[^.;,\n]+)",
                                re.IGNORECASE,
                            ),
                            0.78,
                        ),
                        (
                            re.compile(
                                r"\b(?:get|feel|am)\s+(?:very\s+)?uncomfortable\s+(?:dining|eating)\s+(?:in|at)\s+(?P<object>[^.;,\n]+)",
                                re.IGNORECASE,
                            ),
                            -1.0,
                        ),
                    ]
                )

            MemoryWriter.PREFERENCE_PATTERNS = (
                list(MemoryWriter.PREFERENCE_PATTERNS) + dietary_preference_patterns
            )
            MemoryWriter.HABIT_PREFERENCE_PATTERNS = (
                list(MemoryWriter.HABIT_PREFERENCE_PATTERNS) + dietary_habit_patterns
            )
            setattr(MemoryWriter, marker, base_patch_name)
            memory_patch_report["applied"] = True
            memory_patch_report["added_preference_patterns"] = len(dietary_preference_patterns)
            memory_patch_report["added_habit_patterns"] = len(dietary_habit_patterns)

    route_patch_report = apply_query_parser_route_patch(patch_name)
    route_query_parser_patch = bool(
        route_patch_report.get("applied") or route_patch_report.get("already_present")
    )
    return {
        "name": patch_name,
        "applied": bool(memory_patch_report.get("applied")) or bool(route_patch_report.get("applied")),
        "added_preference_patterns": int(memory_patch_report.get("added_preference_patterns", 0)),
        "added_habit_patterns": int(memory_patch_report.get("added_habit_patterns", 0)),
        "route_query_parser_patch": route_query_parser_patch,
        "components": {
            "memory_writer": memory_patch_report,
            "query_parser": route_patch_report,
        },
    }


def build_answer_prompt(
    question: str,
    preference: str,
    completion: Any,
    bundle: Any,
) -> dict[str, Any]:
    preference_objects = [
        {
            "object_type": item.get("object_type"),
            "object_text": item.get("object_text"),
            "polarity": item.get("polarity"),
            "source_unit_text": item.get("source_unit_text"),
        }
        for item in completion.evidence_objects
        if item.get("object_type") == "preference"
    ]
    return {
        "instruction": (
            "Answer the user question while following the remembered user preference. "
            "Ground the answer in the provided evidence bundle and do not ignore the stored preference."
        ),
        "question": question,
        "remembered_preference": preference,
        "preference_objects": preference_objects,
        "primary_evidence": [
            {
                "text": item.get("text"),
                "summary": item.get("summary"),
                "score": item.get("score"),
            }
            for item in bundle.primary_evidence[:4]
        ],
        "supporting_context": [
            {
                "text": item.get("text"),
                "summary": item.get("summary"),
                "score": item.get("score"),
            }
            for item in bundle.supporting_context[:3]
        ],
        "raw_reference_pointers": list(bundle.raw_reference_pointers),
        "caveat": (
            "DysonSpherain currently assembles grounded context but does not yet ship a built-in final answer "
            "generator, so this prompt is a handoff artifact for a later generation layer rather than a model answer."
        ),
    }


def sign_matches(polarity: float | None, expected: float) -> bool:
    if polarity is None:
        return False
    if expected >= 0:
        return float(polarity) >= 0
    return float(polarity) < 0


def add_distractor_turns(
    runtime: Any,
    inter_turn_pool: list[dict[str, Any]],
    case_index: int,
    limit: int,
    zone_name: str,
) -> int:
    if limit <= 0 or not inter_turn_pool:
        return 0

    from sphere_cli.models import MemoryNode  # type: ignore

    injected = 0
    source = inter_turn_pool[case_index % len(inter_turn_pool)]
    conversation = source.get("conversation") or []
    for turn_index, message in enumerate(conversation[: limit * 2], start=1):
        role = str(message.get("role") or "speaker")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        node = MemoryNode(
            shell=4,
            sector="raw",
            zone=zone_name,
            cell=f"distractor_{turn_index:02d}",
            molecular_type="conversation_turn",
            summary=f"{role}: {short_preference_summary(content)}",
            raw_content=f"{role}: {content}",
            importance=0.2,
            stability_score=0.2,
            stage="staging",
            tags="prefeval,distractor",
        )
        runtime.writeback_memory(node=node, source_kind="conversation_turn")
        injected += 1
    return injected


def case_workspace_root(output_root: Path, case_id: str) -> Path:
    return output_root / "workspaces" / case_id


def case_trace_root(output_root: Path, case_id: str) -> Path:
    return output_root / "traces" / case_id


def prepare_output_root(output_root: Path, clean: bool) -> None:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "traces").mkdir(parents=True, exist_ok=True)
    (output_root / "workspaces").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def run_case(
    runtime_cls: Any,
    memory_node_cls: Any,
    args: argparse.Namespace,
    output_root: Path,
    inter_turn_pool: list[dict[str, Any]],
    topic: str,
    global_case_index: int,
    case_payload: dict[str, Any],
) -> dict[str, Any]:
    case_id = f"case_{global_case_index + 1:04d}"
    workspace_root = case_workspace_root(output_root, case_id)
    trace_root = case_trace_root(output_root, case_id)
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    if trace_root.exists():
        shutil.rmtree(trace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)

    runtime = runtime_cls.from_base_dir(workspace_root)
    preference = str(case_payload["preference"])
    question = str(case_payload["question"])
    expected_polarity = infer_preference_polarity(preference)

    preference_node = memory_node_cls(
        shell=2,
        sector="persona",
        zone=f"prefeval_{topic}",
        cell=case_id,
        molecular_type="preference",
        summary=short_preference_summary(preference),
        raw_content=preference,
        importance=0.8,
        stability_score=0.8,
        stage="long_term",
        tags="prefeval,explicit,preference",
    )
    writeback_report = runtime.writeback_memory(node=preference_node, source_kind="preference")
    distractor_count = add_distractor_turns(
        runtime=runtime,
        inter_turn_pool=inter_turn_pool,
        case_index=global_case_index,
        limit=args.inter_turns,
        zone_name=f"prefeval_{topic}_distractors",
    )

    started = time.perf_counter()
    result = runtime.run_query(
        query=question,
        task_type=args.task_type,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        evidence_top_k=args.evidence_top_k,
        support_top_k=args.support_top_k,
        object_top_k=args.object_top_k,
        cognitive_top_k=args.cognitive_top_k,
    )
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

    evidence = result["evidence"]
    completion = result["completion"]
    bundle = result["bundle"]

    preference_objects = [
        obj for obj in completion.evidence_objects if str(obj.get("object_type") or "") == "preference"
    ]
    route_type = str(evidence.query_route.get("route_type") or "")
    route_ok = route_type == "persona_preference_state"
    object_recalled = bool(preference_objects)
    polarity_match = any(sign_matches(obj.get("polarity"), expected_polarity) for obj in preference_objects)
    core_texts = [str(item.get("text") or "") for item in completion.core_evidence]
    object_source_texts = [str(item.get("source_unit_text") or "") for item in preference_objects]
    normalized_pref = normalize_text(preference)
    preference_text_recalled = any(
        normalize_text(candidate) == normalized_pref or normalized_pref in normalize_text(candidate)
        for candidate in core_texts + object_source_texts
    )
    reactivation_success = route_ok and object_recalled and polarity_match

    failure_reasons: list[str] = []
    if not route_ok:
        failure_reasons.append("route_not_persona_preference_state")
    if not object_recalled:
        failure_reasons.append("preference_object_missing")
    if object_recalled and not polarity_match:
        failure_reasons.append("preference_polarity_mismatch")
    if not preference_text_recalled:
        failure_reasons.append("preference_text_not_grounded")

    prediction = {
        "case_id": case_id,
        "topic": topic,
        "question": question,
        "preference": preference,
        "reactivation_success": reactivation_success,
        "route_type": route_type,
        "route_persona_preference_state": route_ok,
        "preference_object_recalled": object_recalled,
        "preference_polarity_match": polarity_match,
        "preference_text_recalled": preference_text_recalled,
        "failure_reasons": failure_reasons,
        "latency_ms": latency_ms,
        "estimated_input_tokens": int(bundle.debug.get("estimated_input_tokens", 0) or 0),
        "context_token_delta": int(bundle.debug.get("context_token_delta", 0) or 0),
        "distractor_turns_injected": distractor_count,
        "writeback_report": writeback_report,
        "profile": {
            "task_type": evidence.profile.task_type,
            "needs_preference_objects": evidence.profile.needs_preference_objects,
            "needs_personal_context_objects": evidence.profile.needs_personal_context_objects,
            "preferred_object_types": list(evidence.profile.preferred_object_types),
        },
        "answer_prompt": build_answer_prompt(question=question, preference=preference, completion=completion, bundle=bundle),
    }

    trace_payload = {
        "case_id": case_id,
        "source_case_index": global_case_index,
        "preference": preference,
        "question": question,
        "explanation": case_payload.get("explanation"),
        "prediction": prediction,
        "route": evidence.query_route,
        "profile": {
            "task_type": evidence.profile.task_type,
            "needs_exact_evidence": evidence.profile.needs_exact_evidence,
            "needs_multi_hop_evidence": evidence.profile.needs_multi_hop_evidence,
            "needs_preference_objects": evidence.profile.needs_preference_objects,
            "needs_temporal_objects": evidence.profile.needs_temporal_objects,
            "needs_personal_context_objects": evidence.profile.needs_personal_context_objects,
            "needs_relation_objects": evidence.profile.needs_relation_objects,
            "preferred_object_types": list(evidence.profile.preferred_object_types),
            "preference_polarity_hint": evidence.profile.preference_polarity_hint,
        },
        "core_evidence": completion.core_evidence,
        "evidence_objects": completion.evidence_objects,
        "supporting_context": completion.supporting_context,
        "relevant_experience": result["cognitive"].relevant_experience,
        "creative_reflections": result["cognitive"].creative_reflections,
        "alternative_paths": result["cognitive"].alternative_paths,
        "bundle": {
            "primary_evidence": bundle.primary_evidence,
            "supporting_context": bundle.supporting_context,
            "evidence_objects": bundle.evidence_objects,
            "creative_reflections": bundle.creative_reflections,
            "alternative_paths": bundle.alternative_paths,
            "raw_reference_pointers": bundle.raw_reference_pointers,
            "debug": bundle.debug,
        },
        "timings_ms": {
            "total_ms": latency_ms,
            "retrieval_ms": round(float(evidence.timings_ms.get("total_ms", 0.0)), 2),
            "completion_ms": round(float(completion.timings_ms.get("total_ms", 0.0)), 2),
            "cognitive_ms": round(float(result["cognitive"].timings_ms.get("total_ms", 0.0)), 2),
            "assemble_ms": round(float(bundle.debug.get("assemble_ms", 0.0) or 0.0), 2),
        },
    }

    write_json(trace_root / "trace.json", trace_payload)
    return prediction


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    failure_counts = Counter(
        reason for prediction in predictions for reason in prediction.get("failure_reasons", [])
    )
    latencies = [float(prediction["latency_ms"]) for prediction in predictions]
    estimated_tokens = [int(prediction["estimated_input_tokens"]) for prediction in predictions]
    token_delta = [int(prediction["context_token_delta"]) for prediction in predictions]

    def mean_bool(key: str) -> float:
        return round(
            sum(1 for prediction in predictions if prediction.get(key)) / max(1, total),
            4,
        )

    return {
        "total_cases": total,
        "reactivation_success_rate": mean_bool("reactivation_success"),
        "route_persona_preference_state_rate": mean_bool("route_persona_preference_state"),
        "preference_object_recall_rate": mean_bool("preference_object_recalled"),
        "preference_polarity_match_rate": mean_bool("preference_polarity_match"),
        "preference_text_recall_rate": mean_bool("preference_text_recalled"),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "median": round(statistics.median(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "context_tokens": {
            "estimated_input_tokens_mean": round(statistics.fmean(estimated_tokens), 2)
            if estimated_tokens
            else 0.0,
            "context_token_delta_mean": round(statistics.fmean(token_delta), 2)
            if token_delta
            else 0.0,
        },
        "failure_breakdown": dict(sorted(failure_counts.items())),
    }


def render_summary_md(
    args: argparse.Namespace,
    data_path: Path,
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> str:
    failing = [p for p in predictions if not p.get("reactivation_success")]
    lines = [
        "# PrefEval Smoke Summary",
        "",
        "## Scope",
        "",
        f"- topic: `{args.topic}`",
        f"- data_path: `{data_path}`",
        f"- max_cases: `{args.max_cases}`",
        f"- inter_turns: `{args.inter_turns}`",
        f"- preference_patch: `{args.preference_patch}`",
        f"- dyson_root: `{args.dyson_root}`",
        f"- interpreter: `{sys.executable}`",
        "",
        "## Headline",
        "",
        f"- preference reactivation success rate: `{metrics['reactivation_success_rate']:.4f}`",
        f"- persona/preference route hit rate: `{metrics['route_persona_preference_state_rate']:.4f}`",
        f"- preference object recall rate: `{metrics['preference_object_recall_rate']:.4f}`",
        f"- preference polarity match rate: `{metrics['preference_polarity_match_rate']:.4f}`",
        f"- mean latency: `{metrics['latency_ms']['mean']:.2f} ms`",
        f"- mean context token delta: `{metrics['context_tokens']['context_token_delta_mean']:.2f}`",
        "",
        "## Caveat",
        "",
        "- This smoke evaluates preference reactivation readiness inside the current DysonSpherain memory runtime.",
        "- It does not claim official PrefEval final-answer accuracy because the current runtime does not yet ship a built-in answer generator.",
        "- Each case trace includes an `answer_prompt` handoff artifact so a later generation layer can be added without rerunning ingestion logic.",
        "",
        "## Failure Breakdown",
        "",
    ]
    if metrics["failure_breakdown"]:
        for key, value in metrics["failure_breakdown"].items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- none")

    lines.extend(["", "## First Failed Cases", ""])
    if failing:
        for item in failing[:3]:
            lines.append(
                f"- `{item['case_id']}`: route=`{item['route_type']}`, reasons={item['failure_reasons']}"
            )
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ensure_repo_import(args.dyson_root)
    patch_report = apply_memory_writer_preference_patch(args.preference_patch)
    from sphere_cli.models import MemoryNode  # type: ignore
    from sphere_cli.runtime import UnifiedMemoryRuntime  # type: ignore

    data_path, all_cases = load_explicit_cases(args)
    selected_cases = all_cases[args.start_index : args.start_index + max(0, args.max_cases)]
    if not selected_cases:
        raise ValueError("No PrefEval cases selected for the smoke run.")

    prepare_output_root(args.output_root, clean=args.clean_output)
    inter_turn_pool = load_inter_turn_pool(args.prefeval_root)

    started_at = now_iso()
    predictions: list[dict[str, Any]] = []
    for local_index, case_payload in enumerate(selected_cases):
        global_case_index = args.start_index + local_index
        predictions.append(
            run_case(
                runtime_cls=UnifiedMemoryRuntime,
                memory_node_cls=MemoryNode,
                args=args,
                output_root=args.output_root,
                inter_turn_pool=inter_turn_pool,
                topic=args.topic,
                global_case_index=global_case_index,
                case_payload=case_payload,
            )
        )

    metrics = summarize_predictions(predictions)
    finished_at = now_iso()
    manifest = {
        "run_name": "prefeval-wave1-smoke",
        "started_at": started_at,
        "finished_at": finished_at,
        "script_path": str(Path(__file__).resolve()),
        "dyson_root": str(args.dyson_root.resolve()),
        "prefeval_root": str(args.prefeval_root.resolve()),
        "data_path": str(data_path),
        "topic": args.topic,
        "max_cases": len(predictions),
        "start_index": args.start_index,
        "inter_turns": args.inter_turns,
        "preference_patch": args.preference_patch,
        "preference_patch_report": patch_report,
        "task_type": args.task_type,
        "interpreter": sys.executable,
        "command": " ".join(sys.argv),
        "scope_note": (
            "This smoke validates preference reactivation inside the current memory runtime. "
            "It intentionally stops short of official PrefEval final-answer scoring."
        ),
    }

    write_json(args.output_root / "run_manifest.json", manifest)
    write_json(args.output_root / "preference_patch_report.json", patch_report)
    write_json(args.output_root / "metrics.json", metrics)
    write_json(
        args.output_root / "summary.json",
        {
            "manifest": manifest,
            "metrics": metrics,
            "prediction_count": len(predictions),
            "predictions_path": str((args.output_root / "predictions.jsonl").resolve()),
        },
    )
    (args.output_root / "predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )
    (args.output_root / "summary.md").write_text(
        render_summary_md(args=args, data_path=data_path, predictions=predictions, metrics=metrics),
        encoding="utf-8",
    )
    write_json(
        args.output_root / "artifact_manifest.json",
        {
            "files": [
                "artifact_manifest.json",
                "metrics.json",
                "predictions.jsonl",
                "run_manifest.json",
                "summary.json",
                "summary.md",
                "traces/",
                "workspaces/",
            ],
            "note": (
                "Each case trace lives under traces/<case_id>/trace.json and each isolated runtime workspace "
                "lives under workspaces/<case_id>/."
            ),
        },
    )

    print(json.dumps({"manifest": manifest, "metrics": metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
