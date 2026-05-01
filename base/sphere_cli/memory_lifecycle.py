from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment_registry import BenchmarkRun, RUN_TYPE_ORDER, load_registry
from .utils import stable_content_hash


@dataclass
class MemoryConflict:
    conflict_id: str
    project: str
    dataset: str
    conflict_type: str
    run_ids: list[str]
    reason: str
    recommended_winner: str | None = None
    status: str = "open"


def lifecycle_dir(base_dir: Path) -> Path:
    return base_dir / "artifacts" / "memory_lifecycle"


def conflict_path(base_dir: Path) -> Path:
    return lifecycle_dir(base_dir) / "conflicts.json"


def lifecycle_actions_path(base_dir: Path) -> Path:
    return lifecycle_dir(base_dir) / "lifecycle_actions.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conflict_id(project: str, dataset: str, conflict_type: str, run_ids: list[str]) -> str:
    return stable_content_hash(json.dumps([project, dataset, conflict_type, sorted(run_ids)], ensure_ascii=False))[:16]


def _winner(a: BenchmarkRun, b: BenchmarkRun) -> BenchmarkRun:
    a_fallback = bool(a.fallback_in_use)
    b_fallback = bool(b.fallback_in_use)
    if a_fallback != b_fallback:
        return b if a_fallback else a
    a_order = RUN_TYPE_ORDER.get(a.run_type, 0)
    b_order = RUN_TYPE_ORDER.get(b.run_type, 0)
    if a_order != b_order:
        return a if a_order > b_order else b
    a_questions = int(a.question_count or 0)
    b_questions = int(b.question_count or 0)
    if a_questions != b_questions:
        return a if a_questions > b_questions else b
    return a if a.timestamp >= b.timestamp else b


def _primary_metric_delta(a: BenchmarkRun, b: BenchmarkRun) -> tuple[str, float] | None:
    for key in ("recall_frac@10", "recall_any@10", "ndcg_any@10", "candidate_recall@100", "final_recall@10"):
        if key not in a.metrics or key not in b.metrics:
            continue
        try:
            return key, abs(float(a.metrics[key]) - float(b.metrics[key]))
        except (TypeError, ValueError):
            continue
    return None


def detect_conflicts(base_dir: Path, project: str) -> list[MemoryConflict]:
    runs = [run for run in load_registry(base_dir) if run.project.lower() == project.lower()]
    by_dataset: dict[str, list[BenchmarkRun]] = {}
    for run in runs:
        by_dataset.setdefault(run.dataset.lower(), []).append(run)
    conflicts: list[MemoryConflict] = []
    for dataset, dataset_runs in by_dataset.items():
        for index, a in enumerate(dataset_runs):
            for b in dataset_runs[index + 1 :]:
                run_ids = [a.run_id, b.run_id]
                winner = _winner(a, b)
                if bool(a.fallback_in_use) != bool(b.fallback_in_use):
                    conflicts.append(
                        MemoryConflict(
                            conflict_id=_conflict_id(project, dataset, "fallback_conflict", run_ids),
                            project=project,
                            dataset=dataset,
                            conflict_type="fallback_conflict",
                            run_ids=run_ids,
                            reason="fallback and non-fallback runs both exist for the same dataset",
                            recommended_winner=winner.run_id,
                        )
                    )
                if {a.run_type, b.run_type} & {"smoke"} and {a.run_type, b.run_type} & {"full"}:
                    conflicts.append(
                        MemoryConflict(
                            conflict_id=_conflict_id(project, dataset, "smoke_full_conflict", run_ids),
                            project=project,
                            dataset=dataset,
                            conflict_type="smoke_full_conflict",
                            run_ids=run_ids,
                            reason="smoke and full runs both exist; full should drive formal state",
                            recommended_winner=winner.run_id,
                        )
                    )
                if {a.run_type, b.run_type} & {"partial"} and {a.run_type, b.run_type} & {"full"}:
                    conflicts.append(
                        MemoryConflict(
                            conflict_id=_conflict_id(project, dataset, "partial_full_conflict", run_ids),
                            project=project,
                            dataset=dataset,
                            conflict_type="partial_full_conflict",
                            run_ids=run_ids,
                            reason="partial and full runs both exist; partial cannot define formal state",
                            recommended_winner=winner.run_id,
                        )
                    )
                metric_delta = _primary_metric_delta(a, b)
                if metric_delta and a.run_type == b.run_type and bool(a.fallback_in_use) == bool(b.fallback_in_use):
                    metric, delta = metric_delta
                    if delta >= 0.03:
                        conflicts.append(
                            MemoryConflict(
                                conflict_id=_conflict_id(project, dataset, "metric_conflict", run_ids),
                                project=project,
                                dataset=dataset,
                                conflict_type="metric_conflict",
                                run_ids=run_ids,
                                reason=f"{metric} differs by {delta:.4f} between comparable-looking runs",
                                recommended_winner=winner.run_id,
                            )
                        )
    return conflicts


def write_conflict_report(base_dir: Path, conflicts: list[MemoryConflict]) -> Path:
    path = conflict_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(conflict) for conflict in conflicts], ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_conflicts(base_dir: Path) -> list[MemoryConflict]:
    path = conflict_path(base_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [MemoryConflict(**item) for item in payload if isinstance(item, dict)]


def append_lifecycle_action(base_dir: Path, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        "action_id": stable_content_hash(json.dumps([action, payload, _now()], ensure_ascii=False))[:16],
        "action": action,
        "payload": payload,
        "created_at": _now(),
    }
    path = lifecycle_actions_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row
