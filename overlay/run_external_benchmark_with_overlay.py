from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


QUEST_ROOT = Path(__file__).resolve().parents[3]
OVERLAY_ROOT = QUEST_ROOT / "experiments" / "main" / "dyson_overlay"
BENCHMARK_OVERLAY_ROOT = OVERLAY_ROOT / "benchmarks"


def resolve_deepscientist_root() -> Path:
    for candidate in [QUEST_ROOT, *QUEST_ROOT.parents]:
        if candidate.name == "DeepScientist":
            return candidate
    parents = list(QUEST_ROOT.parents)
    fallback_index = 5 if len(parents) > 5 else len(parents) - 1
    return parents[fallback_index]


DEEPSCIENTIST_ROOT = resolve_deepscientist_root()
RECOVERY_PARENT = DEEPSCIENTIST_ROOT / "Quest-002"
RECOVERY_CODE_ROOT_NAME = "sphere_memory_cli_next_main_code_20260417_164120"


def iter_repo_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_value = os.environ.get("DYSONSPHERAIN_REPO_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    if RECOVERY_PARENT.exists():
        recovered = sorted(
            RECOVERY_PARENT.glob(f"recovered-dysonspherain-*/{RECOVERY_CODE_ROOT_NAME}"),
            reverse=True,
        )
        candidates.extend(recovered)
    return candidates


def resolve_default_repo_root() -> Path | None:
    for candidate in iter_repo_root_candidates():
        if candidate.exists():
            return candidate.resolve()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an external DysonSpherain benchmark with the quest-local overlay first on sys.path."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="External DysonSpherain code root that contains benchmarks/ and sphere_cli/.",
    )
    parser.add_argument(
        "benchmark_script",
        help="Benchmark script filename under <repo-root>/benchmarks/, for example longmemeval_benchmark.py.",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the benchmark script.",
    )
    return parser.parse_args()


def ensure_import_order(repo_root: Path) -> None:
    ordered_paths = [
        BENCHMARK_OVERLAY_ROOT.resolve(),
        OVERLAY_ROOT.resolve(),
        repo_root.resolve(),
        (repo_root / "benchmarks").resolve(),
    ]
    ordered_strings = {str(path) for path in ordered_paths}
    sys.path = [entry for entry in sys.path if entry not in ordered_strings]
    for path in reversed(ordered_paths):
        if path.exists():
            sys.path.insert(0, str(path))


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve() if args.repo_root is not None else resolve_default_repo_root()
    if repo_root is None:
        raise SystemExit(
            "No DysonSpherain repo root found. Pass --repo-root or set DYSONSPHERAIN_REPO_ROOT."
        )
    benchmark_path = repo_root / "benchmarks" / args.benchmark_script
    if not benchmark_path.exists():
        raise SystemExit(f"Benchmark script not found: {benchmark_path}")
    ensure_import_order(repo_root)
    sys.argv = [str(benchmark_path)] + list(args.benchmark_args)
    runpy.run_path(str(benchmark_path), run_name="__main__")


if __name__ == "__main__":
    main()
