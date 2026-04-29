from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quest-local EpBench source-and-protocol smoke."
    )
    parser.add_argument(
        "--epbench-root",
        type=Path,
        default=Path("tmp/epbench_source"),
        help="Quest-local EpBench source snapshot root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/experiment/epbench-wave1-smoke"),
        help="Durable output root for manifests, metrics, and summaries.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Explicit EpBench .env path. Defaults to <epbench-root>/.env.",
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=None,
        help="Explicit EpBench data folder. Defaults to <epbench-root>/epbench/data.",
    )
    parser.add_argument(
        "--book-nb-events",
        type=int,
        default=20,
        help="Official quickstart event count to reference in the smoke contract.",
    )
    parser.add_argument(
        "--answering-kind",
        default="prompting",
        help="Official quickstart answering mode to reference.",
    )
    parser.add_argument(
        "--answering-model-name",
        default="gpt-4o-mini-2024-07-18",
        help="Official quickstart answering model to reference.",
    )
    parser.add_argument(
        "--mode",
        choices=("audit_only", "auto_quickstart"),
        default="audit_only",
        help="Whether to only audit setup or also attempt the official quickstart when ready.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output root before running.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def ensure_output_root(output_root: Path, clean_output: bool) -> None:
    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def extract_download_url(readme_text: str) -> str | None:
    match = re.search(r"https://doi\.org/\S+", readme_text)
    return match.group(0) if match else None


def list_first_entries(root: Path, limit: int = 20) -> list[str]:
    if not root.exists():
        return []
    entries = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    return entries[:limit]


def build_quickstart_command(args: argparse.Namespace, env_file: Path, data_folder: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "epbench.experiments.quickstart",
        "--data_folder",
        str(data_folder),
        "--env_file",
        str(env_file),
        "--book_nb_events",
        str(args.book_nb_events),
        "--answering_kind",
        args.answering_kind,
        "--answering_model_name",
        args.answering_model_name,
    ]


def maybe_run_quickstart(
    args: argparse.Namespace,
    env_file: Path,
    data_folder: Path,
    output_root: Path,
) -> dict[str, Any]:
    run_info: dict[str, Any] = {
        "attempted": False,
        "completed": False,
        "exit_code": None,
        "duration_seconds": 0.0,
        "stdout_path": None,
        "stderr_path": None,
        "reason": "audit_only_mode",
    }
    if args.mode != "auto_quickstart":
        return run_info
    if not env_file.exists():
        run_info["reason"] = "env_missing"
        return run_info
    if not data_folder.exists():
        run_info["reason"] = "data_missing"
        return run_info

    stdout_path = output_root / "quickstart.stdout.log"
    stderr_path = output_root / "quickstart.stderr.log"
    command = build_quickstart_command(args, env_file, data_folder)
    start = time.time()
    proc = subprocess.run(
        command,
        cwd=args.epbench_root,
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.time() - start
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    run_info.update(
        {
            "attempted": True,
            "completed": proc.returncode == 0,
            "exit_code": proc.returncode,
            "duration_seconds": round(duration, 3),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "reason": "executed",
            "command": command,
        }
    )
    return run_info


def render_metrics_md(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# EpBench Smoke Metrics",
            "",
            f"- `source_present`: `{metrics['source_present']}`",
            f"- `quickstart_present`: `{metrics['quickstart_present']}`",
            f"- `env_present`: `{metrics['env_present']}`",
            f"- `data_present`: `{metrics['data_present']}`",
            f"- `download_url_present`: `{metrics['download_url_present']}`",
            f"- `quickstart_attempted`: `{metrics['quickstart_attempted']}`",
            f"- `quickstart_completed`: `{metrics['quickstart_completed']}`",
        ]
    )


def render_summary(manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    blockers = manifest["blockers"]
    blocker_lines = ["- none"] if not blockers else [f"- {item}" for item in blockers]
    next_step = manifest["next_step"]
    return "\n".join(
        [
            "# EpBench Smoke Summary",
            "",
            "## What this pass validated",
            "",
            f"- Official source root present: `{metrics['source_present']}`",
            f"- Official quickstart path present: `{metrics['quickstart_present']}`",
            f"- Official benchmark download URL present: `{metrics['download_url_present']}`",
            "",
            "## Current blockers",
            "",
            *blocker_lines,
            "",
            "## Interpretation",
            "",
            textwrap.fill(
                "This smoke is a source-and-protocol audit. It does not claim any EpBench score unless the official quickstart actually runs.",
                width=88,
            ),
            "",
            "## Next step",
            "",
            f"- {next_step}",
        ]
    )


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_output_root(output_root, args.clean_output)

    epbench_root = args.epbench_root.resolve()
    args.epbench_root = epbench_root
    env_file = (args.env_file or (epbench_root / ".env")).resolve()
    data_folder = (args.data_folder or (epbench_root / "epbench" / "data")).resolve()
    quickstart_path = epbench_root / "epbench" / "experiments" / "quickstart.py"
    readme_path = epbench_root / "README.md"

    source_present = epbench_root.exists()
    quickstart_present = quickstart_path.exists()
    env_present = env_file.exists()
    data_present = data_folder.exists()
    readme_text = read_text(readme_path) if readme_path.exists() else ""
    download_url = extract_download_url(readme_text)
    download_url_present = download_url is not None

    blockers: list[str] = []
    if not source_present:
        blockers.append("official_source_missing")
    if source_present and not quickstart_present:
        blockers.append("quickstart_missing")
    if not env_present:
        blockers.append("env_missing")
    if not data_present:
        blockers.append("data_missing")

    quickstart_run = maybe_run_quickstart(args, env_file, data_folder, output_root)

    if quickstart_run["attempted"] and not quickstart_run["completed"]:
        blockers.append("quickstart_failed")

    if "env_missing" in blockers and "data_missing" in blockers:
        next_step = (
            "Download the official produced benchmark data and provide a valid .env so the official quickstart can be attempted."
        )
    elif "data_missing" in blockers:
        next_step = (
            "Download and unpack the official produced benchmark data so the official quickstart has a valid data root."
        )
    elif "env_missing" in blockers:
        next_step = (
            "Provide a valid .env with a supported model API key so the official quickstart can be attempted against the local data root."
        )
    else:
        next_step = "Run the official quickstart and capture the first real EpBench score bundle."

    metrics = {
        "source_present": source_present,
        "quickstart_present": quickstart_present,
        "env_present": env_present,
        "data_present": data_present,
        "download_url_present": download_url_present,
        "quickstart_attempted": quickstart_run["attempted"],
        "quickstart_completed": quickstart_run["completed"],
    }

    manifest = {
        "run_id": "epbench-wave1-smoke-v1",
        "stage": "experiment",
        "created_at": now_iso(),
        "mode": args.mode,
        "epbench_root": str(epbench_root),
        "env_file": str(env_file),
        "data_folder": str(data_folder),
        "quickstart_path": str(quickstart_path),
        "official_download_url": download_url,
        "quickstart_reference_command": build_quickstart_command(args, env_file, data_folder),
        "blockers": blockers,
        "next_step": next_step,
        "epbench_first_entries": list_first_entries(epbench_root, limit=20),
        "quickstart_run": quickstart_run,
    }

    artifact_manifest = {
        "kind": "epbench_smoke",
        "output_root": str(output_root),
        "generated_at": now_iso(),
        "files": [
            "artifact_manifest.json",
            "run_manifest.json",
            "metrics.json",
            "metrics.md",
            "summary.md",
            "runlog.summary.md",
        ],
    }

    write_json(output_root / "artifact_manifest.json", artifact_manifest)
    write_json(output_root / "run_manifest.json", manifest)
    write_json(output_root / "metrics.json", metrics)
    (output_root / "metrics.md").write_text(render_metrics_md(metrics) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text(render_summary(manifest, metrics) + "\n", encoding="utf-8")
    (output_root / "runlog.summary.md").write_text(
        "\n".join(
            [
                "# EpBench Smoke Runlog",
                "",
                f"- created_at: `{manifest['created_at']}`",
                f"- mode: `{args.mode}`",
                f"- blockers: `{', '.join(blockers) if blockers else 'none'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"output_root": str(output_root), "blockers": blockers, "next_step": next_step}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
