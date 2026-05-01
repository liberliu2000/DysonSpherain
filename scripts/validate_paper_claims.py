#!/usr/bin/env python3
"""Lightweight guardrails for paper claims and generated artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "latex" / "main.tex"
FORMAL = ROOT / "artifacts" / "formal_protocol_validation.json"
REQUIRED = [
    "Introduction",
    "Candidate Admission Failures in Long-Horizon Memory",
    "DysonSpherain Method",
    "Artifact-Backed Evaluation Protocol",
    "Experiments",
    "Results",
    "Case Studies",
    "Related Work",
    "Limitations",
    "Ethics and Privacy Considerations",
    "Conclusion",
]


def main() -> int:
    errors: list[str] = []
    text = MAIN.read_text()
    lower = text.lower()

    if "state of the art" in lower or "sota" in lower:
        allowed = "does not claim universal state of the art" in lower or "does not claim sota" in lower
        if not allowed:
            errors.append("Unsupported SOTA wording found.")

    for section in REQUIRED:
        if f"\\section{{{section}}}" not in text:
            errors.append(f"Missing required section: {section}")

    for sentence in re.split(r"[.!?]\s+", lower):
        if "non-comparable" not in sentence and "non_comparable" not in sentence:
            continue
        if not re.search(r"(improv|gain|outperform|better)", sentence):
            continue
        if any(guard in sentence for guard in ("not used", "excluded", "not plotted", "prevents", "do not")):
            continue
        errors.append("Non-comparable artifacts appear near unsupported improvement wording.")

    formal = json.loads(FORMAL.read_text())
    for item in formal["full_benchmarks"]:
        if item.get("fallback_in_use"):
            errors.append(f"Formal result uses fallback: {item['benchmark']}")

    required_files = [
        ROOT / "paper" / "latex" / "tables" / "main_results.tex",
        ROOT / "paper" / "latex" / "tables" / "baseline_comparison.tex",
        ROOT / "paper" / "latex" / "tables" / "ablation_results.tex",
        ROOT / "paper" / "latex" / "tables" / "failure_taxonomy.tex",
        ROOT / "paper" / "latex" / "tables" / "artifact_validation_summary.tex",
        ROOT / "paper" / "latex" / "tables" / "route_policy_hyperparameters.tex",
        ROOT / "paper" / "latex" / "tables" / "clonemem_efficiency_optimization.tex",
        ROOT / "paper" / "latex" / "tables" / "clonemem_efficiency_text.tex",
        ROOT / "paper" / "latex" / "references.bib",
        ROOT / "paper" / "latex" / "figures" / "pipeline_figure.tex",
        ROOT / "paper" / "latex" / "figures" / "benchmark_comparison.pgf",
        ROOT / "paper" / "latex" / "figures" / "efficiency_pareto.pgf",
        ROOT / "paper" / "latex" / "case_studies.tex",
        ROOT / "paper" / "latex" / "tables" / "case_study_examples.tex",
    ]
    for path in required_files:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"Missing generated paper artifact: {path.relative_to(ROOT)}")

    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 1
    print(json.dumps({"status": "passed", "checked_sections": len(REQUIRED)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
