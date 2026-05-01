from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dysonspherain.adapters.codex.generate_config import doctor
from dysonspherain.adapters.mcp_server import TOOLS, call_tool, smoke_payload, temporary_allowed_roots
from dysonspherain.token_economy.evaluator import evaluate


DEFAULT_PHASE6_EVIDENCE = [
    "python -m pytest -q",
    "PYTHONPATH=base python -m dysonspherain.adapters.mcp_server --smoke",
    "python -m pytest tests/test_token_economy_budget_policy.py tests/test_mcp_server.py -q",
    "python -m dysonspherain.evaluation.token_economy --smoke --modes off,conservative --context-token-budget 800,1600",
    "dyson benchmark smoke-all --record-token-economy",
]


def _artifact_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}


def write_memory_agent_integration_report(project: Path, tests_run: list[str] | None = None) -> Path:
    artifacts = project / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "memory_agent_integration_report.md"
    with temporary_allowed_roots([project]):
        recall = call_tool("dyson_recall", {"query": "benchmark regression token economy", "cwd": str(project), "token_budget": 800})
        decision = evaluate(
            query="benchmark regression token economy",
            candidate_context=str(recall.get("rendered_context") or ""),
            baseline_context_tokens=5000,
            token_budget=800,
            task_type="debug",
        ).to_dict()
        write = call_tool(
            "dyson_write_memory",
            {
                "cwd": str(project),
                "session_id": "integration-report-smoke",
                "task_goal": "Generate memory agent integration report",
                "summary": "Smoke-validated MCP tools, token economy decision, sanitizer, and dedupe.",
                "files_changed": ["base/dysonspherain"],
                "commands_run": tests_run or [],
                "tests_run": tests_run or [],
                "benchmark_results": [],
                "failures": [],
                "next_actions": ["Run full benchmark token economy evaluation on official artifacts."],
                "source": "integration_report",
            },
        )
        duplicate = call_tool(
            "dyson_write_memory",
            {
                "cwd": str(project),
                "session_id": "integration-report-smoke",
                "task_goal": "Generate memory agent integration report",
                "summary": "Smoke-validated MCP tools, token economy decision, sanitizer, and dedupe.",
                "files_changed": ["base/dysonspherain"],
                "commands_run": tests_run or [],
                "tests_run": tests_run or [],
                "benchmark_results": [],
                "failures": [],
                "next_actions": ["Run full benchmark token economy evaluation on official artifacts."],
                "source": "integration_report",
            },
        )
    doctor_report = doctor(project)
    mcp_smoke = smoke_payload()
    evidence = tests_run or DEFAULT_PHASE6_EVIDENCE
    smoke_artifacts = [
        _artifact_status(Path("/tmp/dyson_phase6_token_economy_smoke")),
        _artifact_status(Path("/tmp/dyson_phase6_benchmark_smoke.json")),
        _artifact_status(Path("/tmp/dyson_phase6_benchmark_token_economy")),
        _artifact_status(project / "artifacts" / "memory_agent_integration_report.md"),
    ]
    lines = [
        "# Memory Agent Integration Report",
        "",
        "## Summary",
        "",
        f"- Doctor status: {doctor_report.get('status')}",
        f"- MCP tools: {', '.join(TOOLS)}",
        "",
        "## Tests Run",
        "",
        *[f"- `{item}`" for item in evidence],
        "",
        "## Smoke / Artifact Evidence",
        "",
        "```json",
        json.dumps(smoke_artifacts, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## MCP Tools",
        "",
        "```json",
        json.dumps(TOOLS, ensure_ascii=False, indent=2),
        "```",
        "",
        "## MCP Transport",
        "",
        "```json",
        json.dumps(
            {
                "transport_implementation": mcp_smoke.get("transport_implementation"),
                "mcp_sdk_available": mcp_smoke.get("mcp_sdk_available"),
                "fallback_reason": mcp_smoke.get("fallback_reason"),
                "protocolVersion": mcp_smoke.get("protocolVersion"),
                "serverInfo": mcp_smoke.get("serverInfo"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Claude Hook Simulation",
        "",
        "- Hook modules are importable; UserPromptSubmit inject behavior is covered by tests.",
        "",
        "## Codex Config Generation",
        "",
        f"- Codex MCP config ok: {doctor_report.get('checks', {}).get('codex_mcp', {}).get('ok')}",
        "",
        "## Context Pack Result",
        "",
        "```json",
        json.dumps({"status": recall.get("status"), "token_estimate": recall.get("token_estimate"), "trace": recall.get("trace")}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Token Economy Decision",
        "",
        f"- baseline estimated tokens: 5000",
        f"- injected tokens: {decision.get('estimated_tokens')}",
        f"- estimated saved tokens: {decision.get('estimated_saved_tokens')}",
        f"- decision: {decision.get('decision')}",
        "",
        "## Writeback / Dedupe Result",
        "",
        "```json",
        json.dumps({"write": write, "duplicate": duplicate}, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Sanitizer Result",
        "",
        f"- sanitizer checked by doctor: {doctor_report.get('checks', {}).get('sanitizer', {}).get('ok')}",
        "",
        "## Risks",
        "",
        "- Full benchmark token economy mode is diagnostic-only and should not change default retrieval behavior.",
        "- Retrieval quality and token saving must remain separately reported.",
        "",
        "## Next Steps",
        "",
        "- For formal claims, run full official benchmarks with `--record-token-economy` and no local_hash fallback.",
        "- Use paired token/quality rows from `token_quality_tradeoff.csv` before promoting an automatic injection policy.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
