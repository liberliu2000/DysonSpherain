from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


CODEX_BLOCK = """[mcp_servers.dyson-memory]
command = "python"
args = ["-m", "dysonspherain.adapters.mcp_server"]
cwd = "."
tool_timeout_sec = 60
startup_timeout_sec = 20
enabled_tools = [
  "dyson_memory_intent",
  "dyson_recall",
  "dyson_context_pack",
  "dyson_write_memory",
  "dyson_project_state",
  "dyson_token_economy_eval",
  "dyson_search_memory",
  "dyson_timeline",
  "dyson_get_observations",
  "dyson_resume_context"
]

[mcp_servers.dyson-memory.env]
DYSON_HOME = ".dyson"
DYSON_PROJECT_ROOT = "."
"""

CODEX_ENABLED_TOOLS = [
    "dyson_memory_intent",
    "dyson_recall",
    "dyson_context_pack",
    "dyson_write_memory",
    "dyson_project_state",
    "dyson_token_economy_eval",
    "dyson_search_memory",
    "dyson_timeline",
    "dyson_get_observations",
    "dyson_resume_context",
]


CLAUDE_HOOKS = {
    "SessionStart": "python -m dysonspherain.adapters.claude_hooks.session_start",
    "UserPromptSubmit": "python -m dysonspherain.adapters.claude_hooks.user_prompt_submit",
    "PostToolUse": "python -m dysonspherain.adapters.claude_hooks.post_tool_use",
    "Stop": "python -m dysonspherain.adapters.claude_hooks.stop",
    "SessionEnd": "python -m dysonspherain.adapters.claude_hooks.session_end",
    "PostCompact": "python -m dysonspherain.adapters.claude_hooks.post_compact",
}


@dataclass(frozen=True)
class InstallResult:
    path: str
    changed: bool
    backup_path: str | None = None
    warnings: list[str] | None = None

    def to_dict(self) -> dict:
        return {"path": self.path, "changed": self.changed, "backup_path": self.backup_path, "warnings": self.warnings or []}


def _backup(path: Path) -> str | None:
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    return str(backup)


def install_codex_mcp(project: Path) -> InstallResult:
    config = project / ".codex" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    existing = config.read_text(encoding="utf-8") if config.exists() else ""
    if "[mcp_servers.dyson-memory]" in existing:
        if all(tool in existing for tool in CODEX_ENABLED_TOOLS):
            return InstallResult(str(config), changed=False)
        backup = _backup(config)
        tools_block = "enabled_tools = [\n" + "\n".join(f'  "{tool}",' for tool in CODEX_ENABLED_TOOLS) + "\n]"
        updated, count = re.subn(r"enabled_tools\s*=\s*\[[^\]]*\]", tools_block, existing, count=1, flags=re.S)
        if count == 0:
            updated = existing.rstrip() + "\n" + tools_block + "\n"
        config.write_text(updated, encoding="utf-8")
        return InstallResult(str(config), changed=True, backup_path=backup)
    backup = _backup(config)
    prefix = existing.rstrip() + "\n\n" if existing.strip() else ""
    config.write_text(prefix + CODEX_BLOCK, encoding="utf-8")
    return InstallResult(str(config), changed=True, backup_path=backup)


def install_agents_policy(project: Path) -> InstallResult:
    target = project / "AGENTS.md"
    policy = (Path(__file__).with_name("AGENTS.memory.md")).read_text(encoding="utf-8")
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    warnings: list[str] = []
    if "## DysonSpherain Memory Policy" in existing:
        return InstallResult(str(target), changed=False)
    if len(existing) > 24000:
        warnings.append("AGENTS.md is large; policy was appended without memory content.")
    backup = _backup(target)
    target.write_text((existing.rstrip() + "\n\n" if existing.strip() else "") + policy + "\n", encoding="utf-8")
    return InstallResult(str(target), changed=True, backup_path=backup, warnings=warnings)


def install_claude_hooks(project: Path) -> InstallResult:
    settings = project / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if settings.exists():
        try:
            payload = json.loads(settings.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    hooks = payload.setdefault("hooks", {})
    changed = False
    for event, command in CLAUDE_HOOKS.items():
        entries = hooks.setdefault(event, [])
        exists = any(command in json.dumps(entry) for entry in entries)
        if not exists:
            entries.append({"hooks": [{"type": "command", "command": command}]})
            changed = True
    if not changed:
        return InstallResult(str(settings), changed=False)
    backup = _backup(settings)
    settings.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return InstallResult(str(settings), changed=True, backup_path=backup)


def install_plugin_manifests(project: Path) -> InstallResult:
    codex_plugin = project / ".codex-plugin" / "plugin.json"
    claude_plugin = project / ".claude-plugin" / "plugin.json"
    opencode = project / ".opencode" / "dyson-memory.json"
    gemini = project / ".gemini" / "dyson-memory.json"
    manifest = {
        "name": "dyson-memory",
        "displayName": "DysonSpherain Memory",
        "description": "MCP memory, progressive observation recall, hooks, and token economy diagnostics.",
        "mcp": {"command": "python", "args": ["-m", "dysonspherain.adapters.mcp_server"]},
        "tools": [
            "dyson_memory_intent",
            "dyson_search_memory",
            "dyson_timeline",
            "dyson_get_observations",
            "dyson_resume_context",
            "dyson_recall",
            "dyson_context_pack",
            "dyson_write_memory",
            "dyson_project_state",
            "dyson_token_economy_eval",
        ],
    }
    changed = False
    for path in (codex_plugin, claude_plugin, opencode, gemini):
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8", errors="replace") == text:
            continue
        _backup(path)
        path.write_text(text, encoding="utf-8")
        changed = True
    return InstallResult(str(codex_plugin), changed=changed)


def doctor(project: Path) -> dict:
    from dysonspherain.adapters.mcp_server import call_tool, smoke_payload, temporary_allowed_roots
    from dysonspherain.context_pack.schemas import ContextPack, EvidenceItem
    from dysonspherain.context_pack.token_budgeter import fit_context_pack
    from dysonspherain.utils.token_counter import TokenCounter
    from dysonspherain.writeback.deduper import classify_duplicate
    from dysonspherain.writeback.sanitizer import sanitize_payload

    codex = project / ".codex" / "config.toml"
    claude = project / ".claude" / "settings.json"
    agents = project / "AGENTS.md"
    dyson_home = project / ".dyson"
    dyson_home.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {}
    checks["package_import"] = {"ok": True}
    checks["mcp_smoke"] = {"ok": smoke_payload().get("status") == "ok", "tools": smoke_payload().get("tools")}
    with temporary_allowed_roots([project, dyson_home]):
        recall = call_tool("dyson_recall", {"query": "doctor recall check", "cwd": str(project), "token_budget": 200})
        checks["dyson_recall"] = {"ok": recall.get("status") in {"ok", "empty"}, "status": recall.get("status")}
        context_pack = call_tool("dyson_context_pack", {"query": "doctor context check", "cwd": str(project), "token_budget": 800})
        checks["dyson_context_pack"] = {"ok": context_pack.get("status") == "ok", "over_budget": context_pack.get("token_estimate", {}).get("over_budget")}
        with tempfile.TemporaryDirectory(dir=dyson_home) as tmp:
            write = call_tool(
                "dyson_write_memory",
                {
                    "cwd": tmp,
                    "session_id": "doctor",
                    "task_goal": "doctor write check",
                    "summary": "doctor write check",
                    "files_changed": [],
                    "commands_run": [],
                    "tests_run": [],
                    "benchmark_results": [],
                    "failures": [],
                    "next_actions": [],
                    "source": "doctor",
                },
            )
            duplicate = call_tool("dyson_write_memory", {**{"cwd": tmp, "session_id": "doctor", "task_goal": "doctor write check", "summary": "doctor write check", "source": "doctor"}, **{key: [] for key in ("files_changed", "commands_run", "tests_run", "benchmark_results", "failures", "next_actions")}})
    checks["dyson_write_memory"] = {"ok": write.get("status") == "ok" and duplicate.get("status") == "duplicate", "dedupe_status": duplicate.get("status")}
    token_count = TokenCounter().count("doctor token counter")
    checks["token_counter"] = {"ok": token_count.tokens > 0, "fallback_used": token_count.fallback_used, "tokenizer_name": token_count.tokenizer_name}
    pack = ContextPack(summary="doctor", core_evidence=[EvidenceItem(text="doctor evidence")])
    budget = fit_context_pack(pack, 80)
    checks["token_budgeter"] = {"ok": budget.estimated_tokens_after > 0, "over_budget": budget.over_budget}
    sanitized = sanitize_payload({"summary": "OPENAI_API_KEY=sk-abcdef1234567890"})
    checks["sanitizer"] = {"ok": sanitized.has_redaction, **sanitized.to_dict()}
    with tempfile.TemporaryDirectory(dir=dyson_home) as tmp:
        dedupe = classify_duplicate(Path(tmp), "DysonSpherain", "doctor content")
    checks["deduper"] = {"ok": not dedupe.is_duplicate, "dedupe_reason": dedupe.dedupe_reason}
    test_file = dyson_home / ".doctor_write_test"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink(missing_ok=True)
    checks["dyson_home_writable"] = {"ok": True, "path": str(dyson_home)}
    checks["memory_store"] = {"ok": (project / "artifacts").exists() or project.exists(), "path": str(project / "artifacts" / "project_state")}
    checks["codex_mcp"] = {"ok": codex.exists() and "[mcp_servers.dyson-memory]" in codex.read_text(encoding="utf-8", errors="replace"), "path": str(codex)}
    checks["claude_hooks"] = {"ok": claude.exists() and "dysonspherain.adapters.claude_hooks" in claude.read_text(encoding="utf-8", errors="replace"), "path": str(claude)}
    checks["agents_policy"] = {"ok": agents.exists() and "## DysonSpherain Memory Policy" in agents.read_text(encoding="utf-8", errors="replace"), "path": str(agents)}
    warnings = []
    if not checks["codex_mcp"]["ok"]:
        warnings.append("Run: dyson adapters install-codex-mcp --project .")
    if not checks["claude_hooks"]["ok"]:
        warnings.append("Run: dyson adapters install-claude-hooks --project .")
    if not checks["agents_policy"]["ok"]:
        warnings.append("Run: dyson adapters install-agents-policy --project .")
    return {
        "status": "ok" if all(check.get("ok") for check in checks.values()) else "warning",
        "checks": checks,
        "warnings": warnings,
    }
