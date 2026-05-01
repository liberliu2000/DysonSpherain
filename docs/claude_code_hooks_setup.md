# Claude Code Hooks Setup

Install project-level Claude Code hooks:

```bash
dysonspherain adapters install-claude-hooks --project .
```

Installed hooks:

- `SessionStart`: injects a compact project-state summary.
- `UserPromptSubmit`: recalls compact memory context before a prompt.
- `PostToolUse`: records useful tool observations such as tests, edits, and benchmark output.
- `Stop`: writes a final stop summary when useful.
- `SessionEnd`: summarizes and writes durable task memory.
- `PostCompact`: writes useful compaction summaries.

Hook writeback is best-effort and should not block agent exit.

Hook behavior is covered by unit tests for short-prompt skip, context injection, SessionEnd write errors, and PostCompact writeback/dedupe. Hook writeback runs through the same sanitizer and deduper used by the MCP write tool.

`UserPromptSubmit` now includes a token economy note in injected context: decision, injected token estimate, estimated saved tokens, and budget usage ratio.
