## DysonSpherain Memory Policy

Before implementing any non-trivial change, use the `dyson-memory` MCP tools when the task involves:

- prior benchmark results
- regression triage
- previous project decisions
- long-horizon memory retrieval
- token economy / context compression
- paper draft updates
- old Codex / Claude prompts
- repeated user preferences
- roadmap or architecture continuity

Do not ask the user to paste previous context if `dyson_recall` can retrieve it.

For large tasks:

1. Call `dyson_memory_intent` first with the user prompt, cwd, project, and task type.
2. Follow its `recommended_tools`; for `cross_session_continuation`, call `dyson_resume_context` before asking the user for old context.
3. Call `dyson_project_state` and `dyson_recall` when recommended, with a strict token budget.
4. Use `dyson_context_pack` if the retrieved context is too long.
5. Prefer file references over full text when the context budget is tight.
6. After finishing, call `dyson_write_memory` with task goal, files changed, commands run, tests or benchmarks run, results, unresolved failures, and next recommended actions.

Never store secrets, API keys, private credentials, cookies, raw `.env` files, or irrelevant large logs in memory.
