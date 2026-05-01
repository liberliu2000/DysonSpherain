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
3. Call `dyson_project_state`, `dyson_recall`, and `dyson_context_pack` when recommended, with a strict token budget.
4. Call `dyson_token_economy_eval` on the candidate context before using it.
5. If the evaluator returns `inject`, use the rendered context.
6. If it returns `inject_summary_only`, use only the summary and file references.
7. If it returns `return_file_refs_only`, open the referenced files locally instead of copying long prose into the working prompt.
8. If it returns `skip`, continue by reading local files directly; do not inject memory.
9. Prefer file references over full text when the context budget is tight.
10. After finishing, call `dyson_write_memory` with task goal, files changed, commands run, tests or benchmarks run, results, unresolved failures, token economy anomalies, and next recommended actions.

Never copy full raw recall results directly into the prompt. Use compact rendered context, summaries, or file references selected by the token economy evaluator.

Never store secrets, API keys, private credentials, cookies, raw `.env` files, or irrelevant large logs in memory.
