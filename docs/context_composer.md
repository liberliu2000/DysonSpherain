# Context Composer

`create_context_pack` converts retrieval candidates into an auditable Markdown
packet. It stores both the rendered Markdown and structured section payload in
the `context_packs` table.

Default sections:

- Mission State
- Must-Use Anchors
- Recent State
- Relevant Decisions
- Known Pitfalls
- Supporting Evidence
- Creative Bridges
- Conflicts and Invalidated Evidence
- Open Questions
- Excluded Evidence

The composer respects `max_tokens` using the repository token counter. Evidence
that does not fit is listed under Excluded Evidence with a token-budget reason.

Controls:

- `sections`: optional list of section names to include.
- `section_budget`: optional per-section token caps.
- `task_type`: route hint passed into retrieval.
- `agent_role`: role label included in Mission State.
- `include_raw_quotes`, `include_artifact_refs`, `include_debug_trace`.

Formats: `markdown`, `json`, `yaml`, and `text`.
