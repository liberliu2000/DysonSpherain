# Context Compiler

`ds memory context` and `ds agent preflight` generate Markdown packets for
Codex, paper, benchmark, debug, docs, and project modes.

Packets include:

- task objective
- current project state
- benchmark and experiment status
- conflict warnings
- pinned constraints
- relevant artifacts
- suggested execution plan

Packets are token-budgeted and secret-redacted.
