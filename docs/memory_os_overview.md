# Memory OS Overview

DysonSpherain now includes a local external memory layer for long-horizon
engineering, research, and coding-agent workflows.

Core capabilities:

- structured project state
- artifact-backed benchmark registry
- conflict-aware context compilation
- agent preflight packets
- postrun memory write-back
- execution ledger and resume packets
- lifecycle actions for conflict resolution, supersession, pinning, archiving,
  and merging

All formal benchmark and paper claims must remain artifact-backed.

Plan-level acceptance surfaces:

- schema/write-back: `tests/test_memory_schema.py`
- store CRUD/search/archive: `tests/test_memory_store.py`
- CLI remember/search/inspect/update/archive: `tests/test_memory_cli.py`
- context-packet evaluation: `tests/test_memory_os_eval.py`
- local redaction guardrails: `tests/test_security_redaction.py`
