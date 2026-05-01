# Agent Adapter

Agent workflow commands:

```bash
ds agent preflight "Improve retrieval quality without regression"
ds agent postrun ./logs/codex_run.txt --run-id <run_id>
ds agent ledger list
ds agent ledger resume-packet <run_id>
```

`postrun` writes a redacted memory summary and updates the append-only execution
ledger.
