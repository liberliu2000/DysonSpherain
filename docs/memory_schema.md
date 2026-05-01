# Memory Schema

Project memory is stored under `artifacts/project_state/` as:

- project state JSON
- append-only memory JSONL
- benchmark-derived status
- constraints and do-not-do lists
- recent decisions and relevant artifact references

Secrets are redacted before durable write-back.
