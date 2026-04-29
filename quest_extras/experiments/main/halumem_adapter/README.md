## HaluMem Adapter

This adapter runs a quest-local HaluMem smoke against the current DysonSpherain runtime
without injecting benchmark gold memories into the system.

### Current scope

- Input: one HaluMem user trajectory JSON/JSONL file
- Ingestion: user dialogue turns are written into a fresh DysonSpherain workspace
- Outputs:
  - session-level generated payload compatible with HaluMem's expected fields
  - question traces with retrieval and answer diagnostics
  - lightweight proxy metrics and latency summaries

### What this does not claim

- It is not an official full HaluMem score run
- It does not use gold `memory_points` as system memory
- It does not run HaluMem's full judge-heavy evaluation by default

### Typical smoke commands

Small command-path smoke:

```bash
python3 experiments/main/halumem_adapter/run_halumem_smoke.py \
  --max-sessions 2 \
  --max-questions 6
```

Bounded full-trajectory smoke:

```bash
python3 experiments/main/halumem_adapter/run_halumem_smoke.py
```
