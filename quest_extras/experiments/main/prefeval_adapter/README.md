# PrefEval Adapter

This quest-local adapter runs a bounded PrefEval smoke pass against the current
DysonSpherain memory runtime.

It is intentionally honest about the current system boundary:

- the runtime already supports preference writeback, evidence retrieval,
  structured object completion, and trace assembly
- the runtime does not yet ship a built-in final natural-language answer generator
- therefore this smoke measures preference reactivation readiness, not official
  final-answer PrefEval accuracy

## First-pass usage

Run it with the DysonSpherain project virtualenv:

```bash
/home/liber/Projects/DysonSpherain/sphere_memory_cli_next_main_code_20260417_164120/.venv/bin/python \
  experiments/main/prefeval_adapter/run_prefeval_smoke.py \
  --topic travel_restaurant \
  --max-cases 5 \
  --clean-output
```

Durable outputs go to:

- `artifacts/experiment/prefeval-wave1-smoke/run_manifest.json`
- `artifacts/experiment/prefeval-wave1-smoke/metrics.json`
- `artifacts/experiment/prefeval-wave1-smoke/predictions.jsonl`
- `artifacts/experiment/prefeval-wave1-smoke/summary.md`
- `artifacts/experiment/prefeval-wave1-smoke/traces/<case_id>/trace.json`

## Current metric contract

The current smoke summary reports:

- `reactivation_success_rate`
- `route_persona_preference_state_rate`
- `preference_object_recall_rate`
- `preference_polarity_match_rate`
- latency and assembled-context token deltas

These metrics are a bridge between the current memory runtime and a later
generation layer that can be added without rewriting ingestion or trace capture.
