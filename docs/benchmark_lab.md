# Benchmark Lab

Benchmark Lab records benchmark outputs without changing benchmark runners.

Commands:

```bash
dyson benchmark-lab record --artifact BenchmarkResult/run/metrics.json --project DysonSpherain
dyson benchmark-lab compare --current BenchmarkResult/new --baseline BenchmarkResult/old --project DysonSpherain
```

Recorded runs are stored in `benchmark_runs` and also summarized as evidence
capsules with artifact references. Comparisons compute numeric deltas from real
metrics files and write `.memory/artifacts/benchmark_regression_report.json`.

The API exposes `GET /api/benchmark-runs`.

Dashboard JSON files are written under `.memory/artifacts/benchmark_lab/`:

- `benchmark_runs.json`
- `metric_trends.json`
- `regression_report.json`
- `candidate_admission_report.json`
- `latency_report.json`

The local daemon also exposes `GET /api/benchmark-dashboard`.

Benchmark records include a binding payload with dataset, config hash, duration,
hardware metadata when present, git metadata, and normalized quality fields such
as recall, NDCG, candidate recall, and latency.
