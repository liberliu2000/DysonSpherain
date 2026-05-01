# Benchmark Registry

The benchmark registry ingests artifact directories or direct `metrics.json`
files:

```bash
ds runs ingest ../BenchmarkResult --project DysonSpherain
ds runs list --project DysonSpherain
ds runs latest --dataset CloneMem
ds runs compare --a <run_a> --b <run_b>
ds runs explain-regression --dataset CloneMem
```

Formal results must be non-fallback and full-run artifacts. Smoke, partial,
fallback, and oversized artifacts remain distinguishable.
