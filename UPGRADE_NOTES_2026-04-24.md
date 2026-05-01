# DysonSpherain CLI upgrade notes — 2026-04-24

This patch converts the uploaded code snapshot from a benchmark-oriented memory retriever into a safer, more reproducible local memory CLI.

## Implemented

1. **Embedding fail-fast guard**
   - Added `AppConfig.embedding_fail_fast`.
   - Added `SPHERE_EMBEDDING_FAIL_FAST=1`.
   - `EmbeddingProvider(..., fail_fast=True)` now raises instead of silently falling back to `local-hash`.
   - `status` and `doctor` expose provider/model/fallback/load-error fields.

2. **Offline JSON vector backend**
   - Added a persistent JSON vector backend selected by `SPHERE_VECTOR_BACKEND=json` or automatically used when Chroma is unavailable.
   - This keeps smoke tests and basic CLI workflows runnable in clean/offline environments.
   - Chroma remains the intended backend for production-scale corpora.

3. **Default lightweight graph edge writeback**
   - Added `AppConfig.enable_lightweight_edge_writeback`.
   - Normal writeback now creates lightweight graph edges by default after enough related nodes exist.
   - Batch writeback now actually persists generated edges.

4. **Deterministic final answer layer**
   - Added `sphere_cli.answer_generator.EvidenceAnswerGenerator`.
   - Added runtime `answer(...)`.
   - Added CLI `dysonspherain ask ...`.
   - The local answer mode cites memory ids/chunk ids and abstains when no evidence is retrieved.

5. **User-facing CLI operations**
   - Added `dysonspherain doctor`.
   - Added persistent local config commands:
     - `dysonspherain config get`
     - `dysonspherain config set <key> <value>`
     - `dysonspherain config profile fast|balanced|deep|paper|benchmark`
   - Added data governance commands:
     - `dysonspherain memory export`
     - `dysonspherain memory backup`
     - `dysonspherain memory forget <query> --confirm`

6. **Benchmark artifact scaffold**
   - Added `dysonspherain benchmark smoke-all`.
   - Added `base/benchmarks/run_all_benchmarks.py`.
   - Smoke artifacts include embedding provider, fallback state, edge counts, and a clear `valid_for_full_benchmark_claims=false` flag.

7. **Ingestion coverage**
   - Added lightweight `.docx`, `.pptx`, `.xlsx`, `.html`, `.htm` text extraction without mandatory new dependencies.
   - Existing PDF support remains via `pypdf`.

8. **SQLite robustness**
   - New SQLite connections now use WAL, `busy_timeout=30000`, and foreign keys.

9. **Tests**
   - Added `tests/test_guardrails.py` covering:
     - embedding fail-fast behavior;
     - offline JSON vector backend + answer generator;
     - graph edge writeback after related nodes.

## Not completed inside this patch

Full LongMemEval / LoCoMo / KnowMe / CloneMem numbers were not produced because the uploaded zip does not include the required benchmark datasets. The full-run harness now exists, but official numbers must be generated on a machine with the datasets and the intended embedding model installed/cached.

## Recommended full benchmark command

```bash
export SPHERE_EMBEDDING_FAIL_FAST=1
export SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING=1
export SPHERE_ENABLE_LIGHTWEIGHT_EDGE_WRITEBACK=1
export PYTHONPATH="$(pwd)/base"
python base/benchmarks/run_all_benchmarks.py --data-root /path/to/benchmark_data --out benchmark_runs/2026-04-24-current
```

If using the overlay package:

```bash
export PYTHONPATH="$(pwd)/overlay:$(pwd)/base"
```
