# Product CLI

The product CLI is exposed through the existing Typer app.

Common commands:

```bash
dyson init --project DysonSpherain
dyson remember --project DysonSpherain --type decision --text "Keep official benchmark profiles capped."
dyson record --project DysonSpherain --source shell --command "pytest tests/test_product_memory.py" --capture-output
dyson import markdown session.md --project DysonSpherain
dyson search "benchmark profile" --project DysonSpherain
dyson retrieve "benchmark profile" --project DysonSpherain --show-audit --context-pack
dyson wake --project DysonSpherain --task "resume benchmark repair" --max-tokens 4000
dyson inspect cap_xxx --project DysonSpherain
dyson forget --capsule-id cap_xxx --project DysonSpherain
dyson export --project DysonSpherain --format json
dyson index rebuild --project DysonSpherain
dyson index rebuild --project DysonSpherain --backend sentence_transformers --model sentence-transformers/all-MiniLM-L6-v2
dyson index verify --project DysonSpherain
dyson index repair --project DysonSpherain
dyson index embedding-backends --project DysonSpherain
dyson index configure-embedding local_hash_embedding
dyson index configure-embedding sentence_transformers --model sentence-transformers/all-MiniLM-L6-v2 --allow-unavailable
dyson index vector-backends --project DysonSpherain
dyson index configure-vector sqlite_inline
dyson index configure-vector chroma --allow-unavailable
dyson index rebuild-vector --project DysonSpherain
dyson index configure-encryption external_or_os_managed --scope project_volume
dyson index configure-encryption sqlcipher --key-env DYSON_MEMORY_SQLCIPHER_KEY --allow-unavailable
dyson index migrate-sqlcipher --key-env DYSON_MEMORY_SQLCIPHER_KEY
dyson index migrate-sqlcipher --key-env DYSON_MEMORY_SQLCIPHER_KEY --replace
dyson index maintenance --project DysonSpherain
dyson index maintenance --project DysonSpherain --apply sug_xxx --canonical-id cap_xxx
dyson index maintenance --project DysonSpherain --dismiss sug_xxx --reason "not a duplicate"
dyson benchmark-lab record --project DysonSpherain --artifact BenchmarkResult/latest/metrics.json
dyson ui --project DysonSpherain --port 37777
```

Runtime hook commands:

```bash
dyson runtime before-task --project DysonSpherain --task "run KnowMe official profile"
dyson runtime on-error --project DysonSpherain --error-file traceback.txt
dyson runtime after-task --project DysonSpherain --summary "KnowMe rerun completed"
dyson runtime pre-compact --project DysonSpherain
dyson runtime after-benchmark --project DysonSpherain --metrics metrics.json
```

The product commands write to `.memory/` in the current working directory.
