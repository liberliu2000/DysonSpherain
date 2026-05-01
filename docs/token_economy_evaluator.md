# Token Economy Evaluator

Run a standalone smoke evaluation:

```bash
python -m dysonspherain.evaluation.token_economy \
  --smoke \
  --output artifacts/token_economy_smoke \
  --modes conservative,exploratory \
  --baseline-types full_history,naive_recent \
  --context-token-budget 1000,2000
```

Artifacts:

- `per_sample.jsonl`
- `summary.json`
- `summary.md`
- `mode_comparison.csv`
- `token_quality_tradeoff.csv`
- `failure_cases.json`

Token saving and retrieval quality are reported separately. Tokenizer fallback is explicitly recorded in per-sample and summary artifacts.

Benchmark runners accept `--record-token-economy` plus diagnostic thresholds:

```bash
--low-saving-threshold 0.2
--quality-drop-threshold 0.05
--evidence-bloat-threshold 0.85
--metadata-bloat-threshold 0.25
```

`--memory-db` connects the evaluator to the existing memory/runtime assembly path. If a runtime mode is missing, unregistered, or has no memory DB when one is required, the sample is marked unavailable instead of fabricating retrieval output.

Multiple `--context-token-budget` values produce separate artifact rows. By default, oversized single evidence records are not chopped mid-record; pass `--allow-evidence-truncation` only when lossy evidence truncation is acceptable.
