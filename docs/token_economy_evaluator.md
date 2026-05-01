# Token Economy Evaluator

DysonSpherain reports **LLM prompt token saving** separately from **local compute saving**. Prompt saving means fewer tokens are injected into the model prompt. Local compute saving means local cache/runtime work avoided; it is useful operationally but is not added to prompt-token savings.

Run a standalone smoke evaluation:

```bash
dysonspherain evaluate-token-economy-smoke \
  --samples 20 \
  --output artifacts/token_economy_smoke
```

Run a full evaluation from available artifacts:

```bash
dysonspherain evaluate-token-economy \
  --benchmark-artifact-root artifacts \
  --memory-db .dyson \
  --modes off,conservative,exploratory,minimal \
  --baseline-types full_history,naive_recent,manual_summary \
  --context-token-budget 800,1200,1600,2400 \
  --output artifacts/token_economy
```

The module entrypoint remains available:

```bash
python -m dysonspherain.evaluation.token_economy \
  --smoke \
  --output artifacts/token_economy_smoke \
  --modes off,conservative,exploratory,minimal \
  --baseline-types full_history,naive_recent,manual_summary \
  --context-token-budget 1000,2000
```

Artifacts:

- `manifest.json`
- `per_sample.jsonl`
- `summary.json`
- `summary.md`
- `mode_comparison.csv`
- `token_quality_tradeoff.csv`
- `failure_cases.json`
- `tokenizer_calibration.json`
- `ledger_summary.json`
- `final_report.md`

Token saving and retrieval quality are reported separately. Tokenizer fallback is explicitly recorded in per-sample and summary artifacts.

Key `summary.json` fields:

- `llm_prompt_token_economy`: baseline prompt tokens, final injected tokens, estimated saved tokens, and saved ratio.
- `local_compute_economy`: local cache/runtime indicators; these are not counted as prompt-token savings.
- `decision_distribution`: counts for `inject`, `skip`, `inject_summary_only`, and `return_file_refs_only`.
- `fallback_tokenizer_rate`: share of rows using heuristic tokenization.
- `over_budget_rate`: share of rows whose final prompt exceeded the requested context budget.
- `failure_case_counts`: token regression, low saving, bloat, and quality-drop diagnostics.

`mode_comparison.csv` compares mode/baseline/budget groups. `failure_cases.json` lists samples that need inspection before enabling an automatic policy broadly.

Tokenizer calibration can be generated locally:

```bash
dysonspherain calibrate-tokenizer \
  --input sample_data/tokenizer_calibration_samples.jsonl \
  --output artifacts/tokenizer_calibration.json
```

Samples may include `reference_tokens`. If they do not, the calibration command still writes a heuristic distribution artifact.

Benchmark runners accept `--record-token-economy` plus diagnostic thresholds:

```bash
--low-saving-threshold 0.2
--quality-drop-threshold 0.05
--evidence-bloat-threshold 0.85
--metadata-bloat-threshold 0.25
```

`--memory-db` connects the evaluator to the existing memory/runtime assembly path. If a runtime mode is missing, unregistered, or has no memory DB when one is required, the sample is marked unavailable instead of fabricating retrieval output.

Multiple `--context-token-budget` values produce separate artifact rows. By default, oversized single evidence records are not chopped mid-record; pass `--allow-evidence-truncation` only when lossy evidence truncation is acceptable.
