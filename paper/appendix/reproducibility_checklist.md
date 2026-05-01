# Reproducibility Checklist

- Formal runs must use `SPHERE_EMBEDDING_FAIL_FAST=1`.
- Formal runs must use `SPHERE_VECTOR_BACKEND=chroma` and `fallback_in_use=false`.
- Tables must be regenerated from artifacts, not hand-filled.
- Smoke, partial, fallback, and full runs must remain distinguishable.
- Missing baselines/ablations stay pending until artifacts exist.
- Token economy claims must name their artifact source and remain separate from retrieval-quality claims.
- Prompt-token savings and local compute/cache savings must not be added together.
- Current token economy support artifacts:
  - Smoke validation: `artifacts/token_economy_upgrade_smoke`
  - Existing full-compare diagnostic rollup: `artifacts/token_economy_full_compare_diagnostic`
  - Tokenizer calibration sample output: `artifacts/tokenizer_calibration.json`
