# EpBench Adapter

This quest-local adapter establishes a bounded `EpBench` smoke path against the
official `episodic-memory-benchmark` source snapshot.

It is intentionally honest about the current boundary:

- the official source checkout can be prepared under `tmp/epbench_source`
- the official quickstart exists and is reproducible from that checkout
- a real scored quickstart still has one upstream runtime blocker:
  - cached-data rechecking currently trips an internal dataframe-consistency assertion before the official answering stage
- the official answering lane now has a separate bounded probe:
  - `run_epbench_auth_probe.py` loads the official short-book cache directly and tests one real answering call against an official question
- the current `.env` sample key path has already been checked:
  - after pinning `httpx==0.27.2`, the bounded auth probe reaches the standard OpenAI endpoint and returns `401 invalid_api_key`
- the current provider-injection follow-up is also explicit:
  - the quest-local auth probe now accepts `--openai-base-url`, so one cached official question can be rerun against `right.codes` without patching the upstream source snapshot
- the no-key local continuation lane is now explicit:
  - `export_epbench_question_pack.py` exports a focused temporal question pack from cached official short-book assets
  - `run_epbench_real_task_eval_smoke.py` runs a bounded DysonSpherain `real_task_eval` smoke on that exported pack

## First-pass usage

Run the quest-local smoke with the same Python interpreter that is available in
the current workspace:

```bash
python3 experiments/main/epbench_adapter/run_epbench_smoke.py --clean-output
```

Durable outputs go to:

- `artifacts/experiment/epbench-wave1-smoke/artifact_manifest.json`
- `artifacts/experiment/epbench-wave1-smoke/run_manifest.json`
- `artifacts/experiment/epbench-wave1-smoke/metrics.json`
- `artifacts/experiment/epbench-wave1-smoke/metrics.md`
- `artifacts/experiment/epbench-wave1-smoke/summary.md`
- `artifacts/experiment/epbench-wave1-smoke/runlog.summary.md`

For a single-question answering/auth check on cached official short-book data:

```bash
python3 experiments/main/epbench_adapter/run_epbench_auth_probe.py --clean-output
```

For the provider-injected follow-up against the user-validated `right.codes` lane:

```bash
python3 experiments/main/epbench_adapter/run_epbench_auth_probe.py \
  --clean-output \
  --answering-model-name gpt-5.4 \
  --openai-base-url https://right.codes/codex/v1
```

For the bounded multi-question provider smoke on cached official questions:

```bash
python3 experiments/main/epbench_adapter/run_epbench_provider_smoke.py \
  --clean-output \
  --question-indices 0,50,100
```

For the broader fixed-coverage provider smoke across the main official question families:

```bash
python3 experiments/main/epbench_adapter/run_epbench_provider_smoke.py \
  --clean-output \
  --output-root artifacts/experiment/epbench-wave2-provider-coverage-smoke \
  --question-indices 10,50,148,0,62,153,56,58,158,5,394,389
```

Durable outputs go to:

- `artifacts/experiment/epbench-wave1-auth-probe/summary.json`
- `artifacts/experiment/epbench-wave1-auth-probe/prompt.txt`
- `artifacts/experiment/epbench-wave1-auth-probe/answer.txt`
- `artifacts/experiment/epbench-wave1-auth-probe/reasoning.txt`
- `artifacts/experiment/epbench-wave1-auth-probe/traceback.txt`

For the bounded provider smoke bundle:

- `artifacts/experiment/epbench-wave1-provider-smoke/summary.json`
- `artifacts/experiment/epbench-wave1-provider-smoke/summary.md`
- `artifacts/experiment/epbench-wave1-provider-smoke/question-*/summary.json`
- `artifacts/experiment/epbench-wave1-provider-smoke/question-*/stdout.txt`
- `artifacts/experiment/epbench-wave1-provider-smoke/question-*/stderr.txt`

For the broader fixed-coverage provider smoke bundle:

- `artifacts/experiment/epbench-wave2-provider-coverage-smoke/summary.json`
- `artifacts/experiment/epbench-wave2-provider-coverage-smoke/summary.md`
- `artifacts/experiment/epbench-wave2-provider-coverage-smoke/question-*/summary.json`
- `artifacts/experiment/epbench-wave2-provider-coverage-smoke/question-*/stdout.txt`
- `artifacts/experiment/epbench-wave2-provider-coverage-smoke/question-*/stderr.txt`

For the bounded quest-local judge-override score probe:

```bash
python3 experiments/main/epbench_adapter/run_epbench_score_probe.py \
  --clean-output \
  --question-indices 0,50,56,394
```

Durable outputs go to:

- `artifacts/experiment/epbench-wave3-provider-score-probe/summary.json`
- `artifacts/experiment/epbench-wave3-provider-score-probe/summary.md`
- `artifacts/experiment/epbench-wave3-provider-score-probe/question-*/evaluation.json`
- `artifacts/experiment/epbench-wave3-provider-score-probe/question-*/judge_meta.json`

For the no-key temporal question-pack export:

```bash
python3 experiments/main/epbench_adapter/export_epbench_question_pack.py --clean-output
```

Durable outputs go to:

- `artifacts/experiment/epbench-wave1-question-pack/pack_manifest.json`
- `artifacts/experiment/epbench-wave1-question-pack/question_pack_rows.json`
- `artifacts/experiment/epbench-wave1-question-pack/real_task_eval_temporal_full.json`
- `artifacts/experiment/epbench-wave1-question-pack/real_task_eval_temporal_smoke.json`
- `artifacts/experiment/epbench-wave1-question-pack/summary.md`

For the bounded DysonSpherain real-task smoke on the exported pack:

```bash
python3 experiments/main/epbench_adapter/run_epbench_real_task_eval_smoke.py --clean-output
```

Durable outputs go to:

- `artifacts/experiment/epbench-wave1-real-task-smoke/real_task_eval_report.json`
- `artifacts/experiment/epbench-wave1-real-task-smoke/summary.json`
- `artifacts/experiment/epbench-wave1-real-task-smoke/summary.md`
- `artifacts/experiment/epbench-wave1-real-task-smoke/traceback.txt` on failure

## Current smoke contract

The default smoke is a source-and-protocol audit. It confirms:

- the official source root exists
- the official quickstart script is present
- the benchmark download DOI is discoverable from the official README
- whether `.env` and `epbench/data` are already available locally
- whether a single cached short-book question can reach a real answering endpoint cleanly once dependency compatibility is fixed

The current continuation lane for temporal/event-chain evidence is narrower and honest:

- export `latest` and `chronological` cached questions into a quest-local question pack
- run a bounded `real_task_eval` smoke against the official cached short-book text
- treat this as local temporal coverage evidence, not as an official full `EpBench` score

The provider-injected auth probe is also intentionally narrow:

- it only tests whether the quest-local single-question answering call can be redirected to a custom OpenAI-compatible `base_url`
- it does not by itself prove that the full official quickstart or full benchmark run is ready

The broader provider smoke remains bounded and honest as well:

- it expands from 3 questions to 12 fixed cached official questions
- it is designed to cover the main `retrieval_type/get` families rather than to claim a scored full official `EpBench` result

The new judge-override score probe is also explicitly non-official:

- it reuses already-generated provider answers from the fixed coverage bundle
- it overrides the judge model to `gpt-5.4` through `right.codes`
- it exists to test end-to-end scoring mechanics under the current credential set
- it does not replace the official `EpBench` judge policy and does not yield an official comparable score

If both `.env` and `epbench/data` exist, the wrapper can be extended to launch
the official quickstart in a later pass. Until then, this adapter records setup
readiness rather than benchmark performance.
