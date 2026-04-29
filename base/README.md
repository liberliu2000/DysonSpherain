# Sphere Memory CLI

本项目实现一个球壳式本地 CLI 记忆系统，采用双库存储：

- SQLite：节点、chunk 元数据、关系边、zone 索引、任务日志、文件摄取状态
- Chroma：`raw_chunks` 与 `memory_objects` 两类向量索引

当前主干按“三层统一记忆算法”组织：

```text
Query
  -> Evidence retrieval
  -> Structured completion
  -> Cognitive augmentation
  -> Context assembly
  -> Response / action
  -> Writeback
```

- Layer 1 `Evidence Memory`：raw chunk / local window / macro summary
- Layer 2 `Structured Memory`：preference / state_update / solution_card
- Layer 3 `Cognitive Memory`：graph activation / experience / creative reflection

## 当前能力

- 五级索引：`shell -> sector -> zone -> cell -> molecular`
- 自动文件摄取：Markdown / code / logs / PDFs / txt / json / yaml / toml / ini / cfg
- evidence-first 检索：dense + sparse + object retrieval，先找证据再做增强
- 多表示共存：micro chunk / local window / macro summary / structured objects
- 结构对象抽取：`preference` / `state_update` / `solution_card`
- 认知增强后置：evidence completion 后再接 activation / reflection
- 后台维护：批量重嵌入、冷数据压缩、边衰减、zone 裂变
- 增量摄取：文件指纹跟踪、只重建变化文件
- 轮询 watch：持续扫描目录变化并自动入库
- 真实任务评测：支持本地任务集评测，不只依赖公开 benchmark

## 安装

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m sphere_cli.cli init
```

## 关于本地 embedding 与 rerank

默认设计为：

- 优先使用本地 `sentence-transformers` 模型
- 若本地模型不可用，自动降级到内置 `local_hash` embedding
- cross-encoder 同样为可插拔；模型不可用时，`hybrid` 会自动退回规则 rerank

推荐本地准备：

- embedding: `sentence-transformers/all-MiniLM-L6-v2`
- cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## 常用命令

初始化与状态：

```bash
python -m sphere_cli.cli init
python -m sphere_cli.cli status
```

手动写入：

```bash
python -m sphere_cli.cli remember add \
  --shell 1 \
  --sector project \
  --zone sequencer_log_platform \
  --cell parallel_parsing \
  --molecular-type decision \
  --summary "并行解析阶段瓶颈主要是任务切分和写竞争" \
  --content "parsing files parallel 阶段 CPU 利用率低，且 SQLite 存在写锁竞争"
```

批量摄取：

```bash
python -m sphere_cli.cli ingest path ./docs --zone raw_docs
```

增量同步：

```bash
python -m sphere_cli.cli ingest sync ./docs --zone raw_docs
```

目录监听：

```bash
python -m sphere_cli.cli ingest watch ./docs --zone raw_docs --poll-seconds 5 --max-rounds 0
```

检索：

```bash
python -m sphere_cli.cli memory find "sqlite write contention during parallel parsing" --rerank-mode hybrid
python -m sphere_cli.cli memory raw-find "parallel parsing sqlite"
python -m sphere_cli.cli memory trace --task "设计本地CLI记忆系统" --rerank-mode hybrid
```

后台维护：

```bash
python -m sphere_cli.cli maint reembed --batch-size 200
python -m sphere_cli.cli maint compress --cold-days 30 --access-threshold 1 --limit 100
python -m sphere_cli.cli maint decay-edges --factor 0.97 --floor 0.05
python -m sphere_cli.cli maint split-zones --threshold 40 --group-size 20
python -m sphere_cli.cli maint rebuild-representations --limit 200
```

真实任务评测：

```bash
python -m sphere_cli.cli eval run --dataset evaluation/real_tasks_sample.json
```

## 目录结构

```text
sphere_memory_cli/
├─ data/
│  ├─ memory.db
│  ├─ raw/
│  ├─ cache/
│  ├─ exports/
│  └─ vector_db/
├─ sample_docs/
├─ sphere_cli/
│  ├─ cli.py
│  ├─ embedding.py
│  ├─ vector_store.py
│  ├─ ingestion.py
│  ├─ reranker.py
│  ├─ background_tasks.py
│  └─ ...
└─ requirements.txt
```
## Prism Propagation Creative Augmentation

`PrismPropagationEngine` is a secondary augmentation layer that runs only after
the primary evidence-first retrieval and structured completion steps.

- `creative_mode = off` keeps the original cognitive chain behavior unchanged.
- `creative_mode = conservative` keeps only light semantic/temporal support paths for exact tasks.
- `creative_mode = exploratory` adds the full bounded prism propagation layer on top evidence seeds.
- Output is explicitly separated into `primary_evidence`,
  `supporting_context`, `creative_reflections`, and `alternative_paths`.

The engine is intentionally bounded:

- ANN shortlist first, then local neighborhood propagation
- storage-level local adjacency queries instead of generic full graph fetches
- path-level scoring instead of node-only scoring
- adjacency cache, beam-aware creative adjacency cache, local neighborhood cache, and path score cache
- configurable beam width, hop limit, per-hop neighbor cap, and operator gating
- factual and exact-evidence tasks use operator gates plus output gates
- alternative path selection uses greedy MMR reranking for diversity

## Creative Config

The CLI/runtime reads these overrides from environment variables:

```powershell
$env:SPHERE_CREATIVE_MODE = "exploratory"   # or "conservative" / "off"
$env:SPHERE_CREATIVE_BEAM_WIDTH = "6"
$env:SPHERE_CREATIVE_MAX_HOPS = "2"
$env:SPHERE_CREATIVE_NEIGHBORS_PER_HOP = "4"
$env:SPHERE_CREATIVE_ENABLE_ANALOGY = "1"
$env:SPHERE_CREATIVE_ENABLE_CONTRAST = "1"
$env:SPHERE_CREATIVE_ENABLE_TRANSFER = "1"
$env:SPHERE_CREATIVE_ENABLE_TEMPORAL = "1"
$env:SPHERE_CREATIVE_ENABLE_COMPOSITION = "1"
$env:SPHERE_CREATIVE_NOVELTY_WEIGHT = "0.20"
$env:SPHERE_CREATIVE_SUPPORT_WEIGHT = "0.24"
$env:SPHERE_CREATIVE_DIVERSITY_WEIGHT = "0.16"
$env:SPHERE_CREATIVE_CONFLICT_PENALTY = "0.22"
$env:SPHERE_CREATIVE_REFLECTION_GAIN = "0.18"
$env:SPHERE_CREATIVE_MAX_OUTPUT_PATHS = "4"
```

You can inspect the active config with:

```bash
python -m sphere_cli.cli status
```

`status` now reports the normalized creative mode and the active propagation limits.

And trace a creative-enabled run with:

```bash
python -m sphere_cli.cli memory trace --task "Redesign retrieval with stronger local memory reuse" --task-type design
python -m sphere_cli.cli creative reflect --task "Explore alternative retrieval refactors" --task-type creative
```

## Retrieval And Compression Upgrades

The repo now includes a conservative optimization layer on top of the existing
evidence-first chain:

```text
query
  -> lightweight task router
  -> object/profile/state shortcut
  -> temporal proxy prefilter
  -> coarse retrieval (raw + sparse + proxy)
  -> conditional rerank
  -> seed dedup / clustering
  -> structured completion
  -> bounded prism sidecar
  -> context compression
```

Key properties:

- `primary_evidence` still comes from grounded evidence chunks.
- creative / prism output is still isolated from `primary_evidence`.
- preference / persona / state queries can short-circuit through structured memory.
- temporal queries can use proxy-based prefiltering before heavier retrieval.
- writeback dedups exact duplicates, compresses near-duplicates, and stores deltas.
- multi-resolution representations are stored as raw chunks, structured objects, and retrieval proxies.

## Optimization Env Flags

## Overnight Upgrade Notes

This repository now includes a resumable overnight upgrade scaffold in the repo root:

- `codex_overnight_plan.md`
- `codex_progress.json`
- `codex_change_log.md`
- `codex_next_actions.md`
- `codex_validation_log.md`

If a long run is interrupted, read those files first instead of relying on chat history.

## Workspace-Aware CLI

The CLI now supports workspace-aware execution context without breaking the older benchmark-oriented flow.

Global options:

```bash
python -m sphere_cli.cli --workspace overnight --project alpha --session sess-42 --mode balanced --scope-order project,session,global status
```

New context fields are written through storage and retrieval metadata:

- `scope`
- `workspace`
- `project`
- `session_id`
- `source_type`
- `source_ref`
- `extraction_method`
- `confidence`
- `verification_status`
- `updated_at`

Default write behavior now prefers project scope when a project is active, while still preserving explicit `--scope global` or `--scope session:<id>` overrides.

## New Commands

Recall and explain:

```bash
python -m sphere_cli.cli recall "What is still pending?" --task-type qa --explain
python -m sphere_cli.cli memory explain --last-recall
python -m sphere_cli.cli memory explain --node-id mem_xxx
python -m sphere_cli.cli memory-list --project alpha
python -m sphere_cli.cli ask "Summarize the current project state" --task-type qa
```

Artifacts:

```bash
python -m sphere_cli.cli --project alpha artifact-add ./docs/plan.md --artifact-type markdown --summary "Current plan draft"
```

Open loops:

```bash
python -m sphere_cli.cli --project alpha open-loop-add "Finish retrieval trace polish" --details "Add richer recall trace output"
python -m sphere_cli.cli --project alpha open-loop-list
python -m sphere_cli.cli open-loop-update loop_xxx --status closed
```

Workspace inventory:

```bash
python -m sphere_cli.cli workspace-list
```

Project-management notes now also auto-extract lightweight structured objects during normal `remember add` / ingest flows:

- `project`
- `goal`
- `artifact` when a file path like `docs/plan.md` appears in the note
- `open_loop` for `TODO`, `blocked on`, `next step`, and similar pending-work phrasing

If an extracted artifact path resolves to a real local file, writeback now also auto-registers it in `artifact_registry` and links the extracted memory object back to that registry record.

Blocked/pending task prompts with wording like `currently blocked`, `right now`, or `remaining task` are normalized onto the exact/open-loop path instead of drifting into temporal or persona-style routing.

## Narrative Guardrail Upgrades

The retrieval chain now includes conservative, configurable hooks for:

- workspace/project/session scope prioritization
- expanded narrative anchors in identity features
- optional wrong-domain / wrong-role-target / wrong-subtheme / generic-topic penalties
- three-sentence segment spans for narrative-heavy cases
- representative selection that considers specificity and genericness

These hooks are intentionally lightweight and remain additive to the existing evidence-first path.

These new environment variables are read by `AppConfig.from_env`:

```powershell
$env:SPHERE_ENABLE_TASK_ROUTER = "1"
$env:SPHERE_ENABLE_OBJECT_SHORTCUT = "1"
$env:SPHERE_ENABLE_TEMPORAL_PREFILTER = "1"
$env:SPHERE_RETRIEVAL_TOPK_COARSE = "24"
$env:SPHERE_RETRIEVAL_TOPK_FINE = "8"
$env:SPHERE_ENABLE_SEED_CLUSTERING = "1"
$env:SPHERE_ENABLE_SEMANTIC_DEDUP = "1"
$env:SPHERE_ENABLE_CONDITIONAL_RERANK = "1"

$env:SPHERE_ENABLE_INGEST_COMPRESSION = "1"
$env:SPHERE_ENABLE_CONTENT_HASH_DEDUP = "1"
$env:SPHERE_ENABLE_DELTA_MEMORY_WRITER = "1"
$env:SPHERE_ENABLE_STRUCTURED_COMPRESSION = "1"
$env:SPHERE_ENABLE_MULTIRES_SUMMARIES = "1"
$env:SPHERE_ENABLE_RETRIEVAL_PROXY_INDEX = "1"
$env:SPHERE_ENABLE_CONTEXT_COMPRESSOR = "1"

$env:SPHERE_ENABLE_RETRIEVAL_CACHE = "1"
$env:SPHERE_ENABLE_COMPLETION_CACHE = "1"
$env:SPHERE_ENABLE_PROFILE_SNAPSHOT_CACHE = "1"
$env:SPHERE_ENABLE_PRISM_PATH_CACHE = "1"
$env:SPHERE_ENABLE_BENCHMARK_ROUTE_TUNING = "1"
$env:SPHERE_ENABLE_IDENTITY_AWARE_RERANK = "1"
$env:SPHERE_ENABLE_SEGMENT_RERANK = "1"
$env:SPHERE_ENABLE_CONFUSING_CLUSTER_RERANK = "1"
$env:SPHERE_ENABLE_OBJECT_SHORTCUT_CACHE = "1"
$env:SPHERE_ENABLE_IDENTITY_FEATURE_CACHE = "1"
$env:SPHERE_ENABLE_SEGMENT_FEATURE_CACHE = "1"
$env:SPHERE_ENABLE_CONFUSING_CLUSTER_CACHE = "1"
$env:SPHERE_ENABLE_ARTIFACT_REGISTRY = "1"
$env:SPHERE_ENABLE_NOTE_ARTIFACT_AUTO_REGISTER = "1"
$env:SPHERE_ENABLE_OPEN_LOOP_TRACKING = "1"
$env:SPHERE_SEGMENT_RERANK_TOPK_DEFAULT = "12"
$env:SPHERE_CONFUSING_CLUSTER_TOPK_DEFAULT = "20"
```

`python -m sphere_cli.cli status` now reports:

- retrieval feature flags
- compression feature flags
- cache feature flags and cache entry counts
- proxy representation counts
- creative mode bounds

## Evaluation Signals

`real_task_eval.py` now records additional engineering metrics per case:

- route type
- benchmark route profile
- retrieval and completion cache hits
- object shortcut hit rate
- identity-aware rerank trigger count
- segment rerank trigger count
- confusing cluster count and average size
- temporal prefilter hit rate
- coarse / fine / rerank / completion / prism / assemble stage timings
- identity / confusing-cluster / segment-rerank stage timings
- seed compression counts
- context token deltas
- factual contamination rate for final evidence

This makes it easier to compare:

- baseline vs optimized retrieval
- creative mode `off` / `conservative` / `exploratory`
- compression enabled vs disabled

## Tests

Run the minimal regression suite with:

```bash
python -m unittest discover -s tests -v
```

The repo also includes a small fixture-driven CloneMem-style validation slice at `tests/fixtures/clonemem_like_slice.json`, covered by `tests.test_memory_optimizations.MemoryOptimizationTests.test_clonemem_fixture_slice_prefers_correct_narrative_anchor`.


## Hardening patch notes

The 2026-04-24 package adds:

- `SPHERE_EMBEDDING_FAIL_FAST=1` to prevent silent SentenceTransformer -> local hash fallback during benchmark runs.
- `SPHERE_VECTOR_BACKEND=json` for deterministic local smoke tests without Chroma.
- `SPHERE_ENABLE_LIGHTWEIGHT_EDGE_WRITEBACK=1` so graph edges are written by default.
- `dysonspherain ask`, `doctor`, `config`, `memory export`, `memory backup`, `memory forget`, and `benchmark smoke-all`.
- `.docx`, `.pptx`, `.xlsx`, and `.html` ingestion fallbacks.
- `tests/test_guardrails.py` for embedding, vector fallback, answer generation, and edge writeback guardrails.

See the top-level `UPGRADE_NOTES_2026-04-24.md` for the verified test results and remaining full-benchmark requirements.
