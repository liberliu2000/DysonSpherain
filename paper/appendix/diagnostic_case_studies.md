# Diagnostic Case Studies

Cases are referenced from diagnostic JSONL artifacts. No unsupported examples are invented.

## LoCoMo ordering repair case

- source: `reports/diagnostics/locomo_ordering_failures.jsonl`
- first_record: `{"artifact_path": "/Users/yanbo/DysonSpherain/BenchmarkResult/20260427_other_full_rerun_v1/locomo/locomo/chunk_00/reports/diagnostics/locomo_candidate_recall.json", "benchmark_name": "locomo", "candidate_recall@100": 1.0, "channel_stats": {"dense_semantic": {"candidate_count": 29, "gold_hit": true, "gold_rank": 9}, "entity_aware": {"candidate_count": 29, "gold_hit": true, "gold_rank": 3}, "exact_phrase": {"candidate_count": 2, "gold_hit": false, "gold_rank": null}, "lexical_sparse": {"candidate_`

## CloneMem parent-to-segment repair case

- source: `reports/diagnostics/clonemem_parent_hit_segment_miss_examples.jsonl`
- first_record: `{"artifact_path": "/Users/yanbo/DysonSpherain/BenchmarkResult/20260427_clonemem_full_sample_sharded_v1/clonemem/clonemem/chunk_00/reports/diagnostics/clonemem_candidate_recall.json", "benchmark_name": "clonemem", "candidate_recall@100": 0.0, "channel_stats": {"dense_semantic": {"candidate_count": 200, "gold_hit": true, "gold_rank": 136}, "entity_aware": {"candidate_count": 133, "gold_hit": true, "gold_rank": 27}, "exact_phrase": {"candidate_count": 16, "gold_hit": false, "gold_rank": null}, "lex`

## KnowMe profile/entity admission case

- source: `reports/diagnostics/knowme_segment_admission_failures.jsonl`
- first_record: `{"artifact_path": "/Users/yanbo/DysonSpherain/BenchmarkResult/20260427_full_diagnostic_sampling_v2_after_profile_index/knowme/reports/diagnostics/knowme_candidate_recall.json", "benchmark_name": "knowme", "candidate_recall@100": 0.0, "channel_stats": {"dense_semantic": {"candidate_count": 175, "gold_hit": false, "gold_rank": null}, "entity_aware": {"candidate_count": 150, "gold_hit": false, "gold_rank": null}, "exact_phrase": {"candidate_count": 4, "gold_hit": false, "gold_rank": null}, "lexical`

## Remaining failure case

- source: `reports/diagnostics/clonemem_reranker_dropped_gold_examples.jsonl`
- first_record: `{"artifact_path": "/Users/yanbo/DysonSpherain/BenchmarkResult/20260427_clonemem_full_sample_sharded_v1/clonemem/clonemem/chunk_00/reports/diagnostics/clonemem_candidate_recall.json", "benchmark_name": "clonemem", "candidate_recall@100": 0.8, "channel_stats": {"dense_semantic": {"candidate_count": 200, "gold_hit": true, "gold_rank": 1}, "entity_aware": {"candidate_count": 42, "gold_hit": true, "gold_rank": 17}, "exact_phrase": {"candidate_count": 76, "gold_hit": true, "gold_rank": 20}, "lexical_s`
