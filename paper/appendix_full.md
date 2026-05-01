# Appendix

## Algorithms

See `paper/method_algorithms.tex`.

## Hyperparameters

Use artifact-recorded runtime config and route policy config for formal runs.

## Additional Baselines and Ablations

Formal evidence coverage is artifact-backed: 28 available baseline rows, 8
blocked unavailable-model baseline rows, 36/36 mechanism ablations, 4/4
leave-one-benchmark-out rows, paired delta statistics, and 7/7 matched
efficiency sweeps.

## Failure Case Studies

See `paper/appendix/diagnostic_case_studies.md`.

Recent rejected Phase 5 probes should be cited only as diagnostics. The
channel-tail rescue probe improved a 20-question smoke slice but regressed
candidate_recall@100 on 186-question medium validation, so it remains
default-off.

## Reproducibility

See `paper/appendix/reproducibility_checklist.md`.
