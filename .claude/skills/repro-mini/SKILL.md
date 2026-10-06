---
name: repro-mini
description: Rerun the mini pipeline in stages and compare the outputs with the results table. Only the owner starts this.
disable-model-invocation: true
---

Reproduce the mini-scale results. This retrains models and rewrites rows in
`results/metrics_comparison.md`, so only run it on the mini state.

1. Confirm the data is mini. Run `restore-mini-state` first if it is not.
   Stop if train is not 2,388 rows.
2. Save a copy of `results/metrics_comparison.md` outside the repo so the new
   rows can be diffed against it.
3. Run `scripts/run_pipeline.sh` in the `traj/` venv. It stops before the
   review app. Report any stage that fails and do not continue past it.
4. Capture the stdout-only analyses so README numbers have a source:
   `trajflow-seed-variance | tee results/seed_variance.txt`,
   `trajflow-moving-subset-analysis | tee results/moving_subset.txt`,
   `trajflow-finetune-regularization-sweep | tee results/regularization_sweep.txt`.
5. Diff the new table against the saved copy and list every changed number.
   Training is seeded but not guaranteed bit-identical, so report the
   differences and never edit the table to match.
6. Run `make numbers` and show which README numbers no longer match.
7. Do not update README numbers without the owner's approval.
