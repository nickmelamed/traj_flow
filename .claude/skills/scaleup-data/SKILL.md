---
name: scaleup-data
description: Download and preprocess a larger nuScenes split, starting with a capped pilot. Only the owner starts this.
disable-model-invocation: true
---

Prepare scale-up data without damaging the mini state. `trajflow-preprocess`
overwrites `data/processed/` and `data/SCHEMA.md`, so protect them first.

1. Check that `backups/mini/` exists and matches the current mini state. If
   the live processed data is already scale-up, stop and ask.
2. Ask the owner to finish the nuScenes account and licence click-through if
   the data is missing. Do not script around it. Never download full
   nuScenes unless the owner says so in this session.
3. Run `trajflow-preprocess --version v1.0-trainval --max-scenes 100` as a
   pilot, with `--out data/processed_scaleup` so the live splits are not
   overwritten. Raise `--val-scenes-from-train` as the preprocess docstring
   suggests when the scene count grows. The script also rewrites
   `data/SCHEMA.md`, so restore it with `git checkout data/SCHEMA.md` if the
   mini schema should stay.
4. Print per-split row counts, scene counts and easy/hard counts. Check that
   no scene appears in two splits.
5. Remind the owner that checkpoints, corrections and flags are still mini
   and must not be evaluated against this data. Reported README results are
   mini only. Do not log scale-up rows into `results/metrics_comparison.md`
   without a separate table.
