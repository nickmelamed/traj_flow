---
name: restore-mini-state
description: Put the mini-scale data, checkpoints, corrections and flags from backups/mini back into the live directories. Only the owner starts this.
disable-model-invocation: true
---

Restore the mini-scale working state. This overwrites local files that are
not in git, so confirm each step.

1. Show what differs. Print row counts for `data/processed/*.parquet` and
   `backups/mini/processed/*.parquet`, and the row count of both
   `corrections.parquet` files. Mini should be train 2,388, val 701, test
   1,626 and 124 corrections (13 corrected).
2. If the live files are the scale-up run, ask whether to keep a copy first.
   If yes, move them to `backups/scaleup/` instead of deleting them.
3. Ask for a go-ahead, then copy `processed/`, `checkpoints/`, `corrections/`,
   `artifacts/` from `backups/mini/` over the live directories.
4. Re-count and show that the numbers match step 1. Run `python -m pytest -q`.
5. Check that `results/metrics_comparison.md` still equals
   `backups/mini/metrics_comparison.md` apart from the header line that points
   to docs/SPEC.md.
6. Never delete `backups/mini/`.
