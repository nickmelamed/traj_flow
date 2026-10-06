---
name: hitl-review-pass
description: Flag uncertain hard-train examples, run the review app, then fine-tune round 2 and its control. Only the owner starts this.
disable-model-invocation: true
---

Run one human-in-the-loop pass. The review step is interactive, so the owner
does it.

1. Confirm the state is mini (see `restore-mini-state`) and that
   `checkpoints/finetuned_v1.pt` exists.
2. Back up the current `corrections/` and `artifacts/flagged.parquet` to
   `backups/` before anything is overwritten.
3. Run `trajflow-flag-uncertain`. Check that it scored hard TRAIN examples
   and flagged about 10%. Flagging test examples would break the evaluation.
4. Tell the owner to run `trajflow-review-app` and complete the pass. Wait.
5. Check the corrections file. Report reviewed, accepted and corrected counts,
   and confirm every reviewed key is in the train split.
6. Run `trajflow-finetune-round2`, then `trajflow-finetune-round2
   --ablation-no-corrections`, then the two LSTM round-2 scripts.
7. Show the test/hard rows for v1, v2 and the control. State plainly whether
   the corrections helped, including when the answer is no.
