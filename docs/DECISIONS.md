# Design decisions

DRAFT. Each entry was inferred from comments and code, not from you. Edit them
into your own words and delete any that are wrong.

## D1. Scene-level splits, official val list as test
Train, val and test are assigned by scene name. TEST is the untouched official
`mini_val`, and VAL is carved from the tail of `mini_train`. Samples from one
scene are highly correlated, so a sample-level split would leak.

## D2. Hard/easy thresholds tuned empirically
A 15–20 m intersection radius labelled about 97% of examples as hard, which
makes the pretrain/fine-tune split meaningless. The code uses 4 m for
intersections and at least 5 neighbors within 10 m, which gives roughly a
55/45 hard/easy split. The thresholds were chosen by looking at the data, so
"hard" is a project-specific definition, not a nuScenes one.

## D3. Pretrain on easy, fine-tune on hard
This gives the fine-tune a purpose and a measurable out-of-distribution
question (how does an easy-only model do on hard scenes?). It also leaves very
few hard scenes to fine-tune on, which the README names as a source of
overfitting.

## D4. Small models, CPU-friendly
The transformer has about 108k parameters and the LSTM about 41k. Mini has
about 8 training scenes, so larger models would just memorise.

## D5. Absolute heading removed from learned models
Global-frame yaw is tied to each scene's road orientation and differs between
disjoint scenes, so the models could learn train-scene orientation. All other
features are agent-frame. Removing it made the transformer slightly worse
(test minADE 0.602 with it), and XGBoost stopped benefiting. It was kept out
for frame-invariance, and the result is documented rather than hidden.

## D6. Zero-fill plus validity flags for missing values
Short histories and fewer than 3 neighbors are filled with 0 and paired with an
explicit valid flag, so "0" and "unknown" are distinguishable.

## D7. K=6 modes, winner-take-all loss
The model outputs 6 futures with probabilities. The loss is min-of-K ADE plus
cross-entropy on the winning mode, the standard multi-hypothesis recipe.
Metrics take the min over K, so a K=1 baseline and a K=6 model are compared on
the same formula.

## D8. HITL flags TRAIN, not TEST
The original spec said to flag test predictions. Correcting test labels and
then evaluating on them would invalidate the before/after comparison, so
flagging runs on hard TRAIN examples. This is the rule that protects the
headline HITL result.

## D9. Uncertainty = rank average of two signals
Mode-endpoint spread (the model disagreeing with itself) and XGBoost vs
transformer endpoint divergence (two model families disagreeing), each
rank-normalised and averaged, top 10% flagged. Rank normalisation avoids
having to scale metres against metres-squared.

## D10. No-corrections control for round 2
Round 2 changes two things at once: corrected labels and 60 more epochs. The
`--ablation-no-corrections` flag reruns the identical recipe on uncorrected
labels so the effect of the corrections is isolated.

## D11. Model selection by best val minADE
Each training script keeps the best-val checkpoint, and test is only
reported. Round 2 starts with the v1 val score as the bar, so it can only keep
a checkpoint that beats v1 on val.

## D12. Log every run, replace on rerun
`log_metrics` keys rows on (phase, model, split, difficulty) and replaces on
rerun, so the table can't accumulate stale duplicates. Values are not rounded
beyond the 4-decimal display.

## D13. Extra experiments instead of removing losing models
The LSTM, the AR-decoder hybrid and the full-split transformer were added to
explain why a "smarter" transformer loses to simpler models, not to hide the
result. The 2x2 encoder/decoder table is the explanation structure.

## D14. Package layout under `src/trajflow/` with console scripts
The code was restructured from the original flat layout so it is installable
and reproducible. Paths are centralised in `paths.py`.

## D15. Import order and OpenMP environment hacks
`KMP_DUPLICATE_LIB_OK` and `OMP_NUM_THREADS` are set before torch and xgboost
are imported, and xgboost must be imported first, because both bundle
libomp and crash or hang on this macOS setup. This is a workaround.
