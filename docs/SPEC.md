# TrajFlow spec

Status: DRAFT. Written from the code (as of commit 39cea0f) plus the owner's
interview answers. Lines tagged **[inferred]** come from reading the code and
have not been confirmed. Lines tagged **[confirmed]** come from the owner.
Untagged lines are directly observable in the code.

The section numbers 0–7 follow the original build spec that CLAUDE.md used to
hold, so existing references ("Phase 4", "Phase 6") still resolve.

## Summary [confirmed]

TrajFlow is a portfolio trajectory-prediction pipeline on nuScenes mini.
It has constant-velocity, constant-acceleration and XGBoost baselines, a small
transformer (pretrained on easy scenes, fine-tuned on hard ones) and an LSTM.
A human-in-the-loop (HITL) loop flags the transformer's uncertain predictions,
lets a reviewer correct them in Streamlit, and feeds them into a second
fine-tune, with a no-corrections control. The project's value is honestly
measured results, including negative ones.

## Rules that must not be broken

1. **No scene leakage** [confirmed]. Train, val and test are assigned by scene
   name. No scene contributes samples to more than one split. TEST is the
   official nuScenes `mini_val` list (`build_scene_splits` in
   `data/preprocess.py`).
2. **Corrections never touch test** [confirmed, and code and README agree it is TRAIN-only]. HITL flagging and review run on
   hard-scene TRAIN examples only (`hitl/flag_uncertain.py`). Corrected labels
   must never appear in any split used for final evaluation.
3. **Report as measured** [inferred from CLAUDE.md "Do NOT" section and README].
   Never round, cherry-pick, or drop a metric because a model lost. Negative
   results stay in the tables. `metrics_comparison.md` holds unrounded values
   formatted to 4 decimals.
4. **Same test set for every comparison** [inferred]. Baselines, pretrained,
   fine-tuned-v1/v2, control, LSTM and ablations are all evaluated on the same
   test parquet. Not selected by the owner as a hard rule, so confirm.
5. **README numbers match the results table** [confirmed as a claim to
   enforce, see Claims below].
6. **Model selection uses val only** [inferred]. Best checkpoint is chosen by
   val minADE. Test is reported and never used to pick.
7. **No absolute-heading feature in the learned models** [inferred]. Raw
   `heading` is excluded from the transformer context because it leaked scene
   orientation (see `models/transformer.py` and DECISIONS.md).

## Claims the project reports

- **README model-comparison tables match `results/metrics_comparison.md`**
  [confirmed]. The README restates many numbers (minADE, minFDE, miss rate,
  per-difficulty rows, plus bootstrap CI and seed-variance tables).
- **Easy/hard split counts and parameter counts** [confirmed]. For example
  "108k-parameter transformer", "41k-parameter LSTM", "124 of 1,238 flagged",
  "13 labels changed", "63/1626 moving test examples".
- Not selected as claims to enforce: bootstrap CIs and seed variance as a
  separate check, figures matching tables. These are still reported in the
  README, so treat as [inferred] sources of drift.

## Fast checks [confirmed]

- Tests: `python -m pytest -q` (32 tests, about 7s, all on synthetic data).
- Lint: ruff (to be added as a dev dependency, needs owner approval).
- Types: mypy (to be added as a dev dependency, likely noisy on torch code).
- Style: `python3 scripts/agent/check_style.py`.

## Phase 0. Environment and scaffold

Installable package `trajflow` under `src/trajflow/` (PEP 621
`pyproject.toml`, pinned dependencies, Python >= 3.10). 21 `trajflow-*`
console scripts. Paths come from `trajflow/paths.py`. CI runs pytest only.

## Phase 1. Data acquisition and preprocessing

- `trajflow-download` checks the nuScenes setup. Registration and licence
  click-through are manual.
- `trajflow-preprocess` (`data/preprocess.py`) emits one row per
  (vehicle instance, sample) with a full 6 s future. Schema is generated into
  `data/SCHEMA.md`.
- Horizon: 2 s past (4 steps), 6 s future (12 steps), both at 2 Hz, agent frame.
- Features: velocity, acceleration, heading, heading-change rate, distance and
  relative heading to the 3 nearest agents within 60 m (any category).
- Difficulty: "hard" if near an `is_intersection` road segment (within 4 m) or
  at least 5 agents within 10 m. Otherwise "easy".
- Splits: mini_val is TEST, the last 2 scenes (alphabetical) of mini_train are
  VAL, the rest is TRAIN. For other versions the same carving applies to the
  official train/val lists, with `--val-scenes-from-train` and `--max-scenes`.
- Output: `data/processed/{train,val,test}.parquet` (gitignored, regenerable).

Scope: the original spec said mini only. The code now supports trainval
(`--version v1.0-trainval`) and `data/nuscenes/v1.0-trainval` metadata is
present locally. [inferred] The scale-up is infrastructure only. Reported
results are still mini. Confirm.

## Phase 2. Classical baselines

Constant velocity, constant acceleration (double finite difference of the last
two past positions), and XGBoost (`MultiOutputRegressor` over `XGBRegressor`,
24-dim output) on the engineered features, trained on the full train split.
Metrics (`evaluation/metrics.py`): minADE, minFDE, miss rate at 2 m (miss =
min over K of final-point error above 2 m). Single-trajectory models are K=1.

## Phase 3. Transformer pretrain

`models/transformer.py`: self-attention encoder over the past, MLP over
context features, then a one-layer attention block over 6 learned mode tokens.
Outputs K=6 trajectories and mode logits. Loss: winner-take-all min-of-K ADE
plus cross-entropy on the winning mode. Missing values are zero-filled with
validity flags. Trained 150 epochs on the easy split (LR 1e-3, batch 64,
seed 0). The best epoch by val minADE is saved to `checkpoints/pretrained.pt`.

## Phase 4. Fine-tune on hard scenes

`models/finetune.py` starts from the pretrained checkpoint, trains on the hard
split with a lower LR and fewer epochs, giving `fine-tuned-v1`. Compared on the
hard test subset. [inferred] The README reports that this round regressed
versus pretrained (see "Can regularization fix the round-1 regression?").
`finetune_regularization_sweep.py` probes weight decay and dropout.

## Phase 5. HITL flagging and review

- `hitl/flag_uncertain.py` scores hard TRAIN examples by 0.5 times the
  rank of mode-endpoint spread plus 0.5 times the rank of XGBoost-vs-
  transformer endpoint divergence, and flags the top 10% (about 124 of 1,238).
  The original spec said "each test prediction". The code deliberately uses
  TRAIN to avoid leakage (rule 2).
- `hitl/review_app.py` (Streamlit): shows history, nearby lanes, ground truth,
  the 6 modes and the XGBoost prediction. The reviewer accepts, corrects via 3
  adjustable waypoints, or tags a failure mode. Output goes to
  `corrections/` (gitignored).

## Phase 6. Fine-tune round 2

`models/finetune_round2.py` overwrites future labels for reviewed examples,
continues from `finetuned_v1.pt` for 60 epochs at LR 2e-4, and re-evaluates on
the same test set. `--ablation-no-corrections` reruns the identical recipe on
uncorrected labels, so the effect of the corrections is separable from extra
training. Rows are logged as v2 and v2-control.

## Additional experiments (added after the original spec)

- LSTM autoregressive encoder-decoder (`models/lstm.py`), trained on the full
  split, and run through the transformer's pretrain/fine-tune/HITL lineage.
- Transformer-AR hybrid (`models/transformer_ar.py`) isolating decoder style.
- Full-split transformer (`train_transformer_full.py`) as the training-regime
  control.
- Moving-vehicle subset analysis, bootstrap CIs, seed variance
  (`evaluation/`).
- Regularization sweep for round-1 fine-tuning.

## Phase 7. Visualization and README

`viz/scene_overlay.py` writes figures to `results/figures/`. `viz/dashboard.py`
is a Streamlit results dashboard. The README holds the narrative, tables and
limitations.

## Results logging

`evaluation/evaluate.py::log_metrics` rewrites `results/metrics_comparison.md`.
A row is keyed on (phase, model, eval split, difficulty), so rerunning replaces
that row rather than appending a duplicate. `|` and newlines in cells are
sanitized.

## Known gaps and inconsistencies noticed while drafting [inferred]

- `finetune_round2.py` docstring says "~100 accepted, ~24 corrected". The mini
  corrections file (`backups/mini/corrections/`), the README and the results
  table all say 124 reviewed: 111 accepted, 13 corrected, 1,114 unreviewed of
  1,238. The docstring is stale and will be fixed in the style cleanup.
- The working tree is not in the mini state the README describes. Local
  `data/processed/`, `artifacts/flagged.parquet` and `corrections/` come from a
  scale-up run (train 16,878 rows over 72 scenes with 6,778 hard, 679 flagged, and 154
  corrections of which 17 changed a label). The `checkpoints/` are still the mini ones
  (dated 2026-07-08). Rerunning evaluation or round 2 here would mix the two.
  The mini versions are in `backups/mini/`. `results/metrics_comparison.md` is
  tracked and identical to the mini backup.
- `git stash@{0}` ("stashing any uncommitted changes before repo cleanup")
  holds unreviewed edits to README, `results/metrics_comparison.md`,
  `preprocess.py`, `paths.py`, `review_app.py`, the dashboard, scene overlay and
  `tests/test_data_scaleup.py`. It has not been applied or inspected beyond its
  file list.
- `evaluate.py` PREAMBLE points to `CLAUDE.md` for the phase plan. That plan
  now lives here.
- `corrections/` is gitignored, so the HITL labels behind v2 are not
  reproducible from the repo.
- Checkpoints, processed data and nuScenes data have no checksums.
- Several modules need torch and xgboost imported in a specific order
  (OpenMP conflict on macOS).
