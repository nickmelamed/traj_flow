# Trajectory Prediction Pipeline — Build Spec for Claude Code

## Context & Goal
Portfolio project demonstrating three things in one pipeline: a classical ML
baseline, a fine-tuned deep learning model, and a human-in-the-loop active
learning loop — applied to autonomous vehicle trajectory prediction. This
targets autonomy/planning & prediction engineering interviews. Priority:
get a working end-to-end pipeline fast, then iterate. Ship something
complete and honestly-measured over something perfect and half-finished.

## Definition of Done
All 7 phases below complete, with a single `results/metrics_comparison.md`
table showing baseline → pretrained → fine-tuned-v1 → fine-tuned-v2 (post-HITL),
plus a README that reads as a coherent portfolio piece to a technical reviewer
who has never seen the project.

## Ground Rules
- Use nuScenes **mini** only for this pass. Do NOT download full nuScenes
  (100+ GB) — that's a scale-up step, documented separately, not part of v1.
- Favor small, CPU-friendly models. This is a portfolio piece, not SOTA research.
- Commit to git after each phase passes its acceptance check.
- If nuscenes-devkit API details are unclear, check the installed package's
  docstrings/source rather than guessing at function signatures.
- Log every metric to `results/metrics_comparison.md`, even if a phase
  underperforms. Honest negative results are part of the portfolio value —
  do not round or cherry-pick numbers to look better.
- nuScenes requires a free account + license click-through on nuscenes.org
  that can't be scripted. Flag this clearly in Phase 1 and pause for the
  human to complete it rather than trying to work around it.

## Tech Stack
- Python 3.10+
- PyTorch
- nuscenes-devkit
- xgboost
- streamlit
- pandas, numpy
- matplotlib or plotly (visualization)

## Repo Structure
```
trajectory-prediction/
├── README.md
├── pyproject.toml (or requirements.txt)
├── data/
│   ├── download.py        # nuScenes mini fetch/instructions
│   └── preprocess.py      # feature engineering + train/val/test split
├── models/
│   ├── baseline_cv.py     # constant velocity model
│   ├── baseline_xgb.py    # XGBoost on engineered features
│   ├── transformer.py     # trajectory transformer architecture
│   ├── train_pretrain.py  # pretrain on "easy" scenes
│   └── finetune.py        # fine-tune on "hard" / HITL-corrected data
├── evaluation/
│   ├── metrics.py         # minADE, minFDE, miss rate
│   └── evaluate.py        # run eval, append to comparison table
├── hitl/
│   ├── flag_uncertain.py  # disagreement-based flagging
│   └── review_app.py      # Streamlit reviewer UI
├── viz/
│   └── scene_overlay.py   # predicted vs actual trajectory plots
└── results/
    ├── metrics_comparison.md
    └── figures/
```

## Phase 0 — Environment & Scaffold
1. Init git repo, create the structure above.
2. Create `pyproject.toml`/`requirements.txt` pinning the libs above.
3. Stub `README.md` with a project description (fill in fully in Phase 7).

**Acceptance:** `pip install -r requirements.txt` runs clean; structure matches spec.

## Phase 1 — Data Acquisition & Preprocessing
1. `data/download.py`: fetch nuScenes **mini** (v1.0-mini). Registration is
   manual — pause and tell the human what to do if the devkit needs an
   account/license click-through.
2. Use nuscenes-devkit's `PredictHelper` to extract, per agent instance/sample
   token: 2s of past history, 6s of future ground truth (the standard
   nuScenes prediction-challenge horizon).
3. Feature engineer: velocity, acceleration, heading, heading-change rate,
   distance & relative heading to the 3 nearest neighboring agents.
4. Classify each instance as **"easy"** (low neighbor density, no intersection
   nearby) or **"hard"** (high density or near an intersection per the map API).
   This split powers the pretrain/fine-tune structure in Phases 3–4.
5. Save processed data as parquet/pickle with train/val/test split — respect
   nuScenes' official scene-level split to avoid leakage.

**Acceptance:** processed dataset exists with documented schema; easy/hard split
counts logged.

## Phase 2 — Classical Baselines
1. Constant velocity model: extrapolate the final observed velocity/heading
   over the 6s horizon.
2. XGBoost regressor: predict the future waypoint sequence from engineered
   features.
3. Evaluate both with standard metrics: **minADE**, **minFDE**, **Miss Rate @ 2m**.
4. Log to `results/metrics_comparison.md`.

**Acceptance:** baseline table exists comparing both models.

## Phase 3 — Trajectory Transformer: Pretrain
1. Build a compact encoder-decoder / attention model (a few million params —
   don't reach for a large architecture) taking agent history + neighbor
   context, outputting K=6 candidate futures with mode probabilities.
2. Loss: min-of-K displacement loss + cross-entropy on mode probability.
3. Train on the **"easy"** split only.
4. Evaluate on held-out val set, log metrics.

**Acceptance:** checkpoint saved; metrics logged (report honestly even if not
yet better than XGBoost — that's expected pre-fine-tune).

## Phase 4 — Fine-tune on Hard Scenes
1. Load the pretrained checkpoint, fine-tune on the **"hard"** split (lower LR,
   fewer epochs).
2. Evaluate on the hard-scene test set; compare pretrained-only vs fine-tuned.
3. Log to comparison table.

**Acceptance:** table now shows baseline → pretrained → fine-tuned-v1.

## Phase 5 — HITL Flagging + Review Tool
1. `flag_uncertain.py`: score each test prediction by (a) variance across the
   K modes' endpoints and (b) divergence between XGBoost and transformer
   predictions. Flag the top ~10% as "needs review."
2. `hitl/review_app.py` (Streamlit):
   - Show the scene (agent history, map context if available).
   - Overlay the model's top predictions.
   - Let the reviewer accept, correct (adjust the trajectory), or tag a
     failure mode (occlusion, aggressive merge, sensor noise, map ambiguity).
   - Persist corrections as new labeled examples under `corrections/`.

**Acceptance:** app runs via `streamlit run hitl/review_app.py`; one full
review pass completed with corrections saved.

## Phase 6 — Fine-tune Round 2 (post-HITL)
1. Merge HITL-corrected examples into the hard-scene training set.
2. Fine-tune again from the round-1 checkpoint.
3. Re-evaluate on the *same* held-out test set for a fair comparison.
4. Update the table: baseline → pretrained → fine-tuned-v1 → fine-tuned-v2.

**Acceptance:** full 4-row table complete, reported honestly either direction.

## Phase 7 — Visualization & README
1. `viz/scene_overlay.py`: for a handful of representative scenes (include at
   least one "hard" scene showing visible improvement), plot history, ground
   truth, baseline prediction, and final fine-tuned prediction together.
2. Write the full README:
   - Problem statement & motivation
   - Methodology (baseline → pretrain → fine-tune → HITL → fine-tune round 2)
   - Metrics table
   - 2–3 embedded overlay figures
   - Limitations & what you'd do with more compute/data

**Acceptance:** README reads coherently to a reviewer with no prior context.

## Do NOT
- Download full nuScenes/Argoverse2/Waymo in this pass.
- Build distributed training — single-machine scope only.
- Skip logging a metric because a phase underperformed.
- Fabricate or round results to look better.