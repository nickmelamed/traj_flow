"""Flag the most uncertain hard-scene TRAIN predictions for human review.

Uncertainty score combines two signals:
  (a) spread across the transformer's K=6 mode endpoints (the model
      disagreeing with itself)
  (b) divergence between the XGBoost baseline's prediction and the
      transformer's most-likely mode (two different model families
      disagreeing with each other)

Each is rank-normalized to [0, 1] and averaged; the top ~10% highest-
scoring examples are flagged as "needs review" in hitl/review_app.py.

Why TRAIN and not TEST: Phase 6 merges the resulting corrections back
into the hard-scene TRAINING set and then re-evaluates on the *same*
held-out test set for a fair before/after comparison. Flagging (and
therefore correcting) TEST examples would mean training on corrected
test labels and then evaluating on those same instances -- a leak that
would invalidate that comparison. Scoring TRAIN's hard subset instead
targets exactly the population Phase 4 already fine-tunes on, which is
the actual intent of the HITL loop: find the training examples the
model is most uncertain about and get them human-corrected.
"""

import os

# Must be set before torch/xgboost are imported: loading both libraries in
# the same process on this macOS setup (each bundles its own OpenMP runtime)
# deadlocks otherwise -- reordering imports alone (xgboost before torch)
# prevented the segfault but not a subsequent hang. See also review_app.py,
# which has the same constraint.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

# NOTE: xgboost must be imported (directly or via trajflow.models.baseline_xgb)
# before torch in this process -- on this macOS setup, importing torch first
# and then loading/using an xgboost model segfaults (a known OpenMP runtime
# conflict between the two libraries' bundled libomp). Keep this import
# order; see also hitl/review_app.py which has the same constraint.
from trajflow.models.baseline_xgb import MODEL_PATH as XGB_MODEL_PATH
from trajflow.models.baseline_xgb import make_features as xgb_make_features
from trajflow.evaluation.evaluate import filter_difficulty, load_split

import torch
from torch.utils.data import DataLoader

from trajflow.models.transformer import TrajectoryDataset, TrajectoryTransformer
from trajflow.paths import CHECKPOINTS_DIR, FLAGGED_PATH

TRANSFORMER_CHECKPOINT = CHECKPOINTS_DIR / "finetuned_v1.pt"
OUTPUT_PATH = FLAGGED_PATH
TOP_FRACTION = 0.10


@torch.no_grad()
def transformer_predictions(df: pd.DataFrame):
    model = TrajectoryTransformer()
    model.load_state_dict(torch.load(TRANSFORMER_CHECKPOINT))
    model.eval()

    dataset = TrajectoryDataset(df)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_traj, all_logits = [], []
    for past_seq, context, _ in loader:
        traj, logits = model(past_seq, context)
        all_traj.append(traj.numpy())
        all_logits.append(logits.numpy())
    return np.concatenate(all_traj, axis=0), np.concatenate(all_logits, axis=0)


def mode_endpoint_spread(traj: np.ndarray) -> np.ndarray:
    """Mean distance of each mode's endpoint from the across-mode centroid endpoint. traj: [N, K, T, 2]."""
    endpoints = traj[:, :, -1, :]  # [N, K, 2]
    centroid = endpoints.mean(axis=1, keepdims=True)  # [N, 1, 2]
    dists = np.linalg.norm(endpoints - centroid, axis=-1)  # [N, K]
    return dists.mean(axis=1)


def best_mode_endpoint(traj: np.ndarray, logits: np.ndarray) -> np.ndarray:
    best = logits.argmax(axis=1)
    idx = np.arange(len(traj))
    return traj[idx, best, -1, :]


def main() -> None:
    df = filter_difficulty(load_split("train"), "hard").reset_index(drop=True)

    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_preds = xgb_model.predict(xgb_make_features(df)).reshape(len(df), 12, 2)
    xgb_endpoint = xgb_preds[:, -1, :]

    traj, logits = transformer_predictions(df)
    spread = mode_endpoint_spread(traj)
    transformer_endpoint = best_mode_endpoint(traj, logits)

    divergence = np.linalg.norm(xgb_endpoint - transformer_endpoint, axis=-1)

    spread_rank = pd.Series(spread).rank(pct=True).to_numpy()
    divergence_rank = pd.Series(divergence).rank(pct=True).to_numpy()
    uncertainty_score = 0.5 * spread_rank + 0.5 * divergence_rank

    out = df[["instance_token", "sample_token", "scene_name", "difficulty"]].copy()
    out["mode_endpoint_spread"] = spread
    out["xgb_transformer_divergence"] = divergence
    out["uncertainty_score"] = uncertainty_score

    threshold = out["uncertainty_score"].quantile(1 - TOP_FRACTION)
    out["needs_review"] = out["uncertainty_score"] >= threshold

    out = out.sort_values("uncertainty_score", ascending=False).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)

    n_flagged = int(out["needs_review"].sum())
    print(f"Scored {len(out)} hard-scene TRAIN examples.")
    print(f"Flagged {n_flagged} ({n_flagged / len(out):.1%}) for review -> {OUTPUT_PATH}")
    print(out[out["needs_review"]][["scene_name", "mode_endpoint_spread", "xgb_transformer_divergence", "uncertainty_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
