"""XGBoost trajectory baseline.

Predicts the flattened 24-dim future waypoint vector (12 timesteps x 2)
from engineered scalar / past-position / neighbor features, via one
independent XGBoost regressor per target dimension (sklearn's
MultiOutputRegressor wrapping XGBRegressor). Trained on the full train
split (all difficulties — the easy/hard split is reserved for the
transformer pretrain/fine-tune structure in Phases 3-4).

XGBoost's native missing-value handling (missing=np.nan) means we don't
need to impute the NaNs that show up near scene starts (no velocity/
acceleration/heading-change-rate yet) or when fewer than 3 neighbors are
present — they're passed through as-is.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from evaluation.evaluate import filter_difficulty, future_xy, load_split, log_metrics
from evaluation.metrics import batch_metrics

FUTURE_STEPS = 12
FEATURE_COLS = [
    "velocity",
    "acceleration",
    "heading",
    "heading_change_rate",
    "past_x_0",
    "past_y_0",
    "past_x_1",
    "past_y_1",
    "past_x_2",
    "past_y_2",
    "past_x_3",
    "past_y_3",
    "neighbor_dist_0",
    "neighbor_rel_heading_0",
    "neighbor_dist_1",
    "neighbor_rel_heading_1",
    "neighbor_dist_2",
    "neighbor_rel_heading_2",
    "neighbor_density_count",
]
TARGET_COLS = [f"future_{axis}_{i}" for i in range(FUTURE_STEPS) for axis in ("x", "y")]
MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / "xgb_baseline.joblib"


def make_features(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    intersection = df["near_intersection"].to_numpy(dtype=float).reshape(-1, 1)
    return np.concatenate([X, intersection], axis=1)


def train(train_df: pd.DataFrame) -> MultiOutputRegressor:
    X_train = make_features(train_df)
    Y_train = train_df[TARGET_COLS].to_numpy(dtype=float)

    base = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        missing=np.nan,
        n_jobs=-1,
        random_state=0,
    )
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(X_train, Y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def predict(model: MultiOutputRegressor, df: pd.DataFrame) -> np.ndarray:
    X = make_features(df)
    preds_flat = model.predict(X)  # [N, 24]
    return preds_flat.reshape(len(df), FUTURE_STEPS, 2)


def main() -> None:
    train_df = load_split("train")
    model = train(train_df)

    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        preds = predict(model, df)
        gts = future_xy(df)
        metrics = batch_metrics(preds, gts)
        log_metrics(
            phase=2,
            model="XGBoost",
            eval_split="test",
            difficulty=difficulty,
            metrics=metrics,
            notes="underperforms CV: trees underestimate displacement for higher-speed agents (can't extrapolate past training-range leaf values); see README limitations",
        )
        print(f"[XGBoost] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
