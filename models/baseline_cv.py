"""Constant velocity trajectory baseline.

Extrapolates the agent's final observed velocity (agent frame, heading-
aligned, so the y-axis points along the agent's current heading) linearly
over the 6s future horizon. No learning involved — this is the simplest
possible baseline, meant to calibrate how much the learned models below
actually buy us.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from evaluation.evaluate import filter_difficulty, future_xy, load_split, log_metrics
from evaluation.metrics import batch_metrics

DT = 0.5  # seconds between timesteps (2 Hz)
FUTURE_STEPS = 12


def predict_cv(df: pd.DataFrame) -> np.ndarray:
    """Returns [N, FUTURE_STEPS, 2] agent-frame predictions."""
    last_past = df[["past_x_3", "past_y_3"]].to_numpy(dtype=float)  # most recent observed past point
    valid = ~np.isnan(last_past).any(axis=1)

    velocity = np.zeros_like(last_past)  # stationary fallback where history is missing
    velocity[valid] = -last_past[valid] / DT  # (current(0,0) - last_past) / dt

    steps_seconds = (np.arange(1, FUTURE_STEPS + 1) * DT).reshape(1, -1, 1)  # [1, T, 1]
    preds = velocity[:, None, :] * steps_seconds  # [N, T, 2]
    return preds


def main() -> None:
    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        preds = predict_cv(df)
        gts = future_xy(df)
        metrics = batch_metrics(preds, gts)
        log_metrics(phase=2, model="Constant Velocity", eval_split="test", difficulty=difficulty, metrics=metrics)
        print(f"[CV] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
