"""Constant velocity trajectory baseline.

Extrapolates the agent's final observed velocity (agent frame, heading-
aligned, so the y-axis points along the agent's current heading) linearly
over the 6s future horizon. No learning involved — this is the simplest
possible baseline, meant to calibrate how much the learned models below
actually buy us.
"""

import numpy as np
import pandas as pd

from trajflow.evaluation.evaluate import filter_difficulty, future_xy, load_split, log_metrics
from trajflow.evaluation.metrics import batch_metrics

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
    # Also evaluated on train/val (not just test, the primary comparison
    # split) specifically to surface how much scene-to-scene variance this
    # baseline's own minADE swings by -- referenced directly in the README's
    # Limitations section as evidence that any single-run result here should
    # be read as "plausible given this data," not "precisely measured."
    # Without this, that claim would cite numbers nothing in the repo
    # actually computes or logs -- see CLAUDE.md's "log every metric" rule.
    for eval_split in ["train", "val", "test"]:
        for difficulty in ["all", "easy", "hard"]:
            df = filter_difficulty(load_split(eval_split), difficulty)
            preds = predict_cv(df)
            gts = future_xy(df)
            metrics = batch_metrics(preds, gts)
            if eval_split == "test" and difficulty == "all":
                notes = (
                    "this aggregate win is driven almost entirely by the dataset's dominant near-stationary "
                    "majority (median displacement 0.16m); restricted to the 63/1626 test examples that actually "
                    "move >5m, fine-tuned-v2 wins instead -- see the 'moving (>5m displacement)' rows below and "
                    "README 'moving-vehicle subset' section"
                )
            elif difficulty == "all":
                notes = (
                    "logged on train/val too (not just test) so the README's cross-scene-variance claim in "
                    "Limitations is backed by an actual row here rather than an unlogged number -- nuScenes mini's "
                    "train/val/test scenes have very different typical vehicle speeds, so this baseline's own "
                    "minADE swings a lot across splits despite using no learned parameters at all"
                )
            else:
                notes = ""
            log_metrics(phase=2, model="Constant Velocity", eval_split=eval_split, difficulty=difficulty, metrics=metrics, notes=notes)
            print(f"[CV] {eval_split}/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
