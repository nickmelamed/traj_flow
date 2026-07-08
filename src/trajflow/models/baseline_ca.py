"""Constant acceleration trajectory baseline.

Extends the constant-velocity baseline: estimates both a velocity vector
and an acceleration vector from the two most recent observed past
positions (agent frame), then extrapolates with constant-acceleration
kinematics: pos(t) = v*t + 0.5*a*t^2. Falls back to constant velocity
(zero acceleration) if only one past point is available, and to fully
stationary if none are available -- same graceful-degradation style as
models/baseline_cv.py.

Added in a later pass alongside models/lstm.py to broaden the baseline
comparison (CLAUDE.md's Phase 2 originally specced constant-velocity +
XGBoost only); logged under the same phase=2 for consistency since it's
the same category of classical, no-training baseline.
"""

import numpy as np
import pandas as pd

from trajflow.evaluation.evaluate import filter_difficulty, future_xy, load_split, log_metrics
from trajflow.evaluation.metrics import batch_metrics

DT = 0.5  # seconds between timesteps (2 Hz)
FUTURE_STEPS = 12


def predict_ca(df: pd.DataFrame) -> np.ndarray:
    """Returns [N, FUTURE_STEPS, 2] agent-frame predictions."""
    p3 = df[["past_x_3", "past_y_3"]].to_numpy(dtype=float)  # most recent past point (t=-0.5s)
    p2 = df[["past_x_2", "past_y_2"]].to_numpy(dtype=float)  # second most recent (t=-1.0s)
    has_p3 = ~np.isnan(p3).any(axis=1)
    has_p2 = ~np.isnan(p2).any(axis=1)

    velocity = np.zeros_like(p3)
    velocity[has_p3] = -p3[has_p3] / DT  # (current(0,0) - p3) / dt, same as constant-velocity baseline

    acceleration = np.zeros_like(p3)
    has_both = has_p3 & has_p2
    # v_now = -p3/DT (derived above); v_prev = (p3 - p2)/DT; a = (v_now - v_prev) / DT
    acceleration[has_both] = (p2[has_both] - 2 * p3[has_both]) / (DT ** 2)

    steps_seconds = (np.arange(1, FUTURE_STEPS + 1) * DT).reshape(1, -1, 1)  # [1, T, 1]
    preds = velocity[:, None, :] * steps_seconds + 0.5 * acceleration[:, None, :] * steps_seconds ** 2
    return preds


def main() -> None:
    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        preds = predict_ca(df)
        gts = future_xy(df)
        metrics = batch_metrics(preds, gts)
        notes = (
            "substantially UNDERPERFORMS constant velocity (minADE 0.511) despite being the more "
            "sophisticated physics model -- acceleration estimated from a double finite-difference of "
            "noisy position data is itself noisy, and the t^2 term in constant-acceleration kinematics "
            "amplifies that noise over the 6s horizon (e.g. a small spurious acceleration estimate "
            "produces a >100m overshoot by t=6s for an otherwise near-stationary vehicle); a textbook "
            "fragility of naive higher-order kinematic extrapolation, not a bug -- see README limitations"
            if difficulty == "all"
            else ""
        )
        log_metrics(phase=2, model="Constant Acceleration", eval_split="test", difficulty=difficulty, metrics=metrics, notes=notes)
        print(f"[CA] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
