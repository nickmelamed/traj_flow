import numpy as np
import pandas as pd
import pytest

from trajflow.models.baseline_ca import DT, FUTURE_STEPS, predict_ca
from trajflow.models.baseline_cv import predict_cv


def _row(past_x_0=np.nan, past_y_0=np.nan, past_x_1=np.nan, past_y_1=np.nan,
         past_x_2=np.nan, past_y_2=np.nan, past_x_3=np.nan, past_y_3=np.nan):
    return {
        "past_x_0": past_x_0, "past_y_0": past_y_0,
        "past_x_1": past_x_1, "past_y_1": past_y_1,
        "past_x_2": past_x_2, "past_y_2": past_y_2,
        "past_x_3": past_x_3, "past_y_3": past_y_3,
    }


def test_cv_extrapolates_constant_velocity_from_last_two_points():
    # Agent frame: current position is always (0, 0). If the most recent
    # past point (t=-0.5s) was at (-1, 0), the agent moved +1m in x over
    # 0.5s -> velocity (2, 0) m/s -> after 1s it should be at (2, 0).
    df = pd.DataFrame([_row(past_x_3=-1.0, past_y_3=0.0)])
    preds = predict_cv(df)
    assert preds.shape == (1, FUTURE_STEPS, 2)
    np.testing.assert_allclose(preds[0, 1], [2.0, 0.0])  # t=1.0s (index 1)
    np.testing.assert_allclose(preds[0, -1], [12.0, 0.0])  # t=6.0s


def test_cv_falls_back_to_stationary_when_history_missing():
    df = pd.DataFrame([_row()])  # all-NaN past
    preds = predict_cv(df)
    np.testing.assert_allclose(preds[0], np.zeros((FUTURE_STEPS, 2)))


def test_ca_matches_hand_derived_kinematics_with_both_past_points():
    # p2 (t=-1.0s) = (-3, 0), p3 (t=-0.5s) = (-1, 0).
    # v_now = -p3/DT = (2, 0); v_prev = (p3-p2)/DT = (4, 0) -> a = (v_now-v_prev)/DT = (-4, 0).
    df = pd.DataFrame([_row(past_x_2=-3.0, past_y_2=0.0, past_x_3=-1.0, past_y_3=0.0)])
    preds = predict_ca(df)
    t = DT  # first future step, 0.5s
    expected_first = np.array([2.0, 0.0]) * t + 0.5 * np.array([-4.0, 0.0]) * t**2
    np.testing.assert_allclose(preds[0, 0], expected_first, atol=1e-9)


def test_ca_falls_back_to_cv_when_only_one_past_point():
    df = pd.DataFrame([_row(past_x_3=-1.0, past_y_3=0.0)])  # no p2
    ca_preds = predict_ca(df)
    cv_preds = predict_cv(df)
    np.testing.assert_allclose(ca_preds, cv_preds)


def test_ca_amplifies_noise_more_than_cv_over_the_horizon():
    # A small perturbation to the older point (simulating position noise)
    # should move the t=6s CA prediction by much more than it moves the
    # t=6s CV prediction, illustrating the t^2 noise-amplification claim
    # made in baseline_ca.py's module docstring and README.
    clean = pd.DataFrame([_row(past_x_2=-2.0, past_y_2=0.0, past_x_3=-1.0, past_y_3=0.0)])
    noisy = pd.DataFrame([_row(past_x_2=-2.2, past_y_2=0.0, past_x_3=-1.0, past_y_3=0.0)])

    cv_shift = np.linalg.norm(predict_cv(noisy)[0, -1] - predict_cv(clean)[0, -1])
    ca_shift = np.linalg.norm(predict_ca(noisy)[0, -1] - predict_ca(clean)[0, -1])
    assert ca_shift > cv_shift * 5


def test_predictions_have_no_nans_even_with_partial_history():
    df = pd.DataFrame([_row(past_x_3=-1.0, past_y_3=0.0), _row()])
    assert not np.isnan(predict_cv(df)).any()
    assert not np.isnan(predict_ca(df)).any()
