"""Property-based tests for the numeric code, using Hypothesis."""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from trajflow.evaluation.metrics import batch_metrics
from trajflow.hitl.flag_uncertain import mode_endpoint_spread
from trajflow.models.baseline_ca import DT, predict_ca
from trajflow.models.baseline_cv import predict_cv

T = 12
coords = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False, width=32)
shifts = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)
scales = st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False)


@st.composite
def multimodal_case(draw, max_k=4):
    n = draw(st.integers(1, 5))
    k = draw(st.integers(1, max_k))
    preds = draw(arrays(np.float64, (n, k, T, 2), elements=coords))
    gts = draw(arrays(np.float64, (n, T, 2), elements=coords))
    return preds, gts


@settings(max_examples=60, deadline=None)
@given(multimodal_case())
def test_metrics_are_nonnegative_and_miss_rate_is_a_fraction(case):
    preds, gts = case
    m = batch_metrics(preds, gts)
    assert m["minADE"] >= 0 and m["minFDE"] >= 0
    assert 0.0 <= m["MissRate@2m"] <= 1.0
    assert m["N"] == len(gts)


@settings(max_examples=60, deadline=None)
@given(multimodal_case())
def test_predicting_the_ground_truth_gives_zero_error(case):
    _, gts = case
    m = batch_metrics(gts, gts)
    assert m["minADE"] == 0 and m["minFDE"] == 0 and m["MissRate@2m"] == 0


@settings(max_examples=60, deadline=None)
@given(multimodal_case(max_k=3), st.data())
def test_adding_a_mode_never_increases_min_metrics(case, data):
    preds, gts = case
    extra = data.draw(arrays(np.float64, (len(gts), 1, T, 2), elements=coords))
    before = batch_metrics(preds, gts)
    after = batch_metrics(np.concatenate([preds, extra], axis=1), gts)
    assert after["minADE"] <= before["minADE"] + 1e-9
    assert after["minFDE"] <= before["minFDE"] + 1e-9
    assert after["MissRate@2m"] <= before["MissRate@2m"] + 1e-12


@settings(max_examples=60, deadline=None)
@given(multimodal_case(), shifts, shifts)
def test_metrics_are_translation_invariant(case, dx, dy):
    preds, gts = case
    shift = np.array([dx, dy])
    a = batch_metrics(preds, gts)
    b = batch_metrics(preds + shift, gts + shift)
    assert abs(a["minADE"] - b["minADE"]) < 1e-6
    assert abs(a["minFDE"] - b["minFDE"]) < 1e-6


@settings(max_examples=60, deadline=None)
@given(multimodal_case(), scales)
def test_ade_and_fde_scale_linearly_with_distance(case, c):
    preds, gts = case
    a = batch_metrics(preds, gts)
    b = batch_metrics(preds * c, gts * c)
    assert abs(b["minADE"] - c * a["minADE"]) < 1e-6 * max(1.0, c * a["minADE"])
    assert abs(b["minFDE"] - c * a["minFDE"]) < 1e-6 * max(1.0, c * a["minFDE"])


@settings(max_examples=60, deadline=None)
@given(multimodal_case(max_k=1))
def test_single_trajectory_input_matches_k_equals_one(case):
    preds, gts = case
    assert batch_metrics(preds[:, 0], gts) == batch_metrics(preds, gts)


@st.composite
def past_frames(draw):
    n = draw(st.integers(1, 4))
    frame = {}
    for i in range(4):
        frame[f"past_x_{i}"] = draw(arrays(np.float64, n, elements=coords))
        frame[f"past_y_{i}"] = draw(arrays(np.float64, n, elements=coords))
    return pd.DataFrame(frame)


@settings(max_examples=60, deadline=None)
@given(past_frames(), scales)
def test_constant_velocity_scales_with_the_past_positions(df, c):
    scaled = df * c
    np.testing.assert_allclose(predict_cv(scaled), c * predict_cv(df), rtol=1e-6, atol=1e-6)


@settings(max_examples=60, deadline=None)
@given(past_frames())
def test_constant_velocity_first_step_is_minus_last_past_point(df):
    first = predict_cv(df)[:, 0, :]
    expected = -df[["past_x_3", "past_y_3"]].to_numpy()
    np.testing.assert_allclose(first, expected, atol=1e-9)


@settings(max_examples=60, deadline=None)
@given(past_frames())
def test_constant_acceleration_equals_constant_velocity_when_second_point_is_missing(df):
    df = df.copy()
    df["past_x_2"] = np.nan
    df["past_y_2"] = np.nan
    np.testing.assert_allclose(predict_ca(df), predict_cv(df), atol=1e-9)


@settings(max_examples=60, deadline=None)
@given(past_frames())
def test_constant_acceleration_is_cv_plus_a_quadratic_term(df):
    p2 = df[["past_x_2", "past_y_2"]].to_numpy()
    p3 = df[["past_x_3", "past_y_3"]].to_numpy()
    accel = (p2 - 2 * p3) / DT**2
    t = (np.arange(1, T + 1) * DT).reshape(1, -1, 1)
    expected = predict_cv(df) + 0.5 * accel[:, None, :] * t**2
    np.testing.assert_allclose(predict_ca(df), expected, rtol=1e-6, atol=1e-6)


@settings(max_examples=60, deadline=None)
@given(arrays(np.float64, (3, 4, T, 2), elements=coords), shifts, shifts)
def test_mode_endpoint_spread_is_nonnegative_and_translation_invariant(traj, dx, dy):
    spread = mode_endpoint_spread(traj)
    assert (spread >= 0).all()
    np.testing.assert_allclose(spread, mode_endpoint_spread(traj + np.array([dx, dy])), atol=1e-6)


@settings(max_examples=40, deadline=None)
@given(arrays(np.float64, (2, 1, T, 2), elements=coords))
def test_mode_endpoint_spread_is_zero_for_identical_modes(traj):
    same = np.repeat(traj, 5, axis=1)
    np.testing.assert_allclose(mode_endpoint_spread(same), 0.0, atol=1e-9)
