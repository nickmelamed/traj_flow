"""Characterization tests: pin down what the core logic does today.

These use small synthetic frames with hand-worked answers. They describe
current behavior, so a failure means behavior changed, not necessarily that
the new behavior is wrong.
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from trajflow.data.preprocess import DENSITY_THRESHOLD, FUTURE_STEPS, PAST_STEPS, classify_difficulty
from trajflow.evaluation.evaluate import filter_difficulty, future_xy
from trajflow.evaluation.metrics import batch_metrics
from trajflow.evaluation.moving_subset_analysis import bootstrap_ci, per_example_min_ade
from trajflow.models.baseline_xgb import FEATURE_COLS, make_features
from trajflow.models.lstm import LSTMTrajectoryModel
from trajflow.models.transformer import (
    CONTEXT_DIM,
    TrajectoryDataset,
    TrajectoryTransformer,
    min_of_k_loss,
)


def _frame(n=3, **overrides):
    """Return a processed-schema frame whose values are easy to recognise."""
    cols = {
        "instance_token": [f"i{k}" for k in range(n)],
        "sample_token": [f"s{k}" for k in range(n)],
        "difficulty": ["easy"] * n,
        "velocity": [1.0] * n,
        "acceleration": [0.5] * n,
        "heading": [0.3] * n,
        "heading_change_rate": [0.1] * n,
        "near_intersection": [False] * n,
        "neighbor_density_count": [2] * n,
    }
    for i in range(PAST_STEPS):
        cols[f"past_x_{i}"] = [float(i)] * n
        cols[f"past_y_{i}"] = [float(-i)] * n
    for i in range(3):
        cols[f"neighbor_dist_{i}"] = [10.0 + i] * n
        cols[f"neighbor_rel_heading_{i}"] = [0.2 * i] * n
    for i in range(FUTURE_STEPS):
        cols[f"future_x_{i}"] = [float(i + 1)] * n
        cols[f"future_y_{i}"] = [float(2 * (i + 1))] * n
    cols.update(overrides)
    return pd.DataFrame(cols)


# classify_difficulty


@pytest.mark.parametrize(
    "density, near_intersection, expected",
    [
        (0, False, "easy"),
        (DENSITY_THRESHOLD - 1, False, "easy"),
        (DENSITY_THRESHOLD, False, "hard"),
        (0, True, "hard"),
        (DENSITY_THRESHOLD + 3, True, "hard"),
    ],
)
def test_difficulty_is_hard_at_density_threshold_or_near_intersection(density, near_intersection, expected):
    assert classify_difficulty(density, near_intersection) == expected


# evaluate helpers


def test_filter_difficulty_all_returns_every_row_and_others_filter():
    df = _frame(4, difficulty=["easy", "hard", "hard", "easy"])
    assert len(filter_difficulty(df, "all")) == 4
    assert filter_difficulty(df, "hard")["instance_token"].tolist() == ["i1", "i2"]
    assert filter_difficulty(df, "easy")["instance_token"].tolist() == ["i0", "i3"]


def test_future_xy_shape_and_axis_order():
    gts = future_xy(_frame(2))
    assert gts.shape == (2, FUTURE_STEPS, 2)
    np.testing.assert_allclose(gts[0, 0], [1.0, 2.0])
    np.testing.assert_allclose(gts[1, -1], [float(FUTURE_STEPS), 2.0 * FUTURE_STEPS])


# XGBoost features


def test_xgb_features_follow_column_order_and_append_intersection_last():
    df = _frame(2, near_intersection=[True, False])
    X = make_features(df)
    assert X.shape == (2, len(FEATURE_COLS) + 1)
    np.testing.assert_allclose(X[:, :-1], df[FEATURE_COLS].to_numpy(dtype=float))
    np.testing.assert_allclose(X[:, -1], [1.0, 0.0])


def test_xgb_features_exclude_absolute_heading():
    assert "heading" not in FEATURE_COLS


def test_xgb_features_keep_nan_for_missing_values():
    df = _frame(1, past_x_0=[np.nan], neighbor_dist_2=[np.nan])
    X = make_features(df)
    assert np.isnan(X[0, FEATURE_COLS.index("past_x_0")])
    assert np.isnan(X[0, FEATURE_COLS.index("neighbor_dist_2")])


# TrajectoryDataset


def test_dataset_context_has_expected_width_and_layout():
    assert CONTEXT_DIM == 17
    ds = TrajectoryDataset(_frame(2, near_intersection=[True, False]))
    assert ds.context.shape == (2, CONTEXT_DIM)
    # scalars (3), scalar-valid (3), neighbors interleaved (6), neighbor-valid (3), density, intersection
    np.testing.assert_allclose(ds.context[0, :3], [1.0, 0.5, 0.1])
    np.testing.assert_allclose(ds.context[0, 3:6], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(ds.context[0, 6:12], [10.0, 0.0, 11.0, 0.2, 12.0, 0.4])
    np.testing.assert_allclose(ds.context[0, 12:15], [1.0, 1.0, 1.0])
    assert ds.context[0, 15] == 2.0
    assert ds.context[0, 16] == 1.0
    assert ds.context[1, 16] == 0.0


def test_dataset_zero_fills_missing_values_and_marks_them_invalid():
    df = _frame(
        1,
        past_x_0=[np.nan],
        velocity=[np.nan],
        neighbor_dist_1=[np.nan],
        neighbor_rel_heading_1=[np.nan],
    )
    ds = TrajectoryDataset(df)
    assert ds.past_seq[0, 0, 0] == 0.0
    assert ds.past_seq[0, 0, 2] == 0.0
    assert ds.past_seq[0, 1, 2] == 1.0
    assert ds.context[0, 0] == 0.0
    assert ds.context[0, 3] == 0.0
    assert ds.context[0, 8] == 0.0 and ds.context[0, 9] == 0.0
    assert ds.context[0, 13] == 0.0
    assert not np.isnan(ds.context).any()
    assert not np.isnan(ds.past_seq).any()


def test_dataset_item_shapes_and_future_layout():
    ds = TrajectoryDataset(_frame(3))
    past_seq, context, future = ds[1]
    assert past_seq.shape == (PAST_STEPS, 3)
    assert context.shape == (CONTEXT_DIM,)
    assert future.shape == (FUTURE_STEPS, 2)
    assert len(ds) == 3
    np.testing.assert_allclose(future[0].numpy(), [1.0, 2.0])


# min_of_k_loss


def test_min_of_k_loss_picks_closest_mode_and_cross_entropy_on_it():
    gt = torch.zeros(1, 2, 2)
    pred = torch.zeros(1, 2, 2, 2)
    pred[0, 0] = torch.tensor([[3.0, 4.0], [3.0, 4.0]])  # distance 5 at every step
    pred[0, 1] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # distance 1 at every step
    logits = torch.zeros(1, 2)
    total, reg, cls = min_of_k_loss(pred, logits, gt)
    assert reg.item() == pytest.approx(1.0)
    assert cls.item() == pytest.approx(math.log(2))
    assert total.item() == pytest.approx(1.0 + math.log(2))


def test_min_of_k_loss_classification_term_prefers_confident_correct_mode():
    gt = torch.zeros(1, 2, 2)
    pred = torch.zeros(1, 2, 2, 2)
    pred[0, 1] = 1.0
    pred[0, 0] = 5.0
    _, _, cls_right = min_of_k_loss(pred, torch.tensor([[-4.0, 4.0]]), gt)
    _, _, cls_wrong = min_of_k_loss(pred, torch.tensor([[4.0, -4.0]]), gt)
    assert cls_right.item() < cls_wrong.item()


# model output shapes


def test_transformer_forward_shapes_and_finite_output():
    torch.manual_seed(0)
    ds = TrajectoryDataset(_frame(4))
    past = torch.from_numpy(ds.past_seq)
    ctx = torch.from_numpy(ds.context)
    model = TrajectoryTransformer().eval()
    traj, logits = model(past, ctx)
    assert traj.shape == (4, 6, FUTURE_STEPS, 2)
    assert logits.shape == (4, 6)
    assert torch.isfinite(traj).all() and torch.isfinite(logits).all()


def test_lstm_forward_shapes_and_cumulative_decoding():
    torch.manual_seed(0)
    ds = TrajectoryDataset(_frame(2))
    past = torch.from_numpy(ds.past_seq)
    ctx = torch.from_numpy(ds.context)
    model = LSTMTrajectoryModel().eval()
    traj, logits = model(past, ctx)
    assert traj.shape == (2, 6, FUTURE_STEPS, 2)
    assert logits.shape == (2, 6)
    assert torch.isfinite(traj).all()


def test_model_parameter_counts_match_readme_claims():
    transformer = sum(p.numel() for p in TrajectoryTransformer().parameters())
    lstm = sum(p.numel() for p in LSTMTrajectoryModel().parameters())
    assert 100_000 < transformer < 120_000  # README says 108k
    assert 35_000 < lstm < 48_000  # README says 41k


# moving-subset bootstrap helpers


def test_per_example_min_ade_matches_batch_metrics_mean():
    rng = np.random.default_rng(0)
    preds = rng.normal(size=(7, 3, FUTURE_STEPS, 2))
    gts = rng.normal(size=(7, FUTURE_STEPS, 2))
    per_example = per_example_min_ade(preds, gts)
    assert per_example.shape == (7,)
    assert per_example.mean() == pytest.approx(batch_metrics(preds, gts)["minADE"])


def test_per_example_min_ade_accepts_single_trajectory_predictions():
    rng = np.random.default_rng(1)
    preds = rng.normal(size=(5, FUTURE_STEPS, 2))
    gts = rng.normal(size=(5, FUTURE_STEPS, 2))
    assert per_example_min_ade(preds, gts).mean() == pytest.approx(batch_metrics(preds, gts)["minADE"])


def test_bootstrap_ci_collapses_when_every_resample_is_the_same():
    per_example = np.array([1.0, 2.0, 3.0, 4.0])
    boot_idx = np.tile(np.arange(4), (50, 1))
    lo, hi, boot_means = bootstrap_ci(per_example, boot_idx)
    assert lo == pytest.approx(2.5) and hi == pytest.approx(2.5)
    assert boot_means.shape == (50,)


def test_bootstrap_ci_brackets_the_mean_for_random_resamples():
    rng = np.random.default_rng(0)
    per_example = rng.exponential(size=63)
    boot_idx = rng.integers(0, 63, size=(500, 63))
    lo, hi, _ = bootstrap_ci(per_example, boot_idx)
    assert lo < per_example.mean() < hi
