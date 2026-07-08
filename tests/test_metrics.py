import numpy as np
import pytest

from trajflow.evaluation.metrics import batch_metrics


def test_perfect_single_hypothesis_prediction_is_zero_error():
    gts = np.random.default_rng(0).normal(size=(5, 12, 2))
    metrics = batch_metrics(gts.copy(), gts)
    assert metrics["minADE"] == 0.0
    assert metrics["minFDE"] == 0.0
    assert metrics["MissRate@2m"] == 0.0
    assert metrics["N"] == 5


def test_multimodal_uses_best_mode_not_average():
    # K=2: mode 0 is wildly wrong, mode 1 matches ground truth exactly.
    gt = np.zeros((1, 12, 2))
    preds = np.zeros((1, 2, 12, 2))
    preds[0, 0] = 100.0  # bad mode
    preds[0, 1] = 0.0  # perfect mode
    metrics = batch_metrics(preds, gt)
    assert metrics["minADE"] == 0.0
    assert metrics["minFDE"] == 0.0


def test_ade_is_mean_over_time_fde_is_final_step_only():
    gt = np.zeros((1, 3, 2))  # single example, K=1, T=3
    preds = np.array([[[1.0, 0.0], [1.0, 0.0], [4.0, 0.0]]])  # per-step errors: 1, 1, 4
    metrics = batch_metrics(preds, gt)
    assert metrics["minADE"] == pytest.approx((1 + 1 + 4) / 3)
    assert metrics["minFDE"] == pytest.approx(4.0)


def test_miss_rate_threshold_is_strictly_greater_than():
    gt = np.zeros((2, 1, 2))
    preds = np.zeros((2, 1, 2))
    preds[0, 0] = [2.0, 0.0]  # FDE exactly 2.0 -> not a miss (threshold is ">")
    preds[1, 0] = [2.0001, 0.0]  # FDE just above 2.0 -> a miss
    metrics = batch_metrics(preds, gt, miss_threshold=2.0)
    assert metrics["MissRate@2m"] == 0.5


def test_accepts_k1_without_explicit_mode_axis():
    gt = np.random.default_rng(1).normal(size=(4, 12, 2))
    preds_no_k_axis = gt + 1.0
    preds_with_k_axis = preds_no_k_axis[:, None, :, :]
    m1 = batch_metrics(preds_no_k_axis, gt)
    m2 = batch_metrics(preds_with_k_axis, gt)
    assert m1 == m2
