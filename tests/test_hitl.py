import numpy as np
import pandas as pd
import pytest

from trajflow.hitl import flag_uncertain
from trajflow.hitl.flag_uncertain import best_mode_endpoint, mode_endpoint_spread
from trajflow.models.finetune_round2 import merge_corrections
from trajflow.models.transformer import FUTURE_STEPS


def _future_row(instance_token, sample_token, x_fill, y_fill):
    row = {
        "instance_token": instance_token,
        "sample_token": sample_token,
        "difficulty": "hard",
        "scene_name": "scene-0",
    }
    for i in range(FUTURE_STEPS):
        row[f"future_x_{i}"] = x_fill
        row[f"future_y_{i}"] = y_fill
    return row


def _correction_row(instance_token, sample_token, x_fill, y_fill):
    row = {"instance_token": instance_token, "sample_token": sample_token}
    for i in range(FUTURE_STEPS):
        row[f"corrected_future_x_{i}"] = x_fill
        row[f"corrected_future_y_{i}"] = y_fill
    return row


def test_merge_corrections_overwrites_only_reviewed_rows():
    train_df = pd.DataFrame([
        _future_row("inst-1", "samp-1", 1.0, 1.0),
        _future_row("inst-2", "samp-2", 2.0, 2.0),
        _future_row("inst-3", "samp-3", 3.0, 3.0),
    ])
    corrections_df = pd.DataFrame([
        _correction_row("inst-1", "samp-1", 1.0, 1.0),  # "accept as-is": identical to original
        _correction_row("inst-2", "samp-2", 99.0, 99.0),  # actually edited
        # inst-3 was never reviewed -> must stay untouched
    ])

    merged, n_changed = merge_corrections(train_df, corrections_df)

    assert n_changed == 1  # only inst-2's values actually differ
    assert len(merged) == 3  # row count preserved, no rows dropped/added

    row1 = merged[merged["instance_token"] == "inst-1"].iloc[0]
    assert row1["future_x_0"] == 1.0  # unchanged (accept-as-is is a no-op)

    row2 = merged[merged["instance_token"] == "inst-2"].iloc[0]
    assert row2["future_x_0"] == 99.0  # overwritten with the correction

    row3 = merged[merged["instance_token"] == "inst-3"].iloc[0]
    assert row3["future_x_0"] == 3.0  # never flagged/reviewed -> untouched


def test_merge_corrections_with_no_reviews_is_a_full_noop():
    train_df = pd.DataFrame([_future_row("inst-1", "samp-1", 5.0, 5.0)])
    corrections_df = pd.DataFrame(columns=["instance_token", "sample_token"])
    merged, n_changed = merge_corrections(train_df, corrections_df)
    assert n_changed == 0
    assert merged["future_x_0"].iloc[0] == 5.0


def test_mode_endpoint_spread_is_zero_when_modes_agree():
    traj = np.zeros((2, 6, 12, 2))
    traj[:, :, -1, :] = 3.0  # every mode's endpoint identical
    spread = mode_endpoint_spread(traj)
    np.testing.assert_allclose(spread, 0.0)


def test_mode_endpoint_spread_increases_with_disagreement():
    traj = np.zeros((1, 2, 12, 2))
    traj[0, 0, -1] = [0.0, 0.0]
    traj[0, 1, -1] = [10.0, 0.0]
    spread = mode_endpoint_spread(traj)
    assert spread[0] == pytest.approx(5.0)  # each mode is 5m from the centroid at (5, 0)


def test_best_mode_endpoint_follows_argmax_logits():
    traj = np.zeros((1, 3, 12, 2))
    traj[0, 0, -1] = [1.0, 0.0]
    traj[0, 1, -1] = [2.0, 0.0]
    traj[0, 2, -1] = [3.0, 0.0]
    logits = np.array([[0.1, 5.0, 0.2]])  # mode 1 is the confident pick
    endpoint = best_mode_endpoint(traj, logits)
    np.testing.assert_allclose(endpoint[0], [2.0, 0.0])


def _split_frame():
    return pd.DataFrame({
        "instance_token": [f"inst-{i}" for i in range(6)],
        "sample_token": [f"samp-{i}" for i in range(6)],
        "scene_name": ["scene-a", "scene-a", "scene-b", "scene-b", "scene-c", "scene-c"],
        "difficulty": ["hard", "easy", "hard", "easy", "hard", "hard"],
    })


def test_load_candidates_reads_only_the_train_split_and_only_hard_rows(monkeypatch):
    requested = []

    def fake_load_split(name):
        requested.append(name)
        if name != "train":
            raise AssertionError(f"flagging must not read the {name} split")
        return _split_frame()

    monkeypatch.setattr(flag_uncertain, "load_split", fake_load_split)
    df = flag_uncertain.load_candidates()

    assert requested == ["train"]
    assert set(df["difficulty"]) == {"hard"}
    assert list(df["instance_token"]) == ["inst-0", "inst-2", "inst-4", "inst-5"]
    assert list(df.index) == [0, 1, 2, 3]


def test_score_candidates_flags_the_top_tenth_and_keeps_every_row():
    rng = np.random.default_rng(0)
    n, k = 40, 6
    df = pd.DataFrame({
        "instance_token": [f"inst-{i}" for i in range(n)],
        "sample_token": [f"samp-{i}" for i in range(n)],
        "scene_name": ["scene-a"] * n,
        "difficulty": ["hard"] * n,
    })
    traj = rng.normal(size=(n, k, FUTURE_STEPS, 2))
    logits = rng.normal(size=(n, k))
    xgb_endpoint = rng.normal(size=(n, 2))

    out = flag_uncertain.score_candidates(df, xgb_endpoint, traj, logits)

    assert sorted(out["instance_token"]) == sorted(df["instance_token"])
    assert out["uncertainty_score"].is_monotonic_decreasing
    assert int(out["needs_review"].sum()) == round(n * flag_uncertain.TOP_FRACTION)
    flagged_min = out.loc[out["needs_review"], "uncertainty_score"].min()
    unflagged_max = out.loc[~out["needs_review"], "uncertainty_score"].max()
    assert flagged_min > unflagged_max
