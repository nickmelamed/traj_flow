import numpy as np
import pandas as pd
import torch

from trajflow.data.preprocess import FUTURE_STEPS, PAST_STEPS
from trajflow.evaluation.seed_variance import train_loop
from trajflow.models.transformer import TrajectoryTransformer


def _tiny_df(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        row = {
            "velocity": rng.normal(), "acceleration": rng.normal(), "heading_change_rate": rng.normal(),
            "neighbor_density_count": 0, "near_intersection": False,
        }
        for i in range(PAST_STEPS):
            row[f"past_x_{i}"] = rng.normal()
            row[f"past_y_{i}"] = rng.normal()
        for i in range(3):
            row[f"neighbor_dist_{i}"] = np.nan
            row[f"neighbor_rel_heading_{i}"] = np.nan
        for i in range(FUTURE_STEPS):
            row[f"future_x_{i}"] = rng.normal()
            row[f"future_y_{i}"] = rng.normal()
        rows.append(row)
    return pd.DataFrame(rows)


def test_warm_start_falls_back_to_starting_weights_when_untrained():
    """warm_start=True (finetune-style) with 0 training epochs must return
    the model completely unchanged -- this is the safeguard finetune.py/
    finetune_round2.py rely on. Regression test for the bug caught while
    building the seed-variance study: applying this same fallback to a
    from-scratch training run (warm_start should have been False) could
    silently return an untrained, randomly-initialized model.
    """
    torch.manual_seed(0)
    model = TrajectoryTransformer()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    df = _tiny_df(6, seed=1)
    trained, best_val = train_loop(model, df, df, epochs=0, lr=1e-3, warm_start=True)

    for k, v in trained.state_dict().items():
        torch.testing.assert_close(v, original_state[k])


def test_no_warm_start_always_accepts_at_least_one_epoch():
    """warm_start=False (pretrain/lstm/transformer_full-style) must accept
    epoch 1's result unconditionally (best_val_minade starts at +inf), so
    the returned model is never the untrained random initialization even
    if that initialization scores well by chance on this near-stationary
    dataset.
    """
    torch.manual_seed(0)
    model = TrajectoryTransformer()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    df = _tiny_df(6, seed=2)
    trained, best_val = train_loop(model, df, df, epochs=1, lr=1e-2, warm_start=False)

    assert np.isfinite(best_val)
    changed = any(
        not torch.allclose(v, original_state[k]) for k, v in trained.state_dict().items()
    )
    assert changed, "model weights must differ from the random init after warm_start=False training"
