"""Trajectory prediction metrics: minADE, minFDE, Miss Rate @ 2m.

All functions operate on batched numpy arrays so the same code path
handles single-trajectory baselines (K=1) and multimodal models (K>1).
"""

import numpy as np


def batch_metrics(preds: np.ndarray, gts: np.ndarray, miss_threshold: float = 2.0) -> dict:
    """
    :param preds: [N, T, 2] (single trajectory) or [N, K, T, 2] (K candidate futures).
    :param gts: [N, T, 2] ground truth futures.
    :param miss_threshold: FDE (meters) above which a prediction counts as a "miss".
    :return: dict with minADE, minFDE, MissRate@2m (all in meters, unrounded), and N.
    """
    if preds.ndim == 3:
        preds = preds[:, None, :, :]  # add a K=1 axis

    diffs = preds - gts[:, None, :, :]  # [N, K, T, 2]
    dists = np.linalg.norm(diffs, axis=-1)  # [N, K, T]

    ade_per_mode = dists.mean(axis=-1)  # [N, K]
    fde_per_mode = dists[:, :, -1]  # [N, K]

    min_ade = ade_per_mode.min(axis=-1)  # [N]
    min_fde = fde_per_mode.min(axis=-1)  # [N]
    miss = min_fde > miss_threshold

    return {
        "minADE": float(min_ade.mean()),
        "minFDE": float(min_fde.mean()),
        "MissRate@2m": float(miss.mean()),
        "N": int(len(gts)),
    }
