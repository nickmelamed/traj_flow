"""Scene overlay visualizations for the README: history, ground truth, the
constant-velocity baseline's prediction, and the final model's (fine-
tuned-v2, post-HITL) prediction, plotted together with nearby map context
for a handful of representative test examples.

The transformer's single plotted/scored prediction is its BEST-OF-K mode
(closest to ground truth), matching minADE's own selection rule -- not
its highest-probability mode. These give different numbers; picking the
latter here would make per-example captions inconsistent with the
official minADE reported everywhere else in this project.

Example selection (all on the untouched test set, restricted to vehicles
with >5m net displacement over 6s -- see the "moving" filter below):
  1. easy_typical.png / hard_typical.png -- the median example by
     RELATIVE performance (cv_ade - final_ade), not median absolute error.
     Median absolute error can land on an example where the two models'
     relative ranking doesn't match the aggregate trend at all, which
     would make the "typical" claim wrong for that specific figure.
  2. hard_improvement.png -- the hard example where fine-tuned-v2 beats
     the CV baseline by the largest margin, restricted to cases where
     fine-tuned-v2's own absolute error is actually small (<3m) -- so
     "largest improvement" doesn't just surface the least-catastrophic
     case among two failures.

The >5m-displacement "moving vehicle" filter used for example selection
is analytically important too, not just for figure legibility -- see
evaluation/moving_subset_analysis.py, which logs every registered
model's minADE/minFDE/MissRate on that subset to
results/metrics_comparison.md (phase 7). Restricted to it, several
learned models beat constant velocity, which the full-test-set aggregate
(dominated by near-stationary vehicles) obscures. See the README's
"moving-vehicle subset" section.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from data.preprocess import DEFAULT_DATAROOT, FUTURE_STEPS, PAST_STEPS
from evaluation.evaluate import future_xy, load_split
from models.lstm import LSTMTrajectoryModel
from models.transformer import TrajectoryTransformer
from viz.model_registry import load_cv_predict_fn, load_multimodal_predict_fn

from nuscenes.map_expansion.arcline_path_utils import discretize_lane
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import convert_global_coords_to_local

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
MAP_RADIUS = 40.0


def best_of_k(traj: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Reduces [N, K, T, 2] to [N, T, 2] by picking, per example, whichever
    of the K modes is closest to ground truth -- the same "best of K"
    selection minADE itself uses, so the single line drawn here and its
    captioned ADE are consistent with the official metric in
    results/metrics_comparison.md (picking the model's *most confident*
    mode instead, e.g. via argmax over predicted probabilities, is a
    different and typically worse metric -- don't conflate the two).
    """
    dists = np.linalg.norm(traj - gts[:, None, :, :], axis=-1).mean(axis=-1)  # [N, K]
    best_idx = dists.argmin(axis=1)
    return traj[np.arange(len(traj)), best_idx]


def nearby_lane_polylines(nusc_map: NuScenesMap, x: float, y: float, translation: tuple, rotation: tuple, radius: float) -> list:
    nearby = nusc_map.get_records_in_radius(x, y, radius, ["lane", "lane_connector"])
    tokens = nearby.get("lane", []) + nearby.get("lane_connector", [])
    polylines = []
    for token in tokens:
        path = nusc_map.arcline_path_3.get(token, [])
        if not path:
            continue
        pts = np.array(discretize_lane(path, resolution_meters=1.0))
        if len(pts) == 0:
            continue
        polylines.append(convert_global_coords_to_local(pts[:, :2], translation, rotation))
    return polylines


def plot_example(row, model_lines: list, map_cache: dict, helper: PredictHelper, out_path: Path, title: str) -> None:
    """model_lines: list of (label, pred [T,2], color, linestyle) tuples, one per model to overlay."""
    ann = helper.get_sample_annotation(row["instance_token"], row["sample_token"])
    x, y = ann["translation"][0], ann["translation"][1]
    map_name = row["map_name"]
    if map_name not in map_cache:
        map_cache[map_name] = NuScenesMap(dataroot=str(DEFAULT_DATAROOT), map_name=map_name)
    polylines = nearby_lane_polylines(map_cache[map_name], x, y, ann["translation"], ann["rotation"], MAP_RADIUS)

    fig, ax = plt.subplots(figsize=(6, 6))
    for poly in polylines:
        ax.plot(poly[:, 0], poly[:, 1], color="lightgray", linewidth=1, zorder=0)

    past_x = np.array([row[f"past_x_{i}"] for i in range(PAST_STEPS)])
    past_y = np.array([row[f"past_y_{i}"] for i in range(PAST_STEPS)])
    valid = ~np.isnan(past_x)
    ax.plot(np.r_[past_x[valid], 0], np.r_[past_y[valid], 0], "o-", color="tab:blue", label="past", zorder=3)

    future_x = np.array([row[f"future_x_{i}"] for i in range(FUTURE_STEPS)])
    future_y = np.array([row[f"future_y_{i}"] for i in range(FUTURE_STEPS)])
    ax.plot(np.r_[0, future_x], np.r_[0, future_y], "-", color="magenta", linewidth=3, label="ground truth", zorder=5)

    all_pred_x = [future_x]
    all_pred_y = [future_y]
    for i, (label, pred, color, linestyle) in enumerate(model_lines):
        ax.plot(np.r_[0, pred[:, 0]], np.r_[0, pred[:, 1]], linestyle, color=color, linewidth=2, label=label, zorder=2 + i)
        all_pred_x.append(pred[:, 0])
        all_pred_y.append(pred[:, 1])

    ax.scatter([0], [0], color="black", marker="x", s=60, zorder=6)

    # Clip the view to the trajectory content (with padding), not the map's
    # extent -- nearby_lane_polylines pulls in full lane geometry for any
    # lane that merely intersects the MAP_RADIUS patch, which can stretch
    # far beyond it and otherwise shrinks the actual trajectories (the
    # whole point of the figure) down to an unreadable sliver.
    content_x = np.concatenate([past_x[valid], [0.0]] + all_pred_x)
    content_y = np.concatenate([past_y[valid], [0.0]] + all_pred_y)
    pad = max(5.0, 0.25 * max(content_x.max() - content_x.min(), content_y.max() - content_y.min()))
    ax.set_xlim(content_x.min() - pad, content_x.max() + pad)
    ax.set_ylim(content_y.min() - pad, content_y.max() + pad)

    ax.set_xlabel("x (m, agent frame)")
    ax.set_ylabel("y (m, agent frame, heading = +y)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DEFAULT_DATAROOT), verbose=False)
    helper = PredictHelper(nusc)

    test_df = load_split("test").reset_index(drop=True)
    gts = future_xy(test_df)

    cv_traj, _ = load_cv_predict_fn()(test_df)
    cv_preds = cv_traj[:, 0]  # K=1

    final_traj, _ = load_multimodal_predict_fn(TrajectoryTransformer, "finetuned_v2.pt")(test_df)
    final_preds = best_of_k(final_traj, gts)  # matches minADE's own selection rule

    lstm_traj, _ = load_multimodal_predict_fn(LSTMTrajectoryModel, "lstm.pt")(test_df)
    lstm_preds = best_of_k(lstm_traj, gts)

    cv_ade = np.linalg.norm(cv_preds - gts, axis=-1).mean(axis=1)
    final_ade = np.linalg.norm(final_preds - gts, axis=-1).mean(axis=1)
    lstm_ade = np.linalg.norm(lstm_preds - gts, axis=-1).mean(axis=1)
    displacement = np.linalg.norm(gts[:, -1, :], axis=-1)  # net ground-truth movement over 6s

    test_df["cv_ade"] = cv_ade
    test_df["final_ade"] = final_ade
    test_df["lstm_ade"] = lstm_ade
    test_df["improvement"] = cv_ade - final_ade  # positive = fine-tuned-v2 beat CV
    test_df["displacement"] = displacement

    # Restrict example selection to genuinely moving vehicles (>5m net
    # displacement over 6s). Over 90% of test examples are near-stationary
    # (parked cars, median displacement is ~0.16m) -- true to the dataset
    # and discussed in the README, but a "typical" or "improvement" example
    # from the literal unfiltered median would just be a parked car with an
    # invisible trajectory, which illustrates nothing. This selects for
    # legibility, not to hide how much of the dataset is near-stationary.
    moving_mask = test_df["displacement"].to_numpy() > 5.0
    moving = test_df[moving_mask]
    print(f"Moving (>5m displacement) examples: {len(moving)} / {len(test_df)} total test examples.")
    print("(Moving-subset minADE for every model is logged separately by evaluation/moving_subset_analysis.py.)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    map_cache: dict = {}

    # "Typical" is selected by median IMPROVEMENT (cv_ade - final_ade), not
    # median absolute final_ade -- the latter can land on an example where
    # the final model happens to beat CV even though CV wins in aggregate
    # (median absolute error says nothing about which model is better on
    # that specific example), which would contradict the aggregate-metrics
    # story the caption claims to illustrate.
    cv_line = lambda idx: ("constant velocity baseline", cv_preds[idx], "tab:red", "--")
    final_line = lambda idx: ("fine-tuned-v2 (final)", final_preds[idx], "tab:green", "-")

    easy_moving = moving[moving["difficulty"] == "easy"].sort_values("improvement")
    row = easy_moving.iloc[len(easy_moving) // 2]
    plot_example(
        row, [cv_line(row.name), final_line(row.name)], map_cache, helper,
        OUTPUT_DIR / "easy_typical.png", f"{row['scene_name']} (easy, typical moving vehicle) — CV ADE={row['cv_ade']:.2f}m, final ADE={row['final_ade']:.2f}m",
    )

    # Among hard+moving examples, restrict to cases where the final model's
    # own prediction is actually good in absolute terms (final_ade < 3m)
    # before ranking by improvement over CV -- otherwise "largest
    # improvement" surfaces cases where CV fails catastrophically and the
    # final model merely fails somewhat less catastrophically (e.g. 28m vs
    # 19m error), which isn't a convincing illustration of anything.
    hard_moving_good = moving[(moving["difficulty"] == "hard") & (moving["final_ade"] < 3.0)].sort_values("improvement", ascending=False)
    row = hard_moving_good.iloc[0]
    plot_example(
        row, [cv_line(row.name), final_line(row.name)], map_cache, helper,
        OUTPUT_DIR / "hard_improvement.png", f"{row['scene_name']} (hard, largest improvement) — CV ADE={row['cv_ade']:.2f}m, final ADE={row['final_ade']:.2f}m",
    )

    hard_typical_moving = moving[moving["difficulty"] == "hard"].sort_values("improvement")
    row = hard_typical_moving.iloc[len(hard_typical_moving) // 2]
    plot_example(
        row, [cv_line(row.name), final_line(row.name)], map_cache, helper,
        OUTPUT_DIR / "hard_typical.png", f"{row['scene_name']} (hard, typical moving vehicle) — CV ADE={row['cv_ade']:.2f}m, final ADE={row['final_ade']:.2f}m",
    )

    # LSTM showcase: the MEDIAN (typical, not cherry-picked-best) moving
    # example by the LSTM's own error, alongside CV and fine-tuned-v2 for
    # direct comparison -- see README for why this result is exciting but
    # not (yet) a controlled architecture comparison.
    lstm_typical = moving.sort_values("lstm_ade")
    row = lstm_typical.iloc[len(lstm_typical) // 2]
    plot_example(
        row,
        [cv_line(row.name), final_line(row.name), ("LSTM (baseline)", lstm_preds[row.name], "tab:brown", "-")],
        map_cache, helper,
        OUTPUT_DIR / "lstm_typical.png",
        f"{row['scene_name']} (typical moving vehicle)\n"
        f"CV ADE={row['cv_ade']:.2f}m, fine-tuned-v2 ADE={row['final_ade']:.2f}m, LSTM ADE={row['lstm_ade']:.2f}m",
    )

    print(f"Saved figures to {OUTPUT_DIR}:")
    for fname in ["easy_typical.png", "hard_improvement.png", "hard_typical.png", "lstm_typical.png"]:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
