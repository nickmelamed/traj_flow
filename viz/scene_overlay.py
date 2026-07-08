"""Scene overlay visualizations for the README: history, ground truth, the
constant-velocity baseline's prediction, and the final model's (fine-
tuned-v2, post-HITL) prediction, plotted together with nearby map context
for a handful of representative test examples.

Selection (all on the untouched test set):
  1. easy_typical.png  -- median-error easy example
  2. hard_improvement.png -- the hard example where fine-tuned-v2 beats
     the CV baseline by the largest margin (the spec asks for at least
     one such case)
  3. hard_typical.png -- median-error hard example, included for honesty:
     the aggregate metrics (results/metrics_comparison.md) show CV is
     still competitive-to-better than fine-tuned-v2 on hard scenes
     overall, so a single cherry-picked "win" isn't the whole story.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.preprocess import DEFAULT_DATAROOT, FUTURE_STEPS, PAST_STEPS
from evaluation.evaluate import future_xy, load_split
from models.baseline_cv import predict_cv
from models.transformer import TrajectoryDataset, TrajectoryTransformer

from nuscenes.map_expansion.arcline_path_utils import discretize_lane
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import convert_global_coords_to_local

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "finetuned_v2.pt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
MAP_RADIUS = 40.0


def load_model() -> TrajectoryTransformer:
    model = TrajectoryTransformer()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    return model


@torch.no_grad()
def predict_final_best_mode(model: TrajectoryTransformer, df) -> np.ndarray:
    """Returns [N, FUTURE_STEPS, 2]: each row's highest-probability mode."""
    dataset = TrajectoryDataset(df)
    preds = []
    for i in range(len(dataset)):
        past_seq, context, _ = dataset[i]
        traj, logits = model(past_seq.unsqueeze(0), context.unsqueeze(0))
        best = logits.argmax(dim=1).item()
        preds.append(traj[0, best].numpy())
    return np.stack(preds, axis=0)


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


def plot_example(row, cv_pred: np.ndarray, final_pred: np.ndarray, map_cache: dict, helper: PredictHelper, out_path: Path, title: str) -> None:
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

    ax.plot(np.r_[0, cv_pred[:, 0]], np.r_[0, cv_pred[:, 1]], "--", color="tab:red", linewidth=2, label="constant velocity baseline", zorder=2)
    ax.plot(np.r_[0, final_pred[:, 0]], np.r_[0, final_pred[:, 1]], "-", color="tab:green", linewidth=2.5, label="fine-tuned-v2 (final)", zorder=4)

    ax.scatter([0], [0], color="black", marker="x", s=60, zorder=6)

    # Clip the view to the trajectory content (with padding), not the map's
    # extent -- nearby_lane_polylines pulls in full lane geometry for any
    # lane that merely intersects the MAP_RADIUS patch, which can stretch
    # far beyond it and otherwise shrinks the actual trajectories (the
    # whole point of the figure) down to an unreadable sliver.
    content_x = np.concatenate([past_x[valid], future_x, cv_pred[:, 0], final_pred[:, 0], [0.0]])
    content_y = np.concatenate([past_y[valid], future_y, cv_pred[:, 1], final_pred[:, 1], [0.0]])
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
    model = load_model()

    test_df = load_split("test").reset_index(drop=True)
    gts = future_xy(test_df)
    cv_preds = predict_cv(test_df)
    final_preds = predict_final_best_mode(model, test_df)

    cv_ade = np.linalg.norm(cv_preds - gts, axis=-1).mean(axis=1)
    final_ade = np.linalg.norm(final_preds - gts, axis=-1).mean(axis=1)
    displacement = np.linalg.norm(gts[:, -1, :], axis=-1)  # net ground-truth movement over 6s

    test_df["cv_ade"] = cv_ade
    test_df["final_ade"] = final_ade
    test_df["improvement"] = cv_ade - final_ade  # positive = fine-tuned-v2 beat CV
    test_df["displacement"] = displacement

    # Restrict example selection to genuinely moving vehicles (>5m net
    # displacement over 6s). Over 90% of test examples are near-stationary
    # (parked cars, median displacement is ~0.16m) -- true to the dataset
    # and discussed in the README, but a "typical" or "improvement" example
    # from the literal unfiltered median would just be a parked car with an
    # invisible trajectory, which illustrates nothing. This selects for
    # legibility, not to hide how much of the dataset is near-stationary.
    moving = test_df[test_df["displacement"] > 5.0]
    print(f"Moving (>5m displacement) examples: {len(moving)} / {len(test_df)} total test examples.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    map_cache: dict = {}

    easy_moving = moving[moving["difficulty"] == "easy"].sort_values("final_ade")
    row = easy_moving.iloc[len(easy_moving) // 2]
    plot_example(
        row, cv_preds[row.name], final_preds[row.name], map_cache, helper,
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
        row, cv_preds[row.name], final_preds[row.name], map_cache, helper,
        OUTPUT_DIR / "hard_improvement.png", f"{row['scene_name']} (hard, largest improvement) — CV ADE={row['cv_ade']:.2f}m, final ADE={row['final_ade']:.2f}m",
    )

    hard_typical_moving = moving[moving["difficulty"] == "hard"].sort_values("final_ade")
    row = hard_typical_moving.iloc[len(hard_typical_moving) // 2]
    plot_example(
        row, cv_preds[row.name], final_preds[row.name], map_cache, helper,
        OUTPUT_DIR / "hard_typical.png", f"{row['scene_name']} (hard, typical moving vehicle) — CV ADE={row['cv_ade']:.2f}m, final ADE={row['final_ade']:.2f}m",
    )

    print(f"Saved figures to {OUTPUT_DIR}:")
    for fname in ["easy_typical.png", "hard_improvement.png", "hard_typical.png"]:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
