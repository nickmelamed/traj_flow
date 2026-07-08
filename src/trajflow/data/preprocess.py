"""Extract trajectory-prediction examples from nuScenes mini.

For every (vehicle instance, sample) pair with a full 6s future, this:
  * pulls 2s of past + 6s of future agent-frame xy history via PredictHelper
  * engineers velocity / acceleration / heading / heading-change-rate and
    distance + relative heading to the 3 nearest neighboring agents
  * classifies the example as "easy" or "hard" using neighbor density and
    proximity to a map intersection (NuScenesMap)
  * assigns it to a train/val/test split at the SCENE level, so no scene's
    samples appear in more than one split (avoids leakage)

Splits: nuScenes mini only defines mini_train (8 scenes) / mini_val
(2 scenes). We hold mini_val out untouched as our TEST set (matches the
official split exactly, never trained on). We further carve the last
`--val-scenes` scenes of mini_train into our VAL set for model selection,
keeping the rest as TRAIN. All carving is scene-level.

Output: data/processed/{train,val,test}.parquet — schema documented in
data/SCHEMA.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.eval.common.utils import angle_diff, quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.utils.splits import mini_train, mini_val

from trajflow.paths import NUSCENES_ROOT as DEFAULT_DATAROOT
from trajflow.paths import PROCESSED_DIR as DEFAULT_OUT
from trajflow.paths import SCHEMA_PATH

PAST_SECONDS = 2.0
FUTURE_SECONDS = 6.0
PAST_STEPS = 4  # 2 Hz * 2s
FUTURE_STEPS = 12  # 2 Hz * 6s

NEIGHBOR_RADIUS_MAX = 60.0  # meters; ignore agents farther than this as "neighbors"
NEIGHBOR_DENSITY_RADIUS = 10.0  # meters; used for the density half of easy/hard
DENSITY_THRESHOLD = 5  # >= this many neighbors within NEIGHBOR_DENSITY_RADIUS => "hard"
INTERSECTION_RADIUS = 4.0  # meters; used for the intersection half of easy/hard
# These were tuned empirically against this dataset (see preprocessing summary
# output): nuScenes mini's `is_intersection` road_segment polygons are large
# relative to typical inter-vehicle distances, so a naive 15-20m radius
# classified ~97% of examples as "hard" (degenerate for the pretrain/
# fine-tune split in Phases 3-4). These tighter radii instead give a
# realistic ~55/45 hard/easy split.

VEHICLE_PREFIX = "vehicle."
N_VAL_SCENES_FROM_TRAIN = 2


def build_scene_splits(val_scenes_from_train: int) -> dict:
    train_scene_names = sorted(mini_train)
    test_scene_names = sorted(mini_val)
    val_scene_names = train_scene_names[-val_scenes_from_train:]
    train_scene_names = train_scene_names[:-val_scenes_from_train]

    split_of_scene = {}
    split_of_scene.update({s: "train" for s in train_scene_names})
    split_of_scene.update({s: "val" for s in val_scene_names})
    split_of_scene.update({s: "test" for s in test_scene_names})
    return split_of_scene


def build_scene_lookup(nusc: NuScenes) -> dict:
    """Map every sample_token -> its scene name."""
    scene_of_sample = {}
    for scene in nusc.scene:
        sample_token = scene["first_sample_token"]
        while sample_token != "":
            scene_of_sample[sample_token] = scene["name"]
            sample_token = nusc.get("sample", sample_token)["next"]
    return scene_of_sample


def classify_difficulty(
    density_count: int, near_intersection: bool
) -> str:
    if near_intersection or density_count >= DENSITY_THRESHOLD:
        return "hard"
    return "easy"


def extract_examples(
    nusc: NuScenes,
    helper: PredictHelper,
    split_of_scene: dict,
    scene_of_sample: dict,
) -> tuple[list[dict], dict]:
    map_cache: dict[str, NuScenesMap] = {}
    rows = []
    stats = {"candidates": 0, "skipped_short_future": 0, "kept": 0}

    for sample in tqdm(nusc.sample, desc="samples"):
        sample_token = sample["token"]
        scene_name = scene_of_sample[sample_token]
        split = split_of_scene.get(scene_name)
        if split is None:
            continue

        annotations = helper.get_annotations_for_sample(sample_token)
        vehicle_anns = [a for a in annotations if a["category_name"].startswith(VEHICLE_PREFIX)]
        if not vehicle_anns:
            continue

        map_name = helper.get_map_name_from_sample_token(sample_token)
        if map_name not in map_cache:
            map_cache[map_name] = NuScenesMap(dataroot=str(nusc.dataroot), map_name=map_name)
        nusc_map = map_cache[map_name]

        for ann in vehicle_anns:
            stats["candidates"] += 1
            instance_token = ann["instance_token"]

            future = helper.get_future_for_agent(
                instance_token, sample_token, FUTURE_SECONDS, in_agent_frame=True, just_xy=True
            )
            if len(future) < FUTURE_STEPS:
                stats["skipped_short_future"] += 1
                continue
            future = future[:FUTURE_STEPS]

            past = helper.get_past_for_agent(
                instance_token, sample_token, PAST_SECONDS, in_agent_frame=True, just_xy=True
            )
            past = past[:PAST_STEPS]  # closest-past-first
            past_padded = np.full((PAST_STEPS, 2), np.nan)
            if len(past) > 0:
                past_padded[: len(past)] = past
            past_chrono = past_padded[::-1]  # oldest-first, most-recent last

            velocity = helper.get_velocity_for_agent(instance_token, sample_token)
            acceleration = helper.get_acceleration_for_agent(instance_token, sample_token)
            heading_change_rate = helper.get_heading_change_rate_for_agent(instance_token, sample_token)
            heading = quaternion_yaw(Quaternion(ann["rotation"]))

            x, y = ann["translation"][0], ann["translation"][1]

            neighbor_feats = []
            for other in annotations:
                if other["instance_token"] == instance_token:
                    continue
                ox, oy = other["translation"][0], other["translation"][1]
                dist = float(np.hypot(ox - x, oy - y))
                if dist > NEIGHBOR_RADIUS_MAX:
                    continue
                other_yaw = quaternion_yaw(Quaternion(other["rotation"]))
                rel_heading = float(angle_diff(other_yaw, heading, period=2 * np.pi))
                neighbor_feats.append((dist, rel_heading))
            neighbor_feats.sort(key=lambda t: t[0])

            density_count = sum(1 for d, _ in neighbor_feats if d <= NEIGHBOR_DENSITY_RADIUS)
            nearest3 = neighbor_feats[:3] + [(np.nan, np.nan)] * max(0, 3 - len(neighbor_feats))

            nearby = nusc_map.get_records_in_radius(x, y, INTERSECTION_RADIUS, ["road_segment"])
            near_intersection = any(
                nusc_map.get("road_segment", token).get("is_intersection", False)
                for token in nearby.get("road_segment", [])
            )

            difficulty = classify_difficulty(density_count, near_intersection)

            row = {
                "instance_token": instance_token,
                "sample_token": sample_token,
                "scene_name": scene_name,
                "split": split,
                "category_name": ann["category_name"],
                "map_name": map_name,
                "difficulty": difficulty,
                "near_intersection": near_intersection,
                "neighbor_density_count": density_count,
                "velocity": velocity,
                "acceleration": acceleration,
                "heading": heading,
                "heading_change_rate": heading_change_rate,
            }
            for i in range(PAST_STEPS):
                row[f"past_x_{i}"] = past_chrono[i, 0]
                row[f"past_y_{i}"] = past_chrono[i, 1]
            for i in range(FUTURE_STEPS):
                row[f"future_x_{i}"] = future[i, 0]
                row[f"future_y_{i}"] = future[i, 1]
            for i, (d, rh) in enumerate(nearest3):
                row[f"neighbor_dist_{i}"] = d
                row[f"neighbor_rel_heading_{i}"] = rh

            rows.append(row)
            stats["kept"] += 1

    return rows, stats


def write_schema_doc(path: Path) -> None:
    path.write_text(
        f"""# `data/processed/{{train,val,test}}.parquet` schema

One row = one (vehicle instance, sample) trajectory-prediction example.

| column | meaning |
|---|---|
| `instance_token`, `sample_token` | nuScenes identifiers for the agent / timestep |
| `scene_name`, `split` | source scene and its assigned split (train/val/test), scene-level — no leakage |
| `category_name` | nuScenes category, e.g. `vehicle.car` |
| `map_name` | nuScenes map location |
| `difficulty` | `easy` or `hard` — see classification rule below |
| `near_intersection` | bool, agent is within {INTERSECTION_RADIUS}m of a `road_segment` with `is_intersection=True` |
| `neighbor_density_count` | # other agents within {NEIGHBOR_DENSITY_RADIUS}m |
| `velocity`, `acceleration`, `heading`, `heading_change_rate` | from `PredictHelper`, NaN where insufficient history (e.g. first sample in a scene) |
| `past_x_0..{PAST_STEPS - 1}`, `past_y_0..{PAST_STEPS - 1}` | {PAST_SECONDS}s of past xy, agent frame, chronological (index 0 = oldest); NaN-padded if history is shorter |
| `future_x_0..{FUTURE_STEPS - 1}`, `future_y_0..{FUTURE_STEPS - 1}` | {FUTURE_SECONDS}s of future xy, agent frame, chronological ground truth — always fully present (rows without a full future are dropped) |
| `neighbor_dist_0..2`, `neighbor_rel_heading_0..2` | distance (m) and relative heading (rad) to the 3 nearest other agents (any category) within {NEIGHBOR_RADIUS_MAX}m, sorted by distance; NaN-padded if fewer than 3 present |

## Difficulty rule

An example is **hard** if `near_intersection` is True OR `neighbor_density_count >= {DENSITY_THRESHOLD}`;
otherwise **easy**.

## Splits

- `test` = official nuScenes `mini_val` scenes (2 scenes), held out untouched.
- `val` = last {N_VAL_SCENES_FROM_TRAIN} scenes (alphabetically) of official `mini_train`, carved out for model selection.
- `train` = remaining `mini_train` scenes.

All assignment is by scene name, so no scene contributes samples to more than one split.
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-scenes-from-train", type=int, default=N_VAL_SCENES_FROM_TRAIN)
    args = parser.parse_args()

    try:
        nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=True)
    except Exception as exc:  # noqa: BLE001 - surface a clear pointer, not a stack trace
        print(f"ERROR: could not load NuScenes from {args.dataroot}: {exc}", file=sys.stderr)
        print("Run `trajflow-download` first to check/complete the dataset setup.", file=sys.stderr)
        return 1

    helper = PredictHelper(nusc)
    split_of_scene = build_scene_splits(args.val_scenes_from_train)
    scene_of_sample = build_scene_lookup(nusc)

    rows, stats = extract_examples(nusc, helper, split_of_scene, scene_of_sample)
    df = pd.DataFrame(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    write_schema_doc(SCHEMA_PATH)

    print("\n=== Preprocessing summary ===")
    print(f"Candidate vehicle instance-samples: {stats['candidates']}")
    print(f"Skipped (future < {FUTURE_STEPS} steps): {stats['skipped_short_future']}")
    print(f"Kept: {stats['kept']}")

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        out_path = args.out / f"{split}.parquet"
        split_df.to_parquet(out_path, index=False)
        counts = split_df["difficulty"].value_counts().to_dict()
        n_scenes = split_df["scene_name"].nunique()
        print(
            f"  {split}: {len(split_df)} rows, {n_scenes} scenes, "
            f"easy={counts.get('easy', 0)}, hard={counts.get('hard', 0)} -> {out_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
