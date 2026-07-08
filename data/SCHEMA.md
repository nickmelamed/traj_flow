# `data/processed/{train,val,test}.parquet` schema

Generated from nuScenes version `v1.0-mini`.

One row = one (vehicle instance, sample) trajectory-prediction example.

| column | meaning |
|---|---|
| `instance_token`, `sample_token` | nuScenes identifiers for the agent / timestep |
| `scene_name`, `split` | source scene and its assigned split (train/val/test), scene-level — no leakage |
| `category_name` | nuScenes category, e.g. `vehicle.car` |
| `map_name` | nuScenes map location |
| `difficulty` | `easy` or `hard` — see classification rule below |
| `near_intersection` | bool, agent is within 4.0m of a `road_segment` with `is_intersection=True` |
| `neighbor_density_count` | # other agents within 10.0m |
| `velocity`, `acceleration`, `heading`, `heading_change_rate` | from `PredictHelper`, NaN where insufficient history (e.g. first sample in a scene) |
| `past_x_0..3`, `past_y_0..3` | 2.0s of past xy, agent frame, chronological (index 0 = oldest); NaN-padded if history is shorter |
| `future_x_0..11`, `future_y_0..11` | 6.0s of future xy, agent frame, chronological ground truth — always fully present (rows without a full future are dropped) |
| `neighbor_dist_0..2`, `neighbor_rel_heading_0..2` | distance (m) and relative heading (rad) to the 3 nearest other agents (any category) within 60.0m, sorted by distance; NaN-padded if fewer than 3 present |

## Difficulty rule

An example is **hard** if `near_intersection` is True OR `neighbor_density_count >= 5`;
otherwise **easy**.

## Splits

- `test` = official nuScenes `mini_val` scenes, held out untouched.
- `val` = last 2 scenes (alphabetically) of official `mini_train`, carved out for model selection.
- `train` = remaining `mini_train` scenes.

All assignment is by scene name, so no scene contributes samples to more than one split.
