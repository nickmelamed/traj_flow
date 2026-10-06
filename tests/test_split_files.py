import pandas as pd

from trajflow.data.preprocess import build_scene_splits


def _rows_for_scenes(scene_names):
    return pd.DataFrame({
        "scene_name": [s for s in scene_names for _ in range(3)],
        "sample_token": [f"{s}-{i}" for s in scene_names for i in range(3)],
    })


def test_written_split_files_share_no_scene(tmp_path):
    split_of_scene = build_scene_splits(val_scenes_from_train=2, version="v1.0-mini")
    df = _rows_for_scenes(sorted(split_of_scene))
    df["split"] = df["scene_name"].map(split_of_scene)

    for split in ["train", "val", "test"]:
        df[df["split"] == split].to_parquet(tmp_path / f"{split}.parquet", index=False)

    scenes = {
        split: set(pd.read_parquet(tmp_path / f"{split}.parquet")["scene_name"])
        for split in ["train", "val", "test"]
    }
    assert all(scenes.values())
    assert scenes["train"].isdisjoint(scenes["val"])
    assert scenes["train"].isdisjoint(scenes["test"])
    assert scenes["val"].isdisjoint(scenes["test"])
    assert scenes["test"] == set(build_scene_splits(2).keys()) - scenes["train"] - scenes["val"]
