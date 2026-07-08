from trajflow.data.download import _required_subdirs
from trajflow.data.preprocess import build_scene_splits


def test_build_scene_splits_uses_mini_lists_by_default():
    split_of_scene = build_scene_splits(val_scenes_from_train=2, version="v1.0-mini")
    assert set(split_of_scene.values()) == {"train", "val", "test"}
    # mini_train has 8 scenes: 6 train + 2 val; mini_val (test) has 2 scenes.
    counts = {v: list(split_of_scene.values()).count(v) for v in ["train", "val", "test"]}
    assert counts == {"train": 6, "val": 2, "test": 2}


def test_build_scene_splits_uses_full_trainval_lists_for_other_versions():
    split_of_scene = build_scene_splits(val_scenes_from_train=50, version="v1.0-trainval")
    counts = {v: list(split_of_scene.values()).count(v) for v in ["train", "val", "test"]}
    # official train=700, val=150 scenes; 50 carved from train's tail into our val.
    assert counts == {"train": 650, "val": 50, "test": 150}


def test_no_scene_appears_in_more_than_one_split():
    split_of_scene = build_scene_splits(val_scenes_from_train=2, version="v1.0-mini")
    assert len(split_of_scene) == len(set(split_of_scene.keys()))  # each scene has exactly one split assignment


def test_max_scenes_caps_total_and_preserves_train_test_ratio():
    split_of_scene = build_scene_splits(val_scenes_from_train=10, version="v1.0-trainval", max_scenes=100)
    assert len(split_of_scene) == 100
    counts = {v: list(split_of_scene.values()).count(v) for v in ["train", "val", "test"]}
    # official ratio is 700:150 train:test (~82:18); with max_scenes=100 that's ~82 train-pool
    # scenes (val carved from its tail) and ~18 test scenes.
    assert counts["train"] + counts["val"] == 82
    assert counts["val"] == 10
    assert counts["test"] == 18


def test_max_scenes_truncation_is_deterministic_not_random():
    a = build_scene_splits(val_scenes_from_train=2, version="v1.0-trainval", max_scenes=50)
    b = build_scene_splits(val_scenes_from_train=2, version="v1.0-trainval", max_scenes=50)
    assert a == b


def test_max_scenes_none_matches_unbounded_behavior():
    capped_at_everything = build_scene_splits(val_scenes_from_train=2, version="v1.0-mini", max_scenes=10)
    uncapped = build_scene_splits(val_scenes_from_train=2, version="v1.0-mini", max_scenes=None)
    assert capped_at_everything == uncapped


def test_required_subdirs_only_requires_samples_sweeps_for_mini():
    mini_dirs = _required_subdirs("v1.0-mini")
    trainval_dirs = _required_subdirs("v1.0-trainval")
    assert "samples" in mini_dirs and "sweeps" in mini_dirs
    assert "samples" not in trainval_dirs and "sweeps" not in trainval_dirs
    assert "maps" in trainval_dirs and "v1.0-trainval" in trainval_dirs
