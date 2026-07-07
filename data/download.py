"""Fetch / verify the nuScenes mini (v1.0-mini) dataset + map expansion.

nuScenes requires a free account and a license click-through on
nuscenes.org that cannot be scripted. This tool cannot log in or accept
the license on your behalf, so it only does two things:

1. Checks whether v1.0-mini (plus the vector map expansion pack, needed
   for intersection lookups in preprocessing) is already present at the
   expected location and, if so, verifies it loads via the devkit.
2. If anything is missing, prints exact manual download instructions and
   exits with a non-zero status so the pipeline stops cleanly rather than
   guessing at a broken path.

Usage:
    python data/download.py                          # check / verify only
    python data/download.py --archive PATH            # extract an archive
                                                       # (.tgz or .zip) into
                                                       # place, then verify
"""

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent / "nuscenes"
VERSION = "v1.0-mini"

REQUIRED_SUBDIRS = ["maps", "samples", "sweeps", VERSION]
# At least one of these vector map expansion files must exist. This is a
# SEPARATE download from the basic Mini archive (see instructions below) —
# it's what powers intersection-proximity lookups via NuScenesMap.
MAP_EXPANSION_FILES = [
    "singapore-onenorth.json",
    "singapore-hollandvillage.json",
    "singapore-queenstown.json",
    "boston-seaport.json",
]

INSTRUCTIONS = f"""
nuScenes mini is not fully present at:
    {DATA_ROOT}

This dataset requires a free account and a license click-through that
cannot be automated. To proceed:

  1. Go to https://www.nuscenes.org/nuscenes#download and create a free
     account (or log in).
  2. Accept the nuScenes terms of use.
  3. Under "Full dataset (v1.0) > Mini", download "Mini" (~4 GB): a single
     archive (v1.0-mini.tgz) bundling rasterized maps, samples, sweeps,
     and the v1.0-mini metadata.
  4. ALSO under "Map expansion", download the "Map expansion pack (v1.3)"
     (~35 MB, a separate .zip). This project classifies scenes as
     easy/hard using the vector map API (intersection lookups), which
     needs this expansion pack — the basic Mini archive alone is not
     enough for that.
  5. Place both downloaded archives somewhere on disk, then run:

       python data/download.py --archive /path/to/v1.0-mini.tgz
       python data/download.py --archive /path/to/nuscenes-map-expansion-v1.3.zip

     which will extract each into {DATA_ROOT} and verify everything loads.

     (Alternatively, extract them yourself so that this directory exists:
       {DATA_ROOT}/maps/expansion/*.json   <- from the map expansion pack
       {DATA_ROOT}/samples/
       {DATA_ROOT}/sweeps/
       {DATA_ROOT}/v1.0-mini/
      then re-run `python data/download.py` with no arguments to verify.)

Once the files are in place, re-run this script (or continue the
pipeline) and it will proceed automatically.
"""


def has_required_layout(root: Path) -> bool:
    if not all((root / sub).is_dir() for sub in REQUIRED_SUBDIRS):
        return False
    expansion_dir = root / "maps" / "expansion"
    return expansion_dir.is_dir() and any(
        (expansion_dir / fname).is_file() for fname in MAP_EXPANSION_FILES
    )


def extract_archive(archive_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} -> {dest} ...")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest)


def verify_with_devkit(root: Path) -> None:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap

    print(f"Loading NuScenes(version='{VERSION}', dataroot='{root}') ...")
    nusc = NuScenes(version=VERSION, dataroot=str(root), verbose=True)
    n_scenes = len(nusc.scene)
    n_samples = len(nusc.sample)
    print(f"OK: loaded {n_scenes} scenes, {n_samples} samples.")
    if n_scenes == 0:
        raise RuntimeError("Devkit loaded but found 0 scenes — check the archive contents.")

    map_name = nusc.get("log", nusc.scene[0]["log_token"])["location"]
    print(f"Loading NuScenesMap(map_name='{map_name}') to verify map expansion ...")
    nusc_map = NuScenesMap(dataroot=str(root), map_name=map_name)
    print(f"OK: loaded map '{map_name}' with {len(nusc_map.road_segment)} road segments.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to a downloaded archive (v1.0-mini.tgz or the map "
        "expansion .zip) to extract into data/nuscenes/",
    )
    args = parser.parse_args()

    if args.archive is not None:
        if not args.archive.is_file():
            print(f"ERROR: archive not found at {args.archive}", file=sys.stderr)
            return 1
        extract_archive(args.archive, DATA_ROOT)

    if not has_required_layout(DATA_ROOT):
        print(INSTRUCTIONS)
        return 1

    verify_with_devkit(DATA_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
