"""Fetch / verify a nuScenes dataset version (v1.0-mini by default) + map
expansion.

nuScenes requires a free account and a license click-through on
nuscenes.org that cannot be scripted. This tool cannot log in or accept
the license on your behalf, so it only does two things:

1. Checks whether the requested version (plus the vector map expansion
   pack, needed for intersection lookups in preprocessing) is already
   present at the expected location and, if so, verifies it loads via the
   devkit.
2. If anything is missing, prints exact manual download instructions and
   exits with a non-zero status so the pipeline stops cleanly rather than
   guessing at a broken path.

Usage:
    trajflow-download                          # check / verify v1.0-mini
    trajflow-download --archive PATH           # extract an archive
                                                       # (.tgz or .zip) into
                                                       # place, then verify
    trajflow-download --version v1.0-trainval  # check / verify a scale-up
                                                # version instead (see below)

## Scaling up past mini (v1.0-trainval)

This project's own feature pipeline (`data/preprocess.py`) never reads the
`samples/`/`sweeps/` sensor blob directories -- every call it makes
(`PredictHelper.get_past_for_agent`/`get_future_for_agent`/etc.,
`NuScenesMap`) reads only the JSON metadata tables and the small
per-location rasterized map PNGs (verified directly against the installed
nuscenes-devkit source: grepping every method this pipeline calls for any
`samples`/`sweeps`/`.filename`/`get_sample_data_path` reference turns up
nothing). That means scaling from mini's 10 scenes up to trainval's 850
(700 train + 150 val, see `nuscenes.utils.splits`) should NOT require the
~350 GB of camera/lidar "File blobs" nuScenes also offers -- nuScenes'
much smaller "Metadata"-only archive (JSON tables, no sensor data) should
be enough, and the 4 map PNGs already downloaded for mini cover every
trainval scene too (nuScenes has exactly 4 physical locations total,
shared between mini and trainval).

To try it: download nuScenes' "Metadata" archive for Trainval (NOT "File
blobs") from the same download page as mini, extract it so a
`v1.0-trainval/*.json` directory exists under this same `data/nuscenes/`
root (`trajflow-download --archive PATH --version v1.0-trainval` handles
the extraction), then re-run `trajflow-download --version v1.0-trainval`
to verify. No second map-expansion download is needed if you already have
mini's. `preprocess.py --version v1.0-trainval` then uses `train`/`val`
in place of `mini_train`/`mini_val` automatically.

This is unverified against the real archive (fetching it requires the
account/license click-through this tool can't automate) -- if nuScenes'
actual Metadata archive turns out to bundle something this doesn't expect,
`has_required_layout`/`verify_with_devkit` below are the two functions to
adjust. Also worth checking before committing to all 850 scenes:
`preprocess.py`'s feature-extraction loop is a plain per-instance Python
loop with a map-radius query per example, so wall-clock preprocessing time
should scale roughly linearly with scene count -- 850 scenes is ~85x
mini's, so expect preprocessing (and, downstream, training over a
correspondingly larger dataset) to take meaningfully longer, not stay at
mini's few-minutes scale.
"""

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

from trajflow.paths import NUSCENES_ROOT as DATA_ROOT

DEFAULT_VERSION = "v1.0-mini"
# At least one of these vector map expansion files must exist. This is a
# SEPARATE download from the basic Mini archive (see instructions below) —
# it's what powers intersection-proximity lookups via NuScenesMap. Shared
# across every version (mini or trainval) since nuScenes has only 4
# physical map locations total.
MAP_EXPANSION_FILES = [
    "singapore-onenorth.json",
    "singapore-hollandvillage.json",
    "singapore-queenstown.json",
    "boston-seaport.json",
]


def _required_subdirs(version: str) -> list[str]:
    """v1.0-mini's standard archive bundles samples/sweeps alongside the
    metadata, so requiring them is a meaningful "did this actually extract
    right" check for the default path. Any other version is assumed to be
    a metadata-only scale-up (see module docstring) where samples/sweeps
    are neither expected nor needed by this pipeline.
    """
    if version == "v1.0-mini":
        return ["maps", "samples", "sweeps", version]
    return ["maps", version]


def _instructions(version: str) -> str:
    if version == "v1.0-mini":
        return f"""
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

       trajflow-download --archive /path/to/v1.0-mini.tgz
       trajflow-download --archive /path/to/nuscenes-map-expansion-v1.3.zip

     which will extract each into {DATA_ROOT} and verify everything loads.

     (Alternatively, extract them yourself so that this directory exists:
       {DATA_ROOT}/maps/expansion/*.json   <- from the map expansion pack
       {DATA_ROOT}/samples/
       {DATA_ROOT}/sweeps/
       {DATA_ROOT}/v1.0-mini/
      then re-run `trajflow-download` with no arguments to verify.)

Once the files are in place, re-run this script (or continue the
pipeline) and it will proceed automatically.
"""
    return f"""
nuScenes {version} metadata is not fully present at:
    {DATA_ROOT}

This is a scale-up beyond mini (see the "Scaling up" section of this
script's module docstring for why only a metadata download should be
needed, not the full ~350 GB of sensor blobs). To proceed:

  1. Go to https://www.nuscenes.org/nuscenes#download (same free
     account/license click-through as mini).
  2. Under "Full dataset (v1.0) > Trainval", download the "Metadata"
     archive ONLY -- NOT any of the "File blobs" parts (those are the
     camera/lidar sensor data this pipeline never reads).
  3. If you don't already have mini's map expansion pack downloaded, also
     grab "Map expansion pack (v1.3)" under "Map expansion" (~35 MB) --
     skip this if `{DATA_ROOT}/maps/expansion/` already has files from a
     prior mini setup, since nuScenes' 4 map locations are shared across
     every version.
  4. Place the archive(s) somewhere on disk, then run:

       trajflow-download --archive /path/to/metadata-archive --version {version}

     which extracts into {DATA_ROOT} (same root as mini -- maps are
     shared) and verifies everything loads.

     (Alternatively, extract it yourself so that this directory exists:
       {DATA_ROOT}/{version}/*.json
      then re-run `trajflow-download --version {version}` to verify.)

This path is unverified against the real archive layout (see module
docstring) -- if extraction or verification fails in an unexpected way,
that's the first thing to check.
"""


def has_required_layout(root: Path, version: str) -> bool:
    if not all((root / sub).is_dir() for sub in _required_subdirs(version)):
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


def verify_with_devkit(root: Path, version: str) -> None:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap

    print(f"Loading NuScenes(version='{version}', dataroot='{root}') ...")
    nusc = NuScenes(version=version, dataroot=str(root), verbose=True)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to a downloaded archive (v1.0-mini.tgz, a trainval Metadata archive, or the map "
        "expansion .zip) to extract into data/nuscenes/",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help=f"nuScenes version to check/verify (default: {DEFAULT_VERSION}). Use e.g. v1.0-trainval to "
        "scale up -- see this script's module docstring for what that needs (metadata only, not the "
        "sensor-blob 'File blobs' download).",
    )
    args = parser.parse_args()

    if args.archive is not None:
        if not args.archive.is_file():
            print(f"ERROR: archive not found at {args.archive}", file=sys.stderr)
            return 1
        extract_archive(args.archive, DATA_ROOT)

    if not has_required_layout(DATA_ROOT, args.version):
        print(_instructions(args.version))
        return 1

    verify_with_devkit(DATA_ROOT, args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
