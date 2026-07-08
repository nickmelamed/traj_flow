"""Shared filesystem paths for the trajflow pipeline.

All data/artifact directories live at the repo root, alongside `src/`
rather than inside it, so they survive package reinstalls and stay easy
to gitignore independently of the code. `REPO_ROOT` is derived from this
file's own location, which is correct for an editable install
(`pip install -e .`) since that's the only supported way to run this
project.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
NUSCENES_ROOT = DATA_DIR / "nuscenes"
PROCESSED_DIR = DATA_DIR / "processed"
SCHEMA_PATH = DATA_DIR / "SCHEMA.md"

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
FLAGGED_PATH = ARTIFACTS_DIR / "flagged.parquet"

CORRECTIONS_DIR = REPO_ROOT / "corrections"
CORRECTIONS_PATH = CORRECTIONS_DIR / "corrections.parquet"

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "metrics_comparison.md"
FIGURES_DIR = RESULTS_DIR / "figures"
