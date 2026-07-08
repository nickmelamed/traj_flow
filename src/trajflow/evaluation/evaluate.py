"""Shared eval helpers: load processed splits and log metrics to
results/metrics_comparison.md.

Every model script (baseline_cv, baseline_xgb, train_pretrain, finetune)
imports `load_split`, `future_xy`, and `log_metrics` from here so the
comparison table is built up consistently across phases.
"""

import numpy as np
import pandas as pd

from trajflow.paths import PROCESSED_DIR, RESULTS_PATH

FUTURE_STEPS = 12

HEADER = (
    "| Phase | Model | Eval Split | Difficulty | N | minADE (m) | minFDE (m) | Miss Rate @2m | Notes |\n"
)
SEP = "|---|---|---|---|---|---|---|---|---|\n"
PREAMBLE = (
    "# TrajFlow — Metrics Comparison\n\n"
    "Every model / eval-split / difficulty-filter combination run so far, logged honestly\n"
    "(including underperforming results — nothing is rounded or omitted to look better).\n"
    "See `data/SCHEMA.md` for column definitions and `CLAUDE.md` for the phase plan.\n\n"
)


def load_split(split: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")


def filter_difficulty(df: pd.DataFrame, difficulty: str) -> pd.DataFrame:
    if difficulty == "all":
        return df
    return df[df["difficulty"] == difficulty]


def future_xy(df: pd.DataFrame) -> np.ndarray:
    """Returns [N, FUTURE_STEPS, 2] ground-truth future trajectories."""
    xs = df[[f"future_x_{i}" for i in range(FUTURE_STEPS)]].to_numpy(dtype=float)
    ys = df[[f"future_y_{i}" for i in range(FUTURE_STEPS)]].to_numpy(dtype=float)
    return np.stack([xs, ys], axis=-1)


def _read_existing_rows() -> list[str]:
    if not RESULTS_PATH.exists():
        return []
    lines = RESULTS_PATH.read_text().splitlines(keepends=True)
    # data rows are the ones starting with "| " that come after the separator row
    data_rows = []
    in_table = False
    for line in lines:
        if line.startswith("|---"):
            in_table = True
            continue
        if in_table and line.startswith("|"):
            data_rows.append(line)
    return data_rows


def _row_key(line: str) -> tuple:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    # Phase, Model, Eval Split, Difficulty identify a unique row
    return tuple(cells[:4])


def _sanitize_cell(value: str) -> str:
    """Neutralize characters that would corrupt the pipe-delimited markdown
    table (a literal "|" would shift every later column; a newline would
    split one row into two lines `_read_existing_rows` can't reassemble).
    """
    return str(value).replace("|", "/").replace("\n", " ").replace("\r", " ")


def log_metrics(phase, model: str, eval_split: str, difficulty: str, metrics: dict, notes: str = "") -> None:
    """Append (or replace, if the same phase/model/split/difficulty combo
    was already logged in an earlier run) a row in results/metrics_comparison.md.
    """
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model, eval_split, difficulty, notes = (
        _sanitize_cell(model),
        _sanitize_cell(eval_split),
        _sanitize_cell(difficulty),
        _sanitize_cell(notes),
    )
    new_line = (
        f"| {phase} | {model} | {eval_split} | {difficulty} | {metrics['N']} | "
        f"{metrics['minADE']:.4f} | {metrics['minFDE']:.4f} | {metrics['MissRate@2m']:.4f} | {notes} |\n"
    )
    new_key = _row_key(new_line)

    rows = _read_existing_rows()
    rows = [r for r in rows if _row_key(r) != new_key]
    rows.append(new_line)

    with open(RESULTS_PATH, "w") as f:
        f.write(PREAMBLE)
        f.write(HEADER)
        f.write(SEP)
        f.writelines(rows)
