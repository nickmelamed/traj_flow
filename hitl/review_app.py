"""HITL review app for the top ~10% most uncertain hard-scene TRAINING
predictions (flagged by hitl/flag_uncertain.py).

Run with:
    streamlit run hitl/review_app.py

For each flagged example the reviewer sees: the agent's past trajectory,
nearby lane geometry (map context), the ground-truth future, the
transformer's K=6 candidate futures (opacity = mode probability), and the
XGBoost baseline's prediction -- then can accept the ground truth as-is,
edit the future waypoints directly, or tag a failure mode. Every decision
is persisted to corrections/corrections.parquet, keyed by
(instance_token, sample_token) so re-running the app resumes where you
left off (re-submitting the same example overwrites its prior entry).
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Must be set before torch/xgboost are imported: loading both libraries in
# the same process on this macOS setup deadlocks otherwise. See the same
# note in hitl/flag_uncertain.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from models.baseline_xgb import MODEL_PATH as XGB_MODEL_PATH
from models.baseline_xgb import make_features as xgb_make_features
from evaluation.evaluate import load_split

import matplotlib.pyplot as plt
import torch

from nuscenes.map_expansion.arcline_path_utils import discretize_lane
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import convert_global_coords_to_local

from data.preprocess import DEFAULT_DATAROOT, FUTURE_STEPS, PAST_STEPS
from models.transformer import TrajectoryDataset, TrajectoryTransformer

FLAGGED_PATH = Path(__file__).resolve().parent / "flagged.parquet"
CORRECTIONS_PATH = Path(__file__).resolve().parent.parent / "corrections" / "corrections.parquet"
TRANSFORMER_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "finetuned_v1.pt"
MAP_RADIUS = 40.0
FAILURE_MODES = ["none", "occlusion", "aggressive merge", "sensor noise", "map ambiguity"]


@st.cache_resource
def load_nusc():
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DEFAULT_DATAROOT), verbose=False)
    helper = PredictHelper(nusc)
    return nusc, helper


@st.cache_resource
def load_transformer() -> TrajectoryTransformer:
    model = TrajectoryTransformer()
    model.load_state_dict(torch.load(TRANSFORMER_CHECKPOINT))
    model.eval()
    return model


@st.cache_resource
def load_xgb():
    return joblib.load(XGB_MODEL_PATH)


@st.cache_data
def load_flagged() -> pd.DataFrame:
    flagged = pd.read_parquet(FLAGGED_PATH)
    flagged = flagged[flagged["needs_review"]].sort_values("uncertainty_score", ascending=False).reset_index(drop=True)
    train_df = load_split("train")
    merged = flagged.merge(train_df, on=["instance_token", "sample_token", "scene_name", "difficulty"], how="left")
    return merged


def load_corrections() -> pd.DataFrame:
    if CORRECTIONS_PATH.exists():
        return pd.read_parquet(CORRECTIONS_PATH)
    return pd.DataFrame()


def save_correction(record: dict) -> None:
    existing = load_corrections()
    new_row = pd.DataFrame([record])
    if not existing.empty:
        is_same = (existing["instance_token"] == record["instance_token"]) & (
            existing["sample_token"] == record["sample_token"]
        )
        existing = existing[~is_same]
    combined = pd.concat([existing, new_row], ignore_index=True)
    CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(CORRECTIONS_PATH, index=False)


@st.cache_data
def nearby_lane_polylines(map_name: str, x: float, y: float, translation: tuple, rotation: tuple, radius: float):
    nusc_map = NuScenesMap(dataroot=str(DEFAULT_DATAROOT), map_name=map_name)
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
        local = convert_global_coords_to_local(pts[:, :2], translation, rotation)
        polylines.append(local)
    return polylines


def make_plot(row, traj: np.ndarray, logits: np.ndarray, xgb_pred: np.ndarray, map_polylines: list):
    fig, ax = plt.subplots(figsize=(6, 6))

    for poly in map_polylines:
        ax.plot(poly[:, 0], poly[:, 1], color="lightgray", linewidth=1, zorder=0)

    past_x = np.array([row[f"past_x_{i}"] for i in range(PAST_STEPS)])
    past_y = np.array([row[f"past_y_{i}"] for i in range(PAST_STEPS)])
    valid = ~np.isnan(past_x)
    ax.plot(np.r_[past_x[valid], 0], np.r_[past_y[valid], 0], "o-", color="tab:blue", label="past", zorder=3)

    future_x = np.array([row[f"future_x_{i}"] for i in range(FUTURE_STEPS)])
    future_y = np.array([row[f"future_y_{i}"] for i in range(FUTURE_STEPS)])
    ax.plot(np.r_[0, future_x], np.r_[0, future_y], "-", color="black", linewidth=2.5, label="ground truth", zorder=4)

    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    for k in range(traj.shape[0]):
        label = "transformer modes" if k == 0 else None
        ax.plot(
            np.r_[0, traj[k, :, 0]],
            np.r_[0, traj[k, :, 1]],
            "-",
            color="tab:orange",
            alpha=float(0.15 + 0.85 * probs[k]),
            linewidth=1.5,
            zorder=2,
            label=label,
        )

    ax.plot(np.r_[0, xgb_pred[:, 0]], np.r_[0, xgb_pred[:, 1]], "--", color="tab:red", label="XGBoost", zorder=2)

    ax.scatter([0], [0], color="black", marker="x", s=60, zorder=5)
    ax.set_xlabel("x (m, agent frame)")
    ax.set_ylabel("y (m, agent frame, heading = +y)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{row['scene_name']} | difficulty={row['difficulty']} | uncertainty={row['uncertainty_score']:.2f}")
    return fig


def main() -> None:
    st.set_page_config(page_title="TrajFlow HITL Review", layout="wide")
    st.title("TrajFlow — HITL Review")
    st.caption(
        "Reviewing the top ~10% most uncertain HARD-scene TRAINING examples "
        "(flagged by hitl/flag_uncertain.py from mode-endpoint spread + "
        "XGBoost/transformer disagreement)."
    )

    if not FLAGGED_PATH.exists():
        st.error(f"No flagged examples found at {FLAGGED_PATH}. Run `python hitl/flag_uncertain.py` first.")
        return

    flagged = load_flagged()
    corrections = load_corrections()
    reviewed_keys = set()
    if not corrections.empty:
        reviewed_keys = set(zip(corrections["instance_token"], corrections["sample_token"]))

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    n_total = len(flagged)
    n_reviewed = sum(
        1 for _, r in flagged.iterrows() if (r["instance_token"], r["sample_token"]) in reviewed_keys
    )
    st.progress(n_reviewed / n_total if n_total else 0)
    st.caption(f"{n_reviewed} / {n_total} flagged examples reviewed so far")

    col_prev, col_next, _ = st.columns([1, 1, 6])
    if col_prev.button("Previous") and st.session_state.idx > 0:
        st.session_state.idx -= 1
    if col_next.button("Next") and st.session_state.idx < n_total - 1:
        st.session_state.idx += 1
    st.session_state.idx = min(max(st.session_state.idx, 0), max(n_total - 1, 0))

    row = flagged.iloc[st.session_state.idx]
    key = (row["instance_token"], row["sample_token"])
    already_reviewed = key in reviewed_keys
    st.write(
        f"### Example {st.session_state.idx + 1} / {n_total}"
        + (" — _already reviewed (saving again will overwrite)_" if already_reviewed else "")
    )

    nusc, helper = load_nusc()
    model = load_transformer()
    xgb_model = load_xgb()

    ann = helper.get_sample_annotation(row["instance_token"], row["sample_token"])
    x, y = ann["translation"][0], ann["translation"][1]

    row_df = pd.DataFrame([row])
    dataset = TrajectoryDataset(row_df)
    with torch.no_grad():
        past_seq, context, _ = dataset[0]
        traj, logits = model(past_seq.unsqueeze(0), context.unsqueeze(0))
    traj = traj.squeeze(0).numpy()
    logits = logits.squeeze(0).numpy()

    xgb_pred = xgb_model.predict(xgb_make_features(row_df)).reshape(FUTURE_STEPS, 2)

    polylines = nearby_lane_polylines(row["map_name"], x, y, tuple(ann["translation"]), tuple(ann["rotation"]), MAP_RADIUS)

    col_plot, col_review = st.columns([2, 1])
    with col_plot:
        fig = make_plot(row, traj, logits, xgb_pred, polylines)
        st.pyplot(fig)

    with col_review:
        st.write("**Scene context**")
        st.write(f"- Map: `{row['map_name']}`")
        st.write(f"- Near intersection: `{row['near_intersection']}`")
        st.write(f"- Neighbor density: `{row['neighbor_density_count']}`")
        st.write(f"- Mode spread: `{row['mode_endpoint_spread']:.2f}` m")
        st.write(f"- XGB/Transformer divergence: `{row['xgb_transformer_divergence']:.2f}` m")
        st.write(f"- Uncertainty score (percentile rank): `{row['uncertainty_score']:.3f}`")

        future_x = np.array([row[f"future_x_{i}"] for i in range(FUTURE_STEPS)])
        future_y = np.array([row[f"future_y_{i}"] for i in range(FUTURE_STEPS)])

        with st.form(key=f"form_{st.session_state.idx}"):
            decision = st.radio("Decision", ["Accept ground truth as-is", "Correct trajectory", "Skip"])

            st.caption("Editable future waypoints (agent frame, meters) — only used if 'Correct trajectory' is chosen")
            edit_df = pd.DataFrame({"x": future_x, "y": future_y})
            edited = st.data_editor(edit_df, key=f"editor_{st.session_state.idx}", num_rows="fixed")

            failure_mode = st.selectbox("Failure mode tag", FAILURE_MODES)
            notes = st.text_area("Reviewer notes")

            submitted = st.form_submit_button("Save review")

        if submitted:
            if decision == "Correct trajectory":
                corrected_x = edited["x"].to_numpy(dtype=float)
                corrected_y = edited["y"].to_numpy(dtype=float)
            else:
                corrected_x = future_x
                corrected_y = future_y

            record = {
                "instance_token": row["instance_token"],
                "sample_token": row["sample_token"],
                "scene_name": row["scene_name"],
                "difficulty": row["difficulty"],
                "decision": decision,
                "failure_mode": failure_mode,
                "reviewer_notes": notes,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
            for i in range(FUTURE_STEPS):
                record[f"corrected_future_x_{i}"] = float(corrected_x[i])
                record[f"corrected_future_y_{i}"] = float(corrected_y[i])

            save_correction(record)
            st.success("Saved.")
            if st.session_state.idx < n_total - 1:
                st.session_state.idx += 1
            st.rerun()


if __name__ == "__main__":
    main()
