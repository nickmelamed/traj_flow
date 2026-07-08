"""HITL review app for the top ~10% most uncertain hard-scene TRAINING
predictions (flagged by hitl/flag_uncertain.py).

Run with:
    streamlit run hitl/review_app.py

For each flagged example the reviewer sees: the agent's past trajectory,
nearby lane geometry (map context), the ground-truth future, the
transformer's K=6 candidate futures (opacity = mode probability), and the
XGBoost baseline's prediction -- then can accept the ground truth as-is,
correct it by adjusting the car's position at 3 key moments (2s/4s/6s
into the future, auto-smoothed into the 12 required timesteps via a
cubic spline, with a live plot preview), or tag a failure mode. Every
decision is persisted to corrections/corrections.parquet, keyed by
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
import plotly.graph_objects as go
import streamlit as st
import torch
from scipy.interpolate import CubicSpline

from models.baseline_xgb import MODEL_PATH as XGB_MODEL_PATH
from models.baseline_xgb import make_features as xgb_make_features
from evaluation.evaluate import load_split

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
FUTURE_SECONDS = 6.0
DT = FUTURE_SECONDS / FUTURE_STEPS  # 0.5s per step, matching the rest of the codebase
KEY_TIMES = [2.0, 4.0, 6.0]  # seconds; the 3 adjustable checkpoints
KEY_INDICES = [int(round(t / DT)) - 1 for t in KEY_TIMES]  # corresponding 0-indexed rows in future_x/y


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


def spline_from_keypoints(key_points: list) -> np.ndarray:
    """Fit a natural cubic spline from the origin (agent's current position,
    t=0) through the 3 key points at KEY_TIMES seconds, then evaluate it at
    the 12 canonical timesteps. Returns [FUTURE_STEPS, 2].
    """
    times = np.concatenate([[0.0], KEY_TIMES])
    xs = np.concatenate([[0.0], [p[0] for p in key_points]])
    ys = np.concatenate([[0.0], [p[1] for p in key_points]])

    spline_x = CubicSpline(times, xs)
    spline_y = CubicSpline(times, ys)

    canonical_times = np.arange(1, FUTURE_STEPS + 1) * DT
    return np.stack([spline_x(canonical_times), spline_y(canonical_times)], axis=-1)


def make_plot(row, traj: np.ndarray, logits: np.ndarray, xgb_pred: np.ndarray, map_polylines: list, spline_xy: np.ndarray = None, key_points: list = None):
    fig = go.Figure()

    for poly in map_polylines:
        fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], mode="lines", line=dict(color="lightgray", width=1), hoverinfo="skip", showlegend=False))

    past_x = np.array([row[f"past_x_{i}"] for i in range(PAST_STEPS)])
    past_y = np.array([row[f"past_y_{i}"] for i in range(PAST_STEPS)])
    valid = ~np.isnan(past_x)
    fig.add_trace(go.Scatter(x=np.r_[past_x[valid], 0], y=np.r_[past_y[valid], 0], mode="lines+markers", line=dict(color="royalblue"), name="past"))

    future_x = np.array([row[f"future_x_{i}"] for i in range(FUTURE_STEPS)])
    future_y = np.array([row[f"future_y_{i}"] for i in range(FUTURE_STEPS)])

    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    for k in range(traj.shape[0]):
        fig.add_trace(go.Scatter(
            x=np.r_[0, traj[k, :, 0]], y=np.r_[0, traj[k, :, 1]], mode="lines",
            line=dict(color=f"rgba(255,140,0,{0.15 + 0.85 * float(probs[k]):.3f})", width=2),
            name="transformer modes" if k == 0 else None, showlegend=(k == 0),
        ))

    fig.add_trace(go.Scatter(x=np.r_[0, xgb_pred[:, 0]], y=np.r_[0, xgb_pred[:, 1]], mode="lines", line=dict(color="crimson", width=2, dash="dash"), name="XGBoost"))

    # Ground truth is added AFTER transformer modes / XGBoost (and given a
    # bold, otherwise-unused color) so it draws on top and isn't hidden when
    # a prediction line sits almost exactly on top of it -- with black,
    # drawn earlier, that was silently swallowing the ground truth line in
    # examples where the transformer's prediction closely matched reality.
    fig.add_trace(go.Scatter(x=np.r_[0, future_x], y=np.r_[0, future_y], mode="lines", line=dict(color="magenta", width=3), name="ground truth"))

    if spline_xy is not None:
        fig.add_trace(go.Scatter(x=np.r_[0, spline_xy[:, 0]], y=np.r_[0, spline_xy[:, 1]], mode="lines", line=dict(color="green", width=3), name="your correction"))

    if key_points is not None:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in key_points], y=[p[1] for p in key_points], mode="markers+text",
            marker=dict(color="green", size=12, symbol="circle", line=dict(color="white", width=1)),
            text=[f"{t:g}s" for t in KEY_TIMES], textposition="top center", name="your key points",
        ))

    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color="magenta", size=16, symbol="x", line=dict(width=3, color="magenta")), name="current position"))

    fig.update_layout(
        template="plotly_white",  # explicit light background, independent of Streamlit's theme
        # (dark theme was rendering the black ground-truth line invisible against a near-black plot bg)
        xaxis_title="x (m, agent frame)", yaxis_title="y (m, agent frame, heading = +y)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title=dict(text=f"{row['scene_name']} | difficulty={row['difficulty']} | uncertainty={row['uncertainty_score']:.2f}", y=0.98),
        height=650, margin=dict(l=10, r=10, t=60, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    )
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

    idx = st.session_state.idx
    row = flagged.iloc[idx]
    key = (row["instance_token"], row["sample_token"])
    already_reviewed = key in reviewed_keys
    st.write(
        f"### Example {idx + 1} / {n_total}"
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

    future_x = np.array([row[f"future_x_{i}"] for i in range(FUTURE_STEPS)])
    future_y = np.array([row[f"future_y_{i}"] for i in range(FUTURE_STEPS)])

    col_plot, col_review = st.columns([2, 1])

    # NOTE: col_review's widgets are read here, BEFORE col_plot is drawn
    # below, even though col_plot renders on the left -- st.columns() fixes
    # each column's on-screen position independently of which one your code
    # writes to first, so this lets the plot include a live preview built
    # from the values just entered.
    with col_review:
        st.write("**Scene context**")
        st.write(f"- Map: `{row['map_name']}`")
        st.write(f"- Near intersection: `{row['near_intersection']}`")
        st.write(f"- Neighbor density: `{row['neighbor_density_count']}`")
        st.write(f"- Mode spread: `{row['mode_endpoint_spread']:.2f}` m")
        st.write(f"- XGB/Transformer divergence: `{row['xgb_transformer_divergence']:.2f}` m")
        st.write(f"- Uncertainty score (percentile rank): `{row['uncertainty_score']:.3f}`")

        decision = st.radio(
            "Decision", ["Accept ground truth as-is", "Correct trajectory", "Skip"], key=f"decision_{idx}"
        )

        key_points = None
        spline_xy = None
        if decision == "Correct trajectory":
            st.caption(
                "Adjust the car's position at 3 key moments below; the plot updates live with a green "
                "preview and a smooth path is auto-fit through them to fill the 12-row table beneath it "
                "(pre-filled from the ground truth -- only the numbers you actually change matter)."
            )
            key_points = []
            cols = st.columns(3)
            for t, t_idx, col in zip(KEY_TIMES, KEY_INDICES, cols):
                with col:
                    st.write(f"**~{t:g}s**")
                    kx = st.number_input("x", value=float(future_x[t_idx]), key=f"kp_{idx}_{t}_x", format="%.2f")
                    ky = st.number_input("y", value=float(future_y[t_idx]), key=f"kp_{idx}_{t}_y", format="%.2f")
                    key_points.append((kx, ky))
            spline_xy = spline_from_keypoints(key_points)
            table_default = spline_xy
        else:
            table_default = np.stack([future_x, future_y], axis=-1)

        st.caption("Full 12-waypoint table (auto-filled above; edit individual cells here if you want finer control).")
        edit_df = pd.DataFrame({"x": table_default[:, 0], "y": table_default[:, 1]})
        # Keying on the key-point values forces the editor to re-initialize
        # from the latest spline fit whenever they change; manual edits made
        # at a given set of key-point values are preserved until they change.
        editor_key_suffix = hash(tuple(key_points[0] + key_points[1] + key_points[2])) if key_points else "flat"
        edited = st.data_editor(edit_df, key=f"editor_{idx}_{editor_key_suffix}", num_rows="fixed")

        edited_x = edited["x"].to_numpy(dtype=float)
        edited_y = edited["y"].to_numpy(dtype=float)
        max_change = float(np.max(np.abs(np.r_[edited_x - future_x, edited_y - future_y])))
        n_changed_rows = int(np.sum((edited_x != future_x) | (edited_y != future_y)))
        if decision == "Correct trajectory":
            if max_change == 0.0:
                st.warning("No changes detected yet — adjust the key points above before saving.")
            else:
                st.info(f"{n_changed_rows} / {FUTURE_STEPS} waypoints changed (max change {max_change:.2f} m).")

        failure_mode = st.selectbox("Failure mode tag", FAILURE_MODES, key=f"failure_mode_{idx}")
        notes = st.text_area("Reviewer notes", key=f"notes_{idx}")

        submitted = st.button("Save review", key=f"submit_{idx}")

    with col_plot:
        fig = make_plot(row, traj, logits, xgb_pred, polylines, spline_xy=spline_xy, key_points=key_points)
        st.plotly_chart(fig, key=f"plot_{idx}", theme=None)

    if submitted:
        if decision == "Correct trajectory":
            corrected_x = edited_x
            corrected_y = edited_y
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
