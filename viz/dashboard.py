"""Interactive results dashboard for TrajFlow.

Run with:
    streamlit run viz/dashboard.py

Two tabs:
  - Metrics: a sortable/filterable view of results/metrics_comparison.md
    plus grouped bar charts, so the whole project's results can be
    browsed without reading raw markdown.
  - Scene Browser: pick any test example and see history, ground truth,
    and every model's prediction (constant velocity, XGBoost, and all
    three transformer checkpoints) overlaid on the map, read-only (no
    review/correction controls -- that's hitl/review_app.py's job).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from viz.model_registry import MODEL_SPECS

from evaluation.evaluate import future_xy, load_split
from evaluation.metrics import batch_metrics

from nuscenes.map_expansion.arcline_path_utils import discretize_lane
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import convert_global_coords_to_local

from data.preprocess import DEFAULT_DATAROOT, FUTURE_STEPS, PAST_STEPS

METRICS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics_comparison.md"
MAP_RADIUS = 40.0
NUMERIC_COLS = ["minADE (m)", "minFDE (m)", "Miss Rate @2m"]


@st.cache_data
def parse_metrics_table(path: Path) -> pd.DataFrame:
    header = None
    rows = []
    in_table = False
    for line in path.read_text().splitlines():
        if line.startswith("|---"):
            in_table = True
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = cells
            elif in_table:
                rows.append(cells)
    df = pd.DataFrame(rows, columns=header)
    df["N"] = pd.to_numeric(df["N"], errors="coerce").astype("Int64")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_resource
def get_predict_fn(model_key: str):
    spec = next(s for s in MODEL_SPECS if s.key == model_key)
    return spec.loader()


@st.cache_resource
def load_nusc():
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DEFAULT_DATAROOT), verbose=False)
    helper = PredictHelper(nusc)
    return nusc, helper


@st.cache_data
def load_test_df() -> pd.DataFrame:
    df = load_split("test").reset_index(drop=True)
    gts = future_xy(df)
    df["displacement"] = np.linalg.norm(gts[:, -1, :], axis=-1)
    return df


@st.cache_data
def nearby_lane_polylines(map_name: str, x: float, y: float, translation: tuple, rotation: tuple, radius: float) -> list:
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
        polylines.append(convert_global_coords_to_local(pts[:, :2], translation, rotation))
    return polylines


def best_of_k(traj: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """traj: [K, T, 2], gt: [T, 2] -> [T, 2], the mode closest to gt (matches minADE)."""
    dists = np.linalg.norm(traj - gt[None, :, :], axis=-1).mean(axis=-1)
    return traj[dists.argmin()]


def metrics_tab() -> None:
    st.header("Metrics")
    df = parse_metrics_table(METRICS_PATH)

    col1, col2, col3 = st.columns(3)
    split = col1.selectbox("Eval split", sorted(df["Eval Split"].unique()), index=sorted(df["Eval Split"].unique()).index("test") if "test" in df["Eval Split"].unique() else 0)
    difficulties = sorted(df[df["Eval Split"] == split]["Difficulty"].unique())
    difficulty = col2.selectbox("Difficulty", difficulties, index=difficulties.index("all") if "all" in difficulties else 0)
    metric = col3.selectbox("Metric", NUMERIC_COLS)

    filtered = df[(df["Eval Split"] == split) & (df["Difficulty"] == difficulty)]

    st.subheader(f"{metric} by model — {split}/{difficulty}")
    fig = px.bar(
        filtered.sort_values(metric), x="Model", y=metric, color="Model", text_auto=".3f",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, theme=None, key="metrics_bar")

    show_all = st.checkbox(
        "Show all phases/splits/difficulties (unfiltered)", value=False,
        help="Unchecked: table matches the chart above (same split/difficulty). Checked: every row ever logged.",
    )
    st.subheader("Filtered table (matches the chart above)" if not show_all else "Full table (all rows, unfiltered)")
    st.caption(
        "Every model / eval-split / difficulty-filter combination logged so far. "
        "Click a column header to sort. Source: results/metrics_comparison.md."
    )
    table_df = df if show_all else filtered
    st.dataframe(table_df.drop(columns=["Notes"]).sort_values(metric), width="stretch", height=400)

    with st.expander("Notes for the rows above (methodology caveats, honest findings)"):
        for _, row in filtered.iterrows():
            if row["Notes"]:
                st.markdown(f"**{row['Model']}** ({row['Difficulty']}): {row['Notes']}")


def scene_browser_tab() -> None:
    st.header("Scene Browser")
    st.caption(
        "Pick any test example and see every model's prediction overlaid. Read-only -- "
        "for reviewing/correcting training labels, use `streamlit run hitl/review_app.py` instead."
    )

    test_df = load_test_df()
    gts = future_xy(test_df)

    col1, col2 = st.columns(2)
    difficulty_filter = col1.selectbox("Difficulty", ["all", "easy", "hard"], key="sb_difficulty")
    moving_only = col2.checkbox(
        "Moving vehicles only (>5m displacement)", value=False,
        help="Over 90% of test examples are near-stationary; the fine-tuned model actually "
        "beats constant velocity on this subset even though it loses in aggregate -- see README.",
    )

    mask = np.ones(len(test_df), dtype=bool)
    if difficulty_filter != "all":
        mask &= (test_df["difficulty"] == difficulty_filter).to_numpy()
    if moving_only:
        mask &= (test_df["displacement"] > 5.0).to_numpy()
    filtered_df = test_df[mask].reset_index(drop=False).rename(columns={"index": "orig_idx"})

    if len(filtered_df) == 0:
        st.warning("No examples match this filter.")
        return

    if "sb_idx" not in st.session_state:
        st.session_state.sb_idx = 0
    st.session_state.sb_idx = min(st.session_state.sb_idx, len(filtered_df) - 1)

    col_prev, col_next, col_jump = st.columns([1, 1, 4])
    if col_prev.button("Previous", key="sb_prev") and st.session_state.sb_idx > 0:
        st.session_state.sb_idx -= 1
    if col_next.button("Next", key="sb_next") and st.session_state.sb_idx < len(filtered_df) - 1:
        st.session_state.sb_idx += 1
    st.session_state.sb_idx = col_jump.slider("Jump to example", 0, len(filtered_df) - 1, st.session_state.sb_idx, key="sb_slider")

    row = filtered_df.iloc[st.session_state.sb_idx]
    orig_idx = row["orig_idx"]
    gt = gts[orig_idx]
    st.write(f"### Example {st.session_state.sb_idx + 1} / {len(filtered_df)} — {row['scene_name']} (difficulty={row['difficulty']}, displacement={row['displacement']:.2f}m)")

    row_df = pd.DataFrame([row])
    per_model_lines = {}
    per_model_metrics = {}
    for spec in MODEL_SPECS:
        predict_fn = get_predict_fn(spec.key)
        traj, _ = predict_fn(row_df)  # [1, K, T, 2]
        line = best_of_k(traj[0], gt)
        per_model_lines[spec.key] = line
        dist = np.linalg.norm(line - gt, axis=-1)
        per_model_metrics[spec.label] = {"ADE (m)": dist.mean(), "FDE (m)": dist[-1]}

    nusc, helper = load_nusc()
    ann = helper.get_sample_annotation(row["instance_token"], row["sample_token"])
    x, y = ann["translation"][0], ann["translation"][1]
    polylines = nearby_lane_polylines(row["map_name"], x, y, tuple(ann["translation"]), tuple(ann["rotation"]), MAP_RADIUS)

    fig = go.Figure()
    for poly in polylines:
        fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], mode="lines", line=dict(color="lightgray", width=1), showlegend=False, hoverinfo="skip"))

    past_x = np.array([row[f"past_x_{i}"] for i in range(PAST_STEPS)])
    past_y = np.array([row[f"past_y_{i}"] for i in range(PAST_STEPS)])
    valid = ~np.isnan(past_x)
    fig.add_trace(go.Scatter(x=np.r_[past_x[valid], 0], y=np.r_[past_y[valid], 0], mode="lines+markers", line=dict(color="royalblue"), name="past"))

    fig.add_trace(go.Scatter(x=np.r_[0, gt[:, 0]], y=np.r_[0, gt[:, 1]], mode="lines", line=dict(color="magenta", width=3), name="ground truth"))

    for spec in MODEL_SPECS:
        line = per_model_lines[spec.key]
        fig.add_trace(go.Scatter(
            x=np.r_[0, line[:, 0]], y=np.r_[0, line[:, 1]], mode="lines",
            line=dict(color=spec.color, width=2, dash=spec.dash), name=spec.label,
        ))

    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color="black", size=14, symbol="x"), name="current position"))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="x (m, agent frame)", yaxis_title="y (m, agent frame, heading = +y)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=600, margin=dict(l=10, r=10, t=30, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )

    col_plot, col_table = st.columns([2, 1])
    with col_plot:
        st.plotly_chart(fig, theme=None, key=f"sb_plot_{st.session_state.sb_idx}")
    with col_table:
        st.write("**Per-example error** (best-of-K for multimodal models, matching minADE)")
        st.dataframe(pd.DataFrame(per_model_metrics).T.round(3), width="stretch")


def main() -> None:
    st.set_page_config(page_title="TrajFlow Dashboard", layout="wide")
    st.title("TrajFlow — Results Dashboard")

    tab_metrics, tab_scenes = st.tabs(["Metrics", "Scene Browser"])
    with tab_metrics:
        metrics_tab()
    with tab_scenes:
        scene_browser_tab()


if __name__ == "__main__":
    main()
