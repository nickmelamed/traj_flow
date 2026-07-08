"""Shared model-loading + prediction utilities for every model in the
TrajFlow pipeline, used by viz/dashboard.py (and reusable elsewhere, e.g.
viz/scene_overlay.py). Add a new model to the comparison by adding one
ModelSpec entry to MODEL_SPECS -- everything that imports this list
(dashboard tables, charts, scene browser) picks it up automatically.

Every predict_fn has the SAME signature regardless of whether the
underlying model is single-hypothesis (CV, XGBoost -- K=1) or multimodal
(the transformer variants -- K=6):

    predict_fn(df) -> (traj: [N, K, T, 2], probs: [N, K])

K=1 models return probs of all 1.0 (their only mode is certain). This
lets evaluation.metrics.batch_metrics(traj, gts) compute minADE/minFDE
identically for every model (it already handles K=1 or K>1), so numbers
computed here always match the official logged metrics in
results/metrics_comparison.md -- and it lets the scene browser show
per-mode probability (opacity) for multimodal models without special-
casing single-hypothesis ones.
"""

import os
import sys
from pathlib import Path
from typing import Callable, NamedTuple

# Must be set before torch/xgboost are imported: loading both libraries in
# the same process on this macOS setup deadlocks otherwise. See the same
# note in hitl/flag_uncertain.py and hitl/review_app.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from models.baseline_xgb import MODEL_PATH as XGB_MODEL_PATH
from models.baseline_xgb import make_features as xgb_make_features

import torch
from torch.utils.data import DataLoader

from data.preprocess import FUTURE_STEPS
from models.baseline_ca import predict_ca
from models.baseline_cv import predict_cv
from models.lstm import LSTMTrajectoryModel
from models.transformer import TrajectoryDataset, TrajectoryTransformer

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints"

# df -> (traj [N, K, T, 2], probs [N, K])
PredictFn = Callable[[pd.DataFrame], tuple]


class ModelSpec(NamedTuple):
    key: str
    label: str
    color: str
    dash: str  # plotly line dash style: "solid" | "dash" | "dot" | "dashdot"
    loader: Callable[[], PredictFn]  # lazy -- only loads weights when called


def load_cv_predict_fn() -> PredictFn:
    def predict(df: pd.DataFrame):
        traj = predict_cv(df)[:, None, :, :]  # [N, 1, T, 2]
        probs = np.ones((len(df), 1))
        return traj, probs

    return predict


def load_ca_predict_fn() -> PredictFn:
    def predict(df: pd.DataFrame):
        traj = predict_ca(df)[:, None, :, :]  # [N, 1, T, 2]
        probs = np.ones((len(df), 1))
        return traj, probs

    return predict


def load_xgb_predict_fn() -> PredictFn:
    model = joblib.load(XGB_MODEL_PATH)

    def predict(df: pd.DataFrame):
        preds = model.predict(xgb_make_features(df)).reshape(len(df), FUTURE_STEPS, 2)
        traj = preds[:, None, :, :]  # [N, 1, T, 2]
        probs = np.ones((len(df), 1))
        return traj, probs

    return predict


def load_multimodal_predict_fn(model_class, checkpoint_name: str) -> PredictFn:
    """Works for any model sharing TrajectoryTransformer's forward signature
    -- (past_seq, context) -> (traj [B,K,T,2], logits [B,K]) -- which
    includes both TrajectoryTransformer and LSTMTrajectoryModel.
    """
    model = model_class()
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / checkpoint_name))
    model.eval()

    @torch.no_grad()
    def predict(df: pd.DataFrame):
        dataset = TrajectoryDataset(df)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        all_traj, all_probs = [], []
        for past_seq, context, _ in loader:
            traj, logits = model(past_seq, context)
            probs = torch.softmax(logits, dim=-1)
            all_traj.append(traj.numpy())
            all_probs.append(probs.numpy())
        return np.concatenate(all_traj, axis=0), np.concatenate(all_probs, axis=0)

    return predict


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("cv", "Constant Velocity", "crimson", "dash", load_cv_predict_fn),
    ModelSpec("ca", "Constant Acceleration", "purple", "dot", load_ca_predict_fn),
    ModelSpec("xgb", "XGBoost", "darkorange", "dot", load_xgb_predict_fn),
    # NOTE: these three labels must exactly match the ones models/train_pretrain.py,
    # models/finetune.py, and models/finetune_round2.py already log to
    # results/metrics_comparison.md (phases 3/4/6) -- a mismatch means the
    # same model shows up as two separate rows/bars/legend entries anywhere
    # that groups by Model (the dashboard's Metrics tab in particular).
    ModelSpec("pretrained", "Transformer (pretrained, easy-only)", "gray", "dashdot", lambda: load_multimodal_predict_fn(TrajectoryTransformer, "pretrained.pt")),
    ModelSpec("finetuned_v1", "Transformer (fine-tuned-v1, hard)", "steelblue", "dash", lambda: load_multimodal_predict_fn(TrajectoryTransformer, "finetuned_v1.pt")),
    ModelSpec("finetuned_v2", "Transformer (fine-tuned-v2, post-HITL)", "seagreen", "solid", lambda: load_multimodal_predict_fn(TrajectoryTransformer, "finetuned_v2.pt")),
    ModelSpec("transformer_full", "Transformer (full-split)", "black", "dashdot", lambda: load_multimodal_predict_fn(TrajectoryTransformer, "transformer_full.pt")),
    ModelSpec("lstm", "LSTM (baseline)", "brown", "solid", lambda: load_multimodal_predict_fn(LSTMTrajectoryModel, "lstm.pt")),
]
