"""Compact trajectory transformer.

Encodes 2s of past agent-frame motion with a small self-attention encoder,
fuses it with engineered scalar/neighbor context features, and decodes
K=6 candidate futures (+ mode probabilities) via a second small
self-attention block over learned mode tokens.

Missing values (short history near scene starts, fewer than 3 neighbors)
are zero-filled with an explicit validity flag alongside each such
feature, rather than silently imputed — the model can learn to
distinguish "value is 0" from "value is unknown".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from data.preprocess import FUTURE_STEPS, PAST_STEPS

N_SCALAR_FEATS = 3  # velocity, acceleration, heading_change_rate
# NOTE: raw `heading` (absolute global-frame yaw) is deliberately excluded.
# Every other feature here is agent-frame/rotation-invariant; absolute
# heading is tied to each scene's specific road orientation and its mean
# differs meaningfully across train/val/test (disjoint scenes), so the model
# could pick up spurious train-scene-orientation associations that don't
# transfer. `heading_change_rate` (a rate, not an absolute angle) is
# rotation-invariant and stays.
N_NEIGHBORS = 3
CONTEXT_DIM = (N_SCALAR_FEATS * 2) + (N_NEIGHBORS * 2) + N_NEIGHBORS + 2  # scalars+valid, neighbor feats, neighbor-valid, density, intersection


class TrajectoryDataset(Dataset):
    def __init__(self, df):
        past_x = df[[f"past_x_{i}" for i in range(PAST_STEPS)]].to_numpy(dtype=np.float32)
        past_y = df[[f"past_y_{i}" for i in range(PAST_STEPS)]].to_numpy(dtype=np.float32)
        past_valid = (~np.isnan(past_x)).astype(np.float32)
        past_x = np.nan_to_num(past_x, nan=0.0)
        past_y = np.nan_to_num(past_y, nan=0.0)
        self.past_seq = np.stack([past_x, past_y, past_valid], axis=-1).astype(np.float32)  # [N, P, 3]

        scalars = df[["velocity", "acceleration", "heading_change_rate"]].to_numpy(dtype=np.float32)
        scalar_valid = (~np.isnan(scalars)).astype(np.float32)
        scalars = np.nan_to_num(scalars, nan=0.0)

        neighbor_dist_cols = [f"neighbor_dist_{i}" for i in range(N_NEIGHBORS)]
        neighbor_heading_cols = [f"neighbor_rel_heading_{i}" for i in range(N_NEIGHBORS)]
        neighbor_dist = df[neighbor_dist_cols].to_numpy(dtype=np.float32)
        neighbor_heading = df[neighbor_heading_cols].to_numpy(dtype=np.float32)
        neighbor_valid = (~np.isnan(neighbor_dist)).astype(np.float32)
        neighbors = np.nan_to_num(
            np.stack([neighbor_dist, neighbor_heading], axis=-1), nan=0.0
        ).reshape(len(df), -1)  # [N, 6] interleaved dist_0, heading_0, dist_1, heading_1, ...

        density = df["neighbor_density_count"].to_numpy(dtype=np.float32).reshape(-1, 1)
        intersection = df["near_intersection"].to_numpy(dtype=np.float32).reshape(-1, 1)

        self.context = np.concatenate(
            [scalars, scalar_valid, neighbors, neighbor_valid, density, intersection], axis=1
        ).astype(np.float32)  # [N, CONTEXT_DIM]
        assert self.context.shape[1] == CONTEXT_DIM

        future_x = df[[f"future_x_{i}" for i in range(FUTURE_STEPS)]].to_numpy(dtype=np.float32)
        future_y = df[[f"future_y_{i}" for i in range(FUTURE_STEPS)]].to_numpy(dtype=np.float32)
        self.future = np.stack([future_x, future_y], axis=-1).astype(np.float32)  # [N, T, 2]

    def __len__(self):
        return len(self.future)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.past_seq[idx]),
            torch.from_numpy(self.context[idx]),
            torch.from_numpy(self.future[idx]),
        )


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        k_modes: int = 6,
        past_steps: int = PAST_STEPS,
        future_steps: int = FUTURE_STEPS,
        context_dim: int = CONTEXT_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k_modes = k_modes
        self.future_steps = future_steps

        self.past_proj = nn.Linear(3, d_model)
        self.pos_emb = nn.Parameter(torch.randn(past_steps, d_model) * 0.02)
        past_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.past_encoder = nn.TransformerEncoder(past_encoder_layer, num_layers=num_encoder_layers)

        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.mode_emb = nn.Parameter(torch.randn(k_modes, d_model) * 0.02)
        mode_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.mode_decoder = nn.TransformerEncoder(mode_layer, num_layers=1)

        self.traj_head = nn.Linear(d_model, future_steps * 2)
        self.prob_head = nn.Linear(d_model, 1)

    def forward(self, past_seq: torch.Tensor, context: torch.Tensor):
        """
        :param past_seq: [B, P, 3] (x, y, valid)
        :param context: [B, CONTEXT_DIM]
        :return: traj [B, K, T, 2], mode_logits [B, K]
        """
        B = past_seq.shape[0]
        h = self.past_proj(past_seq) + self.pos_emb.unsqueeze(0)
        h = self.past_encoder(h)
        history_emb = h.mean(dim=1)  # [B, d_model]

        context_emb = self.context_mlp(context)  # [B, d_model]
        fused = history_emb + context_emb  # [B, d_model]

        mode_tokens = fused.unsqueeze(1) + self.mode_emb.unsqueeze(0)  # [B, K, d_model]
        mode_tokens = self.mode_decoder(mode_tokens)

        traj = self.traj_head(mode_tokens).view(B, self.k_modes, self.future_steps, 2)
        logits = self.prob_head(mode_tokens).squeeze(-1)  # [B, K]
        return traj, logits


def min_of_k_loss(pred_traj: torch.Tensor, logits: torch.Tensor, gt: torch.Tensor):
    """Winner-take-all min-of-K displacement loss + cross-entropy on the winning mode."""
    diffs = pred_traj - gt.unsqueeze(1)  # [B, K, T, 2]
    dists = diffs.pow(2).sum(-1).clamp(min=1e-9).sqrt()  # [B, K, T]
    ade_per_mode = dists.mean(-1)  # [B, K]
    best_mode = ade_per_mode.argmin(dim=1)  # [B]
    reg_loss = ade_per_mode.gather(1, best_mode.unsqueeze(1)).squeeze(1).mean()
    cls_loss = F.cross_entropy(logits, best_mode)
    return reg_loss + cls_loss, reg_loss.detach(), cls_loss.detach()
