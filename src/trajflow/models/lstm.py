"""Compact LSTM encoder-decoder trajectory model.

A second deep-learning architecture alongside models/transformer.py, added
in a later pass to broaden the comparison beyond a single attention-based
model. Same inputs/outputs and training objective as the transformer
(reuses TrajectoryDataset and min_of_k_loss directly), but a genuinely
different architecture family: an LSTM encodes 2s of past motion, fused
with the same scalar/neighbor context vector, and K=6 candidate futures
are decoded autoregressively (one LSTMCell step per future timestep,
predicting a position delta and feeding it back as the next input) rather
than in parallel via attention.
"""

import torch
from torch import nn

from trajflow.data.preprocess import FUTURE_STEPS, PAST_STEPS
from trajflow.models.transformer import CONTEXT_DIM


class LSTMTrajectoryModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        k_modes: int = 6,
        past_steps: int = PAST_STEPS,
        future_steps: int = FUTURE_STEPS,
        context_dim: int = CONTEXT_DIM,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_modes = k_modes
        self.future_steps = future_steps

        self.encoder = nn.LSTM(input_size=3, hidden_size=hidden_dim, batch_first=True)

        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mode_emb = nn.Parameter(torch.randn(k_modes, hidden_dim) * 0.02)
        self.decoder_cell = nn.LSTMCell(input_size=2, hidden_size=hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 2)  # predicts a position DELTA per step
        self.prob_head = nn.Linear(hidden_dim, 1)

    def forward(self, past_seq: torch.Tensor, context: torch.Tensor):
        """
        :param past_seq: [B, P, 3] (x, y, valid)
        :param context: [B, CONTEXT_DIM]
        :return: traj [B, K, T, 2], mode_logits [B, K]
        """
        B = past_seq.shape[0]
        _, (h_enc, _) = self.encoder(past_seq)
        h_enc = h_enc.squeeze(0)  # [B, hidden_dim]

        context_emb = self.context_mlp(context)  # [B, hidden_dim]
        fused = h_enc + context_emb  # [B, hidden_dim]

        all_traj = []
        all_logits = []
        for k in range(self.k_modes):
            h = fused + self.mode_emb[k].unsqueeze(0)
            c = torch.zeros_like(h)
            pos = torch.zeros(B, 2, device=past_seq.device, dtype=past_seq.dtype)
            step_input = torch.zeros(B, 2, device=past_seq.device, dtype=past_seq.dtype)
            positions = []
            for _ in range(self.future_steps):
                h, c = self.decoder_cell(step_input, (h, c))
                delta = self.output_head(h)
                pos = pos + delta
                positions.append(pos)
                step_input = delta
            all_traj.append(torch.stack(positions, dim=1))  # [B, T, 2]
            all_logits.append(self.prob_head(h).squeeze(-1))  # [B]

        traj = torch.stack(all_traj, dim=1)  # [B, K, T, 2]
        logits = torch.stack(all_logits, dim=1)  # [B, K]
        return traj, logits
