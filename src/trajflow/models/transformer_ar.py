"""Autoregressive-decoder transformer: isolates decoding style (parallel
vs. autoregressive) as a candidate explanation for the LSTM's edge over
TrajectoryTransformer, independent of encoder type.

The existing controlled comparison (train_transformer_full.py vs.
train_lstm.py) showed architecture, not training regime, explains most of
the LSTM's advantage -- but "architecture" bundles two differences at
once: the LSTM has both a different *encoder* (recurrent vs. attention)
and a different *decoder* (autoregressive, one step conditioning on the
last, vs. parallel -- all T future steps produced in a single forward
pass). This model holds the encoder fixed at TrajectoryTransformer's exact
self-attention encoder (same `past_proj`/`pos_emb`/`past_encoder`/
`context_mlp`, fused identically) and swaps in an autoregressive decoder
structurally identical to models/lstm.py's (one `LSTMCell` step per future
timestep, predicting a position delta and feeding it back in):

  TrajectoryTransformer:  attention encoder + parallel decoder
  LSTMTrajectoryModel:    LSTM encoder      + autoregressive decoder
  TransformerARModel:     attention encoder + autoregressive decoder  <- this file

If this model performs close to the LSTM, decoding style is the dominant
factor. If it instead performs close to TrajectoryTransformer, the LSTM's
own (recurrent) encoder matters too, not just its decoder -- see
train_transformer_ar_full.py and README Results for what was actually
found.
"""

import torch
from torch import nn

from trajflow.data.preprocess import FUTURE_STEPS, PAST_STEPS
from trajflow.models.transformer import CONTEXT_DIM


class TransformerARModel(nn.Module):
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

        # Encoder: identical to TrajectoryTransformer's.
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

        # Decoder: identical in structure to LSTMTrajectoryModel's.
        self.mode_emb = nn.Parameter(torch.randn(k_modes, d_model) * 0.02)
        self.decoder_cell = nn.LSTMCell(input_size=2, hidden_size=d_model)
        self.output_head = nn.Linear(d_model, 2)  # predicts a position DELTA per step
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

        all_traj = []
        all_logits = []
        for k in range(self.k_modes):
            h_k = fused + self.mode_emb[k].unsqueeze(0)
            c_k = torch.zeros_like(h_k)
            pos = torch.zeros(B, 2, device=past_seq.device, dtype=past_seq.dtype)
            step_input = torch.zeros(B, 2, device=past_seq.device, dtype=past_seq.dtype)
            positions = []
            for _ in range(self.future_steps):
                h_k, c_k = self.decoder_cell(step_input, (h_k, c_k))
                delta = self.output_head(h_k)
                pos = pos + delta
                positions.append(pos)
                step_input = delta
            all_traj.append(torch.stack(positions, dim=1))  # [B, T, 2]
            all_logits.append(self.prob_head(h_k).squeeze(-1))  # [B]

        traj = torch.stack(all_traj, dim=1)  # [B, K, T, 2]
        logits = torch.stack(all_logits, dim=1)  # [B, K]
        return traj, logits
