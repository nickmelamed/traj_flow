import torch

from trajflow.data.preprocess import FUTURE_STEPS, PAST_STEPS
from trajflow.models.transformer import CONTEXT_DIM
from trajflow.models.transformer_ar import TransformerARModel


def test_forward_shapes():
    model = TransformerARModel()
    B, K = 4, 6
    past_seq = torch.randn(B, PAST_STEPS, 3)
    context = torch.randn(B, CONTEXT_DIM)

    traj, logits = model(past_seq, context)

    assert traj.shape == (B, K, FUTURE_STEPS, 2)
    assert logits.shape == (B, K)


def test_decoder_is_autoregressive_not_parallel():
    """Sanity check that this really is a step-by-step decoder: perturbing
    only the encoder's inputs (past_seq/context) changes every future
    step's prediction (since they all derive from the same fused
    embedding), which is expected for both decoder styles -- but the
    defining autoregressive property is that decoding involves a
    per-timestep LSTMCell loop, verified structurally here by confirming
    the model exposes a single shared `decoder_cell` (unlike
    TrajectoryTransformer's parallel one-shot `mode_decoder`).
    """
    model = TransformerARModel()
    assert isinstance(model.decoder_cell, torch.nn.LSTMCell)
    assert not hasattr(model, "mode_decoder")  # no parallel one-shot decoder block


def test_zero_context_and_history_still_produces_finite_output():
    model = TransformerARModel()
    model.eval()
    past_seq = torch.zeros(2, PAST_STEPS, 3)
    context = torch.zeros(2, CONTEXT_DIM)
    with torch.no_grad():
        traj, logits = model(past_seq, context)
    assert torch.isfinite(traj).all()
    assert torch.isfinite(logits).all()
