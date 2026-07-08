"""Trains TransformerARModel (see models/transformer_ar.py) on the FULL
train split in one pass -- exactly matching how train_lstm.py trains the
LSTM and train_transformer_full.py trains the parallel-decoder
transformer. Same data, same loss, same training budget as both; only the
decoder differs from train_transformer_full.py (autoregressive vs.
parallel) while the encoder differs from train_lstm.py (attention vs.
LSTM). Together, all three full-split models isolate encoder type from
decoder style:

  Transformer (full-split):     attention encoder + parallel decoder
  Transformer-AR (full-split):  attention encoder + autoregressive decoder   <- this script
  LSTM (baseline):               LSTM encoder      + autoregressive decoder
"""

import torch
from torch.utils.data import DataLoader

from trajflow.evaluation.evaluate import filter_difficulty, load_split, log_metrics
from trajflow.models.train_pretrain import evaluate_on_df, set_seed
from trajflow.models.transformer import TrajectoryDataset, min_of_k_loss
from trajflow.models.transformer_ar import TransformerARModel
from trajflow.paths import CHECKPOINTS_DIR

CHECKPOINT_PATH = CHECKPOINTS_DIR / "transformer_ar_full.pt"
EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 64
SEED = 0


def main() -> None:
    set_seed(SEED)

    train_df = load_split("train")
    val_df = load_split("val")

    train_dataset = TrajectoryDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerARModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_minade = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for past_seq, context, gt in train_loader:
            optimizer.zero_grad()
            traj, logits = model(past_seq, context)
            loss, reg, cls = min_of_k_loss(traj, logits, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(gt)
        epoch_loss /= len(train_dataset)

        val_metrics = evaluate_on_df(model, val_df)
        if val_metrics["minADE"] < best_val_minade:
            best_val_minade = val_metrics["minADE"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d} | train_loss={epoch_loss:.4f} | "
                f"val_minADE={val_metrics['minADE']:.4f} | val_minFDE={val_metrics['minFDE']:.4f}"
            )

    model.load_state_dict(best_state)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved best checkpoint (val minADE={best_val_minade:.4f}) to {CHECKPOINT_PATH}")

    val_metrics = evaluate_on_df(model, val_df)
    log_metrics(
        phase=2,
        model="Transformer-AR (full-split, autoregressive decoder)",
        eval_split="val",
        difficulty="all",
        metrics=val_metrics,
        notes="model selection metric (best checkpoint by val minADE); ablation isolating decoder style -- "
        "same self-attention encoder as 'Transformer (full-split)', but an autoregressive LSTMCell decoder "
        "structurally identical to LSTM (baseline)'s, same full-split/one-pass training regime as both; "
        "see README Results",
    )
    print(f"[Transformer-AR-full] val/all: {val_metrics}")

    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        metrics = evaluate_on_df(model, df)
        notes = (
            "3-way isolation: 'Transformer (full-split)' (attention encoder + parallel decoder) vs. this "
            "model (attention encoder + autoregressive decoder) vs. 'LSTM (baseline)' (LSTM encoder + "
            "autoregressive decoder) -- same data/loss/epochs/batch size for all three, only encoder/decoder "
            "type differs; see README Results for which factor (encoder or decoder) actually explains the gap"
            if difficulty == "all"
            else ""
        )
        log_metrics(
            phase=2, model="Transformer-AR (full-split, autoregressive decoder)", eval_split="test",
            difficulty=difficulty, metrics=metrics, notes=notes,
        )
        print(f"[Transformer-AR-full] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
