"""Pretrain the LSTM on the 'easy' split only -- mirrors
models/train_pretrain.py exactly, but for LSTMTrajectoryModel instead of
TrajectoryTransformer.

Why this exists: models/train_lstm.py trains the LSTM on the FULL split in
one pass, as a standalone architecture-comparison point outside the
transformer's pretrain/fine-tune/HITL lineage. README's own Limitations
section flagged the natural follow-up: does the LSTM show the same
scene-specific-overfitting pattern the transformer showed when put through
that same fragmented (pretrain-on-easy -> fine-tune-on-hard -> fine-tune-
again-on-HITL-corrections) pipeline, or is that a transformer-specific
weakness? This script (+ finetune_lstm.py + finetune_lstm_round2.py) is
that experiment. Model selection, loss, and epoch/LR budgets are identical
to the transformer's own lineage for a fair comparison.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from trajflow.evaluation.evaluate import filter_difficulty, load_split, log_metrics
from trajflow.models.lstm import LSTMTrajectoryModel
from trajflow.models.train_pretrain import evaluate_on_df, set_seed
from trajflow.models.transformer import TrajectoryDataset, min_of_k_loss
from trajflow.paths import CHECKPOINTS_DIR

CHECKPOINT_PATH = CHECKPOINTS_DIR / "lstm_pretrained.pt"
EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 64
SEED = 0


def main() -> None:
    set_seed(SEED)

    train_df = filter_difficulty(load_split("train"), "easy")
    val_df = load_split("val")

    train_dataset = TrajectoryDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMTrajectoryModel()
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
        phase=9,
        model="LSTM (pretrained, easy-only)",
        eval_split="val",
        difficulty="all",
        metrics=val_metrics,
        notes="model selection metric (best checkpoint by val minADE); first stage of putting the LSTM "
        "through the same pretrain/fine-tune/HITL lineage as the transformer (mirrors train_pretrain.py "
        "exactly) -- to see if it overfits the same way on 6 scenes, or if that's transformer-specific; "
        "see README Results",
    )
    print(f"[LSTM pretrained] val/all: {val_metrics}")

    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        metrics = evaluate_on_df(model, df)
        log_metrics(
            phase=9,
            model="LSTM (pretrained, easy-only)",
            eval_split="test",
            difficulty=difficulty,
            metrics=metrics,
            notes="trained on easy scenes only, same as Transformer (pretrained, easy-only); compare "
            "against that row directly -- see README Results" if difficulty == "all" else "",
        )
        print(f"[LSTM pretrained] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
