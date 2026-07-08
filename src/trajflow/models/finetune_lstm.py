"""Fine-tune the pretrained LSTM on the 'hard' split -- mirrors
models/finetune.py exactly (same EPOCHS/LR), but continues from
lstm_pretrained.pt instead of pretrained.pt. See train_lstm_pretrain.py
for why this lineage exists.
"""

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from trajflow.evaluation.evaluate import filter_difficulty, load_split, log_metrics
from trajflow.models.lstm import LSTMTrajectoryModel
from trajflow.models.train_pretrain import evaluate_on_df, set_seed
from trajflow.models.transformer import TrajectoryDataset, min_of_k_loss
from trajflow.paths import CHECKPOINTS_DIR

PRETRAINED_CHECKPOINT = CHECKPOINTS_DIR / "lstm_pretrained.pt"
CHECKPOINT_PATH = CHECKPOINTS_DIR / "lstm_finetuned_v1.pt"
EPOCHS = 60
LR = 2e-4
BATCH_SIZE = 64
SEED = 0


def main() -> None:
    set_seed(SEED)

    if not PRETRAINED_CHECKPOINT.exists():
        print(f"ERROR: no pretrained checkpoint at {PRETRAINED_CHECKPOINT}. Run trajflow-lstm-pretrain first.", file=sys.stderr)
        raise SystemExit(1)

    train_df = filter_difficulty(load_split("train"), "hard")
    val_df = load_split("val")

    train_dataset = TrajectoryDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMTrajectoryModel()
    model.load_state_dict(torch.load(PRETRAINED_CHECKPOINT))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    pretrained_val_metrics = evaluate_on_df(model, val_df)
    print(f"[LSTM pretrained, before fine-tuning] val/all: {pretrained_val_metrics}")

    best_val_minade = pretrained_val_metrics["minADE"]
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

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
        model="LSTM (fine-tuned-v1, hard)",
        eval_split="val",
        difficulty="all",
        metrics=val_metrics,
        notes="model selection metric (best checkpoint by val minADE, starting from LSTM pretrained weights)",
    )
    print(f"[LSTM fine-tuned-v1] val/all: {val_metrics}")

    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        metrics = evaluate_on_df(model, df)
        notes = (
            "compare against Transformer (fine-tuned-v1, hard)'s regression (0.758 -> 0.925 test/all) to see "
            "if the LSTM shows the same scene-specific-overfitting pattern on only 6 training scenes, or is "
            "more robust to it -- see README Results"
            if difficulty == "all"
            else ""
        )
        log_metrics(
            phase=9,
            model="LSTM (fine-tuned-v1, hard)",
            eval_split="test",
            difficulty=difficulty,
            metrics=metrics,
            notes=notes,
        )
        print(f"[LSTM fine-tuned-v1] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
