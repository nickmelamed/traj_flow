"""Fine-tune round 2 for the LSTM lineage: merges the SAME HITL corrections
used for the transformer (corrections/corrections.parquet is reviewer
decisions on ground truth, not architecture-specific) into the hard
training set and continues fine-tuning from lstm_finetuned_v1.pt. Mirrors
models/finetune_round2.py exactly (reuses its merge_corrections function
directly rather than re-implementing it), but for the LSTM.
"""

import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from trajflow.evaluation.evaluate import filter_difficulty, load_split, log_metrics
from trajflow.models.finetune_round2 import merge_corrections
from trajflow.models.lstm import LSTMTrajectoryModel
from trajflow.models.train_pretrain import evaluate_on_df, set_seed
from trajflow.models.transformer import TrajectoryDataset, min_of_k_loss
from trajflow.paths import CHECKPOINTS_DIR, CORRECTIONS_PATH

V1_CHECKPOINT = CHECKPOINTS_DIR / "lstm_finetuned_v1.pt"
CHECKPOINT_PATH = CHECKPOINTS_DIR / "lstm_finetuned_v2.pt"
EPOCHS = 60
LR = 2e-4
BATCH_SIZE = 64
SEED = 0


def main() -> None:
    set_seed(SEED)

    if not V1_CHECKPOINT.exists():
        print(f"ERROR: no round-1 checkpoint at {V1_CHECKPOINT}. Run trajflow-lstm-finetune first.", file=sys.stderr)
        raise SystemExit(1)
    if not CORRECTIONS_PATH.exists():
        print(f"ERROR: no corrections found at {CORRECTIONS_PATH}. Complete a review pass via trajflow-review-app first.", file=sys.stderr)
        raise SystemExit(1)

    train_hard_df = filter_difficulty(load_split("train"), "hard").reset_index(drop=True)
    corrections_df = pd.read_parquet(CORRECTIONS_PATH)
    train_df, n_changed = merge_corrections(train_hard_df, corrections_df)
    val_df = load_split("val")

    train_dataset = TrajectoryDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMTrajectoryModel()
    model.load_state_dict(torch.load(V1_CHECKPOINT))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    v1_val_metrics = evaluate_on_df(model, val_df)
    print(f"[LSTM fine-tuned-v1, before round 2] val/all: {v1_val_metrics}")

    best_val_minade = v1_val_metrics["minADE"]
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
        model="LSTM (fine-tuned-v2, post-HITL)",
        eval_split="val",
        difficulty="all",
        metrics=val_metrics,
        notes="model selection metric (best checkpoint by val minADE, starting from LSTM fine-tuned-v1 weights)",
    )
    print(f"[LSTM fine-tuned-v2] val/all: {val_metrics}")

    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        metrics = evaluate_on_df(model, df)
        notes = (
            f"only {n_changed} of {len(train_df)} hard-train labels actually changed (same corrections file "
            f"as the transformer lineage); compare against LSTM (fine-tuned-v1, hard) and Transformer "
            f"(fine-tuned-v2, post-HITL) -- see README Results"
            if difficulty == "hard"
            else ""
        )
        log_metrics(
            phase=9,
            model="LSTM (fine-tuned-v2, post-HITL)",
            eval_split="test",
            difficulty=difficulty,
            metrics=metrics,
            notes=notes,
        )
        print(f"[LSTM fine-tuned-v2] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
