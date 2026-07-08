"""Controlled comparison experiment: trains the SAME TrajectoryTransformer
architecture used by the pretrain/fine-tune/HITL lineage, but on the FULL
train split in one single pass -- exactly matching how models/train_lstm.py
trains the LSTM. Same data, same loss, same training budget as the LSTM;
only the architecture differs.

Why this exists: the LSTM (models/train_lstm.py) substantially
outperforms every transformer checkpoint, but it was trained differently
(one pass over all 2,388 train rows) than the transformer (pretrained on
1,150 easy rows, then fine-tuned on 1,238 hard rows, then fine-tuned again
on HITL-corrected hard rows). That's two confounded variables -- this
script holds training regime constant so the remaining gap (if any)
between this model and the LSTM isolates the effect of architecture.
"""

import torch
from torch.utils.data import DataLoader

from trajflow.evaluation.evaluate import filter_difficulty, load_split, log_metrics
from trajflow.models.train_pretrain import evaluate_on_df, set_seed
from trajflow.models.transformer import TrajectoryDataset, TrajectoryTransformer, min_of_k_loss
from trajflow.paths import CHECKPOINTS_DIR

CHECKPOINT_PATH = CHECKPOINTS_DIR / "transformer_full.pt"
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

    model = TrajectoryTransformer()
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
        model="Transformer (full-split)",
        eval_split="val",
        difficulty="all",
        metrics=val_metrics,
        notes="model selection metric (best checkpoint by val minADE); controlled comparison against "
        "LSTM (baseline) -- same architecture as the pretrained/fine-tuned-v1/v2 lineage, but trained on "
        "the FULL train split in one pass like the LSTM, isolating architecture from training regime",
    )
    print(f"[Transformer-full] val/all: {val_metrics}")

    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        metrics = evaluate_on_df(model, df)
        notes = (
            "controlled comparison: same training regime as LSTM (baseline) -- full train split, one pass, "
            "same loss/epochs/batch size -- so the gap to LSTM's test/all minADE (0.265) isolates the effect "
            "of architecture (attention+parallel decoding vs. recurrent+autoregressive decoding); see README"
            if difficulty == "all"
            else ""
        )
        log_metrics(phase=2, model="Transformer (full-split)", eval_split="test", difficulty=difficulty, metrics=metrics, notes=notes)
        print(f"[Transformer-full] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
