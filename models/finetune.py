"""Fine-tune the pretrained trajectory transformer on the 'hard' split.

Loads the Phase 3 checkpoint (pretrained on 'easy' scenes only) and
continues training on 'hard' scenes with a lower LR and fewer epochs.
Model selection uses the same val-set minADE criterion as pretraining,
for methodological consistency across phases. Primary evaluation is the
hard-scene test set (per the phase spec), with test/all logged alongside
for comparability against the other rows in the table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.evaluate import filter_difficulty, load_split, log_metrics
from models.train_pretrain import evaluate_on_df, set_seed
from models.transformer import TrajectoryDataset, TrajectoryTransformer, min_of_k_loss

PRETRAINED_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "pretrained.pt"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "finetuned_v1.pt"
EPOCHS = 60
LR = 2e-4
BATCH_SIZE = 64
SEED = 0


def main() -> None:
    set_seed(SEED)

    if not PRETRAINED_CHECKPOINT.exists():
        print(f"ERROR: no pretrained checkpoint at {PRETRAINED_CHECKPOINT}. Run models/train_pretrain.py first.", file=sys.stderr)
        raise SystemExit(1)

    train_df = filter_difficulty(load_split("train"), "hard")
    val_df = load_split("val")

    train_dataset = TrajectoryDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TrajectoryTransformer()
    model.load_state_dict(torch.load(PRETRAINED_CHECKPOINT))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    pretrained_val_metrics = evaluate_on_df(model, val_df)
    print(f"[Pretrained, before fine-tuning] val/all: {pretrained_val_metrics}")

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
        phase=4,
        model="Transformer (fine-tuned-v1, hard)",
        eval_split="val",
        difficulty="all",
        metrics=val_metrics,
        notes="model selection metric (best checkpoint by val minADE, starting from Phase 3 pretrained weights)",
    )
    print(f"[Fine-tuned-v1] val/all: {val_metrics}")

    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        metrics = evaluate_on_df(model, df)
        notes = (
            "primary Phase 4 comparison metric (pretrained vs fine-tuned) -- fine-tuning improved "
            "val minADE but regressed test/hard minADE (0.798 -> 0.896) while improving Miss Rate@2m "
            "(0.112 -> 0.086, i.e. fewer complete misses but higher average error); with only 6 train "
            "scenes total this reads as scene-specific overfitting rather than a clean win; see README limitations"
            if difficulty == "hard"
            else ""
        )
        log_metrics(
            phase=4,
            model="Transformer (fine-tuned-v1, hard)",
            eval_split="test",
            difficulty=difficulty,
            metrics=metrics,
            notes=notes,
        )
        print(f"[Fine-tuned-v1] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
