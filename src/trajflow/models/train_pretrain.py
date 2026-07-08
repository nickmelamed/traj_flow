"""Pretrain the compact trajectory transformer on the 'easy' split only.

Model selection: track val-set (all difficulties, per the phase spec's
"held-out val set") minADE each epoch and keep the best checkpoint. Final
metrics are logged both on val (canonical) and on the test splits (for
direct comparability with the baselines and the Phase 4 fine-tuned model).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from trajflow.evaluation.evaluate import filter_difficulty, future_xy, load_split, log_metrics
from trajflow.evaluation.metrics import batch_metrics
from trajflow.models.transformer import TrajectoryDataset, TrajectoryTransformer, min_of_k_loss
from trajflow.paths import CHECKPOINTS_DIR

CHECKPOINT_PATH = CHECKPOINTS_DIR / "pretrained.pt"
EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 64
SEED = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def predict_all(model: TrajectoryTransformer, dataset: TrajectoryDataset) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_traj = []
    for past_seq, context, _ in loader:
        traj, _ = model(past_seq, context)
        all_traj.append(traj.numpy())
    return np.concatenate(all_traj, axis=0)


def evaluate_on_df(model: TrajectoryTransformer, df) -> dict:
    dataset = TrajectoryDataset(df)
    preds = predict_all(model, dataset)
    gts = future_xy(df)
    return batch_metrics(preds, gts)


def main() -> None:
    set_seed(SEED)

    train_df = filter_difficulty(load_split("train"), "easy")
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
        phase=3,
        model="Transformer (pretrained, easy-only)",
        eval_split="val",
        difficulty="all",
        metrics=val_metrics,
        notes="model selection metric (best checkpoint by val minADE)",
    )
    print(f"[Pretrained] val/all: {val_metrics}")

    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        metrics = evaluate_on_df(model, df)
        log_metrics(
            phase=3,
            model="Transformer (pretrained, easy-only)",
            eval_split="test",
            difficulty=difficulty,
            metrics=metrics,
            notes="trained on easy scenes only; test/hard measures out-of-distribution generalization pre-fine-tune. "
            "Note: removing the leaked absolute-heading feature (see XGBoost row) made this model's test metrics "
            "slightly worse, not better (test/all minADE was 0.602 with heading included) -- unlike XGBoost, the "
            "transformer apparently extracted some real (if likely non-generalizable) signal from it; kept removed "
            "for methodological consistency/frame-invariance, see README limitations",
        )
        print(f"[Pretrained] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
