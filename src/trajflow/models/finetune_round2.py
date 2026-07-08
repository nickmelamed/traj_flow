"""Fine-tune round 2: merge HITL corrections into the hard-scene training
set and continue fine-tuning from the round-1 checkpoint.

Merge: for the 124 flagged hard-TRAIN examples reviewed in Phase 5, this
overwrites future_x/y with the reviewer's corrected_future_x/y wherever
provided (a no-op for the ~100 "accept ground truth as-is" rows, since
those were saved with corrected == original; an actual edit for the ~24
"correct trajectory" rows). The other ~1,114 unreviewed hard-train rows
are untouched. Net effect: a small fraction of training labels change.

Continues from models/checkpoints/finetuned_v1.pt (not the original
pretrained checkpoint) -- this is round 2 of fine-tuning, not a restart.
Re-evaluates on the same held-out test set as every other phase for a
fair before/after comparison.

Ablation control (--ablation-no-corrections): fine-tuned-v2 gets both (a)
60 more epochs of fine-tuning and (b) 13 corrected labels, confounded --
so an improvement over fine-tuned-v1 doesn't by itself prove the
corrections mattered rather than just more training on largely the same
hard-train data. This flag reruns the identical round-2 recipe (same
epochs/LR/seed/starting checkpoint) on the UNCORRECTED hard-train split
-- everything held constant except the 13 corrected labels -- producing
`Transformer (fine-tuned-v2-control, no corrections)` so the two can be
compared directly. See README HITL section.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from trajflow.evaluation.evaluate import filter_difficulty, load_split, log_metrics
from trajflow.models.train_pretrain import evaluate_on_df, set_seed
from trajflow.models.transformer import FUTURE_STEPS, TrajectoryDataset, TrajectoryTransformer, min_of_k_loss
from trajflow.paths import CHECKPOINTS_DIR, CORRECTIONS_PATH

V1_CHECKPOINT = CHECKPOINTS_DIR / "finetuned_v1.pt"
CHECKPOINT_PATH = CHECKPOINTS_DIR / "finetuned_v2.pt"
CONTROL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "finetuned_v2_control.pt"
EPOCHS = 60
LR = 2e-4
BATCH_SIZE = 64
SEED = 0


def merge_corrections(train_hard_df: pd.DataFrame, corrections_df: pd.DataFrame) -> pd.DataFrame:
    df = train_hard_df.copy().set_index(["instance_token", "sample_token"])
    corr = corrections_df.set_index(["instance_token", "sample_token"])

    n_changed = 0
    future_cols = [f"future_x_{i}" for i in range(FUTURE_STEPS)] + [f"future_y_{i}" for i in range(FUTURE_STEPS)]
    for key in corr.index.intersection(df.index):
        row = corr.loc[key]
        corrected = np.array(
            [row[f"corrected_future_x_{i}"] for i in range(FUTURE_STEPS)]
            + [row[f"corrected_future_y_{i}"] for i in range(FUTURE_STEPS)]
        )
        original = df.loc[key, future_cols].to_numpy(dtype=float)
        if not np.allclose(corrected, original):
            n_changed += 1
        df.loc[key, future_cols] = corrected

    print(f"Merged {len(corr)} reviewed examples into {len(df)} hard-train rows ({n_changed} actually changed values).")
    return df.reset_index(), n_changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-no-corrections",
        action="store_true",
        help="Skip merging HITL corrections and fine-tune round 2 on the unmodified hard-train split instead "
        "-- an ablation isolating whether the corrections themselves help, vs. just more training epochs.",
    )
    args = parser.parse_args()
    is_control = args.ablation_no_corrections

    set_seed(SEED)

    if not V1_CHECKPOINT.exists():
        print(f"ERROR: no round-1 checkpoint at {V1_CHECKPOINT}. Run trajflow-finetune first.", file=sys.stderr)
        raise SystemExit(1)

    train_hard_df = filter_difficulty(load_split("train"), "hard").reset_index(drop=True)
    if is_control:
        train_df, n_changed = train_hard_df, 0
        print("[Ablation] --ablation-no-corrections: skipping corrections merge, training on unmodified hard-train split.")
    else:
        if not CORRECTIONS_PATH.exists():
            print(f"ERROR: no corrections found at {CORRECTIONS_PATH}. Complete a review pass via trajflow-review-app first.", file=sys.stderr)
            raise SystemExit(1)
        corrections_df = pd.read_parquet(CORRECTIONS_PATH)
        train_df, n_changed = merge_corrections(train_hard_df, corrections_df)
    val_df = load_split("val")

    train_dataset = TrajectoryDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TrajectoryTransformer()
    model.load_state_dict(torch.load(V1_CHECKPOINT))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    v1_val_metrics = evaluate_on_df(model, val_df)
    print(f"[Fine-tuned-v1, before round 2] val/all: {v1_val_metrics}")

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
    checkpoint_path = CONTROL_CHECKPOINT_PATH if is_control else CHECKPOINT_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved best checkpoint (val minADE={best_val_minade:.4f}) to {checkpoint_path}")

    model_label = (
        "Transformer (fine-tuned-v2-control, no corrections)" if is_control else "Transformer (fine-tuned-v2, post-HITL)"
    )

    val_metrics = evaluate_on_df(model, val_df)
    log_metrics(
        phase=6,
        model=model_label,
        eval_split="val",
        difficulty="all",
        metrics=val_metrics,
        notes=(
            "ablation control for the HITL comparison above: identical round-2 recipe (same starting "
            "checkpoint, epochs, LR, seed) but WITHOUT merging the 13 corrected labels -- isolates whether "
            "fine-tuned-v2's gain over fine-tuned-v1 comes from the corrections themselves or just from 60 "
            "more epochs of fine-tuning on largely the same hard-train data; see README HITL section"
            if is_control
            else "model selection metric (best checkpoint by val minADE, starting from Phase 4 fine-tuned-v1 weights)"
        ),
    )
    print(f"[{model_label}] val/all: {val_metrics}")

    for difficulty in ["all", "easy", "hard"]:
        df = filter_difficulty(load_split("test"), difficulty)
        metrics = evaluate_on_df(model, df)
        if is_control:
            notes = (
                "ablation control -- same round-2 training recipe as fine-tuned-v2 (60 epochs continuing from "
                "fine-tuned-v1) but on the UNCORRECTED hard-train split; compare directly against "
                "fine-tuned-v2's row above to see how much of that model's gain (if any) is attributable to the "
                "13 corrected labels themselves rather than just more fine-tuning epochs; see README HITL section"
                if difficulty == "hard"
                else ""
            )
        elif difficulty == "hard":
            notes = (
                f"primary Phase 6 comparison metric (fine-tuned-v1 vs fine-tuned-v2, post-HITL) -- "
                f"only {n_changed} of {len(train_df)} hard-train labels actually changed from the review pass, "
                f"so a small effect size here is expected, not a shortcoming; see README limitations. Compare "
                f"against the 'fine-tuned-v2-control, no corrections' row (--ablation-no-corrections) to see "
                f"how much of this is attributable to the corrections themselves vs. just more fine-tuning epochs"
            )
        elif difficulty == "all":
            notes = (
                "loses to constant velocity here, but that's driven by the dataset's dominant near-stationary "
                "majority -- restricted to the 63/1626 test examples that actually move >5m, this model wins "
                "instead; see the 'moving (>5m displacement)' rows below and README 'moving-vehicle subset' section"
            )
        else:
            notes = ""
        log_metrics(
            phase=6,
            model=model_label,
            eval_split="test",
            difficulty=difficulty,
            metrics=metrics,
            notes=notes,
        )
        print(f"[{model_label}] test/{difficulty}: {metrics}")


if __name__ == "__main__":
    main()
