"""Regularization sweep for fine-tune round 1 (models/finetune.py) -- the
step that showed test-set overfitting (minADE 0.758 -> 0.925 test/all,
confirmed across all 3 seeds in seed_variance.py). "Fine-tuning on 6
scenes is overfitting-prone" was documented as a limitation; this script
asks the natural follow-up: is it fixable, not just documentable?

Sweeps weight_decay x dropout, holding everything else -- epochs, LR,
seed, starting checkpoint, data -- identical to finetune.py. Reuses
evaluation.seed_variance.train_loop directly (warm_start=True, matching
finetune.py's fallback-to-pretrained-weights safeguard) so every run here
follows the exact same recipe finetune.py itself does, just with
regularization added. Transient: no checkpoint is saved, and the
canonical pretrained.pt/finetuned_v1.pt are only ever read, never
overwritten.

Note on early stopping: NOT swept separately, because it wouldn't do
anything here -- finetune.py (and this script, via train_loop) already
select the best-val-minADE checkpoint across all epochs rather than just
returning the final epoch, which is functionally equivalent to early
stopping with perfect hindsight. Extra epochs past the best one are never
selected, so they can't hurt (only cost wall-clock time). Regularization
(weight decay, dropout) is the lever that can actually change which
epoch's weights end up best -- that's why this sweep targets those two
instead.
"""

import itertools

import torch

from trajflow.evaluation.evaluate import filter_difficulty, load_split, log_metrics
from trajflow.evaluation.seed_variance import train_loop
from trajflow.models.finetune import EPOCHS, LR
from trajflow.models.train_pretrain import CHECKPOINT_PATH as PRETRAINED_CHECKPOINT
from trajflow.models.train_pretrain import evaluate_on_df, set_seed
from trajflow.models.transformer import TrajectoryTransformer

SEED = 0
WEIGHT_DECAYS = [0.0, 1e-4, 1e-3]
DROPOUTS = [0.1, 0.3]  # 0.1 matches TrajectoryTransformer's own default (finetune.py's implicit baseline)


def main() -> None:
    if not PRETRAINED_CHECKPOINT.exists():
        raise SystemExit(f"ERROR: no pretrained checkpoint at {PRETRAINED_CHECKPOINT}. Run trajflow-pretrain first.")

    train_df = filter_difficulty(load_split("train"), "hard")
    val_df = load_split("val")
    test_all = load_split("test")
    test_hard = filter_difficulty(test_all, "hard")

    pretrained_state = torch.load(PRETRAINED_CHECKPOINT)

    print(f"Baseline (finetune.py's actual recipe: weight_decay=0.0, dropout=0.1) for reference:")
    print("  test/all minADE=0.9249 (canonical finetuned_v1.pt, logged in results/metrics_comparison.md)")
    print()

    results = []
    for weight_decay, dropout in itertools.product(WEIGHT_DECAYS, DROPOUTS):
        set_seed(SEED)
        model = TrajectoryTransformer(dropout=dropout)
        model.load_state_dict(pretrained_state)

        model, best_val_minade = train_loop(model, train_df, val_df, EPOCHS, LR, warm_start=True, weight_decay=weight_decay)

        test_all_metrics = evaluate_on_df(model, test_all)
        test_hard_metrics = evaluate_on_df(model, test_hard)
        results.append((weight_decay, dropout, best_val_minade, test_all_metrics, test_hard_metrics))
        print(
            f"weight_decay={weight_decay:<8g} dropout={dropout:<4g} "
            f"val_minADE={best_val_minade:.4f}  test/all_minADE={test_all_metrics['minADE']:.4f}  "
            f"test/hard_minADE={test_hard_metrics['minADE']:.4f}"
        )

        label = f"Transformer (fine-tuned-v1, wd={weight_decay:g}, dropout={dropout:g})"
        notes_all = (
            f"regularization sweep over fine-tune round 1 -- same recipe as finetune.py (60 epochs from "
            f"pretrained.pt, LR={LR}) except weight_decay={weight_decay:g} (Adam) and dropout={dropout:g} "
            f"(finetune.py's implicit values are weight_decay=0.0, dropout=0.1); compare against "
            f"'Transformer (fine-tuned-v1, hard)' (test/all minADE 0.9249) to see if this combo mitigates "
            f"the round-1 overfitting regression; see README Results"
        )
        log_metrics(
            phase=10, model=label, eval_split="test", difficulty="all", metrics=test_all_metrics, notes=notes_all,
        )
        log_metrics(
            phase=10, model=label, eval_split="test", difficulty="hard", metrics=test_hard_metrics, notes="",
        )

    print("\nRanked by test/all minADE (lower is better; canonical finetune.py recipe is wd=0.0, dropout=0.1 -> 0.9249):")
    for weight_decay, dropout, best_val_minade, test_all_metrics, test_hard_metrics in sorted(results, key=lambda r: r[3]["minADE"]):
        print(f"  {test_all_metrics['minADE']:.4f}  wd={weight_decay:g} dropout={dropout:g}")


if __name__ == "__main__":
    main()
