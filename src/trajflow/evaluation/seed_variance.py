"""Multi-seed variance study for every trained (non-baseline) model.

Why this exists: every canonical checkpoint in this repo (pretrained.pt,
finetuned_v1.pt, etc.) is a single SEED=0 training run. Given how small
nuScenes mini is (train has only 6 scenes), the headline numbers could be
partly an artifact of that one lucky/unlucky initialization rather than a
property of the method -- this script re-runs the same recipes across
several seeds and reports mean +/- std, so a reader can tell whether e.g.
"fine-tuning regressed test/hard minADE" reflects a real effect or is
within the noise band of a single seed.

Design notes:
  - Reuses each canonical script's exact EPOCHS/LR/BATCH_SIZE constants
    (imported directly, not re-typed) and TrajectoryDataset/min_of_k_loss,
    so the training recipe run here is identical to the one that produced
    the canonical checkpoints -- only the seed differs.
  - Does NOT overwrite any canonical checkpoint (pretrained.pt,
    finetuned_v1.pt, ...); models trained here are transient, kept only in
    memory to compute metrics, then discarded. The canonical SEED=0
    checkpoints used everywhere else in the pipeline (dashboard, HITL
    review, scene overlays) are untouched.
  - Logs one row per model to results/metrics_comparison.md, under a
    distinct eval_split value ("test (N-seed mean +/- std)") that can't
    collide with any canonical row's key, with per-seed values spelled out
    in Notes for full transparency.
  - IMPORTANT: do not set OMP_NUM_THREADS here. Torch's default intra-op
    thread count affects floating-point summation order and therefore the
    exact trained result even at a fixed seed (verified directly during
    this study -- forcing OMP_NUM_THREADS=1 on a plain torch-only script
    changed a seed=0 rerun's test minADE from 0.7584 to 0.6346 with
    identical code). Leave threading at its default so results here are
    comparable to the canonical checkpoints' own numbers. This script
    doesn't import xgboost, so it doesn't need the OpenMP workaround that
    flag_uncertain.py/review_app.py/dashboard.py require.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from trajflow.evaluation.evaluate import filter_difficulty, load_split, log_metrics
from trajflow.models import train_lstm as train_lstm_mod
from trajflow.models import train_pretrain as train_pretrain_mod
from trajflow.models import train_transformer_full as train_transformer_full_mod
from trajflow.models.finetune import EPOCHS as FINETUNE_EPOCHS
from trajflow.models.finetune import LR as FINETUNE_LR
from trajflow.models.finetune_round2 import EPOCHS as ROUND2_EPOCHS
from trajflow.models.finetune_round2 import LR as ROUND2_LR
from trajflow.models.finetune_round2 import merge_corrections
from trajflow.models.lstm import LSTMTrajectoryModel
from trajflow.models.train_pretrain import evaluate_on_df, set_seed
from trajflow.models.transformer import TrajectoryDataset, TrajectoryTransformer, min_of_k_loss
from trajflow.paths import CORRECTIONS_PATH

SEEDS = [0, 1, 2]
BATCH_SIZE = 64  # identical across every canonical script


def train_loop(model, train_df: pd.DataFrame, val_df: pd.DataFrame, epochs: int, lr: float, warm_start: bool) -> tuple:
    """The exact training loop shared (copy-identical) by pretrain / finetune
    / finetune_round2 / train_lstm / train_transformer_full -- extracted
    here once rather than re-run through 5 separate subprocesses, since
    this script needs the trained model object in memory (not just a
    saved checkpoint) to evaluate it and then discard it.

    warm_start must match the canonical script being reproduced:
      - True  (finetune, finetune_round2): `model` arrives with meaningful
        pre-trained weights. best_val_minade starts at the PRE-finetuning
        val minADE, so if no epoch beats it, the returned model correctly
        falls back to those starting weights (a real safeguard against
        fine-tuning making things worse -- matches finetune.py/
        finetune_round2.py exactly).
      - False (pretrain, train_lstm, train_transformer_full): `model`
        arrives freshly initialized (random weights). best_val_minade
        starts at +inf so epoch 1's result is always accepted, matching
        those scripts exactly. Using warm_start=True here by mistake was
        an actual bug caught while running this study: an untrained,
        randomly-initialized model can score deceptively well on minADE
        against this dataset's near-stationary majority (small random
        weights -> near-zero-magnitude output), so the eval-before-
        training fallback could "protect" the model from ever training
        at all if no epoch happened to beat that -- producing identical
        pretrained/fine-tuned-v1/fine-tuned-v2 numbers for 2 of 3 seeds
        before this was caught and fixed.
    """
    train_dataset = TrajectoryDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if warm_start:
        best_val_minade = evaluate_on_df(model, val_df)["minADE"]
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        best_val_minade = float("inf")
        best_state = None

    for _ in range(1, epochs + 1):
        model.train()
        for past_seq, context, gt in train_loader:
            optimizer.zero_grad()
            traj, logits = model(past_seq, context)
            loss, _, _ = min_of_k_loss(traj, logits, gt)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_on_df(model, val_df)
        if val_metrics["minADE"] < best_val_minade:
            best_val_minade = val_metrics["minADE"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val_minade


def run_one_seed(seed: int, corrections_df: pd.DataFrame) -> dict:
    """Trains the pretrain -> finetune-v1 -> finetune-v2(+corrections)
    lineage, plus the LSTM and full-split-transformer comparison points,
    all at this seed. Returns {model_label: {"all": metrics, "hard": metrics}}.
    """
    train_easy = filter_difficulty(load_split("train"), "easy")
    train_hard = filter_difficulty(load_split("train"), "hard").reset_index(drop=True)
    train_full = load_split("train")
    val_df = load_split("val")
    test_all = load_split("test")
    test_hard = filter_difficulty(test_all, "hard")

    def eval_all_hard(model) -> dict:
        return {"all": evaluate_on_df(model, test_all), "hard": evaluate_on_df(model, test_hard)}

    results = {}

    set_seed(seed)
    pretrained, _ = train_loop(TrajectoryTransformer(), train_easy, val_df, train_pretrain_mod.EPOCHS, train_pretrain_mod.LR, warm_start=False)
    results["Transformer (pretrained, easy-only)"] = eval_all_hard(pretrained)

    set_seed(seed)
    finetuned_v1, _ = train_loop(pretrained, train_hard, val_df, FINETUNE_EPOCHS, FINETUNE_LR, warm_start=True)
    results["Transformer (fine-tuned-v1, hard)"] = eval_all_hard(finetuned_v1)

    set_seed(seed)
    train_hard_corrected, _ = merge_corrections(train_hard, corrections_df)
    finetuned_v2, _ = train_loop(finetuned_v1, train_hard_corrected, val_df, ROUND2_EPOCHS, ROUND2_LR, warm_start=True)
    results["Transformer (fine-tuned-v2, post-HITL)"] = eval_all_hard(finetuned_v2)

    set_seed(seed)
    lstm, _ = train_loop(LSTMTrajectoryModel(), train_full, val_df, train_lstm_mod.EPOCHS, train_lstm_mod.LR, warm_start=False)
    results["LSTM (baseline)"] = eval_all_hard(lstm)

    set_seed(seed)
    transformer_full, _ = train_loop(TrajectoryTransformer(), train_full, val_df, train_transformer_full_mod.EPOCHS, train_transformer_full_mod.LR, warm_start=False)
    results["Transformer (full-split)"] = eval_all_hard(transformer_full)

    return results


def main() -> None:
    if not CORRECTIONS_PATH.exists():
        raise SystemExit(f"ERROR: no corrections found at {CORRECTIONS_PATH}. Run the HITL review pass first.")
    corrections_df = pd.read_parquet(CORRECTIONS_PATH)

    per_seed: dict = {}
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        per_seed[seed] = run_one_seed(seed, corrections_df)
        for label, splits in per_seed[seed].items():
            print(f"  {label:45s} test/all minADE={splits['all']['minADE']:.4f}  test/hard minADE={splits['hard']['minADE']:.4f}")

    model_labels = list(per_seed[SEEDS[0]].keys())
    print("\n=== Seed variance summary (mean +/- std across seeds", SEEDS, ") ===")
    for label in model_labels:
        for split_name in ["all", "hard"]:
            values = {metric: [per_seed[s][label][split_name][metric] for s in SEEDS] for metric in ["minADE", "minFDE", "MissRate@2m"]}
            mean = {m: float(np.mean(v)) for m, v in values.items()}
            std = {m: float(np.std(v)) for m, v in values.items()}
            n = per_seed[SEEDS[0]][label][split_name]["N"]

            per_seed_str = ", ".join(f"seed{s}={per_seed[s][label][split_name]['minADE']:.4f}" for s in SEEDS)
            notes = (
                f"{len(SEEDS)}-seed variance study (seeds={SEEDS}): minADE {mean['minADE']:.4f} +/- {std['minADE']:.4f}, "
                f"minFDE {mean['minFDE']:.4f} +/- {std['minFDE']:.4f}, MissRate@2m {mean['MissRate@2m']:.4f} +/- {std['MissRate@2m']:.4f} "
                f"-- individual seeds' minADE: {per_seed_str}. Canonical SEED=0 checkpoints/rows elsewhere in this table are "
                f"unaffected (this script trains transient in-memory copies, never overwrites checkpoints/*.pt)."
            )
            log_metrics(
                phase=8,
                model=label,
                eval_split=f"test ({len(SEEDS)}-seed mean +/- std)",
                difficulty=split_name,
                metrics={"N": n, "minADE": mean["minADE"], "minFDE": mean["minFDE"], "MissRate@2m": mean["MissRate@2m"]},
                notes=notes,
            )
            print(f"  {label:45s} {split_name:4s} minADE={mean['minADE']:.4f} +/- {std['minADE']:.4f}")


if __name__ == "__main__":
    main()
