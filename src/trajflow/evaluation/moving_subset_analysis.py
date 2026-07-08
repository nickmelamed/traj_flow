"""Evaluates every registered model on the "moving vehicles" subset of the
test set (>5m net displacement over 6s) and logs each as its own row in
results/metrics_comparison.md (phase 7, difficulty="moving (>5m
displacement)").

Why this exists: over 90% of test examples are near-stationary (median
displacement 0.16m), so aggregate test/all metrics are dominated by that
majority. Constant velocity wins in aggregate almost entirely because of
it -- restricted to vehicles that actually move, several learned models
do better. This is the single script that evaluates ALL models on that
subset consistently (previously computed ad hoc for just two models
inside viz/scene_overlay.py; this supersedes that).
"""

import numpy as np

from trajflow.evaluation.evaluate import future_xy, load_split, log_metrics
from trajflow.evaluation.metrics import batch_metrics
from trajflow.viz.model_registry import MODEL_SPECS

DISPLACEMENT_THRESHOLD = 5.0


def main() -> None:
    test_df = load_split("test").reset_index(drop=True)
    gts = future_xy(test_df)
    displacement = np.linalg.norm(gts[:, -1, :], axis=-1)
    moving_mask = displacement > DISPLACEMENT_THRESHOLD
    n_moving = int(moving_mask.sum())
    print(f"Moving (>{DISPLACEMENT_THRESHOLD}m displacement) examples: {n_moving} / {len(test_df)} total test examples.")

    results = {}
    for spec in MODEL_SPECS:
        predict_fn = spec.loader()
        traj, _ = predict_fn(test_df)
        metrics = batch_metrics(traj[moving_mask], gts[moving_mask])
        results[spec.label] = metrics["minADE"]
        log_metrics(
            phase=7,
            model=spec.label,
            eval_split="test",
            difficulty=f"moving (>{DISPLACEMENT_THRESHOLD:g}m displacement)",
            metrics=metrics,
            notes=f"restricted to the {n_moving}/{len(test_df)} test examples with >{DISPLACEMENT_THRESHOLD:g}m net "
            "displacement over 6s -- see README 'moving-vehicle subset' section for why this subset matters",
        )
        print(f"  {spec.label:30s} minADE={metrics['minADE']:.4f}  minFDE={metrics['minFDE']:.4f}  MissRate={metrics['MissRate@2m']:.4f}")

    print("\nRanked by minADE on moving vehicles (lower is better):")
    for label, ade in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {ade:.4f}  {label}")


if __name__ == "__main__":
    main()
