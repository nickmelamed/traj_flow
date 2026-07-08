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

Statistical caveat this script exists to quantify, not just state: the
moving subset is only 63 test examples. A 95% bootstrap CI (2,000
resamples) is computed for every model's minADE, and a PAIRED bootstrap
(same resample indices reused across models, since it's the same 63
examples every time) reports the fraction of resamples where a model beats
Constant Velocity -- a direct, non-parametric answer to "is this difference
believable at this sample size," rather than eyeballing point estimates.
"""

import numpy as np

from trajflow.evaluation.evaluate import future_xy, load_split, log_metrics
from trajflow.evaluation.metrics import batch_metrics
from trajflow.viz.model_registry import MODEL_SPECS

DISPLACEMENT_THRESHOLD = 5.0
N_BOOT = 2000
BOOT_SEED = 0
CI = 0.95


def per_example_min_ade(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Same reduction batch_metrics uses internally (min-of-K ADE), but
    returns the per-example [N] array instead of the aggregate mean, so it
    can be bootstrap-resampled.
    """
    if preds.ndim == 3:
        preds = preds[:, None, :, :]
    dists = np.linalg.norm(preds - gts[:, None, :, :], axis=-1)  # [N, K, T]
    return dists.mean(axis=-1).min(axis=-1)  # [N]


def bootstrap_ci(per_example: np.ndarray, boot_idx: np.ndarray) -> tuple[float, float, np.ndarray]:
    """boot_idx: [N_BOOT, n] indices into per_example, shared across models
    so cross-model comparisons below are paired (same resamples).
    Returns (lo, hi, boot_means) for a `CI`-width percentile interval.
    """
    boot_means = per_example[boot_idx].mean(axis=1)  # [N_BOOT]
    lo, hi = np.percentile(boot_means, [(1 - CI) / 2 * 100, (1 + CI) / 2 * 100])
    return float(lo), float(hi), boot_means


def main() -> None:
    test_df = load_split("test").reset_index(drop=True)
    gts = future_xy(test_df)
    displacement = np.linalg.norm(gts[:, -1, :], axis=-1)
    moving_mask = displacement > DISPLACEMENT_THRESHOLD
    n_moving = int(moving_mask.sum())
    print(f"Moving (>{DISPLACEMENT_THRESHOLD}m displacement) examples: {n_moving} / {len(test_df)} total test examples.")
    print(f"Bootstrap: {N_BOOT} resamples, seed={BOOT_SEED}, {CI:.0%} percentile CI, paired across models (same resample indices).")

    moving_gts = gts[moving_mask]
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, n_moving, size=(N_BOOT, n_moving))

    per_example_by_model: dict[str, np.ndarray] = {}
    boot_means_by_model: dict[str, np.ndarray] = {}
    results = {}

    for spec in MODEL_SPECS:
        predict_fn = spec.loader()
        traj, _ = predict_fn(test_df)
        moving_traj = traj[moving_mask]
        metrics = batch_metrics(moving_traj, moving_gts)

        per_ex = per_example_min_ade(moving_traj, moving_gts)
        lo, hi, boot_means = bootstrap_ci(per_ex, boot_idx)
        per_example_by_model[spec.label] = per_ex
        boot_means_by_model[spec.label] = boot_means
        results[spec.label] = metrics["minADE"]

        notes = (
            f"restricted to the {n_moving}/{len(test_df)} test examples with >{DISPLACEMENT_THRESHOLD:g}m net "
            f"displacement over 6s -- see README 'moving-vehicle subset' section for why this subset matters. "
            f"minADE {CI:.0%} bootstrap CI [{lo:.3f}, {hi:.3f}] m ({N_BOOT} resamples of n={n_moving}) -- "
            f"at this sample size, treat point estimates as directional, not precise; see README 'Statistical "
            f"caveats' subsection"
        )
        log_metrics(
            phase=7,
            model=spec.label,
            eval_split="test",
            difficulty=f"moving (>{DISPLACEMENT_THRESHOLD:g}m displacement)",
            metrics=metrics,
            notes=notes,
        )
        print(
            f"  {spec.label:45s} minADE={metrics['minADE']:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  "
            f"minFDE={metrics['minFDE']:.4f}  MissRate={metrics['MissRate@2m']:.4f}"
        )

    print("\nRanked by minADE on moving vehicles (lower is better):")
    for label, ade in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {ade:.4f}  {label}")

    cv_label = "Constant Velocity"
    if cv_label in boot_means_by_model:
        print(f"\nPaired bootstrap vs. {cv_label} (fraction of the {N_BOOT} paired resamples where the model beats CV's minADE):")
        cv_boot = boot_means_by_model[cv_label]
        for label, boot_means in sorted(boot_means_by_model.items(), key=lambda kv: results[kv[0]]):
            if label == cv_label:
                continue
            win_frac = float((boot_means < cv_boot).mean())
            print(f"  {label:45s} beats CV in {win_frac:.1%} of resamples")


if __name__ == "__main__":
    main()
