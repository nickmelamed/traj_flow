#!/usr/bin/env bash
# Runs the full TrajFlow pipeline end-to-end, stopping right before the one
# interactive step (the HITL review app). Re-run with --resume-after-hitl
# once you've completed a review pass to run fine-tune round 2 + viz.
#
# Assumes `pip install -e .` has already been run and nuScenes mini is
# already in place (trajflow-download will tell you if it isn't).
set -euo pipefail

resume_after_hitl=false
for arg in "$@"; do
  case "$arg" in
    --resume-after-hitl) resume_after_hitl=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

run() {
  echo
  echo "=== $* ==="
  "$@"
}

if [ "$resume_after_hitl" = false ]; then
  run trajflow-download
  run trajflow-preprocess
  run trajflow-baseline-cv
  run trajflow-baseline-ca
  run trajflow-baseline-xgb
  run trajflow-pretrain
  run trajflow-finetune
  run trajflow-train-lstm
  run trajflow-train-transformer-full
  run trajflow-train-transformer-ar-full  # ablation: attention encoder + autoregressive decoder, isolates decoder style
  run trajflow-lstm-pretrain              # LSTM through the same pretrain/fine-tune/HITL lineage as the transformer
  run trajflow-lstm-finetune
  run trajflow-flag-uncertain

  cat <<'EOF'

=== Pipeline paused for human review ===
Run the HITL review app now:

    trajflow-review-app

Review at least a few flagged examples (accept / correct / tag), then
re-run this script with --resume-after-hitl to fine-tune round 2 and
regenerate the results/figures.
EOF
  exit 0
fi

run trajflow-finetune-round2
run trajflow-finetune-round2 --ablation-no-corrections  # control: isolates the corrections' effect from "just more epochs"
run trajflow-lstm-finetune-round2                       # same corrections, LSTM lineage -- see README Results
run trajflow-moving-subset-analysis                     # now also logs bootstrap CIs per model, see README
run trajflow-scene-overlay

echo
echo "=== Done. See results/metrics_comparison.md and results/figures/. ==="
echo "For an interactive view: trajflow-dashboard"
echo
echo "Optional, not run above (slow, doesn't touch any canonical checkpoint):"
echo "    trajflow-seed-variance                   # ~10 min: re-trains every lineage across 3 seeds, logs mean +/- std"
echo "    trajflow-finetune-regularization-sweep    # ~1 min: weight-decay/dropout sweep on fine-tune round 1"
