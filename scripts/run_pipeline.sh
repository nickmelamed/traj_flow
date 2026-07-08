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
run trajflow-moving-subset-analysis
run trajflow-scene-overlay

echo
echo "=== Done. See results/metrics_comparison.md and results/figures/. ==="
echo "For an interactive view: trajflow-dashboard"
