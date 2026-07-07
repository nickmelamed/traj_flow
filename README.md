# TrajFlow

A trajectory prediction pipeline for autonomous vehicles, built on nuScenes
**mini**, combining three things in one project:

1. **Classical ML baselines** — constant-velocity and XGBoost models.
2. **A fine-tuned deep learning model** — a compact trajectory transformer,
   pretrained on "easy" driving scenes and fine-tuned on "hard" ones
   (dense traffic / intersections).
3. **A human-in-the-loop (HITL) active learning loop** — uncertain
   predictions are flagged, reviewed and corrected via a Streamlit app, and
   fed back into a second fine-tuning round.

This is a portfolio project for autonomy / planning & prediction engineering
roles. It prioritizes an honestly-measured, end-to-end pipeline over a
state-of-the-art model.

> **Status:** work in progress. This README will be filled in fully once all
> phases are complete (see `CLAUDE.md` for the build spec and phase list).
> Metrics live in [`results/metrics_comparison.md`](results/metrics_comparison.md).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost additionally requires the OpenMP runtime:

```bash
brew install libomp
```

## Repo layout

See `CLAUDE.md` for the full build spec, phase breakdown, and acceptance
criteria driving this project.
