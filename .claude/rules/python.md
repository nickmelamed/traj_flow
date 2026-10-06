---
paths:
  - "src/**/*.py"
  - "scripts/**/*.py"
---

# Python in this repo

- Python 3.11, type hints on new public functions. Use `pathlib` and the
  constants in `trajflow/paths.py`. Do not hardcode data paths.
- Shapes are `[N, T, 2]` for one trajectory and `[N, K, T, 2]` for K modes. Put
  the shape in a docstring or comment when a function takes either.
- Metrics code takes unrounded floats. Round only when formatting a table.
- Seed everything through `set_seed` in `models/train_pretrain.py`. A new
  training script needs a fixed seed and a val-based checkpoint choice.
- Anything that writes to `results/` goes through `log_metrics`.
- Docstrings follow the repo's current Sphinx style (`:param x:`) until a
  linter says otherwise. Do not rewrite existing ones just to change style.
- Streamlit apps (`hitl/review_app.py`, `viz/dashboard.py`) are not covered by
  tests. Keep logic in importable functions so it can be tested.
