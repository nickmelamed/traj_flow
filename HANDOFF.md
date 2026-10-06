# Handoff: finishing the agent-standards retrofit

Written for a fresh Claude Code session. Read this first, then `CLAUDE.md`,
`PROGRESS.md` and `docs/SPEC.md`.

## Where things stand

- PR #1 (`chore/agent-standards`) is merged into `main`. The local branch is
  deleted.
- The branch adds the agent tooling (standards v2.0.0), rewrites `CLAUDE.md`,
  adds `docs/SPEC.md` and `docs/DECISIONS.md`, cleans up comments and the
  README, and adds 34 tests (66 in total). No code behavior changed, and no
  README or results number changed.
- The Stop gate runs the style check and `PY=traj/bin/python make test-fast`.
  Lint, types and the README number check are not in the gate because they
  do not pass yet.

## Ground rules for this repo

- Use the `traj/` venv for everything. The system `python3` has no torch, so
  `python3 -m pytest` fails with collection errors. Use
  `traj/bin/python -m pytest -q`.
- Protected files need the owner's approval before each edit. The list is in
  `.claude/protected-paths` and includes `src/trajflow/evaluation/`,
  `data/preprocess.py`, `hitl/flag_uncertain.py`, `paths.py`,
  `results/metrics_comparison.md`, `data/SCHEMA.md`, `backups/` and
  `corrections/`.
- Never weaken a test or a check to make it pass. Never edit
  `results/metrics_comparison.md` by hand. Rows are written by `log_metrics`.
- Commits need the owner's go-ahead. When asked, run them with
  `timeout 40 git commit ...`, since SSH signing once hung. End each message
  with the `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` line.
- Ask before pushing, merging, tagging, or changing dependencies.
- Do not train or evaluate against the current local data. See task 1.

## Tasks, in the order I would do them

### 1. Restore the mini state before touching data or results (done)

Local `data/processed/`, `artifacts/flagged.parquet` and `corrections/` come
from a scale-up run (train 16,878 rows over 72 scenes, 679 flagged, 154
corrections). The checkpoints are still the mini ones. The README and results
table describe mini (train 2,388, val 701, test 1,626, 124 corrections of
which 13 changed a label). Run the `restore-mini-state` skill first. The mini
copies are in `backups/mini/`, which is gitignored. Do not delete it.

### 2. Confirm the hooks run in a live session

Restart Claude Code, then ask Claude to `cat .env.test`. The read must be
blocked. Only the scripted smoke test has exercised the hooks so far. While
building this branch, edits to protected files did not prompt, which suggests
the hooks were not loaded in that session.

### 3. Fix the `--val-scenes-from-train 0` bug

`build_scene_splits` in `src/trajflow/data/preprocess.py` slices with
`train_scene_names[-val_scenes_from_train:]`. With 0, `[-0:]` returns the
whole list, so every train scene becomes val and train is empty. Nothing
warns. The default is 2, so no reported result is affected.

Write a failing test first (`tests/test_data_scaleup.py` already tests the
function), then fix it, either by rejecting values below 1 with a clear error
or by slicing from `len(...) - n`. Ask the owner which. The file is protected.

### 4. Get ruff to pass, then add it to the gate

`make lint` runs `ruff check src tests`. It reports 72 errors:

| Rule | Count | Notes |
|---|---|---|
| C408 | 23 | `dict()` and `list()` calls, safe to rewrite |
| RUF059 | 21 | unused unpacked variables, rename to `_` |
| I001 | 15 | import sorting, auto-fixable |
| F401 | 8 | unused imports, auto-fixable |
| RUF046 | 2 | unnecessary `int()` cast |
| UP035, F541, RUF013 | 1 each | small |

Run `ruff check --fix` for the 26 safe fixes and review the diff. Check each
other fix by hand, since some files are protected. Keep this as its own PR so
the diff stays reviewable. When it passes, add `make lint` to
`.claude/gate-commands` as `PY=traj/bin/python make lint`.

### 5. Get mypy to pass, then add it to the gate

`make typecheck` runs `mypy src` and reports 72 errors. 47 are
`import-untyped` (nuscenes, scipy, pandas, tqdm have no stubs), 23 are
`arg-type`, and 2 are `assignment`. The first group is a configuration fix, for
example `ignore_missing_imports` for `nuscenes.*` in a `[tool.mypy]` section,
or installing `pandas-stubs` and `types-tqdm`. Both change `pyproject.toml`
(dependencies or config), which needs approval. Do the real type errors after
that. Add `make typecheck` to the gate once it passes.

### 6. Make the README numbers traceable

`make numbers` fails on 20 README numbers. They come from three scripts that
only print to the terminal. After task 1, run the `repro-mini` skill. It saves
`trajflow-seed-variance`, `trajflow-moving-subset-analysis` and
`trajflow-finetune-regularization-sweep` output to `results/*.txt`, which
`make numbers` already lists as sources. Seed variance retrains models and
takes about 10 minutes. Some numbers are derived and will still fail, for
example the per-seed gaps (+0.165, +0.140, +0.210), "97% of the gap closed",
and the per-scene ADEs in the figure captions. For those, either generate a
derived-values table or mark the line `numbers: ok` with the owner's
approval. When `make numbers` passes, add it to the gate.

If the retrained numbers differ from the README (training is seeded but not
bit-identical), report the differences. Do not edit the results table to
match.

### 7. Decide what to do with `stash@{0}`

The stash is named "stashing any uncommitted changes before repo cleanup". It
touches `README.md`, `results/metrics_comparison.md`, `data/SCHEMA.md`,
`.gitignore`, `preprocess.py`, `paths.py`, `review_app.py`, `dashboard.py`,
`scene_overlay.py` and `tests/test_data_scaleup.py`. Its README and results
edits will conflict with the README rewrite. Inspect it read-only
(`git stash show -p stash@{0}`) and tell the owner whether anything is worth
porting. Do not apply it blindly.

### 8. Close the test gaps for the rules

`docs/SPEC.md` rules 1 and 2 are only partly tested.
- Rule 1 (no scene leakage): `test_data_scaleup.py` checks the split function.
  Nothing checks processed parquet files for scene overlap. A test that loads
  a small synthetic frame through the extraction path would need nuScenes, so a
  check on the written parquet (no `scene_name` in two splits) is the cheap
  option.
- Rule 2 (HITL uses TRAIN only): `flag_uncertain.main` is not tested. Move the
  scoring into an importable function, then test that it only reads train rows.
- Uncovered by design: the Streamlit apps, the training loops, and
  `preprocess.extract_examples` (needs nuScenes).

### 9. Confirm the draft docs

`docs/SPEC.md` and `docs/DECISIONS.md` are drafts. Ask the owner to confirm
the lines tagged [inferred] in SPEC.md (rules 4, 6 and 7), and to check two
details in DECISIONS.md that came from code comments, not from the owner:
D2's "roughly 55/45" hard/easy split, and D5's "test minADE 0.602 with the
heading feature". Then drop the "Draft" status lines.

### 10. Smaller items

- Register a `slow` pytest marker in `pyproject.toml` (`make test-fast` uses
  `-m "not slow"`, which warns about the unknown marker if one is used).
- Once `make ci` passes, enable the commented "Project checks" step in
  `.github/workflows/agent-checks.yml`.
- Decide whether to commit the mini `corrections.parquet`. `corrections/` is
  gitignored, so the labels behind fine-tuned-v2 cannot be reproduced from the
  repo. Also consider data checksums for processed data and checkpoints.
- The README rewrite was mechanical plus a manual pass. The owner should read
  it once for tone.
- After PR #1 merges, delete the branch and rebase any follow-up work on
  `main`.

## Useful commands

```bash
traj/bin/python -m pytest -q                       # 66 tests, about 3s
python3 scripts/agent/check_style.py .             # style check, whole repo
python3 scripts/agent/check_style.py --changed .   # style check, changed lines
python3 scripts/agent/smoke_test.py                # hook behavior, scripted
echo '{}' | python3 scripts/agent/stop_gate.py     # run the Stop gate by hand
make numbers                                       # README numbers vs sources
```

## Things that surprised me

- `.claude/style-ignore` does not support trailing comments. Put a comment on
  its own line. `.claude/protected-paths` does support them.
- The style checker only reads comments, docstrings and prose. It does not
  read string literals, so the results table notes (written by the training
  scripts) are skipped. The table itself is in `style-ignore`.
- `trajflow-preprocess` also rewrites `data/SCHEMA.md`. The text template for
  that file lives in `preprocess.py`, so edit both together.
- Several modules need xgboost imported before torch, with `KMP_DUPLICATE_LIB_OK`
  and `OMP_NUM_THREADS` set first. Keep that import order.
