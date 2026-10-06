# Handoff: after the agent-standards retrofit

Written for a fresh Claude Code session. Read this first, then `CLAUDE.md`,
`PROGRESS.md` and `docs/SPEC.md`.

## Where things stand

- `main` holds everything. PR #1 (agent standards) and PRs #2 to #9 are merged,
  and `main` is the only branch, locally and on `origin`.
- `make ci`, `make numbers` and the Stop gate all pass on `main`. The gate runs
  the style check, `make lint`, `make typecheck`, `make numbers` and
  `make test-fast`. There are 77 tests.
- CI runs `make setup && make ci` on Ubuntu with Python 3.12, plus a separate
  `agent-checks` workflow for style and hook checks. Both passed on the head of
  PR #9. The `test` job takes about 2 to 3 minutes and `agent-checks` took up to
  about 6.
- The local data is the mini state the README describes (train 2,388, val 701,
  test 1,626, 124 corrections of which 13 changed a label). `data/CHECKSUMS.sha256`
  holds the sha256 of the processed data, flagged rows, corrections and
  checkpoints, and `shasum -a 256 -c data/CHECKSUMS.sha256` verifies them. The
  scale-up copies are in `backups/scaleup/` and the mini copies are in
  `backups/mini/`. Both are gitignored. Do not delete either.
- The README numbers are traceable. `make numbers` checks them against
  `results/metrics_comparison.md`, the saved analysis outputs
  (`results/seed_variance.txt`, `moving_subset.txt`, `regularization_sweep.txt`,
  `scene_overlay.txt`) and `results/derived_values.txt`.
  `scripts/derive_values.py` writes the last one from the results table and the
  seed variance output. Rerun it after any rerun of the analyses.
- A rerun of the three analyses on the mini state reproduced every value in the
  results table. The README's "97% of the gap" was wrong and now says 96%, since
  the table gives 95.7%.
- The README describes a 100-scene pilot on `v1.0-trainval`. Its processed data
  and corrections are in `backups/scaleup/` (train 16,878 rows, val 3,300, test
  3,789, 154 corrections). Only preprocessing, flagging and review are
  evidenced there. No pilot results are in the results table.
- `corrections/corrections.parquet` is deliberately local and gitignored. The
  repo is public and the file holds nuScenes tokens and trajectory coordinates,
  so committing it could conflict with the nuScenes terms. The checksum file
  identifies the labels behind fine-tuned-v2.
- `docs/SPEC.md` has no `[inferred]` tags left. Rule 3 is marked confirmed.

## What changed in the code

- `build_scene_splits` raises a `ValueError` when `val_scenes_from_train` is
  below 1.
- `preprocess.py` writes `data/processed/VERSION`, and `paths.active_nuscenes_version()`
  reads it. The review app, the dashboard and the scene overlay use it instead
  of hardcoding `v1.0-mini`. A missing marker falls back to mini. The mini
  restore left no marker, which is fine.
- `flag_uncertain.main` is split into `load_candidates` and `score_candidates`
  so rule 2 can be tested. New tests cover rule 1 (written split files share no
  scene) and rule 2 (flagging reads only the hard TRAIN rows).
- Ruff and mypy pass. The import blocks in `flag_uncertain.py`, `review_app.py`
  and `model_registry.py` are fenced with `# isort: off` and `# isort: on`
  because xgboost must load before torch. Missing stubs are ignored for
  nuscenes, pandas, plotly, joblib, pyquaternion, scipy, sklearn and tqdm in
  `pyproject.toml`.
- The `slow` pytest marker is registered.
- `trajflow-scene-overlay` now prints the ADE values quoted in the figure
  captions. The figures themselves did not change.

## Ground rules for this repo

- Use the `traj/` venv for everything. The system `python3` has no torch, so
  `python3 -m pytest` fails with collection errors. Use
  `traj/bin/python -m pytest -q`. A plain `make agent-check` also picks up the
  wrong Python and fails mypy on the missing xgboost stubs, so put `traj/bin`
  first on `PATH` or pass `PY=traj/bin/python`.
- Protected files need the owner's approval before each edit. The list is in
  `.claude/protected-paths`.
- Never weaken a test or a check to make it pass. Never edit
  `results/metrics_comparison.md` by hand. Rows are written by `log_metrics`.
- Commit only when the owner asks, wrapped in `timeout 40 git commit ...`.
  Signing works. The key is loaded in the ssh-agent and
  `gpg.ssh.allowedSignersFile` points at `~/.ssh/allowed_signers`, so
  `git log --format=%G?` shows `G`. If a commit ever hangs, check `ssh-add -l`
  first and give the owner the message to commit. End each message with the
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` line.
- Ask before pushing, merging, tagging, deleting branches, or changing
  dependencies, CI or hooks.
- Do not run `ruff check --fix` over the whole tree. It reorders imports in the
  files above, which brings back the xgboost and torch hang, and it edits
  protected files.
- Do not use `git checkout -- <file>` to undo an experiment on a file that has
  other uncommitted work. It discards all of it.
- Do not run `run_pipeline.sh` on the mini state unless the owner asks. It
  retrains everything and overwrites the mini checkpoints.
- The auto-mode classifier blocked an edit to `.claude/gate-commands` until the
  owner asked for it explicitly. Expect the same for other hook and gate files.

## Open items

1. Confirm the hooks run in a fresh session. Restart Claude Code, then ask it to
   `cat .env.test`. The read must be blocked. This already passed in a live
   session through both the Bash and Read hooks, and the scripted smoke test
   passes, so this only checks the case after a restart.
2. Optional cleanup. The old stash diff is saved at
   `~/Desktop/GitHub/traj_flow-stash-b2e4efe.patch`, outside the repo. Its
   pilot paragraph was reworded into the README, and the rest holds scale-up
   results rows and schema text that should not come back. The owner can delete
   it.

Nothing else is outstanding. The HITL review of the reported run is finished
(124 examples reviewed). More review would be a new experiment with its own
round-2 run, and the reported results are the mini run.

## Useful commands

```bash
traj/bin/python -m pytest -q                       # 77 tests, about 4s
make ci PY=traj/bin/python                         # lint, types, tests, style
make numbers PY=traj/bin/python                    # README numbers vs sources
shasum -a 256 -c data/CHECKSUMS.sha256             # mini artifacts match the checksums
python3 scripts/derive_values.py                   # rewrite results/derived_values.txt
python3 scripts/agent/check_style.py --changed .   # style check, changed lines
python3 scripts/agent/smoke_test.py                # hook behavior, scripted
echo '{}' | python3 scripts/agent/stop_gate.py     # run the Stop gate by hand
gh pr checks <n> --watch                           # wait for CI on a pull request
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
- The seed variance analysis retrains models and takes about 10 minutes. It and
  the other two analyses call `log_metrics`, so rerunning them rewrites rows in
  the results table. The values come out the same, but the row order can move.
- Slow CI runs are not necessarily hung ones. The `test` job has taken between
  2 and 6.5 minutes.
- In zsh, a variable holding several branch names is not split into words in a
  `for` loop. Write the names out. The pipe status array is `pipestatus`, not
  `PIPESTATUS`.
- `git apply --cached --unidiff-zero` on a hunk with no context lines can land
  the change at the wrong line. It put a README layout note under the wrong
  entry once. Check `git diff --cached` before committing a split hunk.
- `.claude/settings.json` denies reading `~/.ssh/id_*`, which also matches the
  public key. Use `ssh-add -L` to get the public key instead.
