# Progress

Current phase and what is left, updated at the end of each phase. The "Now"
section is loaded at the start of each session.

## Now

- [x] Agent standards v2.0.0 installed and merged to `main` (PR #1). Hooks,
  protected paths, reviewer agents and six owner-only repo skills are in place.
- [x] Spec (`docs/SPEC.md`) and decision log (`docs/DECISIONS.md`) drafted from
  the code. Items tagged [inferred] still need the owner's confirmation.
- [x] Style cleanup of comments and docs, with no behavior change.
- [x] Characterization and property tests added (66 tests, fast gate runs them).
- [x] Mini state restored in the live directories. The scale-up copies are in
  `backups/scaleup/`.

## Next

- [ ] Fix `build_scene_splits` when `--val-scenes-from-train 0` is passed. The
  `[-0:]` slice puts every train scene into val and leaves train empty. Write a
  failing test first. The file is protected.
- [ ] Get ruff and mypy to pass (72 errors each, 26 of the ruff ones are
  auto-fixable), then add `make lint typecheck` to `.claude/gate-commands`.
- [ ] Make README numbers traceable, then add `make numbers` to the gate. See
  the open questions below.
- [ ] Decide what to do with the scale-up work (see open questions).
- [ ] Register a `slow` pytest marker in `pyproject.toml` if slow tests appear.
- [ ] Restart Claude Code and confirm the hooks run: ask Claude to
  `cat .env.test` and check it is blocked.

## Done

- Phase 0 assessment: retrofit chosen, no secrets or private data in git or
  history, no notebooks.
- Phase 1: SPEC.md and DECISIONS.md drafted, one interview round.
- Phase 2: installer run, CLAUDE.md rewritten (83 lines), python rule, protected
  paths, Makefile, `backups/` ignored, `dev` extra now has hypothesis, ruff and
  mypy, four skills created.
- Phase 3: style cleanup. Stale docstring numbers in `finetune_round2.py` fixed
  to 111 accepted and 13 corrected.
- Phase 4: tests added, mutation spot-check caught all three mutants.
- README prose rewritten after review. No numbers changed. Read it once for tone.

## Open questions for the owner

- **`git stash@{0}`** holds uncommitted edits from before an earlier cleanup (README,
  results table, preprocess, paths, review app, dashboard, scene overlay,
  `test_data_scaleup.py`). It was left alone on purpose. Its README and results
  edits would collide with the style cleanup, so inspect it before applying.
- **20 README numbers have no source file.** `make numbers` fails on them. They
  come from three scripts that only print: `trajflow-seed-variance`,
  `trajflow-moving-subset-analysis` and `trajflow-finetune-regularization-sweep`.
  The `repro-mini` skill saves their output to `results/*.txt`. Some numbers
  are derived (per-seed gaps, "97% of the gap", per-scene ADEs in captions)
  and will still need `numbers: ok` or a generated table.
- **Spec items still tagged [inferred]:** same test set for every comparison, val-only
  model selection, no absolute heading in learned models, and whether the
  scale-up is infrastructure only.
- **Corrections are gitignored.** The HITL labels behind fine-tuned-v2 are not
  reproducible from the repo. Consider committing the mini `corrections.parquet`.
- **No data checksums** exist for processed data or checkpoints.
