# Progress

Current phase and what is left, updated at the end of each phase. The "Now"
section is loaded at the start of each session.

## Now

- [x] Agent standards v2.0.0 installed and merged to `main` (PR #1). Hooks,
  protected paths, reviewer agents and six owner-only repo skills are in place.
- [x] Spec (`docs/SPEC.md`) and decision log (`docs/DECISIONS.md`) reviewed by
  the owner. Rules 4, 6 and 7 are confirmed. A few lines are still tagged
  [inferred].
- [x] Style cleanup of comments and docs, with no behavior change.
- [x] Characterization and property tests added (66 tests, fast gate runs them).
- [x] Mini state restored in the live directories. The scale-up copies are in
  `backups/scaleup/`.

## Next

- [ ] Push the branch stack and open pull requests (owner decides). Merge
  order is `chore/ruff-clean`, `chore/mypy-clean`, `test/rule-coverage`, then
  `feat/derived-values`. `fix/val-scenes-zero` and
  `feat/processed-version-marker` stand alone.
- [ ] Decide whether to drop `stash@{0}` after the version marker port.
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
- **Corrections are gitignored.** The HITL labels behind fine-tuned-v2 are not
  reproducible from the repo. Consider committing the mini `corrections.parquet`.
- **No data checksums** exist for processed data or checkpoints.
