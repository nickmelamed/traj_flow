# Progress

Current phase and what is left, updated at the end of each phase. The "Now"
section is loaded at the start of each session.

## Now

- [x] Agent standards v2.0.0 installed and merged to `main` (PR #1). Hooks,
  protected paths, reviewer agents and six owner-only repo skills are in place.
- [x] Spec (`docs/SPEC.md`) and decision log (`docs/DECISIONS.md`) reviewed by
  the owner. Rules 3, 4, 6 and 7 are confirmed and no lines are tagged
  [inferred].
- [x] Characterization, property and rule-coverage tests added (77 tests, fast
  gate runs them). Lint and types pass and CI is green on `main`.
- [x] The local data is the mini state, and `data/CHECKSUMS.sha256` verifies it.
  The scale-up copies are in `backups/scaleup/`.
- [x] PRs #1 to #9 are merged and `main` is the only branch.
- [x] The README is checked against `results/metrics_comparison.md` by
  `make numbers`, and it now describes the 100-scene trainval pilot.

## Next

- [ ] Restart Claude Code and confirm the hooks run in a fresh session. Ask
  Claude to `cat .env.test` and check it is blocked. This passed in a live
  session already, so it only needs repeating after a restart.

## Done

- Phase 0 assessment: retrofit chosen, no secrets or private data in git or
  history, no notebooks.
- Phase 1: SPEC.md and DECISIONS.md drafted, one interview round.
- Phase 2: installer run, CLAUDE.md rewritten, python rule, protected paths,
  Makefile, `backups/` ignored, `dev` extra now has hypothesis, ruff and mypy,
  four skills created.
- Phase 3: style cleanup. Stale docstring numbers in `finetune_round2.py` fixed
  to 111 accepted and 13 corrected.
- Phase 4: tests added, mutation spot-check caught all three mutants.
- Feature and fix branches merged as PRs #2 to #7. They cover the ruff and mypy
  cleanup, the rule 1 and rule 2 tests, the derived values file, the
  `val_scenes_from_train` check and the processed version marker.
- PR #9: handoff notes, README tone and one corrected CI claim, test counts,
  the checksum file, resolved `[inferred]` tags and the pilot wording.
- The old stash was dropped and the old `chore/agent-standards` branch was
  deleted.

## Decisions for the owner, already made

- `corrections/corrections.parquet` stays local and gitignored. The repo is
  public and the file holds nuScenes tokens and trajectory coordinates, which
  may fall under the nuScenes terms. The labels behind fine-tuned-v2 are not
  reproducible from the repo, and the checksum file identifies them.
- The HITL review of the reported run is finished, with 124 examples reviewed.
  More review would be a new experiment with its own round-2 run, so it is not
  part of the open work.
