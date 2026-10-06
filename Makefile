# Standard targets. Adjust the tool commands to this repo, keeping the names,
# since CLAUDE.md, the Stop hook, and CI refer to them.
.PHONY: setup lint typecheck test test-fast style numbers agent-check ci

PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests

typecheck:
	$(PY) -m mypy src

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -x -q -m "not slow"

style:
	$(PY) scripts/agent/check_style.py --changed .

# Add the documents that report results, and point --sources at the
# generated tables. Remove this target if the repo reports no results.
numbers:
	$(PY) scripts/agent/check_numbers.py README.md --sources results/metrics_comparison.md results/seed_variance.txt results/moving_subset.txt results/regularization_sweep.txt results/derived_values.txt results/scene_overlay.txt

# The fast checks the Stop hook runs before Claude may end a turn. Keep the
# whole target under a few minutes.
agent-check: lint typecheck test-fast style

ci: lint typecheck test style
	$(PY) scripts/agent/check_tests.py --warn
