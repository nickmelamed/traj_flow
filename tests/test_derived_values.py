import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "derive_values.py"
spec = importlib.util.spec_from_file_location("derive_values", SCRIPT)
derive_values = importlib.util.module_from_spec(spec)
spec.loader.exec_module(derive_values)


def _row(model, difficulty, min_ade):
    return f"| 2 | {model} | test | {difficulty} | 10 | {min_ade} | 1.0000 | 0.1000 | a note |"


def _table():
    rows = [
        _row(derive_values.PRETRAINED, "all", "0.7000"),
        _row(derive_values.FINETUNED_V1, "all", "0.9000"),
        _row(derive_values.WD_REGULARIZED, "all", "0.8000"),
        _row(derive_values.FULL_SPLIT, "all", "0.8000"),
        _row(derive_values.AR_FULL, "all", "0.3000"),
        _row(derive_values.LSTM, "all", "0.2000"),
        _row(derive_values.FULL_SPLIT, derive_values.MOVING, "5.0000"),
        _row(derive_values.AR_FULL, derive_values.MOVING, "4.0000"),
        _row(derive_values.LSTM, derive_values.MOVING, "3.0000"),
    ]
    return "| Phase | Model | Eval Split | Difficulty | N | minADE | minFDE | Miss | Notes |\n" + "\n".join(rows)


def _seeds():
    return (
        "=== seed 0 ===\n"
        f"  {derive_values.PRETRAINED:<45} test/all minADE=0.5000  test/hard minADE=0.6000\n"
        f"  {derive_values.FINETUNED_V1:<45} test/all minADE=0.6500  test/hard minADE=0.6000\n"
        "=== seed 1 ===\n"
        f"  {derive_values.PRETRAINED:<45} test/all minADE=0.8000  test/hard minADE=0.6000\n"
        f"  {derive_values.FINETUNED_V1:<45} test/all minADE=0.7000  test/hard minADE=0.6000\n"
    )


def test_table_min_ade_picks_the_requested_row():
    table = _table()
    assert derive_values.table_min_ade(table, derive_values.AR_FULL, derive_values.MOVING) == 4.0
    assert derive_values.table_min_ade(table, derive_values.AR_FULL, "all") == 0.3
    with pytest.raises(KeyError):
        derive_values.table_min_ade(table, "No such model", "all")


def test_seed_min_ade_groups_values_by_seed():
    parsed = derive_values.seed_min_ade(_seeds())
    assert parsed[0][derive_values.FINETUNED_V1] == 0.65
    assert parsed[1][derive_values.PRETRAINED] == 0.8


def test_share_closed_is_the_fraction_of_the_gap_covered():
    assert derive_values.share_closed(0.8, 0.3, 0.2) == pytest.approx(5 / 6)


def test_derive_reports_gaps_and_shares_with_signs():
    text = derive_values.derive(_table(), _seeds())
    assert "seed 0: +0.1500" in text
    assert "seed 1: -0.1000" in text
    assert "round-1 regression (fine-tuned-v1 minus pretrained): +0.2000" in text
    assert "regression reduction from wd=1e-3: 50.0%" in text
    assert "test/all: 83.3%" in text
    assert "moving subset: 50.0%" in text
