from trajflow.evaluation import evaluate


def _metrics(n=10, ade=1.0, fde=2.0, miss=0.1):
    return {"N": n, "minADE": ade, "minFDE": fde, "MissRate@2m": miss}


def test_log_metrics_appends_new_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS_PATH", tmp_path / "metrics.md")

    evaluate.log_metrics(phase=1, model="A", eval_split="test", difficulty="all", metrics=_metrics())
    evaluate.log_metrics(phase=1, model="B", eval_split="test", difficulty="all", metrics=_metrics())

    rows = evaluate._read_existing_rows()
    assert len(rows) == 2


def test_log_metrics_replaces_same_key_instead_of_duplicating(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS_PATH", tmp_path / "metrics.md")

    evaluate.log_metrics(phase=1, model="A", eval_split="test", difficulty="all", metrics=_metrics(ade=1.0))
    evaluate.log_metrics(phase=1, model="A", eval_split="test", difficulty="all", metrics=_metrics(ade=9.0))

    rows = evaluate._read_existing_rows()
    assert len(rows) == 1
    assert "9.0000" in rows[0]
    assert "1.0000" not in rows[0]


def test_log_metrics_does_not_touch_rows_with_a_different_key(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS_PATH", tmp_path / "metrics.md")

    evaluate.log_metrics(phase=1, model="A", eval_split="test", difficulty="all", metrics=_metrics())
    evaluate.log_metrics(phase=1, model="A", eval_split="test", difficulty="easy", metrics=_metrics())
    evaluate.log_metrics(phase=1, model="A", eval_split="val", difficulty="all", metrics=_metrics())

    assert len(evaluate._read_existing_rows()) == 3


def test_notes_with_pipe_and_newline_do_not_corrupt_the_table(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS_PATH", tmp_path / "metrics.md")

    evaluate.log_metrics(
        phase=1, model="A", eval_split="test", difficulty="all", metrics=_metrics(),
        notes="a | malicious note\nwith a newline",
    )
    rows = evaluate._read_existing_rows()
    assert len(rows) == 1
    # 9 columns (8 separators) even though the note itself tried to inject a "|"
    assert rows[0].count("|") == 10
    assert "\n" not in rows[0].strip()
    key = evaluate._row_key(rows[0])
    assert key == ("1", "A", "test", "all")
