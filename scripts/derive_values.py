#!/usr/bin/env python3
"""Write the README's derived numbers (gaps and shares) from generated results.

Reads ``results/metrics_comparison.md`` and ``results/seed_variance.txt`` and
writes ``results/derived_values.txt``. Every value is computed from the
unrounded table entries, so a README number that matches a line here is
traceable to the results table.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "results" / "metrics_comparison.md"
SEEDS = ROOT / "results" / "seed_variance.txt"
OUT = ROOT / "results" / "derived_values.txt"

PRETRAINED = "Transformer (pretrained, easy-only)"
FINETUNED_V1 = "Transformer (fine-tuned-v1, hard)"
FULL_SPLIT = "Transformer (full-split)"
AR_FULL = "Transformer-AR (full-split, autoregressive decoder)"
LSTM = "LSTM (baseline)"
WD_REGULARIZED = "Transformer (fine-tuned-v1, wd=0.001, dropout=0.1)"
MOVING = "moving (>5m displacement)"


def table_min_ade(text: str, model: str, difficulty: str) -> float:
    """Return the test minADE of ``model`` in the results table."""
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and cells[1] == model and cells[2] == "test" and cells[3] == difficulty:
            return float(cells[5])
    raise KeyError(f"no test/{difficulty} row for {model!r}")


def seed_min_ade(text: str) -> dict[int, dict[str, float]]:
    """Return test/all minADE per model for each seed in the seed variance output."""
    out: dict[int, dict[str, float]] = {}
    seed = None
    for line in text.splitlines():
        m = re.match(r"=== seed (\d+) ===", line)
        if m:
            seed = int(m.group(1))
            out[seed] = {}
            continue
        m = re.match(r"\s+(.+?)\s+test/all minADE=([\d.]+)", line)
        if m and seed is not None:
            out[seed][m.group(1)] = float(m.group(2))
    return out


def share_closed(start: float, end: float, target: float) -> float:
    """Return the fraction of the gap from ``start`` to ``target`` that ``end`` covers."""
    return (start - end) / (start - target)


def derive(table_text: str, seed_text: str) -> str:
    lines = ["Derived from results/metrics_comparison.md and results/seed_variance.txt.", ""]

    seeds = seed_min_ade(seed_text)
    lines.append("Fine-tuned-v1 minus pretrained, test/all minADE, per seed:")
    for seed in sorted(seeds):
        gap = seeds[seed][FINETUNED_V1] - seeds[seed][PRETRAINED]
        lines.append(f"  seed {seed}: {gap:+.4f}")

    pre = table_min_ade(table_text, PRETRAINED, "all")
    v1 = table_min_ade(table_text, FINETUNED_V1, "all")
    wd = table_min_ade(table_text, WD_REGULARIZED, "all")
    lines += ["", "Canonical run, test/all minADE:"]
    lines.append(f"  round-1 regression (fine-tuned-v1 minus pretrained): {v1 - pre:+.4f}")
    lines.append(f"  with wd=1e-3 (minus pretrained): {wd - pre:+.4f}")
    lines.append(f"  regression reduction from wd=1e-3: {100 * (1 - (wd - pre) / (v1 - pre)):.1f}%")

    full, ar, lstm = (table_min_ade(table_text, m, "all") for m in (FULL_SPLIT, AR_FULL, LSTM))
    lines += ["", "Share of the full-split transformer to LSTM gap closed by the AR decoder:"]
    lines.append(f"  test/all: {100 * share_closed(full, ar, lstm):.1f}%")

    full, ar, lstm = (table_min_ade(table_text, m, MOVING) for m in (FULL_SPLIT, AR_FULL, LSTM))
    lines.append(f"  moving subset: {100 * share_closed(full, ar, lstm):.1f}%")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.write_text(derive(TABLE.read_text(), SEEDS.read_text()))
    print(OUT.read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
