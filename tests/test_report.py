"""The report pins its period and builds every figure without a display."""

import pandas as pd
import pytest

from ohlcv_lab import report

SYMBOLS = ["AAAUSDT", "BBBUSDT"]
START = pd.Timestamp("2024-01-15", tz="UTC")
END = pd.Timestamp("2024-03-01", tz="UTC")


@pytest.fixture
def closes(layer):
    return report.load_period(SYMBOLS, "60", START, END, layer[0])


def test_period_is_half_open_and_exact(closes):
    assert closes.index[0] == START
    assert closes.index[-1] == END - pd.Timedelta(hours=1)
    assert len(closes) == (END - START) / pd.Timedelta(hours=1)


def test_period_not_covered_is_an_error(layer):
    # BBBUSDT is listed on 2024-01-11: an earlier start must fail, not shrink the period.
    with pytest.raises(ValueError, match="fetch the period first"):
        report.load_period(SYMBOLS, "60", pd.Timestamp("2024-01-05", tz="UTC"), END, layer[0])


def test_build_writes_every_figure(closes, tmp_path):
    paths = report.build(closes, tmp_path)
    assert [p.stem for p in paths] == list(report.FIGURES)
    assert all(p.stat().st_size > 10_000 for p in paths)


def test_markdown_table_has_a_row_per_coin(closes):
    md = report.markdown_table(report.summary_table(closes, "60")).splitlines()
    assert len(md) == 2 + len(SYMBOLS)
    assert md[2].startswith("| AAA |") and "−" in md[2]  # typographic minus in drawdown
