"""Every SQL query is checked against the pandas code it mirrors, on the same files."""

import numpy as np
import pandas as pd
import pytest

from ohlcv_lab.clean import gap_ranges
from ohlcv_lab.sql import connect, default_params, main, query, run_sql
from ohlcv_lab.stats import drawdown, log_returns, rolling_vol, summary
from ohlcv_lab.storage import save_clean

HOURS = 24 * 70


def candles(start, n, freq="h", seed=0, gap=None):
    """A clean-layer frame: random walk, complete grid, optional NaN gap (a slice)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="open_time").as_unit("ms")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[100, close[:-1]]
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": rng.uniform(1, 10, n),
            "turnover": rng.uniform(100, 1000, n),
        },
        index=index,
    )
    if gap is not None:
        df.iloc[gap] = np.nan
    df["is_gap"] = df["close"].isna()
    df["bad_ohlc"] = False
    df["is_extreme"] = False
    return df


@pytest.fixture(scope="module")
def layer(tmp_path_factory):
    """Two hourly series (one listed later, one with a gap) and a daily one."""
    root = tmp_path_factory.mktemp("clean")
    frames = {
        "AAAUSDT": candles("2024-01-01", HOURS, seed=1),
        "BBBUSDT": candles("2024-01-11", HOURS - 240, seed=2, gap=slice(500, 503)),
    }
    for symbol, df in frames.items():
        save_clean(df, root, symbol, "60")
    save_clean(candles("2024-01-01", 70, freq="D", seed=3), root, "AAAUSDT", "D")
    closes = pd.concat({s: df["close"] for s, df in frames.items()}, axis=1)
    return root, frames, closes


@pytest.fixture
def con(layer):
    return connect(layer[0])


def by_symbol(df, column, symbol):
    return df[df["symbol"] == symbol].set_index("open_time")[column]


def test_partitions_are_typed_and_clock_is_utc(con):
    types = dict(con.execute("SELECT column_name, column_type FROM (DESCRIBE candles)").fetchall())
    # interval=60 next to interval=D: both partitions stay text, so no query breaks.
    assert types["interval"] == types["symbol"] == "VARCHAR"
    assert types["open_time"] == "TIMESTAMP WITH TIME ZONE"
    assert con.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"
    days = run_sql(
        con,
        "SELECT date_trunc('day', open_time) d, count(*) n FROM candles "
        "WHERE symbol = 'AAAUSDT' AND interval = '60' GROUP BY d",
    )
    assert (days["n"] == 24).all()  # whole UTC days, not days cut at local midnight


def test_returns_view_matches_pandas_and_does_not_bridge_gaps(con, layer):
    _, frames, _ = layer
    df = run_sql(con, "SELECT * FROM returns WHERE interval = '60' ORDER BY symbol, open_time")
    for symbol, frame in frames.items():
        expected = log_returns(frame["close"])
        got = by_symbol(df, "log_return", symbol)
        # ln(a / b) here, ln(a) - ln(b) in pandas: equal in math, ~1e-16 apart in floats,
        # which is a large *relative* error for a return close to zero.
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14, equal_nan=True)
    gap_returns = by_symbol(df, "log_return", "BBBUSDT").iloc[500:504]
    assert gap_returns.isna().all()  # three gap candles and the one right after


def test_rolling_vol_matches_pandas_window_exactly(con, layer):
    _, frames, _ = layer
    df = query(con, "rolling_vol")
    for symbol, frame in frames.items():
        expected = rolling_vol(log_returns(frame["close"]), "60", "30D")
        got = by_symbol(df, "ann_vol", symbol)
        np.testing.assert_allclose(got, expected, rtol=1e-9, equal_nan=True)


def test_drawdowns_view_matches_pandas(con, layer):
    _, frames, _ = layer
    df = run_sql(con, "SELECT * FROM drawdowns WHERE interval = '60' ORDER BY symbol, open_time")
    for symbol, frame in frames.items():
        got = by_symbol(df, "drawdown", symbol)
        np.testing.assert_allclose(got, drawdown(frame["close"]), rtol=1e-12, equal_nan=True)


def test_resample_matches_pandas_and_marks_incomplete_days(con, layer):
    _, frames, _ = layer
    df = query(con, "resample", bucket="1 day")
    frame = frames["BBBUSDT"].dropna()
    expected = frame.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    got = df[df["symbol"] == "BBBUSDT"].set_index("bucket")
    for col in expected:
        np.testing.assert_allclose(got[col], expected[col], rtol=1e-12)
    gap_day = frames["BBBUSDT"].index[500].floor("D")
    assert got.loc[gap_day, "candles"] == 21
    assert not got.loc[gap_day, "complete"]
    assert got["complete"].sum() == len(got) - 1


def test_gaps_match_clean_gap_ranges(con, layer):
    _, frames, _ = layer
    df = query(con, "gaps")
    assert len(df) == 1
    expected = gap_ranges(frames["BBBUSDT"]["is_gap"])[0]
    row = df.iloc[0]
    assert (row["symbol"], row["interval"], row["candles"]) == ("BBBUSDT", "60", 3)
    assert (str(row["start"]), str(row["end"])) == (expected["start"], expected["end"])


def test_summary_matches_pandas(con, layer):
    _, _, closes = layer
    got = query(con, "summary").set_index("symbol")
    expected = summary(closes, "60")
    cols = ["n", "ann_vol", "sharpe", "skew", "excess_kurtosis", "var_1%", "var_5%", "max_drawdown"]
    np.testing.assert_allclose(got[cols].astype(float), expected[cols].astype(float), rtol=1e-9)
    assert (got["trough"] == expected["trough"]).all()


def test_correlation_matches_pandas(con, layer):
    _, _, closes = layer
    got = query(con, "correlation").iloc[0]
    expected = closes.apply(log_returns).corr().loc["AAAUSDT", "BBBUSDT"]
    assert got["corr"] == pytest.approx(expected, rel=1e-12)
    assert got["n"] == HOURS - 240 - 1 - 4  # overlap minus first return minus the gap's four


def test_extra_params_are_not_passed(con):
    # DuckDB rejects unused named parameters; run_sql filters them out.
    assert run_sql(con, "SELECT $interval AS i", default_params()).iloc[0, 0] == "60"


def test_cli_runs_named_and_inline_queries(layer, capsys, tmp_path):
    root = layer[0]
    assert main(["summary", "--clean-dir", str(root), "--out", str(tmp_path / "s.csv")]) == 0
    assert "AAAUSDT" in capsys.readouterr().out
    assert pd.read_csv(tmp_path / "s.csv")["symbol"].tolist() == ["AAAUSDT", "BBBUSDT"]
    assert main(["-e", "SELECT count(*) AS n FROM candles", "--clean-dir", str(root)]) == 0
    with pytest.raises(SystemExit):
        main(["--clean-dir", str(root)])  # neither a name nor -e
