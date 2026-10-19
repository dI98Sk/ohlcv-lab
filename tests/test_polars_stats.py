"""Every Polars query is checked against the pandas code it mirrors, on the same files."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ohlcv_lab import polars_stats as ps
from ohlcv_lab.stats import log_returns, rolling_vol, summary


@pytest.fixture
def lf(layer):
    return ps.scan_clean(layer[0])


def column(df: pl.DataFrame, symbol: str, name: str) -> np.ndarray:
    return df.filter(pl.col("symbol") == symbol)[name].to_numpy()


def test_scan_types_partitions_as_text_and_keeps_utc(lf):
    schema = lf.collect_schema()
    assert schema["symbol"] == schema["interval"] == pl.String
    assert schema["open_time"] == pl.Datetime("ms", "UTC")


def test_filter_is_pushed_down_to_the_scan(lf):
    plan = ps.series(lf, "60", ["AAAUSDT"]).select("close").explain()
    scan = plan[plan.index("SCAN") :]
    assert "symbol=AAAUSDT/interval=60" in scan and "interval=D" not in scan
    assert "PROJECT" in scan  # only the needed columns are read


def test_gap_is_null_not_nan(lf):
    close = ps.series(lf, "60", ["BBBUSDT"]).collect()["close"]
    assert close.null_count() == 3 and close.is_nan().sum() == 0


def test_returns_and_rolling_vol_match_pandas(lf, layer):
    _, frames, _ = layer
    returns = ps.with_returns(ps.series(lf, "60")).sort("symbol", "open_time").collect()
    vol = ps.rolling_vol(lf, "60", "30d").collect()
    for symbol, frame in frames.items():
        r = log_returns(frame["close"])
        np.testing.assert_allclose(
            column(returns, symbol, "log_return"), r, rtol=0, atol=1e-14, equal_nan=True
        )
        np.testing.assert_allclose(
            column(vol, symbol, "ann_vol"),
            rolling_vol(r, "60", "30D"),
            rtol=1e-9,
            equal_nan=True,
        )


def test_resample_matches_pandas_and_marks_incomplete_days(lf, layer):
    _, frames, _ = layer
    got = ps.resample(lf, "60", "1d").collect().filter(pl.col("symbol") == "BBBUSDT")
    expected = (
        frames["BBBUSDT"]
        .dropna()
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    )
    for col in expected:
        np.testing.assert_allclose(got[col].to_numpy(), expected[col], rtol=1e-12)
    assert got["candles"].min() == 21
    assert got["complete"].sum() == got.height - 1


def test_summary_matches_pandas(lf, layer):
    _, _, closes = layer
    got = ps.summary(lf, "60").collect().to_pandas().set_index("symbol")
    expected = summary(closes, "60")
    cols = ["n", "ann_vol", "sharpe", "skew", "excess_kurtosis", "var_1%", "var_5%", "max_drawdown"]
    np.testing.assert_allclose(got[cols].astype(float), expected[cols].astype(float), rtol=1e-9)
    assert (got["trough"] == expected["trough"]).all()


def test_polars_defaults_differ_from_pandas():
    """Why summary passes bias=False and interpolation='linear'."""
    x = np.random.default_rng(0).standard_t(4, 2000)
    s = pl.Series(x)
    assert s.kurtosis() != pytest.approx(pd.Series(x).kurt(), rel=1e-6)
    assert s.kurtosis(bias=False) == pytest.approx(pd.Series(x).kurt(), rel=1e-12)
    assert s.quantile(0.01) != pytest.approx(pd.Series(x).quantile(0.01), rel=1e-9)


def test_nan_poisons_polars_but_not_pandas():
    values = [1.0, float("nan"), 3.0]
    assert np.isnan(pl.Series(values).std())
    assert pd.Series(values).std() == pytest.approx(np.std([1, 3], ddof=1))
    assert pl.Series(values).fill_nan(None).std() == pd.Series(values).std()
