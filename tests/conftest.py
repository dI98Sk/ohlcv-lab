"""Synthetic clean layer shared by the SQL and Polars tests."""

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture(scope="session")
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
