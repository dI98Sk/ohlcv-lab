import numpy as np
import pandas as pd
import pytest

from ohlcv_lab.clean import clean, pandas_freq
from ohlcv_lab.storage import compare_formats, load_clean, save_clean

H = 3_600_000
T0 = 1_700_000_000_000 - 1_700_000_000_000 % H  # an hour boundary


def raw_frame(n=300, start=T0, server_time=None, price=100.0):
    """Synthetic raw rows exactly as read_raw returns them: strings + server_time."""
    rng = np.random.default_rng(0)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.r_[price, close[:-1]]
    rows = {
        "open_time": [str(start + i * H) for i in range(n)],
        "open": open_.astype(str),
        "high": (np.maximum(open_, close) * 1.001).astype(str),
        "low": (np.minimum(open_, close) * 0.999).astype(str),
        "close": close.astype(str),
        "volume": ["10"] * n,
        "turnover": ["1000"] * n,
    }
    df = pd.DataFrame(rows)
    df["server_time"] = server_time if server_time is not None else start + (n + 5) * H
    return df


def test_types_index_and_sorting():
    raw = raw_frame().sample(frac=1, random_state=1)  # shuffled, like concatenated pages
    df, report = clean(raw, "TESTUSDT", "60")
    assert str(df.index.dtype) == "datetime64[ms, UTC]"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df["close"].dtype == "float64"
    assert report.missing == 0 and report.unclosed_dropped == 0


def test_unclosed_candle_is_dropped():
    n = 300
    raw = raw_frame(n, server_time=T0 + (n - 1) * H + 60_000)  # last candle opened 1 min ago
    df, report = clean(raw, "TESTUSDT", "60")
    assert report.unclosed_dropped == 1
    assert df.index[-1] == pd.Timestamp(T0 + (n - 2) * H, unit="ms", tz="UTC")


def test_duplicates_exact_and_conflicting():
    raw = raw_frame()
    exact = raw.iloc[[10]]
    changed = raw.iloc[[20]].copy()
    changed["close"] = "999"
    changed["server_time"] += H  # fetched later -> must win
    df, report = clean(pd.concat([raw, exact, changed]), "TESTUSDT", "60")
    assert (report.duplicates, report.conflicting_duplicates) == (1, 1)
    assert len(df) == 300
    assert df["close"].iloc[20] == 999


def test_gaps_are_flagged_not_interpolated():
    raw = raw_frame().drop(index=range(50, 53))  # three missing hours
    df, report = clean(raw, "TESTUSDT", "60")
    assert report.missing == 3
    assert report.gaps == [{"start": str(df.index[50]), "end": str(df.index[52]), "candles": 3}]
    assert df["is_gap"].iloc[50:53].all()
    assert df["close"].iloc[50:53].isna().all()  # no invented prices


def test_late_listing_is_not_a_gap():
    df, report = clean(raw_frame(start=T0 + 1000 * H), "NEWUSDT", "60")
    assert report.missing == 0
    assert df.index[0] == pd.Timestamp(T0 + 1000 * H, unit="ms", tz="UTC")


def test_invariant_violations_are_flagged_and_kept():
    raw = raw_frame()
    raw.loc[5, "low"] = str(float(raw.loc[5, "close"]) * 1.01)  # low above close
    raw.loc[6, "volume"] = "-1"
    df, report = clean(raw, "TESTUSDT", "60")
    assert report.invariant_violations == {
        "low_above_body": 1,
        "high_below_body": 0,
        "negative_volume": 1,
    }
    assert df["bad_ohlc"].sum() == 2 and len(df) == 300


def test_extreme_candle_is_flagged_and_kept():
    raw = raw_frame()
    # Intrabar crash: close barely moves, low is 20% down (like BTC on 2025-10-10).
    raw.loc[250, "low"] = str(float(raw.loc[250, "low"]) * 0.8)
    df, report = clean(raw, "TESTUSDT", "60")
    assert df["is_extreme"].iloc[250]
    assert report.extreme_candles >= 1
    assert df["low"].iloc[250] == pytest.approx(float(raw.loc[250, "low"]))


def test_extreme_flag_does_not_look_ahead():
    raw = raw_frame()
    before, _ = clean(raw, "TESTUSDT", "60")
    raw.loc[290, "low"] = str(float(raw.loc[290, "low"]) * 0.5)  # a crash in the future
    after, _ = clean(raw, "TESTUSDT", "60")
    pd.testing.assert_series_equal(before["is_extreme"].iloc[:290], after["is_extreme"].iloc[:290])


@pytest.mark.parametrize(
    ("interval", "freq"), [("1", "min"), ("60", "h"), ("240", "4h"), ("D", "D")]
)
def test_pandas_freq(interval, freq):
    assert pandas_freq(interval) == freq


def test_parquet_roundtrip_keeps_types_csv_does_not(tmp_path):
    df, _ = clean(raw_frame(), "TESTUSDT", "60")
    save_clean(df, tmp_path, "TESTUSDT", "60")
    assert (tmp_path / "symbol=TESTUSDT" / "interval=60" / "data.parquet").exists()
    pd.testing.assert_frame_equal(load_clean("TESTUSDT", "60", tmp_path), df, check_freq=False)

    cmp = compare_formats(df, tmp_path / "cmp", repeats=1)
    assert cmp.loc["parquet", "dtypes_equal"]
    assert not cmp.loc["csv", "dtypes_equal"]
