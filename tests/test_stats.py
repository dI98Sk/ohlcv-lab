import numpy as np
import pandas as pd
import pytest

from ohlcv_lab.stats import (
    annualized_vol,
    describe_returns,
    drawdown,
    log_returns,
    max_drawdown,
    periods_per_year,
    rolling_vol,
    simple_returns,
    summary,
)


def hourly(values, start="2024-01-01"):
    index = pd.date_range(start, periods=len(values), freq="h", tz="UTC", name="open_time")
    return pd.Series(values, index=index, dtype="float64")


@pytest.mark.parametrize(("interval", "n"), [("1", 525_600), ("60", 8_760), ("D", 365)])
def test_periods_per_year_is_24_7(interval, n):
    assert periods_per_year(interval) == n


def test_returns_are_not_bridged_over_a_gap():
    close = hourly([100, np.nan, 110, 121])
    # 100 -> 110 took two hours; it must not show up as one hourly return of +10%.
    assert log_returns(close).iloc[:3].isna().all()
    assert simple_returns(close).iloc[:3].isna().all()
    assert simple_returns(close).iloc[3] == pytest.approx(0.1)


def test_log_returns_add_up_and_match_simple():
    close = hourly([100, 105, 98, 120, 90])
    r = log_returns(close)
    assert r.sum() == pytest.approx(np.log(90 / 100))
    pd.testing.assert_series_equal(np.expm1(r), simple_returns(close))


def test_annualized_vol_scales_with_sqrt_time():
    rng = np.random.default_rng(0)
    minute = pd.Series(rng.normal(0, 0.001, 60 * 24 * 365))
    hour = minute.groupby(minute.index // 60).sum()  # log returns add up within the hour
    expected = 0.001 * np.sqrt(525_600)
    assert annualized_vol(minute, "1") == pytest.approx(expected, rel=0.01)
    assert annualized_vol(hour, "60") == pytest.approx(expected, rel=0.03)


def test_rolling_vol_includes_the_current_return():
    r = hourly(np.random.default_rng(1).normal(0, 0.01, 2000))
    vol = rolling_vol(r, "60", window="30D")
    t = r.index[1500]
    window = r.loc[t - pd.Timedelta("30D") + pd.Timedelta("1h") : t]  # (t-30D, t]
    assert vol.loc[t] == pytest.approx(window.std() * np.sqrt(8760))
    assert vol.iloc[:359].isna().all()  # less than half a window of history


def test_drawdown_peak_trough_recovery():
    close = hourly([100, 120, 60, 90, 130, 110])
    assert drawdown(close).tolist() == pytest.approx([0, 0, -0.5, -0.25, 0, 110 / 130 - 1])
    dd = max_drawdown(close)
    assert dd["max_drawdown"] == pytest.approx(-0.5)
    assert (dd["peak"], dd["trough"], dd["recovery"]) == (
        close.index[1],
        close.index[2],
        close.index[4],
    )


def test_max_drawdown_without_recovery():
    assert max_drawdown(hourly([100, 80, 90]))["recovery"] is None


def test_fat_tails_break_the_normal_var():
    rng = np.random.default_rng(2)
    normal = describe_returns(pd.Series(rng.normal(0, 0.01, 200_000)), "60")
    fat = describe_returns(pd.Series(rng.standard_t(3, 200_000) * 0.01), "60")

    assert abs(normal["excess_kurtosis"]) < 0.1
    assert normal["var_1%"] == pytest.approx(normal["var_1%_normal"], rel=0.02)

    assert fat["excess_kurtosis"] > 5
    assert fat["tail_share"] > 10 * fat["tail_share_normal"]
    # The normal model is fitted on the same std, yet it misjudges the 1% tail.
    assert abs(fat["var_1%"] / fat["var_1%_normal"] - 1) > 0.1


def test_summary_handles_late_listing_and_gaps():
    old = hourly(np.linspace(100, 200, 50))
    new = hourly(np.linspace(10, 20, 30), start=old.index[20])
    new.iloc[10] = np.nan  # one gap inside the younger series
    closes = pd.concat({"OLD": old, "NEW": new}, axis=1)
    table = summary(closes, "60")
    assert table.loc["OLD", "n"] == 49
    assert table.loc["NEW", "n"] == 27  # 29 returns minus the two touching the gap
    assert table.loc["OLD", "max_drawdown"] == 0
