import pytest

from ohlcv_lab.bench import benchmark, check_agreement


def test_all_engines_agree_on_the_synthetic_layer(layer):
    rows = benchmark(layer[0], "BBBUSDT", "60", repeats=1)  # a series with a gap
    assert [r["engine"] for r in rows] == ["pandas", "polars", "duckdb"]
    assert all(r["best_s"] > 0 and r["days"] == rows[0]["days"] for r in rows)


def test_disagreement_is_an_error():
    good = {"engine": "pandas", "last_vol": 0.5, "kurtosis": 10.0, "days": 30}
    with pytest.raises(AssertionError, match="polars last_vol"):
        check_agreement([good, {**good, "engine": "polars", "last_vol": 0.51}])
