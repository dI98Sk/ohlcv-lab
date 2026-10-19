"""The statistics of ``stats.py`` once more, as lazy Polars queries.

Nothing is read until ``.collect()``: every function takes and returns a
``LazyFrame``, so Polars sees the whole query and can push the symbol/interval
filter down to the file scan (only matching partitions are opened) and read
only the columns the query uses. ``lf.explain()`` shows that plan.

Three defaults differ from pandas and would make the numbers drift apart:

* ``skew``/``kurtosis`` default to ``bias=True`` (population formulas);
  pandas uses the bias-corrected sample ones, hence ``bias=False``;
* ``quantile`` defaults to ``"nearest"``; pandas interpolates linearly;
* NaN is a number to Polars, only null is "missing": a NaN in a column turns
  its ``std()`` into NaN, where pandas would skip it. The clean layer stores
  gaps as null (pyarrow writes pandas' NaN as null), so they are skipped here.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ohlcv_lab.bybit import interval_ms
from ohlcv_lab.stats import periods_per_year
from ohlcv_lab.storage import DEFAULT_CLEAN_DIR

SERIES = ["symbol", "interval"]


def scan_clean(clean_dir: Path = DEFAULT_CLEAN_DIR) -> pl.LazyFrame:
    """Lazy view of the whole clean layer; partitions typed as text, like in ``sql.connect``."""
    return pl.scan_parquet(
        Path(clean_dir) / "**" / "*.parquet",
        hive_partitioning=True,
        hive_schema={"symbol": pl.String, "interval": pl.String},
    )


def candles(duration: str, interval: str) -> int:
    """How many candles of ``interval`` fit in a Polars duration: ('30d', '60') -> 720.

    Parsed by Polars itself: its duration strings ('30d', '4h') are not pandas'
    ('30D'), and pandas is dropping the lowercase 'd'.
    """
    start = datetime(2000, 1, 1)
    end = pl.select(pl.lit(start).dt.offset_by(duration)).item()
    return int((end - start) / timedelta(milliseconds=interval_ms(interval)))


def series(lf: pl.LazyFrame, interval: str, symbols: list[str] | None = None) -> pl.LazyFrame:
    """Filter to one interval (and optionally some symbols); pushed down to the scan."""
    lf = lf.filter(pl.col("interval") == interval)
    return lf if symbols is None else lf.filter(pl.col("symbol").is_in(symbols))


def with_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add ``log_return``; null across a gap because the gap's close is null."""
    log_return = pl.col("close").log().diff().over(SERIES, order_by="open_time")
    return lf.with_columns(log_return.alias("log_return"))


def with_drawdown(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add ``drawdown``: close against its running peak (``cum_max`` skips nulls)."""
    peak = pl.col("close").cum_max().over(SERIES, order_by="open_time")
    return lf.with_columns((pl.col("close") / peak - 1).alias("drawdown"))


def rolling_vol(lf: pl.LazyFrame, interval: str, window: str = "30d") -> pl.LazyFrame:
    """Annualised volatility over a trailing time window; twin of ``stats.rolling_vol``.

    ``closed="right"`` gives the window (t - window, t], the same candles as
    pandas' ``rolling("30D")``. ``min_samples`` counts non-null returns.
    """
    vol = pl.col("log_return").rolling_std_by(
        "open_time", window, min_samples=candles(window, interval) // 2, closed="right"
    )
    return (
        with_returns(series(lf, interval))
        .with_columns((vol.over("symbol") * np.sqrt(periods_per_year(interval))).alias("ann_vol"))
        .select("symbol", "open_time", "ann_vol")
        .sort("symbol", "open_time")
    )


def resample(lf: pl.LazyFrame, interval: str, every: str = "1d") -> pl.LazyFrame:
    """Wider candles (e.g. 1h -> 1d) aligned to UTC; ``complete`` is false for partial buckets."""
    per_bucket = candles(every, interval)
    return (
        series(lf, interval)
        .filter(~pl.col("is_gap"))
        .sort("symbol", "open_time")
        .group_by_dynamic("open_time", every=every, group_by="symbol")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("turnover").sum(),
            pl.len().alias("candles"),
        )
        .with_columns((pl.col("candles") == per_bucket).alias("complete"))
        .sort("symbol", "open_time")
    )


def summary(lf: pl.LazyFrame, interval: str) -> pl.LazyFrame:
    """Per-symbol statistics; twin of ``stats.summary`` (subset) and ``queries/summary.sql``."""
    r = pl.col("log_return")
    ppy = periods_per_year(interval)
    return (
        with_drawdown(with_returns(series(lf, interval)))
        .group_by("symbol")
        .agg(
            r.count().alias("n"),
            (r.std() * np.sqrt(ppy)).alias("ann_vol"),
            (r.mean() / r.std() * np.sqrt(ppy)).alias("sharpe"),
            r.skew(bias=False).alias("skew"),
            r.kurtosis(bias=False).alias("excess_kurtosis"),
            r.quantile(0.01, interpolation="linear").alias("var_1%"),
            r.quantile(0.05, interpolation="linear").alias("var_5%"),
            pl.col("drawdown").min().alias("max_drawdown"),
            pl.col("open_time").get(pl.col("drawdown").arg_min()).alias("trough"),
        )
        .sort("symbol")
    )
