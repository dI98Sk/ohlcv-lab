"""Returns and descriptive statistics on top of the clean layer.

Conventions:

* log returns for anything aggregated over time (they add up), simple returns
  where money is meant (a -50% drawdown is a simple return);
* a return is defined only between two adjacent candles: across a gap it is
  NaN, never one big jump attributed to a single period;
* crypto trades 24/7, so a year is 365 days of candles (8760 for 1h), not 252
  trading days;
* hourly returns are fat-tailed (excess kurtosis 10-28 on the 2023-2026 data),
  so VaR is historical; the normal-model VaR is reported next to it only to
  show how far off it is.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from ohlcv_lab.bybit import interval_ms
from ohlcv_lab.storage import DEFAULT_CLEAN_DIR, load_clean

log = logging.getLogger(__name__)

YEAR_MS = 365 * 24 * 3_600_000
TAIL_SIGMA = 4.0


def periods_per_year(interval: str) -> float:
    """Number of candles in a 24/7 year: '60' -> 8760, '1' -> 525600, 'D' -> 365."""
    return YEAR_MS / interval_ms(interval)


def log_returns(close: pd.Series) -> pd.Series:
    """ln(P_t / P_{t-1}); NaN at a gap and at the first candle after it."""
    return np.log(close).diff()


def simple_returns(close: pd.Series) -> pd.Series:
    """P_t / P_{t-1} - 1, written out so no version of pandas can fill a gap."""
    return close / close.shift() - 1


def annualized_vol(returns: pd.Series, interval: str) -> float:
    return float(returns.std() * np.sqrt(periods_per_year(interval)))


def rolling_vol(returns: pd.Series, interval: str, window: str = "30D") -> pd.Series:
    """Annualised volatility known at the close of each candle.

    The window includes the current return: at the close of t it is already
    known. (Contrast with ``clean.extreme_flags``, which judges candle t and
    therefore must keep it out of its own yardstick.)
    """
    min_periods = int(pd.Timedelta(window) / pd.Timedelta(milliseconds=interval_ms(interval)) / 2)
    return returns.rolling(window, min_periods=min_periods).std() * np.sqrt(
        periods_per_year(interval)
    )


def drawdown(close: pd.Series) -> pd.Series:
    """Simple-return distance from the running peak: 0 at a new high, -0.5 at half of it."""
    return close / close.cummax() - 1


def max_drawdown(close: pd.Series) -> dict:
    """Depth of the worst drawdown with its peak, trough and recovery times."""
    dd = drawdown(close)
    trough = dd.idxmin()
    peak = close.loc[:trough].idxmax()
    after = close.loc[trough:]
    recovered = after[after >= close.loc[peak]]
    return {
        "max_drawdown": float(dd.loc[trough]),
        "peak": peak,
        "trough": trough,
        "recovery": recovered.index[0] if len(recovered) else None,
    }


def describe_returns(returns: pd.Series, interval: str) -> dict:
    """Moments, tails and annualised figures of one return series (NaNs skipped)."""
    r = returns.dropna()
    ppy = periods_per_year(interval)
    mean, std = r.mean(), r.std()
    var_hist = {q: float(r.quantile(q)) for q in (0.01, 0.05)}
    var_norm = {q: float(mean + std * st.norm.ppf(q)) for q in (0.01, 0.05)}
    return {
        "n": len(r),
        "mean": float(mean),
        "std": float(std),
        "ann_return": float(mean * ppy),
        "ann_vol": float(std * np.sqrt(ppy)),
        "sharpe": float(mean / std * np.sqrt(ppy)),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurt()),
        "min": float(r.min()),
        "max": float(r.max()),
        "var_1%": var_hist[0.01],
        "var_1%_normal": var_norm[0.01],
        "var_5%": var_hist[0.05],
        "var_5%_normal": var_norm[0.05],
        "tail_share": float(((r - mean).abs() > TAIL_SIGMA * std).mean()),
        "tail_share_normal": float(2 * st.norm.sf(TAIL_SIGMA)),
    }


def available_symbols(interval: str, clean_dir: Path = DEFAULT_CLEAN_DIR) -> list[str]:
    paths = Path(clean_dir).glob(f"symbol=*/interval={interval}/data.parquet")
    return sorted(p.parent.parent.name.removeprefix("symbol=") for p in paths)


def load_closes(
    symbols: list[str], interval: str, clean_dir: Path = DEFAULT_CLEAN_DIR
) -> pd.DataFrame:
    """Close prices side by side on one time index (outer join: NaN where a coin has no candle)."""
    return pd.concat({s: load_clean(s, interval, clean_dir)["close"] for s in symbols}, axis=1)


def summary(closes: pd.DataFrame, interval: str) -> pd.DataFrame:
    """One row per symbol: return statistics plus the worst drawdown."""
    # No dropna() before log_returns: it would glue the candles around a gap together.
    rows = {
        symbol: {**describe_returns(log_returns(close), interval), **max_drawdown(close)}
        for symbol, close in closes.items()
    }
    return pd.DataFrame(rows).T


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ohlcv-stats", description="Return statistics for every symbol in the clean layer."
    )
    parser.add_argument("--interval", default="60")
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default: <clean-dir>/stats_<interval>.csv)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    symbols = available_symbols(args.interval, args.clean_dir)
    if not symbols:
        parser.error(f"no clean data for interval {args.interval} in {args.clean_dir}")
    closes = load_closes(symbols, args.interval, args.clean_dir)
    table = summary(closes, args.interval)
    corr = closes.apply(log_returns).corr()

    with pd.option_context(
        "display.width", 200, "display.max_columns", None, "display.float_format", "{:.4f}".format
    ):
        cols = [
            "ann_vol",
            "sharpe",
            "skew",
            "excess_kurtosis",
            "var_1%",
            "var_1%_normal",
            "max_drawdown",
        ]
        print(table[cols].astype(float))
        print("\ncorrelation of log returns:")
        print(corr)

    out = args.out or Path(args.clean_dir) / f"stats_{args.interval}.csv"
    table.to_csv(out)
    log.info("statistics -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
