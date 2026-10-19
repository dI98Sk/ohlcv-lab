"""The same job in pandas, Polars and DuckDB: time and peak memory.

Job, on one series of the clean layer: read it, log returns, 30-day rolling
volatility, daily candles, summary statistics. Each engine must produce the
same numbers (checked before anything is reported): a fast wrong answer is
not a result.

Fairness notes, printed with the table:

* time: one warm-up run (file cache, lazy imports), then the best and the
  median of ``repeats`` runs;
* memory: every engine runs in its own subprocess, peak RSS above the
  after-import baseline. ``ru_maxrss`` is in bytes on macOS and KiB on Linux;
* pandas uses one core here, Polars and DuckDB use all of them;
* DuckDB's time includes converting results to pandas, as it is used in sql.py.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ohlcv_lab.storage import DEFAULT_CLEAN_DIR

ENGINES = ["pandas", "polars", "duckdb"]


def run_pandas(clean_dir: Path, symbol: str, interval: str) -> dict:
    from ohlcv_lab.stats import log_returns, rolling_vol, summary
    from ohlcv_lab.storage import load_clean

    df = load_clean(symbol, interval, clean_dir)
    vol = rolling_vol(log_returns(df["close"]), interval, "30D")
    daily = (
        df[~df["is_gap"]]
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    )
    stats = summary(df[["close"]].rename(columns={"close": symbol}), interval)
    return {
        "last_vol": vol.iloc[-1],
        "days": len(daily),
        "kurtosis": stats.loc[symbol, "excess_kurtosis"],
    }


def run_polars(clean_dir: Path, symbol: str, interval: str) -> dict:
    import polars as pl

    from ohlcv_lab import polars_stats as ps

    lf = ps.series(ps.scan_clean(clean_dir), interval, [symbol])
    # One collect_all: Polars can share the common scan between the three plans.
    vol, daily, summary = pl.collect_all(
        [
            ps.rolling_vol(lf, interval, "30d"),
            ps.resample(lf, interval, "1d"),
            ps.summary(lf, interval),
        ]
    )
    return {
        "last_vol": vol["ann_vol"][-1],
        "days": daily.height,
        "kurtosis": summary["excess_kurtosis"][0],
    }


def run_duckdb(clean_dir: Path, symbol: str, interval: str) -> dict:
    from ohlcv_lab.sql import connect, query

    con = connect(clean_dir, symbol)  # only this symbol's files, like the other engines
    vol = query(con, "rolling_vol", interval=interval)
    daily = query(con, "resample", interval=interval, bucket="1 day")
    summary = query(con, "summary", interval=interval)
    return {
        "last_vol": vol["ann_vol"].iloc[-1],
        "days": len(daily),
        "kurtosis": summary["excess_kurtosis"].iloc[0],
    }


RUNNERS = {"pandas": run_pandas, "polars": run_polars, "duckdb": run_duckdb}


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def measure(engine: str, clean_dir: Path, symbol: str, interval: str, repeats: int) -> dict:
    """Run one engine in *this* process; meant to be called in a fresh subprocess."""
    runner = RUNNERS[engine]
    import duckdb  # noqa: F401  - import all engines first so the baseline is comparable
    import pandas  # noqa: F401
    import polars  # noqa: F401

    baseline = peak_rss_mb()
    result = runner(clean_dir, symbol, interval)  # warm-up, also the checked result
    memory = peak_rss_mb() - baseline
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        runner(clean_dir, symbol, interval)
        times.append(time.perf_counter() - t0)
    return {
        "engine": engine,
        "best_s": min(times),
        "median_s": statistics.median(times),
        "peak_mb": memory,
        **{k: float(v) for k, v in result.items()},
    }


def benchmark(clean_dir: Path, symbol: str, interval: str, repeats: int) -> list[dict]:
    rows = []
    for engine in ENGINES:
        cmd = [
            sys.executable, "-m", "ohlcv_lab.bench", "--worker", engine,
            "--clean-dir", str(clean_dir), "--symbol", symbol,
            "--interval", interval, "--repeats", str(repeats),
        ]  # fmt: skip
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        rows.append(json.loads(out.strip().splitlines()[-1]))
    check_agreement(rows)
    return rows


def check_agreement(rows: list[dict]) -> None:
    first = rows[0]
    for row in rows[1:]:
        for key in ("last_vol", "kurtosis"):
            if not np.isclose(row[key], first[key], rtol=1e-9):
                raise AssertionError(f"{row['engine']} {key}={row[key]} != {first[key]}")
        if row["days"] != first["days"]:
            raise AssertionError(f"{row['engine']} days={row['days']} != {first['days']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ohlcv-bench", description="pandas vs Polars vs DuckDB on one clean series."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--worker", choices=ENGINES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        row = measure(args.worker, args.clean_dir, args.symbol, args.interval, args.repeats)
        print(json.dumps(row))
        return 0

    rows = benchmark(args.clean_dir, args.symbol, args.interval, args.repeats)
    print(f"{args.symbol} interval={args.interval}, {args.repeats} runs after a warm-up")
    print(f"{'engine':<8} {'best, s':>8} {'median, s':>10} {'peak, MB':>9}")
    for r in rows:
        print(f"{r['engine']:<8} {r['best_s']:>8.3f} {r['median_s']:>10.3f} {r['peak_mb']:>9.0f}")
    print("results agree: last 30d vol, number of days, kurtosis")
    print("note: pandas runs on one core; Polars and DuckDB on all of them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
