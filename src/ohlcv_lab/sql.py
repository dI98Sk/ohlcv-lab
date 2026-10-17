"""SQL layer: DuckDB queries straight over the clean Parquet files.

:func:`connect` opens an in-memory DuckDB with three views over
``data/clean/symbol=*/interval=*/data.parquet``:

* ``candles``   - the clean layer as is, plus ``symbol`` and ``interval`` from the path;
* ``returns``   - log return of every candle (NULL across a gap);
* ``drawdowns`` - distance of every close from its running peak.

Two connection settings are not optional:

* ``TimeZone = 'UTC'``. DuckDB shows and truncates ``TIMESTAMPTZ`` in the session
  zone, which defaults to the machine's. On a Moscow laptop ``date_trunc('day')``
  cuts days at 21:00 UTC and daily candles stop matching the exchange's.
* ``hive_types``. Partition values are typed from folder names, so ``interval=60``
  becomes BIGINT, and the first ``interval=D`` folder would change the type under
  every query. Both partitions are VARCHAR, always.

Named queries live in ``queries/*.sql`` with ``$name`` parameters; each query
gets only the parameters it mentions, because DuckDB rejects extra ones.
"""

from __future__ import annotations

import argparse
import re
from importlib.resources import files
from pathlib import Path

import duckdb
import pandas as pd

from ohlcv_lab.bybit import interval_ms
from ohlcv_lab.stats import periods_per_year
from ohlcv_lab.storage import DEFAULT_CLEAN_DIR

QUERY_DIR = files("ohlcv_lab") / "queries"
PARAM = re.compile(r"\$(\w+)")

VIEWS = """
CREATE VIEW returns AS
SELECT
    symbol,
    interval,
    open_time,
    close,
    -- LAG takes the previous *row*; it is the previous candle only because the
    -- clean layer has a complete grid with NULL closes at the gaps.
    ln(close / lag(close) OVER (PARTITION BY symbol, interval ORDER BY open_time)) AS log_return
FROM candles;

CREATE VIEW drawdowns AS
SELECT
    symbol,
    interval,
    open_time,
    close,
    -- max() skips NULLs: a gap neither resets nor raises the peak.
    close / max(close) OVER (
        PARTITION BY symbol, interval ORDER BY open_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) - 1 AS drawdown
FROM candles;
"""


def connect(clean_dir: Path = DEFAULT_CLEAN_DIR) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the ``candles``, ``returns`` and ``drawdowns`` views."""
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    pattern = (Path(clean_dir) / "*" / "*" / "data.parquet").as_posix().replace("'", "''")
    con.execute(f"""
        CREATE VIEW candles AS
        SELECT * FROM read_parquet(
            '{pattern}',
            hive_partitioning = true,
            hive_types = {{'symbol': VARCHAR, 'interval': VARCHAR}}
        )
    """)
    con.execute(VIEWS)
    return con


def available_queries() -> list[str]:
    return sorted(
        p.name.removesuffix(".sql") for p in QUERY_DIR.iterdir() if p.name.endswith(".sql")
    )


def load_query(name: str) -> str:
    return (QUERY_DIR / f"{name}.sql").read_text()


def default_params(interval: str = "60", window: str = "30 days", bucket: str = "1 day") -> dict:
    """Every parameter a named query may use, derived from three user-facing ones."""
    step = pd.Timedelta(milliseconds=interval_ms(interval))
    return {
        "interval": interval,
        "ppy": periods_per_year(interval),
        "window": window,
        "min_periods": int(pd.Timedelta(window) / step / 2),  # as in stats.rolling_vol
        "bucket": bucket,
        "candles_per_bucket": int(pd.Timedelta(bucket) / step),
    }


def run_sql(con: duckdb.DuckDBPyConnection, sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute ``sql`` with the subset of ``params`` it actually references."""
    used = set(PARAM.findall(sql))
    return con.execute(sql, {k: v for k, v in (params or {}).items() if k in used}).df()


def query(con: duckdb.DuckDBPyConnection, name: str, **options: str) -> pd.DataFrame:
    """Run a named query; ``options`` are the arguments of :func:`default_params`."""
    return run_sql(con, load_query(name), default_params(**options))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ohlcv-sql",
        description="Run a named query (or any SQL with -e) over the clean layer.",
        epilog="views: candles, returns, drawdowns",
    )
    parser.add_argument("name", nargs="?", choices=available_queries(), help="named query")
    parser.add_argument("-e", "--execute", metavar="SQL", help="run this SQL instead")
    parser.add_argument("--interval", default="60")
    parser.add_argument("--window", default="30 days", help="rolling_vol window")
    parser.add_argument("--bucket", default="1 day", help="resample bucket width")
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--out", type=Path, help="also save the result as CSV")
    args = parser.parse_args(argv)
    if (args.name is None) == (args.execute is None):
        parser.error("give either a query name or -e SQL")

    con = connect(args.clean_dir)
    params = default_params(args.interval, args.window, args.bucket)
    sql = args.execute if args.execute is not None else load_query(args.name)
    result = run_sql(con, sql, params)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(result)
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
