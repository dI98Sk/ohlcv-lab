"""Clean layer on disk: Parquet partitioned by symbol and interval.

Layout (hive style, so DuckDB/Polars/pyarrow recover ``symbol`` and
``interval`` from the path)::

    data/clean/symbol=BTCUSDT/interval=60/data.parquet

Partition values live only in the path, not inside the file.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

DEFAULT_CLEAN_DIR = Path("data/clean")


def partition_path(clean_dir: Path, symbol: str, interval: str) -> Path:
    return Path(clean_dir) / f"symbol={symbol}" / f"interval={interval}" / "data.parquet"


def save_clean(df: pd.DataFrame, clean_dir: Path, symbol: str, interval: str) -> Path:
    path = partition_path(clean_dir, symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=True)
    tmp.replace(path)
    return path


def load_clean(symbol: str, interval: str, clean_dir: Path = DEFAULT_CLEAN_DIR) -> pd.DataFrame:
    """One series with its UTC ``DatetimeIndex`` restored."""
    return pd.read_parquet(partition_path(clean_dir, symbol, interval))


def compare_formats(df: pd.DataFrame, workdir: Path, repeats: int = 5) -> pd.DataFrame:
    """Write ``df`` as CSV and Parquet; compare size, read time and what survives the trip."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path, pq_path = workdir / "sample.csv", workdir / "sample.parquet"
    df.to_csv(csv_path)
    df.to_parquet(pq_path, compression="zstd")

    readers = {
        "csv": (csv_path, lambda: pd.read_csv(csv_path, index_col=0)),
        "parquet": (pq_path, lambda: pd.read_parquet(pq_path)),
    }
    rows = []
    for name, (path, read) in readers.items():
        timings = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            back = read()
            timings.append(time.perf_counter() - t0)
        rows.append(
            {
                "format": name,
                "size_mb": path.stat().st_size / 2**20,
                "read_ms": min(timings) * 1000,
                "index_dtype": str(back.index.dtype),
                "flag_dtype": str(back["is_gap"].dtype),
                "dtypes_equal": back.dtypes.equals(df.dtypes)
                and back.index.dtype == df.index.dtype,
            }
        )
    return pd.DataFrame(rows).set_index("format")
