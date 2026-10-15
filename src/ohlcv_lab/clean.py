"""Raw layer -> clean OHLCV frame with a full UTC time grid and quality flags.

The cleaning never invents prices and never deletes real market moves:

* types are parsed, the index is a ``DatetimeIndex`` in UTC, sorted and unique;
* the candle that was still open at fetch time is dropped (its close is just
  the last trade, not a close);
* the series is reindexed to a complete grid; missing candles stay NaN and get
  ``is_gap=True`` instead of being interpolated;
* OHLC invariant violations and extreme candles are flagged, not removed.

The grid starts at the first candle that exists, so a coin listed later simply
has a shorter history; that is not counted as missing data.

All rolling statistics look only at the past (``closed="left"``): a flag on a
candle must not depend on candles that come after it.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ohlcv_lab.bybit import interval_ms
from ohlcv_lab.fetch import DEFAULT_RAW_DIR
from ohlcv_lab.storage import DEFAULT_CLEAN_DIR, save_clean

log = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close"]
VALUE_COLS = [*PRICE_COLS, "volume", "turnover"]
RAW_COLS = ["open_time", *VALUE_COLS]

EXTREME_RETURN_SIGMA = 8.0
EXTREME_RANGE_RATIO = 10.0
ROLLING_WINDOW = "30D"


def pandas_freq(interval: str) -> str:
    """Bybit interval -> pandas frequency alias ('1' -> 'min', '60' -> 'h', 'D' -> 'D')."""
    if interval == "D":
        return "D"
    minutes = interval_ms(interval) // 60_000
    count, unit = (minutes // 60, "h") if minutes % 60 == 0 else (minutes, "min")
    return unit if count == 1 else f"{count}{unit}"


@dataclass
class QualityReport:
    symbol: str
    interval: str
    raw_rows: int = 0
    unclosed_dropped: int = 0
    duplicates: int = 0
    conflicting_duplicates: int = 0
    rows: int = 0
    first: str | None = None
    last: str | None = None
    missing: int = 0
    gaps: list[dict] = field(default_factory=list)
    invariant_violations: dict[str, int] = field(default_factory=dict)
    extreme_candles: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def read_raw(raw_dir: Path, symbol: str, interval: str, category: str = "linear") -> pd.DataFrame:
    """All rows of all raw files for one series, values still strings.

    Adds ``server_time`` (ms) of the response each row came from: it is needed to
    tell a closed candle from an open one.
    """
    series = Path(raw_dir) / "bybit" / category / symbol / interval
    frames = []
    for path in sorted(series.glob("*.json")):
        response = json.loads(path.read_text())["response"]
        rows = response["result"]["list"]
        if rows:
            frame = pd.DataFrame(rows, columns=RAW_COLS)
            frame["server_time"] = response["time"]
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no raw data for {symbol} {interval} in {series}")
    return pd.concat(frames, ignore_index=True)


def parse_types(raw: pd.DataFrame) -> pd.DataFrame:
    """Strings -> float64 values and an ``open_time`` column of UTC timestamps (ms)."""
    df = raw[VALUE_COLS].astype("float64")
    open_ms = raw["open_time"].astype("int64")
    df.insert(0, "open_time", pd.to_datetime(open_ms, unit="ms", utc=True).dt.as_unit("ms"))
    df["server_time"] = pd.to_datetime(raw["server_time"], unit="ms", utc=True).dt.as_unit("ms")
    return df


def clean(raw: pd.DataFrame, symbol: str, interval: str) -> tuple[pd.DataFrame, QualityReport]:
    """Full cleaning pipeline for one series. Returns the frame and its quality report."""
    report = QualityReport(symbol=symbol, interval=interval, raw_rows=len(raw))
    step = pd.Timedelta(milliseconds=interval_ms(interval))
    df = parse_types(raw)

    # 1. A candle is closed only if its end is not later than the server clock.
    unclosed = df["open_time"] + step > df["server_time"]
    report.unclosed_dropped = int(unclosed.sum())
    df = df[~unclosed]

    # 2. Duplicates: identical rows are harmless, differing ones mean the data
    #    changed between fetches; keep the most recently fetched version.
    report.duplicates = int(df.duplicated(subset=["open_time", *VALUE_COLS]).sum())
    df = df.drop_duplicates(subset=["open_time", *VALUE_COLS])
    report.conflicting_duplicates = int(df["open_time"].duplicated().sum())
    df = df.sort_values(["open_time", "server_time"]).drop_duplicates("open_time", keep="last")
    df = df.drop(columns="server_time").set_index("open_time").sort_index()

    # 3. Complete grid. Missing candles stay NaN: no silent interpolation.
    grid = pd.date_range(df.index[0], df.index[-1], freq=pandas_freq(interval), name="open_time")
    df = df.reindex(grid)
    df["is_gap"] = df["close"].isna()
    report.rows = len(df)
    report.first, report.last = str(df.index[0]), str(df.index[-1])
    report.missing = int(df["is_gap"].sum())
    report.gaps = gap_ranges(df["is_gap"])

    # 4. Invariants. NaN comparisons are False, so gaps are never violations.
    checks = invariant_checks(df)
    report.invariant_violations = {name: int(bad.sum()) for name, bad in checks.items()}
    df["bad_ohlc"] = pd.concat(checks, axis=1).any(axis=1)

    # 5. Extreme candles: flagged, kept.
    df["is_extreme"] = extreme_flags(df)
    report.extreme_candles = int(df["is_extreme"].sum())
    return df, report


def invariant_checks(df: pd.DataFrame) -> dict[str, pd.Series]:
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    return {
        "low_above_body": df["low"] > body_low,
        "high_below_body": df["high"] < body_high,
        "negative_volume": df["volume"] < 0,
    }


def extreme_flags(
    df: pd.DataFrame,
    sigma: float = EXTREME_RETURN_SIGMA,
    range_ratio: float = EXTREME_RANGE_RATIO,
    window: str = ROLLING_WINDOW,
) -> pd.Series:
    """True where the move is extreme compared with the *preceding* window.

    Two signals, because a crash can show up only intrabar: on 2025-10-10 21:00
    BTC closed -0.9% but traded a 13% high-low range.
    """
    log_ret = np.log(df["close"]).diff()
    log_range = np.log(df["high"] / df["low"])
    past = {"window": window, "closed": "left", "min_periods": 100}
    big_move = log_ret.abs() > sigma * log_ret.rolling(**past).std()
    wide_bar = log_range > range_ratio * log_range.rolling(**past).median()
    return big_move | wide_bar


def gap_ranges(is_gap: pd.Series) -> list[dict]:
    """Runs of consecutive missing candles as ``{"start", "end", "candles"}``."""
    run_id = (is_gap != is_gap.shift()).cumsum()
    gaps = []
    for _, run in is_gap[is_gap].groupby(run_id[is_gap]):
        gaps.append({"start": str(run.index[0]), "end": str(run.index[-1]), "candles": len(run)})
    return gaps


def discover_series(raw_dir: Path, category: str = "linear") -> list[tuple[str, str]]:
    """(symbol, interval) pairs present in the raw layer."""
    base = Path(raw_dir) / "bybit" / category
    return sorted((p.parent.name, p.name) for p in base.glob("*/*") if p.is_dir())


def run(raw_dir: Path, clean_dir: Path) -> list[QualityReport]:
    reports = []
    for symbol, interval in discover_series(raw_dir):
        df, report = clean(read_raw(raw_dir, symbol, interval), symbol, interval)
        path = save_clean(df, clean_dir, symbol, interval)
        log.info(
            "%s %s: %d rows %s .. %s | unclosed %d, dup %d/%d, missing %d in %d gap(s), "
            "bad ohlc %d, extreme %d -> %s",
            symbol,
            interval,
            report.rows,
            report.first,
            report.last,
            report.unclosed_dropped,
            report.duplicates,
            report.conflicting_duplicates,
            report.missing,
            len(report.gaps),
            sum(report.invariant_violations.values()),
            report.extreme_candles,
            path,
        )
        reports.append(report)
    out = Path(clean_dir) / "quality_report.json"
    out.write_text(json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False))
    log.info("quality report -> %s", out)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ohlcv-clean", description="Build the clean Parquet layer from data/raw."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    run(args.raw_dir, args.clean_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
