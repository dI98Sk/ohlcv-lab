"""Download Bybit klines into the raw layer: ``data/raw/``.

The raw layer is append-only. Every request window is stored as one JSON file
holding the untouched API response plus the request that produced it, and
everything downstream must be reproducible from these files alone.

Layout::

    data/raw/bybit/<category>/<symbol>/<interval>/<window start>.json

A window whose last candle has not closed yet (by the server clock) is stored
as ``<window start>.partial.json``. It is the only kind of file that gets
replaced: the next run fetches the window again and, once it is complete,
writes the final file and removes the partial one.

Usage::

    ohlcv-fetch --symbols BTCUSDT ETHUSDT --interval 60 --days 1095
    ohlcv-fetch --symbols BTCUSDT --interval 1 --start 2025-10-01 --end 2026-10-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ohlcv_lab.bybit import (
    BASE_URL,
    INTERVAL_MS,
    KLINE_PATH,
    MAX_LIMIT,
    BybitClient,
    Row,
    Window,
    find_joint_issue,
    find_page_issues,
    interval_ms,
    kline_windows,
)

log = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")


@dataclass(frozen=True)
class FetchConfig:
    symbols: tuple[str, ...]
    interval: str
    start: datetime
    end: datetime
    category: str = "linear"
    raw_dir: Path = DEFAULT_RAW_DIR
    pause: float = 0.12
    max_retries: int = 5

    def __post_init__(self) -> None:
        interval_ms(self.interval)
        if not self.symbols:
            raise ValueError("at least one symbol is required")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        if self.start >= self.end:
            raise ValueError(f"start {self.start} must be before end {self.end}")

    @property
    def start_ms(self) -> int:
        return to_ms(self.start)

    @property
    def end_ms(self) -> int:
        return to_ms(self.end)


@dataclass
class FetchStats:
    fetched: int = 0
    skipped: int = 0
    partial: int = 0
    rows: int = 0
    issues: list[str] = field(default_factory=list)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y%m%dT%H%MZ")


def series_dir(raw_dir: Path, category: str, symbol: str, interval: str) -> Path:
    return raw_dir / "bybit" / category / symbol / interval


def write_json_atomic(path: Path, data: Any) -> None:
    """Write to a temp file, then rename: an interrupted run never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    os.replace(tmp, path)


def fetch_series(client: BybitClient, cfg: FetchConfig, symbol: str) -> FetchStats:
    step = interval_ms(cfg.interval)
    out_dir = series_dir(cfg.raw_dir, cfg.category, symbol, cfg.interval)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = FetchStats()
    prev_page: list[Row] = []

    for window in kline_windows(cfg.start_ms, cfg.end_ms, step):
        name = ms_to_str(window.start)
        final_path = out_dir / f"{name}.json"
        partial_path = out_dir / f"{name}.partial.json"

        if final_path.exists():
            page = json.loads(final_path.read_text())["response"]["result"]["list"]
            stats.skipped += 1
        else:
            payload = client.get_kline_page(symbol, cfg.interval, window.start, window.end)
            page = payload["result"]["list"]
            complete = window.end + step <= payload["time"]
            record = make_record(cfg, symbol, window, payload, complete)
            if complete:
                write_json_atomic(final_path, record)
                partial_path.unlink(missing_ok=True)
                stats.fetched += 1
            else:
                write_json_atomic(partial_path, record)
                stats.partial += 1

        for issue in find_page_issues(page, window, step):
            stats.issues.append(f"{name}: {issue}")
        joint = find_joint_issue(page, prev_page, step)
        if joint:
            stats.issues.append(f"{name}: {joint}")
        if page:
            prev_page = page
        stats.rows += len(page)

    return stats


def make_record(
    cfg: FetchConfig, symbol: str, window: Window, payload: dict[str, Any], complete: bool
) -> dict[str, Any]:
    return {
        "request": {
            "url": BASE_URL + KLINE_PATH,
            "params": {
                "category": cfg.category,
                "symbol": symbol,
                "interval": cfg.interval,
                "start": window.start,
                "end": window.end,
                "limit": MAX_LIMIT,
            },
        },
        "fetched_at": datetime.now(UTC).isoformat(),
        "complete": complete,
        "response": payload,
    }


def run(cfg: FetchConfig, client: BybitClient | None = None) -> dict[str, FetchStats]:
    client = client or BybitClient(
        category=cfg.category, pause=cfg.pause, max_retries=cfg.max_retries
    )
    log.info("config: %s", {k: str(v) for k, v in asdict(cfg).items()})
    results = {}
    for symbol in cfg.symbols:
        stats = fetch_series(client, cfg, symbol)
        results[symbol] = stats
        log.info(
            "%s %s: %d rows, %d fetched, %d skipped, %d partial, %d issue(s)",
            symbol,
            cfg.interval,
            stats.rows,
            stats.fetched,
            stats.skipped,
            stats.partial,
            len(stats.issues),
        )
        for issue in stats.issues:
            log.warning("%s %s %s", symbol, cfg.interval, issue)
    return results


def parse_datetime(value: str) -> datetime:
    """Accept ``2025-10-01`` or ``2025-10-01T12:00``; naive values are taken as UTC."""
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ohlcv-fetch", description="Download Bybit klines into the raw layer."
    )
    p.add_argument("--symbols", nargs="+", required=True, help="e.g. BTCUSDT ETHUSDT")
    p.add_argument(
        "--interval",
        default="60",
        choices=list(INTERVAL_MS),
        help="candle interval: minutes or D (default: 60)",
    )
    period = p.add_mutually_exclusive_group(required=True)
    period.add_argument("--start", type=parse_datetime, help="UTC date/time, e.g. 2023-10-01")
    period.add_argument("--days", type=int, help="how many days back from --end")
    p.add_argument("--end", type=parse_datetime, help="UTC date/time (default: now)")
    p.add_argument("--category", default="linear")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--pause", type=float, default=0.12, help="seconds between requests")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> FetchConfig:
    end = args.end or datetime.now(UTC)
    start = args.start or end - timedelta(days=args.days)
    return FetchConfig(
        symbols=tuple(s.upper() for s in args.symbols),
        interval=args.interval,
        start=start,
        end=end,
        category=args.category,
        raw_dir=args.raw_dir,
        pause=args.pause,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    run(config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
