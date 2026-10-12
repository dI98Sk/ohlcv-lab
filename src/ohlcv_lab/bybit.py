"""Minimal client for the public Bybit v5 kline endpoint.

API facts the code relies on (checked against the live API):

* ``GET /v5/market/kline`` is public, no API key is needed.
* ``limit`` defaults to 200, the maximum is 1000.
* ``start`` and ``end`` are inclusive and refer to candle *open* times in ms UTC.
* Rows come newest first, every value is a string:
  ``[startTime, open, high, low, close, volume, turnover]``.
* Errors are reported in ``retCode`` with HTTP 200 (e.g. 10001 for a bad symbol),
  and an unknown interval silently returns ``retCode=0`` with an empty list.
* The newest candle may still be open: its close is just the last traded price.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.bybit.com"
KLINE_PATH = "/v5/market/kline"
MAX_LIMIT = 1000

# Fixed-length intervals only: "W" and "M" have no constant length,
# so pages cannot be checked for gaps.
INTERVAL_MS: dict[str, int] = {
    **{str(m): m * 60_000 for m in (1, 3, 5, 15, 30, 60, 120, 240, 360, 720)},
    "D": 86_400_000,
}

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
# 10000: server timeout, 10006: too many visits (rate limit), 10016: server error
RETRYABLE_RET_CODES = {10000, 10006, 10016}

Row = list[str]


class BybitAPIError(RuntimeError):
    """Bybit answered with a non-zero retCode, or retries were exhausted."""

    def __init__(self, message: str, ret_code: int | None = None) -> None:
        super().__init__(message)
        self.ret_code = ret_code


def interval_ms(interval: str) -> int:
    try:
        return INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(
            f"unsupported interval {interval!r}, expected one of {sorted(INTERVAL_MS)}"
        ) from None


@dataclass(frozen=True)
class Window:
    """Inclusive range of candle open times covered by one request."""

    start: int
    end: int


def kline_windows(start: int, end: int, step: int, limit: int = MAX_LIMIT) -> list[Window]:
    """Split ``[start, end)`` into request windows of ``limit`` candles each.

    Windows are aligned to multiples of ``limit * step`` since the epoch, so the
    same candle always falls into the same window no matter when the download
    runs. That makes raw files stable between runs and lets a rerun skip them.
    """
    if end <= start:
        return []
    span = limit * step
    first = start - start % span
    return [Window(w, w + span - step) for w in range(first, end, span)]


def find_page_issues(rows: list[Row], window: Window, step: int) -> list[str]:
    """Check one page: newest-first order, no duplicates, no holes, inside the window.

    Holes are reported, not fixed: an exchange outage is a real gap in the data.
    """
    issues: list[str] = []
    times = [int(r[0]) for r in rows]
    if len(set(times)) != len(times):
        issues.append(f"{len(times) - len(set(times))} duplicate candle(s)")
    if any(t % step for t in times):
        issues.append("candle open time not aligned to the interval")
    outside = [t for t in times if not window.start <= t <= window.end]
    if outside:
        issues.append(f"{len(outside)} candle(s) outside the requested window")
    for newer, older in zip(times, times[1:], strict=False):
        if newer <= older:
            issues.append("rows are not sorted newest first")
            break
        if newer - older != step:
            issues.append(f"hole of {(newer - older) // step - 1} candle(s) before {newer}")
    return issues


def find_joint_issue(newer_page: list[Row], older_page: list[Row], step: int) -> str | None:
    """Check the seam between two adjacent pages (both newest first)."""
    if not newer_page or not older_page:
        return None
    oldest_of_newer = int(newer_page[-1][0])
    newest_of_older = int(older_page[0][0])
    diff = oldest_of_newer - newest_of_older
    if diff == step:
        return None
    if diff <= 0:
        return f"pages overlap at {oldest_of_newer}"
    return f"hole of {diff // step - 1} candle(s) between pages at {oldest_of_newer}"


class BybitClient:
    """Thin wrapper over ``requests.Session`` with pacing and retries."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        category: str = "linear",
        timeout: float = 10.0,
        pause: float = 0.12,
        max_retries: int = 5,
        backoff: float = 1.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url
        self.category = category
        self.timeout = timeout
        self.pause = pause
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "ohlcv-lab/0.1")
        self._sleep = sleep
        self._last_request = 0.0

    def get_kline_page(
        self, symbol: str, interval: str, start: int, end: int, limit: int = MAX_LIMIT
    ) -> dict[str, Any]:
        """One request; returns the full JSON payload untouched."""
        interval_ms(interval)  # fail fast: Bybit returns an empty list for bad intervals
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be in 1..{MAX_LIMIT}, got {limit}")
        params = {
            "category": self.category,
            "symbol": symbol,
            "interval": interval,
            "start": start,
            "end": end,
            "limit": limit,
        }
        return self._get(KLINE_PATH, params)

    def iter_pages(
        self, symbol: str, interval: str, start: int, end: int
    ) -> Iterator[tuple[Window, dict[str, Any]]]:
        """Yield ``(window, payload)`` for every window covering ``[start, end)``."""
        step = interval_ms(interval)
        for window in kline_windows(start, end, step):
            yield window, self.get_kline_page(symbol, interval, window.start, window.end)

    def get_klines(self, symbol: str, interval: str, start: int, end: int) -> list[Row]:
        """All candles with open time in ``[start, end)``, oldest first, still as strings.

        Logs a warning for every hole or duplicate found inside or between pages.
        """
        step = interval_ms(interval)
        out: list[Row] = []
        prev_page: list[Row] = []
        for window, payload in self.iter_pages(symbol, interval, start, end):
            page = payload["result"]["list"]
            for issue in find_page_issues(page, window, step):
                log.warning("%s %s %s: %s", symbol, interval, window.start, issue)
            issue = find_joint_issue(page, prev_page, step)
            if issue:
                log.warning("%s %s: %s", symbol, interval, issue)
            if page:
                prev_page = page
            out.extend(reversed(page))
        return [r for r in out if start <= int(r[0]) < end]

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code in RETRYABLE_HTTP:
                    reason = f"HTTP {resp.status_code}"
                else:
                    resp.raise_for_status()
                    payload = resp.json()
                    ret_code = payload.get("retCode")
                    if ret_code == 0:
                        return payload
                    message = f"retCode={ret_code} retMsg={payload.get('retMsg')!r}"
                    if ret_code not in RETRYABLE_RET_CODES:
                        raise BybitAPIError(f"{message} params={params}", ret_code)
                    reason = message

            if attempt == self.max_retries:
                raise BybitAPIError(f"giving up after {attempt + 1} attempts: {reason}")
            delay = self.backoff * 2**attempt
            log.warning("request failed (%s), retry %d in %.1fs", reason, attempt + 1, delay)
            self._sleep(delay)
        raise AssertionError("unreachable")

    def _throttle(self) -> None:
        wait = self._last_request + self.pause - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()
