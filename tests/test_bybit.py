import json
from datetime import UTC, datetime

import pytest
import requests

from ohlcv_lab.bybit import (
    BybitAPIError,
    BybitClient,
    Window,
    find_joint_issue,
    find_page_issues,
    kline_windows,
)
from ohlcv_lab.fetch import FetchConfig, fetch_series

H = 3_600_000  # one hour in ms


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Replays scripted responses (or exceptions) and records request params."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok(rows, server_time=10**13):
    return FakeResponse(
        {"retCode": 0, "retMsg": "OK", "result": {"list": rows}, "time": server_time}
    )


def rows_newest_first(start, n, step=H):
    """n synthetic candles starting at ``start``, ordered like the API returns them."""
    return [[str(start + i * step), "1", "2", "0.5", "1.5", "10", "15"] for i in reversed(range(n))]


def make_client(responses):
    session = FakeSession(responses)
    sleeps = []
    client = BybitClient(session=session, pause=0, backoff=1.0, sleep=sleeps.append)
    return client, session, sleeps


def test_windows_are_aligned_and_cover_range():
    windows = kline_windows(start=5 * H, end=2500 * H, step=H, limit=1000)
    assert windows == [Window(0, 999 * H), Window(1000 * H, 1999 * H), Window(2000 * H, 2999 * H)]
    # Same candle -> same window, whatever the requested start is
    assert kline_windows(start=700 * H, end=701 * H, step=H)[0] == windows[0]


def test_retries_rate_limit_then_succeeds_with_exponential_backoff():
    rate_limited = FakeResponse({"retCode": 10006, "retMsg": "Too many visits!"})
    client, session, sleeps = make_client(
        [rate_limited, FakeResponse(status_code=502), requests.ConnectionError("reset"), ok([])]
    )
    payload = client.get_kline_page("BTCUSDT", "60", 0, 999 * H)
    assert payload["retCode"] == 0
    assert len(session.calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_bad_symbol_fails_fast_without_retry():
    bad = FakeResponse({"retCode": 10001, "retMsg": "params error: Symbol Is Invalid"})
    client, session, _ = make_client([bad])
    with pytest.raises(BybitAPIError) as exc:
        client.get_kline_page("NOPEUSDT", "60", 0, 999 * H)
    assert exc.value.ret_code == 10001
    assert len(session.calls) == 1


def test_gives_up_after_max_retries():
    client, session, _ = make_client([FakeResponse(status_code=503)] * 10)
    client.max_retries = 2
    with pytest.raises(BybitAPIError, match="giving up after 3 attempts"):
        client.get_kline_page("BTCUSDT", "60", 0, 999 * H)
    assert len(session.calls) == 3


def test_unknown_interval_is_rejected_before_request():
    client, session, _ = make_client([])
    with pytest.raises(ValueError, match="unsupported interval"):
        client.get_kline_page("BTCUSDT", "7", 0, 1)
    assert session.calls == []


def test_page_issues_detect_duplicates_and_holes():
    rows = rows_newest_first(0, 5)
    assert find_page_issues(rows, Window(0, 999 * H), H) == []

    with_dup = rows[:2] + [rows[1]] + rows[2:]
    assert any("duplicate" in i for i in find_page_issues(with_dup, Window(0, 999 * H), H))

    with_hole = rows[:2] + rows[4:]  # drop two candles
    assert find_page_issues(with_hole, Window(0, 999 * H), H) == [
        f"hole of 2 candle(s) before {3 * H}"
    ]


def test_joint_issue_between_pages():
    older = rows_newest_first(0, 1000)
    assert find_joint_issue(rows_newest_first(1000 * H, 3), older, H) is None
    assert "hole of 1" in find_joint_issue(rows_newest_first(1001 * H, 3), older, H)
    assert "overlap" in find_joint_issue(rows_newest_first(999 * H, 3), older, H)


def test_get_klines_returns_oldest_first_and_trims_to_range():
    client, _, _ = make_client(
        [ok(rows_newest_first(0, 1000)), ok(rows_newest_first(1000 * H, 1000))]
    )
    rows = client.get_klines("BTCUSDT", "60", start=10 * H, end=1500 * H)
    times = [int(r[0]) for r in rows]
    assert times == list(range(10 * H, 1500 * H, H))


def test_fetch_writes_raw_files_and_skips_them_on_rerun(tmp_path):
    cfg = FetchConfig(
        symbols=("BTCUSDT",),
        interval="60",
        start=datetime.fromtimestamp(0, UTC),
        end=datetime.fromtimestamp(1500 * 3600, UTC),
        raw_dir=tmp_path,
    )
    server_now = 1500 * H + 1  # second window is still open at fetch time
    client, session, _ = make_client(
        [
            ok(rows_newest_first(0, 1000), server_now),
            ok(rows_newest_first(1000 * H, 500), server_now),
        ]
    )
    stats = fetch_series(client, cfg, "BTCUSDT")
    assert (stats.fetched, stats.partial, stats.rows, stats.issues) == (1, 1, 1500, [])

    out = tmp_path / "bybit" / "linear" / "BTCUSDT" / "60"
    assert sorted(p.name for p in out.iterdir()) == [
        "19700101T0000Z.json",
        "19700211T1600Z.partial.json",
    ]
    record = json.loads((out / "19700101T0000Z.json").read_text())
    assert record["complete"] is True
    assert record["request"]["params"]["start"] == 0
    assert record["response"]["result"]["list"][0][0] == str(999 * H)  # stored as-is

    # Rerun later: the complete window is skipped, the partial one is completed
    client2, session2, _ = make_client([ok(rows_newest_first(1000 * H, 1000), 3000 * H)])
    stats2 = fetch_series(client2, cfg, "BTCUSDT")
    assert (stats2.skipped, stats2.fetched, stats2.partial) == (1, 1, 0)
    assert len(session2.calls) == 1
    assert sorted(p.name for p in out.iterdir()) == ["19700101T0000Z.json", "19700211T1600Z.json"]
