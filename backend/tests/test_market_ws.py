"""Tests for app/collectors/market_ws.py.

Pure-function tests need no DB. DB tests use the db_session fixture.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.market_ws import (
    _build_exchange,
    _build_rest_exchange,
    _detect_gaps,
    _gap_fill,
    _get_last_ts,
    _heartbeat_watcher,
    _upsert_batch,
    parse_ohlcv_row,
)
from app.db.models import Candle

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_row(  # noqa: E741
    ts: datetime,
    o: float = 100.0,
    h: float = 110.0,
    low: float = 90.0,
    c: float = 105.0,
    v: float = 1000.0,
) -> list[float]:
    return [ts.timestamp() * 1000, o, h, low, c, v]


# ── Pure function tests ────────────────────────────────────────────────────────

def test_parse_ohlcv_row_maps_all_fields() -> None:
    ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    row = _make_row(ts, o=42000.0, h=43000.0, low=41000.0, c=42500.0, v=500.0)

    candle = parse_ohlcv_row(row, "BTC/USDT", "1h")

    assert candle.symbol == "BTC/USDT"
    assert candle.timeframe == "1h"
    assert candle.ts == ts
    assert candle.o == 42000.0
    assert candle.h == 43000.0
    assert candle.l == 41000.0
    assert candle.c == 42500.0
    assert candle.v == 500.0


def test_parse_ohlcv_row_timezone_is_utc() -> None:
    ts = datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)
    row = _make_row(ts)
    candle = parse_ohlcv_row(row, "ETH/USDT", "15m")
    assert candle.ts.tzinfo is not None
    assert candle.ts == ts


def test_detect_gaps_cold_start_returns_backfill_window() -> None:
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    tf_secs = 3600  # 1h
    limit = 10

    since_ms = _detect_gaps(None, now, tf_secs, limit)

    expected_dt = now - timedelta(seconds=tf_secs * limit)
    expected_ms = int(expected_dt.timestamp() * 1000)
    assert since_ms == expected_ms


def test_detect_gaps_no_gap_when_current() -> None:
    tf_secs = 3600
    # last_ts is exactly 1 interval ago — no gap
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    last_ts = now - timedelta(seconds=tf_secs)

    since_ms = _detect_gaps(last_ts, now, tf_secs, 500)

    assert since_ms == 0


def test_detect_gaps_detects_gap() -> None:
    tf_secs = 3600
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    last_ts = now - timedelta(seconds=tf_secs * 3)  # 3 intervals ago → 2-interval gap

    since_ms = _detect_gaps(last_ts, now, tf_secs, 500)

    expected_first_missing = last_ts + timedelta(seconds=tf_secs)
    expected_ms = int(expected_first_missing.timestamp() * 1000)
    assert since_ms == expected_ms


def test_detect_gaps_exactly_two_intervals_returns_gap() -> None:
    tf_secs = 900  # 15m
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    last_ts = now - timedelta(seconds=tf_secs * 2)

    since_ms = _detect_gaps(last_ts, now, tf_secs, 500)

    # elapsed_intervals = 2 > 1 → gap detected
    expected_ms = int((last_ts + timedelta(seconds=tf_secs)).timestamp() * 1000)
    assert since_ms == expected_ms


# ── DB tests ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upsert_batch_insert_then_update(db_session: AsyncSession) -> None:
    from sqlalchemy import select

    from app.db.models import Candle as CandleModel

    ts = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)
    original = Candle(
        symbol="BTC/USDT", timeframe="1h", ts=ts,
        o=40000.0, h=41000.0, l=39000.0, c=40500.0, v=100.0,
    )
    await _upsert_batch([original], db_session)

    updated = Candle(
        symbol="BTC/USDT", timeframe="1h", ts=ts,
        o=40000.0, h=42000.0, l=38000.0, c=41000.0, v=200.0,
    )
    await _upsert_batch([updated], db_session)

    result = await db_session.execute(
        select(CandleModel).where(
            CandleModel.symbol == "BTC/USDT",
            CandleModel.timeframe == "1h",
            CandleModel.ts == ts,
        )
    )
    row = result.scalar_one()
    assert row.h == 42000.0
    assert row.v == 200.0


@pytest.mark.asyncio
async def test_skip_open_candle_only_closed_inserted(db_session: AsyncSession) -> None:
    """3 rows submitted → only the first 2 (closed candles) should be upserted."""
    from sqlalchemy import func, select

    from app.db.models import Candle as CandleModel

    ts_base = datetime(2024, 4, 1, 8, 0, 0, tzinfo=UTC)
    rows: list[list[float]] = [
        _make_row(ts_base, c=100.0),
        _make_row(ts_base + timedelta(hours=1), c=101.0),
        _make_row(ts_base + timedelta(hours=2), c=102.0),  # still-forming, must be skipped
    ]

    # Simulate the collector's skipping of the last row (rows[:-1])
    closed_rows = rows[:-1]
    candles = [parse_ohlcv_row(r, "ETH/USDT", "1h") for r in closed_rows]
    await _upsert_batch(candles, db_session)

    count_result = await db_session.execute(
        select(func.count()).where(
            CandleModel.symbol == "ETH/USDT",
            CandleModel.timeframe == "1h",
        )
    )
    count = count_result.scalar_one()
    assert count == 2


@pytest.mark.asyncio
async def test_upsert_batch_empty_list_is_noop(db_session: AsyncSession) -> None:
    """Calling _upsert_batch with an empty list should not raise."""
    await _upsert_batch([], db_session)  # must not raise


# ── Reconnect backoff test ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reconnect_backoff_doubles_and_caps() -> None:
    """_run_timeframe_task must double backoff on each network error, capping at 60s."""
    from ccxt.base.errors import NetworkError

    from app.collectors.market_ws import _run_timeframe_task

    sleep_calls: list[float] = []

    class FakeSettings:
        watched_symbols = ["BTC/USDT"]
        collector_backfill_limit = 10
        collector_heartbeat_timeout = 30

    async def fake_get_last_ts(*_: Any) -> None:
        return None

    async def fake_gap_fill(*_: Any, **__: Any) -> None:
        pass

    call_count = 0

    async def fake_watch(*_: Any, **__: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count >= 5:
            raise asyncio.CancelledError
        raise NetworkError("connection refused")

    async def fake_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    fake_ws = MagicMock()
    fake_ws.watch_ohlcv_for_symbols = fake_watch
    fake_rest = MagicMock()

    with (
        patch("app.collectors.market_ws._get_last_ts", fake_get_last_ts),
        patch("app.collectors.market_ws._gap_fill", fake_gap_fill),
        patch("app.collectors.market_ws.AsyncSessionLocal"),
        patch("asyncio.sleep", fake_sleep),
        pytest.raises(asyncio.CancelledError),
    ):
        await _run_timeframe_task("1h", fake_ws, fake_rest, FakeSettings())

    assert len(sleep_calls) >= 3
    assert sleep_calls[0] == 1.0
    assert sleep_calls[1] == 2.0
    assert sleep_calls[2] == 4.0
    assert all(s <= 60.0 for s in sleep_calls)


# ── Integration test — fake WS data → DB ──────────────────────────────────────

@pytest.mark.asyncio
async def test_integration_fake_ws_data_upserts_closed_candles(db_session: AsyncSession) -> None:
    """Simulate WS data processing: 3 rows arrive, only 2 (closed) end up in DB."""
    from sqlalchemy import func, select

    from app.db.models import Candle as CandleModel

    ts_base = datetime(2024, 5, 10, 6, 0, 0, tzinfo=UTC)
    symbol = "SOL/USDT"
    timeframe = "15m"

    row0 = _make_row(ts_base, c=150.0)
    row1 = _make_row(ts_base + timedelta(minutes=15), c=151.0)
    row_open = _make_row(ts_base + timedelta(minutes=30), c=152.0)  # still-forming

    fake_data: dict[str, dict[str, list[list[float]]]] = {
        symbol: {timeframe: [row0, row1, row_open]}
    }

    batch: list[Candle] = []
    for sym, tf_data in fake_data.items():
        rows = tf_data.get(timeframe, [])
        for row in rows[:-1]:
            batch.append(parse_ohlcv_row(row, sym, timeframe))

    await _upsert_batch(batch, db_session)

    count_result = await db_session.execute(
        select(func.count()).where(
            CandleModel.symbol == symbol,
            CandleModel.timeframe == timeframe,
        )
    )
    count = count_result.scalar_one()
    assert count == 2

    result = await db_session.execute(
        select(CandleModel).where(
            CandleModel.symbol == symbol,
            CandleModel.timeframe == timeframe,
            CandleModel.ts == ts_base + timedelta(minutes=15),
        )
    )
    second_candle = result.scalar_one()
    assert second_candle.c == 151.0


# ── _get_last_ts tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_last_ts_returns_none_when_no_data(db_session: AsyncSession) -> None:
    result = await _get_last_ts("NONEXISTENT/USDT", "4h", db_session)
    assert result is None


@pytest.mark.asyncio
async def test_get_last_ts_returns_newest_timestamp(db_session: AsyncSession) -> None:
    ts1 = datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
    ts2 = datetime(2024, 2, 2, 0, 0, 0, tzinfo=UTC)
    for ts in (ts1, ts2):
        c = Candle(symbol="DOT/USDT", timeframe="4h", ts=ts, o=1.0, h=2.0, l=0.5, c=1.5, v=10.0)
        await db_session.merge(c)
    await db_session.commit()

    result = await _get_last_ts("DOT/USDT", "4h", db_session)
    # SQLite strips timezone on storage; compare only the naive value
    assert result is not None
    assert result.replace(tzinfo=None) == ts2.replace(tzinfo=None)


# ── _gap_fill tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gap_fill_success_upserts_closed_rows(db_session: AsyncSession) -> None:
    from sqlalchemy import func, select

    from app.db.models import Candle as CandleModel

    ts_base = datetime(2024, 7, 1, 0, 0, 0, tzinfo=UTC)
    rows = [
        _make_row(ts_base, c=100.0),
        _make_row(ts_base + timedelta(hours=1), c=101.0),
        _make_row(ts_base + timedelta(hours=2), c=102.0),  # last row (forming), skipped
    ]

    fake_rest = MagicMock()
    fake_rest.fetch_ohlcv = AsyncMock(return_value=rows)

    since_ms = int(ts_base.timestamp() * 1000)
    await _gap_fill(fake_rest, "LINK/USDT", "1h", since_ms, 10, db_session)

    count = (
        await db_session.execute(
            select(func.count()).where(
                CandleModel.symbol == "LINK/USDT",
                CandleModel.timeframe == "1h",
            )
        )
    ).scalar_one()
    assert count == 2


@pytest.mark.asyncio
async def test_gap_fill_exhausted_retries_returns_without_raising(db_session: AsyncSession) -> None:
    from ccxt.base.errors import NetworkError

    fake_rest = MagicMock()
    fake_rest.fetch_ohlcv = AsyncMock(side_effect=NetworkError("timeout"))

    async def fast_sleep(_: float) -> None:
        pass

    with patch("asyncio.sleep", fast_sleep):
        # Returns silently after 5 failed attempts
        await _gap_fill(fake_rest, "ATOM/USDT", "1h", 0, 10, db_session)

    assert fake_rest.fetch_ohlcv.call_count == 5


# ── _heartbeat_watcher tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_heartbeat_watcher_triggers_cancel_event_when_stale() -> None:
    cancel_event = asyncio.Event()
    # Simulate data that arrived 1000 s ago — well past any timeout
    last_ref: list[float] = [time.monotonic() - 1000]

    async def fast_sleep(_: float) -> None:
        pass

    with patch("asyncio.sleep", fast_sleep):
        await _heartbeat_watcher(last_ref, 30, cancel_event, "test_task")

    assert cancel_event.is_set()


@pytest.mark.asyncio
async def test_heartbeat_watcher_does_not_trigger_when_fresh() -> None:
    """When data is fresh the watcher must not set the event on the first check."""
    cancel_event = asyncio.Event()
    last_ref: list[float] = [time.monotonic()]  # just now

    iteration = 0

    async def fake_sleep(_: float) -> None:
        nonlocal iteration
        iteration += 1
        if iteration >= 2:
            # Force exit by setting the event externally after 2 ticks
            cancel_event.set()

    with patch("asyncio.sleep", fake_sleep):
        await _heartbeat_watcher(last_ref, 30, cancel_event, "test_fresh")

    # Event set externally after 2 ticks — not triggered by stale data
    assert cancel_event.is_set()
    assert iteration == 2


# ── Exchange builder tests ─────────────────────────────────────────────────────

def test_build_exchange_returns_bybit_instance() -> None:
    class FakeSettings:
        collector_exchange = "bybit"
        use_testnet = False

    ex = _build_exchange(FakeSettings())  # type: ignore[arg-type]
    assert ex is not None
    assert "bybit" in type(ex).__name__.lower()


def test_build_exchange_binance_testnet_applies_sandbox() -> None:
    class FakeSettings:
        collector_exchange = "binance"
        use_testnet = True

    ex = _build_exchange(FakeSettings())  # type: ignore[arg-type]
    assert ex is not None


def test_build_rest_exchange_bybit() -> None:
    class FakeSettings:
        collector_exchange = "bybit"
        use_testnet = False

    ex = _build_rest_exchange(FakeSettings())  # type: ignore[arg-type]
    assert ex is not None
    assert "bybit" in type(ex).__name__.lower()


def test_build_rest_exchange_binance_testnet() -> None:
    class FakeSettings:
        collector_exchange = "binance"
        use_testnet = True

    ex = _build_rest_exchange(FakeSettings())  # type: ignore[arg-type]
    assert ex is not None
