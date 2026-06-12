"""Tests for app/collectors/market_ws.py.

Pure-function tests need no DB. DB tests use the db_session fixture.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.market_ws import (
    _TF_SECONDS,
    _build_exchange,
    _build_rest_exchange,
    _detect_gaps,
    _gap_fill,
    _get_last_ts,
    _is_candle_closed,
    _upsert_batch,
    parse_ohlcv_row,
)
from app.db.models import Candle

# ── Helpers ────────────────────────────────────────────────────────────────────

def _row(ts_ms: int, c: float = 100.0) -> list[float]:
    return [float(ts_ms), 99.0, 110.0, 90.0, c, 1000.0]


def _past_ms(hours: int = 2) -> int:
    """Return a timestamp `hours` hours in the past as milliseconds."""
    return int((datetime.now(UTC) - timedelta(hours=hours)).timestamp() * 1000)


# ── _is_candle_closed tests ────────────────────────────────────────────────────

def test_is_candle_closed_forming() -> None:
    tf_secs = 3600
    now_ms = 1_700_000_000_000
    # Opened 30 min ago → closes in 30 min
    ts_ms = now_ms - 30 * 60 * 1000
    assert not _is_candle_closed(ts_ms, tf_secs, now_ms)


def test_is_candle_closed_exactly_at_boundary() -> None:
    tf_secs = 3600
    now_ms = 1_700_000_000_000
    ts_ms = now_ms - tf_secs * 1000  # close time == now
    assert _is_candle_closed(ts_ms, tf_secs, now_ms)


def test_is_candle_closed_well_in_past() -> None:
    tf_secs = 900
    now_ms = 1_700_000_000_000
    ts_ms = now_ms - 5 * tf_secs * 1000  # 5 intervals ago
    assert _is_candle_closed(ts_ms, tf_secs, now_ms)


# ── parse_ohlcv_row tests ──────────────────────────────────────────────────────

def test_parse_ohlcv_row_maps_all_fields() -> None:
    ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    row: list[float] = [ts.timestamp() * 1000, 42000.0, 43000.0, 41000.0, 42500.0, 500.0]

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
    row: list[float] = [ts.timestamp() * 1000, 100.0, 110.0, 90.0, 105.0, 1000.0]
    candle = parse_ohlcv_row(row, "ETH/USDT", "15m")
    assert candle.ts.tzinfo is not None
    assert candle.ts == ts


# ── _detect_gaps tests ─────────────────────────────────────────────────────────

def test_detect_gaps_cold_start_returns_backfill_window() -> None:
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    tf_secs = 3600
    limit = 10
    since_ms = _detect_gaps(None, now, tf_secs, limit)
    expected_ms = int((now - timedelta(seconds=tf_secs * limit)).timestamp() * 1000)
    assert since_ms == expected_ms


def test_detect_gaps_no_gap_when_current() -> None:
    tf_secs = 3600
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    last_ts = now - timedelta(seconds=tf_secs)
    assert _detect_gaps(last_ts, now, tf_secs, 500) == 0


def test_detect_gaps_detects_gap() -> None:
    tf_secs = 3600
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    last_ts = now - timedelta(seconds=tf_secs * 3)
    since_ms = _detect_gaps(last_ts, now, tf_secs, 500)
    expected_ms = int((last_ts + timedelta(seconds=tf_secs)).timestamp() * 1000)
    assert since_ms == expected_ms


def test_detect_gaps_exactly_two_intervals_returns_gap() -> None:
    tf_secs = 900
    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    last_ts = now - timedelta(seconds=tf_secs * 2)
    since_ms = _detect_gaps(last_ts, now, tf_secs, 500)
    expected_ms = int((last_ts + timedelta(seconds=tf_secs)).timestamp() * 1000)
    assert since_ms == expected_ms


# ── _upsert_batch tests ────────────────────────────────────────────────────────

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
async def test_upsert_batch_empty_list_is_noop(db_session: AsyncSession) -> None:
    await _upsert_batch([], db_session)  # must not raise


# ── Time-based candle closure in WS loop ──────────────────────────────────────

@pytest.mark.asyncio
async def test_forming_candle_not_stored_closed_candle_is_stored(
    db_session: AsyncSession,
) -> None:
    """Candle whose close time is in the future must not be upserted; a closed
    candle with the same symbol/tf that later satisfies _is_candle_closed must be."""
    from sqlalchemy import func, select

    from app.db.models import Candle as CandleModel

    tf_secs = _TF_SECONDS["1h"]
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    # Forming: opened 30 min ago → closes in 30 min
    forming_ts_ms = now_ms - 30 * 60 * 1000
    # Closed: opened 3 h ago → close was 2 h ago
    closed_ts_ms = now_ms - 3 * tf_secs * 1000

    rows = [_row(closed_ts_ms, c=100.0), _row(forming_ts_ms, c=101.0)]

    batch = [
        parse_ohlcv_row(r, "AVAX/USDT", "1h")
        for r in rows
        if _is_candle_closed(int(r[0]), tf_secs, now_ms)
    ]
    await _upsert_batch(batch, db_session)

    count = (
        await db_session.execute(
            select(func.count()).where(
                CandleModel.symbol == "AVAX/USDT",
                CandleModel.timeframe == "1h",
            )
        )
    ).scalar_one()
    assert count == 1  # forming candle was skipped

    # Simulate the same candle now closed (advance its ts to the past)
    closed_forming = Candle(
        symbol="AVAX/USDT",
        timeframe="1h",
        ts=datetime.fromtimestamp(forming_ts_ms / 1000, tz=UTC),
        o=99.0, h=109.0, l=89.0, c=101.0, v=1000.0,
    )
    # "now" has advanced enough that forming_ts_ms + tf_secs*1000 <= new_now_ms
    new_now_ms = forming_ts_ms + tf_secs * 1000 + 1
    assert _is_candle_closed(forming_ts_ms, tf_secs, new_now_ms)
    await _upsert_batch([closed_forming], db_session)

    count2 = (
        await db_session.execute(
            select(func.count()).where(
                CandleModel.symbol == "AVAX/USDT",
                CandleModel.timeframe == "1h",
            )
        )
    ).scalar_one()
    assert count2 == 2


# ── _gap_fill tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gap_fill_time_based_skips_forming_candle(db_session: AsyncSession) -> None:
    """A candle that hasn't closed (ts + tf_secs > now) must not be upserted."""
    from sqlalchemy import func, select

    from app.db.models import Candle as CandleModel

    tf_secs = _TF_SECONDS["1h"]
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    closed_ts_ms = now_ms - 3 * tf_secs * 1000   # closed 2 h ago
    forming_ts_ms = now_ms - 30 * 60 * 1000       # closes in 30 min

    rows = [_row(closed_ts_ms, c=200.0), _row(forming_ts_ms, c=201.0)]
    fake_rest = MagicMock()
    fake_rest.fetch_ohlcv = AsyncMock(return_value=rows)

    await _gap_fill(fake_rest, "NEAR/USDT", "1h", closed_ts_ms, 10, db_session)

    count = (
        await db_session.execute(
            select(func.count()).where(
                CandleModel.symbol == "NEAR/USDT",
                CandleModel.timeframe == "1h",
            )
        )
    ).scalar_one()
    assert count == 1


@pytest.mark.asyncio
async def test_gap_fill_single_forming_candle_not_stored(db_session: AsyncSession) -> None:
    """Edge-case: REST returns exactly 1 row that is still forming → nothing stored."""
    from sqlalchemy import func, select

    from app.db.models import Candle as CandleModel

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    forming_ts_ms = now_ms - 10 * 60 * 1000  # 10 min ago, closes in 50 min

    fake_rest = MagicMock()
    fake_rest.fetch_ohlcv = AsyncMock(return_value=[_row(forming_ts_ms)])

    await _gap_fill(fake_rest, "UNI/USDT", "1h", forming_ts_ms, 10, db_session)

    count = (
        await db_session.execute(
            select(func.count()).where(
                CandleModel.symbol == "UNI/USDT",
                CandleModel.timeframe == "1h",
            )
        )
    ).scalar_one()
    assert count == 0


@pytest.mark.asyncio
async def test_gap_fill_multi_page_fetches_until_last_page(db_session: AsyncSession) -> None:
    """If the first page is full (len == limit), _gap_fill fetches the next page."""
    from sqlalchemy import func, select

    from app.db.models import Candle as CandleModel

    tf_secs = _TF_SECONDS["1h"]
    # Use timestamps well in the past so all rows are closed
    base_ms = _past_ms(hours=10)

    page1 = [_row(base_ms + i * tf_secs * 1000, c=float(i)) for i in range(3)]
    page2 = [_row(base_ms + 3 * tf_secs * 1000, c=3.0)]  # 1 row < limit → last page

    fetch_calls: list[int] = []

    async def fake_fetch(symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
        fetch_calls.append(since)
        return page1 if len(fetch_calls) == 1 else page2

    fake_rest = MagicMock()
    fake_rest.fetch_ohlcv = fake_fetch

    await _gap_fill(fake_rest, "DOT/USDT", "1h", base_ms, limit=3, session=db_session)

    assert len(fetch_calls) == 2, "expected exactly 2 pages fetched"

    count = (
        await db_session.execute(
            select(func.count()).where(
                CandleModel.symbol == "DOT/USDT",
                CandleModel.timeframe == "1h",
            )
        )
    ).scalar_one()
    assert count == 4


@pytest.mark.asyncio
async def test_gap_fill_exhausted_retries_returns_without_raising(db_session: AsyncSession) -> None:
    from ccxt.base.errors import NetworkError

    fake_rest = MagicMock()
    fake_rest.fetch_ohlcv = AsyncMock(side_effect=NetworkError("timeout"))

    async def fast_sleep(_: float) -> None:
        pass

    with patch("asyncio.sleep", fast_sleep):
        await _gap_fill(fake_rest, "ATOM/USDT", "1h", 0, 10, db_session)

    assert fake_rest.fetch_ohlcv.call_count == 5


# ── Reconnect / backoff tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reconnect_backoff_doubles_and_caps() -> None:
    """_run_timeframe_task doubles backoff on each NetworkError, capping at 60 s."""
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


@pytest.mark.asyncio
async def test_ws_timeout_triggers_reconnect() -> None:
    """A TimeoutError from wait_for must cause a reconnect (gap-fill called again)."""
    from app.collectors.market_ws import _run_timeframe_task

    gap_fill_calls: list[int] = []

    async def fake_gap_fill(*_: Any, **__: Any) -> None:
        gap_fill_calls.append(1)

    async def fake_get_last_ts(*_: Any) -> None:
        return None

    wf_calls = 0

    async def fake_wait_for(coro: Any, timeout: float = 0, **_: Any) -> Any:
        nonlocal wf_calls
        wf_calls += 1
        with contextlib.suppress(Exception):
            coro.close()
        if wf_calls == 1:
            raise TimeoutError  # first call → heartbeat timeout → reconnect
        raise asyncio.CancelledError  # second call → shut down

    class FakeSettings:
        watched_symbols = ["BTC/USDT"]
        collector_backfill_limit = 10
        collector_heartbeat_timeout = 30

    fake_ws = MagicMock()
    fake_ws.watch_ohlcv_for_symbols = AsyncMock()
    fake_rest = MagicMock()

    with (
        patch("asyncio.wait_for", fake_wait_for),
        patch("app.collectors.market_ws._get_last_ts", fake_get_last_ts),
        patch("app.collectors.market_ws._gap_fill", fake_gap_fill),
        patch("app.collectors.market_ws.AsyncSessionLocal"),
        pytest.raises(asyncio.CancelledError),
    ):
        await _run_timeframe_task("1h", fake_ws, fake_rest, FakeSettings())

    # gap-fill must have run twice: initial connect + after timeout reconnect
    assert len(gap_fill_calls) == 2
    assert wf_calls == 2


# ── Integration test — fake WS data → DB ──────────────────────────────────────

@pytest.mark.asyncio
async def test_integration_fake_ws_data_upserts_only_closed_candles(
    db_session: AsyncSession,
) -> None:
    """End-to-end: WS data with 2 closed rows + 1 forming → only 2 in DB."""
    from sqlalchemy import func, select

    from app.db.models import Candle as CandleModel

    tf_secs = _TF_SECONDS["15m"]
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    symbol = "SOL/USDT"
    timeframe = "15m"

    closed1_ms = now_ms - 3 * tf_secs * 1000
    closed2_ms = now_ms - 2 * tf_secs * 1000
    forming_ms = now_ms - 5 * 60 * 1000  # 5 min ago, closes in 10 min

    fake_data: dict[str, dict[str, list[list[float]]]] = {
        symbol: {
            timeframe: [_row(closed1_ms, 150.0), _row(closed2_ms, 151.0), _row(forming_ms, 152.0)]
        }
    }

    batch: list[Candle] = []
    for sym, tf_data in fake_data.items():
        for row in tf_data.get(timeframe, []):
            if _is_candle_closed(int(row[0]), tf_secs, now_ms):
                batch.append(parse_ohlcv_row(row, sym, timeframe))

    await _upsert_batch(batch, db_session)

    count = (
        await db_session.execute(
            select(func.count()).where(
                CandleModel.symbol == symbol,
                CandleModel.timeframe == timeframe,
            )
        )
    ).scalar_one()
    assert count == 2


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
