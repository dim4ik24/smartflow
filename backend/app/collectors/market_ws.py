"""WebSocket OHLCV collector with auto-reconnect, heartbeat, and REST gap-fill.

One asyncio task per timeframe (15m / 1h / 4h). Each task subscribes to all
configured symbols via watch_ohlcv_for_symbols on a single WS connection.
On every (re)connect the task gap-fills missing candles via REST.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import signal
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, settings
from app.db.models import Candle
from app.db.session import AsyncSessionLocal

# ── Constants ─────────────────────────────────────────────────────────────────

_TF_SECONDS: dict[str, int] = {"15m": 900, "1h": 3600, "4h": 14400}

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ── Exchange factories ────────────────────────────────────────────────────────


def _build_exchange(s: Settings) -> Any:
    """Return a ccxt.pro WebSocket exchange (one shared instance per collector)."""
    import ccxt.pro as ccxtpro  # deferred: speeds up import of other modules

    opts: dict[str, Any] = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    if s.collector_exchange == "bybit":
        return ccxtpro.bybit({"testnet": s.use_testnet, **opts})
    ex = ccxtpro.binance(opts)
    if s.use_testnet:
        ex.set_sandbox_mode(True)
    return ex


def _build_rest_exchange(s: Settings) -> Any:
    """Return a ccxt.async_support REST exchange used exclusively for gap-fill."""
    import ccxt.async_support as ccxt_async  # deferred; separate instance from WS

    opts: dict[str, Any] = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    ex = ccxt_async.bybit(opts) if s.collector_exchange == "bybit" else ccxt_async.binance(opts)
    if s.use_testnet:
        ex.set_sandbox_mode(True)
    return ex


# ── Pure helpers (no I/O — easy to unit-test) ─────────────────────────────────


def parse_ohlcv_row(row: list[float], symbol: str, timeframe: str) -> Candle:
    """Convert one ccxt OHLCV row [ts_ms, o, h, l, c, v] to a Candle ORM object."""
    ts_ms, o, h, l, c, v = row  # noqa: E741 — OHLCV convention
    ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=UTC)
    return Candle(symbol=symbol, timeframe=timeframe, ts=ts, o=o, h=h, l=l, c=c, v=v)


def _detect_gaps(
    last_ts: datetime | None,
    now_utc: datetime,
    tf_secs: int,
    limit: int,
) -> int:
    """Return the since_ms value for REST gap-fill, or 0 if no gap exists.

    Cold-start (no data): fetches the last `limit` candles of history.
    Current (last_ts is the most recent closed candle): returns 0.
    Gap detected: returns the ms timestamp of the first missing candle.
    """
    if last_ts is None:
        since_dt = now_utc - timedelta(seconds=tf_secs * limit)
        return int(since_dt.timestamp() * 1000)
    elapsed_intervals = int((now_utc - last_ts).total_seconds() / tf_secs)
    if elapsed_intervals <= 1:
        return 0
    return int((last_ts + timedelta(seconds=tf_secs)).timestamp() * 1000)


# ── Database helpers ──────────────────────────────────────────────────────────


async def _get_last_ts(
    symbol: str, timeframe: str, session: AsyncSession
) -> datetime | None:
    """Return the newest candle timestamp for (symbol, timeframe), or None."""
    result = await session.execute(
        select(func.max(Candle.ts)).where(
            Candle.symbol == symbol, Candle.timeframe == timeframe
        )
    )
    return result.scalar_one_or_none()


async def _upsert_batch(candles: list[Candle], session: AsyncSession) -> None:
    """Merge (INSERT-or-UPDATE) a list of candles in a single transaction."""
    if not candles:
        return
    for candle in candles:
        await session.merge(candle)
    await session.commit()


# ── Gap-fill via REST ─────────────────────────────────────────────────────────


async def _gap_fill(
    rest_ex: Any,
    symbol: str,
    timeframe: str,
    since_ms: int,
    limit: int,
    session: AsyncSession,
) -> None:
    """Fetch candles missing since `since_ms` via REST and upsert them.

    Retries up to 5 times with exponential backoff on transient network errors.
    """
    from ccxt.base.errors import (
        ExchangeNotAvailable,
        NetworkError,
        RateLimitExceeded,
        RequestTimeout,
    )

    backoff = 1.0
    rows: list[list[float]] = []
    for attempt in range(5):
        try:
            rows = await rest_ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            break
        except RateLimitExceeded:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
        except (NetworkError, ExchangeNotAvailable, RequestTimeout) as exc:
            log.warning(
                "gap_fill_fetch_failed",
                symbol=symbol,
                tf=timeframe,
                attempt=attempt,
                error=str(exc),
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
    else:
        log.error("gap_fill_exhausted_retries", symbol=symbol, tf=timeframe)
        return

    # Skip the last row — it may still be forming at the time of the REST call
    closed = rows[:-1] if len(rows) > 1 else rows
    candles = [parse_ohlcv_row(r, symbol, timeframe) for r in closed]
    await _upsert_batch(candles, session)
    log.info("gap_fill_complete", symbol=symbol, tf=timeframe, count=len(candles))


# ── Heartbeat ─────────────────────────────────────────────────────────────────


async def _heartbeat_watcher(
    last_received_ref: list[float],
    timeout: int,
    cancel_event: asyncio.Event,
    task_name: str,
) -> None:
    """Signal cancel_event if no WS data arrives within `timeout` seconds."""
    interval = max(timeout // 3, 5)
    while not cancel_event.is_set():
        await asyncio.sleep(interval)
        age = time.monotonic() - last_received_ref[0]
        if age > timeout:
            log.warning("heartbeat_timeout", task=task_name, age_secs=round(age, 1))
            cancel_event.set()


# ── Per-timeframe WS task ────────────────────────────────────────────────────


async def _run_timeframe_task(
    timeframe: str,
    ws_ex: Any,
    rest_ex: Any,
    s: Settings,
) -> None:
    """Run the WebSocket receive loop for one timeframe with auto-reconnect.

    Backoff resets only after the first successful batch of data is received,
    so repeated immediate failures accumulate delay correctly.
    """
    from ccxt.base.errors import (
        ExchangeNotAvailable,
        NetworkError,
        RateLimitExceeded,
        RequestTimeout,
    )

    tf_secs = _TF_SECONDS[timeframe]
    sym_tf_pairs = [[sym, timeframe] for sym in s.watched_symbols]
    backoff = 1.0
    bound_log = log.bind(tf=timeframe)

    while True:
        try:
            # ── Gap-fill on every (re)connect ──────────────────────────────────
            async with AsyncSessionLocal() as session:
                for sym in s.watched_symbols:
                    last_ts = await _get_last_ts(sym, timeframe, session)
                    since_ms = _detect_gaps(
                        last_ts, datetime.now(UTC), tf_secs, s.collector_backfill_limit
                    )
                    if since_ms:
                        await _gap_fill(
                            rest_ex, sym, timeframe, since_ms, s.collector_backfill_limit, session
                        )

            bound_log.info("ws_connected", symbols=len(s.watched_symbols))

            # ── Heartbeat ──────────────────────────────────────────────────────
            last_received_ref: list[float] = [time.monotonic()]
            cancel_event = asyncio.Event()
            hb_task = asyncio.create_task(
                _heartbeat_watcher(
                    last_received_ref,
                    s.collector_heartbeat_timeout,
                    cancel_event,
                    f"ws_{timeframe}",
                )
            )

            is_first_batch = True
            try:
                while not cancel_event.is_set():
                    data: dict[str, dict[str, list[list[float]]]] = (
                        await ws_ex.watch_ohlcv_for_symbols(sym_tf_pairs)
                    )
                    last_received_ref[0] = time.monotonic()

                    if is_first_batch:
                        backoff = 1.0  # confirmed live; reset backoff
                        is_first_batch = False

                    batch: list[Candle] = []
                    for sym, tf_data in data.items():
                        rows = tf_data.get(timeframe, [])
                        for row in rows[:-1]:  # skip last (still-forming) candle
                            batch.append(parse_ohlcv_row(row, sym, timeframe))

                    if batch:
                        async with AsyncSessionLocal() as session:
                            await _upsert_batch(batch, session)
                        bound_log.debug("candles_upserted", count=len(batch))

            finally:
                hb_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await hb_task

            bound_log.warning("heartbeat_triggered_reconnect")

        except asyncio.CancelledError:
            bound_log.info("task_cancelled")
            raise

        except (NetworkError, ExchangeNotAvailable, RequestTimeout, RateLimitExceeded) as exc:
            bound_log.warning("ws_network_error", error=str(exc), next_retry_secs=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

        except Exception as exc:
            bound_log.error("ws_unexpected_error", error=str(exc), exc_info=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


# ── Top-level entrypoint ──────────────────────────────────────────────────────


async def run_collector() -> None:
    """Initialize exchanges, start per-timeframe tasks, handle graceful shutdown."""
    log.info(
        "collector_starting",
        exchange=settings.collector_exchange,
        testnet=settings.use_testnet,
        symbols=len(settings.watched_symbols),
        timeframes=settings.watched_timeframes,
    )

    ws_ex = _build_exchange(settings)
    rest_ex = _build_rest_exchange(settings)

    await ws_ex.load_markets()
    log.info("markets_loaded", count=len(ws_ex.markets))

    tasks: list[asyncio.Task[None]] = [
        asyncio.create_task(
            _run_timeframe_task(tf, ws_ex, rest_ex, settings),
            name=f"collector_{tf}",
        )
        for tf in settings.watched_timeframes
    ]

    loop = asyncio.get_running_loop()

    def _handle_signal(sig: signal.Signals) -> None:
        log.info("shutdown_signal", signal=sig.name)
        for t in tasks:
            t.cancel()

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, functools.partial(_handle_signal, sig))
    except NotImplementedError:
        pass  # Windows: rely on default KeyboardInterrupt handling

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        log.info("closing_exchanges")
        await ws_ex.close()
        await rest_ex.close()
        log.info("collector_stopped")
