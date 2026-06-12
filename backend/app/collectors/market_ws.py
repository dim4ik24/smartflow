"""WebSocket OHLCV collector with auto-reconnect, heartbeat, and REST gap-fill.

One asyncio task per timeframe (15m / 1h / 4h). Each task subscribes to all
configured symbols via watch_ohlcv_for_symbols on a single WS connection.
On every (re)connect the task gap-fills missing candles via REST.
"""

from __future__ import annotations

import asyncio
import functools
import signal
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


def _is_candle_closed(ts_ms: int, tf_secs: int, now_ms: int) -> bool:
    """Return True if the candle opened at ts_ms has already closed.

    A candle opened at ts_ms closes at ts_ms + tf_secs*1000. Comparing against
    wall-clock time is more reliable than skipping the last element positionally,
    which breaks when ccxt returns only the current (forming) candle in newUpdates
    mode.
    """
    return ts_ms + tf_secs * 1000 <= now_ms


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
    """Fetch all missing closed candles since `since_ms` via REST and upsert them.

    Loops over pages until fewer than `limit` rows are returned or the last row
    in a page is still forming. Each page retries up to 5 times with exponential
    backoff on transient network errors.
    """
    from ccxt.base.errors import (
        ExchangeNotAvailable,
        NetworkError,
        RateLimitExceeded,
        RequestTimeout,
    )

    tf_secs = _TF_SECONDS[timeframe]
    total = 0

    while True:
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
            break

        if not rows:
            break

        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        closed = [r for r in rows if _is_candle_closed(int(r[0]), tf_secs, now_ms)]
        if closed:
            candles = [parse_ohlcv_row(r, symbol, timeframe) for r in closed]
            await _upsert_batch(candles, session)
            total += len(closed)

        # Stop if this was the last page or the last row is still forming
        if len(rows) < limit or not _is_candle_closed(int(rows[-1][0]), tf_secs, now_ms):
            break
        since_ms = int(rows[-1][0]) + tf_secs * 1000

    if total:
        log.info("gap_fill_complete", symbol=symbol, tf=timeframe, count=total)


# ── Per-timeframe WS task ────────────────────────────────────────────────────


async def _run_timeframe_task(
    timeframe: str,
    ws_ex: Any,
    rest_ex: Any,
    s: Settings,
) -> None:
    """Run the WebSocket receive loop for one timeframe with auto-reconnect.

    Heartbeat is implemented via asyncio.wait_for: if watch_ohlcv_for_symbols
    does not yield within collector_heartbeat_timeout seconds the receive loop
    breaks and triggers a full reconnect (including gap-fill).

    Backoff resets only after the first successful data batch is received.
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
            is_first_batch = True

            # ── Receive loop (heartbeat via wait_for) ──────────────────────────
            while True:
                try:
                    data: dict[str, dict[str, list[list[float]]]] = await asyncio.wait_for(
                        ws_ex.watch_ohlcv_for_symbols(sym_tf_pairs),
                        timeout=s.collector_heartbeat_timeout,
                    )
                except TimeoutError:
                    bound_log.warning("ws_heartbeat_timeout_reconnect")
                    break  # exit receive loop → reconnect

                if is_first_batch:
                    backoff = 1.0  # confirmed live; reset backoff
                    is_first_batch = False

                now_ms = int(datetime.now(UTC).timestamp() * 1000)
                batch: list[Candle] = []
                for sym, tf_data in data.items():
                    for row in tf_data.get(timeframe, []):
                        if _is_candle_closed(int(row[0]), tf_secs, now_ms):
                            batch.append(parse_ohlcv_row(row, sym, timeframe))

                if batch:
                    async with AsyncSessionLocal() as session:
                        await _upsert_batch(batch, session)
                    bound_log.debug("candles_upserted", count=len(batch))

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
