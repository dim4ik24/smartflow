"""Derivatives data collector — funding rate, open interest, long/short ratio.

Fetches perpetual-futures metrics from Bybit/Binance via ccxt REST every
``derivatives_collect_interval_minutes`` minutes and inserts one
DerivativesSnapshot row per symbol. All fetch errors are logged and silenced;
the scheduler job never crashes.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import ccxt
import ccxt.async_support as ccxt_async
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, settings
from app.db.models import DerivativesSnapshot
from app.db.session import AsyncSessionLocal

log = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0

_RETRYABLE: tuple[type[Exception], ...] = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
    ccxt.RateLimitExceeded,
)
_SKIPPABLE: tuple[type[Exception], ...] = (
    ccxt.NotSupported,
    ccxt.BadSymbol,
    ccxt.BadRequest,
    ccxt.PermissionDenied,
)


# ── Retry helper ──────────────────────────────────────────────────────────────

async def _call_with_retry(fn: Any, *args: Any, label: str) -> Any:
    """Call ``await fn(*args)`` with exponential backoff on transient errors.

    Returns None on terminal errors (NotSupported, BadSymbol …) or after
    exhausting retries. Never raises.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return await fn(*args)
        except _SKIPPABLE as exc:
            log.debug("derivatives_not_supported", label=label, error=str(exc))
            return None
        except _RETRYABLE as exc:
            delay = _BACKOFF_BASE ** attempt
            log.warning(
                "derivatives_retry",
                label=label,
                attempt=attempt,
                next_delay_s=delay,
                error=str(exc),
            )
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(delay)
        except Exception as exc:
            log.warning("derivatives_unexpected", label=label, error=str(exc))
            return None
    log.warning("derivatives_exhausted", label=label, attempts=_MAX_RETRIES)
    return None


# ── Per-metric fetchers ────────────────────────────────────────────────────────

async def _fetch_funding_rate(ex: ccxt_async.Exchange, symbol: str) -> float | None:
    raw = await _call_with_retry(ex.fetch_funding_rate, symbol, label=f"fr:{symbol}")
    if raw is None:
        return None
    try:
        return float(raw["fundingRate"])
    except (KeyError, TypeError, ValueError):
        return None


async def _fetch_open_interest(ex: ccxt_async.Exchange, symbol: str) -> float | None:
    raw = await _call_with_retry(ex.fetch_open_interest, symbol, label=f"oi:{symbol}")
    if raw is None:
        return None
    try:
        # Prefer USD-denominated value; fall back to base-currency amount.
        val = (
            raw.get("openInterestValue")
            or raw.get("openInterestAmount")
            or raw.get("openInterest")
        )
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


async def _fetch_long_short_ratio(ex: ccxt_async.Exchange, symbol: str) -> float | None:
    # fetch_long_short_ratio returns a list ordered oldest→newest; take last.
    raw = await _call_with_retry(
        ex.fetch_long_short_ratio, symbol, "5m", label=f"lsr:{symbol}"
    )
    if raw is None:
        return None
    try:
        if isinstance(raw, list):
            return float(raw[-1]["longShortRatio"]) if raw else None
        return float(raw["longShortRatio"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None


# ── Snapshot builder ──────────────────────────────────────────────────────────

async def fetch_snapshot_for_symbol(
    ex: ccxt_async.Exchange,
    symbol: str,
) -> DerivativesSnapshot | None:
    """Fetch all three metrics concurrently; return None only when all fail."""
    ts = datetime.now(UTC)
    fr, oi, lsr = await asyncio.gather(
        _fetch_funding_rate(ex, symbol),
        _fetch_open_interest(ex, symbol),
        _fetch_long_short_ratio(ex, symbol),
    )
    if fr is None and oi is None and lsr is None:
        log.warning("derivatives_all_metrics_failed", symbol=symbol)
        return None
    return DerivativesSnapshot(
        symbol=symbol,
        ts=ts,
        funding_rate=fr,
        open_interest=oi,
        long_short_ratio=lsr,
    )


# ── DB query ─────────────────────────────────────────────────────────────────

async def get_latest_derivatives(
    symbol: str,
    session: AsyncSession,
) -> DerivativesSnapshot | None:
    """Return the most recent DerivativesSnapshot for *symbol*, or None."""
    result = await session.execute(
        select(DerivativesSnapshot)
        .where(DerivativesSnapshot.symbol == symbol)
        .order_by(DerivativesSnapshot.ts.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_prev_derivatives(
    symbol: str,
    session: AsyncSession,
) -> DerivativesSnapshot | None:
    """Return the second-latest snapshot for *symbol*, used to compute ΔOI.

    Returns None when fewer than two snapshots exist (cold start / early run).
    """
    result = await session.execute(
        select(DerivativesSnapshot)
        .where(DerivativesSnapshot.symbol == symbol)
        .order_by(DerivativesSnapshot.ts.desc())
        .limit(2)
    )
    rows = result.scalars().all()
    return rows[1] if len(rows) >= 2 else None


# ── Exchange factory ──────────────────────────────────────────────────────────

def _build_exchange(s: Settings) -> ccxt_async.Exchange:
    opts: dict[str, Any] = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    if s.collector_exchange == "bybit":
        ex: ccxt_async.Exchange = ccxt_async.bybit(opts)
    else:
        ex = ccxt_async.binance(opts)
    if s.use_testnet:
        ex.set_sandbox_mode(True)
    return ex


# ── Scheduler job ─────────────────────────────────────────────────────────────

async def collect_derivatives() -> None:
    """Fetch derivatives metrics for all watched symbols and persist snapshots."""
    ex = _build_exchange(settings)
    try:
        await ex.load_markets()

        snapshots: list[DerivativesSnapshot] = []
        for symbol in settings.watched_symbols:
            snap = await fetch_snapshot_for_symbol(ex, symbol)
            if snap is not None:
                snapshots.append(snap)

        if not snapshots:
            log.warning("derivatives_no_snapshots_collected")
            return

        async with AsyncSessionLocal() as session:
            session.add_all(snapshots)
            await session.commit()

        log.info("derivatives_collected", count=len(snapshots))

    except Exception as exc:
        log.error("derivatives_collect_error", error=str(exc), exc_info=True)
    finally:
        try:
            await ex.close()
        except Exception as exc:
            log.warning("derivatives_close_error", error=str(exc))


def start_derivatives_scheduler(
    scheduler: AsyncIOScheduler | None = None,
) -> AsyncIOScheduler:
    """Register collect_derivatives on *scheduler* and start if not running."""
    if scheduler is None:
        scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        collect_derivatives,
        trigger="interval",
        minutes=settings.derivatives_collect_interval_minutes,
        id="collect_derivatives",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=120,
    )
    if not scheduler.running:
        scheduler.start()
    return scheduler
