"""Signal analysis engine — orchestrator for the full pipeline (SPEC §5).

``analyze_symbol_on_close`` is called whenever a candle closes:
  1. Load OHLCV from DB (4h context + entry TF)
  2. Idempotency check — skip if the latest candle was already analysed
  3. Run SMC analysis (confirmed_only=True to avoid lookahead bias)
  4. Determine trade side from 4h structural direction
  5. Fetch derivatives and news sentiment
  6. Score via scoring.py
  7. Deduplication — skip if an active Signal already covers the same zone
  8. If score >= signal_min_score and no macro gate → create Signal in DB
  9. Always persist AnalysisState so repeated calls on the same candle are no-ops

The caller is responsible for committing the session; this function only flushes
so that the returned Signal already has an assigned ID.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiogram import Bot

import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis import indicators, smc
from app.analysis.scoring import detect_structure_direction, score_setup
from app.collectors.derivatives import get_latest_derivatives, get_prev_derivatives
from app.config import settings
from app.db.models import AnalysisState, Candle, MarketSentiment, NewsItem, Signal
from app.db.session import AsyncSessionLocal

log = structlog.get_logger(__name__)

_MIN_CANDLES = 50     # guard against sparse DB during early onboarding


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _load_candles(
    symbol: str,
    timeframe: str,
    session: AsyncSession,
    limit: int,
) -> pd.DataFrame:
    """Return a time-ordered OHLCV DataFrame (oldest first) from the candles table."""
    result = await session.execute(
        select(Candle)
        .where(Candle.symbol == symbol, Candle.timeframe == timeframe)
        .order_by(Candle.ts.desc())
        .limit(limit)
    )
    rows = list(reversed(result.scalars().all()))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        [{"open": r.o, "high": r.h, "low": r.l, "close": r.c, "volume": r.v} for r in rows],
        index=pd.DatetimeIndex([r.ts for r in rows]),
    )
    return df


def _avg_sentiment(news: list[NewsItem]) -> float | None:
    """Importance-weighted average sentiment across a news list."""
    scored = [
        (n.sentiment, n.importance)
        for n in news
        if n.sentiment is not None and n.importance
    ]
    if not scored:
        return None
    total_w = sum(w for _, w in scored)
    if total_w == 0:
        return None
    return sum(s * w for s, w in scored) / total_w


async def _find_duplicate_signal(
    symbol: str,
    timeframe: str,
    side: str,
    entry_low: float,
    entry_high: float,
    session: AsyncSession,
) -> Signal | None:
    """Return an existing active Signal whose entry zone overlaps [entry_low, entry_high].

    Overlap condition: existing.entry_low <= new_entry_high
                   AND existing.entry_high >= new_entry_low
    """
    result = await session.execute(
        select(Signal).where(
            Signal.symbol == symbol,
            Signal.timeframe == timeframe,
            Signal.side == side,
            Signal.status == "active",
            Signal.entry_low <= entry_high,
            Signal.entry_high >= entry_low,
        )
    )
    return result.scalar_one_or_none()


async def _update_analysis_state(
    symbol: str,
    timeframe: str,
    candle_ts: datetime,
    session: AsyncSession,
) -> None:
    """Upsert the AnalysisState row for (symbol, timeframe)."""
    state_result = await session.execute(
        select(AnalysisState).where(
            AnalysisState.symbol == symbol,
            AnalysisState.timeframe == timeframe,
        )
    )
    state = state_result.scalar_one_or_none()
    if state is None:
        session.add(AnalysisState(symbol=symbol, timeframe=timeframe, last_candle_ts=candle_ts))
    else:
        state.last_candle_ts = candle_ts


# ── Core pipeline ─────────────────────────────────────────────────────────────

async def analyze_symbol_on_close(
    symbol: str,
    trigger_timeframe: str,
    session: AsyncSession,
) -> Signal | None:
    """Run the full analysis pipeline for one symbol+timeframe on candle close.

    Idempotent: returns None immediately when the latest candle was already
    analysed in a previous call. Always updates AnalysisState on completion so
    the caller can unconditionally commit without re-running logic next cycle.

    Returns the new Signal (flushed but not committed) or None.
    """
    # 1. Load candles ─────────────────────────────────────────────────────────
    limit    = settings.analysis_candle_limit
    ctx_df   = await _load_candles(symbol, "4h",              session, limit)
    entry_df = await _load_candles(symbol, trigger_timeframe, session, limit)

    if len(ctx_df) < _MIN_CANDLES or len(entry_df) < _MIN_CANDLES:
        log.debug(
            "engine_insufficient_candles",
            symbol=symbol, tf=trigger_timeframe,
            ctx=len(ctx_df), entry=len(entry_df),
        )
        return None

    # 2. Idempotency check — skip if we already analysed this candle ──────────
    latest_ts_raw = entry_df.index[-1]
    latest_candle_ts: datetime = latest_ts_raw.to_pydatetime()
    if latest_candle_ts.tzinfo is None:
        latest_candle_ts = latest_candle_ts.replace(tzinfo=UTC)

    state_result = await session.execute(
        select(AnalysisState).where(
            AnalysisState.symbol == symbol,
            AnalysisState.timeframe == trigger_timeframe,
        )
    )
    state = state_result.scalar_one_or_none()
    if state is not None:
        # SQLite may return naive datetimes even from DateTime(timezone=True) columns.
        state_ts = state.last_candle_ts
        if state_ts.tzinfo is None:
            state_ts = state_ts.replace(tzinfo=UTC)
        already_analysed = state_ts >= latest_candle_ts
    else:
        already_analysed = False
    if already_analysed:
        log.debug(
            "engine_already_analysed",
            symbol=symbol, tf=trigger_timeframe, ts=latest_candle_ts,
        )
        return None

    # 3. SMC analysis — include_mitigated=True so LIQ_SWEEP zones are visible ─
    try:
        zones_ctx   = smc.analyze(ctx_df,   confirmed_only=True, include_mitigated=True)
        zones_entry = smc.analyze(entry_df, confirmed_only=True, include_mitigated=True)
    except Exception as exc:
        log.warning("engine_smc_failed", symbol=symbol, error=str(exc))
        return None

    # 4. Trade direction from 4h structural context ───────────────────────────
    side = detect_structure_direction(zones_ctx)
    if side is None:
        log.debug("engine_no_direction", symbol=symbol)
        return None

    # 5. Supporting data ───────────────────────────────────────────────────────
    derivatives, prev_derivatives = await asyncio.gather(
        get_latest_derivatives(symbol, session),
        get_prev_derivatives(symbol, session),
    )

    cutoff = datetime.now(UTC) - timedelta(hours=4)
    news_result = await session.execute(
        select(NewsItem).where(
            NewsItem.published_at >= cutoff,
            NewsItem.sentiment.is_not(None),
        )
    )
    all_recent = news_result.scalars().all()
    relevant = [n for n in all_recent if symbol in (n.symbols or [])]
    avg_sent = _avg_sentiment(relevant)

    # 6. Fear & Greed (informational; macro gate is a future FOMC/CPI calendar) ─
    fg_row = (
        await session.execute(
            select(MarketSentiment).order_by(MarketSentiment.ts.desc()).limit(1)
        )
    ).scalar_one_or_none()
    fear_greed_value = fg_row.fear_greed_value if fg_row else None  # noqa: F841 — future use

    macro_flag = False  # placeholder; implement FOMC/CPI calendar in a later etap

    # 7. ATR from entry timeframe ──────────────────────────────────────────────
    atr_series   = indicators.atr(entry_df)
    last_atr_val = atr_series.iloc[-1]
    if pd.isna(last_atr_val) or float(last_atr_val) <= 0:
        log.warning("engine_atr_invalid", symbol=symbol, atr=last_atr_val)
        await _update_analysis_state(symbol, trigger_timeframe, latest_candle_ts, session)
        await session.flush()
        return None
    current_atr = float(last_atr_val)

    # 8. Score ────────────────────────────────────────────────────────────────
    current_price = float(entry_df["close"].iloc[-1])

    result = score_setup(
        symbol=symbol,
        side=side,
        current_price=current_price,
        zones_entry=zones_entry,
        zones_ctx=zones_ctx,
        atr=current_atr,
        derivatives=derivatives,
        prev_derivatives=prev_derivatives,
        avg_sentiment=avg_sent,
    )
    if result is None:
        log.debug("engine_no_valid_setup", symbol=symbol, side=side)
        await _update_analysis_state(symbol, trigger_timeframe, latest_candle_ts, session)
        await session.flush()
        return None

    # 9. Threshold and macro gate ──────────────────────────────────────────────
    if result.score < settings.signal_min_score:
        log.debug(
            "engine_score_below_threshold",
            symbol=symbol, score=result.score, threshold=settings.signal_min_score,
        )
        await _update_analysis_state(symbol, trigger_timeframe, latest_candle_ts, session)
        await session.flush()
        return None

    if macro_flag:
        log.info("engine_macro_gate_active", symbol=symbol)
        await _update_analysis_state(symbol, trigger_timeframe, latest_candle_ts, session)
        await session.flush()
        return None

    # 10. Deduplication ────────────────────────────────────────────────────────
    duplicate = await _find_duplicate_signal(
        symbol, trigger_timeframe, result.side,
        result.entry_low, result.entry_high, session,
    )
    if duplicate is not None:
        log.debug(
            "engine_duplicate_signal",
            symbol=symbol, side=result.side, existing_id=duplicate.id,
        )
        await _update_analysis_state(symbol, trigger_timeframe, latest_candle_ts, session)
        await session.flush()
        return None

    # 11. Persist signal ───────────────────────────────────────────────────────
    signal = Signal(
        symbol=symbol,
        side=result.side,
        timeframe=trigger_timeframe,
        score=result.score,
        entry_low=result.entry_low,
        entry_high=result.entry_high,
        sl=result.sl,
        tp1=result.tp1,
        tp2=result.tp2,
        rr=result.rr,
        factors=result.factors,
        zones=result.zones,
        status="active",
    )
    session.add(signal)
    await _update_analysis_state(symbol, trigger_timeframe, latest_candle_ts, session)
    await session.flush()   # assigns Signal.id and persists AnalysisState in this tx
    log.info(
        "engine_signal_created",
        symbol=symbol, side=side, score=result.score, signal_id=signal.id,
    )
    return signal


# ── Alert dispatch ────────────────────────────────────────────────────────────

_ALERT_CANDLES = 60  # candles loaded for the chart (mirrors alerts._CHART_CANDLES)


async def _dispatch_alert(
    bot: Bot,
    session: AsyncSession,
    signal: Signal,
    symbol: str,
    timeframe: str,
) -> None:
    """Load chart candles and hand off to the alert dispatcher.

    Isolated so that any failure here cannot bubble up and affect AnalysisState
    persistence in the caller.  Lazy-imported to avoid hard coupling between
    app.analysis and app.bot at module load time.
    """
    try:
        from app.bot.alerts import send_signal_alert  # lazy — bot pkg is optional
        candles_df = await _load_candles(symbol, timeframe, session, limit=_ALERT_CANDLES)
        await send_signal_alert(bot, session, signal, candles_df)
    except Exception as exc:
        log.warning("engine_alert_dispatch_failed", symbol=symbol, error=str(exc))


# ── Scheduler job ─────────────────────────────────────────────────────────────

async def run_analysis_cycle(bot: Bot | None = None) -> None:
    """Iterate over all watched symbols and entry timeframes on candle close.

    Each symbol+TF runs in its own session so one failure does not block others.
    Always commits after each call: AnalysisState must persist even when no
    signal was produced, to prevent re-analysis of the same candle next cycle.

    If *bot* is provided, a Telegram alert is dispatched after each new Signal.
    The alert uses the same (committed) session for user queries — safe because
    AsyncSessionLocal is configured with expire_on_commit=False.
    """
    entry_tfs = [tf for tf in settings.watched_timeframes if tf != "4h"]
    for symbol in settings.watched_symbols:
        for tf in entry_tfs:
            try:
                async with AsyncSessionLocal() as session:
                    signal = await analyze_symbol_on_close(symbol, tf, session)
                    await session.commit()   # always: persists AnalysisState + optional Signal
                    if signal is not None:
                        log.info("engine_cycle_signal", symbol=symbol, tf=tf, id=signal.id)
                        if bot is not None:
                            await _dispatch_alert(bot, session, signal, symbol, tf)
            except Exception as exc:
                log.error("engine_cycle_error", symbol=symbol, tf=tf, error=str(exc))
