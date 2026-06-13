"""Signal analysis engine — orchestrator for the full pipeline (SPEC §5).

``analyze_symbol_on_close`` is called whenever a candle closes:
  1. Load OHLCV from DB (4h context + entry TF)
  2. Run SMC analysis (confirmed_only=True to avoid lookahead bias)
  3. Determine trade side from 4h structural direction
  4. Fetch derivatives and news sentiment
  5. Score via scoring.py
  6. If score >= signal_min_score and no macro gate → create Signal in DB

The caller is responsible for committing the session; this function only flushes
so that the returned Signal already has an assigned ID.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis import indicators, smc
from app.analysis.scoring import detect_structure_direction, score_setup
from app.collectors.derivatives import get_latest_derivatives
from app.config import settings
from app.db.models import Candle, MarketSentiment, NewsItem, Signal
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


# ── Core pipeline ─────────────────────────────────────────────────────────────

async def analyze_symbol_on_close(
    symbol: str,
    trigger_timeframe: str,
    session: AsyncSession,
) -> Signal | None:
    """Run the full analysis pipeline for one symbol+timeframe on candle close.

    Returns the new Signal (flushed but not committed) or None.
    The caller decides whether to commit (real flow) or roll back (tests).
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

    # 2. SMC analysis — include_mitigated=True so LIQ_SWEEP zones are visible ─
    try:
        zones_ctx   = smc.analyze(ctx_df,   confirmed_only=True, include_mitigated=True)
        zones_entry = smc.analyze(entry_df, confirmed_only=True, include_mitigated=True)
    except Exception as exc:
        log.warning("engine_smc_failed", symbol=symbol, error=str(exc))
        return None

    # 3. Trade direction from 4h structural context ───────────────────────────
    side = detect_structure_direction(zones_ctx)
    if side is None:
        log.debug("engine_no_direction", symbol=symbol)
        return None

    # 4. Supporting data ───────────────────────────────────────────────────────
    derivatives = await get_latest_derivatives(symbol, session)

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

    # 5. Fear & Greed (informational; macro gate is a future FOMC/CPI calendar) ─
    fg_row = (
        await session.execute(
            select(MarketSentiment).order_by(MarketSentiment.ts.desc()).limit(1)
        )
    ).scalar_one_or_none()
    fear_greed_value = fg_row.fear_greed_value if fg_row else None  # noqa: F841 — future use

    macro_flag = False  # placeholder; implement FOMC/CPI calendar in a later etap

    # 6. ATR from entry timeframe ──────────────────────────────────────────────
    atr_series   = indicators.atr(entry_df)
    last_atr_val = atr_series.iloc[-1]
    if pd.isna(last_atr_val) or float(last_atr_val) <= 0:
        log.warning("engine_atr_invalid", symbol=symbol, atr=last_atr_val)
        return None
    current_atr = float(last_atr_val)

    # 7. Score ────────────────────────────────────────────────────────────────
    current_price = float(entry_df["close"].iloc[-1])
    result = score_setup(
        symbol=symbol,
        side=side,
        current_price=current_price,
        zones_entry=zones_entry,
        zones_ctx=zones_ctx,
        atr=current_atr,
        derivatives=derivatives,
        avg_sentiment=avg_sent,
    )
    if result is None:
        log.debug("engine_no_valid_setup", symbol=symbol, side=side)
        return None

    # 8. Threshold and macro gate ──────────────────────────────────────────────
    if result.score < settings.signal_min_score:
        log.debug(
            "engine_score_below_threshold",
            symbol=symbol, score=result.score, threshold=settings.signal_min_score,
        )
        return None

    if macro_flag:
        log.info("engine_macro_gate_active", symbol=symbol)
        return None

    # 9. Persist signal ────────────────────────────────────────────────────────
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
    await session.flush()   # assign ID; caller commits or rolls back
    log.info(
        "engine_signal_created",
        symbol=symbol, side=side, score=result.score, signal_id=signal.id,
    )
    return signal


# ── Scheduler job ─────────────────────────────────────────────────────────────

async def run_analysis_cycle() -> None:
    """Iterate over all watched symbols and entry timeframes on candle close.

    Each symbol+TF runs in its own session so one failure does not block others.
    """
    entry_tfs = [tf for tf in settings.watched_timeframes if tf != "4h"]
    for symbol in settings.watched_symbols:
        for tf in entry_tfs:
            try:
                async with AsyncSessionLocal() as session:
                    signal = await analyze_symbol_on_close(symbol, tf, session)
                    if signal:
                        await session.commit()
                        log.info("engine_cycle_signal", symbol=symbol, tf=tf, id=signal.id)
            except Exception as exc:
                log.error("engine_cycle_error", symbol=symbol, tf=tf, error=str(exc))
