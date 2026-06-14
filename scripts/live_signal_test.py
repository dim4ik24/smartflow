#!/usr/bin/env python3
"""First live analysis cycle diagnostic for BTC/USDT, ETH/USDT, SOL/USDT.

Steps
-----
1. Ensure DB has >= BACKFILL_MIN candles for every symbol + TF; fetch
   missing candles via ccxt REST (sync, in executor) if needed.
2. Per symbol + entry TF: trace every pipeline stage and print exactly
   why a signal was created or rejected.
3. Run the real analysis engine for the three test symbols; any signal
   that passes all gates is persisted and a Telegram alert is dispatched.

Usage (CWD must be backend/ so pydantic-settings finds .env)
-----
    cd backend && python ../scripts/live_signal_test.py [--force]

    --force  Delete AnalysisState for test symbols first, so the engine
             re-analyses the current candle even if it was already seen.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add backend/ to the module search path before any `app.*` import.
# Works when CWD is backend/ (typical) and also when run from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# smartmoneyconcepts prints a star emoji on import; on Windows cp1251
# terminals this raises UnicodeEncodeError before main() even starts.
# Reconfigure stdout/stderr to UTF-8 before any app.* import fires.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import pandas as pd
import structlog
from sqlalchemy import delete, func, select

from app.analysis import indicators, smc
from app.analysis.engine import _load_candles
from app.analysis.scoring import detect_structure_direction, score_setup
from app.bot.bot import create_bot
from app.collectors.derivatives import get_latest_derivatives, get_prev_derivatives
from app.config import settings
from app.db.models import AnalysisState, Candle, Signal
from app.db.session import AsyncSessionLocal, Base
from app.db.session import engine as db_engine  # SQLAlchemy async engine

log = structlog.get_logger("live_signal_test")

# ── Constants ──────────────────────────────────────────────────────────────────

SYMBOLS: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
ENTRY_TFS: list[str] = [tf for tf in settings.watched_timeframes if tf != "4h"]
CONTEXT_TF = "4h"
ALL_TFS: list[str] = [CONTEXT_TF] + ENTRY_TFS   # backfill context first
BACKFILL_COUNT = 300    # candles to request from ccxt
BACKFILL_MIN = 100      # minimum threshold below which we backfill
_ENGINE_MIN_CANDLES = 50  # mirrors engine.py._MIN_CANDLES

SEP  = "=" * 64
HSEP = "-" * 64

# ── Formatting ─────────────────────────────────────────────────────────────────

def _p(v: object) -> str:
    """Format a float with up to 6 significant figures, or 'None'."""
    if v is None:
        return "None"
    return f"{float(v):.6g}"  # type: ignore[arg-type]


def _yn(flag: object) -> str:
    """ASCII yes/no marker for factor table."""
    return "[+]" if flag else "[ ]"


# ── ccxt backfill ──────────────────────────────────────────────────────────────

def _fetch_ohlcv_sync(symbol: str, tf: str, count: int) -> list[list[float]]:
    """Fetch candles via the ccxt SYNC client (uses requests, not aiohttp/aiodns).

    Running synchronous ccxt in run_in_executor avoids the pycares/UDP-DNS
    issue on Windows without touching any private aiohttp internals.
    """
    import ccxt  # sync -- separate from ccxt.async_support used by the collector

    opts: dict = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    ex = ccxt.bybit(opts) if settings.collector_exchange == "bybit" else ccxt.binance(opts)
    if settings.use_testnet:
        ex.set_sandbox_mode(True)
    return ex.fetch_ohlcv(symbol, tf, limit=count)  # type: ignore[return-value]


async def _backfill(symbol: str, tf: str, session, count: int) -> int:
    from app.collectors.market_ws import parse_ohlcv_row

    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, _fetch_ohlcv_sync, symbol, tf, count)
    candles = [parse_ohlcv_row(row, symbol, tf) for row in rows]
    for c in candles:
        await session.merge(c)
    return len(candles)


async def _count_candles(symbol: str, tf: str, session) -> int:
    result = await session.execute(
        select(func.count()).select_from(Candle).where(
            Candle.symbol == symbol, Candle.timeframe == tf
        )
    )
    return result.scalar_one() or 0


# ── Per-symbol diagnostic ──────────────────────────────────────────────────────

async def _diagnose_pair(symbol: str, tf: str, session) -> None:
    """Run every pipeline step in read-only mode and print the decision at each gate."""
    print(f"\n{HSEP}")
    print(f"  {symbol}  |  {tf}")
    print(HSEP)

    # ── 1. Candle count ────────────────────────────────────────────────────────
    limit = settings.analysis_candle_limit
    ctx_df = await _load_candles(symbol, CONTEXT_TF, session, limit)
    entry_df = await _load_candles(symbol, tf, session, limit)

    print(
        f"  Candles   4h:{len(ctx_df):4d}   {tf}:{len(entry_df):4d}"
        f"   (engine min: {_ENGINE_MIN_CANDLES})"
    )
    if len(ctx_df) < _ENGINE_MIN_CANDLES or len(entry_df) < _ENGINE_MIN_CANDLES:
        print("  [REJECT] engine_insufficient_candles")
        return

    # ── 2. Idempotency ────────────────────────────────────────────────────────
    latest_ts = entry_df.index[-1].to_pydatetime()
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.replace(tzinfo=UTC)

    state = (
        await session.execute(
            select(AnalysisState).where(
                AnalysisState.symbol == symbol, AnalysisState.timeframe == tf
            )
        )
    ).scalar_one_or_none()

    already_analysed = False
    if state:
        state_ts = state.last_candle_ts
        if state_ts.tzinfo is None:
            state_ts = state_ts.replace(tzinfo=UTC)
        already_analysed = state_ts >= latest_ts
        print(
            f"  AnalysisState  last={state_ts:%Y-%m-%d %H:%M UTC}"
            f"   candle={latest_ts:%Y-%m-%d %H:%M UTC}"
        )
    else:
        print(f"  AnalysisState  last=None   candle={latest_ts:%Y-%m-%d %H:%M UTC}")

    if already_analysed:
        print("  [NOTE] engine_already_analysed -- engine will skip unless --force was used")

    # ── 3. SMC analysis ───────────────────────────────────────────────────────
    try:
        zones_ctx = smc.analyze(ctx_df, confirmed_only=True, include_mitigated=True)
        zones_entry = smc.analyze(entry_df, confirmed_only=True, include_mitigated=True)
    except Exception as exc:
        print(f"  [REJECT] engine_smc_failed: {exc}")
        return

    zone_counts: dict[str, int] = {}
    for z in zones_ctx + zones_entry:
        zone_counts[z["type"]] = zone_counts.get(z["type"], 0) + 1
    zone_str = "   ".join(f"{k}:{v}" for k, v in sorted(zone_counts.items())) or "(none)"
    print(f"  Zones   4h:{len(zones_ctx)}  {tf}:{len(zones_entry)}   {zone_str}")

    # ── 4. Direction ──────────────────────────────────────────────────────────
    side = detect_structure_direction(zones_ctx)
    if side is None:
        print("  [REJECT] engine_no_direction -- no BOS/CHOCH in 4h zones")
        return
    direction_str = "LONG  ^" if side == "long" else "SHORT v"
    print(f"  Direction (4h): {direction_str}")

    # ── 5. ATR ────────────────────────────────────────────────────────────────
    atr_series = indicators.atr(entry_df)
    last_atr_raw = atr_series.iloc[-1]
    if pd.isna(last_atr_raw) or float(last_atr_raw) <= 0:
        print(f"  [REJECT] engine_atr_invalid  ATR={last_atr_raw}")
        return
    current_atr = float(last_atr_raw)
    current_price = float(entry_df["close"].iloc[-1])
    print(f"  Price: {_p(current_price)}   ATR(14): {_p(current_atr)}")

    # ── 6. Derivatives ────────────────────────────────────────────────────────
    derivatives = await get_latest_derivatives(symbol, session)
    prev_derivatives = await get_prev_derivatives(symbol, session)
    if derivatives:
        print(
            f"  Derivatives  funding={_p(derivatives.funding_rate)}"
            f"   OI={_p(derivatives.open_interest)}"
            f"   L/S={_p(derivatives.long_short_ratio)}"
        )
    else:
        print("  Derivatives: none in DB  (funding/OI/LSR factors all 0)")

    # ── 7. Scoring ────────────────────────────────────────────────────────────
    result = score_setup(
        symbol=symbol,
        side=side,
        current_price=current_price,
        zones_entry=zones_entry,
        zones_ctx=zones_ctx,
        atr=current_atr,
        derivatives=derivatives,
        prev_derivatives=prev_derivatives,
        avg_sentiment=None,
    )
    if result is None:
        print(
            f"  [REJECT] engine_no_valid_setup\n"
            f"    No active OB matching '{side}' within +-ATR of price,"
            f" or nearest liquidity target gives R:R < {settings.score_min_rr}"
        )
        return

    fac = result.factors
    print(
        f"\n  Score: {result.score}/100   threshold: {settings.signal_min_score}"
        f"\n  Factors:"
        f"\n    {_yn(fac.get('sweep'))}  sweep              +{settings.score_weight_sweep}"
        f"\n    {_yn(fac.get('ob_retest'))}  ob_retest          +{settings.score_weight_ob_retest}"
        f"\n    {_yn(fac.get('fvg'))}  fvg                +{settings.score_weight_fvg}"
        f"\n    {_yn(fac.get('structure_aligned'))}  structure_aligned  +{settings.score_weight_structure}"
        f"\n    {_yn(fac.get('funding_extreme'))}  funding_extreme    +{settings.score_weight_funding}"
        f"  (rate={_p(fac.get('funding_rate'))})"
        f"\n    {_yn(fac.get('oi_rising'))}  oi_rising          +{settings.score_weight_oi_rising}"
        f"  (dOI={_p(fac.get('delta_oi'))})"
        f"\n    {_yn(fac.get('lsr_confirms'))}  lsr_confirms       +{settings.score_weight_lsr}"
        f"  (l/s={_p(fac.get('long_short_ratio'))})"
        f"\n    {_yn(fac.get('sentiment_agrees'))}  sentiment_agrees   +{settings.score_weight_sentiment}"
        f"\n    {_yn(fac.get('premium_discount'))}  premium_discount   +{settings.score_weight_premium_discount}"
    )
    print(
        f"\n  Geometry:"
        f"\n    Entry  {_p(result.entry_low)} - {_p(result.entry_high)}"
        f"\n    SL     {_p(result.sl)}"
        f"\n    TP1    {_p(result.tp1)}   TP2 {_p(result.tp2)}"
        f"\n    R:R    {result.rr:.2f}"
    )

    if result.score < settings.signal_min_score:
        print(
            f"\n  [REJECT] engine_score_below_threshold"
            f"  ({result.score} < {settings.signal_min_score})"
        )
        return

    # ── 8. Duplicate check ────────────────────────────────────────────────────
    dup = (
        await session.execute(
            select(Signal).where(
                Signal.symbol == symbol,
                Signal.timeframe == tf,
                Signal.side == result.side,
                Signal.status == "active",
                Signal.entry_low <= result.entry_high,
                Signal.entry_high >= result.entry_low,
            )
        )
    ).scalar_one_or_none()
    if dup is not None:
        print(f"\n  [REJECT] engine_duplicate_signal  existing id={dup.id}")
        return

    if already_analysed:
        print("\n  [PASS] Score OK -- engine will SKIP (already_analysed, re-run with --force)")
    else:
        print("\n  [PASS] Score OK -> engine will create signal + dispatch alert")


# ── Engine run ────────────────────────────────────────────────────────────────

async def _run_engine(bot) -> None:
    """Run analyze_symbol_on_close for SYMBOLS x ENTRY_TFS only."""
    from app.analysis import engine as eng_mod

    for symbol in SYMBOLS:
        for tf in ENTRY_TFS:
            try:
                async with AsyncSessionLocal() as sess:
                    signal = await eng_mod.analyze_symbol_on_close(symbol, tf, sess)
                    await sess.commit()

                    if signal is not None:
                        print(
                            f"  [SIGNAL]  {symbol}  {tf}"
                            f"  id={signal.id}  {signal.side.upper()}  score={signal.score}"
                        )
                        if bot is not None:
                            await eng_mod._dispatch_alert(bot, sess, signal, symbol, tf)
                            print("            Alert dispatched to Telegram.")
                    else:
                        print(f"  [none]    no signal  {symbol}  {tf}")

            except Exception as exc:
                print(f"  [ERROR]   {symbol}  {tf}: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main(*, force: bool) -> None:
    # ── Init DB ───────────────────────────────────────────────────────────────
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("DB tables ready.\n")

    # ── Step 1: Backfill ──────────────────────────────────────────────────────
    print(SEP)
    print("STEP 1 -- Candle backfill")
    print(SEP)
    async with AsyncSessionLocal() as session:
        for symbol in SYMBOLS:
            for tf in ALL_TFS:
                count = await _count_candles(symbol, tf, session)
                prefix = f"  {symbol}  {tf:4s}  {count:4d} candles"
                if count < BACKFILL_MIN:
                    print(prefix + f"  < {BACKFILL_MIN} -> fetching {BACKFILL_COUNT} via ccxt...", end="", flush=True)
                    n = await _backfill(symbol, tf, session, BACKFILL_COUNT)
                    await session.commit()
                    print(f"  +{n} upserted")
                else:
                    print(prefix + "  ok")

    # ── Step 1b: optionally clear AnalysisState ───────────────────────────────
    if force:
        print(f"\n[--force] Clearing AnalysisState for {SYMBOLS}...")
        async with AsyncSessionLocal() as session:
            for symbol in SYMBOLS:
                for tf in ENTRY_TFS:
                    await session.execute(
                        delete(AnalysisState).where(
                            AnalysisState.symbol == symbol,
                            AnalysisState.timeframe == tf,
                        )
                    )
            await session.commit()
        print("  Done -- engine will re-analyse the current candle.\n")

    # ── Step 2: Diagnostic ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 2 -- Pipeline diagnostic (read-only)")
    print(SEP)
    async with AsyncSessionLocal() as session:
        for symbol in SYMBOLS:
            for tf in ENTRY_TFS:
                await _diagnose_pair(symbol, tf, session)

    # ── Step 3: Engine run ────────────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print("STEP 3 -- Engine run: analyze_symbol_on_close")
    print(SEP)
    print()

    bot = create_bot()
    try:
        await _run_engine(bot)
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass
        await db_engine.dispose()

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmartFlow -- first live signal diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run from backend/: cd backend && python ../scripts/live_signal_test.py",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear AnalysisState for test symbols so the engine re-analyses the current candle",
    )
    args = parser.parse_args()
    asyncio.run(main(force=args.force))
