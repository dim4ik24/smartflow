#!/usr/bin/env python3
"""First live analysis cycle diagnostic for BTC/USDT, ETH/USDT, SOL/USDT.

Steps
-----
0. Collect live data for the 3 test symbols:
     a) Derivatives round 1  (sync ccxt, avoids aiodns on Windows)
     b) News + Fear & Greed  (httpx, no DNS quirks)
     c) Derivatives round 2  (natural delay via news gives a non-zero dOI window)
   AnalysisState is cleared after collection so the engine re-runs fresh.
1. Backfill candles if DB has < BACKFILL_MIN rows per symbol+TF.
2. Per symbol + entry TF: trace every pipeline gate; print rejection reason
   and full factor table with live derivative values.
3. Run analyze_symbol_on_close for the 3 symbols; persist any signal and
   dispatch a Telegram alert.

Usage
-----
    cd backend && python ../scripts/live_signal_test.py [--force] [--skip-collect]

    --force         Clear AnalysisState before running (auto-set when collecting).
    --skip-collect  Skip step 0 (derivatives + news); useful for repeat runs.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# Reconfigure stdout/stderr to UTF-8 before any app.* import fires the
# smartmoneyconcepts star-emoji side-effect that breaks cp1251 terminals.
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
from app.collectors.news import collect_news
from app.config import settings
from app.db.models import AnalysisState, Candle, DerivativesSnapshot, Signal
from app.db.session import AsyncSessionLocal, Base
from app.db.session import engine as db_engine

log = structlog.get_logger("live_signal_test")

# ── Constants ──────────────────────────────────────────────────────────────────

SYMBOLS: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
ENTRY_TFS: list[str] = [tf for tf in settings.watched_timeframes if tf != "4h"]
CONTEXT_TF = "4h"
ALL_TFS: list[str] = [CONTEXT_TF] + ENTRY_TFS
BACKFILL_COUNT = 300
BACKFILL_MIN = 100
_ENGINE_MIN_CANDLES = 50  # mirrors engine.py._MIN_CANDLES

SEP  = "=" * 64
HSEP = "-" * 64


# ── Formatting ─────────────────────────────────────────────────────────────────

def _p(v: object) -> str:
    if v is None:
        return "None"
    return f"{float(v):.6g}"  # type: ignore[arg-type]


def _yn(flag: object) -> str:
    return "[+]" if flag else "[ ]"


# ── ccxt backfill (sync in executor) ──────────────────────────────────────────

def _fetch_ohlcv_sync(symbol: str, tf: str, count: int) -> list[list[float]]:
    import ccxt
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


# ── Derivatives collection (sync ccxt in executor) ────────────────────────────

def _fetch_derivatives_sync(symbols: list[str]) -> list[tuple]:
    """Fetch funding rate, open interest, L/S ratio via sync ccxt.

    Returns list of (symbol, ts, funding_rate, open_interest, long_short_ratio).

    Bybit specifics (tested against V5 API):
    - Linear perpetuals use symbol format "BTC/USDT:USDT" + defaultType="linear".
      The spot "BTC/USDT" type="future" only exposes OHLCV, not derivative data.
    - fetch_open_interest returns openInterestAmount (base units); openInterestValue is None.
    - fetch_long_short_ratio is not implemented in ccxt for bybit; we call the
      V5 publicGetV5MarketAccountRatio endpoint directly (buy/sell ratios -> L/S ratio).

    Always uses mainnet: testnet does not expose funding/OI/LSR endpoints.
    No orders are placed here -- read-only public data.
    """
    import ccxt

    # "linear" = USDT-margined perpetuals on Bybit
    opts: dict = {"enableRateLimit": True, "options": {"defaultType": "linear"}}
    ex = ccxt.bybit(opts) if settings.collector_exchange == "bybit" else ccxt.binance(opts)
    # intentionally no set_sandbox_mode -- testnet lacks these endpoints
    ex.load_markets()

    results: list[tuple] = []
    for symbol in symbols:
        ts = datetime.now(UTC)
        fr: float | None = None
        oi: float | None = None
        lsr: float | None = None

        # "BTC/USDT" -> "BTC/USDT:USDT" (linear perp) for Bybit; unchanged for Binance
        contract = f"{symbol}:USDT" if settings.collector_exchange == "bybit" else symbol
        # "BTC/USDT" -> "BTCUSDT" for raw V5 calls
        raw_sym = symbol.replace("/", "")

        try:
            raw = ex.fetch_funding_rate(contract)
            if raw:
                fr = float(raw["fundingRate"])
        except Exception:
            pass

        try:
            raw = ex.fetch_open_interest(contract)
            if raw:
                # Bybit V5 returns openInterestAmount (base currency), openInterestValue is None
                val = (
                    raw.get("openInterestAmount")
                    or raw.get("openInterestValue")
                    or raw.get("openInterest")
                )
                oi = float(val) if val is not None else None
        except Exception:
            pass

        try:
            if settings.collector_exchange == "bybit":
                # ccxt's fetch_long_short_ratio is NotSupported for bybit;
                # call V5 account-ratio endpoint directly.
                resp = ex.publicGetV5MarketAccountRatio({
                    "category": "linear",
                    "symbol": raw_sym,
                    "period": "5min",
                    "limit": 1,
                })
                items = (resp.get("result") or {}).get("list") or []
                if items:
                    buy = float(items[0].get("buyRatio", 0))
                    sell = float(items[0].get("sellRatio", 0))
                    lsr = round(buy / sell, 4) if sell > 0 else None
            else:
                raw_list = ex.fetch_long_short_ratio(contract, "5m")
                if isinstance(raw_list, list) and raw_list:
                    lsr = float(raw_list[-1]["longShortRatio"])
        except Exception:
            pass

        results.append((symbol, ts, fr, oi, lsr))
    return results


async def _collect_and_store_derivatives(symbols: list[str], round_label: str) -> int:
    """Run one derivatives collection round and persist to DB. Returns stored count."""
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, _fetch_derivatives_sync, symbols)

    snapshots = [
        DerivativesSnapshot(
            symbol=sym, ts=ts,
            funding_rate=fr, open_interest=oi, long_short_ratio=lsr,
        )
        for sym, ts, fr, oi, lsr in rows
        if fr is not None or oi is not None or lsr is not None
    ]

    for sym, ts, fr, oi, lsr in rows:
        tag = "[ok]" if (fr is not None or oi is not None or lsr is not None) else "[skip - all None]"
        print(
            f"  {round_label}  {sym:12s}  "
            f"funding={_p(fr):>10s}  OI={_p(oi):>14s}  L/S={_p(lsr):>6s}  {tag}"
        )

    if snapshots:
        async with AsyncSessionLocal() as session:
            session.add_all(snapshots)
            await session.commit()
    return len(snapshots)


# ── Per-symbol diagnostic ──────────────────────────────────────────────────────

async def _diagnose_pair(symbol: str, tf: str, session) -> None:
    print(f"\n{HSEP}")
    print(f"  {symbol}  |  {tf}")
    print(HSEP)

    # 1. Candle count
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

    # 2. Idempotency
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

    # 3. SMC
    try:
        zones_ctx = smc.analyze(ctx_df, confirmed_only=True, include_mitigated=True)
        zones_entry = smc.analyze(entry_df, confirmed_only=True, include_mitigated=True)
    except Exception as exc:
        print(f"  [REJECT] engine_smc_failed: {exc}")
        return

    zone_counts: dict[str, int] = {}
    for z in zones_ctx + zones_entry:
        zone_counts[z["type"]] = zone_counts.get(z["type"], 0) + 1
    zone_str = "  ".join(f"{k}:{v}" for k, v in sorted(zone_counts.items())) or "(none)"
    print(f"  Zones   4h:{len(zones_ctx)}  {tf}:{len(zones_entry)}   {zone_str}")

    # 4. Direction
    side = detect_structure_direction(zones_ctx)
    if side is None:
        print("  [REJECT] engine_no_direction -- no BOS/CHOCH in 4h zones")
        return
    print(f"  Direction (4h): {side.upper():5s}  ({'LONG ^' if side == 'long' else 'SHORT v'})")

    # 5. ATR
    atr_series = indicators.atr(entry_df)
    last_atr_raw = atr_series.iloc[-1]
    if pd.isna(last_atr_raw) or float(last_atr_raw) <= 0:
        print(f"  [REJECT] engine_atr_invalid  ATR={last_atr_raw}")
        return
    current_atr = float(last_atr_raw)
    current_price = float(entry_df["close"].iloc[-1])
    print(f"  Price: {_p(current_price)}   ATR(14): {_p(current_atr)}")

    # 6. Derivatives from DB
    derivatives = await get_latest_derivatives(symbol, session)
    prev_derivatives = await get_prev_derivatives(symbol, session)
    if derivatives:
        delta_oi = None
        if (derivatives.open_interest is not None
                and prev_derivatives is not None
                and prev_derivatives.open_interest is not None):
            delta_oi = derivatives.open_interest - prev_derivatives.open_interest
        print(
            f"  Derivatives  funding={_p(derivatives.funding_rate):>10s}"
            f"   OI={_p(derivatives.open_interest):>14s}"
            f"   L/S={_p(derivatives.long_short_ratio):>6s}"
            f"   dOI={_p(delta_oi)}"
        )
    else:
        print("  Derivatives: none in DB  (funding/OI/LSR factors all 0)")

    # 7. Scoring
    result = score_setup(
        symbol=symbol,
        side=side,
        current_price=current_price,
        zones_entry=zones_entry,
        zones_ctx=zones_ctx,
        atr=current_atr,
        derivatives=derivatives,
        prev_derivatives=prev_derivatives,
        avg_sentiment=None,  # Gemini not run in this script
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
        f"  (rate={_p(fac.get('funding_rate'))}, threshold=+/-{settings.score_funding_extreme_threshold})"
        f"\n    {_yn(fac.get('oi_rising'))}  oi_rising          +{settings.score_weight_oi_rising}"
        f"  (dOI={_p(fac.get('delta_oi'))})"
        f"\n    {_yn(fac.get('lsr_confirms'))}  lsr_confirms       +{settings.score_weight_lsr}"
        f"  (l/s={_p(fac.get('long_short_ratio'))})"
        f"\n    {_yn(fac.get('sentiment_agrees'))}  sentiment_agrees   +{settings.score_weight_sentiment}"
        f"  (avg_sentiment=None, Gemini not run)"
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

    # 8. Duplicate check
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


# ── Force-signal: bypass ATR/width guard for test chart generation ─────────────

async def _force_test_signal(symbol: str, tf: str, bot, sess) -> Signal | None:
    """Generate a test signal using relaxed OB distance (50× ATR) for chart demos.

    Uses score_setup with a modified Settings copy so production guards are
    not touched.  Returns the persisted Signal or None if no OB found at all.
    """
    from app.analysis import engine as eng_mod
    from app.config import Settings  # noqa: F401  (used below in type annotation)

    limit = settings.analysis_candle_limit
    ctx_df  = await _load_candles(symbol, CONTEXT_TF, sess, limit)
    entry_df = await _load_candles(symbol, tf, sess, limit)
    if len(ctx_df) < _ENGINE_MIN_CANDLES or len(entry_df) < _ENGINE_MIN_CANDLES:
        print(f"  [force]   {symbol} {tf}: insufficient candles")
        return None

    try:
        zones_ctx   = smc.analyze(ctx_df,   confirmed_only=True, include_mitigated=True)
        zones_entry = smc.analyze(entry_df, confirmed_only=True, include_mitigated=True)
    except Exception as exc:
        print(f"  [force]   {symbol} {tf}: SMC failed: {exc}")
        return None

    side = detect_structure_direction(zones_ctx)
    if side is None:
        print(f"  [force]   {symbol} {tf}: no direction")
        return None

    atr_series   = indicators.atr(entry_df)
    last_atr_raw = atr_series.iloc[-1]
    if pd.isna(last_atr_raw) or float(last_atr_raw) <= 0:
        print(f"  [force]   {symbol} {tf}: ATR invalid")
        return None
    current_atr   = float(last_atr_raw)
    current_price = float(entry_df["close"].iloc[-1])

    derivatives      = await get_latest_derivatives(symbol, sess)
    prev_derivatives = await get_prev_derivatives(symbol, sess)

    # Very relaxed settings: any OB within 50 ATRs, any width up to 50 %
    relaxed = settings.model_copy(
        update={
            "score_max_entry_atr_distance": 50.0,
            "score_max_ob_width_pct":       0.50,
            "signal_min_score":             0,
        }
    )

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
        s=relaxed,
    )
    if result is None:
        print(f"  [force]   {symbol} {tf}: no OB found even with relaxed guards")
        return None

    print(
        f"  [force]   {symbol} {tf}  {result.side.upper()}  score={result.score}\n"
        f"            Entry  {_p(result.entry_low)} – {_p(result.entry_high)}\n"
        f"            SL     {_p(result.sl)}   "
        f"TP1 {_p(result.tp1)}   TP2 {_p(result.tp2)}   R:R {result.rr:.2f}"
    )

    signal = Signal(
        symbol=symbol,
        side=result.side,
        timeframe=tf,
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
    sess.add(signal)
    await sess.flush()  # assigns id
    print(f"  [force]   Signal id={signal.id} persisted.")

    if bot is not None:
        await eng_mod._dispatch_alert(bot, sess, signal, symbol, tf)
        print("  [force]   Alert dispatched to Telegram.")

    return signal


# ── Engine run ────────────────────────────────────────────────────────────────

async def _run_engine(bot, *, force_signal: bool = False) -> None:
    from app.analysis import engine as eng_mod

    generated: bool = False
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
                        generated = True
                    else:
                        print(f"  [none]    no signal  {symbol}  {tf}")
            except Exception as exc:
                print(f"  [ERROR]   {symbol}  {tf}: {exc}")

    if not generated and force_signal:
        print(f"\n{'─'*64}")
        print("No organic signal found — running FORCE-SIGNAL on first viable pair...")
        for symbol in SYMBOLS:
            for tf in ENTRY_TFS:
                try:
                    async with AsyncSessionLocal() as sess:
                        sig = await _force_test_signal(symbol, tf, bot, sess)
                        await sess.commit()
                        if sig is not None:
                            return  # one forced signal is enough
                except Exception as exc:
                    print(f"  [force-err]  {symbol} {tf}: {exc}")


# ── Clear AnalysisState ───────────────────────────────────────────────────────

async def _clear_analysis_state() -> None:
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


# ── Entry point ────────────────────────────────────────────────────────────────

async def main(*, force: bool, skip_collect: bool, force_signal: bool) -> None:
    # Init DB
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("DB tables ready.\n")

    # ── Step 0: Data collection ───────────────────────────────────────────────
    if not skip_collect:
        print(SEP)
        print("STEP 0 -- Data collection")
        print(SEP)

        print("\n  [deriv round 1]  fetching funding/OI/LSR via sync ccxt...")
        n1 = await _collect_and_store_derivatives(SYMBOLS, "  round1")

        print(f"\n  [news]  fetching RSS feeds + Fear & Greed index...")
        await collect_news()
        print("  [news]  done (sentiment=None until Gemini runs; sentiment factor stays 0)")

        print(f"\n  [deriv round 2]  second snapshot for delta-OI calculation...")
        n2 = await _collect_and_store_derivatives(SYMBOLS, "  round2")

        print(f"\n  Derivatives stored: round1={n1}  round2={n2}")

        # Auto-force: fresh data always warrants fresh analysis
        print("\n  Clearing AnalysisState (fresh data -> fresh analysis)...")
        await _clear_analysis_state()
        print("  Done.")
        force = True  # engine step will not be blocked by already_analysed

    elif force:
        print("[--force] Clearing AnalysisState...")
        await _clear_analysis_state()
        print("Done.\n")

    # ── Step 1: Backfill ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 1 -- Candle backfill")
    print(SEP)
    async with AsyncSessionLocal() as session:
        for symbol in SYMBOLS:
            for tf in ALL_TFS:
                count = await _count_candles(symbol, tf, session)
                prefix = f"  {symbol}  {tf:4s}  {count:4d} candles"
                if count < BACKFILL_MIN:
                    print(
                        prefix + f"  < {BACKFILL_MIN} -> fetching {BACKFILL_COUNT} via ccxt...",
                        end="", flush=True,
                    )
                    n = await _backfill(symbol, tf, session, BACKFILL_COUNT)
                    await session.commit()
                    print(f"  +{n} upserted")
                else:
                    print(prefix + "  ok")

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
        await _run_engine(bot, force_signal=force_signal)
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
        description="SmartFlow -- live signal diagnostic with data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run from backend/:  cd backend && python ../scripts/live_signal_test.py",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear AnalysisState before running (auto-set when collecting)",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip step 0 (derivatives + news); useful for quick repeat runs",
    )
    parser.add_argument(
        "--force-signal",
        action="store_true",
        help=(
            "If no organic signal is found, generate one using relaxed OB guards "
            "(50x ATR distance, 50%% width) to produce a test chart and Telegram alert."
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(
        force=args.force,
        skip_collect=args.skip_collect,
        force_signal=args.force_signal,
    ))
