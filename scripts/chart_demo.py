#!/usr/bin/env python3
"""Find a historical BTC/USDT 1h setup where score_setup passes production
guards (score >= 50, fallback >= 35) and save the chart as PNG locally.

Usage
-----
    cd backend && python ../scripts/chart_demo.py

Approach
--------
Walk forward through 3 months of 1h candles with a 24h step.
At each step run SMC + score_setup with PRODUCTION settings (no relaxation).
Stop at the first window where score >= MIN_SCORE (50), falling back to 35.
Save the resulting candlestick PNG to the project root as ``chart_demo.png``.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import matplotlib  # noqa: E402  (must be before any pyplot import)
matplotlib.use("Agg")

import pandas as pd  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app.analysis import indicators, smc  # noqa: E402
from app.analysis.scoring import detect_structure_direction, score_setup  # noqa: E402
from app.bot.alerts import render_signal_chart  # noqa: E402
from app.config import settings  # noqa: E402
from app.db.models import Candle, Signal  # noqa: E402
from app.db.session import AsyncSessionLocal, Base  # noqa: E402
from app.db.session import engine as db_engine  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOL       = "BTC/USDT"
ENTRY_TF     = "1h"
CONTEXT_TF   = "4h"
MIN_SCORE    = 50
FALLBACK_SCORE = 35
SMC_WINDOW   = 200   # candles fed to SMC at each step
CHART_CANDLES = 60   # candles in the saved chart
SCAN_STEP    = 24    # step between windows (24h → ~91 iterations for 3 months)
BACKFILL_1H  = 2200  # ~91 days of 1h candles
BACKFILL_4H  = 600   # ~100 days of 4h candles
MIN_NEEDED   = 2100  # trigger backfill if DB has fewer

OUTPUT_PATH  = Path(__file__).resolve().parent.parent / "chart_demo.png"

# ── ccxt backfill (sync in thread) ────────────────────────────────────────────

def _fetch_ohlcv_sync(symbol: str, tf: str, count: int) -> list[list[float]]:
    import ccxt
    opts: dict = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    ex = ccxt.bybit(opts) if settings.collector_exchange == "bybit" else ccxt.binance(opts)
    return ex.fetch_ohlcv(symbol, tf, limit=count)  # type: ignore[return-value]


async def _backfill(symbol: str, tf: str, session, count: int) -> int:
    from app.collectors.market_ws import parse_ohlcv_row
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, _fetch_ohlcv_sync, symbol, tf, count)
    candles = [parse_ohlcv_row(row, symbol, tf) for row in rows]
    for c in candles:
        await session.merge(c)
    return len(candles)


# ── DB loader (all candles, oldest-first) ─────────────────────────────────────

async def _load_all(symbol: str, tf: str, session, limit: int) -> pd.DataFrame:
    result = await session.execute(
        select(Candle)
        .where(Candle.symbol == symbol, Candle.timeframe == tf)
        .order_by(Candle.ts.asc())
        .limit(limit)
    )
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [{"open": r.o, "high": r.h, "low": r.l, "close": r.c, "volume": r.v}
         for r in rows],
        index=pd.DatetimeIndex([r.ts for r in rows]),
    )


# ── Historical scanner ────────────────────────────────────────────────────────

def _scan_history(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    min_score: int,
) -> tuple[int, object] | None:
    """Walk forward in time; return (candle_index, ScoreResult) on first hit."""
    indices = range(SMC_WINDOW, len(df_1h) - 1, SCAN_STEP)
    print(f"  Scanning {len(indices)} windows (step={SCAN_STEP}h) for score>={min_score}...")

    for step_num, idx in enumerate(indices):
        win_1h = df_1h.iloc[idx - SMC_WINDOW : idx + 1]
        current_ts = win_1h.index[-1]

        # 4h context candles available at this moment
        win_4h = df_4h[df_4h.index <= current_ts].tail(SMC_WINDOW)
        if len(win_4h) < 50:
            continue

        # SMC zones
        try:
            zones_ctx   = smc.analyze(win_4h, confirmed_only=True, include_mitigated=True)
            zones_entry = smc.analyze(win_1h, confirmed_only=True, include_mitigated=True)
        except Exception:
            continue

        # Direction from 4h context
        side = detect_structure_direction(zones_ctx)
        if side is None:
            continue

        # ATR + price at this window
        atr_val = indicators.atr(win_1h).iloc[-1]
        if pd.isna(atr_val) or float(atr_val) <= 0:
            continue
        current_atr   = float(atr_val)
        current_price = float(win_1h["close"].iloc[-1])

        # Score with production settings (no overrides)
        result = score_setup(
            symbol=SYMBOL,
            side=side,
            current_price=current_price,
            zones_entry=zones_entry,
            zones_ctx=zones_ctx,
            atr=current_atr,
            derivatives=None,
            prev_derivatives=None,
            avg_sentiment=None,
        )
        if result is not None and result.score >= min_score:
            print(
                f"  FOUND at window {step_num+1}/{len(indices)}"
                f"  ts={current_ts}  {side.upper()}  score={result.score}"
            )
            return (idx, result)

        # Progress tick every 10 windows
        if (step_num + 1) % 10 == 0:
            pct = (step_num + 1) / len(indices) * 100
            best_label = f"(best so far: no hit)" if True else ""
            print(f"  ... {step_num+1}/{len(indices)} ({pct:.0f}%)  {best_label}")

    return None


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Init DB
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Backfill if sparse
    async with AsyncSessionLocal() as session:
        n1h = (await session.execute(
            select(func.count()).select_from(Candle)
            .where(Candle.symbol == SYMBOL, Candle.timeframe == ENTRY_TF)
        )).scalar_one()
        n4h = (await session.execute(
            select(func.count()).select_from(Candle)
            .where(Candle.symbol == SYMBOL, Candle.timeframe == CONTEXT_TF)
        )).scalar_one()

        if n1h < MIN_NEEDED:
            print(f"  Backfilling {SYMBOL} {ENTRY_TF}: {n1h} candles in DB, "
                  f"fetching {BACKFILL_1H}...")
            n = await _backfill(SYMBOL, ENTRY_TF, session, BACKFILL_1H)
            await session.commit()
            print(f"  Done: {n} rows upserted.")

        if n4h < 500:
            print(f"  Backfilling {SYMBOL} {CONTEXT_TF}: {n4h} candles in DB, "
                  f"fetching {BACKFILL_4H}...")
            n = await _backfill(SYMBOL, CONTEXT_TF, session, BACKFILL_4H)
            await session.commit()
            print(f"  Done: {n} rows upserted.")

    # Load candles
    async with AsyncSessionLocal() as session:
        df_1h = await _load_all(SYMBOL, ENTRY_TF, session, BACKFILL_1H)
        df_4h = await _load_all(SYMBOL, CONTEXT_TF, session, BACKFILL_4H)

    print(f"\n  Loaded: {len(df_1h)} x 1h  |  {len(df_4h)} x 4h")
    if len(df_1h) < SMC_WINDOW + 10:
        print("  ERROR: not enough candles. Run without --skip-collect first.")
        return

    # Scan history
    print()
    hit = _scan_history(df_1h, df_4h, MIN_SCORE)
    threshold_used = MIN_SCORE

    if hit is None:
        print(f"\n  No setup with score>={MIN_SCORE} found in {len(df_1h)} candles.")
        print(f"  Falling back to score>={FALLBACK_SCORE}...")
        threshold_used = FALLBACK_SCORE
        hit = _scan_history(df_1h, df_4h, FALLBACK_SCORE)

    if hit is None:
        print(f"\n  No setup found with score>={FALLBACK_SCORE} in 3 months of BTC 1h data.")
        print("  This suggests the SMC OB-proximity guard is highly selective — "
              "which is correct behaviour (high-quality signals only).")
        return

    idx, result = hit
    hit_ts  = df_1h.index[idx]
    hit_price = float(df_1h["close"].iloc[idx])

    print(f"\n{'='*64}")
    print(f"  Historical setup found  (threshold used: >={threshold_used})")
    print(f"{'='*64}")
    print(f"  Symbol:     {SYMBOL}  {ENTRY_TF}")
    print(f"  Timestamp:  {hit_ts}")
    print(f"  Price:      {hit_price:.2f}")
    print(f"  Side:       {result.side.upper()}")
    print(f"  Score:      {result.score}/100")
    print(f"  Entry zone: {result.entry_low:.2f} – {result.entry_high:.2f}"
          f"  (width {(result.entry_high - result.entry_low) / hit_price * 100:.2f}% of price)")
    print(f"  SL:         {result.sl:.2f}")
    print(f"  TP1:        {result.tp1:.2f}")
    print(f"  TP2:        {result.tp2:.2f}")
    print(f"  R:R:        {result.rr:.2f}")
    print(f"\n  Factors:")
    for k, v in result.factors.items():
        if isinstance(v, bool):
            print(f"    {'[+]' if v else '[ ]'}  {k}")
    print(f"\n  Zones in chart ({len(result.zones or [])} total):")
    for z in (result.zones or []):
        print(f"    {z['type']:10s}  {z['direction']:5s}  "
              f"{z.get('price_from', 0):.2f}–{z.get('price_to', 0):.2f}  "
              f"  time_from={z.get('time_from', '?')}")

    # 60 candles ending at the signal candle
    chart_start = max(0, idx - CHART_CANDLES + 1)
    chart_df = df_1h.iloc[chart_start : idx + 1].copy()
    # Keep index naive — mplfinance on Windows does not handle tz-aware DatetimeIndex
    if chart_df.index.tz is not None:
        chart_df.index = chart_df.index.tz_localize(None)
    print(f"\n  Chart window: {chart_df.index[0]} → {chart_df.index[-1]} ({len(chart_df)} candles)")

    # Render via the production alerts function (used in Telegram alerts)
    signal = Signal(
        symbol=SYMBOL,
        side=result.side,
        timeframe=ENTRY_TF,
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
    signal.id = 0
    OUTPUT_PATH.write_bytes(render_signal_chart(signal, chart_df))
    print(f"  Production chart (Telegram style): {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size//1024} KB)")

    # Also render a zoomed version with lighter background for local inspection
    _render_zoomed(result, chart_df, OUTPUT_PATH.with_name("chart_demo_zoomed.png"))


def _render_zoomed(result, chart_df: pd.DataFrame, output: Path) -> None:
    """Standalone zoomed render with y-axis clipped to the signal price range."""
    import mplfinance as mpf
    import matplotlib.pyplot as plt

    df = chart_df.copy()
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:          # strip tz — mplfinance needs naive index
        df.index = df.index.tz_localize(None)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})

    # Y-limits: span from TP2 to SL with 0.5% padding on each side
    y_lo = result.tp2 * 0.995
    y_hi = result.sl  * 1.005

    style = mpf.make_mpf_style(
        base_mpf_style="yahoo",
        rc={"font.size": 8, "axes.labelsize": 7},
    )

    fig, axes = mpf.plot(
        df, type="candle", style=style,
        title=f"BTC/USDT 1h  |  Score {result.score}/100  (zoomed)",
        volume=True, returnfig=True, figsize=(12, 7),
        ylim=(y_lo, y_hi),
    )
    price_ax = axes[0]

    # Zone bands — use integer x-positions (mplfinance internal coordinate).
    # Using df.index (DatetimeIndex) would expand the x-axis to date numbers
    # (~738 000), compressing all candles to the far left.
    _COLORS: dict[str, tuple[str, float]] = {
        "OB":        ("#1565C0", 0.40),
        "FVG":       ("#E65100", 0.28),
        "LIQ_SWEEP": ("#6A1B9A", 0.22),
    }
    n = len(df)
    x_int = list(range(n))
    for zone in (result.zones or []):
        ztype = str(zone.get("type", ""))
        color, alpha = _COLORS.get(ztype, ("#607D8B", 0.14))
        p_lo = float(zone.get("price_from") or 0.0)
        p_hi = float(zone.get("price_to")   or 0.0)
        if p_lo <= 0 or p_hi <= p_lo:
            continue
        ix_start = 0
        try:
            zone_ts = pd.Timestamp(zone.get("time_from"))
            if zone_ts.tzinfo is not None:
                zone_ts = zone_ts.tz_localize(None)
            ix_start = int(min(df.index.searchsorted(zone_ts), n - 1))
        except Exception:
            ix_start = 0
        mask_int = [i >= ix_start for i in x_int]
        if not any(mask_int):
            continue
        price_ax.fill_between(
            x_int, p_lo, p_hi, where=mask_int,
            facecolor=color, alpha=alpha, linewidth=0, zorder=0,
            label=ztype,
        )
    price_ax.set_xlim(-0.5, n - 0.5)

    # Price level lines + right-edge labels
    _LEVELS = [
        ("entry_low",  "#1565C0", "--", 1.0, "Lo"),
        ("entry_high", "#1565C0", "--", 1.0, "Hi"),
        ("sl",         "#C62828", "-",  1.6, "SL"),
        ("tp1",        "#2E7D32", "-",  1.6, "TP1"),
        ("tp2",        "#1B5E20", "-",  1.6, "TP2"),
    ]
    for attr, color, ls, lw, prefix in _LEVELS:
        price = float(getattr(result, attr))
        price_ax.axhline(y=price, color=color, linestyle=ls, linewidth=lw, alpha=0.95)
        price_ax.annotate(
            f"{prefix} {price:.1f}",
            xy=(1.005, price), xycoords=("axes fraction", "data"),
            color=color, fontsize=7, va="center", clip_on=False,
        )

    # Entry zone fill (full-width, semi-transparent, on top of zones)
    price_ax.axhspan(
        result.entry_low, result.entry_high,
        facecolor="#1565C0", alpha=0.10, linewidth=0, zorder=1,
        label="Entry zone",
    )

    buf = __import__("io").BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    output.write_bytes(buf.read())
    plt.close(fig)
    print(f"  Zoomed chart (local inspect):     {output}  ({output.stat().st_size//1024} KB)")


if __name__ == "__main__":
    asyncio.run(main())
