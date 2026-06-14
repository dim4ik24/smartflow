#!/usr/bin/env python3
"""Scoring calibration diagnostic — read-only, no DB writes.

Walks 2 months of historical OHLCV data for BTC/USDT, ETH/USDT, SOL/USDT
on 1h and 15m timeframes, runs score_setup with PRODUCTION guards at every
24h step, and prints:

  1. Score distribution histogram (buckets 0-20, 20-35, 35-50, 50-70, 70+)
  2. Factor hit rate (% of valid setups where each factor fired)
  3. Summary: total valid setups, days covered, estimated weekly signal rate

Usage
-----
    cd backend && python ../scripts/calibrate_score.py

Runtime: ~5-15 min depending on CPU (SMC analysis × windows × symbols × TFs).
"""
from __future__ import annotations

import asyncio
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import pandas as pd
from sqlalchemy import func, select

from app.analysis import indicators, smc
from app.analysis.scoring import detect_structure_direction, score_setup
from app.config import settings
from app.db.models import Candle
from app.db.session import AsyncSessionLocal, Base
from app.db.session import engine as db_engine

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOLS      = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
ENTRY_TFS    = ["1h", "15m"]
CONTEXT_TF   = "4h"
SMC_WINDOW   = 200    # rolling window for SMC + indicators
STEP_CANDLES = {      # scan step (every ~24h)
    "1h":  24,
    "15m": 96,
}
BACKFILL_LIMITS = {   # max candles to fetch per TF
    "1h":  1000,
    "15m": 1000,
    "4h":  600,
}
MIN_NEEDED = {        # trigger backfill if below this
    "1h":  900,
    "15m": 900,
    "4h":  400,
}
SCORE_BUCKETS = [(0, 20), (20, 35), (35, 50), (50, 70), (70, 101)]
FACTOR_KEYS = [
    "sweep", "ob_retest", "fvg", "structure_aligned",
    "funding_extreme", "oi_rising", "lsr_confirms",
    "sentiment_agrees", "premium_discount",
]
FACTOR_WEIGHTS = {
    "sweep":             settings.score_weight_sweep,
    "ob_retest":         settings.score_weight_ob_retest,
    "fvg":               settings.score_weight_fvg,
    "structure_aligned": settings.score_weight_structure,
    "funding_extreme":   settings.score_weight_funding,
    "oi_rising":         settings.score_weight_oi_rising,
    "lsr_confirms":      settings.score_weight_lsr,
    "sentiment_agrees":  settings.score_weight_sentiment,
    "premium_discount":  settings.score_weight_premium_discount,
}

# ── Result accumulator ────────────────────────────────────────────────────────

@dataclass
class CalibStats:
    symbol: str
    tf: str
    windows_scanned: int = 0
    valid_setups: int = 0          # score_setup returned non-None
    score_dist: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    factor_hits: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    days_covered: float = 0.0
    side_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


# ── ccxt backfill ─────────────────────────────────────────────────────────────

def _fetch_sync(symbol: str, tf: str, count: int) -> list[list[float]]:
    import ccxt
    opts: dict = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    ex = ccxt.bybit(opts) if settings.collector_exchange == "bybit" else ccxt.binance(opts)
    return ex.fetch_ohlcv(symbol, tf, limit=count)  # type: ignore[return-value]


async def _backfill(symbol: str, tf: str, session, count: int) -> int:
    from app.collectors.market_ws import parse_ohlcv_row
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, _fetch_sync, symbol, tf, count)
    candles = [parse_ohlcv_row(row, symbol, tf) for row in rows]
    for c in candles:
        await session.merge(c)
    return len(candles)


async def _count(symbol: str, tf: str, session) -> int:
    res = await session.execute(
        select(func.count()).select_from(Candle)
        .where(Candle.symbol == symbol, Candle.timeframe == tf)
    )
    return res.scalar_one()


async def _load_df(symbol: str, tf: str, session, limit: int) -> pd.DataFrame:
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


# ── Single-combination scan ────────────────────────────────────────────────────

def _scan(
    symbol: str,
    tf: str,
    df_entry: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> CalibStats:
    stats = CalibStats(symbol=symbol, tf=tf)
    step = STEP_CANDLES[tf]
    indices = list(range(SMC_WINDOW, len(df_entry) - 1, step))
    stats.windows_scanned = len(indices)

    if indices:
        ts_first = df_entry.index[SMC_WINDOW]
        ts_last  = df_entry.index[indices[-1]]
        stats.days_covered = (ts_last - ts_first).total_seconds() / 86400

    for idx in indices:
        win_entry = df_entry.iloc[idx - SMC_WINDOW : idx + 1]
        current_ts = win_entry.index[-1]

        win_4h = df_4h[df_4h.index <= current_ts].tail(SMC_WINDOW)
        if len(win_4h) < 50:
            continue

        try:
            zones_ctx   = smc.analyze(win_4h,    confirmed_only=True, include_mitigated=True)
            zones_entry = smc.analyze(win_entry, confirmed_only=True, include_mitigated=True)
        except Exception:
            continue

        side = detect_structure_direction(zones_ctx)
        if side is None:
            continue

        atr_val = indicators.atr(win_entry).iloc[-1]
        if pd.isna(atr_val) or float(atr_val) <= 0:
            continue

        current_atr   = float(atr_val)
        current_price = float(win_entry["close"].iloc[-1])

        result = score_setup(
            symbol=symbol,
            side=side,
            current_price=current_price,
            zones_entry=zones_entry,
            zones_ctx=zones_ctx,
            atr=current_atr,
            derivatives=None,
            prev_derivatives=None,
            avg_sentiment=None,
        )

        if result is None:
            continue

        # Valid setup found
        stats.valid_setups += 1
        stats.side_counts[side] += 1

        # Score bucket
        for lo, hi in SCORE_BUCKETS:
            if lo <= result.score < hi:
                stats.score_dist[f"{lo}-{hi-1}"] += 1
                break

        # Factor hits
        for k in FACTOR_KEYS:
            if result.factors.get(k):
                stats.factor_hits[k] += 1

    return stats


# ── Report printer ────────────────────────────────────────────────────────────

def _fmt_bar(pct: float, width: int = 30) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _print_report(all_stats: list[CalibStats]) -> None:
    print("\n" + "=" * 70)
    print("  SCORING CALIBRATION REPORT")
    print("=" * 70)

    # ── Per-combination table ─────────────────────────────────────────────────
    print(f"\n{'Symbol':10s} {'TF':4s} {'Days':5s} {'Windows':8s} {'Valid':6s}  "
          f"{'Valid%':7s}  Long/Short")
    print("-" * 60)
    total_windows = 0
    total_valid   = 0
    total_days_by_key: dict[tuple, float] = {}
    for s in all_stats:
        key = (s.symbol, s.tf)
        total_windows += s.windows_scanned
        total_valid   += s.valid_setups
        pct = s.valid_setups / s.windows_scanned * 100 if s.windows_scanned else 0
        long_n  = s.side_counts.get("long",  0)
        short_n = s.side_counts.get("short", 0)
        total_days_by_key[key] = s.days_covered
        print(f"  {s.symbol:10s} {s.tf:4s} {s.days_covered:5.1f} "
              f"{s.windows_scanned:8d} {s.valid_setups:6d}  "
              f"{pct:6.1f}%  L:{long_n} S:{short_n}")
    print("-" * 60)
    total_pct = total_valid / total_windows * 100 if total_windows else 0
    print(f"  {'TOTAL':10s} {'':4s} {'':5s} {total_windows:8d} {total_valid:6d}  "
          f"{total_pct:6.1f}%")

    # ── Aggregated score distribution ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  SCORE DISTRIBUTION (all symbols × TFs, valid setups only)")
    print(f"{'─'*70}")
    agg_dist: dict[str, int] = defaultdict(int)
    for s in all_stats:
        for k, v in s.score_dist.items():
            agg_dist[k] += v
    bucket_order = [f"{lo}-{hi-1}" for lo, hi in SCORE_BUCKETS]
    for bucket in bucket_order:
        count = agg_dist.get(bucket, 0)
        pct   = count / total_valid * 100 if total_valid else 0
        bar   = _fmt_bar(pct)
        print(f"  {bucket:7s}  {bar} {count:4d} ({pct:5.1f}%)")

    # Also print per-TF breakdown
    for tf in ENTRY_TFS:
        tf_stats = [s for s in all_stats if s.tf == tf]
        tf_valid  = sum(s.valid_setups for s in tf_stats)
        if tf_valid == 0:
            continue
        tf_dist: dict[str, int] = defaultdict(int)
        for s in tf_stats:
            for k, v in s.score_dist.items():
                tf_dist[k] += v
        print(f"\n  {tf} breakdown ({tf_valid} valid setups):")
        for bucket in bucket_order:
            count = tf_dist.get(bucket, 0)
            pct   = count / tf_valid * 100 if tf_valid else 0
            bar   = _fmt_bar(pct, 20)
            print(f"    {bucket:7s}  {bar} {count:3d} ({pct:5.1f}%)")

    # ── Factor hit rates ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  FACTOR HIT RATES (% of valid setups where factor fired)")
    print(f"{'─'*70}")
    agg_hits: dict[str, int] = defaultdict(int)
    for s in all_stats:
        for k, v in s.factor_hits.items():
            agg_hits[k] += v

    max_name = max(len(k) for k in FACTOR_KEYS)
    print(f"\n  {'Factor':20s}  {'Weight':6s}  {'Hits%':6s}  {'Count':6s}  Bar")
    print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*30}")
    for k in FACTOR_KEYS:
        hits  = agg_hits.get(k, 0)
        pct   = hits / total_valid * 100 if total_valid else 0
        w     = FACTOR_WEIGHTS.get(k, "?")
        bar   = _fmt_bar(pct, 30)
        print(f"  {k:20s}  {str(w):6s}  {pct:5.1f}%  {hits:6d}  {bar}")

    # Factor breakdown by TF
    for tf in ENTRY_TFS:
        tf_stats = [s for s in all_stats if s.tf == tf]
        tf_valid  = sum(s.valid_setups for s in tf_stats)
        if tf_valid == 0:
            continue
        tf_hits: dict[str, int] = defaultdict(int)
        for s in tf_stats:
            for k, v in s.factor_hits.items():
                tf_hits[k] += v
        print(f"\n  {tf} factor rates ({tf_valid} setups):")
        for k in FACTOR_KEYS:
            hits = tf_hits.get(k, 0)
            pct  = hits / tf_valid * 100 if tf_valid else 0
            print(f"    {k:20s}  {pct:5.1f}%  ({hits}/{tf_valid})")

    # ── Threshold analysis ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  THRESHOLD ANALYSIS — estimated weekly signals per symbol×TF")
    print(f"{'─'*70}")

    # Collect all individual scores from per-combo stats (we didn't store them)
    # Approximate from bucket counts
    bucket_mids = {
        "0-19":   10,
        "20-34":  27,
        "35-49":  42,
        "50-69":  60,
        "70-100": 85,
    }

    thresholds = [20, 35, 40, 50, 60, 70]
    for sym in SYMBOLS:
        for tf in ENTRY_TFS:
            s = next((x for x in all_stats if x.symbol == sym and x.tf == tf), None)
            if s is None or s.windows_scanned == 0 or s.days_covered < 1:
                continue
            windows_per_day = 1.0 / (STEP_CANDLES[tf] * (0.25 if tf == "15m" else 1)) * 24
            setups_per_day  = s.valid_setups / s.days_covered if s.days_covered > 0 else 0
            setups_per_week = setups_per_day * 7

            # For each threshold, estimate fraction passing
            above = {}
            for thr in thresholds:
                above_count = 0
                for bucket, cnt in s.score_dist.items():
                    lo = int(bucket.split("-")[0])
                    if lo >= thr:
                        above_count += cnt
                above[thr] = above_count

            row_parts = [f"{sym} {tf} ({s.valid_setups} setups / {s.days_covered:.0f}d):"]
            for thr in thresholds:
                cnt = above.get(thr, 0)
                weekly = cnt / s.days_covered * 7 if s.days_covered > 0 else 0
                row_parts.append(f"  ≥{thr}→{weekly:.1f}/wk")
            print("  " + " ".join(row_parts))

    # ── Calibration recommendation ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  CALIBRATION NOTES")
    print(f"{'─'*70}")
    print(f"  Current threshold:  signal_min_score = {settings.signal_min_score}")
    print(f"  Current weights sum: {sum(FACTOR_WEIGHTS.values())} (max score = 100)")
    print()

    # Factors with very low hit rates are candidates for review
    low_factors = [(k, agg_hits.get(k, 0) / total_valid * 100)
                   for k in FACTOR_KEYS
                   if total_valid > 0 and agg_hits.get(k, 0) / total_valid * 100 < 5.0]
    if low_factors:
        print("  Factors with <5% hit rate (rarely contribute to score):")
        for k, pct in sorted(low_factors, key=lambda x: x[1]):
            w = FACTOR_WEIGHTS[k]
            print(f"    {k:20s}  {pct:4.1f}%  weight={w}")
        print()

    print("  Note: 'sentiment' and 'funding/OI/LSR' factors require live data")
    print("  (derivatives + Gemini) and score 0 in this offline scan.")
    print("  Effective max score in offline scan:",
          100 - FACTOR_WEIGHTS["funding_extreme"]
              - FACTOR_WEIGHTS["oi_rising"]
              - FACTOR_WEIGHTS["lsr_confirms"]
              - FACTOR_WEIGHTS["sentiment_agrees"])
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    t0 = time.perf_counter()

    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # ── Backfill missing data ─────────────────────────────────────────────────
    print("Checking / backfilling candle data...")
    async with AsyncSessionLocal() as session:
        for sym in SYMBOLS:
            for tf in list(ENTRY_TFS) + [CONTEXT_TF]:
                n = await _count(sym, tf, session)
                need = MIN_NEEDED[tf]
                limit = BACKFILL_LIMITS[tf]
                if n < need:
                    print(f"  {sym} {tf}: {n} rows → fetching {limit}...", end="", flush=True)
                    added = await _backfill(sym, tf, session, limit)
                    await session.commit()
                    print(f" +{added} done")
                else:
                    print(f"  {sym} {tf}: {n} rows ok")

    # ── Load all data into memory ─────────────────────────────────────────────
    print("\nLoading candles from DB...")
    dfs: dict[tuple[str, str], pd.DataFrame] = {}
    async with AsyncSessionLocal() as session:
        for sym in SYMBOLS:
            for tf in list(ENTRY_TFS) + [CONTEXT_TF]:
                limit = BACKFILL_LIMITS[tf]
                df = await _load_df(sym, tf, session, limit)
                dfs[(sym, tf)] = df
                print(f"  {sym} {tf}: {len(df)} rows"
                      + (f"  [{df.index[0]} → {df.index[-1]}]" if len(df) > 0 else "  EMPTY"))

    # ── Scan ──────────────────────────────────────────────────────────────────
    print(f"\nScanning (SMC_WINDOW={SMC_WINDOW}, step=24h)...")
    print("This may take 5-20 minutes.\n")

    all_stats: list[CalibStats] = []

    for sym in SYMBOLS:
        for tf in ENTRY_TFS:
            df_entry = dfs[(sym, tf)]
            df_4h    = dfs[(sym, CONTEXT_TF)]
            step = STEP_CANDLES[tf]
            n_windows = max(0, (len(df_entry) - SMC_WINDOW - 1) // step)

            if len(df_entry) < SMC_WINDOW + step:
                print(f"  [{sym} {tf}] SKIP — not enough candles ({len(df_entry)})")
                all_stats.append(CalibStats(symbol=sym, tf=tf))
                continue

            print(f"  [{sym} {tf}] ~{n_windows} windows...", end="", flush=True)
            t1 = time.perf_counter()
            stats = _scan(sym, tf, df_entry, df_4h)
            elapsed = time.perf_counter() - t1
            print(f"  valid={stats.valid_setups}/{stats.windows_scanned}"
                  f"  days={stats.days_covered:.0f}  ({elapsed:.0f}s)")
            all_stats.append(stats)

    total_elapsed = time.perf_counter() - t0
    print(f"\nScan complete in {total_elapsed:.0f}s")

    _print_report(all_stats)


if __name__ == "__main__":
    asyncio.run(main())
