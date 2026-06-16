#!/usr/bin/env python3
"""Walk-forward backtest for SmartFlow SMC scoring (Etap 7).

Fetches 2-3 years of historical OHLCV via ccxt (with CSV cache), runs the
same walk-forward signal detection as the live engine, simulates each trade
(entry on zone touch; TP1 at 50% / TP2 at 50%; SL moves to breakeven after
TP1), and reports performance metrics broken down by score bucket and symbol.
Uses vectorbt 1.0.0 for equity-curve statistics (drawdown, Sharpe, Calmar).

Usage
-----
    cd backend && python ../scripts/backtest.py
    cd backend && python ../scripts/backtest.py --years 3 --out ../results/backtest.json

Lookahead guarantee
-------------------
    Scoring uses only candles 0..T (closed).  Trade simulation receives
    df[T+1:T+1+MAX_FILL+MAX_HOLD] — it can never see the signal candle or
    anything before it.  A dedicated pytest test (test_backtest.py) verifies
    this property by appending adversarial candles and asserting invariance.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT        = _SCRIPTS_DIR.parent
_BACKEND     = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from app.analysis import indicators, smc
from app.analysis.scoring import detect_structure_direction, score_setup
from app.config import Settings, settings
from app.db.models import DerivativesSnapshot

# ── Constants ─────────────────────────────────────────────────────────────────

SYMBOLS    = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT",
    "BNB/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT",
]
ENTRY_TFS  = ["1h", "15m"]
CONTEXT_TF = "4h"

SMC_WINDOW   = 200   # rolling window for SMC + indicators
# 4h real-time resolution for both TFs (was 24/96 = 1 day → missed 23/24 candles).
STEP_CANDLES = {"1h": 4, "15m": 16}

# Candles allowed to find zone touch after signal.  Per SPEC §7: signal
# expires 2 h after generation if price never reaches the entry zone.
MAX_FILL_CANDLES = {"1h": 2, "15m": 8}

# Max candles to hold a trade (after fill) waiting for SL or TP.
# Trade is force-closed at mid_entry + unrealised PnL if still open.
MAX_HOLD_CANDLES = {"1h": 200, "15m": 800}  # ~8 days

# Zone deduplication cooldown: a persistent OB zone would fire every step without
# this guard.  One signal per zone per trading day (24 × 1h or 96 × 15m candles).
DEDUP_COOLDOWN = {"1h": 24, "15m": 96}

# Score buckets for the breakdown report.
SCORE_BUCKETS: list[tuple[int, int]] = [(55, 70), (70, 85), (85, 101)]
BUCKET_LABELS: list[str]             = ["55-69", "70-84", "85+"]

CACHE_DIR = _SCRIPTS_DIR / "data" / "ohlcv_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FundingHistory = dict[str, list[tuple[datetime, float]]]

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol:       str
    tf:           str
    score:        int
    bucket:       str
    side:         str
    signal_ts:    pd.Timestamp
    fill_ts:      pd.Timestamp | None  # None = no fill (expired)
    exit_ts:      pd.Timestamp | None
    exit_reason:  str     # "sl" | "tp1_be" | "tp1_tp2" | "no_fill" | "hold_expired"
    r_outcome:    float   # net R (0 if no fill)
    mid_entry:    float
    sl:           float
    tp1:          float
    tp2:          float


@dataclass
class BucketStats:
    bucket:        str
    n_signals:     int = 0
    n_fills:       int = 0
    n_wins:        int = 0          # trades where r_outcome > 0
    total_r:       float = 0.0
    sum_sq_r:      float = 0.0     # for stddev
    worst_r:       float = 0.0
    best_r:        float = 0.0
    wins_r:        float = 0.0     # sum of winning R
    loss_r:        float = 0.0     # sum of losing R (absolute)
    max_drawdown:  float = 0.0     # from vectorbt (percent of capital)
    sharpe:        float | None = None
    calmar:        float | None = None

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_fills if self.n_fills else 0.0

    @property
    def avg_r(self) -> float:
        return self.total_r / self.n_fills if self.n_fills else 0.0

    @property
    def profit_factor(self) -> float:
        return self.wins_r / self.loss_r if self.loss_r else float("inf")

    @property
    def fill_rate(self) -> float:
        return self.n_fills / self.n_signals if self.n_signals else 0.0

# ── OHLCV fetch (paginated, with CSV cache) ────────────────────────────────────

_CCXT_LIMIT = 1000   # candles per ccxt request


def _cache_path(symbol: str, tf: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('/', '_')}_{tf}.csv"


def _load_cache(symbol: str, tf: str) -> pd.DataFrame:
    p = _cache_path(symbol, tf)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df.sort_index()


def _save_cache(symbol: str, tf: str, df: pd.DataFrame) -> None:
    df.to_csv(_cache_path(symbol, tf))


def _tf_to_ms(tf: str) -> int:
    multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    unit = tf[-1]
    num  = int(tf[:-1])
    return num * multipliers[unit]


def fetch_ohlcv(
    symbol: str,
    tf: str,
    since: datetime,
    until: datetime,
) -> pd.DataFrame:
    """Fetch OHLCV from Bybit via ccxt, paginating until *until* is covered.

    Results are returned as a time-ordered DataFrame with tz-naive UTC index.
    """
    import ccxt  # local import — not needed in CI

    opts: dict[str, Any] = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    ex: ccxt.Exchange = (
        ccxt.bybit(opts)
        if settings.collector_exchange == "bybit"
        else ccxt.binance(opts)
    )

    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)
    tf_ms    = _tf_to_ms(tf)
    all_rows: list[list[float]] = []

    while since_ms < until_ms:
        rows = ex.fetch_ohlcv(symbol, tf, since=since_ms, limit=_CCXT_LIMIT)
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        since_ms = last_ts + tf_ms
        # Do NOT break on len(rows) < _CCXT_LIMIT: ccxt may return slightly
        # fewer rows than requested (de-duplication, gaps) even when more data
        # exists.  Termination is handled by `since_ms >= until_ms` above.
        time.sleep(0.25)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_or_fetch(
    symbol: str,
    tf: str,
    years: int,
) -> pd.DataFrame:
    """Return OHLCV, refreshing cache when stale (>24 h old) or incomplete."""
    cached = _load_cache(symbol, tf)
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=years * 365)
    until = datetime.now(timezone.utc).replace(tzinfo=None)

    needs_fetch = (
        cached.empty
        or cached.index[0] > since + timedelta(days=7)  # too short history
        or (until - cached.index[-1]).total_seconds() > 86_400  # stale
    )

    if needs_fetch:
        # Extend cached data forward if we already have some
        fetch_since = (
            since if cached.empty else cached.index[-1].to_pydatetime()
        )
        fresh = fetch_ohlcv(symbol, tf, fetch_since, until)
        if not fresh.empty:
            combined = pd.concat([cached, fresh])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            # Trim to requested window
            combined = combined[combined.index >= since]
            _save_cache(symbol, tf, combined)
            return combined

    return cached[cached.index >= since] if not cached.empty else pd.DataFrame()

# ── Funding rate history ──────────────────────────────────────────────────────

def _fetch_funding_sync(symbol: str) -> list[tuple[datetime, float]]:
    import ccxt

    opts: dict[str, Any] = {"enableRateLimit": True, "options": {"defaultType": "linear"}}
    ex: ccxt.Exchange = (
        ccxt.bybit(opts) if settings.collector_exchange == "bybit" else ccxt.binance(opts)
    )
    contract = f"{symbol}:USDT" if settings.collector_exchange == "bybit" else symbol
    try:
        rows = ex.fetch_funding_rate_history(contract, limit=200)
    except Exception as exc:
        print(f"  WARNING funding fetch failed for {symbol}: {exc}")
        return []
    result = []
    for row in rows or []:
        ts_ms = row.get("timestamp")
        rate  = row.get("fundingRate")
        if ts_ms is not None and rate is not None:
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).replace(tzinfo=None)
            result.append((dt, float(rate)))
    result.sort(key=lambda x: x[0])
    return result


def _lookup_funding(
    history: list[tuple[datetime, float]],
    at: pd.Timestamp,
) -> float | None:
    if not history:
        return None
    at_dt = at.to_pydatetime()
    idx = bisect.bisect_right([h[0] for h in history], at_dt) - 1
    return history[idx][1] if idx >= 0 else None

# ── Trade simulation ──────────────────────────────────────────────────────────

def _simulate_trade(
    df: pd.DataFrame,
    side: str,
    entry_low: float,
    entry_high: float,
    sl: float,
    tp1: float,
    tp2: float,
    tf: str,
) -> tuple[str, float, pd.Timestamp | None, pd.Timestamp | None]:
    """Simulate a single signal trade on *df* (candles AFTER the signal candle).

    The caller MUST pass only df[signal_idx + 1:] to prevent any lookahead.

    Parameters
    ----------
    df:
        OHLCV starting strictly at T+1 (signal candle excluded).
    side:
        "long" or "short".
    entry_low, entry_high:
        OB zone bounds; fill triggers when a candle overlaps the zone.
    sl, tp1, tp2:
        Absolute price levels.  SL is always checked before TP on each candle.
    tf:
        Entry timeframe key ("1h" or "15m").

    Returns
    -------
    exit_reason, r_outcome, fill_ts, exit_ts
        exit_reason: "no_fill" | "sl" | "tp1_be" | "tp1_tp2" | "hold_expired"
        r_outcome: net R-multiple (0 for no_fill)
        fill_ts: timestamp of fill candle or None
        exit_ts: timestamp of exit candle or None
    """
    max_fill = MAX_FILL_CANDLES.get(tf, 2)
    max_hold = MAX_HOLD_CANDLES.get(tf, 200)
    mid_entry = (entry_low + entry_high) / 2.0
    risk = abs(mid_entry - sl)
    if risk <= 0:
        return "no_fill", 0.0, None, None

    # Phase 1: find fill — limit-order semantics: price must reach mid_entry.
    # Long:  price must come DOWN to mid_entry → row["low"]  <= mid_entry.
    # Short: price must come UP   to mid_entry → row["high"] >= mid_entry.
    # The old "zone overlap" check (high>=entry_low AND low<=entry_high) fired
    # when a candle merely clipped the zone edge, creating phantom fills and
    # inflating WR by ~50 pp (confirmed by midR=NO→73% vs midR=YES→23% audit).
    fill_idx: int | None = None
    for i in range(min(max_fill, len(df))):
        row = df.iloc[i]
        if side == "long":
            reached = row["low"] <= mid_entry
        else:
            reached = row["high"] >= mid_entry
        if reached:
            fill_idx = i
            break

    if fill_idx is None:
        return "no_fill", 0.0, None, None

    fill_ts = df.index[fill_idx]

    # Phase 2: simulate after fill — pessimistic (SL checked before TP).
    #
    # Within-bar rule for the fill candle (i == fill_idx):
    #   SL is checked — we may have entered and immediately moved to the stop.
    #   TP is NOT checked — within-bar candle order after entry is unknown;
    #   assuming TP fired before SL on the entry bar would be optimistic.
    # From fill_idx+1 onward both SL and TP are checked normally.
    current_sl = sl
    r_partial  = 0.0        # R already locked from TP1 (50 %)
    tp1_hit    = False

    scan_end = min(fill_idx + 1 + max_hold, len(df))
    for i in range(fill_idx, scan_end):
        row = df.iloc[i]
        is_fill_candle = i == fill_idx

        if side == "long":
            sl_breach = row["low"] <= current_sl
            tp1_level = not tp1_hit and not is_fill_candle and row["high"] >= tp1
            tp2_level = tp1_hit and not is_fill_candle and row["high"] >= tp2
        else:  # short
            sl_breach = row["high"] >= current_sl
            tp1_level = not tp1_hit and not is_fill_candle and row["low"] <= tp1
            tp2_level = tp1_hit and not is_fill_candle and row["low"] <= tp2

        if sl_breach:
            # Full stop: lose 1R. BE stop (after TP1): remaining half exits at
            # entry price → 0 additional PnL (not a loss).
            r_loss = 0.0 if tp1_hit else -1.0
            return (
                "sl" if not tp1_hit else "tp1_be",
                r_partial + r_loss,
                fill_ts,
                df.index[i],
            )

        if tp2_level:
            r2 = abs(tp2 - mid_entry) / risk * 0.5  # remaining 50 % of position
            return "tp1_tp2", r_partial + r2, fill_ts, df.index[i]

        if tp1_level and not tp1_hit:
            tp1_hit    = True
            r_partial  = abs(tp1 - mid_entry) / risk * 0.5
            current_sl = mid_entry  # move stop to breakeven

    # Position still open at end of hold window → close at last close
    last_close = float(df.iloc[scan_end - 1]["close"])
    if side == "long":
        r_rem = (last_close - mid_entry) / risk * (0.5 if tp1_hit else 1.0)
    else:
        r_rem = (mid_entry - last_close) / risk * (0.5 if tp1_hit else 1.0)
    return "hold_expired", r_partial + r_rem, fill_ts, df.index[scan_end - 1]


# ── Walk-forward scan + simulation ────────────────────────────────────────────

def _bucket_label(score: int) -> str:
    for (lo, hi), label in zip(SCORE_BUCKETS, BUCKET_LABELS):
        if lo <= score < hi:
            return label
    return "85+"  # 100 falls here


def scan_and_simulate(
    symbol: str,
    tf: str,
    df_entry: pd.DataFrame,
    df_4h: pd.DataFrame,
    funding_history: FundingHistory,
    s: Settings | None = None,
) -> list[TradeRecord]:
    """Walk-forward scan over *df_entry*; simulate each qualifying signal.

    Only candles 0..T feed the scorer.  Simulation receives df_entry[T+1:].
    This is the core lookahead-safety boundary.

    Parameters
    ----------
    s:
        Optional Settings override (used by proximity_sweep.py to test
        different score_proximity_atr values without reloading data).
        Defaults to the module-level ``settings`` singleton.
    """
    if s is None:
        s = settings
    trades: list[TradeRecord] = []
    step      = STEP_CANDLES[tf]
    min_score = s.signal_min_score

    # Zone deduplication: with small step values the same OB zone can produce
    # a signal on every consecutive step.  Track the candle index when a zone
    # was last signaled; suppress re-signals within DEDUP_COOLDOWN candles.
    # Fingerprint: (side, log-price bucket at 0.25% resolution) — robust
    # across minor SMC zone-edge drift as the rolling window advances.
    _LOG_BUCKET = math.log(1.0025)
    _zone_last: dict[tuple[str, int], int] = {}
    dedup_cd = DEDUP_COOLDOWN[tf]

    for idx in range(SMC_WINDOW, len(df_entry) - 1, step):
        win_entry  = df_entry.iloc[idx - SMC_WINDOW : idx + 1]  # 0..T inclusive
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

        hist_fr = _lookup_funding(funding_history.get(symbol, []), current_ts)
        deriv   = DerivativesSnapshot(
            symbol=symbol,
            ts=current_ts.to_pydatetime(),
            funding_rate=hist_fr,
            open_interest=None,
            long_short_ratio=None,
        )

        result = score_setup(
            symbol=symbol,
            side=side,
            current_price=current_price,
            zones_entry=zones_entry,
            zones_ctx=zones_ctx,
            atr=current_atr,
            derivatives=deriv,
            prev_derivatives=None,
            avg_sentiment=None,
            s=s,
        )

        if result is None or result.score < min_score:
            continue

        # Dedup guard: same OB zone signaled within cooldown window → skip
        mid = (result.entry_low + result.entry_high) / 2.0
        zone_key = (result.side, int(math.log(mid) / _LOG_BUCKET))
        if idx - _zone_last.get(zone_key, -(dedup_cd + 1)) <= dedup_cd:
            continue
        _zone_last[zone_key] = idx

        # ── Simulate trade — only future candles (T+1 onward) ────────────────
        # LOOKAHEAD GUARD: df_entry.iloc[idx + 1:] never includes win_entry.
        df_future = df_entry.iloc[idx + 1 :]

        exit_reason, r_out, fill_ts, exit_ts = _simulate_trade(
            df=df_future,
            side=result.side,
            entry_low=result.entry_low,
            entry_high=result.entry_high,
            sl=result.sl,
            tp1=result.tp1,
            tp2=result.tp2,
            tf=tf,
        )

        trades.append(TradeRecord(
            symbol=symbol,
            tf=tf,
            score=result.score,
            bucket=_bucket_label(result.score),
            side=result.side,
            signal_ts=current_ts,
            fill_ts=fill_ts,
            exit_ts=exit_ts,
            exit_reason=exit_reason,
            r_outcome=r_out,
            mid_entry=(result.entry_low + result.entry_high) / 2.0,
            sl=result.sl,
            tp1=result.tp1,
            tp2=result.tp2,
        ))

    return trades


# ── Statistics ────────────────────────────────────────────────────────────────

def _vbt_portfolio_stats(
    filled_trades: list[TradeRecord],
    df_full: pd.DataFrame,
) -> dict[str, float | None]:
    """Build a vectorbt Portfolio from closed trades and extract drawdown stats.

    Uses sl_stop/tp_stop percentage stops per signal.  TP1 is the primary take-
    profit (vectorbt exits at TP1); TP2 mechanics are captured in R-outcome from
    the manual simulation.  This gives an accurate drawdown and Sharpe from the
    full equity curve.
    """
    if not filled_trades or df_full.empty:
        return {"max_drawdown": 0.0, "sharpe": None, "calmar": None}

    close = df_full["close"].astype(float)
    entries_s  = pd.Series(False, index=close.index)
    exits_s    = pd.Series(False, index=close.index)
    sl_pct_s   = pd.Series(0.0,   index=close.index)
    tp1_pct_s  = pd.Series(0.0,   index=close.index)
    is_short_s = pd.Series(False, index=close.index)

    for t in filled_trades:
        if t.fill_ts not in close.index:
            continue
        ts = t.fill_ts
        entries_s[ts]  = True
        is_short_s[ts] = t.side == "short"
        risk = abs(t.mid_entry - t.sl)
        sl_pct_s[ts]   = risk / t.mid_entry if t.mid_entry > 0 else 0.01
        tp1_dist       = abs(t.tp1 - t.mid_entry)
        tp1_pct_s[ts]  = tp1_dist / t.mid_entry if t.mid_entry > 0 else 0.02

    try:
        import vectorbt as vbt  # lazy — not needed during unit tests

        direction = "both" if is_short_s.any() and (~is_short_s[entries_s]).any() else (
            "shortonly" if is_short_s[entries_s].all() else "longonly"
        )
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries_s & ~is_short_s,
            exits=exits_s,
            short_entries=entries_s & is_short_s,
            short_exits=exits_s,
            sl_stop=sl_pct_s,
            tp_stop=tp1_pct_s,
            init_cash=10_000.0,
            size=1.0,
            size_type="value",
            fees=0.0005,
            freq="1h" if any(t.tf == "1h" for t in filled_trades) else "15min",
        )
        return {
            "max_drawdown": float(pf.max_drawdown()),
            "sharpe":       float(pf.sharpe_ratio()) if pf.sharpe_ratio() is not None else None,
            "calmar":       float(pf.calmar_ratio()) if pf.calmar_ratio() is not None else None,
        }
    except Exception as exc:
        print(f"  WARNING vectorbt portfolio failed: {exc}")
        # Fallback: manual drawdown on R-equity curve
        r_values = [t.r_outcome for t in filled_trades]
        equity = np.cumsum(r_values) + 0.0
        rolling_max = np.maximum.accumulate(equity)
        dd = equity - rolling_max
        return {
            "max_drawdown": float(dd.min()) if len(dd) else 0.0,
            "sharpe":       None,
            "calmar":       None,
        }


def compute_bucket_stats(
    trades: list[TradeRecord],
    df_map: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, BucketStats]:
    """Aggregate trade results into BucketStats per score bucket."""
    stats: dict[str, BucketStats] = {label: BucketStats(bucket=label) for label in BUCKET_LABELS}

    for t in trades:
        bs = stats[t.bucket]
        bs.n_signals += 1
        if t.exit_reason == "no_fill":
            continue
        bs.n_fills += 1
        r = t.r_outcome
        bs.total_r    += r
        bs.sum_sq_r   += r * r
        bs.worst_r     = min(bs.worst_r, r)
        bs.best_r      = max(bs.best_r,  r)
        if r > 0:
            bs.n_wins  += 1
            bs.wins_r  += r
        else:
            bs.loss_r  += abs(r)

    # Compute vectorbt stats per bucket using the first applicable df
    for label, bs in stats.items():
        bucket_trades = [t for t in trades if t.bucket == label and t.fill_ts is not None]
        if not bucket_trades:
            continue

        # Pick the df for the most common symbol+tf in this bucket
        combos: dict[tuple[str, str], int] = defaultdict(int)
        for t in bucket_trades:
            combos[(t.symbol, t.tf)] += 1
        best_key = max(combos, key=lambda k: combos[k])
        df_full  = df_map.get(best_key, pd.DataFrame())

        vbt_stats = _vbt_portfolio_stats(bucket_trades, df_full)
        bs.max_drawdown = vbt_stats.get("max_drawdown", 0.0) or 0.0
        bs.sharpe       = vbt_stats.get("sharpe")
        bs.calmar       = vbt_stats.get("calmar")

    return stats


# ── Report ────────────────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 25) -> str:
    filled = round(max(0.0, min(1.0, pct)) * width)
    return "█" * filled + "░" * (width - filled)


def print_report(
    all_trades: list[TradeRecord],
    bucket_stats: dict[str, BucketStats],
    years: int,
) -> None:
    print("\n" + "=" * 72)
    print("  SMARTFLOW WALK-FORWARD BACKTEST REPORT")
    print(f"  Period: {years} year(s)  |  "
          f"Signals: {len(all_trades)}  |  "
          f"Filled: {sum(1 for t in all_trades if t.fill_ts)}")
    print("=" * 72)

    # ── By score bucket ───────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  METRICS BY SCORE BUCKET")
    print(f"{'─'*72}")
    print(f"  {'Bucket':7s}  {'Signals':>7}  {'Fills':>5}  {'Fill%':>5}  "
          f"{'WinR%':>5}  {'AvgR':>5}  {'PF':>5}  {'MaxDD%':>6}  {'Sharpe':>6}")
    print(f"  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*6}")

    for label in BUCKET_LABELS:
        bs = bucket_stats[label]
        wr  = f"{bs.win_rate * 100:.0f}%"   if bs.n_fills else "—"
        ar  = f"{bs.avg_r:+.2f}"            if bs.n_fills else "—"
        pf  = f"{bs.profit_factor:.2f}"     if bs.n_fills and bs.loss_r else ("∞" if bs.n_fills else "—")
        dd  = f"{bs.max_drawdown * 100:.1f}%" if bs.n_fills else "—"
        sh  = f"{bs.sharpe:.2f}"            if bs.sharpe is not None else "—"
        fr  = f"{bs.fill_rate * 100:.0f}%"
        print(f"  {label:7s}  {bs.n_signals:>7}  {bs.n_fills:>5}  {fr:>5}  "
              f"{wr:>5}  {ar:>5}  {pf:>5}  {dd:>6}  {sh:>6}")

    # ── Score ↔ WR monotonicity ───────────────────────────────────────────────
    active = [(lbl, bucket_stats[lbl]) for lbl in BUCKET_LABELS if bucket_stats[lbl].n_fills >= 5]
    if len(active) >= 2:
        wrs = [(lbl, bs.win_rate * 100) for lbl, bs in active]
        mono = all(wrs[i][1] <= wrs[i+1][1] for i in range(len(wrs)-1))
        arrow = "  →  ".join(f"{lbl} {wr:.0f}%" for lbl, wr in wrs)
        tag = "monotonic ↑ (score predicts WR)" if mono else "non-monotonic (score not predictive)"
        print(f"\n  Score↔WR: {arrow}  [{tag}]")
    else:
        print("\n  Score↔WR: insufficient data (need ≥5 fills per bucket for monotonicity check)")

    # ── By symbol ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  METRICS BY SYMBOL  (filled trades only)")
    print(f"{'─'*72}")
    sym_groups: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in all_trades:
        if t.fill_ts:
            sym_groups[t.symbol].append(t)
    print(f"  {'Symbol':12s}  {'Count':>5}  {'WinR%':>5}  {'AvgR':>5}  {'PF':>5}  BestR   WorstR")
    for sym, trades_s in sorted(sym_groups.items()):
        wr = sum(1 for t in trades_s if t.r_outcome > 0) / len(trades_s) * 100
        ar = sum(t.r_outcome for t in trades_s) / len(trades_s)
        loss_r_ = sum(abs(t.r_outcome) for t in trades_s if t.r_outcome < 0) or 1e-9
        wins_r_ = sum(t.r_outcome for t in trades_s if t.r_outcome > 0)
        pf  = wins_r_ / loss_r_
        br  = max(t.r_outcome for t in trades_s)
        wr_ = min(t.r_outcome for t in trades_s)
        print(f"  {sym:12s}  {len(trades_s):>5}  {wr:4.0f}%  {ar:+.2f}  {pf:.2f}  "
              f"{br:+.2f}R  {wr_:+.2f}R")

    # ── Exit reason distribution ───────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  EXIT REASON DISTRIBUTION  (all signals)")
    print(f"{'─'*72}")
    reasons: dict[str, int] = defaultdict(int)
    for t in all_trades:
        reasons[t.exit_reason] += 1
    for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = cnt / len(all_trades) * 100
        bar = _bar(pct / 100)
        print(f"  {reason:14s}  {bar}  {cnt:4d}  ({pct:.1f}%)")

    # ── hold_expired bias diagnostic ──────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  HOLD_EXPIRED BIAS DIAGNOSTIC")
    print(f"{'─'*72}")
    he_trades = [t for t in all_trades if t.exit_reason == "hold_expired"]
    tp_sl_trades = [t for t in all_trades
                    if t.exit_reason in ("sl", "tp1_be", "tp1_tp2")]
    if he_trades:
        he_avg  = sum(t.r_outcome for t in he_trades) / len(he_trades)
        he_tot  = sum(t.r_outcome for t in he_trades)
        he_pos  = sum(1 for t in he_trades if t.r_outcome > 0) / len(he_trades) * 100
        print(f"  hold_expired trades : {len(he_trades):4d}  "
              f"avg R: {he_avg:+.3f}  total R: {he_tot:+.2f}  "
              f"positive: {he_pos:.0f}%")
        print(f"  (MAX_HOLD {MAX_HOLD_CANDLES['1h']}×1h = "
              f"{MAX_HOLD_CANDLES['1h']}h ≈ "
              f"{MAX_HOLD_CANDLES['1h']//24}d; "
              f"close-at-expiry rewards trend, penalises reversals)")
    else:
        print("  No hold_expired trades.")

    if tp_sl_trades:
        n = len(tp_sl_trades)
        wr  = sum(1 for t in tp_sl_trades if t.r_outcome > 0) / n
        ar  = sum(t.r_outcome for t in tp_sl_trades) / n
        lr  = sum(abs(t.r_outcome) for t in tp_sl_trades if t.r_outcome < 0) or 1e-9
        wr_ = sum(t.r_outcome for t in tp_sl_trades if t.r_outcome > 0)
        pf  = wr_ / lr
        print(f"\n  TP/SL-only  (sl + tp1_be + tp1_tp2, n={n}):")
        print(f"  WinR: {wr*100:.1f}%  |  AvgR: {ar:+.3f}  |  PF: {pf:.2f}")
    else:
        print("\n  No TP/SL-closed trades found.")

    print()


# ── Incremental partial save ──────────────────────────────────────────────────

def _save_partial(
    all_trades: list[TradeRecord],
    completed: list[tuple[str, str]],
    out_path: Path,
) -> None:
    """Persist current trades after each combo so a crash loses at most one combo.

    Written to <out_path>.partial.json alongside the final output.  Contains
    the raw trade list so any post-hoc analysis script can resume from it.
    """
    payload = {
        "partial":       True,
        "updated_at":    datetime.now(timezone.utc).isoformat(),
        "combos_done":   [f"{s} {tf}" for s, tf in completed],
        "total_signals": len(all_trades),
        "total_fills":   sum(1 for t in all_trades if t.fill_ts),
        "trades":        [asdict(t) for t in all_trades],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, default=str), encoding="utf-8")


# ── JSON export ───────────────────────────────────────────────────────────────

def export_json(
    all_trades: list[TradeRecord],
    bucket_stats: dict[str, BucketStats],
    years: int,
    out_path: Path,
) -> None:
    """Export backtest results as JSON consumable by the /stats API endpoint."""

    def _bucket_to_dict(bs: BucketStats) -> dict[str, Any]:
        return {
            "bucket":        bs.bucket,
            "n_signals":     bs.n_signals,
            "n_fills":       bs.n_fills,
            "fill_rate":     round(bs.fill_rate, 4),
            "win_rate":      round(bs.win_rate, 4),
            "avg_r":         round(bs.avg_r, 4),
            "profit_factor": round(bs.profit_factor, 4) if bs.loss_r else None,
            "max_drawdown":  round(bs.max_drawdown, 4),
            "sharpe":        round(bs.sharpe, 4) if bs.sharpe is not None else None,
            "calmar":        round(bs.calmar, 4) if bs.calmar is not None else None,
            "best_r":        round(bs.best_r, 4),
            "worst_r":       round(bs.worst_r, 4),
        }

    # Per-symbol aggregates
    sym_stats: dict[str, Any] = {}
    for sym in SYMBOLS:
        sym_trades = [t for t in all_trades if t.symbol == sym and t.fill_ts]
        if not sym_trades:
            sym_stats[sym] = {"n_fills": 0}
            continue
        wr = sum(1 for t in sym_trades if t.r_outcome > 0) / len(sym_trades)
        ar = sum(t.r_outcome for t in sym_trades) / len(sym_trades)
        loss_r_ = sum(abs(t.r_outcome) for t in sym_trades if t.r_outcome < 0) or 1e-9
        wins_r_ = sum(t.r_outcome for t in sym_trades if t.r_outcome > 0)
        sym_stats[sym] = {
            "n_fills":      len(sym_trades),
            "win_rate":     round(wr, 4),
            "avg_r":        round(ar, 4),
            "profit_factor": round(wins_r_ / loss_r_, 4),
        }

    payload: dict[str, Any] = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "backtest_years":  years,
        "symbols":         SYMBOLS,
        "timeframes":      ENTRY_TFS,
        "signal_min_score": settings.signal_min_score,
        "score_buckets":   BUCKET_LABELS,
        "bucket_stats":    {bs.bucket: _bucket_to_dict(bs) for bs in bucket_stats.values()},
        "symbol_stats":    sym_stats,
        "total_signals":   len(all_trades),
        "total_fills":     sum(1 for t in all_trades if t.fill_ts),
        "exit_reasons":    dict(
            pd.Series([t.exit_reason for t in all_trades]).value_counts()
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  [OK] Results exported → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SmartFlow walk-forward backtest")
    parser.add_argument("--years", type=int, default=2,
                        help="History length in years (default: 2)")
    parser.add_argument("--out",  type=Path,
                        default=_ROOT / "results" / "backtest.json",
                        help="JSON output path")
    args = parser.parse_args()

    print(f"\nSmartFlow backtest  —  {args.years} year(s)  —  score >= {settings.signal_min_score}")
    print("=" * 72)

    # ── Fetch / load OHLCV ───────────────────────────────────────────────────
    print("\nLoading OHLCV data (cache: scripts/data/ohlcv_cache/)...")
    df_map: dict[tuple[str, str], pd.DataFrame] = {}
    tfs_to_fetch = list(set(ENTRY_TFS + [CONTEXT_TF]))

    for sym in SYMBOLS:
        for tf in tfs_to_fetch:
            label = f"  {sym} {tf}"
            t0 = time.time()
            df = load_or_fetch(sym, tf, args.years)
            elapsed = time.time() - t0
            if df.empty:
                print(f"{label}: NO DATA (skipped)")
            else:
                print(f"{label}: {len(df)} rows  "
                      f"[{df.index[0].date()} → {df.index[-1].date()}]  ({elapsed:.1f}s)")
                df_map[(sym, tf)] = df

    # ── Fetch funding rate history ────────────────────────────────────────────
    print("\nFetching funding rate history (~66 days, Bybit)...")
    funding: FundingHistory = {}
    for sym in SYMBOLS:
        rows = _fetch_funding_sync(sym)
        funding[sym] = rows
        if rows:
            dates = [r[0].date() for r in rows]
            print(f"  {sym}: {len(rows)} settlements [{dates[0]} → {dates[-1]}]")
        else:
            print(f"  {sym}: unavailable")

    # ── Walk-forward scan + simulation ───────────────────────────────────────
    n_combos = len(SYMBOLS) * len(ENTRY_TFS)
    print(f"\nScanning (SMC_WINDOW={SMC_WINDOW}, step=4h, {n_combos} symbol×TF combos)...")
    print("Expected runtime: 60-150 min (step=4 for 1h / step=16 for 15m, 8 symbols).\n")
    t_scan_start = time.time()

    all_trades: list[TradeRecord] = []
    completed_combos: list[tuple[str, str]] = []
    partial_path = args.out.with_suffix(".partial.json")
    for sym in SYMBOLS:
        for tf in ENTRY_TFS:
            if (sym, tf) not in df_map or (sym, CONTEXT_TF) not in df_map:
                print(f"  [{sym} {tf}] SKIP — missing data")
                continue

            df_entry = df_map[(sym, tf)]
            df_4h    = df_map[(sym, CONTEXT_TF)]

            t0 = time.time()
            n_windows = max(0, (len(df_entry) - SMC_WINDOW - 1) // STEP_CANDLES[tf])
            print(f"  [{sym} {tf}] ~{n_windows} windows...", end="", flush=True)
            trades = scan_and_simulate(sym, tf, df_entry, df_4h, funding)
            all_trades.extend(trades)
            completed_combos.append((sym, tf))
            fills = sum(1 for t in trades if t.fill_ts)
            print(f"  signals={len(trades)}  fills={fills}  ({time.time()-t0:.0f}s)")
            _save_partial(all_trades, completed_combos, partial_path)

    total_scan = time.time() - t_scan_start
    print(f"\nScan complete in {total_scan:.0f}s  |  "
          f"Total signals: {len(all_trades)}  |  "
          f"Filled: {sum(1 for t in all_trades if t.fill_ts)}")

    # ── Statistics ────────────────────────────────────────────────────────────
    bucket_stats = compute_bucket_stats(all_trades, df_map)

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(all_trades, bucket_stats, args.years)

    # ── Export JSON ───────────────────────────────────────────────────────────
    export_json(all_trades, bucket_stats, args.years, args.out)


if __name__ == "__main__":
    main()
