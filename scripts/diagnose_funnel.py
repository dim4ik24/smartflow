#!/usr/bin/env python3
"""Funnel diagnostic: count walk-forward windows passing each signal filter.

Diagnostic only — does NOT modify any production code.  Reads OHLCV from
the CSV cache produced by backtest.py (no API calls needed).

Usage
-----
    cd backend && python ../scripts/diagnose_funnel.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

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

from app.analysis import indicators, smc  # noqa: E402
from app.analysis.scoring import (  # noqa: E402
    _apply_weights,
    _build_entry_geometry,
    _compute_factors,
    detect_structure_direction,
)
from app.config import settings  # noqa: E402
from app.db.models import DerivativesSnapshot  # noqa: E402

# ── constants — must match backtest.py ───────────────────────────────────────
SYMBOLS    = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
ENTRY_TFS  = ["1h", "15m"]
CONTEXT_TF = "4h"
SMC_WINDOW   = 200
STEP_CANDLES = {"1h": 24, "15m": 96}
CACHE_DIR = _SCRIPTS_DIR / "data" / "ohlcv_cache"

MAX_OB_WIDTH_PCT       = settings.score_max_ob_width_pct        # 0.015
MAX_ENTRY_ATR_DISTANCE = settings.score_max_entry_atr_distance  # 3.0
MIN_RR                 = settings.score_min_rr                   # 2.0
MIN_SCORE              = settings.signal_min_score               # 55

# ── funnel step definitions ───────────────────────────────────────────────────
# Each entry: (display_label, attr_name, previous_step_attr | None)
STEPS: list[tuple[str, str, str | None]] = [
    ("Total windows",                     "total",        None),
    ("4h context OK + SMC parsed",        "has_4h",       "total"),
    ("Has direction (BOS/CHOCH)",         "has_dir",      "has_4h"),
    ("Has OB in direction",               "has_ob_raw",   "has_dir"),
    (f"OB passes width (≤{MAX_OB_WIDTH_PCT*100:.1f}%)",   "has_ob_width", "has_ob_raw"),
    ("OB passes proximity (±1 ATR)",      "has_ob_prox",  "has_ob_width"),
    (f"OB passes dist (≤{MAX_ENTRY_ATR_DISTANCE:.0f} ATR)", "has_ob_dist", "has_ob_prox"),
    (f"RR ≥ {MIN_RR:.1f}",               "has_rr",       "has_ob_dist"),
    (f"score ≥ {MIN_SCORE}",             "has_score",    "has_rr"),
]

# Short column headers for the per-combo table
STEP_SHORT = ["Total", "4h_OK", "Dir", "OB_raw", "Width", "Prox", "Dist", "RR", "Score"]


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class FunnelStats:
    combo:        str
    total:        int = 0  # all walk-forward positions scanned
    has_4h:       int = 0  # 4h data available AND smc.analyze succeeded
    has_dir:      int = 0  # detect_structure_direction(zones_ctx) != None
    has_ob_raw:   int = 0  # any OB in zones_entry matching side, not mitigated
    has_ob_width: int = 0  # at least one OB passes width guard
    has_ob_prox:  int = 0  # at least one of those also passes proximity check
    has_ob_dist:  int = 0  # at least one of those also passes distance guard
    has_rr:       int = 0  # _build_entry_geometry returned non-None (RR ≥ min_rr)
    has_score:    int = 0  # final score ≥ MIN_SCORE


def _add(a: FunnelStats, b: FunnelStats) -> None:
    for _, attr, _ in STEPS:
        setattr(a, attr, getattr(a, attr) + getattr(b, attr))


# ── core funnel loop ──────────────────────────────────────────────────────────

def _diagnose_combo(
    symbol: str,
    tf: str,
    df_entry: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> FunnelStats:
    fs   = FunnelStats(combo=f"{symbol} {tf}")
    step = STEP_CANDLES[tf]

    for idx in range(SMC_WINDOW, len(df_entry) - 1, step):
        fs.total += 1

        win_entry  = df_entry.iloc[idx - SMC_WINDOW : idx + 1]
        current_ts = win_entry.index[-1]

        win_4h = df_4h[df_4h.index <= current_ts].tail(SMC_WINDOW)
        if len(win_4h) < 50:
            continue

        try:
            zones_ctx   = smc.analyze(win_4h,    confirmed_only=True, include_mitigated=True)
            zones_entry = smc.analyze(win_entry, confirmed_only=True, include_mitigated=True)
        except Exception:
            continue
        fs.has_4h += 1

        side = detect_structure_direction(zones_ctx)
        if side is None:
            continue
        fs.has_dir += 1

        atr_val = indicators.atr(win_entry).iloc[-1]
        if pd.isna(atr_val) or float(atr_val) <= 0:
            continue
        atr   = float(atr_val)
        price = float(win_entry["close"].iloc[-1])

        # Step: any unmitigated OB in trade direction
        raw_obs = [
            z for z in zones_entry
            if z["type"] == "OB"
            and z["direction"] == side
            and not z.get("mitigated", False)
        ]
        if not raw_obs:
            continue
        fs.has_ob_raw += 1

        # Step: width guard — OB span must be ≤ MAX_OB_WIDTH_PCT of price
        width_obs = [
            z for z in raw_obs
            if (z["price_to"] - z["price_from"]) / price <= MAX_OB_WIDTH_PCT
        ]
        if not width_obs:
            continue
        fs.has_ob_width += 1

        # Step: proximity — price must be within 1 ATR of the OB boundaries
        prox_obs = [
            z for z in width_obs
            if z["price_from"] - atr <= price <= z["price_to"] + atr
        ]
        if not prox_obs:
            continue
        fs.has_ob_prox += 1

        # Step: distance guard — OB midpoint within MAX_ENTRY_ATR_DISTANCE × ATR
        dist_obs = [
            z for z in prox_obs
            if abs((z["price_from"] + z["price_to"]) / 2.0 - price)
            <= MAX_ENTRY_ATR_DISTANCE * atr
        ]
        if not dist_obs:
            continue
        fs.has_ob_dist += 1

        # Step: RR ≥ MIN_RR (uses nearest liquidity target if available)
        best_ob = max(
            dist_obs,
            key=lambda z: (z.get("strength", 0.0), z.get("time_from") or ""),
        )
        geom = _build_entry_geometry(
            side, best_ob, atr, zones_entry + zones_ctx, MIN_RR
        )
        if geom is None:
            continue
        fs.has_rr += 1

        # Step: score ≥ MIN_SCORE
        entry_low, entry_high, sl, tp1, tp2, rr = geom
        deriv = DerivativesSnapshot(
            symbol=symbol,
            ts=current_ts.to_pydatetime(),
            funding_rate=None,
            open_interest=None,
            long_short_ratio=None,
        )
        factors = _compute_factors(
            side=side,
            current_price=price,
            atr=atr,
            entry_ob=best_ob,
            zones_entry=zones_entry,
            zones_ctx=zones_ctx,
            derivatives=deriv,
            prev_derivatives=None,
            avg_sentiment=None,
            s=settings,
        )
        if _apply_weights(factors, settings) >= MIN_SCORE:
            fs.has_score += 1

    return fs


# ── output helpers ────────────────────────────────────────────────────────────

def _pct(n: int, denom: int) -> str:
    if denom == 0:
        return "   —  "
    return f"{n / denom * 100:5.1f}%"


def print_results(all_stats: list[FunnelStats]) -> None:
    agg = FunnelStats(combo="TOTAL")
    for s in all_stats:
        _add(agg, s)

    # ── Aggregate funnel ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  AGGREGATE SIGNAL FUNNEL  (all combos)")
    print(f"  width≤{MAX_OB_WIDTH_PCT*100:.1f}%  "
          f"dist≤{MAX_ENTRY_ATR_DISTANCE:.0f}ATR  "
          f"RR≥{MIN_RR:.1f}  score≥{MIN_SCORE}")
    print("=" * 70)
    print(f"\n  {'Step':<38}  {'Count':>7}  {'% prev':>7}  {'% total':>7}")
    print("  " + "─" * 38 + "  " + "─" * 7 + "  " + "─" * 7 + "  " + "─" * 7)

    total_all = agg.total
    for label, attr, prev_attr in STEPS:
        cnt  = getattr(agg, attr)
        prev = getattr(agg, prev_attr) if prev_attr else cnt
        p_prev  = _pct(cnt, prev)  if prev_attr else "      —"
        p_total = _pct(cnt, total_all)
        print(f"  {label:<38}  {cnt:>7}  {p_prev:>7}  {p_total:>7}")

    # ── Per-combo table ───────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PER-COMBO BREAKDOWN  (raw counts at each step)")
    print("─" * 70)

    combo_col = 14
    step_col  = 7
    header = f"  {'Combo':<{combo_col}}"
    for sh in STEP_SHORT:
        header += f"  {sh:>{step_col}}"
    print(header)
    print("  " + "─" * combo_col + (f"  {'─'*step_col}") * len(STEP_SHORT))

    for s in all_stats + [agg]:
        line = f"  {s.combo:<{combo_col}}"
        for _, attr, _ in STEPS:
            cnt = getattr(s, attr)
            line += f"  {cnt:>{step_col}}"
        print(line)

    # ── Per-step drop analysis ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  DROP ANALYSIS  (windows lost at each step, by combo)")
    print("─" * 70)

    attrs = [attr for _, attr, _ in STEPS]
    combo_col2 = 14
    print(f"\n  {'Combo':<{combo_col2}}", end="")
    for _, attr, _prev_attr in STEPS[1:]:
        # label of what this step removes (short)
        sh = STEP_SHORT[attrs.index(attr)]
        print(f"  {sh:>{step_col}}", end="")
    print()
    print("  " + "─" * combo_col2 + (f"  {'─'*step_col}") * (len(STEPS) - 1))

    for s in all_stats + [agg]:
        line = f"  {s.combo:<{combo_col2}}"
        for _, attr, prev_attr in STEPS[1:]:
            if prev_attr is None:
                continue
            dropped = getattr(s, prev_attr) - getattr(s, attr)
            line += f"  {dropped:>{step_col}}"
        print(line)

    print()
    print("  NOTE: Each 'drop' = windows that passed the previous step but not this one.")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nSmartFlow Funnel Diagnostic")
    print("=" * 72)

    df_map: dict[tuple[str, str], pd.DataFrame] = {}
    all_tfs = list(set(ENTRY_TFS + [CONTEXT_TF]))

    print("\nLoading cached OHLCV...")
    for sym in SYMBOLS:
        for tf in all_tfs:
            p = CACHE_DIR / f"{sym.replace('/', '_')}_{tf}.csv"
            if not p.exists():
                print(f"  {sym} {tf}: NOT FOUND — run backtest.py --years 2 first")
                continue
            df = pd.read_csv(str(p), index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            df = df.sort_index()
            print(f"  {sym} {tf}: {len(df)} rows  "
                  f"[{df.index[0].date()} → {df.index[-1].date()}]")
            df_map[(sym, tf)] = df

    if not df_map:
        print("\nERROR: No cached data found. Run backtest.py first.")
        sys.exit(1)

    print("\nRunning funnel analysis (same SMC analysis as backtest — ~30-60 min)...\n")

    all_stats: list[FunnelStats] = []
    for sym in SYMBOLS:
        for tf in ENTRY_TFS:
            if (sym, tf) not in df_map or (sym, CONTEXT_TF) not in df_map:
                print(f"  [{sym} {tf}] SKIP — missing cache data")
                continue
            t0 = time.time()
            print(f"  [{sym} {tf}] scanning...", end="", flush=True)
            fs = _diagnose_combo(
                sym, tf, df_map[(sym, tf)], df_map[(sym, CONTEXT_TF)]
            )
            all_stats.append(fs)
            print(f"  done ({time.time() - t0:.0f}s)")

    print_results(all_stats)


if __name__ == "__main__":
    main()
