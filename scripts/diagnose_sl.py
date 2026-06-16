#!/usr/bin/env python3
"""SL placement diagnostic — one scan pass, three stop distances.

Each qualifying signal is simulated with SL_MULTS = [0.5, 1.0, 1.5] × ATR
in a single SMC-analysis pass.  TP targets are held at the scorer's original
levels so only the stop distance varies.  This cleanly isolates whether the
current 0.5-ATR stop is too tight (noise stops) or the system has no edge.

Key output
----------
  - WR / PF / avg-R for each SL multiplier (same signal set)
  - "Noise-SL rate": fraction of 0.5-ATR SL hits that become wins at 1.5 ATR
  - Per score-bucket breakdown
  - Verdict: if PF stays <1 at 1.5 ATR → no edge regardless of stop width

Usage
-----
    cd backend && python ../scripts/diagnose_sl.py
    cd backend && python ../scripts/diagnose_sl.py --years 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT        = _SCRIPTS_DIR.parent
_BACKEND     = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_SCRIPTS_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from backtest import (                                      # noqa: E402
    BUCKET_LABELS,
    CONTEXT_TF,
    DEDUP_COOLDOWN,
    ENTRY_TFS,
    SMC_WINDOW,
    STEP_CANDLES,
    SYMBOLS,
    TradeRecord,
    _bucket_label,
    _fetch_funding_sync,
    _lookup_funding,
    _simulate_trade,
    load_or_fetch,
)
from app.analysis import indicators, smc                    # noqa: E402
from app.analysis.scoring import detect_structure_direction, score_setup  # noqa: E402
from app.config import settings                             # noqa: E402
from app.db.models import DerivativesSnapshot              # noqa: E402

SL_MULTS: list[float] = [0.5, 1.0, 1.5]


# ── Core: one scan pass, N simulations per signal ─────────────────────────────

def _scan_multi_sl(
    symbol: str,
    tf: str,
    df_entry: pd.DataFrame,
    df_4h: pd.DataFrame,
    funding_history: dict[str, list],
    sl_mults: list[float],
) -> dict[float, list[TradeRecord]]:
    """Walk-forward scan: SMC+scoring once per window, simulate per sl_mult.

    TP1/TP2 are fixed at the scorer's original levels (computed with 0.5 ATR).
    Only the SL price differs across sl_mults.  Returns parallel lists — index
    i in each list corresponds to the same signal.
    """
    result_by_mult: dict[float, list[TradeRecord]] = {m: [] for m in sl_mults}
    step      = STEP_CANDLES[tf]
    min_score = settings.signal_min_score

    _LOG_BUCKET = math.log(1.0025)
    _zone_last: dict[tuple[str, int], int] = {}
    dedup_cd   = DEDUP_COOLDOWN[tf]

    for idx in range(SMC_WINDOW, len(df_entry) - 1, step):
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
        )

        if result is None or result.score < min_score:
            continue

        # Dedup: same zone key as backtest.py to get identical signal set
        mid = (result.entry_low + result.entry_high) / 2.0
        zone_key = (result.side, int(math.log(mid) / _LOG_BUCKET))
        if idx - _zone_last.get(zone_key, -(dedup_cd + 1)) <= dedup_cd:
            continue
        _zone_last[zone_key] = idx

        df_future = df_entry.iloc[idx + 1 :]

        # Simulate once per SL level — TP stays at scorer's original price
        for sl_mult in sl_mults:
            sim_sl = (
                result.entry_low  - sl_mult * current_atr if result.side == "long"
                else result.entry_high + sl_mult * current_atr
            )
            exit_reason, r_out, fill_ts, exit_ts = _simulate_trade(
                df=df_future,
                side=result.side,
                entry_low=result.entry_low,
                entry_high=result.entry_high,
                sl=sim_sl,
                tp1=result.tp1,
                tp2=result.tp2,
                tf=tf,
            )
            result_by_mult[sl_mult].append(TradeRecord(
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
                mid_entry=mid,
                sl=sim_sl,
                tp1=result.tp1,
                tp2=result.tp2,
            ))

    return result_by_mult


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _stats(trades: list[TradeRecord]) -> dict[str, Any]:
    filled = [t for t in trades if t.fill_ts]
    if not filled:
        return {"n_signals": len(trades), "n_fills": 0,
                "win_rate": 0.0, "avg_r": 0.0, "profit_factor": 0.0}
    wins   = [t for t in filled if t.r_outcome > 0]
    loss_r = sum(abs(t.r_outcome) for t in filled if t.r_outcome <= 0)
    wins_r = sum(t.r_outcome for t in wins)
    return {
        "n_signals":     len(trades),
        "n_fills":       len(filled),
        "win_rate":      len(wins) / len(filled),
        "avg_r":         sum(t.r_outcome for t in filled) / len(filled),
        "profit_factor": wins_r / loss_r if loss_r else float("inf"),
    }


def _bucket_stats(trades: list[TradeRecord]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        buckets[t.bucket].append(t)
    return {lbl: _stats(buckets.get(lbl, [])) for lbl in BUCKET_LABELS}


# ── Incremental partial save ──────────────────────────────────────────────────

def _save_partial_diag(
    all_by_mult: dict[float, list[TradeRecord]],
    completed: list[tuple[str, str]],
    out_path: Path,
) -> None:
    """Persist multi-SL results after each combo (crash-safe incremental save)."""
    payload = {
        "partial":     True,
        "updated_at":  datetime.now(timezone.utc).isoformat(),
        "combos_done": [f"{s} {tf}" for s, tf in completed],
        "sl_mults":    sorted(all_by_mult.keys()),
        "by_mult": {
            str(m): {
                "n_signals": len(trades),
                "n_fills":   sum(1 for t in trades if t.fill_ts),
                "trades":    [asdict(t) for t in trades],
            }
            for m, trades in all_by_mult.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, default=str), encoding="utf-8")


# ── Noise-SL analysis ─────────────────────────────────────────────────────────

def _noise_sl_report(
    by_mult: dict[float, list[TradeRecord]],
) -> None:
    """Compare SL-0.5 outcomes vs SL-1.5 on the same filled trades.

    'Noise SL': hit SL at 0.5 ATR, would have been profitable at 1.5 ATR.
    """
    baseline = by_mult[0.5]
    wide     = by_mult[1.5]
    assert len(baseline) == len(wide), "Signal lists must be parallel"

    filled_sl_idx = [
        i for i, t in enumerate(baseline)
        if t.fill_ts and t.exit_reason == "sl"
    ]
    n_sl = len(filled_sl_idx)
    if n_sl == 0:
        print("  No SL trades in baseline — noise-SL analysis skipped.")
        return

    rescued_win   = [i for i in filled_sl_idx if wide[i].r_outcome > 0]
    rescued_tp2   = [i for i in filled_sl_idx if wide[i].exit_reason == "tp1_tp2"]
    rescued_tp1be = [i for i in filled_sl_idx if wide[i].exit_reason == "tp1_be"]
    still_sl      = [i for i in filled_sl_idx if wide[i].exit_reason == "sl"]
    to_no_fill    = [i for i in filled_sl_idx if wide[i].exit_reason == "no_fill"]

    noise_rate = len(rescued_win) / n_sl * 100

    print(f"\n  SL hits at 0.5 ATR       : {n_sl}")
    print(f"  → still SL at 1.5 ATR    : {len(still_sl)}  "
          f"({len(still_sl)/n_sl*100:.0f}%)")
    print(f"  → rescued → tp1_tp2      : {len(rescued_tp2)}")
    print(f"  → rescued → tp1_be       : {len(rescued_tp1be)}")
    print(f"  → became no_fill          : {len(to_no_fill)}")
    print(f"\n  Noise-SL rate (→ profit)  : {len(rescued_win)}/{n_sl}  "
          f"= {noise_rate:.1f}%")
    if noise_rate >= 40:
        print("  [!] >40% noise SL — stop is too tight, wider SL is worth testing")
    elif noise_rate >= 20:
        print("  [~] 20-40% noise SL — marginal; wider SL may help slightly")
    else:
        print("  [✓] <20% noise SL — stop width is not the main problem")


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(
    all_by_mult: dict[float, list[TradeRecord]],
) -> None:
    mults = sorted(all_by_mult)
    col = 13

    # ── Overall comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SL DIAGNOSTIC — OVERALL COMPARISON  (same signal set)")
    print("=" * 72)
    hdr = f"  {'Metric':<22}"
    for m in mults:
        hdr += f"  {f'SL×{m:.1f}ATR':>{col}}"
    print(hdr)
    print("  " + "─" * 22 + (f"  {'─'*col}") * len(mults))

    all_st = [_stats(all_by_mult[m]) for m in mults]
    rows: list[tuple[str, str]] = [
        ("Signals",       "n_signals"),
        ("Filled",        "n_fills"),
        ("Win rate",      "win_rate"),
        ("Avg R",         "avg_r"),
        ("Profit factor", "profit_factor"),
    ]
    for label, key in rows:
        line = f"  {label:<22}"
        for st in all_st:
            v = st[key]
            if key == "win_rate":
                line += f"  {v*100:>12.1f}%"
            elif key in ("avg_r", "profit_factor"):
                line += f"  {v:>13.3f}"
            else:
                line += f"  {int(v):>{col}}"
        print(line)

    # ── Verdict ───────────────────────────────────────────────────────────────
    pf_base = all_st[0]["profit_factor"]
    pf_wide = all_st[-1]["profit_factor"]
    delta   = pf_wide - pf_base
    print(f"\n  PF delta (1.5 vs 0.5 ATR): {delta:+.3f}", end="  ")
    if pf_wide >= 1.0:
        print("[IMPROVEMENT → wider SL warrants a full backtest]")
    elif delta >= 0.15:
        print("[marginal improvement — PF still <1]")
    else:
        print("[no meaningful improvement — system lacks edge]")

    # ── Per-bucket breakdown ──────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  BREAKDOWN BY SCORE BUCKET")
    print("─" * 72)
    for bucket in BUCKET_LABELS:
        bstats = [_bucket_stats(all_by_mult[m])[bucket] for m in mults]
        if all(bs["n_fills"] == 0 for bs in bstats):
            continue
        print(f"\n  [{bucket}]")
        hdr2 = f"    {'Metric':<20}"
        for m in mults:
            hdr2 += f"  {f'SL×{m:.1f}':>{col}}"
        print(hdr2)
        print("    " + "─" * 20 + (f"  {'─'*col}") * len(mults))
        for label, key in rows:
            line = f"    {label:<20}"
            for bs in bstats:
                v = bs[key]
                if key == "win_rate":
                    line += f"  {v*100:>12.1f}%"
                elif key in ("avg_r", "profit_factor"):
                    line += f"  {v:>13.3f}"
                else:
                    line += f"  {int(v):>{col}}"
            print(line)

    # ── Exit reason distribution for each mult ────────────────────────────────
    print("\n" + "─" * 72)
    print("  EXIT REASON  (filled trades)")
    print("─" * 72)
    hdr3 = f"  {'Reason':<14}"
    for m in mults:
        hdr3 += f"  {f'SL×{m:.1f}':>{col}}"
    print(hdr3)
    print("  " + "─" * 14 + (f"  {'─'*col}") * len(mults))
    all_reasons = {"sl", "tp1_tp2", "tp1_be", "hold_expired", "no_fill"}
    for reason in sorted(all_reasons):
        line = f"  {reason:<14}"
        for m in mults:
            n = sum(1 for t in all_by_mult[m] if t.exit_reason == reason)
            line += f"  {n:>{col}}"
        print(line)

    # ── Noise-SL analysis ─────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  NOISE-SL ANALYSIS  (0.5 ATR SL hits vs 1.5 ATR outcomes)")
    print("─" * 72)
    _noise_sl_report(all_by_mult)

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SmartFlow SL placement diagnostic")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Subset of symbols to scan, e.g. BTC/USDT ETH/USDT SOL/USDT",
    )
    args = parser.parse_args()

    scan_symbols = args.symbols if args.symbols else SYMBOLS

    print(f"\nSL diagnostic  —  {args.years} year(s)  —  mults={SL_MULTS}"
          f"  symbols={scan_symbols}")
    print("=" * 72)

    # ── Load OHLCV (all from cache — no network calls expected) ──────────────
    print("\nLoading OHLCV (from cache)...")
    df_map: dict[tuple[str, str], pd.DataFrame] = {}
    for sym in scan_symbols:
        for tf in list(set(ENTRY_TFS + [CONTEXT_TF])):
            df = load_or_fetch(sym, tf, args.years)
            if not df.empty:
                df_map[(sym, tf)] = df
                print(f"  {sym} {tf}: {len(df)} rows")

    print("\nLoading funding rates...")
    funding: dict[str, list] = {}
    for sym in scan_symbols:
        rows = _fetch_funding_sync(sym)
        funding[sym] = rows
        print(f"  {sym}: {len(rows)} settlements" if rows else f"  {sym}: unavailable")

    # ── Scan (one pass per symbol×TF, 3 simulations per signal) ──────────────
    n_combos = len(scan_symbols) * len(ENTRY_TFS)
    print(f"\nScanning {n_combos} combos × {len(SL_MULTS)} SL levels in one pass...")
    print(f"Runtime: ~{n_combos * 7} min estimate.\n")

    all_by_mult: dict[float, list[TradeRecord]] = {m: [] for m in SL_MULTS}
    completed_combos: list[tuple[str, str]] = []
    partial_path = _ROOT / "results" / "diagnose_sl.partial.json"
    t_total = time.time()

    for sym in scan_symbols:
        for tf in ENTRY_TFS:
            if (sym, tf) not in df_map or (sym, CONTEXT_TF) not in df_map:
                print(f"  [{sym} {tf}] SKIP — missing data")
                continue
            t0 = time.time()
            print(f"  [{sym} {tf}]...", end="", flush=True)
            by_mult = _scan_multi_sl(
                sym, tf,
                df_map[(sym, tf)],
                df_map[(sym, CONTEXT_TF)],
                funding,
                SL_MULTS,
            )
            for m, trades in by_mult.items():
                all_by_mult[m].extend(trades)
            completed_combos.append((sym, tf))
            n_sig   = len(by_mult[SL_MULTS[0]])
            n_fills = sum(1 for t in by_mult[SL_MULTS[0]] if t.fill_ts)
            print(f"  signals={n_sig}  fills(0.5ATR)={n_fills}  ({time.time()-t0:.0f}s)")
            _save_partial_diag(all_by_mult, completed_combos, partial_path)

    elapsed = time.time() - t_total
    n_sig_total   = len(all_by_mult[SL_MULTS[0]])
    n_fills_total = sum(1 for t in all_by_mult[SL_MULTS[0]] if t.fill_ts)
    print(f"\nDone in {elapsed:.0f}s  |  "
          f"signals={n_sig_total}  fills(0.5ATR)={n_fills_total}")

    print_report(all_by_mult)


if __name__ == "__main__":
    main()
