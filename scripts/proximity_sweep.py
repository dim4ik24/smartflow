#!/usr/bin/env python3
"""Proximity-guard sweep: compare backtest results across score_proximity_atr values.

Loads OHLCV + funding data ONCE, then runs the walk-forward scan three times
(one per proximity value) without re-fetching.  Reports signals, win rate,
profit factor, and score-bucket breakdown for each value.

Usage
-----
    cd backend && python ../scripts/proximity_sweep.py
    cd backend && python ../scripts/proximity_sweep.py --values 1.0 2.0 3.0 5.0
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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

from app.config import settings as _base_settings  # noqa: E402

# Import shared helpers from backtest (data loading, simulation, stats).
# sys.path already includes _BACKEND; backtest.py is in _SCRIPTS_DIR which is
# also importable because we execute from _BACKEND via `python ../scripts/...`.
sys.path.insert(0, str(_SCRIPTS_DIR))
from backtest import (  # noqa: E402
    BUCKET_LABELS,
    CONTEXT_TF,
    ENTRY_TFS,
    SYMBOLS,
    TradeRecord,
    _fetch_funding_sync,
    load_or_fetch,
    scan_and_simulate,
)

YEARS = 2


# ── stats helpers ─────────────────────────────────────────────────────────────

def _compute_simple_stats(
    trades: list[TradeRecord],
) -> dict[str, Any]:
    filled = [t for t in trades if t.fill_ts]
    if not filled:
        return {
            "n_signals": len(trades), "n_fills": 0,
            "win_rate": 0.0, "avg_r": 0.0, "profit_factor": 0.0,
        }
    wins   = [t for t in filled if t.r_outcome > 0]
    losses = [t for t in filled if t.r_outcome <= 0]
    wins_r = sum(t.r_outcome for t in wins)
    loss_r = sum(abs(t.r_outcome) for t in losses)
    return {
        "n_signals":     len(trades),
        "n_fills":       len(filled),
        "win_rate":      len(wins) / len(filled),
        "avg_r":         sum(t.r_outcome for t in filled) / len(filled),
        "profit_factor": wins_r / loss_r if loss_r else float("inf"),
    }


def _compute_bucket_stats(
    trades: list[TradeRecord],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        buckets[t.bucket].append(t)

    result: dict[str, dict[str, Any]] = {}
    for label in BUCKET_LABELS:
        bt = buckets.get(label, [])
        filled = [t for t in bt if t.fill_ts]
        if not filled:
            result[label] = {
                "n_signals": len(bt), "n_fills": 0,
                "win_rate": 0.0, "avg_r": 0.0, "profit_factor": 0.0,
            }
            continue
        wins   = [t for t in filled if t.r_outcome > 0]
        losses = [t for t in filled if t.r_outcome <= 0]
        wins_r = sum(t.r_outcome for t in wins)
        loss_r = sum(abs(t.r_outcome) for t in losses)
        result[label] = {
            "n_signals":     len(bt),
            "n_fills":       len(filled),
            "win_rate":      len(wins) / len(filled),
            "avg_r":         sum(t.r_outcome for t in filled) / len(filled),
            "profit_factor": wins_r / loss_r if loss_r else float("inf"),
        }
    return result


# ── report ────────────────────────────────────────────────────────────────────

def print_report(
    results: list[tuple[float, list[TradeRecord]]],
) -> None:
    prox_values = [pv for pv, _ in results]

    # ── Overall comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  PROXIMITY SWEEP — OVERALL COMPARISON")
    print("=" * 72)
    col = 11
    header = f"  {'Metric':<20}"
    for pv in prox_values:
        header += f"  {f'prox={pv:.1f}':>{col}}"
    print(header)
    print("  " + "─" * 20 + (f"  {'─'*col}") * len(prox_values))

    rows: list[tuple[str, str]] = [
        ("Signals",      "n_signals"),
        ("Filled",       "n_fills"),
        ("Win rate",     "win_rate"),
        ("Avg R",        "avg_r"),
        ("Profit factor","profit_factor"),
    ]
    all_stats = [_compute_simple_stats(trades) for _, trades in results]

    for label, key in rows:
        line = f"  {label:<20}"
        for st in all_stats:
            v = st[key]
            if key == "win_rate":
                line += f"  {v*100:>10.1f}%"
            elif key in ("avg_r", "profit_factor"):
                line += f"  {v:>11.3f}"
            else:
                line += f"  {int(v):>{col}}"
        print(line)

    # ── Per-bucket breakdown ──────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  BREAKDOWN BY SCORE BUCKET")
    print("─" * 72)

    for bucket in BUCKET_LABELS:
        print(f"\n  [{bucket}]")
        bstats = [_compute_bucket_stats(trades)[bucket] for _, trades in results]
        col2 = 11
        hdr = f"    {'Metric':<18}"
        for pv in prox_values:
            hdr += f"  {f'prox={pv:.1f}':>{col2}}"
        print(hdr)
        print("    " + "─" * 18 + (f"  {'─'*col2}") * len(prox_values))
        for label, key in rows:
            line = f"    {label:<18}"
            for bs in bstats:
                v = bs[key]
                if key == "win_rate":
                    line += f"  {v*100:>10.1f}%"
                elif key in ("avg_r", "profit_factor"):
                    line += f"  {v:>11.3f}"
                else:
                    line += f"  {int(v):>{col2}}"
            print(line)

    # ── Per-symbol fill count ─────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  FILLS BY SYMBOL")
    print("─" * 72)
    col3 = 11
    hdr2 = f"  {'Symbol':<14}"
    for pv in prox_values:
        hdr2 += f"  {f'prox={pv:.1f}':>{col3}}"
    print(hdr2)
    print("  " + "─" * 14 + (f"  {'─'*col3}") * len(prox_values))
    for sym in SYMBOLS:
        line = f"  {sym:<14}"
        for _, trades in results:
            n = sum(1 for t in trades if t.fill_ts and t.symbol == sym)
            line += f"  {n:>{col3}}"
        print(line)

    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Proximity guard sweep")
    parser.add_argument(
        "--values", nargs="+", type=float, default=[1.0, 2.0, 3.0],
        help="score_proximity_atr values to sweep (default: 1.0 2.0 3.0)",
    )
    args = parser.parse_args()
    prox_values: list[float] = args.values

    print(f"\nProximity sweep: {prox_values}  ({YEARS} years, BTC/ETH/SOL 1h+15m)")
    print("=" * 72)

    # ── Load data once ────────────────────────────────────────────────────────
    print("\nLoading OHLCV (cache / fetch)...")
    df_map: dict[tuple[str, str], pd.DataFrame] = {}
    all_tfs = list(set(ENTRY_TFS + [CONTEXT_TF]))
    for sym in SYMBOLS:
        for tf in all_tfs:
            t0 = time.time()
            df = load_or_fetch(sym, tf, YEARS)
            if df.empty:
                print(f"  {sym} {tf}: NO DATA")
            else:
                d0, d1 = df.index[0].date(), df.index[-1].date()
                print(f"  {sym} {tf}: {len(df)} rows [{d0} → {d1}]  ({time.time()-t0:.1f}s)")
                df_map[(sym, tf)] = df

    print("\nLoading funding rate history...")
    funding: dict[str, list[tuple]] = {}
    for sym in SYMBOLS:
        rows = _fetch_funding_sync(sym)
        funding[sym] = rows
        print(f"  {sym}: {len(rows)} settlements" if rows else f"  {sym}: unavailable")

    # ── Run sweep ─────────────────────────────────────────────────────────────
    results: list[tuple[float, list[TradeRecord]]] = []

    for pv in prox_values:
        custom_s = _base_settings.model_copy(update={"score_proximity_atr": pv})
        all_trades: list[TradeRecord] = []

        print(f"\n--- proximity={pv:.1f} ---")
        t_start = time.time()
        for sym in SYMBOLS:
            for tf in ENTRY_TFS:
                if (sym, tf) not in df_map or (sym, CONTEXT_TF) not in df_map:
                    continue
                t0 = time.time()
                print(f"  [{sym} {tf}]...", end="", flush=True)
                trades = scan_and_simulate(
                    sym, tf,
                    df_map[(sym, tf)],
                    df_map[(sym, CONTEXT_TF)],
                    funding,
                    s=custom_s,
                )
                all_trades.extend(trades)
                fills = sum(1 for t in trades if t.fill_ts)
                print(f"  signals={len(trades)}  fills={fills}  ({time.time()-t0:.0f}s)")

        total_t = time.time() - t_start
        total_fills = sum(1 for t in all_trades if t.fill_ts)
        print(f"  => {len(all_trades)} signals, {total_fills} fills  ({total_t:.0f}s total)")
        results.append((pv, all_trades))

    print_report(results)


if __name__ == "__main__":
    main()
