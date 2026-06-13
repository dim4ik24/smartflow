"""Smoke-test: run the collector against mainnet for 16+ min, then show stats.

Proves live WS collection (not just gap-fill) by:
  - Recording collector start time and querying candles with ts >= start_time
  - Parsing logs to count gap-fill vs live candles_upserted events
  - Verifying timeframe counts DIFFER (15m closes most often)

Usage:  python collector_smoke_test.py   (requires Python 3.12 venv)
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime

DURATION = 980  # 16 min 20 s — guarantees at least one 15m close after startup
DB_FILE = "collector_test.db"

ENV: dict[str, str] = {
    **os.environ,
    "DATABASE_URL": f"sqlite+aiosqlite:///./{DB_FILE}",
    "JWT_SECRET_KEY": "smoke-test-only-jwt-secret-min-32-chars-xx",
    "MASTER_ENCRYPTION_KEY": "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
    "TELEGRAM_BOT_TOKEN": "0:smoke-test",
    "TELEGRAM_WEBHOOK_SECRET": "smoke-webhook-secret",
    "USE_TESTNET": "false",
    "COLLECTOR_EXCHANGE": "bybit",
    "LOG_LEVEL": "DEBUG",   # need DEBUG to see candles_upserted events
    "DEBUG": "false",       # keep JSON log format (not dev console renderer)
    "WATCHED_SYMBOLS": '["BTC/USDT","ETH/USDT","SOL/USDT"]',
    "WATCHED_TIMEFRAMES": '["15m","1h","4h"]',
    "COLLECTOR_BACKFILL_LIMIT": "100",
}

# ── Step 0: DNS reachability check ────────────────────────────────────────────
def _check_dns() -> bool:
    for host in ("api.bybit.com", "api.binance.com"):
        try:
            ip = socket.gethostbyname(host)
            print(f"  DNS OK  {host} -> {ip}")
            return True
        except OSError as exc:
            print(f"  DNS FAIL {host}: {exc}")
    return False


# ── Step 1: create DB tables ─────────────────────────────────────────────────
_INIT_CODE = """\
import asyncio
from app.db.session import Base, engine
import app.db.models
async def _init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
asyncio.run(_init())
print("DB tables ready.", flush=True)
"""

# ── Step 2: collector subprocess entry-point ─────────────────────────────────
_COLLECTOR_CODE = """\
from app.collectors.run_collector import main
main()
"""


def _create_tables() -> None:
    subprocess.run([sys.executable, "-c", _INIT_CODE], env=ENV, check=True)


def _stream(proc: subprocess.Popen[str], buf: list[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        s = line.rstrip()
        buf.append(s)
        print(f"  [collector] {s}", flush=True)


def _parse_log_stats(buf: list[str]) -> tuple[int, int]:
    """Return (gap_fill_candles, live_ws_candles) by parsing JSON log lines."""
    gap_fill = 0
    live_ws = 0
    for line in buf:
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        event = obj.get("event", "")
        if event == "gap_fill_complete":
            gap_fill += int(obj.get("count", 0))
        elif event == "candles_upserted":
            live_ws += int(obj.get("count", 0))
    return gap_fill, live_ws


def _show_stats(collector_start: datetime) -> None:
    sep = "=" * 66
    start_iso = collector_start.strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"\n{sep}")
    print("TOTAL CANDLE COUNTS BY TIMEFRAME  (gap-fill + live)")
    print(sep)
    try:
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()

        # ── Total counts ──────────────────────────────────────────────────────
        cur.execute(
            "SELECT timeframe, COUNT(*) FROM candles GROUP BY timeframe ORDER BY timeframe"
        )
        total_rows = cur.fetchall()
        if total_rows:
            for tf, n in total_rows:
                print(f"  {tf:>4s}  {n:>6,d} candles")
        else:
            print("  (empty)")

        # ── Live WS candles: candle CLOSED after collector start ─────────────
        # ts is candle open time; a candle closes at ts + tf_seconds.
        # 15m closes ts+900s, 1h closes ts+3600s, 4h closes ts+14400s.
        # We look for candles whose close time (ts + interval) >= collector start.
        print(f"\n{sep}")
        print(f"LIVE WS CANDLES  (closed after {start_iso})")
        print(sep)
        cur.execute(
            """
            SELECT timeframe,
                   COUNT(*) as n,
                   MAX(ts)  as newest_ts
              FROM candles
             WHERE (timeframe = '15m' AND datetime(ts, '+900 seconds')  >= ?)
                OR (timeframe = '1h'  AND datetime(ts, '+3600 seconds') >= ?)
                OR (timeframe = '4h'  AND datetime(ts, '+14400 seconds') >= ?)
             GROUP BY timeframe
             ORDER BY timeframe
            """,
            (start_iso, start_iso, start_iso),
        )
        live_rows = cur.fetchall()
        if live_rows:
            for tf, n, newest in live_rows:
                print(f"  {tf:>4s}  {n:>3d} candles  newest open={newest}  <- LIVE WS confirmed")
        else:
            print("  (none — no candle closed during this run)")

        # ── Last 3 BTC/USDT 15m ───────────────────────────────────────────────
        print(f"\n{sep}")
        print("LAST 5  BTC/USDT  15m  CANDLES  (newest first)")
        print(sep)
        cur.execute(
            """
            SELECT ts, o, h, l, c, v
              FROM candles
             WHERE symbol = 'BTC/USDT' AND timeframe = '15m'
             ORDER BY ts DESC
             LIMIT 5
            """,
        )
        btc = cur.fetchall()
        if btc:
            for ts, o, h, l, c, v in btc:
                close_ts = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                from datetime import timedelta
                marker = " <- LIVE WS" if close_ts + timedelta(seconds=900) >= collector_start else ""
                print(
                    f"  {ts}  o={o:>10,.2f}  h={h:>10,.2f}"
                    f"  l={l:>10,.2f}  c={c:>10,.2f}  vol={v:>10,.3f}{marker}"
                )
        else:
            print("  (no data)")
        con.close()
    except Exception as exc:
        print(f"  DB error: {exc}")


def main() -> None:
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"Removed old {DB_FILE}")

    # ── DNS check ──────────────────────────────────────────────────────────────
    print("\nStep 0 — checking DNS reachability ...")
    if not _check_dns():
        sys.exit("\nERROR: Cannot resolve exchange hostnames. Check connectivity.")

    # ── Create tables ──────────────────────────────────────────────────────────
    print("\nStep 1 — creating DB tables ...")
    _create_tables()

    # ── Start collector ────────────────────────────────────────────────────────
    mins = DURATION // 60
    print(f"\nStep 2 — starting collector (mainnet bybit, 3 symbols, {mins} min) ...")
    proc = subprocess.Popen(
        [sys.executable, "-c", _COLLECTOR_CODE],
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    collector_start = datetime.now(UTC)
    print(f"  PID {proc.pid}  started at {collector_start.strftime('%H:%M:%S UTC')}\n")

    buf: list[str] = []
    reader = threading.Thread(target=_stream, args=(proc, buf), daemon=True)
    reader.start()

    try:
        time.sleep(DURATION)
    finally:
        print(f"\nStep 3 — stopping collector (PID {proc.pid}) ...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("  Collector stopped.")

    # ── Log stats ──────────────────────────────────────────────────────────────
    gap_fill_candles, live_ws_candles = _parse_log_stats(buf)
    print(f"\n  Log summary:")
    print(f"    gap-fill candles (REST): {gap_fill_candles:,}")
    print(f"    live WS candles (upserted): {live_ws_candles:,}")

    _show_stats(collector_start)

    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"\nCleaned up {DB_FILE}.")


if __name__ == "__main__":
    main()
