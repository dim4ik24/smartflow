"""Smoke-test: run the collector against mainnet for ~3.5 min, then show stats.

Usage:  python collector_smoke_test.py   (requires Python 3.12 venv)
"""
from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time

DURATION = 210  # seconds
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
    "LOG_LEVEL": "INFO",
    "DEBUG": "false",
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
    r = subprocess.run([sys.executable, "-c", _INIT_CODE], env=ENV, check=True)
    if r.returncode != 0:
        sys.exit("Table creation failed.")


def _stream(proc: subprocess.Popen[str], buf: list[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        s = line.rstrip()
        buf.append(s)
        print(f"  [collector] {s}", flush=True)


def _show_stats() -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("CANDLE COUNTS BY TIMEFRAME")
    print(sep)
    try:
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute(
            "SELECT timeframe, COUNT(*) FROM candles GROUP BY timeframe ORDER BY timeframe"
        )
        rows = cur.fetchall()
        if rows:
            for tf, n in rows:
                print(f"  {tf:>4s}  {n:>6,d} candles")
        else:
            print("  (empty — collector may not have received WS data in time)")

        print(f"\n{sep}")
        print("LAST 3  BTC/USDT  15m  CANDLES  (newest first)")
        print(sep)
        cur.execute(
            """
            SELECT ts, o, h, l, c, v
              FROM candles
             WHERE symbol = 'BTC/USDT' AND timeframe = '15m'
             ORDER BY ts DESC
             LIMIT 3
            """
        )
        btc = cur.fetchall()
        if btc:
            for ts, o, h, l, c, v in btc:
                print(
                    f"  {ts}   o={o:>10,.2f}  h={h:>10,.2f}"
                    f"  l={l:>10,.2f}  c={c:>10,.2f}  vol={v:>12,.3f}"
                )
        else:
            print("  (no BTC/USDT 15m data yet)")
        con.close()
    except Exception as exc:
        print(f"  DB error: {exc}")


def main() -> None:
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"Removed old {DB_FILE}")

    # ── DNS check ──────────────────────────────────────────────────────────────
    print("\nStep 0 — checking DNS reachability …")
    if not _check_dns():
        print(
            "\nERROR: Cannot resolve exchange hostnames via Python socket.\n"
            "Check network connectivity and try again."
        )
        sys.exit(1)

    # ── Create tables ──────────────────────────────────────────────────────────
    print("\nStep 1 — creating DB tables …")
    _create_tables()

    # ── Start collector ────────────────────────────────────────────────────────
    print(
        f"\nStep 2 — starting collector "
        f"(mainnet bybit, 3 symbols, {DURATION}s) …"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", _COLLECTOR_CODE],
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    print(f"  PID {proc.pid} — streaming logs:\n")

    buf: list[str] = []
    reader = threading.Thread(target=_stream, args=(proc, buf), daemon=True)
    reader.start()

    try:
        time.sleep(DURATION)
    finally:
        print(f"\nStep 3 — stopping collector (PID {proc.pid}) …")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("  Collector stopped.")

    _show_stats()

    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"\nCleaned up {DB_FILE}.")


if __name__ == "__main__":
    main()
