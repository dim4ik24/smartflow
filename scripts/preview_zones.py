"""Preview SMC zones on real BTC/USDT 4h candles.

Usage (run from anywhere — paths resolve relative to this file):
    python scripts/preview_zones.py

The script:
  1. Loads BTC/USDT 4h candles from the local SQLite DB.
  2. If fewer than MIN_CANDLES rows exist, back-fills via Bybit mainnet REST.
  3. Runs analyze(confirmed_only=True) on the last FETCH_LIMIT candles.
  4. Renders an interactive Plotly chart with all SMC zones overlaid.
  5. Saves scripts/preview_zones.html and opens it in the browser.
"""
from __future__ import annotations

import os
import sys
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Force UTF-8 output so Windows cp1251 doesn't choke on emoji in 3rd-party libs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Path bootstrap ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent   # …/scripts/
BACKEND_DIR = SCRIPT_DIR.parent / "backend"    # …/backend/
OUTPUT_HTML = SCRIPT_DIR / "preview_zones.html"
sys.path.insert(0, str(BACKEND_DIR))

# ── Dummy env vars so config.py validates without a real .env ─────────────────
_DUMMY: dict[str, str] = {
    "JWT_SECRET_KEY":          "x" * 32,
    "MASTER_ENCRYPTION_KEY":   "0" * 64,
    "TELEGRAM_BOT_TOKEN":      "0:AA" + "x" * 35,
    "TELEGRAM_WEBHOOK_SECRET": "preview_script",
}
for _k, _v in _DUMMY.items():
    os.environ.setdefault(_k, _v)

import ccxt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from sqlalchemy import create_engine, inspect as sa_inspect  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.analysis.smc import analyze  # noqa: E402
from app.config import settings  # noqa: E402
from app.db.models import Base, Candle  # noqa: E402

# ── Script constants ───────────────────────────────────────────────────────────
SYMBOL = "BTC/USDT"
TIMEFRAME = "4h"
MIN_CANDLES = 50
FETCH_LIMIT = 200

# ── Zone rendering styles ──────────────────────────────────────────────────────
# shape: "rect"  → filled rectangle (OB, FVG, PREM, DISC)
# shape: "hline" → horizontal line  (BOS, CHOCH, EQH, EQL, LIQ_SWEEP)
# Each entry may have "long" / "short" keys, or "any" for direction-agnostic.
_ZONE_STYLES: dict[str, dict[str, Any]] = {
    "OB": {
        "shape": "rect",
        "long":  {"fill": "rgba(0,200,83,0.22)",   "edge": "rgba(0,200,83,0.85)"},
        "short": {"fill": "rgba(229,57,53,0.22)",   "edge": "rgba(229,57,53,0.85)"},
        "label": "Order Block",
        "legend_color": "#00c853",
        "legend_symbol": "square",
    },
    "FVG": {
        "shape": "rect",
        "long":  {"fill": "rgba(0,230,118,0.15)",  "edge": "rgba(0,230,118,0.65)"},
        "short": {"fill": "rgba(255,82,82,0.15)",   "edge": "rgba(255,82,82,0.65)"},
        "label": "Fair Value Gap",
        "legend_color": "#00e676",
        "legend_symbol": "square",
    },
    "PREM": {
        "shape": "rect",
        "any":   {"fill": "rgba(233,30,99,0.10)",  "edge": "rgba(233,30,99,0.45)"},
        "label": "Premium Zone",
        "legend_color": "#e91e63",
        "legend_symbol": "square",
    },
    "DISC": {
        "shape": "rect",
        "any":   {"fill": "rgba(0,200,83,0.10)",   "edge": "rgba(0,200,83,0.45)"},
        "label": "Discount Zone",
        "legend_color": "#69f0ae",
        "legend_symbol": "square",
    },
    "BOS": {
        "shape": "hline",
        "long":  {"color": "#1de9b6", "dash": "dash",     "width": 1.5},
        "short": {"color": "#ff5252", "dash": "dash",     "width": 1.5},
        "label": "Break of Structure",
        "legend_color": "#1de9b6",
        "legend_symbol": "line-ew",
    },
    "CHOCH": {
        "shape": "hline",
        "long":  {"color": "#40c4ff", "dash": "dot",      "width": 2.0},
        "short": {"color": "#ff6d00", "dash": "dot",      "width": 2.0},
        "label": "Change of Character",
        "legend_color": "#40c4ff",
        "legend_symbol": "line-ew",
    },
    "EQH": {
        "shape": "hline",
        "any":   {"color": "#ffd600", "dash": "dashdot",  "width": 1.0},
        "label": "Equal Highs",
        "legend_color": "#ffd600",
        "legend_symbol": "line-ew",
    },
    "EQL": {
        "shape": "hline",
        "any":   {"color": "#ffd600", "dash": "dashdot",  "width": 1.0},
        "label": "Equal Lows",
        "legend_color": "#ffd600",
        "legend_symbol": "line-ew",
    },
    "LIQ_SWEEP": {
        "shape": "hline",
        "any":   {"color": "#ce93d8", "dash": "longdash", "width": 1.5},
        "label": "Liquidity Sweep",
        "legend_color": "#ce93d8",
        "legend_symbol": "line-ew",
    },
}


# ── Database helpers ───────────────────────────────────────────────────────────

def _make_engine():
    url = settings.database_url.replace("+aiosqlite", "")
    if url.startswith("sqlite:///./"):
        db_path = BACKEND_DIR / url.removeprefix("sqlite:///./")
        url = f"sqlite:///{db_path}"
    return create_engine(url, echo=False)


def _load_candles(engine) -> pd.DataFrame:
    insp = sa_inspect(engine)
    if not insp.has_table("candles"):
        return pd.DataFrame()
    with Session(engine) as session:
        rows = (
            session.query(Candle)
            .filter(Candle.symbol == SYMBOL, Candle.timeframe == TIMEFRAME)
            .order_by(Candle.ts.asc())
            .all()
        )
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        ts = r.ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        records.append({
            "ts": ts,
            "open": float(r.o), "high": float(r.h),
            "low":  float(r.l), "close": float(r.c),
            "volume": float(r.v),
        })
    df = pd.DataFrame(records).set_index("ts")
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df


def _backfill(engine) -> None:
    print(f"  Fetching {FETCH_LIMIT} × {TIMEFRAME} candles from Bybit mainnet …")
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "future"}})
    raw = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=FETCH_LIMIT)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for ts_ms, o, h, l, c, v in raw:
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
            session.merge(Candle(
                symbol=SYMBOL, timeframe=TIMEFRAME, ts=ts,
                o=o, h=h, l=l, c=c, v=v,
            ))
        session.commit()
    print(f"  Saved {len(raw)} candles.")


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _zone_style(zone: dict) -> dict:
    sdef = _ZONE_STYLES[zone["type"]]
    direction = zone.get("direction", "long")
    return sdef.get(direction, sdef.get("any", {}))  # type: ignore[return-value]


def _parse_zone_ts(ts_str: str | None, fallback: pd.Timestamp) -> pd.Timestamp:
    if ts_str is None:
        return fallback
    ts = pd.Timestamp(ts_str)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def _iso(ts: pd.Timestamp) -> str:
    """Plotly-compatible ISO datetime string."""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _add_zone_shapes(
    fig: go.Figure,
    zones: list[dict],
    last_ts: pd.Timestamp,
) -> None:
    for z in zones:
        sdef = _ZONE_STYLES.get(z["type"])
        if sdef is None:
            continue
        style = _zone_style(z)
        t0 = _parse_zone_ts(z["time_from"], last_ts)
        t1 = _parse_zone_ts(z["time_to"], last_ts)

        if sdef["shape"] == "rect":
            fig.add_shape(
                type="rect",
                x0=_iso(t0), x1=_iso(t1),
                y0=z["price_from"], y1=z["price_to"],
                fillcolor=style["fill"],
                line=dict(color=style["edge"], width=1),
                layer="below",
                xref="x", yref="y",
            )
        else:  # hline
            price = z["price_from"]
            fig.add_shape(
                type="line",
                x0=_iso(t0), x1=_iso(t1),
                y0=price, y1=price,
                line=dict(
                    color=style["color"],
                    dash=style["dash"],
                    width=style["width"],
                ),
                xref="x", yref="y",
            )


def _add_hover_markers(
    fig: go.Figure,
    zones: list[dict],
    last_ts: pd.Timestamp,
) -> None:
    """Invisible markers at zone centres — hovering reveals zone details."""
    for z in zones:
        sdef = _ZONE_STYLES.get(z["type"])
        if sdef is None:
            continue
        style = _zone_style(z)
        t0 = _parse_zone_ts(z["time_from"], last_ts)
        t1 = _parse_zone_ts(z["time_to"], last_ts)
        mid_t = t0 + (t1 - t0) / 2
        p0, p1 = z["price_from"], z["price_to"]
        mid_p = (p0 + p1) / 2.0

        color = style.get("color") or style.get("edge") or "#ffffff"
        hover = (
            f"<b>{sdef['label']}</b>  ({z.get('direction', '')})<br>"
            f"Price: {p0:.2f} – {p1:.2f}<br>"
            f"From: {z['time_from']}<br>"
            f"To: {z['time_to'] or 'open'}<br>"
            f"Strength: {z.get('strength', 0):.2f}"
            + ("  <i>mitigated</i>" if z.get("mitigated") else "")
        )
        fig.add_trace(go.Scatter(
            x=[_iso(mid_t)], y=[mid_p],
            mode="markers",
            marker=dict(size=8, color=color, opacity=0.0),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))


def _add_legend_entries(fig: go.Figure, present_types: set[str]) -> None:
    for ztype, sdef in _ZONE_STYLES.items():
        if ztype not in present_types:
            continue
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                size=11,
                color=sdef["legend_color"],
                symbol=sdef["legend_symbol"],
                line=dict(color=sdef["legend_color"], width=2),
            ),
            name=sdef["label"],
            showlegend=True,
        ))


def build_chart(df: pd.DataFrame, zones: list[dict]) -> go.Figure:
    last_ts = df.index[-1]
    n_active = sum(1 for z in zones if not z.get("mitigated"))
    title = (
        f"SMC Zones — {SYMBOL} {TIMEFRAME}  |  "
        f"{len(zones)} zones ({n_active} active, {len(zones)-n_active} mitigated)  |  "
        f"confirmed_only=True"
    )

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=[_iso(ts) for ts in df.index],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name=f"{SYMBOL} {TIMEFRAME}",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        showlegend=True,
    ))

    _add_zone_shapes(fig, zones, last_ts)
    _add_hover_markers(fig, zones, last_ts)
    _add_legend_entries(fig, {z["type"] for z in zones})

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(
            title="Time (UTC)",
            rangeslider=dict(visible=False),
            tickformat="%b %d %H:%M",
            tickangle=-40,
        ),
        yaxis=dict(
            title="Price (USDT)",
            fixedrange=False,
            side="right",
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="#555555",
            borderwidth=1,
            font=dict(size=11),
        ),
        height=750,
        margin=dict(l=10, r=90, t=55, b=55),
        hovermode="closest",
    )
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    engine = _make_engine()

    print("Loading candles from DB …")
    df = _load_candles(engine)

    if len(df) < MIN_CANDLES:
        print(f"  Only {len(df)} rows — back-filling from Bybit …")
        _backfill(engine)
        df = _load_candles(engine)

    df = df.tail(FETCH_LIMIT)
    print(f"  {len(df)} candles  [{df.index[0]}  →  {df.index[-1]}]")

    print("Running SMC analysis (confirmed_only=True, include_mitigated=True) …")
    zones = analyze(df, confirmed_only=True, include_mitigated=True)
    counts: dict[str, int] = {}
    for z in zones:
        counts[z["type"]] = counts.get(z["type"], 0) + 1
    print(f"  {len(zones)} zones: {counts}")

    print("Rendering chart …")
    fig = build_chart(df, zones)
    fig.write_html(str(OUTPUT_HTML), include_plotlyjs="cdn")
    print(f"  Saved → {OUTPUT_HTML}")

    webbrowser.open(OUTPUT_HTML.as_uri())
    print("Done — check your browser.")


if __name__ == "__main__":
    main()
