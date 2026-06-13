"""Preview SMC zones on real BTC/USDT 4h candles.

Usage (run from anywhere — paths resolve relative to this file):
    python scripts/preview_zones.py

The script:
  1. Loads BTC/USDT 4h candles from the local SQLite DB.
  2. If fewer than MIN_CANDLES rows exist, back-fills via Bybit mainnet REST.
  3. Runs analyze(confirmed_only=True, include_mitigated=True).
  4. Generates a self-contained HTML on Lightweight Charts (TradingView).
  5. Saves scripts/preview_zones.html and opens it in the browser.
"""
from __future__ import annotations

import json
import os
import sys
import webbrowser
from datetime import UTC, datetime
from pathlib import Path

# Force UTF-8 — smartmoneyconcepts prints a star emoji on import.
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
from sqlalchemy import create_engine, inspect as sa_inspect  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.analysis.smc import analyze  # noqa: E402
from app.config import settings  # noqa: E402
from app.db.models import Base, Candle  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
SYMBOL = "BTC/USDT"
TIMEFRAME = "4h"
MIN_CANDLES = 50
FETCH_LIMIT = 200

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


# ── Data serialisation ─────────────────────────────────────────────────────────

def _df_to_lwc(df: pd.DataFrame) -> list[dict]:
    """DataFrame → Lightweight Charts candlestick format (Unix seconds)."""
    return [
        {
            "time": int(ts.timestamp()),
            "open": row["open"], "high": row["high"],
            "low": row["low"],   "close": row["close"],
        }
        for ts, row in df.iterrows()
    ]


def _parse_zone_ts(ts_str: str | None, fallback: pd.Timestamp) -> int:
    if ts_str is None:
        return int(fallback.timestamp())
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def _zones_to_json(zones: list[dict], last_ts: pd.Timestamp) -> list[dict]:
    out = []
    for z in zones:
        out.append({
            **z,
            "time_from_unix": _parse_zone_ts(z["time_from"], last_ts),
            "time_to_unix":   _parse_zone_ts(z["time_to"],   last_ts),
        })
    return out


# ── HTML generation ────────────────────────────────────────────────────────────

# Plain string — no f-string, uses __PLACEHOLDER__ replacements at the end.
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SMC Zones — __SYMBOL__ __TF__</title>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; background: #131722; color: #d1d4dc;
      font: 12px/1.5 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    body { display: flex; flex-direction: column; }

    /* ── Header ─────────────────────────────────── */
    #hdr {
      flex-shrink: 0;
      display: flex; align-items: center; gap: 14px;
      padding: 6px 14px;
      background: #1a1f2e;
      border-bottom: 1px solid #2a2e39;
    }
    #hdr h1 { font-size: 13px; font-weight: 600; color: #d1d4dc; white-space: nowrap; }
    #stats  { font-size: 11px; color: #787b86; white-space: nowrap; }
    .ctrl   { display: flex; align-items: center; gap: 5px; margin-left: auto; }
    .ctrl label { cursor: pointer; color: #d1d4dc; }
    input[type=checkbox] { cursor: pointer; accent-color: #2962ff; width: 13px; height: 13px; }

    /* ── Chart area ─────────────────────────────── */
    #wrap { flex: 1; position: relative; overflow: hidden; }
    #chart { width: 100%; height: 100%; }

    /* ── Legend overlay ─────────────────────────── */
    #legend {
      position: absolute; top: 10px; left: 10px; z-index: 10;
      background: rgba(19,23,34,0.88);
      border: 1px solid #2a2e39; border-radius: 4px;
      padding: 7px 10px; font-size: 11px; line-height: 1.85;
      pointer-events: none; backdrop-filter: blur(3px);
      min-width: 170px;
    }
    .lr { display: flex; align-items: center; gap: 6px; }
    .ic-rect { width: 11px; height: 11px; border-radius: 2px; flex-shrink: 0; }
    .ic-line { width: 16px; height: 0; flex-shrink: 0; }
    .ic-dot  { width: 8px;  height: 8px; border-radius: 50%; flex-shrink: 0; }
    .lname   { flex: 1; }
    .lcnt    { color: #787b86; padding-left: 6px; white-space: nowrap; }
    .lsep    { border-top: 1px solid #2a2e39; margin: 4px 0; }
  </style>
</head>
<body>

<div id="hdr">
  <h1>SMC Zones — __SYMBOL__ __TF__ &nbsp;|&nbsp; confirmed_only=True</h1>
  <span id="stats">…</span>
  <div class="ctrl">
    <input type="checkbox" id="chk-mit">
    <label for="chk-mit">Show mitigated zones</label>
  </div>
</div>

<div id="wrap">
  <div id="chart"></div>
  <div id="legend"></div>
</div>

<script>
/* ── Embedded data ─────────────────────────────────────────────────────────── */
const CANDLES = __CANDLES_JSON__;
const ZONES   = __ZONES_JSON__;

/* ── Zone style config ─────────────────────────────────────────────────────── */
// shape: 'rect'   → filled rectangle via ISeriesPrimitive
// shape: 'hline'  → createPriceLine
// shape: 'marker' → setMarkers
const CFG = {
  OB:  { shape:'rect',   label:'Order Block',
    long:  { fill:'rgba(38,166,154,.22)',  border:'rgba(38,166,154,.9)',  leg:'#26a69a' },
    short: { fill:'rgba(239,83,80,.22)',   border:'rgba(239,83,80,.9)',   leg:'#ef5350' } },
  FVG: { shape:'rect',   label:'Fair Value Gap',
    long:  { fill:'rgba(100,221,23,.15)', border:'rgba(100,221,23,.7)',  leg:'#64dd17' },
    short: { fill:'rgba(255,171,0,.15)',  border:'rgba(255,171,0,.7)',   leg:'#ffab00' } },
  PREM:{ shape:'rect',   label:'Premium Zone',
    any:   { fill:'rgba(233,30,99,.09)',  border:'rgba(233,30,99,.5)',   leg:'#e91e63' } },
  DISC:{ shape:'rect',   label:'Discount Zone',
    any:   { fill:'rgba(0,230,118,.09)', border:'rgba(0,230,118,.5)',   leg:'#00e676' } },
  BOS: { shape:'hline',  label:'Break of Structure',
    long:  { color:'#1de9b6', ls:2, lw:1, leg:'#1de9b6' },
    short: { color:'#ff5252', ls:2, lw:1, leg:'#ff5252' } },
  CHOCH:{ shape:'hline', label:'Chg of Character',
    long:  { color:'#40c4ff', ls:1, lw:2, leg:'#40c4ff' },
    short: { color:'#ff6d00', ls:1, lw:2, leg:'#ff6d00' } },
  EQH: { shape:'hline',  label:'Equal Highs',
    any:   { color:'#ffd600', ls:4, lw:1, leg:'#ffd600' } },
  EQL: { shape:'hline',  label:'Equal Lows',
    any:   { color:'#ffd600', ls:4, lw:1, leg:'#ffd600' } },
  LIQ_SWEEP:{ shape:'marker', label:'Liq. Sweep',
    any:   { color:'#ce93d8', leg:'#ce93d8' } },
};
const TYPE_ORDER = ['OB','FVG','PREM','DISC','BOS','CHOCH','EQH','EQL','LIQ_SWEEP'];

/* ── Chart init ────────────────────────────────────────────────────────────── */
const wrap  = document.getElementById('wrap');
const chartEl = document.getElementById('chart');
const chart = LightweightCharts.createChart(chartEl, {
  layout: { background:{ type:'solid', color:'#131722' }, textColor:'#d1d4dc', fontSize:11 },
  grid:   { vertLines:{ color:'#1e2233' }, horzLines:{ color:'#1e2233' } },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  rightPriceScale: { borderColor:'#2a2e39' },
  timeScale: { borderColor:'#2a2e39', timeVisible:true, secondsVisible:false, barSpacing:8 },
  width:  chartEl.clientWidth,
  height: chartEl.clientHeight,
});

new ResizeObserver(() => chart.resize(wrap.clientWidth, wrap.clientHeight)).observe(wrap);

const candles = chart.addCandlestickSeries({
  upColor:'#26a69a',   downColor:'#ef5350',
  borderUpColor:'#26a69a', borderDownColor:'#ef5350',
  wickUpColor:'#26a69a',   wickDownColor:'#ef5350',
});
candles.setData(CANDLES);
chart.timeScale().fitContent();

/* ── Zone rectangle primitive ──────────────────────────────────────────────── */
class RectRenderer {
  constructor(prim) { this._p = prim; }

  draw(target) {
    const { _chart:ch, _series:ser, _zones:zones } = this._p;
    if (!ch || !ser || !zones.length) return;

    target.useBitmapCoordinateSpace(({ context:ctx, horizontalPixelRatio:hpr, verticalPixelRatio:vpr }) => {
      for (const z of zones) {
        const cfg = CFG[z.type];
        if (!cfg || cfg.shape !== 'rect') continue;
        const s = cfg[z.direction] ?? cfg.any;
        if (!s) continue;

        const cx0 = ch.timeScale().timeToCoordinate(z.time_from_unix);
        const cx1 = ch.timeScale().timeToCoordinate(z.time_to_unix);
        const cy0 = ser.priceToCoordinate(z.price_to);    // higher price → smaller y
        const cy1 = ser.priceToCoordinate(z.price_from);  // lower price  → larger  y
        if (cx0==null||cx1==null||cy0==null||cy1==null) continue;

        /* convert logical → bitmap pixels */
        const bx0 = Math.round(Math.min(cx0, cx1) * hpr);
        const bx1 = Math.round(Math.max(cx0, cx1) * hpr);
        const by0 = Math.round(Math.min(cy0, cy1) * vpr);
        const by1 = Math.round(Math.max(cy0, cy1) * vpr);
        const bw = bx1 - bx0, bh = by1 - by0;
        if (bw < 1 || bh < 1) continue;

        ctx.save();
        ctx.globalAlpha = 1;
        ctx.fillStyle = s.fill;
        ctx.fillRect(bx0, by0, bw, bh);
        ctx.strokeStyle = s.border;
        ctx.lineWidth = hpr;
        ctx.strokeRect(bx0 + 0.5, by0 + 0.5, bw - 1, bh - 1);

        /* type label inside the rectangle */
        if (bw > 22 * hpr) {
          ctx.fillStyle = s.border;
          const fs = Math.round(9 * hpr);
          ctx.font = `bold ${fs}px monospace`;
          const arrow = z.direction === 'long' ? ' ↑' : (z.direction === 'short' ? ' ↓' : '');
          ctx.fillText(z.type + arrow, bx0 + 3 * hpr, by0 + 11 * vpr);
        }
        ctx.restore();
      }
    });
  }
}

class RectPaneView {
  constructor(p) { this._r = new RectRenderer(p); }
  renderer() { return this._r; }
}

class RectPrimitive {
  constructor() {
    this._zones = []; this._chart = null; this._series = null;
    this._requestUpdate = null; this._view = new RectPaneView(this);
  }
  attached({ chart, series, requestUpdate }) {
    this._chart = chart; this._series = series; this._requestUpdate = requestUpdate;
  }
  detached() { this._chart = this._series = this._requestUpdate = null; }
  updateAllViews() {}
  paneViews() { return [this._view]; }
  setZones(z) { this._zones = z; if (this._requestUpdate) this._requestUpdate(); }
}

const rectPrim = new RectPrimitive();
candles.attachPrimitive(rectPrim);

/* ── Zone rendering orchestrator ───────────────────────────────────────────── */
let priceLinesActive = [];

function zoneStyle(z) {
  const cfg = CFG[z.type];
  return cfg ? (cfg[z.direction] ?? cfg.any ?? null) : null;
}

function render(zones) {
  /* 1 – rectangles via primitive */
  rectPrim.setZones(zones.filter(z => (CFG[z.type] || {}).shape === 'rect'));

  /* 2 – horizontal price lines (BOS / CHOCH / EQH / EQL) */
  priceLinesActive.forEach(pl => candles.removePriceLine(pl));
  priceLinesActive = [];
  for (const z of zones) {
    const cfg = CFG[z.type];
    if (!cfg || cfg.shape !== 'hline') continue;
    const s = zoneStyle(z);
    if (!s) continue;
    const arrow = z.direction === 'long' ? ' ↑' : (z.direction === 'short' ? ' ↓' : '');
    const pl = candles.createPriceLine({
      price: z.price_from, color: s.color,
      lineStyle: s.ls, lineWidth: s.lw,
      title: z.type + arrow + (z.mitigated ? ' ·' : ''),
      axisLabelVisible: true,
    });
    priceLinesActive.push(pl);
  }

  /* 3 – sweep markers */
  const markers = zones
    .filter(z => z.type === 'LIQ_SWEEP')
    .map(z => ({
      time: z.time_from_unix,
      position: z.direction === 'long' ? 'belowBar' : 'aboveBar',
      color: '#ce93d8',
      shape: z.direction === 'long' ? 'arrowUp' : 'arrowDown',
      text: 'Sweep', size: 1,
    }))
    .sort((a, b) => a.time - b.time);
  candles.setMarkers(markers);

  /* 4 – legend + stats */
  refreshLegend(zones);
}

/* ── Legend builder ────────────────────────────────────────────────────────── */
function refreshLegend(zones) {
  const counts = {};
  for (const z of zones) {
    if (!counts[z.type]) counts[z.type] = { a: 0, m: 0 };
    z.mitigated ? counts[z.type].m++ : counts[z.type].a++;
  }

  const el = document.getElementById('legend');
  el.innerHTML = '';

  for (const type of TYPE_ORDER) {
    if (!counts[type]) continue;
    const cfg = CFG[type];
    const s = cfg.long ?? cfg.any ?? {};
    const color = s.leg ?? s.color ?? '#fff';
    const { a, m } = counts[type];

    const row = document.createElement('div');
    row.className = 'lr';

    const ic = document.createElement('div');
    if (cfg.shape === 'rect') {
      ic.className = 'ic-rect';
      ic.style.cssText = `background:${color};border:1px solid ${color};opacity:.75`;
    } else if (cfg.shape === 'hline') {
      ic.className = 'ic-line';
      const dash = s.ls === 1 ? 'dotted' : 'dashed';
      ic.style.cssText = `border-top:2px ${dash} ${color}`;
    } else {
      ic.className = 'ic-dot';
      ic.style.background = color;
    }

    const nm = document.createElement('span');
    nm.className = 'lname'; nm.textContent = cfg.label;

    const cnt = document.createElement('span');
    cnt.className = 'lcnt';
    cnt.textContent = m ? `${a} (+${m})` : `${a}`;

    row.append(ic, nm, cnt);
    el.appendChild(row);
  }

  const total  = zones.length;
  const active = zones.filter(z => !z.mitigated).length;
  document.getElementById('stats').textContent =
    `${active} active / ${total} total zones  ·  ${CANDLES.length} candles (__DATE_RANGE__)`;
}

/* ── Checkbox ──────────────────────────────────────────────────────────────── */
document.getElementById('chk-mit').addEventListener('change', e => {
  render(e.target.checked ? ZONES : ZONES.filter(z => !z.mitigated));
});

/* ── Initial render (active zones only) ────────────────────────────────────── */
render(ZONES.filter(z => !z.mitigated));
</script>
</body>
</html>
"""


def _build_html(
    candles_json: list[dict],
    zones_json: list[dict],
    date_range: str,
) -> str:
    html = _HTML_TEMPLATE
    html = html.replace("__SYMBOL__", SYMBOL)
    html = html.replace("__TF__", TIMEFRAME)
    html = html.replace("__DATE_RANGE__", date_range)
    html = html.replace("__CANDLES_JSON__", json.dumps(candles_json, separators=(",", ":")))
    html = html.replace("__ZONES_JSON__",   json.dumps(zones_json,   separators=(",", ":")))
    return html


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
    active = sum(1 for z in zones if not z.get("mitigated"))
    print(f"  {len(zones)} zones ({active} active): {counts}")

    last_ts = df.index[-1]
    candles_data = _df_to_lwc(df)
    zones_data   = _zones_to_json(zones, last_ts)

    date_range = (
        df.index[0].strftime("%b %d") + " – " + df.index[-1].strftime("%b %d, %Y")
    )

    print("Generating Lightweight Charts HTML …")
    html = _build_html(candles_data, zones_data, date_range)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"  Saved → {OUTPUT_HTML}")

    webbrowser.open(OUTPUT_HTML.as_uri())
    print("Done — check your browser.")


if __name__ == "__main__":
    main()
