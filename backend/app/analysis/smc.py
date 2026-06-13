"""SMC (Smart Money Concepts) analysis — wrapper over smartmoneyconcepts.

Exposes a single public function ``analyze()`` that maps raw OHLCV data to a
unified list of zone dicts matching the ``signals.zones`` JSON schema
documented in SPEC.md §3.

Zone types
----------
OB          Order Block (unfilled / currently active)
FVG         Fair Value Gap
BOS         Break of Structure (structural event)
CHOCH       Change of Character (structural event)
EQH         Equal Highs  — active buy-side liquidity pool
EQL         Equal Lows   — active sell-side liquidity pool
LIQ_SWEEP   Swept liquidity (historical)
PREM        Premium zone (above 50 % of last swing range)
DISC        Discount zone (below 50 % of last swing range)

Each zone dict keys
-------------------
type        : str        — one of the types above
direction   : str        — "long" (bullish) | "short" (bearish)
price_from  : float      — lower price boundary
price_to    : float      — upper price boundary
time_from   : str        — ISO-8601 UTC
time_to     : str | None — ISO-8601 UTC when zone closed/broken, else None
strength    : float      — 0.0–1.0 (library-specific metric)
mitigated   : bool       — True when zone is already filled / swept
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import smartmoneyconcepts as _smc_pkg

_smc = _smc_pkg.smc

ZONE_TYPES = frozenset(
    {"OB", "FVG", "BOS", "CHOCH", "EQH", "EQL", "LIQ_SWEEP", "PREM", "DISC"}
)
_REQUIRED_COLS = frozenset({"open", "high", "low", "close", "volume"})
_MIN_CANDLES = 10
# ORM Candle model uses single-letter column names; normalize them on entry.
_ORM_COL_MAP: dict[str, str] = {
    "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt(ts: pd.Timestamp) -> str:
    return str(ts.strftime("%Y-%m-%dT%H:%M:%SZ"))


def _pos_to_ts(ohlc: pd.DataFrame, pos: Any) -> str | None:
    """Convert a positional integer index (float from library) to ISO timestamp."""
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return None
    i = int(pos)
    return _fmt(ohlc.index[i]) if 0 <= i < len(ohlc) else None


def _normalize_columns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Rename ORM short names (o/h/l/c/v) to standard names if present."""
    if _ORM_COL_MAP.keys() & set(ohlc.columns):
        return ohlc.rename(columns=_ORM_COL_MAP)
    return ohlc


def _validate(ohlc: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(ohlc.columns)
    if missing:
        raise ValueError(f"OHLC DataFrame missing columns: {sorted(missing)}")
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise TypeError("OHLC DataFrame index must be a DatetimeIndex")
    if len(ohlc) < _MIN_CANDLES:
        raise ValueError(f"Need at least {_MIN_CANDLES} candles; got {len(ohlc)}")


# ── Extractor functions ───────────────────────────────────────────────────────


def _extract_bos_choch(ohlc: pd.DataFrame, shl: pd.DataFrame) -> list[dict[str, Any]]:
    bc = _smc.bos_choch(ohlc, shl)
    zones: list[dict[str, Any]] = []
    for pos, row in bc.iterrows():
        for col in ("BOS", "CHOCH"):
            val = row[col]
            if pd.isna(val) or val == 0:
                continue
            level = float(row["Level"])
            broken_ts = _pos_to_ts(ohlc, row["BrokenIndex"])
            zones.append(
                {
                    "type": col,
                    "direction": "long" if val > 0 else "short",
                    "price_from": level,
                    "price_to": level,
                    "time_from": _fmt(ohlc.index[pos]),
                    "time_to": broken_ts,
                    "strength": 1.0,
                    "mitigated": broken_ts is not None,
                }
            )
    return zones


def _extract_ob(
    ohlc: pd.DataFrame,
    shl: pd.DataFrame,
    include_mitigated: bool,
) -> list[dict[str, Any]]:
    ob = _smc.ob(ohlc, shl)
    zones: list[dict[str, Any]] = []
    for pos, row in ob.iterrows():
        val = row["OB"]
        if pd.isna(val) or val == 0:
            continue
        mit_idx = row["MitigatedIndex"]
        # MitigatedIndex == 0 is the library's sentinel for "not yet mitigated"
        mitigated = not (pd.isna(mit_idx) or float(mit_idx) == 0)
        if mitigated and not include_mitigated:
            continue
        pct = row["Percentage"]
        strength = min(1.0, max(0.0, float(pct) / 100.0)) if not pd.isna(pct) else 0.0
        zones.append(
            {
                "type": "OB",
                "direction": "long" if val > 0 else "short",
                "price_from": float(row["Bottom"]),
                "price_to": float(row["Top"]),
                "time_from": _fmt(ohlc.index[pos]),
                "time_to": _pos_to_ts(ohlc, mit_idx if mitigated else None),
                "strength": strength,
                "mitigated": mitigated,
            }
        )
    return zones


def _extract_fvg(ohlc: pd.DataFrame, include_mitigated: bool) -> list[dict[str, Any]]:
    fvg = _smc.fvg(ohlc)
    zones: list[dict[str, Any]] = []
    for pos, row in fvg.iterrows():
        val = row["FVG"]
        if pd.isna(val) or val == 0:
            continue
        mit_idx = row["MitigatedIndex"]
        mitigated = not pd.isna(mit_idx)
        if mitigated and not include_mitigated:
            continue
        zones.append(
            {
                "type": "FVG",
                "direction": "long" if val > 0 else "short",
                "price_from": float(row["Bottom"]),
                "price_to": float(row["Top"]),
                "time_from": _fmt(ohlc.index[pos]),
                "time_to": _pos_to_ts(ohlc, mit_idx),
                "strength": 0.7,
                "mitigated": mitigated,
            }
        )
    return zones


def _extract_liquidity(
    ohlc: pd.DataFrame,
    shl: pd.DataFrame,
    range_pct: float,
    include_mitigated: bool,
) -> list[dict[str, Any]]:
    liq = _smc.liquidity(ohlc, shl, range_percent=range_pct)
    zones: list[dict[str, Any]] = []
    for pos, row in liq.iterrows():
        val = row["Liquidity"]
        if pd.isna(val) or val == 0:
            continue
        swept_idx = row["Swept"]
        swept = not pd.isna(swept_idx)
        if swept and not include_mitigated:
            continue
        level = float(row["Level"])
        half = level * range_pct / 2
        # Liquidity = 1 → equal highs (buy-side); -1 → equal lows (sell-side).
        # A sweep of buy-side liquidity is a bearish move; sell-side is bullish.
        is_highs = val > 0
        zones.append(
            {
                "type": "LIQ_SWEEP" if swept else ("EQH" if is_highs else "EQL"),
                "direction": "short" if is_highs else "long",
                "price_from": level - half,
                "price_to": level + half,
                "time_from": _fmt(ohlc.index[pos]),
                "time_to": _pos_to_ts(ohlc, swept_idx if swept else row.get("End")),
                "strength": 0.8,
                "mitigated": swept,
            }
        )
    return zones


def _extract_premium_discount(
    ohlc: pd.DataFrame, shl: pd.DataFrame
) -> list[dict[str, Any]]:
    """Derive premium/discount zones from the most recent swing range."""
    highs = shl[shl["HighLow"] == 1]
    lows = shl[shl["HighLow"] == -1]
    if highs.empty or lows.empty:
        return []
    swing_high = float(highs["Level"].iloc[-1])
    swing_low = float(lows["Level"].iloc[-1])
    if swing_high <= swing_low:
        return []
    mid = (swing_high + swing_low) / 2.0
    now = _fmt(ohlc.index[-1])
    return [
        {
            "type": "PREM",
            "direction": "short",
            "price_from": mid,
            "price_to": swing_high,
            "time_from": now,
            "time_to": None,
            "strength": 0.6,
            "mitigated": False,
        },
        {
            "type": "DISC",
            "direction": "long",
            "price_from": swing_low,
            "price_to": mid,
            "time_from": now,
            "time_to": None,
            "strength": 0.6,
            "mitigated": False,
        },
    ]


# ── Public API ────────────────────────────────────────────────────────────────


def analyze(
    ohlc: pd.DataFrame,
    *,
    swing_length: int = 10,
    liquidity_range_pct: float = 0.002,
    include_mitigated: bool = False,
) -> list[dict[str, Any]]:
    """Run SMC analysis on an OHLCV DataFrame and return a unified zone list.

    Parameters
    ----------
    ohlc:
        DataFrame with DatetimeIndex (UTC) and either standard columns
        ``open, high, low, close, volume`` or ORM short names
        ``o, h, l, c, v`` (normalized automatically).
    swing_length:
        Candles to look back/forward for swing point detection (5–20 typical).
    liquidity_range_pct:
        Fraction of price within which highs/lows are treated as equal
        (e.g. ``0.002`` = 0.2 %).
    include_mitigated:
        When *False* (default) only active/unmitigated zones are returned.

    Returns
    -------
    list[dict]
        Zone dicts matching the ``signals.zones`` JSON schema.

    Raises
    ------
    ValueError
        If required columns are missing or fewer than 10 rows are provided.
    TypeError
        If the DataFrame index is not a DatetimeIndex.
    """
    ohlc = _normalize_columns(ohlc)
    _validate(ohlc)
    shl = _smc.swing_highs_lows(ohlc, swing_length=swing_length)
    zones: list[dict[str, Any]] = []
    zones.extend(_extract_bos_choch(ohlc, shl))
    zones.extend(_extract_ob(ohlc, shl, include_mitigated))
    zones.extend(_extract_fvg(ohlc, include_mitigated))
    zones.extend(_extract_liquidity(ohlc, shl, liquidity_range_pct, include_mitigated))
    zones.extend(_extract_premium_discount(ohlc, shl))
    return zones
