"""Signal scoring — confluence of SMC + derivatives + sentiment factors (SPEC §5-6).

``score_setup`` is a pure function: given pre-fetched zones, derivatives, and
sentiment, it applies configurable weights to compute a score 0-100 and builds
the entry/SL/TP geometry.  Returns None when:
  - No active OB matching the trade side is found near the current price.
  - Computed R:R to the nearest liquidity target < settings.score_min_rr.

All weights come from the Settings object (SPEC §6: "ваги в config, не хардкод").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.config import settings as _settings
from app.db.models import DerivativesSnapshot

# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    symbol: str
    side: str                    # "long" | "short"
    score: int                   # 0-100
    entry_low: float
    entry_high: float
    sl: float
    tp1: float
    tp2: float
    rr: float
    factors: dict[str, Any]      # which factors fired + raw values (→ signals.factors)
    zones: list[dict[str, Any]]  # active zones included in signal (→ signals.zones)


# ── Structure detection ────────────────────────────────────────────────────────

def detect_structure_direction(zones: list[dict[str, Any]]) -> str | None:
    """Return 'long' or 'short' from the most recent BOS/CHOCH zone, or None."""
    structural = [z for z in zones if z["type"] in ("BOS", "CHOCH")]
    if not structural:
        return None
    latest = max(structural, key=lambda z: z.get("time_from") or "")
    return str(latest["direction"])


# ── Zone queries (pure helpers) ────────────────────────────────────────────────

def _find_entry_ob(
    zones: list[dict[str, Any]],
    side: str,
    price: float,
    atr: float,
    max_ob_width_pct: float = 0.015,
) -> dict[str, Any] | None:
    """Find the strongest active OB near current price that matches trade side.

    The *max_ob_width_pct* guard rejects zones that span more than the given
    fraction of price (default 1.5 %).  The ``smartmoneyconcepts`` library
    occasionally emits OBs whose ``Top`` equals a distant swing-high and
    ``Bottom`` equals a swing-low — producing a zone that is many ATRs wide.
    Those zones are not tradeable single-candle order blocks and must be
    filtered before the geometry builder uses their price_from/price_to as the
    entry range, which would yield a nonsensical mid-entry point far from the
    current market price.
    """
    candidates = [
        z for z in zones
        if z["type"] == "OB"
        and z["direction"] == side
        and not z.get("mitigated", False)
        and (z["price_to"] - z["price_from"]) / price <= max_ob_width_pct
        and z["price_from"] - atr <= price <= z["price_to"] + atr
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda z: (z.get("strength", 0.0), z.get("time_from") or ""))


def _has_fvg_in_zone(
    zones: list[dict[str, Any]],
    side: str,
    low: float,
    high: float,
) -> bool:
    """True if any active FVG matching side overlaps the [low, high] entry band."""
    return any(
        z["type"] == "FVG"
        and z["direction"] == side
        and not z.get("mitigated", False)
        and z["price_from"] < high
        and z["price_to"] > low
        for z in zones
    )


def _has_sweep(zones: list[dict[str, Any]], side: str) -> bool:
    """True if a confirmed liquidity sweep confirming this side is present.

    LIQ_SWEEP direction='long' = sell-side lows were swept (bullish implication).
    LIQ_SWEEP direction='short' = buy-side highs were swept (bearish implication).
    """
    return any(z["type"] == "LIQ_SWEEP" and z["direction"] == side for z in zones)


def _in_premium_or_discount(
    zones: list[dict[str, Any]],
    side: str,
    price: float,
) -> bool:
    """True if price sits in the DISC zone (for long) or PREM zone (for short)."""
    target = "DISC" if side == "long" else "PREM"
    return any(
        z["type"] == target and z["price_from"] <= price <= z["price_to"]
        for z in zones
    )


def _find_tp_target(
    zones: list[dict[str, Any]],
    side: str,
    entry_mid: float,
) -> float | None:
    """Return the nearest unmitigated liquidity level above (long) or below (short)."""
    if side == "long":
        candidates = [
            z for z in zones
            if z["type"] == "EQH"
            and not z.get("mitigated", False)
            and z["price_from"] > entry_mid
        ]
        if candidates:
            return float(min(candidates, key=lambda z: z["price_from"])["price_from"])
    else:
        candidates = [
            z for z in zones
            if z["type"] == "EQL"
            and not z.get("mitigated", False)
            and z["price_to"] < entry_mid
        ]
        if candidates:
            return float(max(candidates, key=lambda z: z["price_to"])["price_to"])
    return None


# ── Entry geometry ─────────────────────────────────────────────────────────────

def _build_entry_geometry(
    side: str,
    entry_ob: dict[str, Any],
    atr: float,
    zones_all: list[dict[str, Any]],
    min_rr: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Build (entry_low, entry_high, sl, tp1, tp2, rr) or None when RR < min_rr.

    SL is placed 0.5 × ATR beyond the OB boundary.
    TP1 is the nearest liquidity target when available; ATR-based (2× risk) otherwise.
    TP2 is always ATR-based (3× risk from mid-entry), giving room for a runner.
    """
    entry_low  = entry_ob["price_from"]
    entry_high = entry_ob["price_to"]
    mid_entry  = (entry_low + entry_high) / 2.0
    direction  = 1 if side == "long" else -1

    sl = entry_low - 0.5 * atr if side == "long" else entry_high + 0.5 * atr
    risk = abs(mid_entry - sl)
    if risk <= 0:
        return None

    tp_target = _find_tp_target(zones_all, side, mid_entry)
    if tp_target is not None:
        rr = abs(tp_target - mid_entry) / risk
        if rr < min_rr:
            return None          # liquidity target too close for acceptable R:R
        tp1 = tp_target
        tp2 = mid_entry + direction * abs(tp_target - mid_entry) * 1.5
    else:
        # ATR-based fallback: R:R = 2.0 guaranteed
        tp1 = mid_entry + direction * 2.0 * risk
        tp2 = mid_entry + direction * 3.0 * risk
        rr  = 2.0

    return entry_low, entry_high, sl, tp1, tp2, rr


# ── Factor evaluation ──────────────────────────────────────────────────────────

def _compute_factors(
    *,
    side: str,
    current_price: float,
    entry_ob: dict[str, Any],
    zones_entry: list[dict[str, Any]],
    zones_ctx: list[dict[str, Any]],
    derivatives: DerivativesSnapshot | None,
    prev_derivatives: DerivativesSnapshot | None,
    avg_sentiment: float | None,
    s: Settings,
) -> dict[str, Any]:
    all_zones = zones_entry + zones_ctx

    has_sweep = _has_sweep(all_zones, side)

    has_fvg = _has_fvg_in_zone(
        zones_entry, side, entry_ob["price_from"], entry_ob["price_to"]
    )

    # Both 4h and 1h structural direction must agree with the intended side.
    dir_4h = detect_structure_direction(zones_ctx)
    dir_1h = detect_structure_direction(zones_entry)
    structure_aligned = (dir_4h == side) and (dir_1h == side)

    # Derivatives factors.
    funding_rate: float | None = None
    long_short_ratio: float | None = None
    open_interest: float | None = None
    delta_oi: float | None = None
    funding_extreme = False
    oi_rising = False
    lsr_confirms = False

    if derivatives is not None:
        funding_rate = derivatives.funding_rate
        long_short_ratio = derivatives.long_short_ratio
        open_interest = derivatives.open_interest

        if funding_rate is not None:
            thr = s.score_funding_extreme_threshold
            if side == "long" and funding_rate <= -thr or side == "short" and funding_rate >= thr:
                funding_extreme = True

        # ΔOI: rising open interest means more contracts are opening → conviction.
        if (open_interest is not None
                and prev_derivatives is not None
                and prev_derivatives.open_interest is not None):
            delta_oi = open_interest - prev_derivatives.open_interest
            oi_rising = delta_oi > 0

        # Long/short ratio: crowd positioning confirms or contradicts the side.
        if long_short_ratio is not None:
            lsr_confirms = (
                (side == "long" and long_short_ratio >= 1.0) or
                (side == "short" and long_short_ratio < 1.0)
            )

    sentiment_agrees = False
    if avg_sentiment is not None:
        thr = s.score_sentiment_threshold
        sentiment_agrees = (
            (side == "long" and avg_sentiment >= thr) or
            (side == "short" and avg_sentiment <= -thr)
        )

    in_pd = _in_premium_or_discount(all_zones, side, current_price)

    return {
        "sweep":             has_sweep,
        "ob_retest":         True,           # having an entry OB IS the retest
        "fvg":               has_fvg,
        "structure_aligned": structure_aligned,
        "funding_extreme":   funding_extreme,
        "funding_rate":      funding_rate,
        "oi_rising":         oi_rising,
        "open_interest":     open_interest,
        "delta_oi":          delta_oi,
        "lsr_confirms":      lsr_confirms,
        "long_short_ratio":  long_short_ratio,
        "sentiment_agrees":  sentiment_agrees,
        "avg_sentiment":     avg_sentiment,
        "premium_discount":  in_pd,
    }


def _apply_weights(factors: dict[str, Any], s: Settings) -> int:
    """Map factor flags to a 0-100 score using configurable weights."""
    score = 0
    if factors.get("sweep"):
        score += s.score_weight_sweep
    if factors.get("ob_retest"):
        score += s.score_weight_ob_retest
    if factors.get("fvg"):
        score += s.score_weight_fvg
    if factors.get("structure_aligned"):
        score += s.score_weight_structure
    if factors.get("funding_extreme"):
        score += s.score_weight_funding
    if factors.get("oi_rising"):
        score += s.score_weight_oi_rising
    if factors.get("lsr_confirms"):
        score += s.score_weight_lsr
    if factors.get("sentiment_agrees"):
        score += s.score_weight_sentiment
    if factors.get("premium_discount"):
        score += s.score_weight_premium_discount
    return min(100, score)


# ── Public API ─────────────────────────────────────────────────────────────────

def score_setup(
    *,
    symbol: str,
    side: str,
    current_price: float,
    zones_entry: list[dict[str, Any]],
    zones_ctx: list[dict[str, Any]],
    atr: float,
    derivatives: DerivativesSnapshot | None,
    prev_derivatives: DerivativesSnapshot | None = None,
    avg_sentiment: float | None,
    s: Settings | None = None,
) -> ScoreResult | None:
    """Score a potential setup and build the signal geometry.

    Parameters
    ----------
    symbol:
        Trading pair ("BTC/USDT").
    side:
        "long" or "short" — determined by engine.py from 4h context.
    current_price:
        Latest close price (used for zone proximity and premium/discount check).
    zones_entry:
        SMC zones from the entry timeframe (1h / 15m), ``include_mitigated=True``.
    zones_ctx:
        SMC zones from the 4h context timeframe, ``include_mitigated=True``.
    atr:
        Average True Range of the entry timeframe — used for SL placement.
    derivatives:
        Latest DerivativesSnapshot for this symbol, or None.
    prev_derivatives:
        Second-latest snapshot for ΔOI computation; None on cold start.
    avg_sentiment:
        Importance-weighted average news sentiment (−10..+10), or None.
    s:
        Settings instance; defaults to module-level singleton.

    Returns
    -------
    ScoreResult or None
        None when no valid entry OB is found, or R:R < score_min_rr.
    """
    if s is None:
        s = _settings

    entry_ob = _find_entry_ob(zones_entry, side, current_price, atr, s.score_max_ob_width_pct)
    if entry_ob is None:
        return None

    geom = _build_entry_geometry(side, entry_ob, atr, zones_entry + zones_ctx, s.score_min_rr)
    if geom is None:
        return None
    entry_low, entry_high, sl, tp1, tp2, rr = geom

    factors = _compute_factors(
        side=side,
        current_price=current_price,
        entry_ob=entry_ob,
        zones_entry=zones_entry,
        zones_ctx=zones_ctx,
        derivatives=derivatives,
        prev_derivatives=prev_derivatives,
        avg_sentiment=avg_sentiment,
        s=s,
    )
    score = _apply_weights(factors, s)

    # Include all active non-OB zones that influenced the signal for display.
    signal_zones: list[dict[str, Any]] = [entry_ob]
    for z in zones_entry + zones_ctx:
        if z is entry_ob:
            continue
        included = ("LIQ_SWEEP", "FVG", "PREM", "DISC", "EQH", "EQL")
        if z["type"] in included and not z.get("mitigated"):
            signal_zones.append(z)

    return ScoreResult(
        symbol=symbol,
        side=side,
        score=score,
        entry_low=entry_low,
        entry_high=entry_high,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        rr=rr,
        factors=factors,
        zones=signal_zones,
    )
