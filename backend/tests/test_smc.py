"""Tests for app/analysis/smc.py — SMC analysis wrapper.

Synthetic OHLCV datasets are constructed to produce known patterns so
that each test asserts specific structural properties rather than exact
library outputs (which vary with internal algorithm versions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis.smc import ZONE_TYPES, analyze

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_trending(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Reproducible OHLCV: uptrend → consolidation → downtrend.

    120 candles is enough for swing detection with swing_length=5 and
    ensures BOS, OB, FVG, and liquidity zones are present.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    third = n // 3
    changes = np.concatenate(
        [
            rng.normal(0.003, 0.004, third),         # uptrend
            rng.normal(0.000, 0.003, third),         # consolidation
            rng.normal(-0.003, 0.004, n - 2 * third),  # downtrend
        ]
    )
    closes = 60_000.0 * np.cumprod(1 + changes)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    wicks = np.abs(rng.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + wicks)
    lows = np.minimum(opens, closes) * (1 - wicks)
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.uniform(10, 200, n),
        },
        index=idx,
    )


def _make_fvg(bullish: bool = True) -> pd.DataFrame:
    """40-candle OHLC with a single deliberate FVG at position 20.

    Bullish FVG: candle[19].high < candle[21].low  (gap below current price).
    Bearish FVG: candle[19].low  > candle[21].high (gap above current price).
    """
    n, k, base = 40, 20, 100.0
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    opens = np.full(n, base, dtype=float)
    closes = np.full(n, base, dtype=float)
    highs = np.full(n, base + 0.5, dtype=float)
    lows = np.full(n, base - 0.5, dtype=float)
    volumes = np.full(n, 100.0)

    if bullish:
        # prev candle: high = 101 (below the gap)
        highs[k - 1] = base + 1.0
        # FVG candle: bullish, body between 101.5 and 104
        opens[k], closes[k] = base + 1.5, base + 4.0
        highs[k], lows[k] = base + 4.0, base + 1.5
        # next candle: low = 103 > prev high 101 → gap [101, 103]
        lows[k + 1] = base + 3.0
        opens[k + 1], closes[k + 1] = base + 3.5, base + 5.0
        highs[k + 1] = base + 6.0
    else:
        # prev candle: low = 99 (above the gap)
        lows[k - 1] = base - 1.0
        # FVG candle: bearish, body between 98.5 and 96
        opens[k], closes[k] = base - 1.5, base - 4.0
        lows[k], highs[k] = base - 4.0, base - 1.5
        # next candle: high = 97 < prev low 99 → gap [97, 99]
        highs[k + 1] = base - 3.0
        opens[k + 1], closes[k + 1] = base - 3.5, base - 5.0
        lows[k + 1] = base - 6.0

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trending() -> pd.DataFrame:
    return _make_trending()


@pytest.fixture(scope="module")
def trending_zones(trending: pd.DataFrame) -> list[dict]:
    return analyze(trending, swing_length=5, include_mitigated=True)


# ── Validation ────────────────────────────────────────────────────────────────


def test_validate_missing_columns() -> None:
    idx = pd.date_range("2026-01-01", periods=15, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": np.ones(15), "high": np.ones(15), "low": np.ones(15)}, index=idx)
    with pytest.raises(ValueError, match="missing columns"):
        analyze(df)


def test_validate_non_datetime_index() -> None:
    df = pd.DataFrame(
        {col: np.ones(20) for col in ("open", "high", "low", "close", "volume")},
        index=range(20),
    )
    with pytest.raises(TypeError, match="DatetimeIndex"):
        analyze(df)


def test_validate_too_few_candles() -> None:
    idx = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {col: np.ones(5) for col in ("open", "high", "low", "close", "volume")},
        index=idx,
    )
    with pytest.raises(ValueError, match="10 candles"):
        analyze(df)


# ── Return type & schema ──────────────────────────────────────────────────────


def test_analyze_returns_list(trending_zones: list[dict]) -> None:
    assert isinstance(trending_zones, list)


def test_analyze_not_empty(trending_zones: list[dict]) -> None:
    assert len(trending_zones) > 0


def test_zone_required_keys(trending_zones: list[dict]) -> None:
    required = {
        "type", "direction", "price_from", "price_to",
        "time_from", "time_to", "strength", "mitigated",
    }
    for z in trending_zones:
        assert required.issubset(z.keys()), f"Missing keys in zone: {z}"


def test_zone_type_values(trending_zones: list[dict]) -> None:
    for z in trending_zones:
        assert z["type"] in ZONE_TYPES, f"Unknown type: {z['type']}"


def test_zone_direction_values(trending_zones: list[dict]) -> None:
    for z in trending_zones:
        assert z["direction"] in ("long", "short"), f"Bad direction: {z['direction']}"


def test_zone_price_from_le_price_to(trending_zones: list[dict]) -> None:
    for z in trending_zones:
        assert z["price_from"] <= z["price_to"], (
            f"price_from > price_to in {z['type']}: {z['price_from']} > {z['price_to']}"
        )


def test_zone_strength_range(trending_zones: list[dict]) -> None:
    for z in trending_zones:
        assert 0.0 <= z["strength"] <= 1.0, f"strength out of range: {z}"


def test_zone_mitigated_is_bool(trending_zones: list[dict]) -> None:
    for z in trending_zones:
        assert isinstance(z["mitigated"], bool), f"mitigated not bool: {z}"


def test_no_nan_price_fields(trending_zones: list[dict]) -> None:
    for z in trending_zones:
        assert z["price_from"] == z["price_from"], f"NaN price_from in {z}"  # NaN != NaN
        assert z["price_to"] == z["price_to"], f"NaN price_to in {z}"


# ── FVG detection ─────────────────────────────────────────────────────────────


def test_bullish_fvg_detected() -> None:
    df = _make_fvg(bullish=True)
    zones = analyze(df, include_mitigated=True)
    fvgs = [z for z in zones if z["type"] == "FVG" and z["direction"] == "long"]
    assert fvgs, "No bullish FVG found in synthetic data with explicit gap"
    # Gap between prev high (101) and next low (103): price_from ≈ 101, price_to ≈ 103
    gap = fvgs[0]
    assert gap["price_from"] >= 100.0
    assert gap["price_to"] <= 105.0


def test_bearish_fvg_detected() -> None:
    df = _make_fvg(bullish=False)
    zones = analyze(df, include_mitigated=True)
    fvgs = [z for z in zones if z["type"] == "FVG" and z["direction"] == "short"]
    assert fvgs, "No bearish FVG found in synthetic data with explicit gap"
    gap = fvgs[0]
    assert gap["price_from"] < gap["price_to"]


def test_fvg_price_range_valid() -> None:
    for bullish in (True, False):
        df = _make_fvg(bullish=bullish)
        zones = analyze(df, include_mitigated=True)
        for z in (z for z in zones if z["type"] == "FVG"):
            assert z["price_from"] < z["price_to"], f"FVG zero-width or inverted: {z}"


# ── Premium / Discount ────────────────────────────────────────────────────────


def test_prem_disc_present(trending_zones: list[dict]) -> None:
    types = {z["type"] for z in trending_zones}
    assert "PREM" in types, "No PREM zone found"
    assert "DISC" in types, "No DISC zone found"


def test_prem_above_disc(trending_zones: list[dict]) -> None:
    prem = next(z for z in trending_zones if z["type"] == "PREM")
    disc = next(z for z in trending_zones if z["type"] == "DISC")
    # Midpoint is the shared boundary: DISC.price_to == PREM.price_from
    assert abs(disc["price_to"] - prem["price_from"]) < 1e-6, (
        f"DISC top ({disc['price_to']}) != PREM bottom ({prem['price_from']})"
    )
    assert prem["price_to"] > disc["price_from"]


def test_prem_direction_short(trending_zones: list[dict]) -> None:
    prem = next(z for z in trending_zones if z["type"] == "PREM")
    assert prem["direction"] == "short"


def test_disc_direction_long(trending_zones: list[dict]) -> None:
    disc = next(z for z in trending_zones if z["type"] == "DISC")
    assert disc["direction"] == "long"


# ── include_mitigated flag ────────────────────────────────────────────────────


def test_default_excludes_mitigated_ob_fvg(trending: pd.DataFrame) -> None:
    zones = analyze(trending, swing_length=5, include_mitigated=False)
    for z in zones:
        if z["type"] in ("OB", "FVG", "EQH", "EQL"):
            assert not z["mitigated"], f"Mitigated zone leaked with include_mitigated=False: {z}"


def test_include_mitigated_adds_zones(trending: pd.DataFrame) -> None:
    without = analyze(trending, swing_length=5, include_mitigated=False)
    with_ = analyze(trending, swing_length=5, include_mitigated=True)
    assert len(with_) >= len(without), "include_mitigated=True should not reduce zone count"


# ── Column normalization ───────────────────────────────────────────────────────


def test_analyze_accepts_orm_columns() -> None:
    """analyze() must accept ORM short names (o/h/l/c/v) without external rename."""
    df = _make_trending()
    orm_df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
    zones = analyze(orm_df, swing_length=5, include_mitigated=True)
    assert isinstance(zones, list)
    assert len(zones) > 0, "ORM-format DataFrame produced no zones"


def test_analyze_orm_and_standard_same_result() -> None:
    """ORM-format and standard-format DataFrames must produce identical zones."""
    df = _make_trending()
    orm_df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
    zones_std = analyze(df, swing_length=5, include_mitigated=True)
    zones_orm = analyze(orm_df, swing_length=5, include_mitigated=True)
    assert len(zones_std) == len(zones_orm), (
        f"Zone count differs: standard={len(zones_std)}, orm={len(zones_orm)}"
    )
    for std, orm in zip(zones_std, zones_orm, strict=True):
        assert std == orm, f"Zone mismatch:\n  standard: {std}\n  orm:      {orm}"


# ── confirmed_only (lookahead bias guard) ─────────────────────────────────────


def _zone_keys(zones: list[dict]) -> set[tuple]:
    """Stable identity key for a zone: (type, time_from, price_from rounded)."""
    return {
        (z["type"], z["time_from"], round(z["price_from"], 4))
        for z in zones
        if z["type"] not in ("PREM", "DISC")  # PREM/DISC time_from reflects run-time, not event
    }


def test_confirmed_only_is_subset_of_raw() -> None:
    """confirmed_only=True can only remove zones, never add new ones."""
    df = _make_trending(n=80, seed=42)
    raw = analyze(df, swing_length=5, include_mitigated=True, confirmed_only=False)
    confirmed = analyze(df, swing_length=5, include_mitigated=True, confirmed_only=True)
    assert _zone_keys(confirmed).issubset(_zone_keys(raw))
    assert len(confirmed) <= len(raw)


def test_confirmed_only_removes_choch_from_edge_swing() -> None:
    """Concrete lookahead proof.

    With n=80, seed=42, swing_length=5 the library marks position 79
    (the very last bar) as a swing LOW.  Confirming a swing at the last bar
    requires 5 future candles that do not exist yet — this is lookahead bias.
    That swing creates a CHOCH zone.

    confirmed_only=True zeros shl positions 75-79 so the unconfirmed swing
    at 79 is ignored.  The CHOCH must disappear from the result.
    """
    df = _make_trending(n=80, seed=42)
    swing_length = 5

    raw = analyze(df, swing_length=swing_length, include_mitigated=True, confirmed_only=False)
    confirmed = analyze(df, swing_length=swing_length, include_mitigated=True, confirmed_only=True)

    raw_choch = [z for z in raw if z["type"] == "CHOCH"]
    conf_choch = [z for z in confirmed if z["type"] == "CHOCH"]

    assert raw_choch, (
        "Expected at least one CHOCH in unfiltered analysis for n=80, seed=42, swing_length=5"
    )
    assert len(conf_choch) < len(raw_choch), (
        f"confirmed_only=True must reduce CHOCH count "
        f"(raw={len(raw_choch)}, confirmed={len(conf_choch)})"
    )


def test_confirmed_only_stable_on_extension() -> None:
    """Stability invariant: zones confirmed on D must persist when D is extended.

    If a zone from confirmed_only=True on D[0..N] disappeared after analysing
    D[0..N+swing_length], it must have depended on the tail of D as implicit
    future candles — i.e. it was still biased.

    We slice both datasets from the same base series so the first N candles
    are byte-for-byte identical.
    """
    swing_length = 5
    n_short = 80

    df_base = _make_trending(n=n_short + swing_length, seed=42)
    df_short = df_base.iloc[:n_short]
    df_long = df_base  # full n_short + swing_length candles

    keys_short = _zone_keys(
        analyze(df_short, swing_length=swing_length, include_mitigated=True, confirmed_only=True)
    )
    keys_long = _zone_keys(
        analyze(df_long, swing_length=swing_length, include_mitigated=True, confirmed_only=True)
    )

    assert keys_short.issubset(keys_long), (
        "Confirmed zones from shorter dataset disappeared after extension — "
        "they still depended on future candles.\n"
        f"Missing after extension: {keys_short - keys_long}"
    )
