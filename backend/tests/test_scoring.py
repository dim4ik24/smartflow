"""Tests for app/analysis/scoring.py."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.analysis.scoring import (
    _apply_weights,
    _build_entry_geometry,
    _has_sweep,
    _in_premium_or_discount,
    detect_structure_direction,
    score_setup,
)
from app.config import Settings

# ── Helpers ───────────────────────────────────────────────────────────────────

def _settings(**overrides: object) -> Settings:
    """Return a Settings-like object with test defaults and optional overrides."""
    defaults: dict[str, object] = {
        "score_weight_sweep": 25,
        "score_weight_ob_retest": 20,
        "score_weight_fvg": 10,
        "score_weight_structure": 15,
        "score_weight_funding": 10,
        "score_weight_oi_rising": 3,
        "score_weight_lsr": 2,
        "score_weight_sentiment": 10,
        "score_weight_premium_discount": 5,
        "score_min_rr": 2.0,
        "score_funding_extreme_threshold": 0.0001,
        "score_sentiment_threshold": 1.0,
    }
    defaults.update(overrides)
    s = MagicMock(spec=Settings)
    for k, v in defaults.items():
        setattr(s, k, v)
    return s  # type: ignore[return-value]


def _ob(side: str, low: float, high: float, *, strength: float = 0.8) -> dict:
    return {
        "type": "OB", "direction": side,
        "price_from": low, "price_to": high,
        "time_from": "2026-01-01T00:00:00Z", "time_to": None,
        "strength": strength, "mitigated": False,
    }


def _bos(side: str, ts: str = "2026-01-01T00:00:00Z") -> dict:
    return {
        "type": "BOS", "direction": side,
        "price_from": 100.0, "price_to": 100.0,
        "time_from": ts, "time_to": "2026-01-01T01:00:00Z",
        "strength": 1.0, "mitigated": True,
    }


def _eqh(level: float) -> dict:
    return {
        "type": "EQH", "direction": "short",
        "price_from": level - 0.1, "price_to": level + 0.1,
        "time_from": "2026-01-01T00:00:00Z", "time_to": None,
        "strength": 0.8, "mitigated": False,
    }


def _eql(level: float) -> dict:
    return {
        "type": "EQL", "direction": "long",
        "price_from": level - 0.1, "price_to": level + 0.1,
        "time_from": "2026-01-01T00:00:00Z", "time_to": None,
        "strength": 0.8, "mitigated": False,
    }


def _sweep(side: str) -> dict:
    return {
        "type": "LIQ_SWEEP", "direction": side,
        "price_from": 99.0, "price_to": 101.0,
        "time_from": "2026-01-01T00:00:00Z", "time_to": "2026-01-01T00:30:00Z",
        "strength": 0.9, "mitigated": True,
    }


def _fvg(side: str, low: float, high: float) -> dict:
    return {
        "type": "FVG", "direction": side,
        "price_from": low, "price_to": high,
        "time_from": "2026-01-01T00:00:00Z", "time_to": None,
        "strength": 0.7, "mitigated": False,
    }


def _disc(low: float, high: float) -> dict:
    return {
        "type": "DISC", "direction": "long",
        "price_from": low, "price_to": high,
        "time_from": "2026-01-01T00:00:00Z", "time_to": None,
        "strength": 0.6, "mitigated": False,
    }


def _prem(low: float, high: float) -> dict:
    return {
        "type": "PREM", "direction": "short",
        "price_from": low, "price_to": high,
        "time_from": "2026-01-01T00:00:00Z", "time_to": None,
        "strength": 0.6, "mitigated": False,
    }


def _derivatives(
    *,
    funding_rate: float | None = None,
    open_interest: float | None = None,
    long_short_ratio: float | None = None,
) -> MagicMock:
    d = MagicMock()
    d.funding_rate = funding_rate
    d.open_interest = open_interest
    d.long_short_ratio = long_short_ratio
    return d


# ── detect_structure_direction ────────────────────────────────────────────────

class TestDetectStructureDirection:
    def test_returns_direction_of_latest_bos(self) -> None:
        zones = [
            _bos("short", "2026-01-01T00:00:00Z"),
            _bos("long",  "2026-01-01T02:00:00Z"),  # latest
        ]
        assert detect_structure_direction(zones) == "long"

    def test_choch_takes_precedence_when_latest(self) -> None:
        zones = [
            _bos("long", "2026-01-01T00:00:00Z"),
            {**_bos("short", "2026-01-01T04:00:00Z"), "type": "CHOCH"},
        ]
        assert detect_structure_direction(zones) == "short"

    def test_empty_zones_returns_none(self) -> None:
        assert detect_structure_direction([]) is None

    def test_no_structural_zones_returns_none(self) -> None:
        # Only OBs and FVGs, no BOS/CHOCH
        assert detect_structure_direction([_ob("long", 99, 101), _fvg("long", 100, 102)]) is None


# ── _has_sweep ────────────────────────────────────────────────────────────────

class TestHasSweep:
    def test_matching_sweep_detected(self) -> None:
        assert _has_sweep([_sweep("long")], "long") is True

    def test_opposite_sweep_not_detected(self) -> None:
        assert _has_sweep([_sweep("short")], "long") is False

    def test_no_sweep_zones(self) -> None:
        assert _has_sweep([_ob("long", 99, 101)], "long") is False


# ── _in_premium_or_discount ───────────────────────────────────────────────────

class TestInPremiumOrDiscount:
    def test_long_in_discount_zone(self) -> None:
        assert _in_premium_or_discount([_disc(95, 100)], "long", 98.0) is True

    def test_long_above_discount_zone(self) -> None:
        assert _in_premium_or_discount([_disc(95, 100)], "long", 105.0) is False

    def test_short_in_premium_zone(self) -> None:
        assert _in_premium_or_discount([_prem(100, 110)], "short", 105.0) is True


# ── _build_entry_geometry ─────────────────────────────────────────────────────

class TestBuildEntryGeometry:
    def test_long_geometry_rr_2(self) -> None:
        ob = _ob("long", 98.0, 100.0)
        result = _build_entry_geometry("long", ob, 2.0, [], 2.0)
        assert result is not None
        _entry_low, _entry_high, sl, tp1, tp2, rr = result
        assert sl == pytest.approx(97.0)   # 98 - 0.5*2
        assert rr  == pytest.approx(2.0)
        assert tp2 > tp1                  # tp2 is a runner target, further than tp1

    def test_insufficient_rr_to_eqh_returns_none(self) -> None:
        # mid_entry=99, sl=97, risk=2 → tp_needed for rr>=2 at 103
        # EQH at 100.5 → rr=(100.5-99)/2=0.75 < 2 → reject
        ob   = _ob("long", 98.0, 100.0)
        eqh  = _eqh(100.5)
        result = _build_entry_geometry("long", ob, 2.0, [eqh], 2.0)
        assert result is None

    def test_sufficient_rr_to_eqh_passes(self) -> None:
        # EQH at 105 → rr=(105-99)/2=3.0 >= 2 → pass
        ob   = _ob("long", 98.0, 100.0)
        eqh  = _eqh(105.0)
        result = _build_entry_geometry("long", ob, 2.0, [eqh], 2.0)
        assert result is not None
        *_, rr = result
        assert rr == pytest.approx(2.95)  # _eqh(105.0) → price_from=104.9; rr=(104.9-99)/2

    def test_short_geometry_sl_above_ob(self) -> None:
        ob = _ob("short", 100.0, 102.0)
        result = _build_entry_geometry("short", ob, 2.0, [], 2.0)
        assert result is not None
        _entry_low, _entry_high, sl, tp1, _tp2, _rr = result
        assert sl == pytest.approx(103.0)  # 102 + 0.5*2
        assert tp1 < 100.0                 # TP below entry for short


# ── _apply_weights ────────────────────────────────────────────────────────────

class TestApplyWeights:
    def test_all_flags_give_100(self) -> None:
        s = _settings()
        keys = ("sweep", "ob_retest", "fvg", "structure_aligned",
                "funding_extreme", "oi_rising", "lsr_confirms",
                "sentiment_agrees", "premium_discount")
        f = dict.fromkeys(keys, True)
        assert _apply_weights(f, s) == 100

    def test_only_ob_retest(self) -> None:
        s = _settings()
        f = {"ob_retest": True}
        assert _apply_weights(f, s) == 20   # weight_ob_retest

    def test_capped_at_100_even_if_overweight(self) -> None:
        # If weights sum > 100 due to misconfiguration, we still cap.
        s = _settings(
            score_weight_sweep=50, score_weight_ob_retest=50,
            score_weight_fvg=50,
        )
        f = {"sweep": True, "ob_retest": True, "fvg": True}
        assert _apply_weights(f, s) == 100

    def test_no_flags_zero(self) -> None:
        s = _settings()
        assert _apply_weights({}, s) == 0


# ── score_setup ───────────────────────────────────────────────────────────────

class TestScoreSetup:
    def _base_long_setup(self, price: float = 100.0) -> dict:
        """Minimal kwargs for a valid long setup with OB at [99, 101].

        Excludes derivatives and avg_sentiment so callers can inject them freely.
        """
        return {
            "symbol": "BTC/USDT",
            "side": "long",
            "current_price": price,
            "zones_entry": [_ob("long", 99.0, 101.0)],
            "zones_ctx": [_bos("long")],
            "atr": 1.0,
        }

    def test_no_ob_returns_none(self) -> None:
        result = score_setup(
            symbol="BTC/USDT", side="long", current_price=100.0,
            zones_entry=[], zones_ctx=[], atr=1.0,
            derivatives=None, avg_sentiment=None, s=_settings(),
        )
        assert result is None

    def test_ob_present_returns_result(self) -> None:
        result = score_setup(
            **self._base_long_setup(), derivatives=None, avg_sentiment=None, s=_settings()
        )
        assert result is not None
        assert result.side == "long"
        assert result.score >= 20           # at minimum ob_retest weight

    def test_rr_insufficient_to_eqh_returns_none(self) -> None:
        # EQH at 100.5 → price_from=100.4; mid=100, sl=98.5, risk=1.5 → rr≈0.27 < 2
        result = score_setup(
            symbol="BTC/USDT", side="long", current_price=100.0,
            zones_entry=[_ob("long", 99.0, 101.0), _eqh(100.5)],
            zones_ctx=[_bos("long")],
            atr=1.0, derivatives=None, avg_sentiment=None,
            s=_settings(score_min_rr=2.0),
        )
        assert result is None

    def test_all_factors_maximum_score(self) -> None:
        # All 9 factors fire → score = 100
        der = _derivatives(funding_rate=-0.001, long_short_ratio=1.5, open_interest=1e6)
        prev_der = _derivatives(open_interest=0.9e6)   # lower than current → ΔOI > 0
        result = score_setup(
            symbol="BTC/USDT", side="long", current_price=100.0,
            zones_entry=[
                _ob("long", 99.0, 101.0),
                _sweep("long"),
                _fvg("long", 99.5, 100.5),
                _bos("long"),                   # 1h structure
                _disc(95.0, 102.0),             # price (100) inside discount zone
            ],
            zones_ctx=[_bos("long")],           # 4h structure agrees
            atr=1.0,
            derivatives=der,
            prev_derivatives=prev_der,
            avg_sentiment=3.0,                  # positive → sentiment_agrees for long
            s=_settings(),
        )
        assert result is not None
        assert result.score == 100

    def test_funding_negative_confirms_long(self) -> None:
        der = _derivatives(funding_rate=-0.001)   # below -0.0001 threshold
        result = score_setup(
            **self._base_long_setup(), derivatives=der, avg_sentiment=None, s=_settings()
        )
        assert result is not None
        assert result.factors["funding_extreme"] is True

    def test_funding_positive_does_not_confirm_long(self) -> None:
        der = _derivatives(funding_rate=0.001)    # positive = longs paying (wrong for long setup)
        result = score_setup(
            **self._base_long_setup(), derivatives=der, avg_sentiment=None, s=_settings()
        )
        assert result is not None
        assert result.factors["funding_extreme"] is False

    def test_funding_positive_confirms_short(self) -> None:
        der = _derivatives(funding_rate=0.001)
        result = score_setup(
            symbol="BTC/USDT", side="short", current_price=100.0,
            zones_entry=[_ob("short", 99.0, 101.0)],
            zones_ctx=[_bos("short")],
            atr=1.0, derivatives=der, avg_sentiment=None,
            s=_settings(),
        )
        assert result is not None
        assert result.factors["funding_extreme"] is True

    def test_sentiment_agrees_long(self) -> None:
        result = score_setup(
            **self._base_long_setup(), derivatives=None, avg_sentiment=2.5, s=_settings()
        )
        assert result is not None
        assert result.factors["sentiment_agrees"] is True
        assert result.score >= 20 + 10   # ob_retest + sentiment

    def test_sentiment_disagrees_for_long(self) -> None:
        result = score_setup(
            **self._base_long_setup(), derivatives=None, avg_sentiment=-2.5, s=_settings()
        )
        assert result is not None
        assert result.factors["sentiment_agrees"] is False

    def test_structure_both_aligned(self) -> None:
        result = score_setup(
            symbol="BTC/USDT", side="long", current_price=100.0,
            zones_entry=[_ob("long", 99.0, 101.0), _bos("long", "2026-01-01T03:00:00Z")],
            zones_ctx=[_bos("long", "2026-01-01T01:00:00Z")],
            atr=1.0, derivatives=None, avg_sentiment=None,
            s=_settings(),
        )
        assert result is not None
        assert result.factors["structure_aligned"] is True
        assert result.score >= 20 + 15   # ob_retest + structure

    def test_result_contains_entry_ob_in_zones(self) -> None:
        result = score_setup(
            **self._base_long_setup(), derivatives=None, avg_sentiment=None, s=_settings()
        )
        assert result is not None
        ob_zones = [z for z in result.zones if z["type"] == "OB"]
        assert len(ob_zones) == 1

    def test_lsr_confirms_long_when_lsr_ge_1(self) -> None:
        der = _derivatives(long_short_ratio=1.2)
        result = score_setup(
            **self._base_long_setup(), derivatives=der, avg_sentiment=None, s=_settings()
        )
        assert result is not None
        assert result.factors["lsr_confirms"] is True

    def test_lsr_does_not_confirm_long_when_lsr_lt_1(self) -> None:
        der = _derivatives(long_short_ratio=0.8)
        result = score_setup(
            **self._base_long_setup(), derivatives=der, avg_sentiment=None, s=_settings()
        )
        assert result is not None
        assert result.factors["lsr_confirms"] is False

    def test_oi_rising_when_oi_increases(self) -> None:
        der = _derivatives(open_interest=1_100_000.0)
        prev_der = _derivatives(open_interest=1_000_000.0)
        result = score_setup(
            **self._base_long_setup(),
            derivatives=der, prev_derivatives=prev_der,
            avg_sentiment=None, s=_settings(),
        )
        assert result is not None
        assert result.factors["oi_rising"] is True
        assert result.factors["delta_oi"] == pytest.approx(100_000.0)

    def test_oi_not_rising_when_oi_decreases(self) -> None:
        der = _derivatives(open_interest=900_000.0)
        prev_der = _derivatives(open_interest=1_000_000.0)
        result = score_setup(
            **self._base_long_setup(),
            derivatives=der, prev_derivatives=prev_der,
            avg_sentiment=None, s=_settings(),
        )
        assert result is not None
        assert result.factors["oi_rising"] is False

    def test_oi_not_rising_when_no_prev(self) -> None:
        der = _derivatives(open_interest=1_000_000.0)
        result = score_setup(
            **self._base_long_setup(),
            derivatives=der, prev_derivatives=None,
            avg_sentiment=None, s=_settings(),
        )
        assert result is not None
        assert result.factors["oi_rising"] is False
        assert result.factors["delta_oi"] is None
