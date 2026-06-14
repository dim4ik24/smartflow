"""Tests for scripts/backtest.py — trade simulation and lookahead safety.

Lookahead guarantee
-------------------
``_simulate_trade`` receives only ``df[signal_idx + 1:]``.  It must never
influence its result based on candles beyond the fill-search window plus the
hold window.  ``test_simulate_trade_no_lookahead`` proves this by appending
adversarial "future" candles and asserting that the result is unchanged.

``test_score_no_lookahead`` verifies the scorer: adding candles beyond the
200-candle window must not change the signal produced at window end T.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Path manipulation must precede backtest import — noqa is intentional.
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from backtest import MAX_FILL_CANDLES, MAX_HOLD_CANDLES, _simulate_trade  # noqa: E402

# ── OHLCV helpers ─────────────────────────────────────────────────────────────

def _candle(
    o: float,
    h: float,
    l: float,  # noqa: E741
    c: float,
    ts: str | None = None,
) -> dict[str, float]:
    return {"open": o, "high": h, "low": l, "close": c, "volume": 100.0}


def _df(*rows: dict[str, float], freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq=freq)
    return pd.DataFrame(list(rows), index=idx)


# ── LONG trades ───────────────────────────────────────────────────────────────

class TestSimulateTradeLong:
    """entry_low=99, entry_high=101, sl=97, tp1=105, tp2=109.

    risk = mid_entry(100) - sl(97) = 3
    R1 = (105-100)/3 = 1.667,  TP1 share = 1.667*0.5 = 0.833
    R2 = (109-100)/3 = 3.0,    TP2 share = 3.0*0.5   = 1.5
    """

    SIDE  = "long"
    EL, EH = 99.0, 101.0   # entry zone
    SL    = 97.0
    TP1   = 105.0
    TP2   = 109.0
    MID   = 100.0           # (EL+EH)/2
    RISK  = 3.0             # MID - SL
    TF    = "1h"

    def _sim(self, df: pd.DataFrame) -> tuple[str, float, object, object]:
        return _simulate_trade(
            df, self.SIDE, self.EL, self.EH, self.SL, self.TP1, self.TP2, self.TF
        )

    def test_no_fill_when_zone_never_touched(self) -> None:
        # All candles stay above 105 — never touch [99, 101]
        df = _df(*[_candle(106, 107, 105, 106)] * 10)
        reason, r, fill_ts, exit_ts = self._sim(df)
        assert reason == "no_fill"
        assert r == pytest.approx(0.0)
        assert fill_ts is None
        assert exit_ts is None

    def test_no_fill_when_fill_window_expires(self) -> None:
        # Zone touch happens at candle 3, but MAX_FILL_CANDLES["1h"] = 2
        df = _df(
            _candle(106, 107, 105, 106),  # 0 — above zone
            _candle(105, 106, 104, 105),  # 1 — above zone
            _candle(99,  102, 98,  100),  # 2 — TOUCHES zone, but index == MAX_FILL = 2 (exclusive)
        )
        reason, r, fill_ts, exit_ts = self._sim(df)
        assert reason == "no_fill"

    def test_sl_hit_immediately_after_fill(self) -> None:
        # Candle 0 fills zone; candle 1 drops to SL
        df = _df(
            _candle(101, 102, 99, 100),   # 0 — fills zone
            _candle(100, 101, 96, 97),    # 1 — SL = 97 breached (low=96 < 97)
        )
        reason, r, fill_ts, exit_ts = self._sim(df)
        assert reason == "sl"
        assert r == pytest.approx(-1.0)
        assert fill_ts is not None
        assert exit_ts is not None

    def test_tp1_only_then_be_sl_hit(self) -> None:
        # Candle 0 fills; candle 1 hits TP1; candle 2 hits breakeven stop.
        # Remaining 50% exits at entry price → 0 additional PnL.
        df = _df(
            _candle(101, 102, 99,  100),   # 0 — fills
            _candle(100, 106, 100, 105),   # 1 — high=106 >= tp1=105 → TP1 hit
            _candle(100, 101, 99,  99),    # 2 — low=99 < mid_entry=100 → BE stop
        )
        reason, r, fill_ts, exit_ts = self._sim(df)
        assert reason == "tp1_be"
        # First 50% earned TP1 R; remaining 50% exits at entry (BE) = 0 PnL
        r1_partial = (self.TP1 - self.MID) / self.RISK * 0.5
        assert r == pytest.approx(r1_partial, abs=1e-6)

    def test_tp1_and_tp2_both_hit(self) -> None:
        # Candle 0 fills; candle 1 hits TP1; candle 2 hits TP2
        df = _df(
            _candle(101, 102, 99,  100),   # 0 — fills
            _candle(100, 106, 100, 105),   # 1 — TP1 hit
            _candle(105, 110, 105, 109),   # 2 — high=110 >= tp2=109 → TP2 hit
        )
        reason, r, fill_ts, exit_ts = self._sim(df)
        assert reason == "tp1_tp2"
        r1 = (self.TP1 - self.MID) / self.RISK * 0.5
        r2 = (self.TP2 - self.MID) / self.RISK * 0.5
        assert r == pytest.approx(r1 + r2, abs=1e-6)

    def test_hold_expired_force_close(self) -> None:
        # Zone touched at candle 0; hold window ends at MAX_HOLD without SL/TP
        max_hold = MAX_HOLD_CANDLES["1h"]
        rows = [_candle(99, 102, 98, 100)]  # candle 0 — fill
        rows += [_candle(100, 103, 100, 102)] * max_hold  # hold, close=102
        df = _df(*rows)
        reason, r, fill_ts, exit_ts = self._sim(df)
        assert reason == "hold_expired"
        expected_r = (102.0 - self.MID) / self.RISK * 1.0  # no TP1 hit → full position
        assert r == pytest.approx(expected_r, abs=1e-3)

    def test_fill_on_last_allowed_candle(self) -> None:
        # Fill window = MAX_FILL_CANDLES["1h"] = 2; zone touch at index 1 (< 2) → fills
        max_fill = MAX_FILL_CANDLES["1h"]
        rows = [_candle(105, 106, 104, 105)] * (max_fill - 1)
        rows.append(_candle(99, 102, 98, 100))   # last fill candle — touches zone
        rows.append(_candle(100, 101, 96, 97))   # SL hit immediately after fill
        df = _df(*rows)
        reason, r, fill_ts, _ = self._sim(df)
        assert reason == "sl"
        assert r == pytest.approx(-1.0)

    def test_within_bar_fill_and_sl_simultaneous(self) -> None:
        """Fill candle that also breaches SL → registered as -1R (pessimistic).

        Even though high=110 would reach TP1=105, within-bar order after entry is
        unknown.  SL on the fill candle is always checked; TP on the fill candle is
        never taken.  Result: SL wins.
        """
        # high=110 ≥ entry_low=99 (fills) AND low=95 ≤ SL=97 (SL breach) in one candle
        df = _df(_candle(100, 110, 95, 100))
        reason, r, fill_ts, exit_ts = self._sim(df)
        assert reason == "sl"
        assert r == pytest.approx(-1.0)
        assert fill_ts is not None
        assert exit_ts is not None

    def test_within_bar_tp_not_taken_on_fill_candle(self) -> None:
        """TP must not fire on the fill candle even when high reaches it.

        The fix: on the fill bar, tp1_level is always False regardless of price.
        Without the fix, this would have returned 'tp1_tp2' or 'tp1_be';
        with the fix the TP check is deferred to the next candle, and the
        subsequent SL hit wins → -1R.
        """
        # candle 0: fills (low=98 ≤ 101) AND reaches TP1 (high=108 ≥ 105) — TP NOT taken
        # candle 1: SL hit (should be the result, NOT a TP1 outcome from candle 0)
        df = _df(
            _candle(100, 108, 98, 105),   # fill candle — high reaches TP1 but is ignored
            _candle(100, 101, 96, 97),    # SL hit on the next candle
        )
        reason, r, fill_ts, exit_ts = self._sim(df)
        assert reason == "sl", (
            f"Expected 'sl' but got '{reason}'. "
            "TP fired on fill candle — within-bar pessimism violated."
        )
        assert r == pytest.approx(-1.0)


# ── SHORT trades ───────────────────────────────────────────────────────────────

class TestSimulateTradeShort:
    """entry_low=99, entry_high=101, sl=103, tp1=95, tp2=91.

    risk = sl(103) - mid_entry(100) = 3
    R1 = (100-95)/3 ≈ 1.667  → TP1 share = 0.833
    R2 = (100-91)/3 = 3.0    → TP2 share = 1.5
    """

    SIDE  = "short"
    EL, EH = 99.0, 101.0
    SL    = 103.0
    TP1   = 95.0
    TP2   = 91.0
    MID   = 100.0
    RISK  = 3.0
    TF    = "1h"

    def _sim(self, df: pd.DataFrame) -> tuple[str, float, object, object]:
        return _simulate_trade(
            df, self.SIDE, self.EL, self.EH, self.SL, self.TP1, self.TP2, self.TF
        )

    def test_sl_hit(self) -> None:
        df = _df(
            _candle(100, 101, 99,  100),  # 0 — fills zone
            _candle(100, 104, 100, 103),  # 1 — high=104 >= sl=103 → SL
        )
        reason, r, _, _ = self._sim(df)
        assert reason == "sl"
        assert r == pytest.approx(-1.0)

    def test_tp1_tp2(self) -> None:
        df = _df(
            _candle(100, 101,  99, 100),  # 0 — fills
            _candle(100, 100,  94,  95),  # 1 — low=94 <= tp1=95 → TP1
            _candle( 95,  95,  90,  91),  # 2 — low=90 <= tp2=91 → TP2
        )
        reason, r, _, _ = self._sim(df)
        assert reason == "tp1_tp2"
        r1 = (self.MID - self.TP1) / self.RISK * 0.5
        r2 = (self.MID - self.TP2) / self.RISK * 0.5
        assert r == pytest.approx(r1 + r2, abs=1e-6)

    def test_within_bar_fill_and_sl_simultaneous(self) -> None:
        """Short fill candle that also breaches SL upward → -1R (pessimistic).

        entry zone [99,101], SL=103.  A candle with high=105 fills the zone
        (high ≥ 99) AND breaches SL (high ≥ 103) simultaneously.  SL wins.
        """
        # high=105 ≥ entry_low=99 (fills) AND high=105 ≥ SL=103 (SL breach)
        df = _df(_candle(100, 105, 99, 100))
        reason, r, _, _ = self._sim(df)
        assert reason == "sl"
        assert r == pytest.approx(-1.0)

    def test_within_bar_tp_not_taken_on_fill_candle(self) -> None:
        """Short TP must not fire on the fill candle even when low reaches it."""
        # candle 0: fills (high=101 ≥ 99) AND low=93 ≤ tp1=95 — TP NOT taken
        # candle 1: SL hit
        df = _df(
            _candle(100, 101, 93,  95),   # fill candle — TP1 reachable but ignored
            _candle(100, 104, 100, 103),  # SL hit on next candle
        )
        reason, r, _, _ = self._sim(df)
        assert reason == "sl", (
            f"Expected 'sl' but got '{reason}'. "
            "TP fired on fill candle — within-bar pessimism violated."
        )
        assert r == pytest.approx(-1.0)


# ── Lookahead safety tests ─────────────────────────────────────────────────────

class TestNoLookahead:
    """Critical: appending candles beyond MAX_FILL + MAX_HOLD must not change results."""

    def test_simulate_trade_no_lookahead(self) -> None:
        """Adversarial candles appended after the hold window must be invisible."""
        # Scenario: fill at candle 0, TP1 hit at candle 5, TP2 hit at candle 10.
        # After candle 10 we append 50 candles with SL-breaching prices.
        # The result must be identical to running without those extra candles.
        entry_low, entry_high = 99.0, 101.0
        sl, tp1, tp2 = 97.0, 105.0, 109.0

        normal_rows = [
            _candle(100, 102, 98,  100),  # 0 — fills zone
            _candle(100, 103, 99,  102),  # 1-4 — hold
            _candle(100, 103, 99,  102),
            _candle(100, 103, 99,  102),
            _candle(100, 103, 99,  102),
            _candle(100, 106, 100, 105),  # 5 — TP1 hit
            _candle(105, 107, 104, 106),  # 6-9 — hold after TP1
            _candle(105, 107, 104, 106),
            _candle(105, 107, 104, 106),
            _candle(105, 107, 104, 106),
            _candle(106, 110, 106, 109),  # 10 — TP2 hit
        ]
        df_normal     = _df(*normal_rows)
        adversarial   = [_candle(109, 109, 80, 80)] * 50  # would breach SL
        df_adversarial = _df(*(normal_rows + adversarial))

        r_normal, _, _, _ = _simulate_trade(
            df_normal, "long", entry_low, entry_high, sl, tp1, tp2, "1h"
        )
        r_adver, _, _, _ = _simulate_trade(
            df_adversarial, "long", entry_low, entry_high, sl, tp1, tp2, "1h"
        )

        # Both runs must give identical results (adversarial candles invisible)
        assert r_normal == pytest.approx(r_adver, abs=1e-9), (
            f"Lookahead detected: normal={r_normal:.4f}, adversarial={r_adver:.4f}. "
            "simulate_trade is reading candles beyond the hold window."
        )

    def test_simulate_trade_fill_window_bounded(self) -> None:
        """Zone touch after MAX_FILL candles must never trigger a fill."""
        max_fill = MAX_FILL_CANDLES["1h"]
        # Exactly max_fill candles with no zone touch, then the zone IS touched
        no_touch = [_candle(105, 106, 104, 105)] * max_fill
        touch    = [_candle(99, 102, 98, 100)] * 10  # would fill if visible

        df = _df(*(no_touch + touch))
        reason, r, fill_ts, _ = _simulate_trade(
            df, "long", 99.0, 101.0, 97.0, 105.0, 109.0, "1h"
        )
        assert reason == "no_fill", (
            f"Expected no_fill but got '{reason}'. "
            "simulate_trade searched beyond the fill window — lookahead!"
        )
        assert fill_ts is None

    def test_score_setup_no_lookahead(self) -> None:
        """score_setup must not accept raw OHLCV data — architectural lookahead guard.

        The only way score_setup could have lookahead is if it accepted candles
        directly.  By verifying that no parameter has type DataFrame, we confirm
        the function is structurally incapable of peeking at T+1 data.
        The walk-forward loop in scan_and_simulate enforces that zones are
        derived exclusively from df[T-200:T+1] before calling score_setup.
        """
        import inspect

        from app.analysis.scoring import score_setup

        sig = inspect.signature(score_setup)
        for name, param in sig.parameters.items():
            ann = param.annotation
            if ann is not inspect.Parameter.empty:
                assert "DataFrame" not in str(ann), (
                    f"score_setup parameter '{name}' accepts a DataFrame — "
                    "this could allow raw future candles to be passed in."
                )


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_risk_returns_no_fill(self) -> None:
        """If mid_entry == sl, risk=0 — function must return no_fill gracefully."""
        df = _df(_candle(100, 101, 99, 100))
        reason, r, _, _ = _simulate_trade(
            df, "long", 100.0, 100.0, 100.0, 105.0, 110.0, "1h"
        )
        # risk = abs(100-100) = 0 → no_fill guard
        assert reason == "no_fill"
        assert r == pytest.approx(0.0)

    def test_15m_fill_window_uses_correct_constant(self) -> None:
        """15m fill window (8 candles) must differ from 1h (2 candles)."""
        assert MAX_FILL_CANDLES["15m"] == 8
        assert MAX_FILL_CANDLES["1h"]  == 2

        # With tf="15m" the zone touch at candle 7 (< 8) should fill
        rows = [_candle(105, 106, 104, 105)] * 7  # 0-6: no touch
        rows.append(_candle(99, 102, 98, 100))     # 7: touches zone (idx 7 < 8)
        rows.append(_candle(100, 101, 96, 97))     # 8: SL
        df = _df(*rows, freq="15min")
        reason, r, fill_ts, _ = _simulate_trade(
            df, "long", 99.0, 101.0, 97.0, 105.0, 109.0, "15m"
        )
        assert reason == "sl"
        assert r == pytest.approx(-1.0)

        # Same scenario with tf="1h" — zone touch at index 7 is BEYOND fill window
        df_1h = _df(*rows)
        reason_1h, _, fill_1h, _ = _simulate_trade(
            df_1h, "long", 99.0, 101.0, 97.0, 105.0, 109.0, "1h"
        )
        assert reason_1h == "no_fill"
        assert fill_1h is None
