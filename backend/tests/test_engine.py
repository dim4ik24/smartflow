"""Tests for app/analysis/engine.py."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import func
from sqlalchemy import select as sa_select

from app.analysis.engine import _avg_sentiment, analyze_symbol_on_close
from app.analysis.scoring import ScoreResult
from app.db.models import AnalysisState, Candle, Signal

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_candle(symbol: str, tf: str, i: int) -> Candle:
    ts = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i)
    return Candle(
        symbol=symbol, timeframe=tf, ts=ts,
        o=50000.0, h=50200.0, l=49800.0, c=50100.0, v=500.0,
    )


def _seed_candles(session: Any, symbol: str, n: int = 55) -> None:
    """Add n candles for both 4h and 1h to the session and flush."""
    for i in range(n):
        session.add(_make_candle(symbol, "4h", i))
        session.add(_make_candle(symbol, "1h", i))


_BOS_LONG = {
    "type": "BOS", "direction": "long",
    "price_from": 50000.0, "price_to": 50000.0,
    "time_from": "2026-01-01T00:00:00Z", "time_to": "2026-01-01T04:00:00Z",
    "strength": 1.0, "mitigated": True,
}
_SCORE_RESULT = ScoreResult(
    symbol="BTC/USDT", side="long", score=80,
    entry_low=49900.0, entry_high=50100.0, sl=49400.0,
    tp1=51100.0, tp2=51600.0, rr=2.4,
    factors={"sweep": True, "ob_retest": True}, zones=[],
)


# ── _avg_sentiment ────────────────────────────────────────────────────────────

class TestAvgSentiment:
    def test_weighted_average(self) -> None:
        news = [
            MagicMock(sentiment=4, importance=2),
            MagicMock(sentiment=8, importance=3),
        ]
        result = _avg_sentiment(news)  # type: ignore[arg-type]
        assert result == pytest.approx((4 * 2 + 8 * 3) / 5)

    def test_empty_returns_none(self) -> None:
        assert _avg_sentiment([]) is None

    def test_none_sentiment_excluded(self) -> None:
        news = [
            MagicMock(sentiment=None, importance=2),
            MagicMock(sentiment=6, importance=1),
        ]
        assert _avg_sentiment(news) == pytest.approx(6.0)  # type: ignore[arg-type]


# ── analyze_symbol_on_close ───────────────────────────────────────────────────

class TestAnalyzeSymbolOnClose:
    async def test_no_candles_returns_none(self, db_session: Any) -> None:
        result = await analyze_symbol_on_close("UNKNOWN/USDT", "1h", db_session)
        assert result is None

    async def test_no_4h_structure_returns_none(self, db_session: Any) -> None:
        _seed_candles(db_session, "BTC/USDT")
        await db_session.flush()

        with (
            patch("app.analysis.engine.smc.analyze", return_value=[]),  # no BOS/CHOCH
        ):
            result = await analyze_symbol_on_close("BTC/USDT", "1h", db_session)
        assert result is None

    async def test_score_below_threshold_no_signal(self, db_session: Any) -> None:
        _seed_candles(db_session, "BTC/USDT")
        await db_session.flush()

        low_score = ScoreResult(
            symbol="BTC/USDT", side="long", score=60,   # below min_score=70
            entry_low=49900.0, entry_high=50100.0, sl=49400.0,
            tp1=51100.0, tp2=51600.0, rr=2.4,
            factors={}, zones=[],
        )
        with (
            patch("app.analysis.engine.smc.analyze", return_value=[_BOS_LONG]),
            patch("app.analysis.engine.score_setup", return_value=low_score),
            patch("app.analysis.engine.get_latest_derivatives",
                  new_callable=AsyncMock, return_value=None),
            patch("app.analysis.engine.get_prev_derivatives",
                  new_callable=AsyncMock, return_value=None),
        ):
            result = await analyze_symbol_on_close("BTC/USDT", "1h", db_session)
        assert result is None

    async def test_score_setup_returns_none_no_signal(self, db_session: Any) -> None:
        _seed_candles(db_session, "ETH/USDT")
        await db_session.flush()

        with (
            patch("app.analysis.engine.smc.analyze", return_value=[_BOS_LONG]),
            patch("app.analysis.engine.score_setup", return_value=None),  # no valid OB / RR
            patch("app.analysis.engine.get_latest_derivatives",
                  new_callable=AsyncMock, return_value=None),
            patch("app.analysis.engine.get_prev_derivatives",
                  new_callable=AsyncMock, return_value=None),
        ):
            result = await analyze_symbol_on_close("ETH/USDT", "1h", db_session)
        assert result is None

    async def test_valid_signal_created_and_flushed(self, db_session: Any) -> None:
        _seed_candles(db_session, "SOL/USDT")
        await db_session.flush()

        with (
            patch("app.analysis.engine.smc.analyze", return_value=[_BOS_LONG]),
            patch("app.analysis.engine.score_setup", return_value=_SCORE_RESULT),
            patch("app.analysis.engine.get_latest_derivatives",
                  new_callable=AsyncMock, return_value=None),
            patch("app.analysis.engine.get_prev_derivatives",
                  new_callable=AsyncMock, return_value=None),
        ):
            signal = await analyze_symbol_on_close("SOL/USDT", "1h", db_session)

        assert signal is not None
        assert isinstance(signal, Signal)
        assert signal.symbol == "SOL/USDT"
        assert signal.side == "long"
        assert signal.score == 80
        assert signal.status == "active"
        assert signal.id is not None      # flush assigned an ID

    async def test_signal_fields_match_score_result(self, db_session: Any) -> None:
        _seed_candles(db_session, "BNB/USDT")
        await db_session.flush()

        with (
            patch("app.analysis.engine.smc.analyze", return_value=[_BOS_LONG]),
            patch("app.analysis.engine.score_setup", return_value=_SCORE_RESULT),
            patch("app.analysis.engine.get_latest_derivatives",
                  new_callable=AsyncMock, return_value=None),
            patch("app.analysis.engine.get_prev_derivatives",
                  new_callable=AsyncMock, return_value=None),
        ):
            signal = await analyze_symbol_on_close("BNB/USDT", "1h", db_session)

        assert signal is not None
        assert signal.entry_low  == pytest.approx(_SCORE_RESULT.entry_low)
        assert signal.entry_high == pytest.approx(_SCORE_RESULT.entry_high)
        assert signal.sl  == pytest.approx(_SCORE_RESULT.sl)
        assert signal.tp1 == pytest.approx(_SCORE_RESULT.tp1)
        assert signal.tp2 == pytest.approx(_SCORE_RESULT.tp2)
        assert signal.rr  == pytest.approx(_SCORE_RESULT.rr)
        assert signal.timeframe == "1h"


# ── Idempotency and deduplication ─────────────────────────────────────────────

class TestIdempotencyAndDedup:
    async def test_same_candle_ts_returns_none(self, db_session: Any) -> None:
        """Second call with the same latest candle ts is a no-op (idempotent)."""
        _seed_candles(db_session, "BTC/USDT", n=55)
        await db_session.flush()

        # The last seeded candle's ts is hour 54.
        last_ts = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=54)
        db_session.add(AnalysisState(
            symbol="BTC/USDT", timeframe="1h", last_candle_ts=last_ts,
        ))
        await db_session.flush()

        result = await analyze_symbol_on_close("BTC/USDT", "1h", db_session)
        assert result is None

    async def test_new_candle_ts_runs_analysis(self, db_session: Any) -> None:
        """Older last_candle_ts in state → analysis runs and produces a signal."""
        _seed_candles(db_session, "BTC/USDT", n=55)
        await db_session.flush()

        # State records an older ts → analysis should proceed.
        old_ts = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=10)
        db_session.add(AnalysisState(
            symbol="BTC/USDT", timeframe="1h", last_candle_ts=old_ts,
        ))
        await db_session.flush()

        with (
            patch("app.analysis.engine.smc.analyze", return_value=[_BOS_LONG]),
            patch("app.analysis.engine.score_setup", return_value=_SCORE_RESULT),
            patch("app.analysis.engine.get_latest_derivatives",
                  new_callable=AsyncMock, return_value=None),
            patch("app.analysis.engine.get_prev_derivatives",
                  new_callable=AsyncMock, return_value=None),
        ):
            signal = await analyze_symbol_on_close("BTC/USDT", "1h", db_session)

        assert signal is not None
        assert signal.symbol == "BTC/USDT"

    async def test_duplicate_active_signal_not_created(self, db_session: Any) -> None:
        """Active signal with overlapping entry zone blocks a duplicate."""
        _seed_candles(db_session, "BTC/USDT", n=55)
        await db_session.flush()

        # Pre-existing active signal whose zone fully overlaps the incoming one.
        existing = Signal(
            symbol="BTC/USDT", side="long", timeframe="1h",
            score=80, status="active",
            entry_low=49900.0, entry_high=50100.0,
            sl=49400.0, tp1=51100.0, tp2=51600.0, rr=2.4,
            factors={}, zones=[],
        )
        db_session.add(existing)
        await db_session.flush()

        with (
            patch("app.analysis.engine.smc.analyze", return_value=[_BOS_LONG]),
            patch("app.analysis.engine.score_setup", return_value=_SCORE_RESULT),
            patch("app.analysis.engine.get_latest_derivatives",
                  new_callable=AsyncMock, return_value=None),
            patch("app.analysis.engine.get_prev_derivatives",
                  new_callable=AsyncMock, return_value=None),
        ):
            result = await analyze_symbol_on_close("BTC/USDT", "1h", db_session)

        assert result is None  # duplicate blocked
        # The original signal is still the only one in the session.
        count = (await db_session.execute(
            sa_select(func.count()).select_from(Signal).where(Signal.symbol == "BTC/USDT")
        )).scalar()
        assert count == 1
