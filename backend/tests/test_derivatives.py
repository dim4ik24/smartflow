"""Tests for app/collectors/derivatives.py."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import ccxt
import pytest

from app.collectors.derivatives import (
    _call_with_retry,
    _fetch_funding_rate,
    _fetch_long_short_ratio,
    _fetch_open_interest,
    collect_derivatives,
    fetch_snapshot_for_symbol,
    get_latest_derivatives,
)
from app.db.models import DerivativesSnapshot

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_exchange(**methods: Any) -> MagicMock:
    ex = MagicMock()
    for name, val in methods.items():
        setattr(ex, name, val)
    return ex


# ── _call_with_retry ──────────────────────────────────────────────────────────

class TestCallWithRetry:
    async def test_success_first_attempt(self) -> None:
        fn = AsyncMock(return_value={"data": 42})
        result = await _call_with_retry(fn, "sym", label="t")
        assert result == {"data": 42}
        fn.assert_awaited_once_with("sym")

    async def test_retries_on_network_error_then_succeeds(self) -> None:
        fn = AsyncMock(side_effect=[ccxt.NetworkError("down"), {"data": 1}])
        with patch("app.collectors.derivatives.asyncio.sleep", new_callable=AsyncMock):
            result = await _call_with_retry(fn, label="t")
        assert result == {"data": 1}
        assert fn.await_count == 2

    async def test_exhausts_retries_returns_none(self) -> None:
        fn = AsyncMock(side_effect=ccxt.NetworkError("always down"))
        with patch("app.collectors.derivatives.asyncio.sleep", new_callable=AsyncMock) as slp:
            result = await _call_with_retry(fn, label="t")
        assert result is None
        assert fn.await_count == 3          # _MAX_RETRIES
        assert slp.await_count == 2         # sleep between attempts, not after last

    async def test_not_supported_returns_none_immediately(self) -> None:
        fn = AsyncMock(side_effect=ccxt.NotSupported("nope"))
        result = await _call_with_retry(fn, label="t")
        assert result is None
        fn.assert_awaited_once()            # no retry on terminal error

    async def test_backoff_doubles(self) -> None:
        fn = AsyncMock(side_effect=ccxt.NetworkError("err"))
        with patch("app.collectors.derivatives.asyncio.sleep", new_callable=AsyncMock) as slp:
            await _call_with_retry(fn, label="t")
        delays = [c.args[0] for c in slp.await_args_list]
        assert delays == [1.0, 2.0]         # 2**0, 2**1

    async def test_unexpected_exception_returns_none(self) -> None:
        fn = AsyncMock(side_effect=ValueError("unexpected"))
        result = await _call_with_retry(fn, label="t")
        assert result is None
        fn.assert_awaited_once()


# ── _fetch_funding_rate ───────────────────────────────────────────────────────

class TestFetchFundingRate:
    async def test_success(self) -> None:
        ex = _make_exchange(fetch_funding_rate=AsyncMock(return_value={"fundingRate": 0.0001}))
        assert await _fetch_funding_rate(ex, "BTC/USDT") == pytest.approx(0.0001)

    async def test_missing_key_returns_none(self) -> None:
        ex = _make_exchange(fetch_funding_rate=AsyncMock(return_value={"other": 0.0}))
        assert await _fetch_funding_rate(ex, "BTC/USDT") is None

    async def test_not_supported_returns_none(self) -> None:
        ex = _make_exchange(
            fetch_funding_rate=AsyncMock(side_effect=ccxt.NotSupported("nope"))
        )
        assert await _fetch_funding_rate(ex, "BTC/USDT") is None

    async def test_network_error_retries_then_returns_none(self) -> None:
        ex = _make_exchange(
            fetch_funding_rate=AsyncMock(side_effect=ccxt.NetworkError("err"))
        )
        with patch("app.collectors.derivatives.asyncio.sleep", new_callable=AsyncMock):
            result = await _fetch_funding_rate(ex, "BTC/USDT")
        assert result is None


# ── _fetch_open_interest ──────────────────────────────────────────────────────

class TestFetchOpenInterest:
    async def test_prefers_value_field(self) -> None:
        ex = _make_exchange(fetch_open_interest=AsyncMock(
            return_value={"openInterestValue": 1_000_000.0, "openInterestAmount": 20.0}
        ))
        assert await _fetch_open_interest(ex, "BTC/USDT") == pytest.approx(1_000_000.0)

    async def test_fallback_to_amount_field(self) -> None:
        ex = _make_exchange(fetch_open_interest=AsyncMock(
            return_value={"openInterestAmount": 20.0}
        ))
        assert await _fetch_open_interest(ex, "BTC/USDT") == pytest.approx(20.0)

    async def test_legacy_open_interest_field(self) -> None:
        ex = _make_exchange(fetch_open_interest=AsyncMock(
            return_value={"openInterest": 15.5}
        ))
        assert await _fetch_open_interest(ex, "BTC/USDT") == pytest.approx(15.5)

    async def test_not_supported_returns_none(self) -> None:
        ex = _make_exchange(
            fetch_open_interest=AsyncMock(side_effect=ccxt.NotSupported("nope"))
        )
        assert await _fetch_open_interest(ex, "BTC/USDT") is None

    async def test_network_error_returns_none(self) -> None:
        ex = _make_exchange(
            fetch_open_interest=AsyncMock(side_effect=ccxt.NetworkError("timeout"))
        )
        with patch("app.collectors.derivatives.asyncio.sleep", new_callable=AsyncMock):
            result = await _fetch_open_interest(ex, "BTC/USDT")
        assert result is None


# ── _fetch_long_short_ratio ───────────────────────────────────────────────────

class TestFetchLongShortRatio:
    async def test_success_list_response(self) -> None:
        ex = _make_exchange(fetch_long_short_ratio=AsyncMock(
            return_value=[{"longShortRatio": 1.5, "timestamp": 123}]
        ))
        assert await _fetch_long_short_ratio(ex, "BTC/USDT") == pytest.approx(1.5)

    async def test_takes_last_item_from_list(self) -> None:
        ex = _make_exchange(fetch_long_short_ratio=AsyncMock(
            return_value=[
                {"longShortRatio": 1.0, "timestamp": 100},
                {"longShortRatio": 1.8, "timestamp": 200},
            ]
        ))
        assert await _fetch_long_short_ratio(ex, "BTC/USDT") == pytest.approx(1.8)

    async def test_empty_list_returns_none(self) -> None:
        ex = _make_exchange(fetch_long_short_ratio=AsyncMock(return_value=[]))
        assert await _fetch_long_short_ratio(ex, "BTC/USDT") is None

    async def test_not_supported_returns_none(self) -> None:
        ex = _make_exchange(
            fetch_long_short_ratio=AsyncMock(side_effect=ccxt.NotSupported("no"))
        )
        assert await _fetch_long_short_ratio(ex, "BTC/USDT") is None

    async def test_passes_5m_timeframe(self) -> None:
        ex = _make_exchange(fetch_long_short_ratio=AsyncMock(
            return_value=[{"longShortRatio": 1.2}]
        ))
        await _fetch_long_short_ratio(ex, "BTC/USDT")
        ex.fetch_long_short_ratio.assert_awaited_once_with("BTC/USDT", "5m")


# ── fetch_snapshot_for_symbol ─────────────────────────────────────────────────

class TestFetchSnapshotForSymbol:
    async def test_all_metrics_success(self) -> None:
        ex = _make_exchange(
            fetch_funding_rate=AsyncMock(return_value={"fundingRate": 0.0001}),
            fetch_open_interest=AsyncMock(return_value={"openInterestValue": 1_000_000.0}),
            fetch_long_short_ratio=AsyncMock(return_value=[{"longShortRatio": 1.3}]),
        )
        snap = await fetch_snapshot_for_symbol(ex, "BTC/USDT")
        assert snap is not None
        assert snap.symbol == "BTC/USDT"
        assert snap.funding_rate == pytest.approx(0.0001)
        assert snap.open_interest == pytest.approx(1_000_000.0)
        assert snap.long_short_ratio == pytest.approx(1.3)

    async def test_partial_failure_returns_snapshot_with_nones(self) -> None:
        ex = _make_exchange(
            fetch_funding_rate=AsyncMock(return_value={"fundingRate": 0.0002}),
            fetch_open_interest=AsyncMock(side_effect=ccxt.NotSupported("no")),
            fetch_long_short_ratio=AsyncMock(side_effect=ccxt.NotSupported("no")),
        )
        snap = await fetch_snapshot_for_symbol(ex, "ETH/USDT")
        assert snap is not None
        assert snap.funding_rate == pytest.approx(0.0002)
        assert snap.open_interest is None
        assert snap.long_short_ratio is None

    async def test_all_fail_returns_none(self) -> None:
        ex = _make_exchange(
            fetch_funding_rate=AsyncMock(side_effect=ccxt.NotSupported("no")),
            fetch_open_interest=AsyncMock(side_effect=ccxt.NotSupported("no")),
            fetch_long_short_ratio=AsyncMock(side_effect=ccxt.NotSupported("no")),
        )
        assert await fetch_snapshot_for_symbol(ex, "SOL/USDT") is None

    async def test_ts_is_utc_aware(self) -> None:
        ex = _make_exchange(
            fetch_funding_rate=AsyncMock(return_value={"fundingRate": 0.0}),
            fetch_open_interest=AsyncMock(side_effect=ccxt.NotSupported("no")),
            fetch_long_short_ratio=AsyncMock(side_effect=ccxt.NotSupported("no")),
        )
        snap = await fetch_snapshot_for_symbol(ex, "BTC/USDT")
        assert snap is not None
        assert snap.ts.tzinfo is not None


# ── get_latest_derivatives ────────────────────────────────────────────────────

class TestGetLatestDerivatives:
    async def test_returns_most_recent_row(self, db_session: Any) -> None:
        older = DerivativesSnapshot(
            symbol="BTC/USDT",
            ts=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            funding_rate=0.0001,
            open_interest=None,
            long_short_ratio=None,
        )
        newer = DerivativesSnapshot(
            symbol="BTC/USDT",
            ts=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            funding_rate=0.0003,
            open_interest=500_000.0,
            long_short_ratio=1.2,
        )
        db_session.add_all([older, newer])
        await db_session.flush()

        result = await get_latest_derivatives("BTC/USDT", db_session)
        assert result is not None
        assert result.funding_rate == pytest.approx(0.0003)

    async def test_unknown_symbol_returns_none(self, db_session: Any) -> None:
        result = await get_latest_derivatives("NONEXISTENT/USDT", db_session)
        assert result is None

    async def test_filters_by_symbol(self, db_session: Any) -> None:
        snap = DerivativesSnapshot(
            symbol="ETH/USDT",
            ts=datetime(2026, 1, 2, tzinfo=UTC),
            funding_rate=0.0005,
            open_interest=None,
            long_short_ratio=None,
        )
        db_session.add(snap)
        await db_session.flush()

        result = await get_latest_derivatives("BTC/USDT", db_session)
        assert result is None


# ── collect_derivatives ───────────────────────────────────────────────────────

class TestCollectDerivatives:
    async def test_saves_snapshots_for_all_symbols(self) -> None:
        mock_ex = MagicMock()
        mock_ex.load_markets = AsyncMock()
        mock_ex.close = AsyncMock()
        mock_ex.fetch_funding_rate = AsyncMock(return_value={"fundingRate": 0.0001})
        mock_ex.fetch_open_interest = AsyncMock(
            return_value={"openInterestValue": 500_000.0}
        )
        mock_ex.fetch_long_short_ratio = AsyncMock(
            return_value=[{"longShortRatio": 1.4}]
        )

        # Use MagicMock for the session so that add_all() stays synchronous.
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        with (
            patch("app.collectors.derivatives._build_exchange", return_value=mock_ex),
            patch("app.collectors.derivatives.AsyncSessionLocal", return_value=mock_session),
            patch("app.collectors.derivatives.settings") as mock_s,
        ):
            mock_s.watched_symbols = ["BTC/USDT", "ETH/USDT"]
            await collect_derivatives()

        mock_ex.load_markets.assert_awaited_once()
        mock_ex.close.assert_awaited_once()
        mock_session.add_all.assert_called_once()
        saved: list[DerivativesSnapshot] = mock_session.add_all.call_args[0][0]
        assert len(saved) == 2
        assert all(isinstance(s, DerivativesSnapshot) for s in saved)
        symbols = {s.symbol for s in saved}
        assert symbols == {"BTC/USDT", "ETH/USDT"}
        mock_session.commit.assert_awaited_once()

    async def test_exchange_closed_even_on_error(self) -> None:
        mock_ex = MagicMock()
        mock_ex.load_markets = AsyncMock(side_effect=ccxt.NetworkError("down"))
        mock_ex.close = AsyncMock()

        with (
            patch("app.collectors.derivatives._build_exchange", return_value=mock_ex),
            patch("app.collectors.derivatives.settings") as mock_s,
        ):
            mock_s.watched_symbols = ["BTC/USDT"]
            await collect_derivatives()          # must not raise

        mock_ex.close.assert_awaited_once()

    async def test_all_symbols_fail_does_not_save(self) -> None:
        mock_ex = MagicMock()
        mock_ex.load_markets = AsyncMock()
        mock_ex.close = AsyncMock()
        mock_ex.fetch_funding_rate = AsyncMock(side_effect=ccxt.NotSupported("no"))
        mock_ex.fetch_open_interest = AsyncMock(side_effect=ccxt.NotSupported("no"))
        mock_ex.fetch_long_short_ratio = AsyncMock(side_effect=ccxt.NotSupported("no"))

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        with (
            patch("app.collectors.derivatives._build_exchange", return_value=mock_ex),
            patch("app.collectors.derivatives.AsyncSessionLocal", return_value=mock_session),
            patch("app.collectors.derivatives.settings") as mock_s,
        ):
            mock_s.watched_symbols = ["BTC/USDT"]
            await collect_derivatives()

        mock_session.add_all.assert_not_called()
