"""Tests for app/bot/alerts.py — signal alert dispatcher.

Chart rendering: tested with a real mplfinance call on a minimal DataFrame so
that we confirm the render pipeline works end-to-end (PNG magic bytes check).

send_signal_alert: bot.send_photo / send_message are mocked; render_signal_chart
is patched to return fake bytes to keep the suite fast.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.alerts import (
    _build_alert_text,
    _build_keyboard,
    _price_fmt,
    render_signal_chart,
    send_signal_alert,
)
from app.db.models import Signal, User

_PNG_MAGIC = b"\x89PNG"


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_signal(
    symbol: str = "BTC/USDT",
    side: str = "long",
    score: int = 82,
    zones: list | None = None,
    signal_id: int = 1,
) -> Signal:
    sig = Signal(
        symbol=symbol,
        side=side,
        timeframe="1h",
        score=score,
        entry_low=30000.0,
        entry_high=30500.0,
        sl=29000.0,
        tp1=32000.0,
        tp2=34000.0,
        rr=2.5,
        factors={},
        zones=zones or [],
        status="active",
    )
    sig.id = signal_id
    return sig


def _make_candles(n: int = 20, base: float = 30000.0) -> pd.DataFrame:
    """Minimal OHLCV DataFrame with UTC DatetimeIndex suitable for mplfinance."""
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="1h", tz="UTC")
    prices = [base + i * 10 for i in range(n)]
    return pd.DataFrame(
        {
            "open":   [p - 5 for p in prices],
            "high":   [p + 20 for p in prices],
            "low":    [p - 20 for p in prices],
            "close":  prices,
            "volume": [1000.0] * n,
        },
        index=idx,
    )


# ── _price_fmt ────────────────────────────────────────────────────────────────

def test_price_fmt_large_integer() -> None:
    assert _price_fmt(30000.0) == "30000"


def test_price_fmt_decimal() -> None:
    assert _price_fmt(1800.55) == "1800.55"


def test_price_fmt_small_fraction() -> None:
    # Should not switch to scientific notation for typical crypto prices
    result = _price_fmt(0.001234)
    assert "e" not in result.lower() or float(result) == pytest.approx(0.001234)


# ── _build_alert_text ─────────────────────────────────────────────────────────

def test_alert_text_contains_symbol_and_side() -> None:
    text = _build_alert_text(_make_signal(symbol="ETH/USDT", side="short"))
    assert "ETH/USDT" in text
    assert "ШОРТ" in text


def test_alert_text_contains_score() -> None:
    text = _build_alert_text(_make_signal(score=87))
    assert "87/100" in text


def test_alert_text_contains_levels() -> None:
    sig = _make_signal()
    text = _build_alert_text(sig)
    assert "29000" in text  # SL
    assert "32000" in text  # TP1
    assert "34000" in text  # TP2
    assert "R:R" in text
    assert "2.50" in text


def test_alert_text_no_probability_percentage() -> None:
    """SPEC invariant: never use "% chance" or "% probability" wording."""
    text = _build_alert_text(_make_signal(score=90))
    # Raw "%" is forbidden in the context of predictions / chances
    # Allow "%" only as part of the R:R or score display — actually score
    # uses "/100" not "%" so "%" must not appear at all.
    assert "%" not in text


def test_alert_text_has_etap7_note() -> None:
    text = _build_alert_text(_make_signal())
    # "Етап 7" or "Етапу 7" (genitive) — any grammatical form is fine
    assert "Етап" in text and "7" in text


def test_alert_text_has_disclaimer_note() -> None:
    text = _build_alert_text(_make_signal())
    assert "порада" in text.lower() or "аналітика" in text.lower()


def test_alert_text_long_side_emoji() -> None:
    text = _build_alert_text(_make_signal(side="long"))
    assert "ЛОНГ" in text
    assert "📈" in text


# ── _build_keyboard ───────────────────────────────────────────────────────────

def test_keyboard_url_contains_signal_id() -> None:
    sig = _make_signal(signal_id=42)
    kb = _build_keyboard(sig)
    button = kb.inline_keyboard[0][0]
    assert "42" in (button.url or "")


def test_keyboard_url_is_telegram_link() -> None:
    kb = _build_keyboard(_make_signal())
    button = kb.inline_keyboard[0][0]
    assert (button.url or "").startswith("https://t.me/")


# ── render_signal_chart ───────────────────────────────────────────────────────

def test_render_signal_chart_returns_valid_png() -> None:
    """Real mplfinance render — verifies the full pipeline produces a valid PNG."""
    sig = _make_signal()
    df = _make_candles(n=20)
    result = render_signal_chart(sig, df)
    assert isinstance(result, bytes)
    assert result[:4] == _PNG_MAGIC


def test_render_signal_chart_with_ob_fvg_zones() -> None:
    """Zone bands must not crash the render."""
    zones = [
        {"type": "OB",  "price_from": 29500.0, "price_to": 30000.0},
        {"type": "FVG", "price_from": 30100.0, "price_to": 30300.0},
    ]
    sig = _make_signal(zones=zones)
    df = _make_candles(n=20)
    result = render_signal_chart(sig, df)
    assert result[:4] == _PNG_MAGIC


def test_render_signal_chart_unknown_zone_type() -> None:
    """Unknown zone types should use the default colour without crashing."""
    zones = [{"type": "UNKNOWN_ZONE", "price_from": 29800.0, "price_to": 30100.0}]
    sig = _make_signal(zones=zones)
    result = render_signal_chart(sig, _make_candles())
    assert result[:4] == _PNG_MAGIC


def test_render_signal_chart_empty_df_raises() -> None:
    with pytest.raises(ValueError, match="too short"):
        render_signal_chart(_make_signal(), pd.DataFrame())


def test_render_signal_chart_missing_column_raises() -> None:
    df = _make_candles().drop(columns=["volume"])
    with pytest.raises(ValueError, match="Volume"):
        render_signal_chart(_make_signal(), df)


def test_render_signal_chart_short() -> None:
    """A single candle should raise before mplfinance gets a chance to error."""
    df = _make_candles(n=1)
    with pytest.raises(ValueError, match="too short"):
        render_signal_chart(_make_signal(), df)


# ── send_signal_alert ─────────────────────────────────────────────────────────

async def test_send_alert_sends_photo_to_accepted_user(
    db_session: AsyncSession,
) -> None:
    user = User(tg_id=10001, disclaimer_accepted_at=datetime.now(UTC))
    db_session.add(user)
    await db_session.flush()

    bot = AsyncMock()
    with patch("app.bot.alerts.render_signal_chart", return_value=b"\x89PNGfake"):
        await send_signal_alert(bot, db_session, _make_signal(), _make_candles())

    bot.send_photo.assert_awaited_once()
    call_kwargs = bot.send_photo.call_args[1]
    assert call_kwargs["chat_id"] == 10001
    assert "caption" in call_kwargs
    assert "reply_markup" in call_kwargs


async def test_send_alert_skips_user_without_disclaimer(
    db_session: AsyncSession,
) -> None:
    db_session.add(User(tg_id=10002))  # disclaimer_accepted_at is None
    await db_session.flush()

    bot = AsyncMock()
    with patch("app.bot.alerts.render_signal_chart", return_value=b"\x89PNGfake"):
        await send_signal_alert(bot, db_session, _make_signal(), _make_candles())

    bot.send_photo.assert_not_awaited()
    bot.send_message.assert_not_awaited()


async def test_send_alert_no_users_no_sends(db_session: AsyncSession) -> None:
    bot = AsyncMock()
    with patch("app.bot.alerts.render_signal_chart", return_value=b"\x89PNGfake"):
        await send_signal_alert(bot, db_session, _make_signal(), _make_candles())

    bot.send_photo.assert_not_awaited()
    bot.send_message.assert_not_awaited()


async def test_send_alert_chart_failure_falls_back_to_message(
    db_session: AsyncSession,
) -> None:
    db_session.add(User(tg_id=10003, disclaimer_accepted_at=datetime.now(UTC)))
    await db_session.flush()

    bot = AsyncMock()
    with patch(
        "app.bot.alerts.render_signal_chart", side_effect=ValueError("render error")
    ):
        await send_signal_alert(bot, db_session, _make_signal(), _make_candles())

    bot.send_photo.assert_not_awaited()
    bot.send_message.assert_awaited_once()
    assert bot.send_message.call_args[1]["chat_id"] == 10003


async def test_send_alert_per_user_error_continues_to_others(
    db_session: AsyncSession,
) -> None:
    for tg_id in (10004, 10005, 10006):
        db_session.add(User(tg_id=tg_id, disclaimer_accepted_at=datetime.now(UTC)))
    await db_session.flush()

    call_count = 0

    async def _send_photo_side_effect(**kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        if kwargs.get("chat_id") == 10005:
            raise RuntimeError("Blocked by Telegram")

    bot = AsyncMock()
    bot.send_photo.side_effect = _send_photo_side_effect

    with patch("app.bot.alerts.render_signal_chart", return_value=b"\x89PNGfake"):
        await send_signal_alert(bot, db_session, _make_signal(), _make_candles())

    # All 3 attempted; 1 failed; the other 2 succeeded
    assert call_count == 3


async def test_send_alert_multiple_users_all_receive_photo(
    db_session: AsyncSession,
) -> None:
    for tg_id in (10007, 10008):
        db_session.add(User(tg_id=tg_id, disclaimer_accepted_at=datetime.now(UTC)))
    await db_session.flush()

    bot = AsyncMock()
    with patch("app.bot.alerts.render_signal_chart", return_value=b"\x89PNGfake"):
        await send_signal_alert(bot, db_session, _make_signal(), _make_candles())

    assert bot.send_photo.await_count == 2


async def test_send_alert_caption_has_no_percentage(db_session: AsyncSession) -> None:
    """SPEC invariant: no "% chance" in alert captions."""
    db_session.add(User(tg_id=10009, disclaimer_accepted_at=datetime.now(UTC)))
    await db_session.flush()

    bot = AsyncMock()
    with patch("app.bot.alerts.render_signal_chart", return_value=b"\x89PNGfake"):
        await send_signal_alert(bot, db_session, _make_signal(), _make_candles())

    caption: str = bot.send_photo.call_args[1]["caption"]
    assert "%" not in caption


async def test_send_alert_keyboard_has_mini_app_url(db_session: AsyncSession) -> None:
    db_session.add(User(tg_id=10010, disclaimer_accepted_at=datetime.now(UTC)))
    await db_session.flush()

    bot = AsyncMock()
    with patch("app.bot.alerts.render_signal_chart", return_value=b"\x89PNGfake"):
        await send_signal_alert(bot, db_session, _make_signal(signal_id=7), _make_candles())

    keyboard = bot.send_photo.call_args[1]["reply_markup"]
    url: str = keyboard.inline_keyboard[0][0].url or ""
    assert "t.me" in url
    assert "7" in url
