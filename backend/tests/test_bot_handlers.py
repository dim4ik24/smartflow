"""Tests for Telegram bot handlers and middlewares.

All aiogram objects (Message, CallbackQuery, User) are mocked with AsyncMock /
MagicMock so that no real Telegram connection is required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.handlers.help import cmd_help
from app.bot.handlers.signals import _format_signal, cmd_signals
from app.bot.handlers.start import accept_disclaimer, cmd_start
from app.bot.handlers.stats import cmd_stats
from app.bot.middlewares import DisclaimerMiddleware
from app.db.models import Signal, User

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _tg_user(tg_id: int = 123456, username: str = "testuser") -> MagicMock:
    u = MagicMock()
    u.id = tg_id
    u.username = username
    return u


def _mock_message(text: str = "/start", tg_id: int = 123456) -> AsyncMock:
    msg = AsyncMock()
    msg.text = text
    msg.from_user = _tg_user(tg_id)
    return msg


def _mock_callback(data: str = "accept_disclaimer", tg_id: int = 123456) -> AsyncMock:
    cb = AsyncMock()
    cb.data = data
    cb.from_user = _tg_user(tg_id)
    cb.message = AsyncMock()
    return cb


def _make_signal(
    symbol: str = "BTC/USDT",
    side: str = "long",
    score: int = 82,
    timeframe: str = "1h",
) -> Signal:
    return Signal(
        symbol=symbol,
        side=side,
        timeframe=timeframe,
        score=score,
        entry_low=30000.0,
        entry_high=30500.0,
        sl=29000.0,
        tp1=32000.0,
        tp2=34000.0,
        rr=2.5,
        factors={},
        zones=[],
        status="active",
    )


# ── _format_signal (pure function) ────────────────────────────────────────────

def test_format_signal_long() -> None:
    sig = _make_signal(symbol="ETH/USDT", side="long", score=90)
    text = _format_signal(sig, 1)
    assert "ETH/USDT" in text
    assert "ЛОНГ" in text
    assert "90/100" in text
    assert "R:R 2.50" in text


def test_format_signal_short() -> None:
    sig = _make_signal(symbol="SOL/USDT", side="short", score=75)
    text = _format_signal(sig, 2)
    assert "ШОРТ" in text
    assert "SOL/USDT" in text
    assert "2." in text  # index prefix


def test_format_signal_contains_levels() -> None:
    sig = _make_signal()
    text = _format_signal(sig, 1)
    assert "29000" in text  # SL
    assert "32000" in text  # TP1
    assert "34000" in text  # TP2


# ── /start handler ────────────────────────────────────────────────────────────

async def test_cmd_start_sends_disclaimer(db_session: AsyncSession) -> None:
    msg = _mock_message("/start")
    await cmd_start(msg, db_session)
    msg.answer.assert_awaited_once()
    call_text: str = msg.answer.call_args[0][0]
    assert "Дисклеймер" in call_text or "дисклеймер" in call_text.lower()
    # Keyboard must be present
    kwargs = msg.answer.call_args[1]
    assert kwargs.get("reply_markup") is not None


async def test_cmd_start_no_from_user(db_session: AsyncSession) -> None:
    msg = _mock_message("/start")
    msg.from_user = None
    await cmd_start(msg, db_session)
    msg.answer.assert_not_awaited()


# ── accept_disclaimer callback ────────────────────────────────────────────────

async def test_accept_disclaimer_creates_new_user(db_session: AsyncSession) -> None:
    tg_id = 9001
    cb = _mock_callback(tg_id=tg_id)

    await accept_disclaimer(cb, db_session)

    cb.answer.assert_awaited()
    cb.message.edit_text.assert_awaited_once()

    from sqlalchemy import select
    result = await db_session.execute(select(User).where(User.tg_id == tg_id))
    user = result.scalar_one_or_none()
    assert user is not None
    assert user.disclaimer_accepted_at is not None


async def test_accept_disclaimer_sets_timestamp_for_existing_user(
    db_session: AsyncSession,
) -> None:
    tg_id = 9002
    existing = User(tg_id=tg_id, username="existing")
    db_session.add(existing)
    await db_session.flush()

    cb = _mock_callback(tg_id=tg_id)
    await accept_disclaimer(cb, db_session)

    from sqlalchemy import select
    result = await db_session.execute(select(User).where(User.tg_id == tg_id))
    user = result.scalar_one_or_none()
    assert user is not None
    assert user.disclaimer_accepted_at is not None


async def test_accept_disclaimer_idempotent(db_session: AsyncSession) -> None:
    tg_id = 9003
    original_ts = datetime(2025, 1, 1, tzinfo=UTC)
    existing = User(tg_id=tg_id, username="existing2", disclaimer_accepted_at=original_ts)
    db_session.add(existing)
    await db_session.flush()

    cb = _mock_callback(tg_id=tg_id)
    await accept_disclaimer(cb, db_session)

    from sqlalchemy import select
    result = await db_session.execute(select(User).where(User.tg_id == tg_id))
    user = result.scalar_one_or_none()
    assert user is not None
    # timestamp was already set — should not be overwritten
    assert user.disclaimer_accepted_at == original_ts


async def test_accept_disclaimer_no_from_user(db_session: AsyncSession) -> None:
    cb = _mock_callback()
    cb.from_user = None
    await accept_disclaimer(cb, db_session)
    cb.answer.assert_awaited_once()
    cb.message.edit_text.assert_not_awaited()


async def test_accept_disclaimer_message_none(db_session: AsyncSession) -> None:
    tg_id = 9004
    cb = _mock_callback(tg_id=tg_id)
    cb.message = None
    await accept_disclaimer(cb, db_session)
    cb.answer.assert_awaited()
    # No AttributeError should be raised when message is None


# ── /signals handler ──────────────────────────────────────────────────────────

async def test_cmd_signals_no_active(db_session: AsyncSession) -> None:
    msg = _mock_message("/signals")
    await cmd_signals(msg, db_session)
    msg.answer.assert_awaited_once()
    assert "немає" in msg.answer.call_args[0][0]


async def test_cmd_signals_shows_active_signals(db_session: AsyncSession) -> None:
    sig = _make_signal()
    db_session.add(sig)
    await db_session.flush()

    msg = _mock_message("/signals")
    await cmd_signals(msg, db_session)
    msg.answer.assert_awaited_once()
    response: str = msg.answer.call_args[0][0]
    assert "BTC/USDT" in response
    assert "82" in response


async def test_cmd_signals_respects_max_limit(db_session: AsyncSession) -> None:
    for i in range(7):
        db_session.add(_make_signal(symbol=f"COIN{i}/USDT", score=70 + i))
    await db_session.flush()

    msg = _mock_message("/signals")
    await cmd_signals(msg, db_session)
    response: str = msg.answer.call_args[0][0]
    # At most 5 signals shown; "COIN6/USDT" is 7th → may be missing
    # Count number of signal entries by looking for pattern "N. "
    # The response should show at most 5 entries
    count = sum(1 for i in range(1, 8) if f"{i}. " in response)
    assert count <= 5


async def test_cmd_signals_excludes_inactive(db_session: AsyncSession) -> None:
    sig = _make_signal()
    sig.status = "expired"
    db_session.add(sig)
    await db_session.flush()

    msg = _mock_message("/signals")
    await cmd_signals(msg, db_session)
    assert "немає" in msg.answer.call_args[0][0]


# ── /stats handler ────────────────────────────────────────────────────────────

async def test_cmd_stats_stub(db_session: AsyncSession) -> None:
    msg = _mock_message("/stats")
    await cmd_stats(msg)
    msg.answer.assert_awaited_once()
    assert "Етап" in msg.answer.call_args[0][0]


# ── /help handler ─────────────────────────────────────────────────────────────

async def test_cmd_help_lists_commands() -> None:
    msg = _mock_message("/help")
    await cmd_help(msg)
    msg.answer.assert_awaited_once()
    response: str = msg.answer.call_args[0][0]
    for cmd in ("/start", "/signals", "/stats", "/help"):
        assert cmd in response


# ── DisclaimerMiddleware ──────────────────────────────────────────────────────

async def test_disclaimer_middleware_allows_start(db_session: AsyncSession) -> None:
    middleware = DisclaimerMiddleware()
    handler = AsyncMock(return_value=None)
    msg = _mock_message("/start", tg_id=88001)

    await middleware(handler, msg, {"session": db_session})

    handler.assert_awaited_once()
    msg.answer.assert_not_awaited()


async def test_disclaimer_middleware_blocks_without_disclaimer(
    db_session: AsyncSession,
) -> None:
    middleware = DisclaimerMiddleware()
    handler = AsyncMock(return_value=None)
    msg = _mock_message("/signals", tg_id=88002)

    await middleware(handler, msg, {"session": db_session})

    handler.assert_not_awaited()
    msg.answer.assert_awaited_once()
    blocked_text: str = msg.answer.call_args[0][0]
    assert "/start" in blocked_text


async def test_disclaimer_middleware_allows_with_disclaimer(
    db_session: AsyncSession,
) -> None:
    tg_id = 88003
    user = User(tg_id=tg_id, disclaimer_accepted_at=datetime.now(UTC))
    db_session.add(user)
    await db_session.flush()

    middleware = DisclaimerMiddleware()
    handler = AsyncMock(return_value=None)
    msg = _mock_message("/signals", tg_id=tg_id)

    await middleware(handler, msg, {"session": db_session})

    handler.assert_awaited_once()


async def test_disclaimer_middleware_no_from_user(db_session: AsyncSession) -> None:
    middleware = DisclaimerMiddleware()
    handler = AsyncMock(return_value=None)
    msg = _mock_message("/signals")
    msg.from_user = None

    result = await middleware(handler, msg, {"session": db_session})

    handler.assert_not_awaited()
    assert result is None


async def test_disclaimer_middleware_blocks_user_without_accepted(
    db_session: AsyncSession,
) -> None:
    tg_id = 88004
    user = User(tg_id=tg_id)  # disclaimer_accepted_at is None
    db_session.add(user)
    await db_session.flush()

    middleware = DisclaimerMiddleware()
    handler = AsyncMock(return_value=None)
    msg = _mock_message("/help", tg_id=tg_id)

    await middleware(handler, msg, {"session": db_session})

    handler.assert_not_awaited()
    msg.answer.assert_awaited_once()
