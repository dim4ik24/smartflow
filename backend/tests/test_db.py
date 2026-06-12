"""Tests for async DB engine, session factory, and ORM models."""

from __future__ import annotations

import contextlib
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_engine_executes_query() -> None:
    from app.db.session import engine

    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_get_db_yields_async_session() -> None:
    from app.db.session import get_db

    gen = get_db()
    session = await gen.__anext__()
    assert isinstance(session, AsyncSession)
    with contextlib.suppress(StopAsyncIteration):
        await gen.aclose()


@pytest.mark.asyncio
async def test_all_tables_created(create_tables: None) -> None:
    from app.db.session import engine

    async with engine.connect() as conn:
        table_names: list[str] = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )

    expected = {
        "users", "api_keys", "candles", "signals",
        "positions", "audit_log", "payments", "news_items",
    }
    assert expected.issubset(set(table_names))


@pytest.mark.asyncio
async def test_user_crud(db_session: AsyncSession) -> None:
    from app.db.models import User

    user = User(tg_id=123456789, username="tester")
    db_session.add(user)
    await db_session.flush()

    assert user.id is not None
    assert user.plan == "free"
    assert user.risk_pct == 1.0
    assert user.max_positions == 3
    assert isinstance(user.created_at, datetime)


@pytest.mark.asyncio
async def test_position_client_order_id_is_uuid(db_session: AsyncSession) -> None:
    from app.db.models import Position, Signal, User

    user = User(tg_id=999888777)
    db_session.add(user)
    await db_session.flush()

    signal = Signal(
        symbol="BTC/USDT",
        side="long",
        timeframe="1h",
        score=85,
        entry_low=60000.0,
        entry_high=60500.0,
        sl=58000.0,
        tp1=63000.0,
        tp2=66000.0,
        rr=2.5,
    )
    db_session.add(signal)
    await db_session.flush()

    position = Position(
        user_id=user.id,
        signal_id=signal.id,
        exchange="bybit",
        qty=0.01,
        entry_price=60200.0,
        sl=58000.0,
        tp1=63000.0,
        tp2=66000.0,
    )
    db_session.add(position)
    await db_session.flush()

    assert isinstance(position.client_order_id, uuid.UUID)


@pytest.mark.asyncio
async def test_candle_composite_primary_key(db_session: AsyncSession) -> None:
    from app.db.models import Candle

    ts = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    candle = Candle(
        symbol="ETH/USDT",
        timeframe="1h",
        ts=ts,
        o=2200.0,
        h=2250.0,
        l=2180.0,
        c=2230.0,
        v=1500.0,
    )
    db_session.add(candle)
    await db_session.flush()

    fetched = await db_session.get(Candle, ("ETH/USDT", "1h", ts))
    assert fetched is not None
    assert fetched.c == 2230.0
