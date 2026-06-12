"""Async SQLAlchemy engine, session factory, and declarative Base.

DATABASE_URL examples
---------------------
SQLite  (dev/test):  sqlite+aiosqlite:///./smartflow.db
                     sqlite+aiosqlite:///:memory:          ← uses StaticPool automatically
PostgreSQL (prod):   postgresql+asyncpg://user:pass@host:5432/smartflow
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from app.config import settings

# Build engine kwargs; keep SQLite and PostgreSQL differences explicit.
_engine_kwargs: dict[str, Any] = {
    "echo": settings.debug,
    "pool_pre_ping": True,
}

if settings.database_url == "sqlite+aiosqlite:///:memory:":
    # StaticPool reuses a single connection so all calls share the same
    # in-memory database — required for :memory: in tests.
    _engine_kwargs["poolclass"] = StaticPool
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
elif settings.database_url.startswith("postgresql"):
    _engine_kwargs["pool_size"] = 10
    _engine_kwargs["max_overflow"] = 20

engine = create_async_engine(settings.database_url, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base — all ORM models inherit from this."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields a session, commits on success, rolls back on error."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
