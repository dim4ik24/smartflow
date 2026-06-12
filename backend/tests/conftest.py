"""Test configuration.

Sets required environment variables BEFORE any app module is imported so that
pydantic-settings reads test values (not a real .env file) on first load.
"""

from __future__ import annotations

import os
import secrets

# ── Required env vars for Settings (must be set before importing app) ─────────
os.environ.setdefault("JWT_SECRET_KEY", "test-only-jwt-secret-key-min-32-chars-xyz")
os.environ.setdefault("MASTER_ENCRYPTION_KEY", secrets.token_hex(32))
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0:test-token")
os.environ.setdefault("TELEGRAM_WEBHOOK_SECRET", "test-webhook-secret")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")

# ── App imports (after env vars are set) ──────────────────────────────────────
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal, Base, engine


@pytest.fixture(scope="session", autouse=True)
async def create_tables() -> None:
    """Create all tables once per test session in the in-memory database."""
    import app.db.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@pytest.fixture
async def db_session() -> AsyncSession:
    """Yield a session that rolls back after each test (no side effects)."""
    async with AsyncSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def client() -> AsyncClient:
    """Async HTTPX test client wired to the FastAPI app."""
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
