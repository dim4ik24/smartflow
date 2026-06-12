"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Relative imports avoid shadowing the `app` package name with the FastAPI instance.
from .config import settings
from .db.session import Base, engine

# ── Logging setup ─────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.BoundLogger,
    cache_logger_on_first_use=True,
)
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, settings.log_level),
    format="%(message)s",
)

log: structlog.BoundLogger = structlog.get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    log.info("startup", environment=settings.environment, debug=settings.debug)
    # Register all models before create_all so their tables are included.
    from . import db  # noqa: F401
    from .db import models  # noqa: F401

    async with engine.begin() as conn:
        # Alembic handles migrations in production; this covers dev/test.
        await conn.run_sync(Base.metadata.create_all)

    yield

    await engine.dispose()
    log.info("shutdown")


# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit_default],
)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartFlow API",
    description="Crypto analytics: SMC signals, scoring, and auto-trading.",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.get("/health", tags=["infra"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}
