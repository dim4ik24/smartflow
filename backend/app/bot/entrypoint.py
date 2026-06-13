"""Standalone entry-point for the Telegram bot (long polling).

Long polling is used (vs webhook) because it requires no public URL, no TLS
termination setup, and no webhook secret validation on every request — all of
which simplify local development and the initial production rollout.  Webhook
can be introduced in a later etap once a stable public domain is set up.
See DECISIONS.md for the full rationale.

Run locally:
    cd backend && python -m app.bot.entrypoint

systemd service:
    see deploy/bot.service
"""

from __future__ import annotations

import asyncio
import logging
import sys

import structlog

from app.config import settings

# ── Logging setup (mirrors app/main.py) ───────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        (
            structlog.dev.ConsoleRenderer()
            if settings.debug
            else structlog.processors.JSONRenderer()
        ),
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

log = structlog.get_logger(__name__)


async def main() -> None:
    import app.db.models  # noqa: F401 — registers models so create_all sees them
    from app.bot.bot import create_bot, create_dispatcher
    from app.db.session import Base, engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    bot = create_bot()
    dp = create_dispatcher()

    log.info("bot_starting", environment=settings.environment)
    try:
        await dp.start_polling(bot, allowed_updates=["message", "callback_query"])
    finally:
        await bot.session.close()
        await engine.dispose()
        log.info("bot_stopped")


if __name__ == "__main__":
    asyncio.run(main())
