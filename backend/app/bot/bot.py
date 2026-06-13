"""Bot and Dispatcher factory functions.

Usage:
    bot = create_bot()
    dp  = create_dispatcher()
    await dp.start_polling(bot)
"""

from __future__ import annotations

import sys

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode

from app.bot.handlers import help as help_cmd
from app.bot.handlers import signals, start, stats
from app.bot.middlewares import DbSessionMiddleware, DisclaimerMiddleware
from app.config import settings

log = structlog.get_logger(__name__)


def _apply_threaded_resolver(session: AiohttpSession) -> None:
    """Replace aiodns with the stdlib ThreadedResolver on Windows only.

    On Linux/prod, aiodns works correctly and should stay as the default
    (it is faster and avoids blocking the event loop).  On Windows, pycares
    cannot open UDP DNS sockets in some sandbox/restricted environments
    (error 11 — "Could not contact DNS servers"), while the OS resolver
    reached via a thread pool works fine.

    AiohttpSession has no public API for swapping the resolver, so we write
    into the private _connector_init dict that create_session() passes to
    TCPConnector(**...).  If aiogram ever removes that attribute the bot
    still starts — DNS will fall back to whatever aiohttp picks by default.
    """
    if sys.platform != "win32":
        return
    try:
        from aiohttp.resolver import ThreadedResolver

        session._connector_init["resolver"] = ThreadedResolver()  # type: ignore[attr-defined]
        log.debug("bot_threaded_resolver_applied")
    except Exception:
        log.warning(
            "bot_threaded_resolver_failed",
            msg="Could not inject ThreadedResolver; DNS may fail on Windows",
        )


def create_bot() -> Bot:
    """Create an aiogram Bot with HTML parse mode enabled globally."""
    bot_session = AiohttpSession()
    _apply_threaded_resolver(bot_session)
    return Bot(
        token=settings.telegram_bot_token,
        session=bot_session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


def create_dispatcher() -> Dispatcher:
    """Create a Dispatcher with all middlewares and routers registered.

    Middleware order matters:
      - DbSessionMiddleware on dp.update: runs first for ALL updates, injects session.
      - DisclaimerMiddleware on dp.message: runs second for messages only,
        checks disclaimer acceptance; callback queries bypass it so that the
        "accept_disclaimer" callback always works.
    """
    dp = Dispatcher()

    dp.update.middleware(DbSessionMiddleware())
    dp.message.middleware(DisclaimerMiddleware())

    dp.include_router(start.router)
    dp.include_router(signals.router)
    dp.include_router(stats.router)
    dp.include_router(help_cmd.router)

    return dp
