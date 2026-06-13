"""Bot and Dispatcher factory functions.

Usage:
    bot = create_bot()
    dp  = create_dispatcher()
    await dp.start_polling(bot)
"""

from __future__ import annotations

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from app.bot.handlers import help as help_cmd
from app.bot.handlers import signals, start, stats
from app.bot.middlewares import DbSessionMiddleware, DisclaimerMiddleware
from app.config import settings

log = structlog.get_logger(__name__)


def create_bot() -> Bot:
    """Create an aiogram Bot with HTML parse mode enabled globally."""
    return Bot(
        token=settings.telegram_bot_token,
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
