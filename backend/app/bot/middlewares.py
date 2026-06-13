"""aiogram 3 middlewares for the SmartFlow Telegram bot.

DbSessionMiddleware — injects an AsyncSession into handler data["session"]
  for all update types (messages, callbacks, etc.).

DisclaimerMiddleware — blocks message commands until User.disclaimer_accepted_at
  is set. /start is always allowed through.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject
from sqlalchemy import select

from app.db.models import User
from app.db.session import AsyncSessionLocal

log = structlog.get_logger(__name__)

_BLOCKED_REPLY = "Спочатку прийміть дисклеймер — введіть /start"


class DbSessionMiddleware(BaseMiddleware):
    """Provides an AsyncSession for every update; commits on success, rolls back on error."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        async with AsyncSessionLocal() as session:
            data["session"] = session
            try:
                result = await handler(event, data)
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise


class DisclaimerMiddleware(BaseMiddleware):
    """Blocks incoming messages until User.disclaimer_accepted_at is set.

    /start always passes through so users can accept at any time.
    Registered only on dp.message — callback queries are not affected.
    """

    async def __call__(  # type: ignore[override]
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        # /start is always allowed
        if event.text and event.text.startswith("/start"):
            return await handler(event, data)

        from_user = event.from_user
        if from_user is None:
            return None

        session = data["session"]
        result = await session.execute(select(User).where(User.tg_id == from_user.id))
        user = result.scalar_one_or_none()

        if user is None or user.disclaimer_accepted_at is None:
            await event.answer(_BLOCKED_REPLY)
            return None

        return await handler(event, data)
