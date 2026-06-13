"""Handler for /start and the disclaimer acceptance callback."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User

log = structlog.get_logger(__name__)

router = Router(name="start")

_DISCLAIMER = (
    "<b>SmartFlow — аналітичний інструмент, не фінансова порада.</b>\n\n"
    "Сигнали відображають технічні сетапи та показники на основі бектестових "
    "даних. Торгівля криптовалютою пов'язана з ризиком втрати коштів. "
    "Ви берете на себе повну відповідальність за свої торгові рішення. "
    "Ні SmartFlow, ні його розробники не несуть відповідальності за фінансові втрати."
)

_DISCLAIMER_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Приймаю ✅", callback_data="accept_disclaimer")]
    ]
)


@router.message(CommandStart())
async def cmd_start(message: Message, session: AsyncSession) -> None:  # noqa: ARG001
    if message.from_user is None:
        return
    await message.answer(
        f"👋 Вітаємо у <b>SmartFlow</b>!\n\n"
        f"⚠️ <b>Дисклеймер</b>\n\n{_DISCLAIMER}\n\n"
        "Натисніть «Приймаю», щоб продовжити.",
        reply_markup=_DISCLAIMER_KEYBOARD,
    )


@router.callback_query(F.data == "accept_disclaimer")
async def accept_disclaimer(callback: CallbackQuery, session: AsyncSession) -> None:
    tg_user = callback.from_user
    if tg_user is None:
        await callback.answer()
        return

    result = await session.execute(select(User).where(User.tg_id == tg_user.id))
    user = result.scalar_one_or_none()

    now = datetime.now(UTC)
    if user is None:
        user = User(
            tg_id=tg_user.id,
            username=tg_user.username,
            disclaimer_accepted_at=now,
        )
        session.add(user)
    elif user.disclaimer_accepted_at is None:
        user.disclaimer_accepted_at = now

    await session.flush()
    await callback.answer("Дисклеймер прийнято!")

    confirmation = (
        "✅ <b>Дякуємо!</b> Ви прийняли дисклеймер.\n\n"
        "Тепер ви можете користуватись усіма функціями SmartFlow.\n\n"
        "<b>Доступні команди:</b>\n"
        "/signals — активні торгові сетапи\n"
        "/stats — статистика\n"
        "/help — довідка"
    )
    if callback.message is not None:
        await callback.message.edit_text(confirmation)

    log.info("disclaimer_accepted", tg_id=tg_user.id, username=tg_user.username)
