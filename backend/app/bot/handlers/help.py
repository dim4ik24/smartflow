"""Handler for /help — lists available commands."""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

router = Router(name="help")

_HELP_TEXT = (
    "📖 <b>SmartFlow — довідка</b>\n\n"
    "<b>Доступні команди:</b>\n"
    "/start — почати / показати дисклеймер\n"
    "/signals — останні активні торгові сетапи (до 5)\n"
    "/stats — статистика торгових сигналів\n"
    "/help — ця довідка\n\n"
    "SmartFlow відстежує ринок 24/7 і сповіщає про сетапи за Smart Money "
    "Concepts. Кожен сигнал містить вхід, стоп-лос, тейк-профіт та R:R.\n\n"
    "⚠️ SmartFlow — аналітичний інструмент, не фінансова порада."
)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(_HELP_TEXT)
