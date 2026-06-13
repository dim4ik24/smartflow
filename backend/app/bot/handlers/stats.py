"""Handler for /stats — stub until Etap 7 paper-trading statistics."""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

router = Router(name="stats")


@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    await message.answer(
        "📈 <b>Статистика</b>\n\n"
        "Статистика paper trading буде доступна у Етапі 7.\n"
        "Слідкуйте за оновленнями!"
    )
