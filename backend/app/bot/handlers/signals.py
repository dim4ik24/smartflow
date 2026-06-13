"""Handler for /signals — last N active signals."""

from __future__ import annotations

import structlog
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Signal

log = structlog.get_logger(__name__)

router = Router(name="signals")

_MAX_SIGNALS = 5


def _format_signal(sig: Signal, index: int) -> str:
    side_label = "ЛОНГ 📈" if sig.side == "long" else "ШОРТ 📉"
    return (
        f"{index}. <b>{sig.symbol}</b> — {side_label} | Score <b>{sig.score}/100</b>\n"
        f"   Таймфрейм: {sig.timeframe}\n"
        f"   Вхід: {sig.entry_low:.6g} – {sig.entry_high:.6g}\n"
        f"   SL: {sig.sl:.6g} | TP1: {sig.tp1:.6g} | TP2: {sig.tp2:.6g}\n"
        f"   R:R {sig.rr:.2f}"
    )


@router.message(Command("signals"))
async def cmd_signals(message: Message, session: AsyncSession) -> None:
    result = await session.execute(
        select(Signal)
        .where(Signal.status == "active")
        .order_by(Signal.created_at.desc())
        .limit(_MAX_SIGNALS)
    )
    signals = result.scalars().all()

    if not signals:
        await message.answer("Наразі активних сигналів немає.")
        return

    header = f"📊 <b>Активні сигнали</b> (останні {len(signals)}):\n"
    body = "\n\n".join(_format_signal(sig, i) for i, sig in enumerate(signals, 1))
    await message.answer(f"{header}\n{body}")
