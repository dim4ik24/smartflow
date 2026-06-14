"""Signal alert dispatcher (SPEC §bot/alerts.py).

Called by the analysis engine immediately after a new Signal is persisted.
Renders a candlestick PNG via mplfinance and broadcasts it to every User
whose disclaimer_accepted_at is set.

Chart layout
------------
- Last _CHART_CANDLES candles of the entry timeframe
- OB / FVG / LIQ_SWEEP zones as horizontal bands (axhspan)
- Dashed lines: entry_low / entry_high (blue)
- Solid lines: SL (red), TP1 / TP2 (green)
- Short text labels at the right axis edge for each level

Alert text
----------
Includes symbol, side, score/100, all levels and R:R.
NEVER uses "% probability" or "% chance" wording — only score and a note
that historical win-rate statistics will be available in Etap 7.
"""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING

# Set non-interactive backend BEFORE any pyplot / mplfinance import so that
# rendering works on headless servers and in CI without a display.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mplfinance as mpf  # noqa: E402
import pandas as pd  # noqa: E402
import structlog  # noqa: E402
from aiogram.types import (  # noqa: E402
    BufferedInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.db.models import Signal, User  # noqa: E402

if TYPE_CHECKING:
    from aiogram import Bot

log = structlog.get_logger(__name__)

# Number of candles shown in the chart (balances detail vs. readability)
_CHART_CANDLES = 60

# Deep-link to the Mini App — placeholder until the frontend is deployed
_MINI_APP_URL = "https://t.me/Smart_Floww_bot/app"

# Zone type → (fill color, alpha)
_ZONE_COLORS: dict[str, tuple[str, float]] = {
    "OB":        ("#2196F3", 0.18),   # blue  — order block
    "FVG":       ("#FF9800", 0.18),   # amber — fair-value gap
    "LIQ_SWEEP": ("#9C27B0", 0.12),   # purple — liquidity sweep
}
_ZONE_COLOR_DEFAULT: tuple[str, float] = ("#9E9E9E", 0.10)

# (signal attribute, line color, linestyle, linewidth, right-edge label prefix)
_LEVEL_SPECS: list[tuple[str, str, str, float, str]] = [
    ("entry_low",  "#1565C0", "--", 0.8, "Lo"),
    ("entry_high", "#1565C0", "--", 0.8, "Hi"),
    ("sl",         "#C62828", "-",  1.4, "SL"),
    ("tp1",        "#2E7D32", "-",  1.4, "TP1"),
    ("tp2",        "#1B5E20", "-",  1.4, "TP2"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _price_fmt(v: float) -> str:
    """Up to 6 significant figures, no scientific notation for typical crypto prices."""
    return f"{v:.6g}"


def _build_alert_text(signal: Signal) -> str:
    side_emoji = "📈" if signal.side == "long" else "📉"
    side_label = "ЛОНГ" if signal.side == "long" else "ШОРТ"
    return (
        f"{side_emoji} <b>{signal.symbol} — {side_label}</b>\n"
        f"Score: <b>{signal.score}/100</b> | Таймфрейм: {signal.timeframe}\n\n"
        f"<b>Рівні:</b>\n"
        f"  Вхід: {_price_fmt(signal.entry_low)} – {_price_fmt(signal.entry_high)}\n"
        f"  SL:   {_price_fmt(signal.sl)}\n"
        f"  TP1:  {_price_fmt(signal.tp1)}\n"
        f"  TP2:  {_price_fmt(signal.tp2)}\n"
        f"  R:R   {signal.rr:.2f}\n\n"
        f"<i>Score {signal.score}/100 — технічний рейтинг сетапу за конфлюенсом "
        f"факторів SMC. Історична статистика win rate буде доступна після Етапу 7.</i>\n\n"
        f"⚠️ Аналітика, не фінансова порада."
    )


def _build_keyboard(signal: Signal) -> InlineKeyboardMarkup:
    signal_id = signal.id or 0
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(
                text="📊 Відкрити графік",
                url=f"{_MINI_APP_URL}?signal={signal_id}",
            )]
        ]
    )


# ── Chart rendering ───────────────────────────────────────────────────────────

def render_signal_chart(signal: Signal, candles_df: pd.DataFrame) -> bytes:
    """Render a candlestick PNG for *signal*.

    This function is **synchronous** (mplfinance/matplotlib are CPU-bound).
    Call it via ``asyncio.run_in_executor`` to avoid blocking the event loop.

    Returns raw PNG bytes.  Raises ``ValueError`` on invalid input so the
    caller can fall back to a text-only message.
    """
    df = candles_df.tail(_CHART_CANDLES).copy()
    if len(df) < 2:  # noqa: PLR2004
        raise ValueError(f"candles_df too short ({len(df)} rows) to render a chart")

    df.index = pd.DatetimeIndex(df.index)
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise ValueError(f"candles_df missing required column '{col}'")

    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        rc={"font.size": 8, "axes.labelsize": 7},
    )

    fig, axes = mpf.plot(
        df,
        type="candle",
        style=style,
        title=f"{signal.symbol} {signal.timeframe}  |  Score {signal.score}/100",
        volume=True,
        returnfig=True,
        figsize=(10, 6),
    )
    price_ax = axes[0]

    # ── Zone bands (time-bounded rectangles) ─────────────────────────────────
    # mplfinance uses integer x-positions (0 … N-1) internally, not datetime.
    # Passing df.index (DatetimeIndex) to fill_between would expand the x-axis
    # to matplotlib date numbers (~738 000), compressing all candles to the left
    # edge.  We use integer positions and convert time_from to a candle index.
    n = len(df)
    x_int = list(range(n))
    for zone in (signal.zones or []):
        ztype = str(zone.get("type", ""))
        color, alpha = _ZONE_COLORS.get(ztype, _ZONE_COLOR_DEFAULT)
        p_lo = float(zone.get("price_from") or 0.0)
        p_hi = float(zone.get("price_to") or 0.0)
        if p_lo <= 0 or p_hi <= p_lo:
            continue

        # Map zone's time_from to a candle index; default to 0 (chart left edge).
        ix_start = 0
        time_from_raw = zone.get("time_from")
        try:
            zone_ts = pd.Timestamp(time_from_raw)
            if zone_ts.tzinfo is not None:
                zone_ts = zone_ts.tz_localize(None)
            idx_found = df.index.searchsorted(zone_ts)
            ix_start = int(min(idx_found, n - 1))
        except Exception:
            ix_start = 0

        mask_int = [i >= ix_start for i in x_int]
        if not any(mask_int):
            continue
        price_ax.fill_between(
            x_int,
            p_lo,
            p_hi,
            where=mask_int,
            facecolor=color,
            alpha=alpha,
            linewidth=0,
            zorder=0,
        )

    # Restore integer x-axis limits after fill_between (in case it drifted).
    price_ax.set_xlim(-0.5, n - 0.5)

    # ── Key price levels + right-edge labels ──────────────────────────────────
    for attr, color, ls, lw, prefix in _LEVEL_SPECS:
        price = float(getattr(signal, attr))
        price_ax.axhline(y=price, color=color, linestyle=ls, linewidth=lw, alpha=0.9)
        price_ax.annotate(
            f"{prefix} {_price_fmt(price)}",
            xy=(1.005, price),
            xycoords=("axes fraction", "data"),
            color=color,
            fontsize=6.5,
            va="center",
            clip_on=False,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    result = buf.read()
    plt.close(fig)
    return result


# ── Dispatcher ────────────────────────────────────────────────────────────────

async def send_signal_alert(
    bot: Bot,
    session: AsyncSession,
    signal: Signal,
    candles_df: pd.DataFrame,
) -> None:
    """Broadcast signal alert (chart PNG + text) to all disclaimer-accepted users.

    Strategy
    --------
    1. Load users with disclaimer_accepted_at set from *session*.
    2. Render chart in a thread pool executor (mplfinance is synchronous/CPU-bound).
    3. If chart render fails → fall back to text-only ``send_message``.
    4. Per-user send errors are caught and logged; one blocked/deactivated user
       does not abort delivery to the rest.
    """
    result = await session.execute(
        select(User).where(User.disclaimer_accepted_at.is_not(None))
    )
    users = result.scalars().all()

    if not users:
        log.debug("alerts_no_users", signal_id=signal.id)
        return

    # Render in thread pool so the event loop is not blocked during PNG generation
    loop = asyncio.get_running_loop()
    chart_bytes: bytes | None
    try:
        chart_bytes = await loop.run_in_executor(
            None, render_signal_chart, signal, candles_df
        )
    except Exception as exc:
        log.warning("alerts_chart_render_failed", signal_id=signal.id, error=str(exc))
        chart_bytes = None

    text = _build_alert_text(signal)
    keyboard = _build_keyboard(signal)

    sent = 0
    for user in users:
        try:
            if chart_bytes is not None:
                await bot.send_photo(
                    chat_id=user.tg_id,
                    photo=BufferedInputFile(
                        chart_bytes, filename=f"signal_{signal.id}.png"
                    ),
                    caption=text,
                    reply_markup=keyboard,
                )
            else:
                await bot.send_message(
                    chat_id=user.tg_id,
                    text=text,
                    reply_markup=keyboard,
                )
            sent += 1
        except Exception as exc:
            log.warning(
                "alerts_user_send_failed",
                tg_id=user.tg_id,
                signal_id=signal.id,
                error=str(exc),
            )

    log.info("alerts_dispatched", signal_id=signal.id, total=len(users), sent=sent)
