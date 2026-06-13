"""ORM models — full SmartFlow database schema (spec §3).

All timestamps are timezone-aware UTC.
JSON columns use sqlalchemy.JSON, compatible with both SQLite and PostgreSQL.
Enum columns use native_enum=False for cross-database portability.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ── users ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    tg_id: Mapped[int] = mapped_column(sa.BigInteger, unique=True, nullable=False, index=True)
    username: Mapped[str | None] = mapped_column(sa.String(64))
    plan: Mapped[str] = mapped_column(
        sa.Enum("free", "pro", "auto", name="plan_enum", native_enum=False),
        default="free",
        server_default="free",
        nullable=False,
    )
    plan_until: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    risk_pct: Mapped[float] = mapped_column(
        sa.Float, default=1.0, server_default="1.0", nullable=False
    )
    max_positions: Mapped[int] = mapped_column(
        sa.Integer, default=3, server_default="3", nullable=False
    )
    autotrade_paused_until: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    disclaimer_accepted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, nullable=False
    )

    api_keys: Mapped[list[ApiKey]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    positions: Mapped[list[Position]] = relationship(back_populates="user")
    payments: Mapped[list[Payment]] = relationship(back_populates="user")
    audit_logs: Mapped[list[AuditLog]] = relationship(back_populates="user")


# ── api_keys ──────────────────────────────────────────────────────────────────

class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    exchange: Mapped[str] = mapped_column(
        sa.Enum("binance", "bybit", name="exchange_enum", native_enum=False),
        nullable=False,
    )
    key_encrypted: Mapped[bytes] = mapped_column(sa.LargeBinary, nullable=False)
    secret_encrypted: Mapped[bytes] = mapped_column(sa.LargeBinary, nullable=False)
    nonce: Mapped[bytes] = mapped_column(sa.LargeBinary, nullable=False)
    # "****a3f9" — last 4 chars of the public key, shown in UI only
    label_mask: Mapped[str] = mapped_column(sa.String(12), nullable=False)
    perms_checked_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, nullable=False
    )

    user: Mapped[User] = relationship(back_populates="api_keys")


# ── candles ───────────────────────────────────────────────────────────────────

class Candle(Base):
    __tablename__ = "candles"

    symbol: Mapped[str] = mapped_column(sa.String(20), primary_key=True)
    timeframe: Mapped[str] = mapped_column(sa.String(4), primary_key=True)  # 15m/1h/4h
    ts: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), primary_key=True)
    o: Mapped[float] = mapped_column(sa.Float, nullable=False)
    h: Mapped[float] = mapped_column(sa.Float, nullable=False)
    l: Mapped[float] = mapped_column(sa.Float, nullable=False)  # noqa: E741 — OHLCV convention
    c: Mapped[float] = mapped_column(sa.Float, nullable=False)
    v: Mapped[float] = mapped_column(sa.Float, nullable=False)


# ── signals ───────────────────────────────────────────────────────────────────

class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(sa.String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(
        sa.Enum("long", "short", name="side_enum", native_enum=False),
        nullable=False,
    )
    timeframe: Mapped[str] = mapped_column(sa.String(4), nullable=False)
    score: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    entry_low: Mapped[float] = mapped_column(sa.Float, nullable=False)
    entry_high: Mapped[float] = mapped_column(sa.Float, nullable=False)
    sl: Mapped[float] = mapped_column(sa.Float, nullable=False)
    tp1: Mapped[float] = mapped_column(sa.Float, nullable=False)
    tp2: Mapped[float] = mapped_column(sa.Float, nullable=False)
    rr: Mapped[float] = mapped_column(sa.Float, nullable=False)
    # {"sweep": true, "ob_retest": true, "funding": -0.018, ...}
    factors: Mapped[dict] = mapped_column(sa.JSON, nullable=False, default=dict)  # type: ignore[type-arg]
    # [{"type": "OB", "price_from": ..., "price_to": ..., "time_from": ..., "time_to": ...}]
    zones: Mapped[list] = mapped_column(sa.JSON, nullable=False, default=list)  # type: ignore[type-arg]
    ai_explanation: Mapped[str | None] = mapped_column(sa.Text)
    news_context: Mapped[dict | None] = mapped_column(sa.JSON)  # type: ignore[type-arg]
    status: Mapped[str] = mapped_column(
        sa.Enum("active", "expired", "tp", "sl", name="signal_status_enum", native_enum=False),
        default="active",
        server_default="active",
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
    resolved_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))

    positions: Mapped[list[Position]] = relationship(back_populates="signal")


# ── positions ─────────────────────────────────────────────────────────────────

class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        sa.ForeignKey("users.id"), nullable=False, index=True
    )
    signal_id: Mapped[int] = mapped_column(sa.ForeignKey("signals.id"), nullable=False)
    exchange: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    # clientOrderId sent to the exchange — guarantees idempotency
    client_order_id: Mapped[uuid.UUID] = mapped_column(
        sa.Uuid, unique=True, nullable=False, default=uuid.uuid4
    )
    qty: Mapped[float] = mapped_column(sa.Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(sa.Float, nullable=False)
    sl: Mapped[float] = mapped_column(sa.Float, nullable=False)
    tp1: Mapped[float] = mapped_column(sa.Float, nullable=False)
    tp2: Mapped[float] = mapped_column(sa.Float, nullable=False)
    status: Mapped[str] = mapped_column(
        sa.Enum(
            "pending", "open", "closed", "cancelled",
            name="position_status_enum",
            native_enum=False,
        ),
        default="pending",
        server_default="pending",
        nullable=False,
    )
    pnl_usd: Mapped[float | None] = mapped_column(sa.Float)
    pnl_pct: Mapped[float | None] = mapped_column(sa.Float)
    opened_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, nullable=False
    )
    closed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="positions")
    signal: Mapped[Signal] = relationship(back_populates="positions")


# ── audit_log ─────────────────────────────────────────────────────────────────

class AuditLog(Base):
    """Append-only log: no UPDATE or DELETE at application level."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        sa.ForeignKey("users.id"), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(sa.Text, nullable=False)
    payload: Mapped[dict] = mapped_column(sa.JSON, nullable=False, default=dict)  # type: ignore[type-arg]
    exchange_response: Mapped[dict | None] = mapped_column(sa.JSON)  # type: ignore[type-arg]
    ts: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    user: Mapped[User] = relationship(back_populates="audit_logs")


# ── payments ──────────────────────────────────────────────────────────────────

class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        sa.ForeignKey("users.id"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(
        sa.Enum("stars", "crypto", name="provider_enum", native_enum=False),
        nullable=False,
    )
    amount: Mapped[float] = mapped_column(sa.Float, nullable=False)
    currency: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    plan: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    external_id: Mapped[str | None] = mapped_column(sa.String(128))
    is_recurring: Mapped[bool] = mapped_column(sa.Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), default=_utcnow, nullable=False
    )

    user: Mapped[User] = relationship(back_populates="payments")


# ── derivatives_snapshot ─────────────────────────────────────────────────────

class DerivativesSnapshot(Base):
    """Perpetual-futures derivatives data per symbol, one row per collection run."""

    __tablename__ = "derivatives_snapshot"
    __table_args__ = (
        sa.Index("ix_derivatives_snapshot_symbol_ts", "symbol", "ts"),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    ts: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    funding_rate: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    open_interest: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    long_short_ratio: Mapped[float | None] = mapped_column(sa.Float, nullable=True)


# ── market_sentiment ──────────────────────────────────────────────────────────

class MarketSentiment(Base):
    """One row per hourly Fear & Greed update. Deduped by ts."""

    __tablename__ = "market_sentiment"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, unique=True, index=True
    )
    fear_greed_value: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    classification: Mapped[str] = mapped_column(sa.String(32), nullable=False)


# ── news_items ────────────────────────────────────────────────────────────────

class NewsItem(Base):
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    source: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    title: Mapped[str] = mapped_column(sa.Text, nullable=False)
    url: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    symbols: Mapped[list] = mapped_column(sa.JSON, nullable=False, default=list)  # type: ignore[type-arg]
    sentiment: Mapped[int | None] = mapped_column(sa.Integer)   # -10..+10
    importance: Mapped[int | None] = mapped_column(sa.Integer)  # 1..5
    published_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False, index=True
    )
