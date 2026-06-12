"""Application settings loaded from .env via pydantic-settings.

All secret fields are required — no code-level defaults for credentials.
Tests set required env vars in tests/conftest.py before importing this module.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    environment: Literal["development", "staging", "production"] = "development"

    # ── Database ──────────────────────────────────────────────────────────────
    # SQLite (dev):  sqlite+aiosqlite:///./smartflow.db
    # PostgreSQL:    postgresql+asyncpg://user:pass@host:5432/smartflow
    database_url: str = "sqlite+aiosqlite:///./smartflow.db"

    # ── Security — JWT ────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 15

    # ── Security — AES-256-GCM master key for exchange API keys ───────────────
    # Must be exactly 64 hex characters (= 32 bytes).
    # Generate: python -c "import secrets; print(secrets.token_hex(32))"
    master_encryption_key: str = Field(min_length=64, max_length=64)

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_bot_token: str
    telegram_webhook_secret: str
    telegram_webhook_url: str = ""

    # ── AI ────────────────────────────────────────────────────────────────────
    gemini_api_key: str = ""

    # ── Data sources ──────────────────────────────────────────────────────────
    cryptopanic_api_key: str = ""
    fear_greed_url: str = "https://api.alternative.me/fng/"

    # ── Billing — NOWPayments ─────────────────────────────────────────────────
    nowpayments_api_key: str = ""
    nowpayments_ipn_secret: str = ""

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["https://smartflow.app", "http://localhost:3000"]

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_default: str = "60/minute"
    rate_limit_auth: str = "10/minute"
    rate_limit_trade: str = "5/minute"

    # ── Trading defaults ──────────────────────────────────────────────────────
    use_testnet: bool = True
    signal_min_score: int = Field(default=70, ge=0, le=100)
    signal_ttl_hours: int = 2
    signal_entry_drift_pct: float = 0.5
    macro_event_window_minutes: int = 30
    daily_loss_pause_pct: float = Field(default=5.0, ge=0.0)
    max_concurrent_positions: int = Field(default=3, ge=1)
    risk_pct_min: float = Field(default=0.5, ge=0.0)
    risk_pct_max: float = Field(default=3.0, le=100.0)

    # ── Collector ─────────────────────────────────────────────────────────────
    collector_exchange: Literal["binance", "bybit"] = "bybit"
    watched_symbols: list[str] = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
        "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT",
    ]
    watched_timeframes: list[str] = ["15m", "1h", "4h"]
    collector_heartbeat_timeout: int = 30   # seconds; reconnect if no data
    collector_backfill_limit: int = 500     # candles per REST fetch on gap-fill

    @field_validator("master_encryption_key")
    @classmethod
    def validate_master_key(cls, v: str) -> str:
        # min_length=64 / max_length=64 already enforce 32-byte output for valid hex.
        # This check catches strings that are 64 chars but contain non-hex characters.
        try:
            bytes.fromhex(v)
        except ValueError as exc:
            raise ValueError("master_encryption_key must be a valid hex string") from exc
        return v


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (use for FastAPI Depends)."""
    return Settings()


# Module-level singleton for use outside of FastAPI dependency injection.
settings: Settings = get_settings()
