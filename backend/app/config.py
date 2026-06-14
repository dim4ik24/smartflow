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
    gemini_model: str = "gemini-2.0-flash"
    sentiment_batch_size: int = 20
    sentiment_analyze_interval_minutes: int = 10

    # ── Data sources (no API keys required) ──────────────────────────────────
    fear_greed_url: str = "https://api.alternative.me/fng/"
    coingecko_api_url: str = "https://api.coingecko.com/api/v3"

    # ── News collector ────────────────────────────────────────────────────────
    news_collect_interval_minutes: int = 10
    news_rss_feeds: list[str] = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://www.theblock.co/rss.xml",
        "https://decrypt.co/feed",
        "https://bitcoinmagazine.com/.rss/full/",
    ]
    # Mapping from base ticker to search terms (word-boundary matched in news text).
    # Multi-word terms like "NEAR Protocol" are matched as full phrases.
    coin_synonyms: dict[str, list[str]] = {
        "BTC":   ["Bitcoin",        "BTC"],
        "ETH":   ["Ethereum",       "ETH",   "Ether"],
        "SOL":   ["Solana",         "SOL"],
        "BNB":   ["BNB",            "Binance Coin"],
        "XRP":   ["XRP",            "Ripple"],
        "DOGE":  ["Dogecoin",       "DOGE"],
        "ADA":   ["Cardano",        "ADA"],
        "AVAX":  ["Avalanche",      "AVAX"],
        "DOT":   ["Polkadot",       "DOT"],
        "LINK":  ["Chainlink",      "LINK"],
        "MATIC": ["Polygon",        "MATIC",  "POL"],
        "UNI":   ["Uniswap",        "UNI"],
        "ATOM":  ["Cosmos",         "ATOM"],
        "LTC":   ["Litecoin",       "LTC"],
        "NEAR":  ["NEAR Protocol",  "NEAR"],
    }

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
    derivatives_collect_interval_minutes: int = 5

    # ── Scoring weights (SPEC §6; calibrated by backtest) ─────────────────────
    score_weight_sweep: int = 25
    score_weight_ob_retest: int = 20
    score_weight_fvg: int = 10
    score_weight_structure: int = 15
    score_weight_funding: int = 10
    score_weight_oi_rising: int = 3   # ΔOI > 0: open interest grew → conviction
    score_weight_lsr: int = 2         # long/short ratio confirms direction
    score_weight_sentiment: int = 10
    score_weight_premium_discount: int = 5
    score_min_rr: float = 2.0
    score_funding_extreme_threshold: float = 0.00005  # |funding| ≥ this = extreme
    score_sentiment_threshold: float = 1.0            # |avg_sentiment| ≥ this = agrees
    # Maximum OB width as a fraction of current price (e.g. 0.015 = 1.5 %).
    # Wider zones are swing-range library artifacts, not single-candle order blocks.
    score_max_ob_width_pct: float = 0.015
    # Maximum distance from current price to OB mid-point, expressed in ATR units.
    # If the mid-entry is further than this the market hasn't reached the zone yet
    # ("setup not ripe") and the candidate is discarded.
    score_max_entry_atr_distance: float = 3.0
    # Maximum distance (in ATR units) from current price for a FVG to be counted
    # as "confirming" the setup.  The FVG's price range must overlap the band
    # [current_price ± score_fvg_max_atr_distance × ATR].
    score_fvg_max_atr_distance: float = 3.0
    # Only FVGs formed within the last N candles are considered recent enough to
    # be relevant.  In a 200-candle rolling window almost all FVGs are already
    # mitigated, so stale ones add noise rather than signal.
    score_fvg_recency_candles: int = 25
    analysis_candle_limit: int = 200

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
