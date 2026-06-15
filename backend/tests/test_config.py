"""Tests for Settings loading and field validation."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError


def test_settings_reads_test_env() -> None:
    from app.config import settings

    assert settings.database_url == "sqlite+aiosqlite:///:memory:"
    assert settings.jwt_algorithm == "HS256"
    assert settings.jwt_expire_minutes == 15
    assert settings.signal_min_score == 55
    assert settings.risk_pct_min == 0.5
    assert settings.risk_pct_max == 3.0
    assert settings.max_concurrent_positions == 3
    assert settings.daily_loss_pause_pct == 5.0
    assert settings.use_testnet is True


def test_master_key_decodes_to_32_bytes() -> None:
    from app.config import settings

    raw = bytes.fromhex(settings.master_encryption_key)
    assert len(raw) == 32


def test_invalid_master_key_not_hex() -> None:
    from app.config import Settings

    backup = os.environ.copy()
    try:
        os.environ["MASTER_ENCRYPTION_KEY"] = "zz" + "00" * 31  # invalid hex
        with pytest.raises(ValidationError, match="valid hex string"):
            Settings()
    finally:
        os.environ.clear()
        os.environ.update(backup)


def test_invalid_master_key_too_short() -> None:
    # "ab" * 10 = 20 chars, fails pydantic min_length=64 before the custom validator.
    from app.config import Settings

    backup = os.environ.copy()
    try:
        os.environ["MASTER_ENCRYPTION_KEY"] = "ab" * 10
        with pytest.raises(ValidationError, match="at least 64 characters"):
            Settings()
    finally:
        os.environ.clear()
        os.environ.update(backup)


def test_invalid_master_key_too_short_string() -> None:
    from app.config import Settings

    backup = os.environ.copy()
    try:
        os.environ["MASTER_ENCRYPTION_KEY"] = "aabb"  # only 2 bytes
        with pytest.raises(ValidationError):
            Settings()
    finally:
        os.environ.clear()
        os.environ.update(backup)


def test_signal_min_score_bounds() -> None:
    from app.config import Settings

    backup = os.environ.copy()
    try:
        os.environ["SIGNAL_MIN_SCORE"] = "150"
        with pytest.raises(ValidationError):
            Settings()
    finally:
        os.environ.clear()
        os.environ.update(backup)


def test_cors_origins_default_contains_expected() -> None:
    from app.config import settings

    assert isinstance(settings.cors_origins, list)
    assert len(settings.cors_origins) >= 1
