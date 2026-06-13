"""Tests for app/analysis/indicators.py.

Synthetic OHLCV datasets with known properties are used so that each test
asserts a specific, analytically derivable result rather than a black-box
library output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis.indicators import (
    add_indicators,
    atr,
    ema,
    relative_volume,
    volume_spike,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 100, base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Reproducible trending OHLCV with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    changes = rng.normal(0.001, 0.005, n)
    closes = base * np.cumprod(1 + changes)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    wicks = np.abs(rng.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + wicks)
    lows = np.minimum(opens, closes) * (1 - wicks)
    volumes = rng.uniform(100.0, 1000.0, n)
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )


def _make_flat(n: int = 50, price: float = 100.0, volume: float = 200.0) -> pd.DataFrame:
    """Flat OHLCV — no price movement, uniform volume."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.full(n, price, dtype=float),
            "high": np.full(n, price, dtype=float),
            "low": np.full(n, price, dtype=float),
            "close": np.full(n, price, dtype=float),
            "volume": np.full(n, volume, dtype=float),
        },
        index=idx,
    )


def _as_orm(df: pd.DataFrame) -> pd.DataFrame:
    """Rename standard OHLCV columns to ORM short names (o/h/l/c/v)."""
    return df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})


# ── EMA ───────────────────────────────────────────────────────────────────────


def test_ema_constant_series_equals_constant() -> None:
    """EMA of a constant series must equal that constant after warm-up."""
    n, price, period = 50, 150.0, 10
    series = pd.Series(np.full(n, price, dtype=float))
    result = ema(series, period)
    valid = result.dropna()
    assert len(valid) > 0
    assert valid.to_numpy() == pytest.approx(price)


def test_ema_period_1_equals_price() -> None:
    """EMA(period=1) has alpha=1 so every output equals the input."""
    df = _make_ohlcv()
    result = ema(df["close"], period=1)
    pd.testing.assert_series_equal(result, df["close"], check_names=False)


def test_ema_nan_prefix_length() -> None:
    """First period-1 values must be NaN."""
    period = 10
    series = pd.Series(np.arange(50, dtype=float))
    result = ema(series, period)
    assert result.iloc[: period - 1].isna().all()
    assert not pd.isna(result.iloc[period - 1])


def test_ema_length_matches_input() -> None:
    series = pd.Series(np.arange(60, dtype=float))
    assert len(ema(series, 5)) == 60


def test_ema_bad_period_raises() -> None:
    with pytest.raises(ValueError, match="period"):
        ema(pd.Series([1.0, 2.0]), period=0)


# ── ATR ───────────────────────────────────────────────────────────────────────


def test_atr_known_values() -> None:
    """Manually verified ATR for a 3-bar sequence with period=2."""
    # Bar 0: prev_close treated as close[0]=100 → TR[0] = max(4, 2, 2) = 4
    # Bar 1: prev_close=100, high=105, low=99   → TR[1] = max(6, 5, 1) = 6
    # Bar 2: prev_close=103, high=104, low=101  → TR[2] = max(3, 1, 2) = 3
    # ATR[1] = mean(4, 6) = 5.0
    # ATR[2] = (5.0 * 1 + 3) / 2 = 4.0
    idx = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 103.0],
            "high": [102.0, 105.0, 104.0],
            "low": [98.0, 99.0, 101.0],
            "close": [100.0, 103.0, 102.0],
            "volume": [100.0, 100.0, 100.0],
        },
        index=idx,
    )
    result = atr(df, period=2)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(5.0)
    assert result.iloc[2] == pytest.approx(4.0)


def test_atr_flat_is_zero() -> None:
    """ATR of a perfectly flat series must be zero after warm-up."""
    df = _make_flat(n=30)
    result = atr(df, period=5)
    valid = result.dropna()
    assert valid.to_numpy() == pytest.approx(0.0, abs=1e-10)


def test_atr_positive_on_real_data() -> None:
    df = _make_ohlcv(n=100)
    result = atr(df, period=14)
    valid = result.dropna()
    assert (valid > 0).all()


def test_atr_nan_prefix_length() -> None:
    """First period-1 values must be NaN."""
    period = 7
    df = _make_ohlcv(n=50)
    result = atr(df, period=period)
    assert result.iloc[: period - 1].isna().all()
    assert not pd.isna(result.iloc[period - 1])


def test_atr_length_matches_input() -> None:
    df = _make_ohlcv(n=80)
    assert len(atr(df, period=14)) == 80


def test_atr_too_few_rows_returns_all_nan() -> None:
    df = _make_flat(n=3)
    result = atr(df, period=14)
    assert result.isna().all()


def test_atr_bad_period_raises() -> None:
    df = _make_flat()
    with pytest.raises(ValueError, match="period"):
        atr(df, period=0)


def test_atr_accepts_orm_columns() -> None:
    df = _make_ohlcv(n=30)
    result_std = atr(df, period=5)
    result_orm = atr(_as_orm(df), period=5)
    pd.testing.assert_series_equal(result_std, result_orm)


# ── Relative volume ───────────────────────────────────────────────────────────


def test_relative_volume_uniform_is_one() -> None:
    """With uniform volume, rel_vol must equal 1.0 for every valid bar."""
    df = _make_flat(n=30, volume=500.0)
    result = relative_volume(df, period=5)
    valid = result.dropna()
    assert valid.to_numpy() == pytest.approx(1.0)


def test_relative_volume_spike_exact() -> None:
    """A bar with 3× normal volume produces rel_vol == 3.0."""
    n, period, spike_pos = 30, 5, 20
    volumes = np.full(n, 100.0)
    volumes[spike_pos] = 300.0
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.ones(n),
            "high": np.ones(n) + 0.5,
            "low": np.ones(n) - 0.5,
            "close": np.ones(n),
            "volume": volumes,
        },
        index=idx,
    )
    result = relative_volume(df, period=period)
    # lookback window [15..19] = all 100 → avg = 100; rel_vol = 300 / 100 = 3.0
    assert result.iloc[spike_pos] == pytest.approx(3.0)


def test_relative_volume_nan_prefix() -> None:
    """First `period` values must be NaN (no complete lookback window yet)."""
    period = 10
    df = _make_ohlcv(n=50)
    result = relative_volume(df, period=period)
    assert result.iloc[:period].isna().all()
    assert not pd.isna(result.iloc[period])


def test_relative_volume_length_matches() -> None:
    df = _make_ohlcv(n=60)
    assert len(relative_volume(df, period=10)) == 60


def test_relative_volume_bad_period_raises() -> None:
    df = _make_flat()
    with pytest.raises(ValueError, match="period"):
        relative_volume(df, period=0)


def test_relative_volume_accepts_orm_columns() -> None:
    df = _make_ohlcv(n=40)
    result_std = relative_volume(df, period=5)
    result_orm = relative_volume(_as_orm(df), period=5)
    pd.testing.assert_series_equal(result_std, result_orm)


# ── Volume spike ──────────────────────────────────────────────────────────────


def test_volume_spike_detects_spike() -> None:
    """A 3× spike with threshold=2.0 must be flagged True."""
    n, spike_pos = 30, 20
    volumes = np.full(n, 100.0)
    volumes[spike_pos] = 300.0
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.ones(n),
            "high": np.ones(n) + 0.5,
            "low": np.ones(n) - 0.5,
            "close": np.ones(n),
            "volume": volumes,
        },
        index=idx,
    )
    result = volume_spike(df, period=5, threshold=2.0)
    assert result.iloc[spike_pos] is True or result.iloc[spike_pos] == True  # noqa: E712
    # Normal bars after warm-up must be False
    assert not result.iloc[25]


def test_volume_spike_is_bool_series() -> None:
    df = _make_ohlcv(n=40)
    result = volume_spike(df, period=5)
    valid = result.dropna()
    assert valid.dtype == bool


def test_volume_spike_uniform_volume_no_spikes() -> None:
    """Uniform volume never triggers a spike regardless of threshold."""
    df = _make_flat(n=30, volume=100.0)
    result = volume_spike(df, period=5, threshold=1.5)
    assert not result.dropna().any()


def test_volume_spike_bad_threshold_raises() -> None:
    df = _make_flat()
    with pytest.raises(ValueError, match="threshold"):
        volume_spike(df, threshold=0.0)


# ── add_indicators ────────────────────────────────────────────────────────────


def test_add_indicators_columns_present() -> None:
    """add_indicators must add atr, ema_N, rel_vol, vol_spike columns."""
    df = _make_ohlcv(n=100)
    periods = (10, 20, 50)
    result = add_indicators(df, ema_periods=periods, atr_period=5, volume_period=5)
    expected_new = {"atr", "rel_vol", "vol_spike"} | {f"ema_{p}" for p in periods}
    assert expected_new.issubset(result.columns)


def test_add_indicators_does_not_mutate_input() -> None:
    df = _make_ohlcv(n=50)
    original_cols = set(df.columns)
    add_indicators(df)
    assert set(df.columns) == original_cols


def test_add_indicators_length_preserved() -> None:
    df = _make_ohlcv(n=80)
    result = add_indicators(df)
    assert len(result) == 80


def test_add_indicators_orm_format() -> None:
    """add_indicators must accept ORM short column names (o/h/l/c/v)."""
    df = _make_ohlcv(n=60)
    orm_df = _as_orm(df)
    result = add_indicators(orm_df, ema_periods=(10,), atr_period=5, volume_period=5)
    assert "atr" in result.columns
    assert "ema_10" in result.columns
    # Standard names should be present in output (normalized from ORM)
    assert "close" in result.columns


def test_add_indicators_orm_matches_standard() -> None:
    """ORM-format and standard-format must yield identical indicator values."""
    df = _make_ohlcv(n=60)
    periods = (10, 20)
    kw = {"ema_periods": periods, "atr_period": 5, "volume_period": 5}
    result_std = add_indicators(df, **kw)
    result_orm = add_indicators(_as_orm(df), **kw)
    indicator_cols = ["atr", "rel_vol", "vol_spike"] + [f"ema_{p}" for p in periods]
    for col in indicator_cols:
        pd.testing.assert_series_equal(
            result_std[col], result_orm[col], check_names=False
        )
