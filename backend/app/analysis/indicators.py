"""Technical indicators — pure-pandas implementation (no external TA libraries).

Public API
----------
ema(series, period)
    Exponential Moving Average.
atr(ohlc, period=14)
    Average True Range (Wilder smoothing).
relative_volume(ohlc, period=20)
    Current bar volume as a multiple of its rolling lookback average.
volume_spike(ohlc, period=20, threshold=2.0)
    Boolean Series — True where volume ≥ threshold × rolling average.
add_indicators(ohlc, ...)
    Convenience wrapper: returns a copy of ohlc with all indicators as
    new columns (atr, ema_N, rel_vol, vol_spike).

Column format
-------------
All functions accept either standard names (open / high / low / close / volume)
or ORM short names (o / h / l / c / v) — normalized internally, no rename needed
by the caller.
"""
from __future__ import annotations

import pandas as pd

_ORM_COL_MAP: dict[str, str] = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _normalize_columns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Rename ORM short names (o/h/l/c/v) to standard names if present."""
    if _ORM_COL_MAP.keys() & set(ohlc.columns):
        return ohlc.rename(columns=_ORM_COL_MAP)
    return ohlc


def _require_columns(df: pd.DataFrame, cols: frozenset[str]) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")


# ── Public functions ──────────────────────────────────────────────────────────


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (alpha = 2 / (period + 1)).

    Uses Wilder-style initialisation via pandas ewm: the first valid EMA value
    is the simple mean of the first ``period`` observations; afterwards the
    standard EMA recurrence applies.

    Returns a Series of the same length; values before the warm-up period are
    NaN.

    Parameters
    ----------
    series:
        Numeric Series (typically ``ohlc["close"]``).
    period:
        Look-back window; must be >= 1.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1; got {period}")
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing.

    True Range = max(high − low, |high − prev_close|, |low − prev_close|).
    For the very first bar, prev_close is treated as that bar's close so
    TR[0] = high[0] − low[0].

    Wilder initialisation: ATR[period−1] = mean(TR[0 : period]);
    subsequent: ATR[i] = (ATR[i−1] × (period−1) + TR[i]) / period.

    Returns a Series indexed like ``ohlc`` with NaN where data is insufficient.

    Parameters
    ----------
    ohlc:
        OHLCV DataFrame (standard or ORM column names).
    period:
        Smoothing window; must be >= 1.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1; got {period}")
    df = _normalize_columns(ohlc)
    _require_columns(df, frozenset({"high", "low", "close"}))

    prev_close = df["close"].shift(1).fillna(df["close"])
    tr: pd.Series = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out = pd.Series(float("nan"), index=df.index, name="atr", dtype=float)
    if len(tr) < period:
        return out
    out.iloc[period - 1] = tr.iloc[:period].mean()
    for i in range(period, len(tr)):
        out.iloc[i] = (out.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
    return out


def relative_volume(ohlc: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume of the current bar relative to the rolling mean of the previous
    ``period`` bars (look-back only — the current bar is excluded from the
    average to avoid self-reference).

    A value of 1.0 means average volume; 2.0 means double average, etc.
    Returns NaN for the first ``period`` bars (insufficient history).

    Parameters
    ----------
    ohlc:
        OHLCV DataFrame (standard or ORM column names).
    period:
        Rolling window length for the lookback average; must be >= 1.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1; got {period}")
    df = _normalize_columns(ohlc)
    _require_columns(df, frozenset({"volume"}))

    avg = df["volume"].shift(1).rolling(period, min_periods=period).mean()
    out = df["volume"] / avg
    out.name = "rel_vol"
    return out


def volume_spike(
    ohlc: pd.DataFrame,
    period: int = 20,
    threshold: float = 2.0,
) -> pd.Series:
    """True where bar volume ≥ ``threshold`` × rolling lookback average.

    Parameters
    ----------
    ohlc:
        OHLCV DataFrame (standard or ORM column names).
    period:
        Rolling window for the baseline average; must be >= 1.
    threshold:
        Multiplier above which a bar is flagged as a spike; must be > 0.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0; got {threshold}")
    out = relative_volume(ohlc, period) >= threshold
    out.name = "vol_spike"
    return out


def add_indicators(
    ohlc: pd.DataFrame,
    *,
    atr_period: int = 14,
    ema_periods: tuple[int, ...] = (20, 50, 200),
    volume_period: int = 20,
    volume_spike_threshold: float = 2.0,
) -> pd.DataFrame:
    """Return a copy of ``ohlc`` with all indicator columns appended.

    The returned DataFrame uses standard column names (open/high/low/close/
    volume) regardless of the input format.

    New columns
    -----------
    atr          : Average True Range
    ema_<N>      : EMA for each value in *ema_periods* (e.g. ema_20, ema_50)
    rel_vol      : Relative volume (current ÷ prior rolling average)
    vol_spike    : Boolean — True when rel_vol ≥ volume_spike_threshold

    Parameters
    ----------
    ohlc:
        OHLCV DataFrame (standard or ORM column names, DatetimeIndex).
    atr_period:
        ATR smoothing window (default 14).
    ema_periods:
        Tuple of EMA periods to compute (default (20, 50, 200)).
    volume_period:
        Rolling window for relative-volume baseline (default 20).
    volume_spike_threshold:
        Relative-volume multiplier that triggers the spike flag (default 2.0).
    """
    df = _normalize_columns(ohlc).copy()
    df["atr"] = atr(df, atr_period)
    close = df["close"]
    for p in ema_periods:
        df[f"ema_{p}"] = ema(close, p)
    df["rel_vol"] = relative_volume(df, volume_period)
    df["vol_spike"] = volume_spike(df, volume_period, volume_spike_threshold)
    return df
