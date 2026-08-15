"""Causal displacement features and event detection."""

from __future__ import annotations

import pandas as pd


def displacement_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    body_ratio_threshold: float = 0.65,
    range_atr_threshold: float = 1.5,
    volume_z_threshold: float | None = 1.5,
    volume_period: int = 20,
) -> pd.DataFrame:
    """Calculate normalized candle-displacement features.

    Displacement is deliberately multi-factor: range relative to ATR, body
    share of total range, and optionally abnormal volume. The returned flag
    is descriptive; it is not a trading signal.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if atr_period < 1 or volume_period < 1:
        raise ValueError("periods must be >= 1")
    if range_atr_threshold <= 0 or not 0 < body_ratio_threshold <= 1:
        raise ValueError("invalid displacement thresholds")

    result = df.copy()
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    open_ = result["open"].astype(float)
    close = result["close"].astype(float)
    volume = result["volume"].astype(float)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(atr_period, min_periods=atr_period).mean()

    candle_range = high - low
    body = (close - open_).abs()
    body_ratio = body.div(candle_range.replace(0, pd.NA)).fillna(0.0)
    range_atr = candle_range.div(atr.replace(0, pd.NA))

    result["true_range"] = true_range
    result["atr"] = atr
    result["range_atr"] = range_atr
    result["body_ratio"] = body_ratio
    result["direction"] = (close > open_).map({True: "bullish", False: "bearish"})

    if volume_z_threshold is not None:
        mean = volume.rolling(volume_period, min_periods=volume_period).mean()
        std = volume.rolling(volume_period, min_periods=volume_period).std(ddof=0)
        result["volume_zscore"] = (volume - mean).div(std.replace(0, pd.NA))
        volume_ok = result["volume_zscore"] >= volume_z_threshold
    else:
        result["volume_zscore"] = pd.NA
        volume_ok = True

    result["is_displacement"] = (
        (result["range_atr"] >= range_atr_threshold)
        & (result["body_ratio"] >= body_ratio_threshold)
        & volume_ok
    )
    return result
