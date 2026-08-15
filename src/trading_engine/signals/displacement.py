"""Causal displacement features and post-sweep event detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


def displacement_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    body_ratio_threshold: float = 0.65,
    range_atr_threshold: float = 1.5,
    volume_z_threshold: float | None = 1.5,
    volume_period: int = 20,
) -> pd.DataFrame:
    """Calculate normalized candle-displacement features."""
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
    body_ratio = (close - open_).abs().div(candle_range.replace(0, pd.NA)).fillna(0.0)
    result["true_range"] = true_range
    result["atr"] = atr
    result["range_atr"] = candle_range.div(atr.replace(0, pd.NA))
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


@dataclass(frozen=True)
class DisplacementEvent:
    timestamp: datetime
    direction: str
    start_price: float
    end_price: float
    move: float
    bars_after_sweep: int
    range_multiple: float
    body_ratio: float
    broke_structure: bool


def detect_post_sweep_displacement(
    df: pd.DataFrame,
    sweep_timestamp: datetime,
    direction: str,
    structure_level: float | None = None,
    max_bars: int = 3,
    range_multiple_threshold: float = 1.5,
    body_ratio_threshold: float = 0.7,
) -> DisplacementEvent | None:
    """Find the first qualifying displacement after a liquidity sweep."""
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if direction not in {"bullish", "bearish"}:
        raise ValueError("direction must be bullish or bearish")
    if max_bars < 1 or range_multiple_threshold <= 0 or not 0 <= body_ratio_threshold <= 1:
        raise ValueError("invalid displacement thresholds")

    bars = df.reset_index()
    timestamp_col = df.index.name or "index"
    bars = bars.rename(columns={timestamp_col: "timestamp"})
    prior = bars.loc[bars["timestamp"] <= sweep_timestamp].copy()
    future = bars.loc[bars["timestamp"] > sweep_timestamp].head(max_bars)
    if future.empty or prior.empty:
        return None
    prior_range = (prior["high"] - prior["low"]).replace(0, pd.NA).dropna()
    if prior_range.empty:
        return None
    baseline = float(prior_range.median())
    start_price = float(prior.iloc[-1]["close"])
    for i, (_, bar) in enumerate(future.iterrows(), start=1):
        high, low = float(bar["high"]), float(bar["low"])
        open_, close = float(bar["open"]), float(bar["close"])
        candle_range = high - low
        if candle_range <= 0:
            continue
        body_ratio = abs(close - open_) / candle_range
        move = close - start_price
        directional = move > 0 if direction == "bullish" else move < 0
        range_multiple = candle_range / baseline if baseline > 0 else 0.0
        broke = ((close > structure_level) if direction == "bullish" else (close < structure_level)) if structure_level is not None else False
        if directional and range_multiple >= range_multiple_threshold and body_ratio >= body_ratio_threshold:
            return DisplacementEvent(bar["timestamp"], direction, start_price, close, move, i, range_multiple, body_ratio, broke)
    return None
