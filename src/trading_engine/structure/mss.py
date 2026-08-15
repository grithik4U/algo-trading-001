"""Causal market-structure shift (MSS) detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class SwingPoint:
    timestamp: datetime
    price: float
    kind: str  # swing_high / swing_low


@dataclass(frozen=True)
class StructureShift:
    timestamp: datetime
    direction: str
    broken_level: float
    swing_timestamp: datetime
    displacement_confirmed: bool


def confirm_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> list[SwingPoint]:
    """Return only swings confirmed after ``right`` future bars have closed."""
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("high and low columns are required")
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")

    points: list[SwingPoint] = []
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    for i in range(left, len(df) - right):
        high_window = highs.iloc[i - left : i + right + 1]
        low_window = lows.iloc[i - left : i + right + 1]
        if highs.iloc[i] == high_window.max() and (high_window == highs.iloc[i]).sum() == 1:
            points.append(SwingPoint(df.index[i], float(highs.iloc[i]), "swing_high"))
        if lows.iloc[i] == low_window.min() and (low_window == lows.iloc[i]).sum() == 1:
            points.append(SwingPoint(df.index[i], float(lows.iloc[i]), "swing_low"))
    return sorted(points, key=lambda x: x.timestamp)


def detect_mss(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    direction: str,
    start_after: datetime | None = None,
) -> StructureShift | None:
    """Detect the first close through the latest opposing confirmed swing."""
    if "close" not in df.columns:
        raise ValueError("close column is required")
    if direction not in {"bullish", "bearish"}:
        raise ValueError("direction must be bullish or bearish")

    eligible = [s for s in swings if start_after is None or s.timestamp >= start_after]
    opposing_kind = "swing_high" if direction == "bullish" else "swing_low"
    targets = [s for s in eligible if s.kind == opposing_kind]
    if not targets:
        return None
    target = targets[-1]

    future = df.loc[df.index > target.timestamp]
    for timestamp, row in future.iterrows():
        close = float(row["close"])
        broken = close > target.price if direction == "bullish" else close < target.price
        if broken:
            return StructureShift(timestamp, direction, target.price, target.timestamp, False)
    return None
