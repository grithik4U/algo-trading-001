"""Swing-point detection.

A swing is only *confirmed* after ``right_bars`` future observations exist.
Callers must use the confirmation timestamp for live/backtest signal timing;
the pivot timestamp describes where the structural extreme occurred.
"""

from __future__ import annotations

import pandas as pd


def detect_swings(
    df: pd.DataFrame,
    left_bars: int = 2,
    right_bars: int = 2,
) -> pd.DataFrame:
    """Return confirmed swing highs/lows for OHLC data.

    Required columns: ``high`` and ``low``.

    Returns a copy with:
      - ``swing_high``: pivot high price or NaN
      - ``swing_low``: pivot low price or NaN
      - ``swing_high_confirmed_at``: confirmation timestamp
      - ``swing_low_confirmed_at``: confirmation timestamp

    The implementation deliberately does not expose a pivot as confirmed
    until the right-side bars have occurred, reducing look-ahead risk.
    """
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must be >= 1")
    required = {"high", "low"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["swing_high"] = float("nan")
    out["swing_low"] = float("nan")
    out["swing_high_confirmed_at"] = pd.NaT
    out["swing_low_confirmed_at"] = pd.NaT

    for pivot_pos in range(left_bars, len(out) - right_bars):
        left = pivot_pos - left_bars
        right = pivot_pos + right_bars
        high_window = out["high"].iloc[left : right + 1]
        low_window = out["low"].iloc[left : right + 1]
        pivot_high = out["high"].iloc[pivot_pos]
        pivot_low = out["low"].iloc[pivot_pos]

        if pivot_high == high_window.max() and (high_window == pivot_high).sum() == 1:
            pivot_index = out.index[pivot_pos]
            confirmed_index = out.index[right]
            out.at[pivot_index, "swing_high"] = pivot_high
            out.at[pivot_index, "swing_high_confirmed_at"] = confirmed_index

        if pivot_low == low_window.min() and (low_window == pivot_low).sum() == 1:
            pivot_index = out.index[pivot_pos]
            confirmed_index = out.index[right]
            out.at[pivot_index, "swing_low"] = pivot_low
            out.at[pivot_index, "swing_low_confirmed_at"] = confirmed_index

    return out
