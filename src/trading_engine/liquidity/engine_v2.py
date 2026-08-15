"""Temporary corrected liquidity engine implementation."""

from __future__ import annotations

import pandas as pd
from trading_engine.structure.swings import detect_swings


class LiquidityEngine:
    def __init__(self, left_bars=2, right_bars=2, external_lookback=50, equal_tolerance=0.0005):
        if external_lookback <= 0:
            raise ValueError("external_lookback must be > 0")
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.external_lookback = external_lookback
        self.equal_tolerance = equal_tolerance

    def levels(self, df: pd.DataFrame) -> pd.DataFrame:
        swings = detect_swings(df, self.left_bars, self.right_bars)
        records = []
        latest_position = len(df) - 1
        for position, (timestamp, row) in enumerate(swings.iterrows()):
            scope = "external" if latest_position - position >= self.external_lookback else "internal"
            strength = 1.0 if scope == "external" else 0.5
            if pd.notna(row["swing_high"]):
                records.append({"timestamp": timestamp, "price": float(row["swing_high"]), "side": "BSL", "scope": scope, "strength": strength, "confirmed_at": row["swing_high_confirmed_at"]})
            if pd.notna(row["swing_low"]):
                records.append({"timestamp": timestamp, "price": float(row["swing_low"]), "side": "SSL", "scope": scope, "strength": strength, "confirmed_at": row["swing_low_confirmed_at"]})
        return pd.DataFrame(records, columns=["timestamp", "price", "side", "scope", "strength", "confirmed_at"])
