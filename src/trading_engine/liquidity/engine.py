"""Initial market-liquidity engine.

This module deliberately treats "internal" and "external" liquidity as
research labels, not facts. The first implementation creates structural
levels from confirmed swings and classifies them using a configurable
lookback horizon. More sophisticated session/HTF scoring will be added in
later phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from trading_engine.structure.swings import detect_swings


Side = Literal["BSL", "SSL"]
Scope = Literal["internal", "external"]


@dataclass(frozen=True)
class LiquidityLevel:
    """A candidate buy-side or sell-side liquidity pool."""

    timestamp: pd.Timestamp
    price: float
    side: Side
    scope: Scope
    strength: float


class LiquidityEngine:
    """Generate structural liquidity candidates from OHLC data.

    ``external_lookback`` controls how far back a confirmed swing can be
    before it is treated as an external structural reference. This is a
    first-pass classification and should be validated empirically.
    """

    def __init__(
        self,
        left_bars: int = 2,
        right_bars: int = 2,
        external_lookback: int = 50,
        equal_tolerance: float = 0.0005,
    ) -> None:
        if external_lookback <= 0:
            raise ValueError("external_lookback must be > 0")
        if equal_tolerance < 0:
            raise ValueError("equal_tolerance must be >= 0")
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.external_lookback = external_lookback
        self.equal_tolerance = equal_tolerance

    def levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return confirmed structural liquidity candidates.

        The returned rows use the pivot timestamp. ``confirmed_at`` is kept
        separately so downstream strategies can enforce causal timing.
        """
        swings = detect_swings(df, self.left_bars, self.right_bars)
        records: list[dict] = []

        for timestamp, row in swings.iterrows():
            if pd.notna(row["swing_high"]):
                records.append(
                    self._record(
                        timestamp,
                        float(row["swing_high"]),
                        "BSL",
                        row["swing_high_confirmed_at"],
                        len(df.loc[:timestamp]) - 1,
                    )
                )
            if pd.notna(row["swing_low"]):
                records.append(
                    self._record(
                        timestamp,
                        float(row["swing_low"]),
                        "SSL",
                        row["swing_low_confirmed_at"],
                        len(df.loc[:timestamp]) - 1,
                    )
                )

        columns = [
            "timestamp",
            "price",
            "side",
            "scope",
            "strength",
            "confirmed_at",
        ]
        return pd.DataFrame(records, columns=columns)

    def _record(
        self,
        timestamp: pd.Timestamp,
        price: float,
        side: Side,
        confirmed_at: pd.Timestamp,
        position: int,
    ) -> dict:
        scope: Scope = (
            "external" if position >= self.external_lookback else "internal"
        )
        # Initial neutral strength score. Later versions will incorporate
        # touches, equal-level clustering, timeframe, volume and age.
        strength = 1.0 if scope == "external" else 0.5
        return {
            "timestamp": timestamp,
            "price": price,
            "side": side,
            "scope": scope,
            "strength": strength,
            "confirmed_at": confirmed_at,
        }
