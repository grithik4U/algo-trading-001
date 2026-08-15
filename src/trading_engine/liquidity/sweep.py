"""Causal detection of liquidity raids/sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .detector import LiquidityCandidate


@dataclass(frozen=True)
class LiquiditySweep:
    pool_timestamp: datetime
    sweep_timestamp: datetime
    pool_price: float
    sweep_extreme: float
    side: str
    penetration: float
    reclaimed: bool


def detect_sweeps(
    df: pd.DataFrame,
    pools: list[LiquidityCandidate],
    tolerance: float = 0.0,
) -> list[LiquiditySweep]:
    """Detect a first post-pool bar that trades through a pool and optionally reclaims it.

    Buy-side pools are swept by a high above the level; sell-side pools by a
    low below the level. Reclaim means the close returns through the pool in
    the opposite direction on that same bar. Only bars after pool timestamp
    are considered.
    """
    required = {"high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    bars = df.reset_index()
    timestamp_col = df.index.name or "index"
    bars = bars.rename(columns={timestamp_col: "timestamp"})
    events: list[LiquiditySweep] = []

    for pool in pools:
        future = bars.loc[bars["timestamp"] > pool.timestamp]
        for _, bar in future.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            level = float(pool.price)

            if pool.side == "buy_side" and high > level + tolerance:
                events.append(
                    LiquiditySweep(
                        pool.timestamp,
                        bar["timestamp"],
                        level,
                        high,
                        pool.side,
                        high - level,
                        close <= level,
                    )
                )
                break
            if pool.side == "sell_side" and low < level - tolerance:
                events.append(
                    LiquiditySweep(
                        pool.timestamp,
                        bar["timestamp"],
                        level,
                        low,
                        pool.side,
                        level - low,
                        close >= level,
                    )
                )
                break

    return events
