"""Causal classification of external and internal liquidity pools."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


LiquidityKind = Literal["external", "internal"]
LiquiditySide = Literal["buy_side", "sell_side"]


@dataclass(frozen=True)
class LiquidityPool:
    timestamp: datetime
    price: float
    side: LiquiditySide
    kind: LiquidityKind
    strength: float
    source: str


def classify_liquidity(
    *,
    timestamp: datetime,
    price: float,
    side: LiquiditySide,
    swing_price: float,
    range_high: float,
    range_low: float,
    source: str,
    strength: float = 1.0,
) -> LiquidityPool:
    """Classify a confirmed swing liquidity pool relative to a parent range.

    External liquidity is a confirmed swing beyond the current parent range
    boundary. Internal liquidity lies inside that range. The caller is
    responsible for supplying causally confirmed swing/range information.
    """
    if range_high < range_low:
        raise ValueError("range_high must be >= range_low")
    if strength < 0:
        raise ValueError("strength must be >= 0")

    if side == "buy_side":
        kind = "external" if swing_price >= range_high else "internal"
    elif side == "sell_side":
        kind = "external" if swing_price <= range_low else "internal"
    else:
        raise ValueError("unsupported liquidity side")

    return LiquidityPool(
        timestamp=timestamp,
        price=price,
        side=side,
        kind=kind,
        strength=strength,
        source=source,
    )
