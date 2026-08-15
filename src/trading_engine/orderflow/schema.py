"""Canonical market-data contracts for professional order-flow research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


TradeSide = Literal["buy", "sell", "unknown"]


@dataclass(frozen=True)
class TradeTick:
    """One executed trade classified by aggressor side."""

    timestamp: datetime
    price: float
    size: float
    side: TradeSide
    bid: float | None = None
    ask: float | None = None


@dataclass(frozen=True)
class BookLevel:
    """One visible L2 price level at a point in time."""

    timestamp: datetime
    price: float
    size: float
    side: Literal["bid", "ask"]
    level: int


@dataclass(frozen=True)
class OrderFlowSnapshot:
    """Normalized order-flow snapshot used by downstream signal modules."""

    timestamp: datetime
    buy_volume: float
    sell_volume: float
    delta: float
    cumulative_delta: float
    bid_volume: float
    ask_volume: float
    imbalance_ratio: float | None
