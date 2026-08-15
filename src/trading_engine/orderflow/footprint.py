"""Price-level footprint aggregation from classified executed trades."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .schema import TradeTick


@dataclass(frozen=True)
class FootprintRow:
    timestamp: datetime
    price: float
    buy_volume: float
    sell_volume: float
    delta: float
    total_volume: float
    imbalance_ratio: float | None
    dominant_side: Literal["buy", "sell", "neutral"]


def build_footprint(
    trades: list[TradeTick],
    price_tick: float,
) -> list[FootprintRow]:
    """Aggregate executed volume by timestamp bucket and price tick.

    Trades must already have a reliable aggressor-side classification. This
    function does not guess trade direction from candle data.
    """
    if not trades:
        return []
    if price_tick <= 0:
        raise ValueError("price_tick must be > 0")

    buckets: dict[tuple[datetime, float], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for trade in trades:
        price = round(round(trade.price / price_tick) * price_tick, 10)
        key = (trade.timestamp, price)
        if trade.side == "buy":
            buckets[key][0] += trade.size
        elif trade.side == "sell":
            buckets[key][1] += trade.size
        else:
            raise ValueError("footprint requires buy/sell classified trades")

    rows: list[FootprintRow] = []
    for (timestamp, price), (buy, sell) in sorted(buckets.items()):
        total = buy + sell
        if buy > sell:
            dominant = "buy"
        elif sell > buy:
            dominant = "sell"
        else:
            dominant = "neutral"
        rows.append(
            FootprintRow(
                timestamp=timestamp,
                price=price,
                buy_volume=buy,
                sell_volume=sell,
                delta=buy - sell,
                total_volume=total,
                imbalance_ratio=(buy / sell if sell > 0 else None),
                dominant_side=dominant,
            )
        )
    return rows
