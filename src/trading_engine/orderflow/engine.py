"""Order-flow aggregation from executed trades and L2 snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from .schema import BookLevel, OrderFlowSnapshot, TradeTick


def aggregate_trades(
    trades: Iterable[TradeTick],
    cumulative_delta_start: float = 0.0,
) -> OrderFlowSnapshot:
    """Aggregate classified executed trades into a true trade-flow snapshot."""
    trades = list(trades)
    if not trades:
        raise ValueError("trades cannot be empty")

    buy_volume = sum(t.size for t in trades if t.side == "buy")
    sell_volume = sum(t.size for t in trades if t.side == "sell")
    delta = buy_volume - sell_volume
    cumulative_delta = cumulative_delta_start + delta
    timestamp = max(t.timestamp for t in trades)

    return OrderFlowSnapshot(
        timestamp=timestamp,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        delta=delta,
        cumulative_delta=cumulative_delta,
        bid_volume=sell_volume,
        ask_volume=buy_volume,
        imbalance_ratio=(buy_volume / sell_volume if sell_volume > 0 else None),
    )


def aggregate_l2(
    levels: Iterable[BookLevel],
    timestamp: datetime,
    trade_flow: OrderFlowSnapshot,
) -> OrderFlowSnapshot:
    """Attach visible L2 bid/ask liquidity to an existing trade-flow snapshot."""
    levels = list(levels)
    if not levels:
        raise ValueError("levels cannot be empty")

    bid_volume = sum(level.size for level in levels if level.side == "bid")
    ask_volume = sum(level.size for level in levels if level.side == "ask")
    imbalance = ask_volume / bid_volume if bid_volume > 0 else None

    return OrderFlowSnapshot(
        timestamp=timestamp,
        buy_volume=trade_flow.buy_volume,
        sell_volume=trade_flow.sell_volume,
        delta=trade_flow.delta,
        cumulative_delta=trade_flow.cumulative_delta,
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        imbalance_ratio=imbalance,
    )
