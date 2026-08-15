"""Fusion of executed flow, price response and L2 observations."""

from __future__ import annotations

from dataclasses import dataclass

from .footprint import FootprintRow
from .l2_events import L2Event


@dataclass(frozen=True)
class FlowFusionEvent:
    timestamp: object
    price: float
    delta: float
    price_response: float
    l2_support: bool
    l2_opposition: bool
    classification: str


def fuse_flow_and_l2(
    row: FootprintRow,
    next_price: float,
    l2_events: list[L2Event],
    delta_threshold: float,
    response_threshold: float,
) -> FlowFusionEvent:
    """Produce an evidence-based flow event without claiming trader intent."""
    if delta_threshold < 0 or response_threshold < 0:
        raise ValueError("thresholds must be >= 0")

    response = next_price - row.price
    relevant = [e for e in l2_events if e.timestamp >= row.timestamp and e.price == row.price]
    replenished = any(e.event_type == "replenishment" for e in relevant)
    pulled = any(e.event_type == "pull" for e in relevant)

    strong_delta = abs(row.delta) >= delta_threshold
    weak_response = abs(response) <= response_threshold

    if strong_delta and weak_response and row.delta > 0 and replenished:
        classification = "buy_absorption_candidate"
    elif strong_delta and weak_response and row.delta < 0 and replenished:
        classification = "sell_absorption_candidate"
    elif strong_delta and row.delta > 0 and response > response_threshold and pulled:
        classification = "buy_liquidity_consumption"
    elif strong_delta and row.delta < 0 and response < -response_threshold and pulled:
        classification = "sell_liquidity_consumption"
    else:
        classification = "unconfirmed"

    return FlowFusionEvent(
        timestamp=row.timestamp,
        price=row.price,
        delta=row.delta,
        price_response=response,
        l2_support=replenished,
        l2_opposition=pulled,
        classification=classification,
    )
