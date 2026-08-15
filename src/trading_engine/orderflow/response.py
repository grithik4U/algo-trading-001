"""Price-response tests for absorption and exhaustion candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .footprint import FootprintRow


@dataclass(frozen=True)
class FlowResponseEvent:
    timestamp: datetime
    price: float
    side: str
    delta: float
    total_volume: float
    price_response: float
    kind: str


def classify_flow_response(
    row: FootprintRow,
    next_price: float,
    delta_threshold: float,
    response_threshold: float,
) -> FlowResponseEvent:
    """Classify extreme executed flow relative to the following price response.

    ``next_price`` must be the first subsequent closed price selected by the
    caller. A strong delta with little adverse price response is an
    absorption candidate; strong delta with strong movement in the same
    direction is continuation/execution confirmation.
    """
    if delta_threshold < 0 or response_threshold < 0:
        raise ValueError("thresholds must be >= 0")

    response = next_price - row.price
    if abs(row.delta) < delta_threshold:
        kind = "none"
    elif row.delta > 0 and response <= response_threshold:
        kind = "buy_absorption_candidate"
    elif row.delta < 0 and response >= -response_threshold:
        kind = "sell_absorption_candidate"
    elif row.delta > 0 and response > response_threshold:
        kind = "buy_continuation"
    elif row.delta < 0 and response < -response_threshold:
        kind = "sell_continuation"
    else:
        kind = "none"

    return FlowResponseEvent(
        timestamp=row.timestamp,
        price=row.price,
        side="buy" if row.delta > 0 else "sell" if row.delta < 0 else "neutral",
        delta=row.delta,
        total_volume=row.total_volume,
        price_response=response,
        kind=kind,
    )
