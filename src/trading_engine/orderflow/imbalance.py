"""Footprint imbalance and absorption-oriented features."""

from __future__ import annotations

from dataclasses import dataclass

from .footprint import FootprintRow


@dataclass(frozen=True)
class ImbalanceEvent:
    timestamp: object
    price: float
    side: str
    ratio: float | None
    delta: float
    total_volume: float
    kind: str


def detect_stacked_imbalances(
    rows: list[FootprintRow],
    ratio_threshold: float = 3.0,
    minimum_total_volume: float = 0.0,
    min_stack: int = 3,
) -> list[ImbalanceEvent]:
    """Detect consecutive same-side footprint imbalances at adjacent prices."""
    if ratio_threshold <= 1:
        raise ValueError("ratio_threshold must be > 1")
    if min_stack < 1:
        raise ValueError("min_stack must be >= 1")

    candidates: list[ImbalanceEvent] = []
    for row in rows:
        if row.total_volume < minimum_total_volume:
            continue
        if row.imbalance_ratio is not None and row.imbalance_ratio >= ratio_threshold:
            candidates.append(ImbalanceEvent(row.timestamp, row.price, "buy", row.imbalance_ratio, row.delta, row.total_volume, "buy_imbalance"))
        elif row.imbalance_ratio is not None and row.imbalance_ratio <= 1.0 / ratio_threshold:
            candidates.append(ImbalanceEvent(row.timestamp, row.price, "sell", 1.0 / row.imbalance_ratio, row.delta, row.total_volume, "sell_imbalance"))

    events: list[ImbalanceEvent] = []
    run: list[ImbalanceEvent] = []
    for event in candidates:
        adjacent = (
            run
            and event.timestamp == run[-1].timestamp
            and event.side == run[-1].side
            and event.price > run[-1].price
        )
        if adjacent:
            run.append(event)
        else:
            if len(run) >= min_stack:
                events.extend(run)
            run = [event]
    if len(run) >= min_stack:
        events.extend(run)
    return events


def classify_absorption(
    row: FootprintRow,
    delta_threshold: float,
    price_response_threshold: float,
) -> str:
    """Classify a single footprint row when price-response information is supplied.

    ``price_response_threshold`` is the maximum normalized price response
    supplied by the caller. This function intentionally avoids inventing price
    movement from volume alone.
    """
    if delta_threshold < 0 or price_response_threshold < 0:
        raise ValueError("thresholds must be >= 0")
    if abs(row.delta) < delta_threshold:
        return "none"
    if price_response_threshold == 0:
        return "absorption_candidate"
    return "absorption_candidate"
