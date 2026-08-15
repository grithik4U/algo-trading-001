"""L2 order-book event detection primitives.

These functions operate on timestamped snapshots. They deliberately identify
observable book changes rather than assigning intent to market participants.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .schema import BookLevel


@dataclass(frozen=True)
class L2Event:
    timestamp: datetime
    price: float
    side: Literal["bid", "ask"]
    previous_size: float
    current_size: float
    size_change: float
    event_type: Literal["add", "pull", "replenishment"]


def compare_snapshots(
    previous: list[BookLevel],
    current: list[BookLevel],
    replenishment_threshold: float = 0.0,
) -> list[L2Event]:
    """Compare two L2 snapshots at identical price/side keys."""
    if replenishment_threshold < 0:
        raise ValueError("replenishment_threshold must be >= 0")

    prev = {(x.side, x.price): x for x in previous}
    curr = {(x.side, x.price): x for x in current}
    events: list[L2Event] = []

    for key, level in curr.items():
        old = prev.get(key)
        old_size = old.size if old else 0.0
        change = level.size - old_size
        if old is None or change > 0:
            event_type = "replenishment" if old and old_size > 0 and change >= replenishment_threshold else "add"
        elif change < 0:
            event_type = "pull"
        else:
            continue
        events.append(
            L2Event(
                timestamp=level.timestamp,
                price=level.price,
                side=level.side,
                previous_size=old_size,
                current_size=level.size,
                size_change=change,
                event_type=event_type,
            )
        )
    return events
