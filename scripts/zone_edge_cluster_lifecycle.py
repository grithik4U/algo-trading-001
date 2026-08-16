"""Cluster-level lifecycle arbitration for zone-edge research.

Research/diagnostic only. This module converts node-level events into one
independent structural interaction per persistent price cluster/lifecycle.
It does not alter v2 scoring or future-outcome calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class ClusterEvent:
    timestamp: object
    event: str
    direction: str | None
    cluster_key: str
    representative_zone: str
    source_events: int


def arbitrate(events, *, reset_after_retest: bool = True):
    """Suppress overlapping node events that belong to one interaction.

    Events must be ordered by timestamp and expose ``timestamp``, ``event``,
    ``direction``, ``cluster_key`` and ``zone_key`` attributes.

    A cluster may emit one BREAKOUT and, after that, one RETEST. Additional
    node-level events in the same active lifecycle are suppressed. A new
    lifecycle is permitted only after the RETEST (or after a BREAKOUT when
    reset_after_retest is False).
    """
    active = {}
    emitted = []
    suppressed = []
    source_counts = defaultdict(int)

    for event in sorted(events, key=lambda e: e.timestamp):
        key = event.cluster_key
        state = active.get(key, "IDLE")
        source_counts[key] += 1

        if event.event == "BREAKOUT":
            if state in {"BROKEN", "RETESTED"}:
                suppressed.append(event)
                continue
            active[key] = "BROKEN"
            emitted.append(ClusterEvent(
                event.timestamp, "BREAKOUT", event.direction, key,
                event.zone_key, 1,
            ))
            continue

        if event.event == "RETEST":
            if state != "BROKEN":
                suppressed.append(event)
                continue
            active[key] = "RETESTED"
            emitted.append(ClusterEvent(
                event.timestamp, "RETEST", event.direction, key,
                event.zone_key, 1,
            ))
            continue

        # Preserve future event types, but never let them create a second
        # independent lifecycle while a cluster is active.
        if state in {"BROKEN", "RETESTED"}:
            suppressed.append(event)
        else:
            emitted.append(ClusterEvent(
                event.timestamp, event.event, event.direction, key,
                event.zone_key, 1,
            ))

    return emitted, suppressed, dict(source_counts)


def audit(events):
    """Return a compact independence report for node-level events."""
    retained, suppressed, counts = arbitrate(events)
    raw = len(events)
    return {
        "raw_events": raw,
        "independent_events": len(retained),
        "suppressed_as_same_interaction": len(suppressed),
        "retained_ratio": (len(retained) / raw if raw else 0.0),
        "max_raw_events_single_cluster": max(counts.values(), default=0),
    }
