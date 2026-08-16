#!/usr/bin/env python3
"""Persistent structural-cluster independence audit.

Research-only audit. It does not alter v2 scoring or trading signals.

Unlike the snapshot-local C0/C1 identifiers, this audit assigns a persistent
cluster identity by price-overlap/center proximity and tracks each structural
interaction through time. A cluster may produce at most one breakout and one
retest per lifecycle. The purpose is to measure whether node-level events are
independent observations before longer validation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from collections import Counter, defaultdict

from zone_edge_walkforward_v2 import _load_bars, _build_profile_snapshots, _find_events

# Conservative research constants; scoring is intentionally untouched.
CLUSTER_GAP_ATR = 0.75
CLUSTER_GAP_PCT = 0.0015
RETEST_MIN_BARS = 3
RETEST_MAX_BARS = 120

@dataclass
class PersistentCluster:
    cid: int
    low: float
    high: float
    last_seen: object
    lifecycle: str = "IDLE"
    breakout_seen: bool = False
    retest_seen: bool = False
    last_event_time: object = None


def _range(event):
    lo = float(getattr(event, "zone_low", getattr(event, "price", 0.0)))
    hi = float(getattr(event, "zone_high", getattr(event, "price", lo)))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _compatible(c, lo, hi, atr):
    gap = max(atr * CLUSTER_GAP_ATR, ((lo + hi) / 2.0) * CLUSTER_GAP_PCT)
    return lo <= c.high + gap and hi >= c.low - gap


def audit(days: int):
    bars = _load_bars(days)
    snapshots = _build_profile_snapshots(bars)
    events, _, _ = _find_events(bars, snapshots)
    events = sorted(events, key=lambda e: e.timestamp)

    clusters: list[PersistentCluster] = []
    assignments = {}
    retained = []
    suppressed = []
    next_id = 0

    for e in events:
        lo, hi = _range(e)
        atr = float(getattr(e, "atr", 0.0) or 0.0)
        matches = [c for c in clusters if _compatible(c, lo, hi, atr)]
        if matches:
            c = min(matches, key=lambda x: abs(((x.low+x.high)/2)-((lo+hi)/2)))
            c.low = min(c.low, lo)
            c.high = max(c.high, hi)
        else:
            c = PersistentCluster(next_id, lo, hi, e.timestamp)
            clusters.append(c)
            next_id += 1
        c.last_seen = e.timestamp
        assignments[id(e)] = c.cid

        kind = getattr(e, "kind", getattr(e, "event_type", "")).upper()
        # A new breakout starts a lifecycle only if this cluster is not already
        # in a live breakout/retest lifecycle.
        if kind == "BREAKOUT":
            if c.lifecycle == "IDLE":
                c.lifecycle = "BROKEN"
                c.breakout_seen = True
                c.retest_seen = False
                c.last_event_time = e.timestamp
                retained.append(e)
            else:
                suppressed.append((e, c.cid, "same_breakout_lifecycle"))
        elif kind == "RETEST":
            if c.lifecycle == "BROKEN" and c.breakout_seen and not c.retest_seen:
                dt = e.timestamp - (c.last_event_time or e.timestamp)
                if timedelta(minutes=RETEST_MIN_BARS) <= dt <= timedelta(minutes=RETEST_MAX_BARS):
                    c.retest_seen = True
                    c.lifecycle = "IDLE"
                    c.last_event_time = e.timestamp
                    retained.append(e)
                else:
                    suppressed.append((e, c.cid, "outside_persistent_lifecycle_window"))
            else:
                suppressed.append((e, c.cid, "no_new_cluster_lifecycle"))
        else:
            suppressed.append((e, c.cid, "unsupported_event_type"))

    raw_counts = Counter(getattr(e, "kind", getattr(e, "event_type", "")).upper() for e in events)
    ret_counts = Counter(getattr(e, "kind", getattr(e, "event_type", "")).upper() for e in retained)
    cluster_counts = Counter(assignments.values())

    print("=== BTCUSDT PERSISTENT CLUSTER INDEPENDENCE AUDIT ===")
    print(f"dataset={bars[0].timestamp} -> {bars[-1].timestamp}")
    print(f"raw_events={len(events)}")
    print("raw_event_counts=" + " ".join(f"{k.lower()}={raw_counts[k]}" for k in ("BREAKOUT","RETEST","SWEEP","REJECTION")))
    print(f"persistent_clusters={len(clusters)}")
    print(f"independent_events={len(retained)}")
    print(f"suppressed_as_same_persistent_lifecycle={len(suppressed)}")
    print(f"retained_ratio={(len(retained)/len(events)*100) if events else 0:.2f}%")
    print(f"max_raw_events_single_persistent_cluster={max(cluster_counts.values(), default=0)}")
    print("independent_event_counts=" + " ".join(f"{k.lower()}={ret_counts[k]}" for k in ("BREAKOUT","RETEST","SWEEP","REJECTION")))
    print("top_persistent_clusters:")
    for cid, n in cluster_counts.most_common(10):
        c = clusters[cid]
        print(f"  P{cid}:{c.low:.2f}-{c.high:.2f} | raw_events={n}")
    print("Interpretation: persistent IDs follow structural price overlap across snapshots; this is an independence audit only and does not alter v2 edge scoring.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7)
    args = p.parse_args()
    audit(args.days)


if __name__ == "__main__":
    main()
