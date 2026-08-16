#!/usr/bin/env python3
"""Persistent structural-cluster independence audit.

Research-only audit. It does not alter v2 scoring or trading signals.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from zone_edge_historical_test import _load_klines, _build_profiles_from_bars
from zone_edge_walkforward_v2 import _build_snapshots, _find_events

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


def _event_range(event):
    lo = float(event.low)
    hi = float(event.high)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _compatible(cluster, lo, hi, atr):
    center = (lo + hi) / 2.0
    gap = max(atr * CLUSTER_GAP_ATR, center * CLUSTER_GAP_PCT)
    return lo <= cluster.high + gap and hi >= cluster.low - gap


def _event_kind(event):
    return str(event.event).upper()


def audit(days: int):
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)

    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT", page_limit=1000))
    bars = _load_klines(provider, start, end)
    profiles = _build_profiles_from_bars(bars)
    snapshots = _build_snapshots(profiles, bars)
    events, _, _ = _find_events(bars, snapshots)
    events = sorted(events, key=lambda e: e.timestamp)

    clusters: list[PersistentCluster] = []
    assignments: dict[int, int] = {}
    retained = []
    suppressed = []
    next_id = 0

    for event in events:
        lo, hi = _event_range(event)
        atr = float(getattr(event, "atr", 0.0) or 0.0)
        matches = [c for c in clusters if _compatible(c, lo, hi, atr)]
        if matches:
            cluster = min(
                matches,
                key=lambda c: abs(((c.low + c.high) / 2.0) - ((lo + hi) / 2.0)),
            )
            cluster.low = min(cluster.low, lo)
            cluster.high = max(cluster.high, hi)
        else:
            cluster = PersistentCluster(next_id, lo, hi, event.timestamp)
            clusters.append(cluster)
            next_id += 1

        cluster.last_seen = event.timestamp
        assignments[id(event)] = cluster.cid
        kind = _event_kind(event)

        if kind == "BREAKOUT":
            if cluster.lifecycle == "IDLE":
                cluster.lifecycle = "BROKEN"
                cluster.breakout_seen = True
                cluster.retest_seen = False
                cluster.last_event_time = event.timestamp
                retained.append(event)
            else:
                suppressed.append((event, cluster.cid, "same_breakout_lifecycle"))
        elif kind == "RETEST":
            if cluster.lifecycle == "BROKEN" and cluster.breakout_seen and not cluster.retest_seen:
                dt = event.timestamp - (cluster.last_event_time or event.timestamp)
                if timedelta(minutes=RETEST_MIN_BARS) <= dt <= timedelta(minutes=RETEST_MAX_BARS):
                    cluster.retest_seen = True
                    cluster.lifecycle = "IDLE"
                    cluster.last_event_time = event.timestamp
                    retained.append(event)
                else:
                    suppressed.append((event, cluster.cid, "outside_persistent_lifecycle_window"))
            else:
                suppressed.append((event, cluster.cid, "no_new_cluster_lifecycle"))
        else:
            suppressed.append((event, cluster.cid, "unsupported_event_type"))

    raw_counts = Counter(_event_kind(e) for e in events)
    retained_counts = Counter(_event_kind(e) for e in retained)
    cluster_counts = Counter(assignments.values())

    print("=== BTCUSDT PERSISTENT CLUSTER INDEPENDENCE AUDIT ===")
    print(f"dataset={bars[0].timestamp} -> {bars[-1].timestamp}")
    print(f"raw_events={len(events)}")
    print(
        "raw_event_counts="
        + " ".join(f"{k.lower()}={raw_counts[k]}" for k in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION"))
    )
    print(f"persistent_clusters={len(clusters)}")
    print(f"independent_events={len(retained)}")
    print(f"suppressed_as_same_persistent_lifecycle={len(suppressed)}")
    print(f"retained_ratio={(len(retained) / len(events) * 100) if events else 0:.2f}%")
    print(f"max_raw_events_single_persistent_cluster={max(cluster_counts.values(), default=0)}")
    print(
        "independent_event_counts="
        + " ".join(f"{k.lower()}={retained_counts[k]}" for k in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION"))
    )
    print("top_persistent_clusters:")
    for cid, count in cluster_counts.most_common(10):
        cluster = clusters[cid]
        print(f"  P{cid}:{cluster.low:.2f}-{cluster.high:.2f} | raw_events={count}")
    print(
        "Interpretation: persistent IDs follow structural price overlap across snapshots; "
        "this is an independence audit only and does not alter v2 edge scoring."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    audit(args.days)


if __name__ == "__main__":
    main()
