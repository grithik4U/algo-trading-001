#!/usr/bin/env python3
"""Persistent structural-cluster independence audit.

Research-only audit. It does not alter v2 scoring or trading signals.

Important: persistent matching is deliberately conservative. We match event
cluster centers across time; we do NOT grow a persistent cluster by repeatedly
unioning overlapping event ranges. That avoids the transitive-merging failure
where several nearby snapshots collapse an entire price region into one
cluster.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from zone_edge_historical_test import _load_klines, _build_profiles_from_bars
from zone_edge_walkforward_v2 import _build_snapshots, _find_events

PERSISTENT_MATCH_ATR = 0.25
PERSISTENT_MATCH_MIN = 5.0
PERSISTENT_MAX_GAP_MINUTES = 120
RETEST_MIN_BARS = 3
RETEST_MAX_BARS = 120
ATR_PERIOD = 14


@dataclass
class PersistentCluster:
    cid: int
    center: float
    low: float
    high: float
    last_seen: pd.Timestamp
    lifecycle: str = "IDLE"
    breakout_seen: bool = False
    retest_seen: bool = False
    last_event_time: pd.Timestamp | None = None


def _event_range(event):
    lo = float(event.low)
    hi = float(event.high)
    return (lo, hi) if hi >= lo else (hi, lo)


def _atr(bars):
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()


def _match_distance(atr_value):
    if atr_value is not None and np.isfinite(atr_value):
        return max(PERSISTENT_MATCH_MIN, float(atr_value) * PERSISTENT_MATCH_ATR)
    return PERSISTENT_MATCH_MIN


def _dataset_range(bars):
    if bars.empty:
        return None, None
    return bars.index.min(), bars.index.max()


def audit(days: int):
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)

    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT", page_limit=1000))
    bars = _load_klines(provider, start, end)
    profiles = _build_profiles_from_bars(bars)
    snapshots = _build_snapshots(profiles, bars)
    events, _, _ = _find_events(bars, snapshots)
    events = sorted(events, key=lambda e: e.timestamp)
    atr = _atr(bars)

    clusters: list[PersistentCluster] = []
    assignments: dict[int, int] = {}
    retained = []
    suppressed = []
    next_id = 0

    for event in events:
        lo, hi = _event_range(event)
        center = float(event.center)
        atr_value = atr.loc[event.timestamp] if event.timestamp in atr.index else np.nan
        max_distance = _match_distance(atr_value)

        candidates = []
        for cluster in clusters:
            age_minutes = (event.timestamp - cluster.last_seen).total_seconds() / 60.0
            if age_minutes > PERSISTENT_MAX_GAP_MINUTES:
                continue
            distance = abs(cluster.center - center)
            if distance <= max_distance:
                candidates.append((distance, cluster))

        if candidates:
            _, cluster = min(candidates, key=lambda item: item[0])
            # Move the persistent center modestly toward the current center;
            # never expand the structural range transitively.
            cluster.center = (cluster.center + center) / 2.0
            cluster.low = min(cluster.low, lo)
            cluster.high = max(cluster.high, hi)
        else:
            cluster = PersistentCluster(next_id, center, lo, hi, event.timestamp)
            clusters.append(cluster)
            next_id += 1

        cluster.last_seen = event.timestamp
        assignments[id(event)] = cluster.cid
        kind = str(event.event).upper()

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

    raw_counts = Counter(str(e.event).upper() for e in events)
    retained_counts = Counter(str(e.event).upper() for e in retained)
    cluster_counts = Counter(assignments.values())
    dataset_start, dataset_end = _dataset_range(bars)

    print("=== BTCUSDT PERSISTENT CLUSTER INDEPENDENCE AUDIT ===")
    print(f"dataset={dataset_start} -> {dataset_end}")
    print(f"raw_events={len(events)}")
    print("raw_event_counts=" + " ".join(f"{k.lower()}={raw_counts[k]}" for k in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION")))
    print(f"persistent_clusters={len(clusters)}")
    print(f"independent_events={len(retained)}")
    print(f"suppressed_as_same_persistent_lifecycle={len(suppressed)}")
    print(f"retained_ratio={(len(retained) / len(events) * 100) if events else 0:.2f}%")
    print(f"max_raw_events_single_persistent_cluster={max(cluster_counts.values(), default=0)}")
    print("independent_event_counts=" + " ".join(f"{k.lower()}={retained_counts[k]}" for k in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION")))
    print("top_persistent_clusters:")
    for cid, count in cluster_counts.most_common(10):
        cluster = clusters[cid]
        print(f"  P{cid}:{cluster.low:.2f}-{cluster.high:.2f} | center={cluster.center:.2f} | raw_events={count}")
    print(
        "Interpretation: persistent IDs use conservative center/temporal matching; "
        "this is an independence audit only and does not alter v2 edge scoring."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    audit(args.days)


if __name__ == "__main__":
    main()
