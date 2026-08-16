#!/usr/bin/env python3
"""Audit persistent-cluster event lifecycles without changing v2 scoring."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from zone_edge_historical_test import _load_klines, _build_profiles_from_bars
from zone_edge_walkforward_v2 import _build_snapshots, _find_events
from zone_edge_persistent_cluster_audit import _atr, _event_range, _match_distance

PERSISTENT_MAX_GAP_MINUTES = 120
NEW_INTERACTION_GAP_MINUTES = 5
RETEST_MAX_MINUTES = 120


def _load(days: int):
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT", page_limit=1000))
    bars = _load_klines(provider, start, end)
    profiles = _build_profiles_from_bars(bars)
    snapshots = _build_snapshots(profiles, bars)
    events, _, _ = _find_events(bars, snapshots)
    return bars, sorted(events, key=lambda e: e.timestamp)


def _assign_persistent_clusters(events, bars):
    """Reproduce the conservative persistent-ID matching used by the audit."""
    atr = _atr(bars)
    clusters = []
    next_id = 0
    assigned = defaultdict(list)

    for event in events:
        lo, hi = _event_range(event)
        center = float(event.center)
        atr_value = atr.loc[event.timestamp] if event.timestamp in atr.index else np.nan
        max_distance = _match_distance(atr_value)

        candidates = []
        for c in clusters:
            age = (event.timestamp - c["last_seen"]).total_seconds() / 60.0
            if age <= PERSISTENT_MAX_GAP_MINUTES and abs(c["center"] - center) <= max_distance:
                candidates.append((abs(c["center"] - center), c))

        if candidates:
            _, c = min(candidates, key=lambda x: x[0])
            c["center"] = (c["center"] + center) / 2.0
            c["low"] = min(c["low"], lo)
            c["high"] = max(c["high"], hi)
        else:
            c = {"id": next_id, "center": center, "low": lo, "high": hi, "last_seen": event.timestamp}
            clusters.append(c)
            next_id += 1
        c["last_seen"] = event.timestamp
        assigned[c["id"]].append(event)
    return clusters, assigned


def _audit_lifecycles(assigned):
    independent = 0
    suppressed = 0
    lifecycle_counts = Counter()
    lifecycle_examples = []
    suppression_reasons = Counter()

    for cid, events in assigned.items():
        events = sorted(events, key=lambda e: e.timestamp)
        lifecycle = 0
        state = "IDLE"
        last_event_ts = None
        last_breakout_ts = None
        seen_retest = False

        for event in events:
            typ = str(event.event).upper()
            ts = event.timestamp

            if state == "IDLE":
                # Any new interaction beginning after the prior lifecycle is independent.
                lifecycle += 1
                independent += 1
                state = "BROKEN" if typ == "BREAKOUT" else "IDLE"
                last_event_ts = ts
                last_breakout_ts = ts if typ == "BREAKOUT" else None
                seen_retest = False
                continue

            # BROKEN lifecycle: immediate retest belongs to the same interaction.
            if typ == "RETEST" and last_breakout_ts is not None:
                gap = (ts - last_breakout_ts).total_seconds() / 60.0
                if 0 <= gap <= RETEST_MAX_MINUTES and not seen_retest:
                    suppressed += 1
                    suppression_reasons["breakout_retest_same_lifecycle"] += 1
                    seen_retest = True
                    state = "IDLE"
                    last_event_ts = ts
                    continue

            # A later breakout after the lifecycle has completed is independent.
            if typ == "BREAKOUT":
                lifecycle += 1
                independent += 1
                state = "BROKEN"
                last_event_ts = ts
                last_breakout_ts = ts
                seen_retest = False
                continue

            suppressed += 1
            suppression_reasons["repeated_within_lifecycle"] += 1
            last_event_ts = ts

        lifecycle_counts[cid] = lifecycle
        if len(events) >= 5:
            lifecycle_examples.append((cid, len(events), lifecycle))

    return independent, suppressed, lifecycle_counts, suppression_reasons, lifecycle_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    bars, events = _load(args.days)
    clusters, assigned = _assign_persistent_clusters(events, bars)
    independent, suppressed, lifecycle_counts, reasons, examples = _audit_lifecycles(assigned)

    raw_counts = Counter(str(e.event).upper() for e in events)
    total = independent + suppressed
    ratio = independent / total * 100 if total else 0.0
    print("=== BTCUSDT PERSISTENT CLUSTER LIFECYCLE AUDIT ===")
    print(f"dataset={bars.index.min()} -> {bars.index.max()}")
    print(f"raw_events={len(events)}")
    print("raw_event_counts=" + " ".join(f"{k.lower()}={raw_counts[k]}" for k in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION")))
    print(f"persistent_clusters={len(clusters)}")
    print(f"independent_lifecycles={independent}")
    print(f"suppressed_within_lifecycle={suppressed}")
    print(f"retained_ratio={ratio:.2f}%")
    print(f"max_lifecycles_single_cluster={max(lifecycle_counts.values(), default=0)}")
    print("suppression_reasons=" + " ".join(f"{k}={v}" for k, v in reasons.most_common()))
    print("lifecycle_distribution=" + " ".join(f"{k}:{v}" for k, v in Counter(lifecycle_counts.values()).most_common(10)))
    print("sample_high_density_clusters:")
    for cid, raw, life in sorted(examples, key=lambda x: (-x[1], x[0]))[:10]:
        print(f"  P{cid} | raw_events={raw} lifecycles={life}")
    print("Interpretation: one breakout + its immediate retest is treated as one lifecycle; a later breakout after lifecycle completion is retained as a new interaction. Audit only; v2 edge scoring is unchanged.")


if __name__ == "__main__":
    main()
