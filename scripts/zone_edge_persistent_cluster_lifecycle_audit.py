#!/usr/bin/env python3
"""Audit whether persistent-cluster event suppressions are true duplicates.

This is an audit only. It does not modify v2 scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta

from zone_edge_persistent_cluster_audit import _load_bars, _build_profile_snapshots, _find_events

COOLDOWN_MINUTES = 5
NEW_INTERACTION_GAP_MINUTES = 5


def _ts(e):
    for name in ("timestamp", "event_timestamp", "time", "ts"):
        if hasattr(e, name):
            return getattr(e, name)
        if isinstance(e, dict) and name in e:
            return e[name]
    return None


def _etype(e):
    for name in ("event_type", "type"):
        if hasattr(e, name):
            return str(getattr(e, name)).upper()
        if isinstance(e, dict) and name in e:
            return str(e[name]).upper()
    return "UNKNOWN"


def _cluster(e):
    for name in ("persistent_cluster_id", "persistent_id", "cluster_id"):
        if hasattr(e, name):
            return getattr(e, name)
        if isinstance(e, dict) and name in e:
            return e[name]
    return None


def _zone(e):
    for name in ("zone_id", "zone_key", "key"):
        if hasattr(e, name):
            return getattr(e, name)
        if isinstance(e, dict) and name in e:
            return e[name]
    return None


def _audit(events):
    by_cluster = defaultdict(list)
    for e in events:
        c = _cluster(e)
        if c is not None:
            by_cluster[c].append(e)

    duplicate = 0
    independent = 0
    lifecycle_counts = Counter()
    examples = []

    for cid, evs in by_cluster.items():
        evs.sort(key=lambda e: _ts(e))
        lifecycle = 0
        last_ts = None
        last_type = None
        for e in evs:
            ts = _ts(e)
            typ = _etype(e)
            # A breakout followed by its immediate retest is one lifecycle.
            # A new event after a meaningful gap / completed retest starts a new lifecycle.
            if last_ts is None or ts - last_ts >= timedelta(minutes=NEW_INTERACTION_GAP_MINUTES):
                lifecycle += 1
                independent += 1
            elif typ == "RETEST" and last_type == "BREAKOUT":
                duplicate += 1
            else:
                duplicate += 1
            last_ts, last_type = ts, typ
        lifecycle_counts[cid] = lifecycle
        if len(evs) >= 5 and len(examples) < 10:
            examples.append((cid, len(evs), lifecycle))

    return independent, duplicate, lifecycle_counts, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    bars = _load_bars(args.days)
    snapshots = _build_profile_snapshots(bars)
    events, _, _ = _find_events(bars, snapshots)

    ts_col = "timestamp" if "timestamp" in bars.columns else bars.columns[0]
    print("=== BTCUSDT PERSISTENT CLUSTER LIFECYCLE AUDIT ===")
    print(f"dataset={bars.iloc[0][ts_col]} -> {bars.iloc[-1][ts_col]}")
    print(f"raw_events={len(events)}")

    independent, suppressed, lifecycle_counts, examples = _audit(events)
    total = independent + suppressed
    ratio = independent / total * 100 if total else 0.0
    print(f"independent_lifecycles={independent}")
    print(f"suppressed_within_lifecycle={suppressed}")
    print(f"retained_ratio={ratio:.2f}%")
    print(f"persistent_clusters_with_events={len(lifecycle_counts)}")
    print(f"max_lifecycles_single_cluster={max(lifecycle_counts.values(), default=0)}")
    print("lifecycle_distribution=" + " ".join(f"{k}:{v}" for k, v in Counter(lifecycle_counts.values()).most_common(10)))
    print("sample_high_density_clusters:")
    for cid, raw, life in sorted(examples, key=lambda x: (-x[1], x[0])):
        print(f"  {cid} | raw_events={raw} lifecycles={life}")
    print("Interpretation: events inside one completed interaction are suppressed; a new interaction after the configured gap is retained. Audit only; v2 edge scoring is unchanged.")


if __name__ == "__main__":
    main()
