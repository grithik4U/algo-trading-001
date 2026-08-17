#!/usr/bin/env python3
"""Temporal stability audit for canonical independent zone-edge events.

Uses one 30-day dataset and evaluates the canonical independent lifecycle
subset inside four non-overlapping 7-day windows. This is a robustness/audit
step only: it does not optimize thresholds, change V2 scoring, or generate
trading signals.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta

from zone_edge_persistent_cluster_lifecycle_audit import (
    _assign_persistent_clusters,
    _independent_events,
    _load,
)
from zone_edge_historical_test import HORIZONS, _baseline, _future_metrics

WINDOW_DAYS = 7
WINDOWS = 4
EVENT_TYPES = ("BREAKOUT", "RETEST")


def _score_window(bars, events, start, end):
    subset = [e for e in events if start <= e.timestamp < end]
    result = {kind: {} for kind in EVENT_TYPES}
    for horizon in HORIZONS:
        baseline = _baseline(bars, subset, horizon)
        for kind in EVENT_TYPES:
            rows = []
            for e in subset:
                if e.event != kind or e.direction not in {"UP", "DOWN"}:
                    continue
                metrics = _future_metrics(bars, e, horizon)
                if metrics is None or baseline.get(e.direction) is None:
                    continue
                move, mfe, mae = metrics
                rows.append((move, baseline[e.direction], mfe, mae))
            result[kind][horizon] = rows
    return subset, result


def _fmt(rows):
    if not rows:
        return "n=0 edge=NA fav=NA adv=NA"
    edge = sum(r[0] - r[1] for r in rows) / len(rows)
    fav = sum(r[2] > r[3] for r in rows) / len(rows) * 100.0
    adv = sum(r[3] > r[2] for r in rows) / len(rows) * 100.0
    return f"n={len(rows)} edge={edge:+.2f} fav={fav:5.1f}% adv={adv:5.1f}%"


def main():
    p = argparse.ArgumentParser(description="30-day temporal stability audit")
    p.add_argument("--days", type=int, default=30, choices=(30,))
    p.add_argument("--gap", type=int, default=5)
    args = p.parse_args()

    bars, raw = _load(args.days)
    _, assigned = _assign_persistent_clusters(raw, bars)
    events = _independent_events(assigned, args.gap)

    end = bars.index.max() + timedelta(minutes=1)
    print("=== BTCUSDT TEMPORAL STABILITY AUDIT ===")
    print(f"dataset={bars.index.min()} -> {bars.index.max()}")
    print(f"raw_events={len(raw)} independent_events={len(events)} retained_ratio={100*len(events)/len(raw):.2f}%" if raw else "raw_events=0 independent_events=0 retained_ratio=0.00%")
    counts = Counter(e.event for e in events)
    print("independent_event_counts=" + " ".join(f"{k.lower()}={counts.get(k, 0)}" for k in EVENT_TYPES))
    print(f"windows={WINDOWS} x {WINDOW_DAYS}d, interaction_gap={args.gap}m")
    print("\nwindow | event | n | 5m edge | 15m edge | 30m edge | 60m edge")
    print("-------|-------|---|----------|-----------|-----------|----------")

    all_edges = {kind: {h: [] for h in HORIZONS} for kind in EVENT_TYPES}
    valid_windows = 0
    for idx in range(WINDOWS - 1, -1, -1):
        start = end - timedelta(days=(idx + 1) * WINDOW_DAYS)
        stop = end - timedelta(days=idx * WINDOW_DAYS)
        subset, scored = _score_window(bars, events, start, stop)
        if subset:
            valid_windows += 1
        label = f"W{WINDOWS-idx}"
        for kind in EVENT_TYPES:
            parts = []
            n_for_label = 0
            for horizon in HORIZONS:
                rows = scored[kind][horizon]
                n_for_label = max(n_for_label, len(rows))
                if rows:
                    edge = sum(r[0] - r[1] for r in rows) / len(rows)
                    all_edges[kind][horizon].append(edge)
                    parts.append(f"{edge:+.2f}")
                else:
                    parts.append("NA")
            print(f"{label} | {kind:<7} | {n_for_label:3d} | {parts[0]:>8} | {parts[1]:>9} | {parts[2]:>9} | {parts[3]:>8}")

    print("\n=== STABILITY SUMMARY ===")
    print("event | horizon | windows_with_data | mean_edge | min_edge | max_edge | positive_windows")
    print("------|---------|-------------------|-----------|----------|----------|----------------")
    for kind in EVENT_TYPES:
        for horizon in HORIZONS:
            vals = all_edges[kind][horizon]
            if not vals:
                print(f"{kind:<6} | {horizon:7} | 0                 | NA        | NA       | NA       | 0")
                continue
            positive = sum(v > 0 for v in vals)
            print(
                f"{kind:<6} | {horizon:7} | {len(vals):17d} | "
                f"{sum(vals)/len(vals):+9.2f} | {min(vals):+8.2f} | {max(vals):+8.2f} | {positive:16d}"
            )

    print("\nInterpretation: this is a temporal robustness audit. A positive mean is not a pass by itself; the purpose is to detect whether an observed edge is concentrated in one week. No threshold optimization, no V2 scoring change, no trading signal.")


if __name__ == "__main__":
    main()
