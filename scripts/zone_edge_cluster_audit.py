"""Cluster-level independence audit for zone-edge walk-forward events.

Research/diagnostic only. This deliberately does not change the existing v2
scoring logic. It collapses simultaneous/nearby node events into one
structural interaction for an independence audit, while retaining the
underlying v2 events for comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta

import numpy as np

from zone_edge_walkforward_v2 import (
    _build_profiles_from_bars,
    _build_snapshots,
    _find_events,
    _load_klines,
    BinanceConfig,
    BinancePublicData,
)

MAX_INTERACTION_MINUTES = 15
INTERACTION_ATR_MULTIPLIER = 0.50


def _atr_series(bars, period=14):
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev = close.shift(1)
    tr = np.maximum.reduce([
        (high - low).to_numpy(),
        (high - prev).abs().to_numpy(),
        (low - prev).abs().to_numpy(),
    ])
    return bars.index.to_series().map(
        dict(zip(bars.index, np.asarray(tr, dtype=float)))
    ).rolling(period, min_periods=period).mean()


def _independent_events(events, bars):
    """Keep one representative event per local structural interaction.

    Events are considered the same interaction when they occur within the
    same short time window and their zone centers are within a fraction of
    the contemporaneous ATR. This is an audit-level independence filter;
    the original v2 events remain untouched.
    """
    if not events:
        return [], []

    atr = _atr_series(bars)
    kept = []
    suppressed = []

    for event in sorted(events, key=lambda e: e.timestamp):
        atr_value = atr.asof(event.timestamp)
        price_radius = float(atr_value) * INTERACTION_ATR_MULTIPLIER if np.isfinite(atr_value) else 0.0
        duplicate = None
        for prior in reversed(kept):
            if event.timestamp - prior.timestamp > timedelta(minutes=MAX_INTERACTION_MINUTES):
                break
            if abs(event.center - prior.center) <= max(price_radius, 5.0):
                duplicate = prior
                break
        if duplicate is None:
            kept.append(event)
        else:
            suppressed.append((event, duplicate))

    return kept, suppressed


def main():
    parser = argparse.ArgumentParser(description="Audit cluster-level independence without changing v2 scoring")
    parser.add_argument("--days", type=int, default=7, choices=(7, 30, 90))
    args = parser.parse_args()

    end = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT", page_limit=1000))
    bars = _load_klines(provider, start, end)
    profiles = _build_profiles_from_bars(bars)
    snapshots = _build_snapshots(profiles, bars)
    events, acceptances, audit = _find_events(bars, snapshots)
    independent, suppressed = _independent_events(events, bars)

    print("=== BTCUSDT CLUSTER INDEPENDENCE AUDIT ===")
    print(f"dataset={bars.index.min()} -> {bars.index.max()}")
    print(f"raw_events={len(events)}")
    print(f"raw_breakouts={sum(e.event == 'BREAKOUT' for e in events)} raw_retests={sum(e.event == 'RETEST' for e in events)}")
    print(f"independent_events={len(independent)}")
    print(f"suppressed_as_same_interaction={len(suppressed)}")
    print(f"retained_ratio={len(independent) / len(events):.2%}" if events else "retained_ratio=0.00%")
    print(f"raw_unique_zones={len(audit['event_zone'])}")
    print(f"raw_unique_clusters={len(audit['event_cluster'])}")
    print(f"max_raw_events_single_cluster={max(audit['event_cluster'].values(), default=0)}")

    counts = Counter(e.event for e in independent)
    print("independent_event_counts=" + " ".join(
        f"{kind.lower()}={counts.get(kind, 0)}"
        for kind in ("BREAKOUT", "RETEST", "REJECTION", "SWEEP")
    ))

    print("top_raw_clusters:")
    for key, count in audit["event_cluster"].most_common(10):
        print(f"  {key} | raw_events={count}")

    print("\nInterpretation: this is an independence audit only. It does not replace v2 events or alter edge scoring.")
    print("The next validation step is to inspect whether the retained interaction count and suppression ratio are structurally reasonable.")


if __name__ == "__main__":
    main()
