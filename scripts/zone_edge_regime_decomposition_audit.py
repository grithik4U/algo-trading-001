#!/usr/bin/env python3
"""Regime-conditioned independent-event audit.

Research only. Uses frozen event/lifecycle logic and does not alter V2 scoring.
Regimes are descriptive (dataset terciles) rather than optimized trading rules.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from zone_edge_persistent_cluster_lifecycle_audit import _load, _assign_persistent_clusters, _independent_events
from zone_edge_historical_test import HORIZONS, _baseline, _future_metrics

ATR_WINDOW = 60
VOL_WINDOW = 60
TREND_WINDOW = 60


def _atr(bars, window=14):
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev = close.shift(1)
    tr = np.maximum.reduce([
        (high - low).to_numpy(),
        (high - prev).abs().to_numpy(),
        (low - prev).abs().to_numpy(),
    ])
    return bars.index.to_series().index.to_series() if False else __import__("pandas").Series(tr, index=bars.index).rolling(window, min_periods=window).mean()


def _regime_series(bars):
    import pandas as pd
    close = bars["close"].astype(float)
    atr = _atr(bars)
    vol = bars["volume"].astype(float).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean()
    trend = (close - close.shift(TREND_WINDOW)).abs() / atr.replace(0, np.nan)

    def terciles(s):
        valid = s.dropna()
        q1, q2 = valid.quantile(1/3), valid.quantile(2/3)
        return q1, q2

    atr_q = terciles(atr)
    vol_q = terciles(vol)
    trend_q = terciles(trend)

    def label(value, qs, labels):
        if not np.isfinite(value):
            return None
        q1, q2 = qs
        return labels[0] if value <= q1 else labels[1] if value <= q2 else labels[2]

    rows = {}
    for ts in bars.index:
        rows[ts] = {
            "volatility": label(float(atr.loc[ts]), atr_q, ("LOW", "MEDIUM", "HIGH")),
            "volume": label(float(vol.loc[ts]), vol_q, ("LOW", "MEDIUM", "HIGH")),
            "trend_magnitude": label(float(trend.loc[ts]), trend_q, ("RANGE", "MID", "TREND")),
        }
    return rows, atr_q, vol_q, trend_q


def _edge_for_group(bars, all_events, group, horizon):
    baseline = _baseline(bars, all_events, horizon)
    rows = []
    for e in group:
        m = _future_metrics(bars, e, horizon)
        if m is None or e.direction not in {"UP", "DOWN"} or baseline.get(e.direction) is None:
            continue
        move, mfe, mae = m
        rows.append((move, baseline[e.direction], mfe, mae))
    if not rows:
        return None
    avg = float(np.mean([x[0] for x in rows]))
    base = float(np.mean([x[1] for x in rows]))
    fav = float(np.mean([x[0] > 0 for x in rows]) * 100)
    return len(rows), avg, base, avg - base, fav


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    args = p.parse_args()

    bars, raw = _load(args.days)
    _, assigned = _assign_persistent_clusters(raw, bars)
    events = [e for e in _independent_events(assigned, 5) if e.event in {"BREAKOUT", "RETEST"} and e.direction in {"UP", "DOWN"}]
    regimes, atr_q, vol_q, trend_q = _regime_series(bars)

    print("=== BTCUSDT REGIME DECOMPOSITION AUDIT ===")
    print(f"dataset={bars.index.min()} -> {bars.index.max()}")
    print(f"raw_events={len(raw)} independent_directional_events={len(events)}")
    print("regime_method=dataset_terciles; descriptive_only; frozen_event_logic")
    print(f"atr_terciles={atr_q[0]:.4f},{atr_q[1]:.4f}")
    print(f"volume_terciles={vol_q[0]:.4f},{vol_q[1]:.4f}")
    print(f"trend_terciles={trend_q[0]:.4f},{trend_q[1]:.4f}")

    dims = ("volatility", "volume", "trend_magnitude")
    for dim in dims:
        print(f"\n=== {dim.upper()} ===")
        labels = ("LOW", "MEDIUM", "HIGH") if dim != "trend_magnitude" else ("RANGE", "MID", "TREND")
        for label in labels:
            group = [e for e in events if regimes.get(e.timestamp, {}).get(dim) == label]
            print(f"\n{label} | directional_events={len(group)}")
            for event_type in ("BREAKOUT", "RETEST"):
                subset = [e for e in group if e.event == event_type]
                line = [f"{event_type} n={len(subset)}"]
                for h in HORIZONS:
                    r = _edge_for_group(bars, events, subset, h)
                    if r is None:
                        line.append(f"{h}m=n/a")
                    else:
                        n, avg, base, edge, fav = r
                        line.append(f"{h}m=edge={edge:+.2f},fav={fav:.1f}%,n={n}")
                print("  " + " | ".join(line))

    # Candidate structural subset × regime overlay: low-status retests.
    print("\n=== LOW-STATUS RETEST × REGIME ===")
    low_retests = [e for e in events if e.event == "RETEST" and e.status == "LOW"]
    for dim in dims:
        print(f"\n{dim}")
        labels = ("LOW", "MEDIUM", "HIGH") if dim != "trend_magnitude" else ("RANGE", "MID", "TREND")
        for label in labels:
            subset = [e for e in low_retests if regimes.get(e.timestamp, {}).get(dim) == label]
            line = [f"{label} n={len(subset)}"]
            for h in HORIZONS:
                r = _edge_for_group(bars, events, subset, h)
                line.append(f"{h}m={('edge=%+.2f,n=%d' % (r[3], r[0])) if r else 'n/a'}")
            print("  " + " | ".join(line))

    print("\nInterpretation: regime decomposition only. No thresholds are promoted and V2 scoring/lifecycle rules are unchanged.")


if __name__ == "__main__":
    main()
