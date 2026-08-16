"""Walk-forward zone-edge validation v2.

Research/diagnostic only. Builds on the existing walk-forward validator but
makes two independence rules explicit:

1. RETEST requires confirmed breakout, minimum bars of separation, and
   measurable displacement away from the zone before returning.
2. Overlapping/nearby HVN/LVN zones are treated as one structural interaction
   cluster for event counting. Original HVN/LVN labels are retained.

No future profile window is used for a historical event.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from zone_edge_historical_test import (
    ACCEPTANCE_BARS,
    BREAKOUT_BARS,
    HORIZONS,
    LIFECYCLE_COOLDOWN_BARS,
    MIN_EDGE_SAMPLE,
    _acceptance_resolutions,
    _baseline,
    _build_long_history_zones,
    _build_profiles_from_bars,
    _future_metrics,
    _load_klines,
    _outcome,
)

ZONE_UPDATE_MINUTES = 60
MIN_PROFILE_HISTORY = 8
ZONE_MATCH_TOLERANCE = 5.0
AUDIT_SAMPLE_SIZE = 12
RETEST_MIN_BARS = 3
RETEST_MIN_ATR = 0.50
CLUSTER_MAX_ATR = 0.25
ATR_PERIOD = 14


@dataclass(frozen=True)
class Event:
    timestamp: pd.Timestamp
    node_type: str
    low: float
    high: float
    center: float
    status: str
    event: str
    direction: str | None
    entry: float
    zone_key: str
    cluster_key: str


@dataclass(frozen=True)
class Acceptance:
    timestamp: pd.Timestamp
    node_type: str
    low: float
    high: float
    center: float
    status: str
    entry: float
    zone_key: str
    cluster_key: str


@dataclass
class ZoneLifecycle:
    node_type: str
    low: float
    high: float
    center: float
    status: str
    zone_key: str
    cluster_key: str
    state: str = "IDLE"
    inside_streak: int = 0
    outside_up: int = 0
    outside_down: int = 0
    bars_since_touch: int | None = None
    rejection_direction: str | None = None
    broken_direction: str | None = None
    cooldown: int = 0
    breakout_price: float | None = None
    breakout_timestamp: pd.Timestamp | None = None
    max_displacement: float = 0.0
    last_seen_snapshot: int = -1


def _touches(low, high, bar_low, bar_high):
    return bar_high >= low and bar_low <= high


def _inside(low, high, price):
    return low <= price <= high


def _zone_key(z):
    return f"{z['node_type']}:{round(float(z['center']) / ZONE_MATCH_TOLERANCE) * ZONE_MATCH_TOLERANCE:.2f}"


def _atr(bars):
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()


def _cluster_zones(zones, atr_value):
    """Cluster overlapping/nearby zones so one price interaction is one cluster."""
    if not zones:
        return []
    gap = max(float(atr_value) * CLUSTER_MAX_ATR, 1e-9) if np.isfinite(atr_value) else 0.0
    ordered = sorted(zones, key=lambda z: (float(z["low"]), float(z["high"])))
    clusters = []
    current = [ordered[0]]
    current_high = float(ordered[0]["high"])
    for z in ordered[1:]:
        low = float(z["low"])
        if low <= current_high + gap:
            current.append(z)
            current_high = max(current_high, float(z["high"]))
        else:
            clusters.append(current)
            current = [z]
            current_high = float(z["high"])
    clusters.append(current)

    out = []
    for idx, group in enumerate(clusters):
        cluster_key = f"C{idx}:{float(min(z['low'] for z in group)):.2f}-{float(max(z['high'] for z in group)):.2f}"
        for z in group:
            item = dict(z)
            item["cluster_key"] = cluster_key
            out.append(item)
    return out


def _match_states(previous, zones, snapshot_id):
    used = set()
    result = []
    for z in zones:
        best = None
        best_score = float("inf")
        for i, state in enumerate(previous):
            if i in used or state.zone_key != _zone_key(z):
                continue
            center_distance = abs(state.center - float(z["center"]))
            overlap = max(0.0, min(state.high, float(z["high"])) - max(state.low, float(z["low"])))
            if center_distance <= ZONE_MATCH_TOLERANCE or overlap > 0:
                score = center_distance - overlap
                if score < best_score:
                    best_score = score
                    best = i
        if best is not None:
            used.add(best)
            state = previous[best]
            state.low = float(z["low"])
            state.high = float(z["high"])
            state.center = float(z["center"])
            state.status = z["status"]
            state.cluster_key = z["cluster_key"]
            state.last_seen_snapshot = snapshot_id
            result.append(state)
        else:
            result.append(ZoneLifecycle(
                node_type=z["node_type"], low=float(z["low"]), high=float(z["high"]),
                center=float(z["center"]), status=z["status"], zone_key=_zone_key(z),
                cluster_key=z["cluster_key"], last_seen_snapshot=snapshot_id,
            ))
    return result


def _build_snapshots(profiles, bars):
    snapshots = []
    if len(profiles) < MIN_PROFILE_HISTORY:
        return snapshots
    atr = _atr(bars)
    last_update = None
    for i in range(MIN_PROFILE_HISTORY - 1, len(profiles)):
        profile_end = profiles[i][1]
        if last_update is not None and profile_end - last_update < timedelta(minutes=ZONE_UPDATE_MINUTES):
            continue
        prefix = profiles[: i + 1]
        zones = _build_long_history_zones(prefix, "HVN") + _build_long_history_zones(prefix, "LVN")
        atr_value = atr.loc[:profile_end].iloc[-1] if not atr.loc[:profile_end].empty else np.nan
        zones = _cluster_zones(zones, atr_value)
        snapshots.append((profile_end, zones, i + 1))
        last_update = profile_end
    return snapshots


def _snapshot_for_timestamp(snapshots, timestamp, pointer):
    while pointer + 1 < len(snapshots) and snapshots[pointer + 1][0] <= timestamp:
        pointer += 1
    if pointer < 0 or not snapshots or snapshots[pointer][0] > timestamp:
        return pointer, None
    return pointer, snapshots[pointer]


def _find_events(bars, snapshots):
    events, acceptances, states = [], [], []
    pointer = -1
    snapshot_id = -1
    audit = {"transitions": Counter(), "event_zone": Counter(), "event_cluster": Counter(), "samples": []}
    atr = _atr(bars)

    def transition(s, new, ts, reason):
        old = s.state
        s.state = new
        audit["transitions"][f"{old}->{new}"] += 1
        if len(audit["samples"]) < AUDIT_SAMPLE_SIZE:
            audit["samples"].append((ts, s.zone_key, s.cluster_key, old, new, reason, s.center))

    def emit(kind, s, ts, direction, close):
        events.append(Event(ts, s.node_type, s.low, s.high, s.center, s.status, kind, direction, close, s.zone_key, s.cluster_key))
        audit["event_zone"][s.zone_key] += 1
        audit["event_cluster"][s.cluster_key] += 1

    for ts, row in bars.iterrows():
        pointer, snapshot = _snapshot_for_timestamp(snapshots, ts, pointer)
        if snapshot is not None and snapshot[2] != snapshot_id:
            snapshot_id = snapshot[2]
            states = _match_states(states, snapshot[1], snapshot_id)
        if not states:
            continue
        low, high, close = float(row["low"]), float(row["high"]), float(row["close"])
        atr_now = atr.loc[ts]
        for s in states:
            if s.last_seen_snapshot != snapshot_id:
                continue
            if s.cooldown:
                s.cooldown -= 1
                continue
            touched = _touches(s.low, s.high, low, high)
            inside = _inside(s.low, s.high, close)
            above, below = close > s.high, close < s.low

            if s.state == "BROKEN":
                if s.breakout_price is None:
                    continue
                displacement = abs(close - s.breakout_price)
                s.max_displacement = max(s.max_displacement, displacement)
                bars_after = int((ts - s.breakout_timestamp).total_seconds() / 60) if s.breakout_timestamp is not None else 0
                atr_threshold = (float(atr_now) * RETEST_MIN_ATR) if np.isfinite(atr_now) else 0.0
                required = max(atr_threshold, abs(s.high - s.low))
                if bars_after >= RETEST_MIN_BARS and s.max_displacement >= required and touched:
                    emit("RETEST", s, ts, s.broken_direction, close)
                    transition(s, "COOLDOWN", ts, "confirmed_retest")
                    s.cooldown = LIFECYCLE_COOLDOWN_BARS
                    s.broken_direction = None
                continue

            if s.state == "COOLDOWN":
                continue

            if s.state == "IDLE":
                if touched:
                    s.bars_since_touch = 0
                    s.inside_streak = 1 if inside else 0
                    transition(s, "TOUCHED", ts, "touch")
                continue

            if touched:
                s.bars_since_touch = 0
                if inside:
                    s.outside_up = s.outside_down = 0
                    s.rejection_direction = None
                    s.inside_streak += 1
                    if s.state == "TOUCHED" and s.inside_streak >= ACCEPTANCE_BARS:
                        acceptances.append(Acceptance(ts, s.node_type, s.low, s.high, s.center, s.status, close, s.zone_key, s.cluster_key))
                        transition(s, "ACCEPTED", ts, "acceptance_confirmed")
                continue

            if s.bars_since_touch is not None:
                s.bars_since_touch += 1

            if s.state in {"TOUCHED", "ACCEPTED"}:
                if above:
                    s.outside_up += 1; s.outside_down = 0; s.rejection_direction = "UP"
                elif below:
                    s.outside_down += 1; s.outside_up = 0; s.rejection_direction = "DOWN"
                else:
                    s.outside_up = s.outside_down = 0

                if s.outside_up >= BREAKOUT_BARS or s.outside_down >= BREAKOUT_BARS:
                    direction = "UP" if s.outside_up >= BREAKOUT_BARS else "DOWN"
                    emit("BREAKOUT", s, ts, direction, close)
                    transition(s, "BROKEN", ts, f"breakout_{direction.lower()}")
                    s.broken_direction = direction
                    s.breakout_price = close
                    s.breakout_timestamp = ts
                    s.max_displacement = 0.0
                    s.inside_streak = 0
                    s.bars_since_touch = None
                    s.rejection_direction = None
                    s.outside_up = s.outside_down = 0
                    continue

                if s.bars_since_touch is not None and s.bars_since_touch > MAX_REJECTION_BARS:
                    # This is a failed interaction, retained as a rejection event
                    # only after the breakout window expires without confirmation.
                    emit("REJECTION", s, ts, s.rejection_direction, close)
                    transition(s, "COOLDOWN", ts, "rejection_window_expired")
                    s.cooldown = LIFECYCLE_COOLDOWN_BARS
                    s.bars_since_touch = None
                    s.rejection_direction = None
                    s.outside_up = s.outside_down = 0

    return sorted(events, key=lambda e: e.timestamp), sorted(acceptances, key=lambda a: a.timestamp), audit


def main():
    p = argparse.ArgumentParser(description="Strict walk-forward zone-edge validation v2")
    p.add_argument("--days", type=int, default=7, choices=(7, 30, 90))
    p.add_argument("--audit", action="store_true")
    args = p.parse_args()
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT", page_limit=1000))
    bars = _load_klines(provider, start, end)
    if bars.empty:
        raise RuntimeError("No Binance kline data returned")
    profiles = _build_profiles_from_bars(bars)
    snapshots = _build_snapshots(profiles, bars)
    events, acceptances, audit = _find_events(bars, snapshots)
    print("=== BTCUSDT WALK-FORWARD V2 ===")
    print("look_ahead_policy=profile_window_end_<=_event_timestamp")
    print(f"dataset={bars.index.min()} -> {bars.index.max()}")
    print(f"bars={len(bars)} profiles={len(profiles)} snapshots={len(snapshots)}")
    print(f"events={len(events)} acceptances={len(acceptances)}")
    counts = Counter(e.event for e in events)
    clusters = len(audit["event_cluster"])
    print(f"event_counts=" + " ".join(f"{k.lower()}={counts.get(k, 0)}" for k in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION")))
    print(f"unique_event_zones={len(audit['event_zone'])} unique_event_clusters={clusters}")
    print(f"max_events_single_zone={max(audit['event_zone'].values(), default=0)}")
    print(f"max_events_single_cluster={max(audit['event_cluster'].values(), default=0)}")
    if args.audit:
        print("state_transitions=" + " ".join(f"{k}={v}" for k, v in sorted(audit["transitions"].items())))
        print("sample_transitions:")
        for x in audit["samples"]:
            ts, key, cluster, old, new, reason, center = x
            print(f"  {ts} | {key} | {old}->{new} | reason={reason} | cluster={cluster} | center={center:.2f}")
        print("top_clusters:")
        for key, count in audit["event_cluster"].most_common(10):
            print(f"  {key} | events={count}")

    acceptance_results = _acceptance_resolutions(bars, acceptances)
    for horizon in HORIZONS:
        baseline = _baseline(bars, events, horizon)
        rows = []
        for e in events:
            metrics = _future_metrics(bars, e, horizon)
            if metrics is None or baseline.get(e.direction) is None:
                continue
            move, mfe, mae = metrics
            rows.append((e, move, baseline[e.direction], mfe, mae, _outcome(mfe, mae)))
        print(f"\n=== {horizon}M V2 EDGE ===")
        print("event | n | favorable | adverse | avg dir | avg baseline | edge")
        print("------|---|-----------|---------|----------|---------------|-----")
        for kind in ("BREAKOUT", "RETEST", "REJECTION", "SWEEP"):
            subset = [r for r in rows if r[0].event == kind]
            if not subset:
                continue
            avg_dir = np.mean([r[1] for r in subset]); avg_base = np.mean([r[2] for r in subset])
            fav = np.mean([r[5] == "FAVORABLE" for r in subset]) * 100
            adv = np.mean([r[5] == "ADVERSE" for r in subset]) * 100
            print(f"{kind:<9}| {len(subset):>4} | {fav:>9.1f}% | {adv:>7.1f}% | {avg_dir:>8.2f} | {avg_base:>13.2f} | {avg_dir-avg_base:+.2f}")
        for node_type in ("HVN", "LVN"):
            for status in ("HIGH_ACTIVE", "MEDIUM_ACTIVE", "DEVELOPING", "LOW", "HISTORICAL"):
                subset = [r for r in rows if r[0].node_type == node_type and r[0].status == status]
                if len(subset) < MIN_EDGE_SAMPLE:
                    continue
                avg_dir = np.mean([r[1] for r in subset]); avg_base = np.mean([r[2] for r in subset])
                fav = np.mean([r[5] == "FAVORABLE" for r in subset]) * 100
                adv = np.mean([r[5] == "ADVERSE" for r in subset]) * 100
                print(f"{node_type:<8}| {status:<13} n={len(subset):>4} | fav={fav:>5.1f}% adv={adv:>5.1f}% | avg={avg_dir:+.2f} base={avg_base:+.2f} edge={avg_dir-avg_base:+.2f}")
        print(f"baseline anchors: UP={baseline.get('UP')} DOWN={baseline.get('DOWN')}")

    print("\n=== ACCEPTANCE RESOLUTION ===")
    print(f"acceptance_results={len(acceptance_results)}")
    for horizon in HORIZONS:
        subset = [r for r in acceptance_results if r.horizon == horizon]
        if not subset: continue
        up = sum(r.resolution == "RESOLVES_UP" for r in subset)
        down = sum(r.resolution == "RESOLVES_DOWN" for r in subset)
        failed = sum(r.resolution in {"FAILED_UP", "FAILED_DOWN"} for r in subset)
        held = sum(r.resolution == "HELD" for r in subset)
        print(f"{horizon:>3}m | n={len(subset):>4} | up={100*up/len(subset):>5.1f}% down={100*down/len(subset):>5.1f}% failed={100*failed/len(subset):>5.1f}% held={100*held/len(subset):>5.1f}%")

    print("\nResearch rules: no look-ahead; breakout requires prior touch; retest requires >=3 bars + >=0.5 ATR or one zone-width displacement; nearby overlapping zones are clustered.")
    print("This is historical validation, not a trading signal.")


if __name__ == "__main__":
    main()
