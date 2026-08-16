"""Walk-forward BTCUSDT zone-edge validation with strict no-look-ahead.

Research/diagnostic only. This script is intentionally separate from
zone_edge_historical_test.py until the walk-forward results are validated.

Key guarantee:
    At event time t, structural zones are built only from profile windows whose
    window_end is <= t. No future profile/node can influence the zone used for
    the event.

Lifecycle rule:
    A zone cannot become BROKEN merely because a new snapshot is created while
    price is already outside it. Price must first interact with/touch the zone.
    A post-touch move away is held as a rejection candidate until it either
    resolves back into the zone (REJECTION) or establishes BREAKOUT.
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
    MAX_REJECTION_BARS,
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


@dataclass
class ZoneLifecycle:
    node_type: str
    low: float
    high: float
    center: float
    status: str
    zone_key: str
    state: str = "IDLE"
    inside_streak: int = 0
    outside_up: int = 0
    outside_down: int = 0
    bars_since_touch: int | None = None
    rejection_direction: str | None = None
    broken_direction: str | None = None
    cooldown: int = 0
    last_seen_snapshot: int = -1


def _touches(low, high, bar_low, bar_high):
    return bar_high >= low and bar_low <= high


def _inside(low, high, price):
    return low <= price <= high


def _direction(event, close, low, high):
    if event == "BREAKOUT":
        if close > high:
            return "UP"
        if close < low:
            return "DOWN"
    if event in {"SWEEP", "REJECTION"}:
        center = (low + high) / 2.0
        if close > center:
            return "UP"
        if close < center:
            return "DOWN"
    return None


def _zone_key(z):
    return f"{z['node_type']}:{round(float(z['center']) / ZONE_MATCH_TOLERANCE) * ZONE_MATCH_TOLERANCE:.2f}"


def _match_states(previous, zones, snapshot_id):
    """Carry lifecycle state forward when a zone persists between snapshots."""
    used = set()
    result = []
    for z in zones:
        best = None
        best_score = float("inf")
        for i, state in enumerate(previous):
            if i in used or state.node_type != z["node_type"]:
                continue
            center_distance = abs(state.center - z["center"])
            overlap = max(0.0, min(state.high, z["high"]) - max(state.low, z["low"]))
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
            state.zone_key = _zone_key(z)
            state.last_seen_snapshot = snapshot_id
            result.append(state)
        else:
            result.append(
                ZoneLifecycle(
                    node_type=z["node_type"],
                    low=float(z["low"]),
                    high=float(z["high"]),
                    center=float(z["center"]),
                    status=z["status"],
                    zone_key=_zone_key(z),
                    last_seen_snapshot=snapshot_id,
                )
            )
    return result


def _build_walk_forward_snapshots(profiles):
    """Build zone snapshots using only profiles available at each snapshot time."""
    snapshots = []
    if len(profiles) < MIN_PROFILE_HISTORY:
        return snapshots

    last_update = None
    for i in range(MIN_PROFILE_HISTORY - 1, len(profiles)):
        profile_end = profiles[i][1]
        if last_update is not None:
            elapsed = profile_end - last_update
            if elapsed < timedelta(minutes=ZONE_UPDATE_MINUTES):
                continue
        prefix = profiles[: i + 1]
        zones = _build_long_history_zones(prefix, "HVN") + _build_long_history_zones(prefix, "LVN")
        snapshots.append((profile_end, zones, i + 1))
        last_update = profile_end
    return snapshots


def _snapshot_for_timestamp(snapshots, timestamp, pointer):
    while pointer + 1 < len(snapshots) and snapshots[pointer + 1][0] <= timestamp:
        pointer += 1
    if pointer < 0 or not snapshots or snapshots[pointer][0] > timestamp:
        return pointer, None
    return pointer, snapshots[pointer]


def _find_walk_forward_events(bars, snapshots):
    events = []
    acceptances = []
    states = []
    snapshot_pointer = -1
    snapshot_id = -1
    audit = {
        "state_transitions": Counter(),
        "zone_snapshots": Counter(),
        "zone_first_seen": Counter(),
        "event_zone_counts": Counter(),
        "samples": [],
    }

    def transition(state, new_state, ts, reason):
        old = state.state
        state.state = new_state
        audit["state_transitions"][f"{old}->{new_state}"] += 1
        if len(audit["samples"]) < AUDIT_SAMPLE_SIZE and old != new_state:
            audit["samples"].append((ts, state.zone_key, old, new_state, reason, state.center))

    def emit(event, state, ts, direction, close):
        events.append(
            Event(
                ts,
                state.node_type,
                state.low,
                state.high,
                state.center,
                state.status,
                event,
                direction,
                close,
                state.zone_key,
            )
        )
        audit["event_zone_counts"][state.zone_key] += 1

    for ts, row in bars.iterrows():
        snapshot_pointer, snapshot = _snapshot_for_timestamp(snapshots, ts, snapshot_pointer)
        if snapshot is not None and snapshot[2] != snapshot_id:
            snapshot_id = snapshot[2]
            audit["zone_snapshots"][snapshot_id] += 1
            previous_keys = {s.zone_key for s in states}
            states = _match_states(states, snapshot[1], snapshot_id)
            for state in states:
                if state.zone_key not in previous_keys:
                    audit["zone_first_seen"][state.zone_key] += 1

        if not states:
            continue

        bar_low = float(row["low"])
        bar_high = float(row["high"])
        close = float(row["close"])

        for state in states:
            if state.last_seen_snapshot != snapshot_id:
                continue

            touched = _touches(state.low, state.high, bar_low, bar_high)
            inside = _inside(state.low, state.high, close)
            above = close > state.high
            below = close < state.low

            if state.cooldown:
                state.cooldown -= 1
                continue

            if state.state == "BROKEN":
                if touched:
                    emit("RETEST", state, ts, state.broken_direction, close)
                    transition(state, "COOLDOWN", ts, "retest")
                    state.cooldown = LIFECYCLE_COOLDOWN_BARS
                    state.broken_direction = None
                continue

            if state.state == "COOLDOWN":
                continue

            # A newly discovered zone cannot be declared broken just because
            # price is already outside it. It must first interact with the zone.
            if state.state == "IDLE":
                if touched:
                    state.bars_since_touch = 0
                    state.inside_streak = 1 if inside else 0
                    if inside and state.inside_streak >= ACCEPTANCE_BARS:
                        acceptances.append(Acceptance(ts, state.node_type, state.low, state.high, state.center, state.status, close, state.zone_key))
                        transition(state, "ACCEPTED", ts, "acceptance_confirmed")
                    else:
                        transition(state, "TOUCHED", ts, "touch")
                continue

            # Existing interaction lifecycle.
            if touched:
                state.bars_since_touch = 0
                if inside:
                    state.outside_up = 0
                    state.outside_down = 0
                    state.rejection_direction = None
                    state.inside_streak += 1
                    if state.state == "TOUCHED" and state.inside_streak >= ACCEPTANCE_BARS:
                        acceptances.append(Acceptance(ts, state.node_type, state.low, state.high, state.center, state.status, close, state.zone_key))
                        transition(state, "ACCEPTED", ts, "acceptance_confirmed")
                else:
                    state.inside_streak = 0
                    state.rejection_direction = "UP" if above else "DOWN" if below else None
                continue

            if state.bars_since_touch is not None:
                state.bars_since_touch += 1

            # Only a previously touched/accepted zone can establish breakout.
            if state.state in {"TOUCHED", "ACCEPTED"}:
                if above:
                    state.outside_up += 1
                    state.outside_down = 0
                    state.rejection_direction = "UP"
                elif below:
                    state.outside_down += 1
                    state.outside_up = 0
                    state.rejection_direction = "DOWN"
                else:
                    state.outside_up = 0
                    state.outside_down = 0

                if state.outside_up >= BREAKOUT_BARS or state.outside_down >= BREAKOUT_BARS:
                    direction = "UP" if state.outside_up >= BREAKOUT_BARS else "DOWN"
                    emit("BREAKOUT", state, ts, direction, close)
                    transition(state, "BROKEN", ts, f"breakout_{direction.lower()}")
                    state.broken_direction = direction
                    state.inside_streak = 0
                    state.bars_since_touch = None
                    state.rejection_direction = None
                    state.outside_up = 0
                    state.outside_down = 0
                    continue

                # If price leaves the zone but does not establish a breakout,
                # wait for a return inside the zone. That resolves as rejection.
                if state.rejection_direction and state.bars_since_touch <= MAX_REJECTION_BARS and state.outside_up + state.outside_down > 0:
                    # Keep the candidate pending until it either re-enters or
                    # establishes BREAKOUT. This avoids classifying a breakout
                    # as a rejection prematurely.
                    continue

                if state.bars_since_touch is not None and state.bars_since_touch > MAX_REJECTION_BARS:
                    emit("REJECTION", state, ts, state.rejection_direction, close)
                    transition(state, "COOLDOWN", ts, "rejection_window_expired")
                    state.cooldown = LIFECYCLE_COOLDOWN_BARS
                    state.bars_since_touch = None
                    state.rejection_direction = None
                    state.outside_up = 0
                    state.outside_down = 0
                    continue

            if state.state == "ACCEPTED" and (above or below):
                # The breakout branch above normally resolves this first. This
                # fallback simply closes acceptance if the zone definition moved.
                transition(state, "COOLDOWN", ts, "acceptance_resolved_outside")
                state.cooldown = LIFECYCLE_COOLDOWN_BARS
                state.bars_since_touch = None
                state.rejection_direction = None

    return sorted(events, key=lambda e: e.timestamp), sorted(acceptances, key=lambda a: a.timestamp), audit


def _print_summary(bars, profiles, snapshots, events, acceptances):
    print("=== BTCUSDT WALK-FORWARD ZONE EDGE TEST ===")
    print("walk_forward=True")
    print("look_ahead_policy=profile_window_end_must_be_<=_event_timestamp")
    print(f"dataset={bars.index.min()} -> {bars.index.max()}")
    print(f"bars={len(bars)}")
    print(f"profile_windows={len(profiles)}")
    print(f"zone_snapshots={len(snapshots)}")
    print(f"minimum_profile_history={MIN_PROFILE_HISTORY}")
    print(f"events={len(events)}")
    print(f"acceptances={len(acceptances)}")


def _print_audit(bars, snapshots, events, acceptances, audit):
    print("\n=== EVENT / LIFECYCLE AUDIT ===")
    counts = Counter(e.event for e in events)
    days = max((bars.index.max() - bars.index.min()).total_seconds() / 86400.0, 1 / 1440)
    print(f"events_per_day={len(events) / days:.2f}")
    print("event_counts=" + " ".join(f"{k.lower()}={counts.get(k, 0)}" for k in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION")))
    unique_event_zones = len(audit["event_zone_counts"])
    max_events_one_zone = max(audit["event_zone_counts"].values(), default=0)
    print(f"unique_zones_with_events={unique_event_zones}")
    print(f"max_events_single_zone={max_events_one_zone}")
    print(f"acceptances={len(acceptances)}")
    print("state_transitions=" + " ".join(f"{k}={v}" for k, v in sorted(audit["state_transitions"].items())))
    print("sample_transitions:")
    for ts, key, old, new, reason, center in audit["samples"]:
        print(f"  {ts} | {key} | {old}->{new} | reason={reason} | center={center:.2f}")
    print("zone_event_frequency_top10:")
    for key, count in audit["event_zone_counts"].most_common(10):
        print(f"  {key} | events={count}")
    print("audit_note=event counts should be reviewed for lifecycle independence before 30d validation")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward BTCUSDT zone edge validation")
    parser.add_argument("--days", type=int, default=7, choices=(7, 30, 90))
    parser.add_argument("--audit", action="store_true", help="print event-density and lifecycle diagnostics")
    args = parser.parse_args()

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT", page_limit=1000))
    bars = _load_klines(provider, start, end)
    if bars.empty:
        raise RuntimeError("No Binance kline data returned")

    profiles = _build_profiles_from_bars(bars)
    snapshots = _build_walk_forward_snapshots(profiles)
    events, acceptances, audit = _find_walk_forward_events(bars, snapshots)
    acceptance_results = _acceptance_resolutions(bars, acceptances)
    _print_summary(bars, profiles, snapshots, events, acceptances)
    if args.audit:
        _print_audit(bars, snapshots, events, acceptances, audit)

    for horizon in HORIZONS:
        baseline = _baseline(bars, events, horizon)
        rows = []
        for event in events:
            metrics = _future_metrics(bars, event, horizon)
            if metrics is None or baseline.get(event.direction) is None:
                continue
            move, mfe, mae = metrics
            rows.append((event, move, baseline[event.direction], mfe, mae, _outcome(mfe, mae)))

        print(f"\n=== {horizon}M WALK-FORWARD EDGE ===")
        print("event | n | favorable | adverse | avg dir | avg baseline | edge")
        print("------|---|-----------|---------|----------|---------------|-----")
        for event_type in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION"):
            subset = [r for r in rows if r[0].event == event_type]
            if not subset:
                continue
            avg_dir = np.mean([r[1] for r in subset])
            avg_base = np.mean([r[2] for r in subset])
            fav = np.mean([r[5] == "FAVORABLE" for r in subset]) * 100
            adv = np.mean([r[5] == "ADVERSE" for r in subset]) * 100
            print(f"{event_type:<9}| {len(subset):>4} | {fav:>9.1f}% | {adv:>7.1f}% | {avg_dir:>8.2f} | {avg_base:>13.2f} | {avg_dir-avg_base:+.2f}")

        for node_type in ("HVN", "LVN"):
            for status in ("HIGH_ACTIVE", "MEDIUM_ACTIVE", "DEVELOPING", "LOW", "HISTORICAL"):
                subset = [r for r in rows if r[0].node_type == node_type and r[0].status == status]
                if len(subset) < MIN_EDGE_SAMPLE:
                    continue
                avg_dir = np.mean([r[1] for r in subset])
                avg_base = np.mean([r[2] for r in subset])
                fav = np.mean([r[5] == "FAVORABLE" for r in subset]) * 100
                adv = np.mean([r[5] == "ADVERSE" for r in subset]) * 100
                print(f"{node_type:<8}| {status:<13} n={len(subset):>4} | fav={fav:>5.1f}% adv={adv:>5.1f}% | avg={avg_dir:+.2f} base={avg_base:+.2f} edge={avg_dir-avg_base:+.2f}")

        if baseline["UP"] is not None and baseline["DOWN"] is not None:
            print(f"baseline anchors: UP={baseline['UP']:.2f} DOWN={baseline['DOWN']:.2f}")
        else:
            print("baseline anchors: insufficient data")

    print("\n=== ACCEPTANCE RESOLUTION ===")
    print(f"acceptance_results={len(acceptance_results)}")
    for horizon in HORIZONS:
        subset = [r for r in acceptance_results if r.horizon == horizon]
        if not subset:
            continue
        up = sum(r.resolution == "RESOLVES_UP" for r in subset)
        down = sum(r.resolution == "RESOLVES_DOWN" for r in subset)
        failed = sum(r.resolution in {"FAILED_UP", "FAILED_DOWN"} for r in subset)
        held = sum(r.resolution == "HELD" for r in subset)
        print(f"{horizon:>3}m | n={len(subset):>4} | up={100*up/len(subset):>5.1f}% down={100*down/len(subset):>5.1f}% failed={100*failed/len(subset):>5.1f}% held={100*held/len(subset):>5.1f}%")

    print("\nInterpretation: positive edge means walk-forward event movement exceeded the independent non-event baseline.")
    print("No zone uses a profile window that ends after the event timestamp.")
    print("A BREAKOUT now requires prior zone interaction; pre-existing outside price cannot create a breakout.")
    print("Acceptance remains separate from directional scoring.")
    print("This is historical validation, not a trading signal.")


if __name__ == "__main__":
    main()
