"""Validate whether zone interactions have an edge versus an independent BTCUSDT baseline.

Research/diagnostic only. Does not generate trade signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.data.dataset import load_aligned_binance_dataset
from zone_interaction_test import (
    ACCEPTANCE_BARS,
    BREAKOUT_BARS,
    MAX_REJECTION_BARS,
    _build_profiles,
    _build_zones,
)

HORIZONS = (5, 15, 30, 60)
MIN_EDGE_SAMPLE = 5


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


def _contains(low: float, high: float, price: float) -> bool:
    return low <= price <= high


def _touches(low: float, high: float, bar_low: float, bar_high: float) -> bool:
    return bar_high >= low and bar_low <= high


def _direction(event: str, close: float, low: float, high: float) -> str | None:
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


def _find_events(dataset, zones) -> list[Event]:
    events: list[Event] = []
    for z in zones:
        low, high = z["low"], z["high"]
        inside_streak = 0
        outside_streak = 0
        last_touch_bar: int | None = None
        broken_direction: str | None = None
        last_state = "UNTESTED"

        for i, (ts, row) in enumerate(dataset.bars.iterrows()):
            bar_low = float(row["low"])
            bar_high = float(row["high"])
            close = float(row["close"])
            touched = _touches(low, high, bar_low, bar_high)
            inside = _contains(low, high, close)
            above = close > high
            below = close < low

            if touched:
                last_touch_bar = i
                if broken_direction is not None:
                    events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "RETEST", broken_direction, close))
                    broken_direction = None
                    inside_streak = 0
                    outside_streak = 0
                    last_state = "RETEST"
                    continue

                if bar_high > high and bar_low < low and inside:
                    events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "SWEEP", _direction("SWEEP", close, low, high), close))
                    inside_streak = 1
                    outside_streak = 0
                    last_state = "SWEEP"
                    continue

                if inside:
                    inside_streak += 1
                    outside_streak = 0
                    last_state = "ACCEPTANCE" if inside_streak >= ACCEPTANCE_BARS else "TOUCH"
                else:
                    inside_streak = 0
                    outside_streak = 0
                    last_state = "TOUCH"
                continue

            if last_touch_bar is not None:
                bars_since = i - last_touch_bar
                if bars_since <= MAX_REJECTION_BARS and last_state in {"TOUCH", "SWEEP"}:
                    events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "REJECTION", _direction("REJECTION", close, low, high), close))
                    last_state = "REJECTION"

            if above or below:
                outside_streak += 1
                inside_streak = 0
                if outside_streak >= BREAKOUT_BARS and last_state not in {"BREAKOUT", "CONFIRMED_BREAKOUT"}:
                    direction = "UP" if above else "DOWN"
                    events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "BREAKOUT", direction, close))
                    broken_direction = direction
                    last_state = "BREAKOUT"
            else:
                outside_streak = 0

    return sorted(events, key=lambda e: e.timestamp)


def _position(dataset, timestamp: pd.Timestamp) -> int | None:
    try:
        pos = dataset.bars.index.get_loc(timestamp)
    except KeyError:
        return None
    return pos if isinstance(pos, int) else None


def _future_metrics(dataset, event: Event, horizon: int) -> tuple[float, float, float] | None:
    pos = _position(dataset, event.timestamp)
    if pos is None or pos + horizon >= len(dataset.bars) or event.direction is None:
        return None
    future = dataset.bars.iloc[pos + 1 : pos + horizon + 1]
    if future.empty:
        return None
    final_close = float(future["close"].iloc[-1])
    raw_move = final_close - event.entry
    if event.direction == "DOWN":
        move = -raw_move
        mfe = event.entry - float(future["low"].min())
        mae = float(future["high"].max()) - event.entry
    else:
        move = raw_move
        mfe = float(future["high"].max()) - event.entry
        mae = event.entry - float(future["low"].min())
    return move, max(0.0, mfe), max(0.0, mae)


def _event_positions(dataset, events) -> set[int]:
    positions: set[int] = set()
    for event in events:
        pos = _position(dataset, event.timestamp)
        if pos is not None:
            positions.add(pos)
    return positions


def _baseline_by_direction(dataset, events: list[Event], horizon: int) -> dict[str, float | None]:
    bars = dataset.bars
    event_positions = _event_positions(dataset, events)
    blocked: set[int] = set()
    for pos in event_positions:
        blocked.update(range(max(0, pos - horizon + 1), min(len(bars), pos + horizon)))

    sums = {"UP": 0.0, "DOWN": 0.0}
    counts = {"UP": 0, "DOWN": 0}
    for pos in range(len(bars) - horizon):
        if pos in blocked:
            continue
        entry = float(bars["close"].iloc[pos])
        future_close = float(bars["close"].iloc[pos + horizon])
        raw = future_close - entry
        sums["UP"] += raw
        sums["DOWN"] += -raw
        counts["UP"] += 1
        counts["DOWN"] += 1

    return {
        direction: (sums[direction] / counts[direction] if counts[direction] else None)
        for direction in ("UP", "DOWN")
    }


def _format_baseline(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "N/A"


def _outcome(mfe: float, mae: float) -> str:
    threshold = max(1.0, 0.25 * (mfe + mae))
    if mfe - mae > threshold:
        return "FAVORABLE"
    if mae - mfe > threshold:
        return "ADVERSE"
    return "MIXED"


def _print_breakdown(rows, horizon: int) -> None:
    print(f"\n=== {horizon}M EDGE ===")
    print("event | type | status | n | favorable | adverse | avg dir | avg baseline | edge")
    print("------|------|--------|---|-----------|---------|----------|---------------|-----")

    for event_type in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION"):
        subset = [r for r in rows if r[0].event == event_type]
        if not subset:
            continue
        fav = sum(r[5] == "FAVORABLE" for r in subset) / len(subset) * 100
        adv = sum(r[5] == "ADVERSE" for r in subset) / len(subset) * 100
        avg_dir = sum(r[1] for r in subset) / len(subset)
        avg_base = sum(r[2] for r in subset) / len(subset)
        print(f"{event_type:<8}| ALL  | ALL    | {len(subset):>3} | {fav:>9.1f}% | {adv:>7.1f}% | {avg_dir:>8.2f} | {avg_base:>13.2f} | {avg_dir - avg_base:>+.2f}")

    for node_type in ("HVN", "LVN"):
        for status in ("HIGH_ACTIVE", "MEDIUM_ACTIVE", "DEVELOPING", "LOW", "HISTORICAL"):
            subset = [r for r in rows if r[0].node_type == node_type and r[0].status == status]
            if len(subset) < MIN_EDGE_SAMPLE:
                continue
            avg_dir = sum(r[1] for r in subset) / len(subset)
            avg_base = sum(r[2] for r in subset) / len(subset)
            fav = sum(r[5] == "FAVORABLE" for r in subset) / len(subset) * 100
            adv = sum(r[5] == "ADVERSE" for r in subset) / len(subset) * 100
            print(f"ALL     | {node_type:<4} | {status:<13} | {len(subset):>3} | {fav:>9.1f}% | {adv:>7.1f}% | {avg_dir:>8.2f} | {avg_base:>13.2f} | {avg_dir - avg_base:>+.2f}")


def main() -> None:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(hours=4)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT"))
    dataset = load_aligned_binance_dataset(provider, interval="1m", start=start, end=end, bar_limit=1000)
    profiles = _build_profiles(dataset)
    zones = _build_zones(profiles, "HVN") + _build_zones(profiles, "LVN")
    zones.sort(key=lambda z: (z["center"], z["node_type"]))
    events = _find_events(dataset, zones)

    print("=== BTCUSDT ZONE EDGE TEST ===")
    print(f"dataset={dataset.start} -> {dataset.end}")
    print(f"bars={len(dataset.bars)}")
    print(f"zones={len(zones)}")
    print(f"events={len(events)}")
    print(f"current_price={float(dataset.bars['close'].iloc[-1]):.2f}")
    print(f"horizons={','.join(f'{h}m' for h in HORIZONS)}")

    for horizon in HORIZONS:
        baseline = _baseline_by_direction(dataset, events, horizon)
        rows = []
        for event in events:
            metrics = _future_metrics(dataset, event, horizon)
            if metrics is None or event.direction is None:
                continue
            base = baseline.get(event.direction)
            if base is None:
                continue
            move, mfe, mae = metrics
            rows.append((event, move, base, mfe, mae, _outcome(mfe, mae)))
        _print_breakdown(rows, horizon)
        print(f"baseline anchors: UP={_format_baseline(baseline['UP'])} DOWN={_format_baseline(baseline['DOWN'])}")
        if baseline["UP"] is None or baseline["DOWN"] is None:
            print(f"WARNING: insufficient independent baseline anchors for {horizon}m within the current 4-hour dataset.")

    print("\n=== RECENT DIRECTIONAL EVENTS ===")
    print("time | type | status | zone | event | dir | entry")
    print("-----|------|--------|------|-------|-----|------")
    directional = [e for e in events if e.direction is not None]
    for event in list(reversed(directional))[:25]:
        print(f"{event.timestamp.strftime('%H:%M')} | {event.node_type:<4} | {event.status:<13} | {event.low:.2f}->{event.high:.2f} | {event.event:<9} | {event.direction:<4} | {event.entry:.2f}")

    print("\nInterpretation: positive edge means the event's average directional move beat the independent non-event BTCUSDT baseline.")
    print("Baseline anchors exclude zone-event bars and forward windows overlapping zone events.")
    print("Minimum sample filter applies only to the zone-status breakdown. This is historical validation, not a trading signal.")


if __name__ == "__main__":
    main()
