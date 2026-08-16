"""Validate whether zone interactions have an edge versus a matched BTCUSDT baseline.

This is a research/diagnostic script. It does not generate trade signals.

The test reuses the temporal HVN/LVN zone builder from zone_interaction_test.py,
then evaluates the first confirmed interaction per zone/bar and compares the
forward directional move against the same-direction unconditional market move.
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
    STEP,
    WINDOW_SIZE,
    _build_profiles,
    _build_zones,
)

HORIZONS = (5, 15, 30, 60)
MIN_EDGE_SAMPLE = 5
NEAR_ZONE_TICKS = 2.0


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
                    direction = broken_direction
                    events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "RETEST", direction, close))
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
                    if inside_streak >= ACCEPTANCE_BARS:
                        last_state = "ACCEPTANCE"
                    else:
                        last_state = "TOUCH"
                else:
                    inside_streak = 0
                    outside_streak = 0
                    last_state = "TOUCH"
                continue

            if last_touch_bar is not None:
                bars_since = i - last_touch_bar
                if bars_since <= MAX_REJECTION_BARS and last_state in {"TOUCH", "SWEEP"}:
                    direction = _direction("REJECTION", close, low, high)
                    events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "REJECTION", direction, close))
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


def _future_metrics(dataset, event: Event, horizon: int) -> tuple[float, float, float] | None:
    bars = dataset.bars
    try:
        pos = bars.index.get_loc(event.timestamp)
    except KeyError:
        return None
    if not isinstance(pos, int):
        return None
    end = pos + horizon
    if end >= len(bars):
        return None

    future = bars.iloc[pos + 1 : end + 1]
    if future.empty:
        return None

    final_close = float(future["close"].iloc[-1])
    move = final_close - event.entry
    if event.direction == "DOWN":
        move = -move
        mfe = event.entry - float(future["low"].min())
        mae = float(future["high"].max()) - event.entry
    elif event.direction == "UP":
        mfe = float(future["high"].max()) - event.entry
        mae = event.entry - float(future["low"].min())
    else:
        return None
    return move, max(0.0, mfe), max(0.0, mae)


def _baseline(dataset, event: Event, horizon: int) -> float | None:
    bars = dataset.bars
    try:
        pos = bars.index.get_loc(event.timestamp)
    except KeyError:
        return None
    if not isinstance(pos, int) or pos + horizon >= len(bars):
        return None
    future_close = float(bars["close"].iloc[pos + horizon])
    move = future_close - event.entry
    return -move if event.direction == "DOWN" else move


def _outcome(mfe: float, mae: float) -> str:
    threshold = max(1.0, 0.25 * (mfe + mae))
    if mfe - mae > threshold:
        return "FAVORABLE"
    if mae - mfe > threshold:
        return "ADVERSE"
    return "MIXED"


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
        rows = []
        for event in events:
            metrics = _future_metrics(dataset, event, horizon)
            baseline = _baseline(dataset, event, horizon)
            if metrics is None or baseline is None:
                continue
            move, mfe, mae = metrics
            rows.append((event, move, baseline, mfe, mae, _outcome(mfe, mae)))

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
            print(f"{event_type:<8}| ALL  | ALL    | {len(subset):>2} | {fav:>9.1f}% | {adv:>7.1f}% | {avg_dir:>8.2f} | {avg_base:>13.2f} | {avg_dir-avg_base:>+.2f}")

        for node_type in ("HVN", "LVN"):
            for status in ("HIGH_ACTIVE", "MEDIUM_ACTIVE", "DEVELOPING", "LOW", "HISTORICAL"):
                subset = [r for r in rows if r[0].node_type == node_type and r[0].status == status]
                if len(subset) < MIN_EDGE_SAMPLE:
                    continue
                avg_dir = sum(r[1] for r in subset) / len(subset)
                avg_base = sum(r[2] for r in subset) / len(subset)
                fav = sum(r[5] == "FAVORABLE" for r in subset) / len(subset) * 100
                print(f"ALL     | {node_type:<4} | {status:<6} | {len(subset):>2} | {fav:>9.1f}% | {'-':>7} | {avg_dir:>8.2f} | {avg_base:>13.2f} | {avg_dir-avg_base:>+.2f}")

    print("\n=== RECENT DIRECTIONAL EVENTS ===")
    print("time | type | status | zone | event | dir | entry")
    print("-----|------|--------|------|-------|-----|------")
    for event in reversed([e for e in events if e.direction is not None])[:25]:
        print(f"{event.timestamp.strftime('%H:%M')} | {event.node_type:<4} | {event.status:<8} | {event.low:.2f}->{event.high:.2f} | {event.event:<9} | {event.direction:<4} | {event.entry:.2f}")

    print("\nInterpretation: positive edge means the zone event beat the matched unconditional BTCUSDT move in the event direction.")
    print("Minimum sample filter applies only to the zone-status breakdown. This is historical validation, not a trading signal.")


if __name__ == "__main__":
    main()
