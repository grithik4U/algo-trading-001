"""Measure forward price outcomes after temporal HVN/LVN zone interactions.

This is a validation script, not a trading strategy. It reuses the same temporal
zone construction and interaction rules as zone_interaction_test.py, then measures
what price actually did after each meaningful interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.data.dataset import load_aligned_binance_dataset

from zone_interaction_test import (
    ZoneState,
    _build_profiles,
    _build_zones,
    _process_bar,
)

HORIZONS = (5, 15, 30, 60)
MIN_EVENT_GAP_BARS = 2


@dataclass
class Outcome:
    timestamp: pd.Timestamp
    node_type: str
    low: float
    high: float
    center: float
    event: str
    direction: str | None
    entry: float
    forward: dict[int, dict[str, float | str]]


def _direction_for_event(zone: ZoneState, event: str, close: float) -> str | None:
    if event == "BREAKOUT":
        return zone.broken_direction
    if event == "RETEST":
        return zone.broken_direction
    if event in {"REJECTION", "SWEEP"}:
        if close > zone.center:
            return "UP"
        if close < zone.center:
            return "DOWN"
    if event == "ACCEPTANCE":
        return None
    return None


def _classify(event: str, direction: str | None, entry: float, zone: ZoneState,
              future: pd.DataFrame) -> tuple[str, float, float]:
    if future.empty:
        return "NO_DATA", 0.0, 0.0

    highs = future["high"].astype(float)
    lows = future["low"].astype(float)
    closes = future["close"].astype(float)

    up_move = float(highs.max() - entry)
    down_move = float(entry - lows.min())

    if direction == "UP":
        favorable = up_move
        adverse = down_move
    elif direction == "DOWN":
        favorable = down_move
        adverse = up_move
    else:
        # For acceptance, measure whether price remains in/returns to the zone.
        favorable = max(0.0, float(zone.high - closes.iloc[-1]))
        adverse = max(0.0, float(closes.iloc[-1] - zone.high))

    if event == "BREAKOUT" and direction in {"ABOVE", "BELOW"}:
        direction = "UP" if direction == "ABOVE" else "DOWN"

    if direction in {"UP", "DOWN"}:
        if favorable >= adverse * 1.25 and favorable >= 2.0:
            label = "FAVORABLE"
        elif adverse >= favorable * 1.25 and adverse >= 2.0:
            label = "ADVERSE"
        else:
            label = "MIXED"
    else:
        final_close = float(closes.iloc[-1])
        if zone.low <= final_close <= zone.high:
            label = "HELD_ZONE"
        elif abs(final_close - zone.center) <= 3.0:
            label = "NEAR_ZONE"
        else:
            label = "LEFT_ZONE"

    return label, favorable, adverse


def _collect_outcomes(dataset, states: list[ZoneState]) -> list[Outcome]:
    bars = dataset.bars
    events: list[Outcome] = []
    last_event_bar: dict[int, int] = {}

    for bar_index, (timestamp, row) in enumerate(bars.iterrows()):
        for zone in states:
            previous_count = len(zone.events)
            previous_direction = zone.broken_direction
            _process_bar(zone, timestamp, row)
            if len(zone.events) == previous_count:
                continue

            event_name = zone.events[-1].rsplit(":", 1)[-1]
            if event_name not in {"ACCEPTANCE", "SWEEP", "REJECTION", "BREAKOUT", "RETEST"}:
                continue

            previous_bar = last_event_bar.get(zone.zone_id)
            if previous_bar is not None and bar_index - previous_bar < MIN_EVENT_GAP_BARS:
                continue
            last_event_bar[zone.zone_id] = bar_index

            close = float(row["close"])
            direction = previous_direction
            if event_name == "BREAKOUT":
                direction = zone.broken_direction
            elif event_name in {"SWEEP", "REJECTION"}:
                direction = "UP" if close >= zone.center else "DOWN"
            else:
                direction = None

            forward: dict[int, dict[str, float | str]] = {}
            for horizon in HORIZONS:
                future = bars.iloc[bar_index + 1 : bar_index + 1 + horizon]
                if future.empty:
                    continue
                label, favorable, adverse = _classify(
                    event_name, direction, close, zone, future
                )
                last_close = float(future["close"].iloc[-1])
                forward[horizon] = {
                    "close": last_close,
                    "move": last_close - close,
                    "mfe": favorable,
                    "mae": adverse,
                    "outcome": label,
                }

            events.append(
                Outcome(
                    timestamp=timestamp,
                    node_type=zone.node_type,
                    low=zone.low,
                    high=zone.high,
                    center=zone.center,
                    event=event_name,
                    direction=direction,
                    entry=close,
                    forward=forward,
                )
            )

    return events


def _print_event_table(events: list[Outcome]) -> None:
    print("\n=== RECENT ZONE OUTCOMES ===")
    print("time | type | zone | event | dir | entry | 15m move | 15m MFE | 15m MAE | outcome")
    print("-----|------|------|-------|-----|-------|----------|----------|----------|--------")

    for event in events[-25:][::-1]:
        result = event.forward.get(15)
        if not result:
            continue
        print(
            f"{event.timestamp.strftime('%H:%M')} | {event.node_type:<3} | "
            f"{event.low:.2f}->{event.high:.2f} | {event.event:<11} | "
            f"{str(event.direction or '-'):>4} | {event.entry:>7.2f} | "
            f"{float(result['move']):>+8.2f} | {float(result['mfe']):>8.2f} | "
            f"{float(result['mae']):>8.2f} | {result['outcome']}"
        )


def _print_summary(events: list[Outcome]) -> None:
    print("\n=== OUTCOME SUMMARY ===")
    print("event | n | favorable | mixed | adverse | no_data")
    print("------|---|-----------|-------|---------|--------")

    for event_name in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION", "ACCEPTANCE"):
        rows = [e.forward[15] for e in events if e.event == event_name and 15 in e.forward]
        if not rows:
            print(f"{event_name:<11} |   0 |       0.0% |   0.0% |     0.0% |    0.0%")
            continue
        counts = {key: sum(1 for r in rows if r["outcome"] == key) for key in
                  ("FAVORABLE", "MIXED", "ADVERSE", "NO_DATA")}
        n = len(rows)
        print(
            f"{event_name:<11} | {n:>2} | {counts['FAVORABLE']/n*100:>9.1f}% | "
            f"{counts['MIXED']/n*100:>5.1f}% | {counts['ADVERSE']/n*100:>7.1f}% | "
            f"{counts['NO_DATA']/n*100:>6.1f}%"
        )

    print("\n=== HORIZON CHECK ===")
    print("event | horizon | n | avg move | avg MFE | avg MAE")
    print("------|---------|---|----------|----------|--------")
    for event_name in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION", "ACCEPTANCE"):
        for horizon in HORIZONS:
            rows = [e.forward[horizon] for e in events if e.event == event_name and horizon in e.forward]
            if not rows:
                continue
            avg_move = sum(float(r["move"]) for r in rows) / len(rows)
            avg_mfe = sum(float(r["mfe"]) for r in rows) / len(rows)
            avg_mae = sum(float(r["mae"]) for r in rows) / len(rows)
            print(
                f"{event_name:<11} | {horizon:>7}m | {len(rows):>2} | "
                f"{avg_move:>+8.2f} | {avg_mfe:>8.2f} | {avg_mae:>7.2f}"
            )


def main() -> None:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(hours=4)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT"))
    dataset = load_aligned_binance_dataset(
        provider, interval="1m", start=start, end=end, bar_limit=1000
    )

    profiles = _build_profiles(dataset)
    zones = _build_zones(profiles, "HVN") + _build_zones(profiles, "LVN")
    zones.sort(key=lambda z: (z["center"], z["node_type"]))
    states = [
        ZoneState(
            zone_id=i + 1,
            node_type=z["node_type"],
            low=z["low"],
            high=z["high"],
            center=z["center"],
            coverage=z["coverage"],
            status=z["status"],
        )
        for i, z in enumerate(zones)
    ]

    events = _collect_outcomes(dataset, states)

    print("=== BTCUSDT ZONE OUTCOME TEST ===")
    print(f"dataset={dataset.start} -> {dataset.end}")
    print(f"bars={len(dataset.bars)}")
    print(f"zones={len(states)}")
    print(f"events={len(events)}")
    print(f"current_price={float(dataset.bars['close'].iloc[-1]):.2f}")
    print(f"horizons={','.join(f'{h}m' for h in HORIZONS)}")

    _print_event_table(events)
    _print_summary(events)

    print("\nOutcome rules:")
    print("- BREAKOUT direction is taken from the side of the confirmed close outside the zone.")
    print("- SWEEP/REJECTION direction is inferred from the close relative to the zone center.")
    print("- FAVORABLE means MFE materially exceeded MAE; ADVERSE means the reverse; otherwise MIXED.")
    print("- Acceptance is direction-neutral and classified by whether price remains in/near the zone.")
    print("- This script measures historical behavior only; it does not generate trade signals.")


if __name__ == "__main__":
    main()
