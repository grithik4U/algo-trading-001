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

from zone_interaction_test import ZoneState, _build_profiles, _build_zones, _process_bar

HORIZONS = (5, 15, 30, 60)
MIN_EVENT_GAP_BARS = 2
MEASURED_EVENTS = ("BREAKOUT", "RETEST", "SWEEP", "REJECTION")


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


def _normalize_direction(event: str, direction: str | None) -> str | None:
    if event in {"BREAKOUT", "RETEST"}:
        if direction == "ABOVE":
            return "UP"
        if direction == "BELOW":
            return "DOWN"
    return direction if direction in {"UP", "DOWN"} else None


def _classify(event: str, direction: str | None, entry: float, future: pd.DataFrame) -> tuple[str, float, float]:
    if future.empty:
        return "NO_DATA", 0.0, 0.0

    highs = future["high"].astype(float)
    lows = future["low"].astype(float)

    up_move = max(0.0, float(highs.max() - entry))
    down_move = max(0.0, float(entry - lows.min()))

    if direction == "UP":
        favorable = up_move
        adverse = down_move
    elif direction == "DOWN":
        favorable = down_move
        adverse = up_move
    else:
        return "UNDEFINED_DIRECTION", up_move, down_move

    if favorable >= adverse * 1.25 and favorable >= 2.0:
        label = "FAVORABLE"
    elif adverse >= favorable * 1.25 and adverse >= 2.0:
        label = "ADVERSE"
    else:
        label = "MIXED"

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
            if event_name not in MEASURED_EVENTS:
                continue

            previous_bar = last_event_bar.get(zone.zone_id)
            if previous_bar is not None and bar_index - previous_bar < MIN_EVENT_GAP_BARS:
                continue
            last_event_bar[zone.zone_id] = bar_index

            close = float(row["close"])
            if event_name in {"BREAKOUT", "RETEST"}:
                direction = _normalize_direction(event_name, previous_direction)
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
                    event_name, direction, close, future
                )
                last_close = float(future["close"].iloc[-1])
                signed_move = last_close - close
                directional_move = (
                    signed_move if direction == "UP" else -signed_move
                    if direction == "DOWN" else 0.0
                )
                forward[horizon] = {
                    "close": last_close,
                    "move": signed_move,
                    "directional_move": directional_move,
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
    print("time | type | zone | event | dir | entry | 15m move | dir move | 15m MFE | 15m MAE | outcome")
    print("-----|------|------|-------|-----|-------|----------|----------|----------|----------|--------")

    for event in events[-25:][::-1]:
        result = event.forward.get(15)
        if not result:
            continue
        print(
            f"{event.timestamp.strftime('%H:%M')} | {event.node_type:<3} | "
            f"{event.low:.2f}->{event.high:.2f} | {event.event:<11} | "
            f"{str(event.direction or '-'):>4} | {event.entry:>7.2f} | "
            f"{float(result['move']):>+8.2f} | {float(result['directional_move']):>+8.2f} | "
            f"{float(result['mfe']):>8.2f} | {float(result['mae']):>8.2f} | {result['outcome']}"
        )


def _print_summary(events: list[Outcome]) -> None:
    print("\n=== OUTCOME SUMMARY ===")
    print("event | n | favorable | mixed | adverse | avg dir move")
    print("------|---|-----------|-------|---------|-------------")

    for event_name in MEASURED_EVENTS:
        rows = [e.forward[15] for e in events if e.event == event_name and 15 in e.forward]
        if not rows:
            print(f"{event_name:<11} |   0 |       0.0% |   0.0% |     0.0% |        +0.00")
            continue
        counts = {key: sum(1 for r in rows if r["outcome"] == key) for key in
                  ("FAVORABLE", "MIXED", "ADVERSE")}
        n = len(rows)
        avg_directional_move = sum(float(r["directional_move"]) for r in rows) / n
        print(
            f"{event_name:<11} | {n:>2} | {counts['FAVORABLE']/n*100:>9.1f}% | "
            f"{counts['MIXED']/n*100:>5.1f}% | {counts['ADVERSE']/n*100:>7.1f}% | "
            f"{avg_directional_move:>+11.2f}"
        )

    print("\n=== HORIZON CHECK ===")
    print("event | horizon | n | avg dir move | avg MFE | avg MAE")
    print("------|---------|---|--------------|----------|--------")
    for event_name in MEASURED_EVENTS:
        for horizon in HORIZONS:
            rows = [e.forward[horizon] for e in events if e.event == event_name and horizon in e.forward]
            if not rows:
                continue
            avg_move = sum(float(r["directional_move"]) for r in rows) / len(rows)
            avg_mfe = sum(float(r["mfe"]) for r in rows) / len(rows)
            avg_mae = sum(float(r["mae"]) for r in rows) / len(rows)
            print(
                f"{event_name:<11} | {horizon:>7}m | {len(rows):>2} | "
                f"{avg_move:>+12.2f} | {avg_mfe:>8.2f} | {avg_mae:>7.2f}"
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
    print("- BREAKOUT and RETEST direction follows the confirmed breakout side.")
    print("- SWEEP/REJECTION direction is inferred from close versus zone center.")
    print("- Directional move is positive when price moves in the event direction.")
    print("- FAVORABLE means MFE materially exceeded MAE; ADVERSE means the reverse; otherwise MIXED.")
    print("- Acceptance is excluded from directional outcome scoring because it has no inherent direction.")
    print("- This script measures historical behavior only; it does not generate trade signals.")


if __name__ == "__main__":
    main()
