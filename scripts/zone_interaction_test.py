"""Detect deterministic price interactions with temporal HVN/LVN structural zones."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.data.dataset import load_aligned_binance_dataset
from trading_engine.volume_profile.trade_profile import build_trade_volume_profile

from profile_temporal_test import (
    MAX_ZONE_WIDTH,
    MIN_ZONE_COVERAGE,
    NODE_MATCH_TOLERANCE,
    ZONE_GAP_TOLERANCE,
    _cluster_nodes,
    _consolidate_zones,
    _zone_coverage,
)

WINDOW_SIZE = timedelta(hours=1)
STEP = timedelta(minutes=15)
PROFILE_BIN_TICKS = 25

APPROACH_TOLERANCE = 2.00
ACCEPTANCE_BARS = 3
BREAKOUT_BARS = 2
MAX_REJECTION_BARS = 3


@dataclass
class ZoneState:
    zone_id: int
    node_type: str
    low: float
    high: float
    center: float
    coverage: float
    status: str
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    touch_count: int = 0
    acceptance_count: int = 0
    rejection_count: int = 0
    sweep_count: int = 0
    breakout_count: int = 0
    retest_count: int = 0
    state: str = "UNTESTED"
    last_interaction: str = "NONE"
    events: list[str] = field(default_factory=list)
    inside_streak: int = 0
    outside_streak: int = 0
    bars_since_touch: int | None = None
    broken_direction: str | None = None


def _build_profiles(dataset):
    windows = []
    cursor = dataset.start
    while cursor + WINDOW_SIZE <= dataset.end:
        window_end = cursor + WINDOW_SIZE
        trades = dataset.trades.loc[
            (dataset.trades.index >= cursor) & (dataset.trades.index < window_end)
        ]
        if len(trades) >= 100:
            windows.append((cursor, window_end, trades))
        cursor += STEP

    profiles = []
    for window_start, window_end, trades in windows:
        profile = build_trade_volume_profile(
            trades,
            tick_size=0.01,
            profile_bin_ticks=PROFILE_BIN_TICKS,
            node_smoothing_bins=3,
            node_prominence=0.25,
            node_min_separation_bins=3,
            node_min_relative_volume=1.0,
        )
        profiles.append((window_start, window_end, profile))
    return profiles


def _build_zones(profiles, node_type: str):
    nodes = []
    for _, _, profile in profiles:
        nodes.extend(profile.hvn_nodes if node_type == "HVN" else profile.lvn_nodes)

    clusters = _cluster_nodes(nodes)
    zones = _consolidate_zones(clusters)
    total_windows = len(profiles)
    result = []

    for zone in zones:
        matches = _zone_coverage(zone, profiles, node_type)
        coverage = len(matches) / max(1, total_windows)
        if coverage < MIN_ZONE_COVERAGE:
            continue
        recent_cutoff = max(0, total_windows - 4)
        recent = bool(matches and matches[-1] >= recent_cutoff)
        if coverage >= 0.75 and recent:
            status = "HIGH_ACTIVE"
        elif coverage >= 0.50 and recent:
            status = "MEDIUM_ACTIVE"
        elif coverage >= 0.50:
            status = "HISTORICAL"
        elif coverage >= MIN_ZONE_COVERAGE and recent:
            status = "DEVELOPING"
        else:
            status = "LOW"
        result.append(
            {
                "node_type": node_type,
                "low": float(zone["low"]),
                "high": float(zone["high"]),
                "center": float(zone["center"]),
                "coverage": coverage,
                "status": status,
            }
        )
    return result


def _contains(zone: ZoneState, price: float) -> bool:
    return zone.low <= price <= zone.high


def _range_touches(zone: ZoneState, low: float, high: float) -> bool:
    return high >= zone.low and low <= zone.high


def _record(zone: ZoneState, timestamp: pd.Timestamp, event: str, state: str) -> None:
    ts = timestamp.isoformat()
    zone.events.append(f"{ts}:{event}")
    zone.last_seen = timestamp.to_pydatetime()
    zone.last_interaction = event
    zone.state = state


def _process_bar(zone: ZoneState, timestamp: pd.Timestamp, row: pd.Series) -> None:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    touches = _range_touches(zone, low, high)
    inside = _contains(zone, close)
    above = close > zone.high
    below = close < zone.low

    if touches:
        if zone.first_seen is None:
            zone.first_seen = timestamp.to_pydatetime()
        zone.touch_count += 1
        zone.bars_since_touch = 0

        if zone.broken_direction is not None:
            zone.retest_count += 1
            _record(zone, timestamp, "RETEST", "RETEST")
            zone.broken_direction = None
            zone.inside_streak = 0
            zone.outside_streak = 0
            return

        if high > zone.high and low < zone.low and inside:
            zone.sweep_count += 1
            _record(zone, timestamp, "SWEEP", "SWEEP")
            zone.inside_streak = 1
            zone.outside_streak = 0
            return

        if inside:
            zone.inside_streak += 1
            zone.outside_streak = 0
            if zone.inside_streak >= ACCEPTANCE_BARS:
                zone.acceptance_count += 1
                _record(zone, timestamp, "ACCEPTANCE", "ACCEPTANCE")
            else:
                _record(zone, timestamp, "TOUCH", "TOUCH")
        else:
            zone.inside_streak = 0
            zone.outside_streak = 0
            _record(zone, timestamp, "TOUCH", "TOUCH")
        return

    if zone.bars_since_touch is not None:
        zone.bars_since_touch += 1
        if zone.bars_since_touch <= MAX_REJECTION_BARS and zone.state in {"TOUCH", "SWEEP"}:
            zone.rejection_count += 1
            _record(zone, timestamp, "REJECTION", "REJECTION")

    if above or below:
        direction = "ABOVE" if above else "BELOW"
        zone.outside_streak += 1
        zone.inside_streak = 0
        if zone.outside_streak >= BREAKOUT_BARS and zone.state not in {"BREAKOUT", "CONFIRMED_BREAKOUT"}:
            zone.breakout_count += 1
            zone.broken_direction = direction
            _record(zone, timestamp, "BREAKOUT", "BREAKOUT")
    else:
        zone.outside_streak = 0


def _latest_event(zone: ZoneState) -> str:
    return zone.events[-1].split(":", 1)[1] if zone.events else "NONE"


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

    for timestamp, row in dataset.bars.iterrows():
        for zone in states:
            _process_bar(zone, timestamp, row)

    current_price = float(dataset.bars["close"].iloc[-1])
    print("=== BTCUSDT ZONE INTERACTION TEST ===")
    print(f"dataset={dataset.start} -> {dataset.end}")
    print(f"bars={len(dataset.bars)}")
    print(f"zones={len(states)}")
    print(f"current_price={current_price:.2f}")
    print()
    print("zone | type | distance | status | state | touches | accepts | rejects | sweeps | breakouts | retests | last")
    print("-----|------|----------|--------|-------|---------|----------|----------|--------|-----------|---------|-----")

    for zone in sorted(states, key=lambda z: (-z.coverage, abs(z.center - current_price), z.node_type)):
        if current_price < zone.low:
            distance = zone.low - current_price
        elif current_price > zone.high:
            distance = current_price - zone.high
        else:
            distance = 0.0
        print(
            f"{zone.low:.2f}->{zone.high:.2f} | {zone.node_type:<3} | {distance:>8.2f} | "
            f"{zone.status:<13} | {zone.state:<19} | {zone.touch_count:>7} | "
            f"{zone.acceptance_count:>8} | {zone.rejection_count:>8} | {zone.sweep_count:>6} | "
            f"{zone.breakout_count:>9} | {zone.retest_count:>7} | {_latest_event(zone)}"
        )

    print("\n=== ACTIVE/RECENT INTERACTIONS ===")
    active = [z for z in states if z.last_interaction != "NONE"]
    for zone in sorted(active, key=lambda z: z.last_seen or datetime.min.replace(tzinfo=timezone.utc), reverse=True)[:20]:
        print(
            f"{zone.node_type} {zone.low:.2f}->{zone.high:.2f} | "
            f"state={zone.state} | last={zone.last_seen.isoformat() if zone.last_seen else 'N/A'} | "
            f"touches={zone.touch_count} accepts={zone.acceptance_count} rejects={zone.rejection_count} "
            f"sweeps={zone.sweep_count} breakouts={zone.breakout_count} retests={zone.retest_count}"
        )

    print("\nState rules: acceptance requires 3 consecutive closes inside; breakout requires 2 consecutive closes outside.")
    print("Sweep requires a wick through both zone boundaries with the close back inside.")
    print("Rejection is recorded when price leaves a touched/swept zone within 3 bars without establishing acceptance.")
    print("Overlapping HVN/LVN zones are tracked independently; no structure is deleted or collapsed.")


if __name__ == "__main__":
    main()
