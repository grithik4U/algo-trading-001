"""Long-history BTCUSDT zone-edge validation.

Research/diagnostic only. Does not generate trade signals.

Directional events are generated through independent zone lifecycles so the
same interaction is not counted repeatedly. Acceptance is measured separately
as a zone-resolution outcome rather than being forced into directional scoring.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.volume_profile.trade_profile import build_trade_volume_profile
from zone_interaction_test import (
    ACCEPTANCE_BARS,
    BREAKOUT_BARS,
    MAX_REJECTION_BARS,
    _cluster_nodes,
    _consolidate_zones,
    _zone_coverage,
)

WINDOW_SIZE = timedelta(hours=1)
STEP = timedelta(minutes=15)
PROFILE_BIN_TICKS = 25
HORIZONS = (5, 15, 30, 60)
MIN_EDGE_SAMPLE = 20
LIFECYCLE_COOLDOWN_BARS = 15
ACCEPTANCE_RESOLUTION_HORIZON = 60


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


@dataclass(frozen=True)
class Acceptance:
    timestamp: pd.Timestamp
    node_type: str
    low: float
    high: float
    center: float
    status: str
    entry: float


@dataclass(frozen=True)
class AcceptanceResolution:
    acceptance: Acceptance
    horizon: int
    resolution: str
    move: float
    time_to_resolution: int | None


def _load_klines(provider, start, end):
    cursor = start
    pages = []
    page = provider.config.page_limit
    while cursor < end:
        frame = provider.get_klines("1m", start_time=cursor, end_time=end, limit=page)
        if frame.empty:
            break
        pages.append(frame)
        last = frame.index.max().to_pydatetime()
        if last >= end or len(frame) < page:
            break
        nxt = last + timedelta(minutes=1)
        if nxt <= cursor:
            raise RuntimeError("Binance kline pagination cursor did not advance")
        cursor = nxt
    if not pages:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    result = pd.concat(pages)
    result = result[~result.index.duplicated(keep="first")].sort_index()
    return result[(result.index >= pd.Timestamp(start)) & (result.index <= pd.Timestamp(end))]


def _bars_to_profile_trades(bars):
    if bars.empty:
        return pd.DataFrame(columns=["price", "quantity", "buyer_maker"])
    rows = []
    for ts, row in bars.iterrows():
        volume = float(row["volume"])
        if volume <= 0:
            continue
        prices = [float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])]
        weights = [0.20, 0.30, 0.20, 0.30]
        bullish = float(row["close"]) >= float(row["open"])
        for price, weight in zip(prices, weights):
            rows.append({"timestamp": ts, "price": price, "quantity": volume * weight, "buyer_maker": not bullish})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["price", "quantity", "buyer_maker"])
    return frame.set_index("timestamp").sort_index()


def _build_profiles_from_bars(bars):
    profiles = []
    cursor = bars.index.min().to_pydatetime()
    end = bars.index.max().to_pydatetime() + timedelta(minutes=1)
    while cursor + WINDOW_SIZE <= end:
        window_end = cursor + WINDOW_SIZE
        window = bars.loc[(bars.index >= cursor) & (bars.index < window_end)]
        if len(window) >= 30:
            profile = build_trade_volume_profile(
                _bars_to_profile_trades(window),
                tick_size=0.01,
                profile_bin_ticks=PROFILE_BIN_TICKS,
                node_smoothing_bins=3,
                node_prominence=0.25,
                node_min_separation_bins=3,
                node_min_relative_volume=1.0,
            )
            a, b = pd.Timestamp(cursor), pd.Timestamp(window_end)
            a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
            b = b.tz_localize("UTC") if b.tzinfo is None else b.tz_convert("UTC")
            profiles.append((a, b, profile))
        cursor += STEP
    return profiles


def _build_long_history_zones(profiles, node_type):
    nodes = []
    for _, _, profile in profiles:
        nodes.extend(profile.hvn_nodes if node_type == "HVN" else profile.lvn_nodes)
    clusters = _cluster_nodes(nodes)
    zones = _consolidate_zones(clusters)
    total = len(profiles)
    result = []
    for zone in zones:
        matches = _zone_coverage(zone, profiles, node_type)
        coverage = len(matches) / max(1, total)
        if coverage < 0.05:
            continue
        recent = bool(matches and matches[-1] >= max(0, total - 4))
        if coverage >= 0.75 and recent:
            status = "HIGH_ACTIVE"
        elif coverage >= 0.50 and recent:
            status = "MEDIUM_ACTIVE"
        elif coverage >= 0.50:
            status = "HISTORICAL"
        elif recent:
            status = "DEVELOPING"
        else:
            status = "LOW"
        result.append({"node_type": node_type, "low": float(zone["low"]), "high": float(zone["high"]), "center": float(zone["center"]), "coverage": coverage, "status": status})
    return result


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


def _find_events(bars, zones):
    """Create one lifecycle at a time per zone.

    A lifecycle can contain one confirmed BREAKOUT and at most one RETEST.
    A TOUCH/SWEEP/REJECTION lifecycle is closed after its directional event.
    A cooldown prevents immediate re-counting of the same interaction.
    ACCEPTANCE is stored separately and resolved independently.
    """
    events = []
    acceptances = []

    for z in zones:
        low, high = z["low"], z["high"]
        state = "IDLE"
        inside_streak = 0
        outside_up = outside_down = 0
        bars_since_touch = None
        broken_direction = None
        cooldown = 0

        for ts, row in bars.iterrows():
            bar_low = float(row["low"])
            bar_high = float(row["high"])
            close = float(row["close"])
            touched = _touches(low, high, bar_low, bar_high)
            inside = _inside(low, high, close)
            above = close > high
            below = close < low

            if cooldown:
                cooldown -= 1
                if touched:
                    cooldown = LIFECYCLE_COOLDOWN_BARS
                continue

            # After a confirmed breakout, only the first retest belongs to it.
            if state == "BROKEN":
                if touched:
                    events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "RETEST", broken_direction, close))
                    state = "COOLDOWN"
                    cooldown = LIFECYCLE_COOLDOWN_BARS
                    broken_direction = None
                continue

            if state == "COOLDOWN":
                cooldown = LIFECYCLE_COOLDOWN_BARS
                continue

            # Confirmed breakout: do not emit a new breakout on every bar.
            if above:
                outside_up += 1
                outside_down = 0
            elif below:
                outside_down += 1
                outside_up = 0
            else:
                outside_up = outside_down = 0

            if outside_up >= BREAKOUT_BARS or outside_down >= BREAKOUT_BARS:
                direction = "UP" if outside_up >= BREAKOUT_BARS else "DOWN"
                events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "BREAKOUT", direction, close))
                state = "BROKEN"
                broken_direction = direction
                inside_streak = 0
                bars_since_touch = None
                continue

            if touched:
                bars_since_touch = 0
                if inside:
                    inside_streak += 1
                    if inside_streak >= ACCEPTANCE_BARS and state != "ACCEPTED":
                        acceptances.append(Acceptance(ts, z["node_type"], low, high, z["center"], z["status"], close))
                        state = "ACCEPTED"
                else:
                    inside_streak = 0
                    if state == "IDLE":
                        state = "TOUCHED"
                continue

            if bars_since_touch is not None:
                bars_since_touch += 1

            # One rejection per lifecycle. This fixes the old behaviour where
            # every bar after a touch could become another rejection.
            if state in {"IDLE", "TOUCHED"} and bars_since_touch is not None and bars_since_touch <= MAX_REJECTION_BARS and (above or below):
                events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "REJECTION", _direction("REJECTION", close, low, high), close))
                state = "COOLDOWN"
                cooldown = LIFECYCLE_COOLDOWN_BARS
                bars_since_touch = None
                continue

            # Acceptance lifecycle ends on the first confirmed close outside.
            if state == "ACCEPTED" and (above or below):
                state = "COOLDOWN"
                cooldown = LIFECYCLE_COOLDOWN_BARS
                bars_since_touch = None
                continue

            if bars_since_touch is not None and bars_since_touch > MAX_REJECTION_BARS:
                state = "COOLDOWN"
                cooldown = LIFECYCLE_COOLDOWN_BARS
                bars_since_touch = None

    return sorted(events, key=lambda x: x.timestamp), sorted(acceptances, key=lambda x: x.timestamp)


def _position(bars, timestamp):
    try:
        pos = bars.index.get_loc(timestamp)
    except KeyError:
        return None
    return pos if isinstance(pos, int) else None


def _future_metrics(bars, event, horizon):
    pos = _position(bars, event.timestamp)
    if pos is None or pos + horizon >= len(bars) or event.direction is None:
        return None
    future = bars.iloc[pos + 1:pos + horizon + 1]
    final_close = float(future["close"].iloc[-1])
    raw = final_close - event.entry
    if event.direction == "DOWN":
        return -raw, max(0.0, event.entry - float(future["low"].min())), max(0.0, float(future["high"].max()) - event.entry)
    return raw, max(0.0, float(future["high"].max()) - event.entry), max(0.0, event.entry - float(future["low"].min()))


def _baseline(bars, events, horizon):
    event_positions = {_position(bars, e.timestamp) for e in events}
    event_positions.discard(None)
    blocked = set()
    for pos in event_positions:
        blocked.update(range(max(0, pos - horizon + 1), min(len(bars), pos + horizon)))
    values = {"UP": [], "DOWN": []}
    for pos in range(len(bars) - horizon):
        if pos in blocked:
            continue
        raw = float(bars["close"].iloc[pos + horizon]) - float(bars["close"].iloc[pos])
        values["UP"].append(raw)
        values["DOWN"].append(-raw)
    return {d: (float(np.mean(v)) if v else None) for d, v in values.items()}


def _outcome(mfe, mae):
    threshold = max(1.0, 0.25 * (mfe + mae))
    if mfe - mae > threshold:
        return "FAVORABLE"
    if mae - mfe > threshold:
        return "ADVERSE"
    return "MIXED"


def _acceptance_resolutions(bars, acceptances):
    """Classify first confirmed resolution after acceptance for each horizon."""
    results = []
    for acceptance in acceptances:
        pos = _position(bars, acceptance.timestamp)
        if pos is None:
            continue
        max_h = min(ACCEPTANCE_RESOLUTION_HORIZON, len(bars) - pos - 1)
        if max_h <= 0:
            continue
        future = bars.iloc[pos + 1:pos + max_h + 1]
        first_dir = None
        first_pos = None
        for offset, (_, row) in enumerate(future.iterrows(), start=1):
            close = float(row["close"])
            if close > acceptance.high:
                first_dir, first_pos = "UP", offset
                break
            if close < acceptance.low:
                first_dir, first_pos = "DOWN", offset
                break

        for horizon in HORIZONS:
            if horizon > max_h:
                continue
            window = bars.iloc[pos + 1:pos + horizon + 1]
            final_close = float(window["close"].iloc[-1])
            move = final_close - acceptance.entry
            if first_dir is None or first_pos > horizon:
                resolution = "HELD"
                ttr = None
            elif first_dir == "UP":
                resolution = "RESOLVES_UP"
                ttr = first_pos
                if final_close <= acceptance.high:
                    resolution = "FAILED_UP"
            else:
                resolution = "RESOLVES_DOWN"
                ttr = first_pos
                if final_close >= acceptance.low:
                    resolution = "FAILED_DOWN"
            results.append(AcceptanceResolution(acceptance, horizon, resolution, move, ttr))
    return results


def _print_acceptance_summary(results):
    print("\n=== ACCEPTANCE RESOLUTION ===")
    print("horizon | type | n | up | down | failed | held | avg move | avg time")
    print("--------|------|---|----|------|--------|------|----------|---------")
    for horizon in HORIZONS:
        for node_type in ("ALL", "HVN", "LVN"):
            subset = [r for r in results if r.horizon == horizon and (node_type == "ALL" or r.acceptance.node_type == node_type)]
            if not subset:
                continue
            n = len(subset)
            up = sum(r.resolution == "RESOLVES_UP" for r in subset)
            down = sum(r.resolution == "RESOLVES_DOWN" for r in subset)
            failed = sum(r.resolution in {"FAILED_UP", "FAILED_DOWN"} for r in subset)
            held = sum(r.resolution == "HELD" for r in subset)
            moves = [r.move for r in subset]
            times = [r.time_to_resolution for r in subset if r.time_to_resolution is not None]
            avg_time = np.mean(times) if times else float("nan")
            print(f"{horizon:>7} | {node_type:<4} | {n:>3} | {100*up/n:>4.1f}% | {100*down/n:>5.1f}% | {100*failed/n:>6.1f}% | {100*held/n:>4.1f}% | {np.mean(moves):>+8.2f} | {avg_time:>7.1f}m")


def _print_independence_summary(events, acceptances):
    counts = {name: sum(e.event == name for e in events) for name in ("BREAKOUT", "RETEST", "SWEEP", "REJECTION")}
    print("\n=== INDEPENDENCE / LIFECYCLE CHECK ===")
    print(f"directional_events={len(events)}")
    print(f"acceptance_events={len(acceptances)}")
    print(" | ".join(f"{k.lower()}={v}" for k, v in counts.items()))
    print(f"lifecycle_cooldown_bars={LIFECYCLE_COOLDOWN_BARS}")
    print("A zone lifecycle can produce one confirmed breakout and at most one retest before cooldown.")
    print("Acceptance is tracked separately and resolved at 5m/15m/30m/60m.")


def main():
    parser = argparse.ArgumentParser(description="Long-history BTCUSDT zone edge validation")
    parser.add_argument("--days", type=int, default=7, choices=(7, 30, 90))
    args = parser.parse_args()

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT", page_limit=1000))
    bars = _load_klines(provider, start, end)
    if bars.empty:
        raise RuntimeError("No Binance kline data returned")

    profiles = _build_profiles_from_bars(bars)
    zones = _build_long_history_zones(profiles, "HVN") + _build_long_history_zones(profiles, "LVN")
    events, acceptances = _find_events(bars, zones)
    acceptance_results = _acceptance_resolutions(bars, acceptances)

    print("=== BTCUSDT HISTORICAL ZONE EDGE TEST ===")
    print(f"requested_days={args.days}")
    print(f"dataset={bars.index.min()} -> {bars.index.max()}")
    print(f"bars={len(bars)}")
    print(f"profile_windows={len(profiles)}")
    print(f"zones={len(zones)}")
    print(f"events={len(events)}")
    print(f"acceptances={len(acceptances)}")
    print("profile_source=1m OHLCV volume approximation")
    print("NOTE: multi-day profiles are approximate; use zone_edge_test.py for trade-level 4h validation.")

    for horizon in HORIZONS:
        baseline = _baseline(bars, events, horizon)
        rows = []
        for event in events:
            metrics = _future_metrics(bars, event, horizon)
            if metrics is None or baseline.get(event.direction) is None:
                continue
            move, mfe, mae = metrics
            rows.append((event, move, baseline[event.direction], mfe, mae, _outcome(mfe, mae)))

        print(f"\n=== {horizon}M EDGE ===")
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
            print(f"{event_type:<8}| {len(subset):>4} | {fav:>9.1f}% | {adv:>7.1f}% | {avg_dir:>8.2f} | {avg_base:>13.2f} | {avg_dir-avg_base:+.2f}")

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

    _print_acceptance_summary(acceptance_results)
    _print_independence_summary(events, acceptances)
    print("\nInterpretation: positive edge means event directional movement exceeded the independent non-event BTCUSDT baseline.")
    print("Acceptance is analyzed separately as zone resolution, not forced into directional favorable/adverse scoring.")
    print("Directional events use one lifecycle plus cooldown to reduce repeated observations of the same interaction.")
    print("This is historical validation, not a trading signal.")


if __name__ == "__main__":
    main()
