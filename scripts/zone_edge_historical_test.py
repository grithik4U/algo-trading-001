"""Longer-history zone-edge validation using paginated 1m OHLCV data.

Research/diagnostic only. Does not generate trade signals.

For multi-day runs this script uses a deterministic 1m OHLCV volume-profile
approximation rather than downloading millions of Binance aggregate trades.
The existing zone_edge_test.py remains the trade-level 4-hour diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.volume_profile.trade_profile import build_trade_volume_profile
from zone_interaction_test import ACCEPTANCE_BARS, BREAKOUT_BARS, MAX_REJECTION_BARS, _zone_coverage, _cluster_nodes, _consolidate_zones

WINDOW_SIZE = timedelta(hours=1)
STEP = timedelta(minutes=15)
PROFILE_BIN_TICKS = 25
HORIZONS = (5, 15, 30, 60)
MIN_EDGE_SAMPLE = 20

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

def _load_klines(provider: BinancePublicData, start: datetime, end: datetime) -> pd.DataFrame:
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
        next_cursor = last + timedelta(minutes=1)
        if next_cursor <= cursor:
            raise RuntimeError("Binance kline pagination cursor did not advance")
        cursor = next_cursor
    if not pages:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    result = pd.concat(pages)
    result = result[~result.index.duplicated(keep="first")].sort_index()
    return result[(result.index >= pd.Timestamp(start)) & (result.index <= pd.Timestamp(end))]

def _bars_to_profile_trades(bars: pd.DataFrame) -> pd.DataFrame:
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

def _build_profiles_from_bars(bars: pd.DataFrame):
    profiles = []
    cursor = bars.index.min().to_pydatetime()
    end = bars.index.max().to_pydatetime() + timedelta(minutes=1)
    while cursor + WINDOW_SIZE <= end:
        window_end = cursor + WINDOW_SIZE
        window = bars.loc[(bars.index >= cursor) & (bars.index < window_end)]
        if len(window) >= 30:
            trades = _bars_to_profile_trades(window)
            profile = build_trade_volume_profile(
                trades, tick_size=0.01, profile_bin_ticks=PROFILE_BIN_TICKS,
                node_smoothing_bins=3, node_prominence=0.25,
                node_min_separation_bins=3, node_min_relative_volume=1.0,
            )
            # cursor/window_end are already timezone-aware when Binance returns UTC timestamps.
            start_ts = pd.Timestamp(cursor)
            end_ts = pd.Timestamp(window_end)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")
            profiles.append((start_ts, end_ts, profile))
        cursor += STEP
    return profiles

def _build_long_history_zones(profiles, node_type: str):
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
        if coverage < 0.05:
            continue
        recent_cutoff = max(0, total_windows - 4)
        recent = bool(matches and matches[-1] >= recent_cutoff)
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

def _contains(low, high, price):
    return low <= price <= high

def _touches(low, high, bar_low, bar_high):
    return bar_high >= low and bar_low <= high

def _direction(event, close, low, high):
    if event == "BREAKOUT":
        if close > high: return "UP"
        if close < low: return "DOWN"
    if event in {"SWEEP", "REJECTION"}:
        center = (low + high) / 2.0
        if close > center: return "UP"
        if close < center: return "DOWN"
    return None

def _find_events(bars, zones):
    events = []
    for z in zones:
        low, high = z["low"], z["high"]
        inside_streak = outside_streak = 0
        last_touch = None
        broken_direction = None
        last_state = "UNTESTED"
        for i, (ts, row) in enumerate(bars.iterrows()):
            bar_low, bar_high, close = float(row["low"]), float(row["high"]), float(row["close"])
            touched = _touches(low, high, bar_low, bar_high)
            inside = _contains(low, high, close)
            above, below = close > high, close < low
            if touched:
                last_touch = i
                if broken_direction:
                    events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "RETEST", broken_direction, close))
                    broken_direction = None; inside_streak = outside_streak = 0; last_state = "RETEST"; continue
                if bar_high > high and bar_low < low and inside:
                    events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "SWEEP", _direction("SWEEP", close, low, high), close))
                    inside_streak, outside_streak, last_state = 1, 0, "SWEEP"; continue
                if inside:
                    inside_streak += 1; outside_streak = 0
                    last_state = "ACCEPTANCE" if inside_streak >= ACCEPTANCE_BARS else "TOUCH"
                else:
                    inside_streak = outside_streak = 0; last_state = "TOUCH"
                continue
            if last_touch is not None and i - last_touch <= MAX_REJECTION_BARS and last_state in {"TOUCH", "SWEEP"}:
                events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "REJECTION", _direction("REJECTION", close, low, high), close))
                last_state = "REJECTION"
            if above or below:
                outside_streak += 1; inside_streak = 0
                if outside_streak >= BREAKOUT_BARS and last_state not in {"BREAKOUT", "CONFIRMED_BREAKOUT"}:
                    direction = "UP" if above else "DOWN"
                    events.append(Event(ts, z["node_type"], low, high, z["center"], z["status"], "BREAKOUT", direction, close))
                    broken_direction = direction; last_state = "BREAKOUT"
            else:
                outside_streak = 0
    return sorted(events, key=lambda e: e.timestamp)

def _position(bars, timestamp):
    try: pos = bars.index.get_loc(timestamp)
    except KeyError: return None
    return pos if isinstance(pos, int) else None

def _future_metrics(bars, event, horizon):
    pos = _position(bars, event.timestamp)
    if pos is None or pos + horizon >= len(bars) or event.direction is None: return None
    future = bars.iloc[pos + 1:pos + horizon + 1]
    final_close = float(future["close"].iloc[-1]); raw = final_close - event.entry
    if event.direction == "DOWN":
        return -raw, max(0.0, event.entry - float(future["low"].min())), max(0.0, float(future["high"].max()) - event.entry)
    return raw, max(0.0, float(future["high"].max()) - event.entry), max(0.0, event.entry - float(future["low"].min()))

def _baseline(bars, events, horizon):
    event_positions = {_position(bars, e.timestamp) for e in events}; event_positions.discard(None)
    blocked = set()
    for pos in event_positions: blocked.update(range(max(0, pos - horizon + 1), min(len(bars), pos + horizon)))
    values = {"UP": [], "DOWN": []}
    for pos in range(len(bars) - horizon):
        if pos in blocked: continue
        entry = float(bars["close"].iloc[pos]); raw = float(bars["close"].iloc[pos + horizon]) - entry
        values["UP"].append(raw); values["DOWN"].append(-raw)
    return {d: (float(np.mean(v)) if v else None) for d, v in values.items()}

def _outcome(mfe, mae):
    threshold = max(1.0, 0.25 * (mfe + mae))
    if mfe - mae > threshold: return "FAVORABLE"
    if mae - mfe > threshold: return "ADVERSE"
    return "MIXED"

def main():
    parser = argparse.ArgumentParser(description="Long-history BTCUSDT zone edge validation")
    parser.add_argument("--days", type=int, default=7, choices=(7, 30, 90))
    args = parser.parse_args()
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0); start = end - timedelta(days=args.days)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT", page_limit=1000))
    bars = _load_klines(provider, start, end)
    if bars.empty: raise RuntimeError("No Binance kline data returned")
    profiles = _build_profiles_from_bars(bars)
    zones = _build_long_history_zones(profiles, "HVN") + _build_long_history_zones(profiles, "LVN")
    events = _find_events(bars, zones)
    print("=== BTCUSDT HISTORICAL ZONE EDGE TEST ===")
    print(f"requested_days={args.days}"); print(f"dataset={bars.index.min()} -> {bars.index.max()}"); print(f"bars={len(bars)}"); print(f"profile_windows={len(profiles)}"); print(f"zones={len(zones)}"); print(f"events={len(events)}"); print("profile_source=1m OHLCV volume approximation"); print("NOTE: multi-day profiles are approximate; use zone_edge_test.py for trade-level 4h validation.")
    for horizon in HORIZONS:
        base = _baseline(bars, events, horizon); rows=[]
        for event in events:
            metrics = _future_metrics(bars, event, horizon)
            if metrics is None or base.get(event.direction) is None: continue
            move,mfe,mae=metrics; rows.append((event,move,base[event.direction],mfe,mae,_outcome(mfe,mae)))
        print(f"\n=== {horizon}M EDGE ==="); print("event | n | favorable | adverse | avg dir | avg baseline | edge"); print("------|---|-----------|---------|----------|---------------|-----")
        for event_type in ("BREAKOUT","RETEST","SWEEP","REJECTION"):
            subset=[r for r in rows if r[0].event==event_type]
            if not subset: continue
            avg_dir=np.mean([r[1] for r in subset]); avg_base=np.mean([r[2] for r in subset]); fav=np.mean([r[5]=="FAVORABLE" for r in subset])*100; adv=np.mean([r[5]=="ADVERSE" for r in subset])*100
            print(f"{event_type:<8}| {len(subset):>3} | {fav:>9.1f}% | {adv:>7.1f}% | {avg_dir:>8.2f} | {avg_base:>13.2f} | {avg_dir-avg_base:+.2f}")
        for node_type in ("HVN","LVN"):
            for status in ("HIGH_ACTIVE","MEDIUM_ACTIVE","DEVELOPING","LOW","HISTORICAL"):
                subset=[r for r in rows if r[0].node_type==node_type and r[0].status==status]
                if len(subset)<MIN_EDGE_SAMPLE: continue
                avg_dir=np.mean([r[1] for r in subset]); avg_base=np.mean([r[2] for r in subset]); fav=np.mean([r[5]=="FAVORABLE" for r in subset])*100; adv=np.mean([r[5]=="ADVERSE" for r in subset])*100
                print(f"{node_type:<8}| {status:<13} n={len(subset):>4} | fav={fav:>5.1f}% adv={adv:>5.1f}% | avg={avg_dir:+.2f} base={avg_base:+.2f} edge={avg_dir-avg_base:+.2f}")
        if base["UP"] is not None and base["DOWN"] is not None: print(f"baseline anchors: UP={base['UP']:.2f} DOWN={base['DOWN']:.2f}")
        else: print("baseline anchors: insufficient data")
    print("\nInterpretation: positive edge means event directional movement exceeded the independent non-event BTCUSDT baseline.")
    print("Acceptance is excluded from directional scoring. This is historical validation, not a trading signal.")

if __name__ == "__main__": main()
