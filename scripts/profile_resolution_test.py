"""Compare the same trade profile across multiple price resolutions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.data.dataset import load_aligned_binance_dataset
from trading_engine.volume_profile.trade_profile import build_trade_volume_profile


def _stable_zones(profiles, node_type: str):
    """Cluster cross-resolution nodes without transitive chaining."""
    items = []
    for ticks, profile in profiles.items():
        nodes = profile.hvn_nodes if node_type == "HVN" else profile.lvn_nodes
        for node in nodes:
            items.append((node.center, ticks, node))
    items.sort(key=lambda item: item[0])

    clusters = []
    for center, ticks, node in items:
        tolerance = max(0.5, ticks * 0.01 * 1.5)
        target = None
        for cluster in clusters:
            if abs(center - cluster["anchor"]) <= max(tolerance, cluster["anchor_tolerance"]):
                target = cluster
                break
        if target is None:
            target = {
                "items": [],
                "anchor": center,
                "anchor_tolerance": tolerance,
            }
            clusters.append(target)
        target["items"].append((center, ticks, node))

    stable = []
    resolution_count = len(profiles)
    for cluster in clusters:
        by_resolution = {}
        for center, ticks, node in cluster["items"]:
            existing = by_resolution.get(ticks)
            if existing is None or node.prominence > existing.prominence:
                by_resolution[ticks] = node
        count = len(by_resolution)
        if count < 2:
            continue
        centers = [node.center for node in by_resolution.values()]
        avg_center = sum(centers) / len(centers)
        half_widths = [node.high - node.low for node in by_resolution.values()]
        strengths = [node.prominence for node in by_resolution.values()]
        stable.append({
            "low": avg_center - max(half_widths) / 2,
            "high": avg_center + max(half_widths) / 2,
            "center": avg_center,
            "resolutions": count,
            "coverage": count / resolution_count,
            "mean_prominence": sum(strengths) / len(strengths),
            "max_prominence": max(strengths),
        })

    # Remove overlapping duplicate clusters, retaining the one with stronger
    # cross-resolution coverage/prominence.
    stable.sort(key=lambda z: (-z["coverage"], -z["mean_prominence"], z["center"]))
    selected = []
    for zone in stable:
        if any(zone["low"] <= other["high"] and zone["high"] >= other["low"] for other in selected):
            continue
        selected.append(zone)
    return sorted(selected, key=lambda z: z["center"])


def main() -> None:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(minutes=60)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT"))
    dataset = load_aligned_binance_dataset(
        provider, interval="1m", start=start, end=end, bar_limit=1000
    )
    trades = dataset.trades

    resolutions = (10, 25, 50, 100)
    profiles = {}
    print("=== BTCUSDT PROFILE RESOLUTION TEST ===")
    print(f"window={dataset.start} -> {dataset.end}")
    print(f"bars={len(dataset.bars)}")
    print(f"trades={len(trades)}")
    print()
    print("resolution  bin_size  POC       VAH       VAL       HVNs  LVNs")
    print("----------  --------  --------  --------  --------  ----  ----")

    for ticks in resolutions:
        profile = build_trade_volume_profile(
            trades,
            tick_size=0.01,
            profile_bin_ticks=ticks,
            node_smoothing_bins=3,
            node_prominence=0.25,
            node_min_separation_bins=3,
            node_min_relative_volume=1.0,
        )
        profiles[ticks] = profile
        print(
            f"{ticks:>10}  ${ticks * 0.01:<7.2f}  "
            f"{profile.poc:>8.2f}  {profile.vah:>8.2f}  {profile.val:>8.2f}  "
            f"{len(profile.hvn_nodes):>4}  {len(profile.lvn_nodes):>4}"
        )

    print("\n=== NODES BY RESOLUTION ===")
    for ticks, profile in profiles.items():
        print(f"\n${ticks * 0.01:.2f} bins")
        print("  HVN:", ", ".join(f"{n.center:.2f}" for n in profile.hvn_nodes) or "none")
        print("  LVN:", ", ".join(f"{n.center:.2f}" for n in profile.lvn_nodes) or "none")

    for node_type in ("HVN", "LVN"):
        print(f"\n=== STABLE {node_type} ZONES ===")
        zones = _stable_zones(profiles, node_type)
        for zone in zones:
            strength = "HIGH" if zone["coverage"] == 1.0 else "MEDIUM"
            print(
                f"{zone['low']:.2f} -> {zone['high']:.2f} | "
                f"center≈{zone['center']:.2f} | "
                f"resolutions={zone['resolutions']}/{len(profiles)} | "
                f"mean_prominence={zone['mean_prominence']:.2f} | {strength}"
            )


if __name__ == "__main__":
    main()
