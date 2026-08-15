"""Compare the same trade profile across multiple price resolutions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.data.dataset import load_aligned_binance_dataset
from trading_engine.volume_profile.trade_profile import build_trade_volume_profile


def _stable_zones(profiles, node_type: str):
    """Cluster nearby nodes so each structural zone is reported once."""
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
        for cluster in reversed(clusters):
            if center - cluster["high"] <= max(tolerance, cluster["tolerance"]):
                target = cluster
                break
        if target is None:
            target = {"items": [], "low": center, "high": center, "tolerance": tolerance}
            clusters.append(target)
        target["items"].append((center, ticks, node))
        target["low"] = min(target["low"], node.low)
        target["high"] = max(target["high"], node.high)
        target["tolerance"] = max(target["tolerance"], tolerance)

    stable = []
    resolution_count = len(profiles)
    for cluster in clusters:
        by_resolution = {}
        for center, ticks, node in cluster["items"]:
            by_resolution[ticks] = node
        count = len(by_resolution)
        if count >= 2:
            centers = [node.center for node in by_resolution.values()]
            avg_center = sum(centers) / len(centers)
            strengths = [node.prominence for node in by_resolution.values()]
            stable.append({
                "low": cluster["low"],
                "high": cluster["high"],
                "center": avg_center,
                "resolutions": count,
                "coverage": count / resolution_count,
                "mean_prominence": sum(strengths) / len(strengths),
                "max_prominence": max(strengths),
            })
    return stable


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
        zones.sort(key=lambda z: (-z["coverage"], -z["mean_prominence"], z["center"]))
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
