"""Test structural volume-profile persistence across rolling time windows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.data.dataset import load_aligned_binance_dataset
from trading_engine.volume_profile.trade_profile import build_trade_volume_profile


def _cluster_nodes(nodes, tolerance: float = 1.0):
    clusters = []
    for node in sorted(nodes, key=lambda n: n.center):
        target = None
        for cluster in clusters:
            if abs(node.center - cluster["center"]) <= tolerance:
                target = cluster
                break
        if target is None:
            target = {"center": node.center, "nodes": []}
            clusters.append(target)
        target["nodes"].append(node)
        target["center"] = sum(n.center for n in target["nodes"]) / len(target["nodes"])
    return clusters


def main() -> None:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    # Fetch enough data for overlapping 60-minute profiles stepped every 15 minutes.
    start = end - timedelta(hours=4)
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT"))
    dataset = load_aligned_binance_dataset(
        provider, interval="1m", start=start, end=end, bar_limit=1000
    )

    windows = []
    window_size = timedelta(hours=1)
    step = timedelta(minutes=15)
    cursor = dataset.start
    while cursor + window_size <= dataset.end:
        window_end = cursor + window_size
        trades = dataset.trades.loc[(dataset.trades.index >= cursor) & (dataset.trades.index < window_end)]
        if len(trades) >= 100:
            windows.append((cursor, window_end, trades))
        cursor += step

    profiles = []
    for window_start, window_end, trades in windows:
        profile = build_trade_volume_profile(
            trades,
            tick_size=0.01,
            profile_bin_ticks=25,
            node_smoothing_bins=3,
            node_prominence=0.25,
            node_min_separation_bins=3,
            node_min_relative_volume=1.0,
        )
        profiles.append((window_start, window_end, profile))

    print("=== BTCUSDT TEMPORAL PROFILE TEST ===")
    print(f"dataset={dataset.start} -> {dataset.end}")
    print(f"windows={len(profiles)}")
    print("\nwindow_end            POC       VAH       VAL       HVNs  LVNs")
    print("--------------------  --------  --------  --------  ----  ----")
    for window_start, window_end, profile in profiles:
        print(
            f"{window_end.isoformat():<20}  {profile.poc:>8.2f}  {profile.vah:>8.2f}  {profile.val:>8.2f}  "
            f"{len(profile.hvn_nodes):>4}  {len(profile.lvn_nodes):>4}"
        )

    for node_type in ("HVN", "LVN"):
        all_nodes = []
        for _, _, profile in profiles:
            nodes = profile.hvn_nodes if node_type == "HVN" else profile.lvn_nodes
            all_nodes.extend(nodes)
        clusters = _cluster_nodes(all_nodes, tolerance=1.0)
        print(f"\n=== TEMPORALLY PERSISTENT {node_type} ZONES ===")
        persistent = []
        for cluster in clusters:
            window_ids = set()
            for i, (_, _, profile) in enumerate(profiles):
                nodes = profile.hvn_nodes if node_type == "HVN" else profile.lvn_nodes
                if any(abs(node.center - cluster["center"]) <= 1.0 for node in nodes):
                    window_ids.add(i)
            coverage = len(window_ids) / max(1, len(profiles))
            if len(window_ids) >= 2:
                persistent.append((coverage, cluster["center"], len(window_ids)))

        for coverage, center, count in sorted(persistent, reverse=True):
            strength = "HIGH" if coverage >= 0.75 else "MEDIUM"
            print(
                f"center≈{center:.2f} | windows={count}/{len(profiles)} | "
                f"coverage={coverage:.0%} | {strength}"
            )


if __name__ == "__main__":
    main()
