"""Compare the same trade profile across multiple price resolutions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.data.dataset import load_aligned_binance_dataset
from trading_engine.volume_profile.trade_profile import build_trade_volume_profile


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

    def persistent(node_type: str) -> None:
        print(f"\n=== PERSISTENT {node_type} CENTERS ===")
        all_nodes = [
            (ticks, node)
            for ticks, profile in profiles.items()
            for node in (profile.hvn_nodes if node_type == "HVN" else profile.lvn_nodes)
        ]
        for ticks, node in all_nodes:
            count = sum(
                any(
                    abs(other.center - node.center)
                    <= max(0.5, max(ticks, other_ticks) * 0.01 * 1.5)
                    for other in (
                        other_profile.hvn_nodes
                        if node_type == "HVN"
                        else other_profile.lvn_nodes
                    )
                )
                for other_ticks, other_profile in profiles.items()
                if other_ticks != ticks
            ) + 1
            if count >= 2:
                print(f"center≈{node.center:.2f}, resolutions={count}/{len(profiles)}")

    persistent("HVN")
    persistent("LVN")


if __name__ == "__main__":
    main()
