"""Compare the same trade profile across multiple price resolutions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from trading_engine.data.binance import BinanceConfig, BinancePublicData
from trading_engine.volume_profile.trade_profile import build_trade_volume_profile


def main() -> None:
    config = BinanceConfig(symbol="BTCUSDT", interval="1m", limit=61)
    provider = BinancePublicData(config)
    bars = provider.fetch_klines()
    trades = provider.fetch_agg_trades(start_time=bars.index[0], end_time=bars.index[-1])

    resolutions = (10, 25, 50, 100)
    profiles = {}
    print("=== BTCUSDT PROFILE RESOLUTION TEST ===")
    print(f"window={bars.index[0]} -> {bars.index[-1]}")
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

    print("\n=== PERSISTENT HVN CENTERS ===")
    hvn_centers = []
    for profile in profiles.values():
        hvn_centers.extend(n.center for n in profile.hvn_nodes)
    for center in sorted(set(round(x, 2) for x in hvn_centers)):
        count = sum(
            any(abs(n.center - center) <= max(0.5, ticks * 0.01 * 1.5) for n in profile.hvn_nodes)
            for ticks, profile in profiles.items()
        )
        if count >= 2:
            print(f"center≈{center:.2f}, resolutions={count}/{len(profiles)}")

    print("\n=== PERSISTENT LVN CENTERS ===")
    lvn_centers = []
    for profile in profiles.values():
        lvn_centers.extend(n.center for n in profile.lvn_nodes)
    for center in sorted(set(round(x, 2) for x in lvn_centers)):
        count = sum(
            any(abs(n.center - center) <= max(0.5, ticks * 0.01 * 1.5) for n in profile.lvn_nodes)
            for ticks, profile in profiles.items()
        )
        if count >= 2:
            print(f"center≈{center:.2f}, resolutions={count}/{len(profiles)}")


if __name__ == "__main__":
    main()
