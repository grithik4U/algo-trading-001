"""Build and inspect a real BTCUSDT trade-level volume profile."""

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

    profile = build_trade_volume_profile(dataset.trades, tick_size=0.01)

    print("=== BTCUSDT TRADE VOLUME PROFILE ===")
    print(f"window={dataset.start} -> {dataset.end}")
    print(f"bars={len(dataset.bars)}")
    print(f"trades={len(dataset.trades)}")
    print(f"POC={profile.poc}")
    print(f"VAH={profile.vah}")
    print(f"VAL={profile.val}")
    print(f"HVNs={profile.hvns}")
    print(f"LVNs={profile.lvns}")
    print(f"total_volume={profile.volume_at_price.sum():.8f}")
    print(f"total_delta={profile.delta_at_price.sum():.8f}")


if __name__ == "__main__":
    main()
