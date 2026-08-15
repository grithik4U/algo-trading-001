"""Inspect live Binance public data and validate normalized schemas."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_engine.data.binance import BinanceConfig, BinancePublicData


def main() -> None:
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT"))
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=1000)

    bars = provider.get_klines("1m", start_time=start, end_time=end, limit=1000)
    trades = provider.get_agg_trades(start_time=start, end_time=end, limit=1000)

    print("=== BINANCE SAMPLE ===")
    print(f"symbol=BTCUSDT")
    print(f"bars={len(bars)}")
    print(f"trades={len(trades)}")
    print(f"bars_monotonic={bars.index.is_monotonic_increasing}")
    print(f"trades_monotonic={trades.index.is_monotonic_increasing}")
    print(f"bars_columns={list(bars.columns)}")
    print(f"trades_columns={list(trades.columns)}")
    if not bars.empty:
        print(f"bar_range={bars.index.min()} -> {bars.index.max()}")
        print(bars.tail(3).to_string())
    if not trades.empty:
        print(f"trade_range={trades.index.min()} -> {trades.index.max()}")
        print(trades.tail(3).to_string())


if __name__ == "__main__":
    main()
