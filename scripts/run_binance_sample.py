"""Fetch a small real BTCUSDT sample through the Binance provider.

Usage:
    python scripts/run_binance_sample.py --interval 1m --limit 1000
"""

from __future__ import annotations

import argparse

from trading_engine.data.binance import BinanceConfig, BinancePublicData


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    provider = BinancePublicData(BinanceConfig(symbol=args.symbol))
    bars = provider.get_klines(args.interval, limit=args.limit)
    trades = provider.get_agg_trades(limit=min(args.limit, 1000))

    print(f"bars={len(bars)}")
    print(f"trades={len(trades)}")
    if not bars.empty:
        print(f"bar_range={bars.index.min()} -> {bars.index.max()}")
        print(bars.tail(3).to_string())
    if not trades.empty:
        print(f"trade_range={trades.index.min()} -> {trades.index.max()}")
        print(trades.tail(3).to_string())


if __name__ == "__main__":
    main()
