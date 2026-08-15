"""Aligned historical market dataset assembled from provider data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .binance import BinancePublicData


@dataclass(frozen=True)
class MarketDataSet:
    symbol: str
    interval: str
    start: datetime
    end: datetime
    bars: pd.DataFrame
    trades: pd.DataFrame


def load_aligned_binance_dataset(
    provider: BinancePublicData,
    *,
    interval: str,
    start: datetime,
    end: datetime,
    bar_limit: int = 1000,
    trade_limit: int = 1000,
) -> MarketDataSet:
    """Load bars and trades for the same requested UTC interval.

    The provider may paginate/limit each endpoint independently. This function
    validates the returned timestamps and refuses a dataset whose returned
    range falls outside the requested interval.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    if end <= start:
        raise ValueError("end must be after start")

    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    bars = provider.get_klines(interval, start_time=start, end_time=end, limit=bar_limit)
    trades = provider.get_agg_trades(start_time=start, end_time=end, limit=trade_limit)

    for frame, name in ((bars, "bars"), (trades, "trades")):
        if not frame.empty:
            if frame.index.min() < pd.Timestamp(start) or frame.index.max() > pd.Timestamp(end):
                raise ValueError(f"{name} contains timestamps outside requested interval")
            if not frame.index.is_monotonic_increasing:
                raise ValueError(f"{name} timestamps are not monotonic")

    return MarketDataSet(
        symbol=provider.config.symbol,
        interval=interval,
        start=start,
        end=end,
        bars=bars,
        trades=trades,
    )
