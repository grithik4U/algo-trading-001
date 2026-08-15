"""Binance public market-data adapter.

No API key is required for these public endpoints. The adapter is deliberately
small: it normalizes Binance candles and aggregate trades into the schemas
used by the research engine, while leaving strategy logic provider-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://data-api.binance.vision"


@dataclass(frozen=True)
class BinanceConfig:
    symbol: str = "BTCUSDT"
    timeout_seconds: int = 20


class BinancePublicData:
    def __init__(self, config: BinanceConfig | None = None) -> None:
        self.config = config or BinanceConfig()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        response = requests.get(
            f"{BASE_URL}{path}", params=params, timeout=self.config.timeout_seconds
        )
        response.raise_for_status()
        return response.json()

    def get_klines(
        self,
        interval: str = "1m",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        params: dict[str, Any] = {"symbol": self.config.symbol, "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.timestamp() * 1000)
        rows = self._get("/api/v3/klines", params)
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trade_count", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        numeric = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]
        frame[numeric] = frame[numeric].astype(float)
        frame["trade_count"] = frame["trade_count"].astype(int)
        return frame.set_index("timestamp").sort_index()[["open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume"]]

    def get_agg_trades(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        params: dict[str, Any] = {"symbol": self.config.symbol, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.timestamp() * 1000)
        rows = self._get("/api/v3/aggTrades", params)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame = frame.rename(columns={"a": "agg_trade_id", "p": "price", "q": "quantity", "f": "first_trade_id", "l": "last_trade_id", "T": "timestamp", "m": "buyer_maker", "M": "best_match"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame[["price", "quantity"]] = frame[["price", "quantity"]].astype(float)
        return frame.set_index("timestamp").sort_index()


def to_utc_milliseconds(value: datetime) -> int:
    """Convert an aware datetime to Unix milliseconds."""
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return int(value.astimezone(timezone.utc).timestamp() * 1000)
