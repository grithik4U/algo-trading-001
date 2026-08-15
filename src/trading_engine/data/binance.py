"""Binance public market-data adapter with chronological pagination."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://data-api.binance.vision"


@dataclass(frozen=True)
class BinanceConfig:
    symbol: str = "BTCUSDT"
    timeout_seconds: int = 20
    page_limit: int = 1000


class BinancePublicData:
    def __init__(self, config: BinanceConfig | None = None) -> None:
        self.config = config or BinanceConfig()
        if not 1 <= self.config.page_limit <= 1000:
            raise ValueError("page_limit must be between 1 and 1000")

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        response = requests.get(f"{BASE_URL}{path}", params=params, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _validate_window(start_time: datetime | None, end_time: datetime | None) -> None:
        if start_time is not None and start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware")
        if end_time is not None and end_time.tzinfo is None:
            raise ValueError("end_time must be timezone-aware")
        if start_time is not None and end_time is not None and end_time <= start_time:
            raise ValueError("end_time must be after start_time")

    def get_klines(self, interval: str = "1m", start_time: datetime | None = None, end_time: datetime | None = None, limit: int = 1000) -> pd.DataFrame:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        self._validate_window(start_time, end_time)
        params: dict[str, Any] = {"symbol": self.config.symbol, "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time.astimezone(timezone.utc).timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.astimezone(timezone.utc).timestamp() * 1000)
        rows = self._get("/api/v3/klines", params)
        columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        numeric = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]
        frame[numeric] = frame[numeric].astype(float)
        frame["trade_count"] = frame["trade_count"].astype(int)
        return frame.set_index("timestamp").sort_index()[["open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume"]]

    def get_agg_trades(self, start_time: datetime | None = None, end_time: datetime | None = None, limit: int = 1000) -> pd.DataFrame:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        self._validate_window(start_time, end_time)
        params: dict[str, Any] = {"symbol": self.config.symbol, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time.astimezone(timezone.utc).timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.astimezone(timezone.utc).timestamp() * 1000)
        rows = self._get("/api/v3/aggTrades", params)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame = frame.rename(columns={"a": "agg_trade_id", "p": "price", "q": "quantity", "f": "first_trade_id", "l": "last_trade_id", "T": "timestamp", "m": "buyer_maker", "M": "best_match"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame[["price", "quantity"]] = frame[["price", "quantity"]].astype(float)
        return frame.set_index("timestamp").sort_index()

    def get_agg_trades_window(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Page aggregate trades chronologically until the requested window is exhausted."""
        self._validate_window(start_time, end_time)
        start = start_time.astimezone(timezone.utc)
        end = end_time.astimezone(timezone.utc)
        cursor = start
        pages: list[pd.DataFrame] = []

        while cursor < end:
            frame = self.get_agg_trades(start_time=cursor, end_time=end, limit=self.config.page_limit)
            if frame.empty:
                break
            pages.append(frame)
            last_ts = frame.index.max().to_pydatetime()
            if last_ts >= end:
                break
            if len(frame) < self.config.page_limit:
                break
            next_cursor = last_ts + timedelta(milliseconds=1)
            if next_cursor <= cursor:
                raise RuntimeError("Binance pagination cursor did not advance")
            cursor = next_cursor

        if not pages:
            return pd.DataFrame(columns=["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "buyer_maker", "best_match"])
        result = pd.concat(pages)
        result = result[~result.index.duplicated(keep="first")].sort_index()
        return result[(result.index >= pd.Timestamp(start)) & (result.index <= pd.Timestamp(end))]


def to_utc_milliseconds(value: datetime) -> int:
    """Convert an aware datetime to Unix milliseconds."""
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return int(value.astimezone(timezone.utc).timestamp() * 1000)
