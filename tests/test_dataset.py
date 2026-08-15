from datetime import datetime, timezone

import pandas as pd
import pytest

from trading_engine.data.dataset import load_aligned_binance_dataset


class FakeConfig:
    symbol = "BTCUSDT"


class FakeProvider:
    config = FakeConfig()

    def __init__(self, bars, trades):
        self.bars = bars
        self.trades = trades

    def get_klines(self, interval, start_time, end_time, limit):
        return self.bars

    def get_agg_trades(self, start_time, end_time, limit):
        return self.trades


def test_dataset_requires_timezone_aware_bounds():
    with pytest.raises(ValueError):
        load_aligned_binance_dataset(
            FakeProvider(pd.DataFrame(), pd.DataFrame()),
            interval="1m",
            start=datetime(2026, 1, 1),
            end=datetime(2026, 1, 1, 0, 1),
        )


def test_dataset_rejects_data_outside_requested_window():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    bars = pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(start), pd.Timestamp(end)]))
    trades = pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(start)]))
    # Provider output is deliberately outside the requested end bound.
    bars = pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(minutes=1)]))
    with pytest.raises(ValueError):
        load_aligned_binance_dataset(FakeProvider(bars, trades), interval="1m", start=start, end=end)
