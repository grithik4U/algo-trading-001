import pandas as pd

from trading_engine.data.binance import BinanceConfig, BinancePublicData


def test_klines_normalization(monkeypatch):
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT"))
    rows = [[
        1760000000000, "100.0", "101.0", "99.0", "100.5", "12.0",
        1760000059999, "1200.0", 42, "6.0", "600.0", "0"
    ]]
    monkeypatch.setattr(provider, "_get", lambda path, params: rows)

    frame = provider.get_klines()

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is not None
    assert frame.iloc[0]["close"] == 100.5
    assert frame.iloc[0]["trade_count"] == 42


def test_aggregate_trade_normalization(monkeypatch):
    provider = BinancePublicData(BinanceConfig(symbol="BTCUSDT"))
    rows = [{"a": 1, "p": "100.25", "q": "0.5", "f": 1, "l": 1, "T": 1760000000000, "m": True, "M": True}]
    monkeypatch.setattr(provider, "_get", lambda path, params: rows)

    frame = provider.get_agg_trades()

    assert frame.iloc[0]["price"] == 100.25
    assert frame.iloc[0]["quantity"] == 0.5
    assert frame.index.tz is not None
