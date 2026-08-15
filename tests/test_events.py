import pandas as pd

from trading_engine.liquidity.events import detect_pool_events


def test_bsl_sweep_is_classified_after_reclaim():
    index = pd.date_range("2026-01-01", periods=4, freq="min")
    df = pd.DataFrame(
        {
            "high": [99.5, 100.5, 100.4, 100.0],
            "low": [99.0, 99.8, 99.7, 99.2],
            "close": [99.2, 100.2, 99.8, 99.5],
        },
        index=index,
    )
    pools = pd.DataFrame({"pool_id": [1], "side": ["BSL"], "price": [100.0]})

    events = detect_pool_events(df, pools, max_bars=3)

    assert len(events) == 1
    assert events.iloc[0]["event_type"] == "sweep"
    assert events.iloc[0]["event_timestamp"] == index[2]


def test_ssl_breakout_is_classified_when_price_accepts_below():
    index = pd.date_range("2026-01-01", periods=3, freq="min")
    df = pd.DataFrame(
        {
            "high": [100.2, 99.9, 99.5],
            "low": [99.8, 99.0, 98.5],
            "close": [100.0, 99.5, 98.8],
        },
        index=index,
    )
    pools = pd.DataFrame({"pool_id": [2], "side": ["SSL"], "price": [99.5]})

    events = detect_pool_events(df, pools, max_bars=2)

    assert len(events) == 1
    assert events.iloc[0]["event_type"] == "breakout"
