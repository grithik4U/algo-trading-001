import pandas as pd

from trading_engine.liquidity.detector import LiquidityCandidate
from trading_engine.liquidity.sweep import detect_sweeps


def test_buy_side_pool_is_swept_and_reclaimed():
    index = pd.date_range("2026-01-01", periods=3, freq="min")
    df = pd.DataFrame(
        {
            "high": [105, 105.8, 106],
            "low": [103, 104, 103],
            "close": [104, 105.0, 104.5],
        },
        index=index,
    )
    pools = [LiquidityCandidate(index[0], 105.0, "buy_side", "equal_highs", 2, 2.0)]

    events = detect_sweeps(df, pools)

    assert len(events) == 1
    assert events[0].sweep_timestamp == index[1]
    assert events[0].sweep_extreme == 105.8
    assert events[0].reclaimed


def test_sell_side_pool_is_swept_without_reclaim():
    index = pd.date_range("2026-01-01", periods=3, freq="min")
    df = pd.DataFrame(
        {
            "high": [102, 101, 103],
            "low": [100, 99.0, 98],
            "close": [101, 99.8, 99],
        },
        index=index,
    )
    pools = [LiquidityCandidate(index[0], 100.0, "sell_side", "equal_lows", 2, 2.0)]

    events = detect_sweeps(df, pools)

    assert len(events) == 1
    assert not events[0].reclaimed
