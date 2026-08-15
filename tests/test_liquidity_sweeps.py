import pandas as pd

from trading_engine.liquidity.sweeps import detect_sweeps


def test_bsl_sweep_requires_reclaim():
    index = pd.date_range("2026-01-01 09:00", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "high": [100.0, 101.5, 102.0],
            "low": [99.0, 100.0, 100.5],
            "close": [99.5, 100.2, 101.5],
        },
        index=index,
    )
    levels = pd.DataFrame(
        [
            {
                "timestamp": index[0],
                "price": 101.0,
                "side": "BSL",
                "scope": "internal",
                "strength": 0.5,
                "confirmed_at": index[0],
            }
        ]
    )

    result = detect_sweeps(df, levels)
    assert len(result) == 1
    assert result.iloc[0]["timestamp"] == index[1]
    assert result.iloc[0]["sweep_depth"] == 0.5


def test_bsl_breakout_is_not_a_reclaim_sweep():
    index = pd.date_range("2026-01-01 09:00", periods=2, freq="min", tz="UTC")
    df = pd.DataFrame(
        {"high": [100.0, 102.0], "low": [99.0, 100.5], "close": [99.5, 101.5]},
        index=index,
    )
    levels = pd.DataFrame(
        [
            {
                "timestamp": index[0],
                "price": 101.0,
                "side": "BSL",
                "scope": "external",
                "strength": 1.0,
                "confirmed_at": index[0],
            }
        ]
    )

    assert detect_sweeps(df, levels).empty
