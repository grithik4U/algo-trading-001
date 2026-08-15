import pandas as pd

from trading_engine.structure.breaks import detect_structure_breaks


def test_bullish_break_occurs_after_confirmed_high():
    index = pd.date_range("2026-01-01", periods=4, freq="min")
    df = pd.DataFrame(
        {
            "high": [100, 101, 102, 104],
            "low": [98, 99, 100, 101],
            "close": [99, 100, 101, 103],
        },
        index=index,
    )
    swings = pd.DataFrame(
        {
            "timestamp": [index[1]],
            "price": [101.0],
            "side": ["high"],
        }
    )

    events = detect_structure_breaks(df, swings)

    assert len(events) == 1
    assert events.iloc[0]["direction"] == "bullish"
    assert events.iloc[0]["break_timestamp"] == index[3]


def test_bearish_break_occurs_below_confirmed_low():
    index = pd.date_range("2026-01-01", periods=4, freq="min")
    df = pd.DataFrame(
        {
            "high": [102, 101, 100, 99],
            "low": [100, 99, 98, 96],
            "close": [101, 100, 99, 97],
        },
        index=index,
    )
    swings = pd.DataFrame(
        {
            "timestamp": [index[1]],
            "price": [99.0],
            "side": ["low"],
        }
    )

    events = detect_structure_breaks(df, swings)

    assert len(events) == 1
    assert events.iloc[0]["direction"] == "bearish"
    assert events.iloc[0]["break_timestamp"] == index[3]
