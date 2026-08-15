import pandas as pd

from trading_engine.signals.fvg_lifecycle import track_fvg_lifecycle


def test_bullish_fvg_becomes_mitigated_after_full_fill():
    index = pd.date_range("2026-01-01", periods=5, freq="min")
    df = pd.DataFrame(
        {
            "high": [100, 104, 103, 103, 102],
            "low": [99, 101, 102, 100, 99],
            "close": [99.5, 103, 102.5, 100.5, 99.5],
        },
        index=index,
    )
    fvgs = pd.DataFrame(
        {
            "created_timestamp": [index[2]],
            "direction": ["bullish"],
            "lower_price": [100.0],
            "upper_price": [102.0],
        }
    )

    result = track_fvg_lifecycle(df, fvgs)

    assert result.iloc[0]["state"] == "mitigated"
    assert result.iloc[0]["mitigation_timestamp"] == index[3]
    assert result.iloc[0]["fill_fraction"] == 1.0


def test_bearish_fvg_is_invalidated_by_close_above_gap():
    index = pd.date_range("2026-01-01", periods=4, freq="min")
    df = pd.DataFrame(
        {
            "high": [105, 103, 101, 104],
            "low": [103, 100, 99, 102],
            "close": [104, 101, 100, 103.5],
        },
        index=index,
    )
    fvgs = pd.DataFrame(
        {
            "created_timestamp": [index[2]],
            "direction": ["bearish"],
            "lower_price": [101.0],
            "upper_price": [103.0],
        }
    )

    result = track_fvg_lifecycle(df, fvgs)

    assert result.iloc[0]["state"] == "invalidated"
    assert result.iloc[0]["invalidation_timestamp"] == index[3]
