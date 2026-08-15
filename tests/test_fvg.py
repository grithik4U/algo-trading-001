import pandas as pd

from trading_engine.signals.fvg import detect_fvgs


def test_bullish_fvg_and_displacement_link():
    index = pd.date_range("2026-01-01", periods=3, freq="min")
    df = pd.DataFrame(
        {
            "high": [100.0, 104.0, 103.0],
            "low": [99.0, 101.0, 102.0],
            "is_displacement": [False, True, False],
        },
        index=index,
    )

    fvgs = detect_fvgs(df)

    assert len(fvgs) == 1
    assert fvgs.iloc[0]["direction"] == "bullish"
    assert fvgs.iloc[0]["lower_price"] == 100.0
    assert fvgs.iloc[0]["upper_price"] == 102.0
    assert bool(fvgs.iloc[0]["middle_displacement"])


def test_bearish_fvg():
    index = pd.date_range("2026-01-01", periods=3, freq="min")
    df = pd.DataFrame(
        {
            "high": [105.0, 103.0, 101.0],
            "low": [103.0, 100.0, 99.0],
        },
        index=index,
    )

    fvgs = detect_fvgs(df)

    assert len(fvgs) == 1
    assert fvgs.iloc[0]["direction"] == "bearish"
    assert fvgs.iloc[0]["lower_price"] == 101.0
    assert fvgs.iloc[0]["upper_price"] == 103.0
