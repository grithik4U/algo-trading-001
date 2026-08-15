import pandas as pd

from trading_engine.structure.mss import confirm_swings, detect_mss


def test_confirmed_swing_high_is_not_emitted_before_right_bars():
    index = pd.date_range("2026-01-01", periods=7, freq="min")
    df = pd.DataFrame(
        {
            "high": [100, 101, 105, 102, 101, 100, 99],
            "low": [98, 99, 100, 99, 98, 97, 96],
            "close": [99, 100, 104, 101, 100, 99, 98],
        },
        index=index,
    )
    swings = confirm_swings(df, left=2, right=2)
    highs = [s for s in swings if s.kind == "swing_high"]
    assert highs
    assert highs[0].timestamp == index[2]


def test_bullish_mss_requires_close_above_confirmed_swing_high():
    index = pd.date_range("2026-01-01", periods=8, freq="min")
    df = pd.DataFrame(
        {
            "high": [100, 101, 105, 102, 101, 106, 108, 109],
            "low": [98, 99, 100, 99, 98, 102, 105, 106],
            "close": [99, 100, 104, 101, 100, 105.5, 107, 108],
        },
        index=index,
    )
    swings = confirm_swings(df, left=1, right=1)
    event = detect_mss(df, swings, "bullish")
    assert event is not None
    assert event.direction == "bullish"
    assert event.broken_level == 105
