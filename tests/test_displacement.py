import pandas as pd

from trading_engine.signals.displacement import detect_post_sweep_displacement, displacement_features


def test_large_body_range_can_be_displacement():
    n = 20
    index = pd.date_range("2026-01-01", periods=n + 2, freq="min")
    opens = [100.0] * n + [100.0, 100.0]
    highs = [100.5] * n + [104.0, 101.0]
    lows = [99.5] * n + [99.8, 100.0]
    closes = [100.1] * n + [103.5, 100.5]
    volumes = [100.0] * n + [500.0, 100.0]
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=index,
    )

    result = displacement_features(df, atr_period=14, body_ratio_threshold=0.65, range_atr_threshold=1.5, volume_z_threshold=1.5, volume_period=20)
    assert bool(result.iloc[-2]["is_displacement"])


def test_post_sweep_bullish_displacement_is_detected():
    index = pd.date_range("2026-01-01", periods=5, freq="min")
    df = pd.DataFrame(
        {
            "open": [100, 100, 99.5, 100.0, 100.2],
            "high": [101, 101, 100.5, 101.0, 102.0],
            "low": [99, 99, 99.0, 99.8, 100.0],
            "close": [100, 100, 100.0, 100.8, 101.8],
        },
        index=index,
    )

    event = detect_post_sweep_displacement(df, index[2], "bullish", structure_level=100.7, max_bars=3, range_multiple_threshold=1.0, body_ratio_threshold=0.6)
    assert event is not None
    assert event.timestamp == index[3]
    assert event.direction == "bullish"
    assert event.broke_structure
