import pandas as pd

from trading_engine.signals.displacement import displacement_features


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

    result = displacement_features(
        df,
        atr_period=14,
        body_ratio_threshold=0.65,
        range_atr_threshold=1.5,
        volume_z_threshold=1.5,
        volume_period=20,
    )

    assert bool(result.iloc[-2]["is_displacement"])
