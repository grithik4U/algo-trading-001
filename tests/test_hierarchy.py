import pandas as pd

from trading_engine.structure.hierarchy import classify_structural_levels


def test_higher_timeframe_levels_are_external_by_default():
    levels = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "price": [100.0, 99.0],
            "side": ["high", "low"],
            "timeframe": ["4H", "5min"],
        }
    )

    result = classify_structural_levels(levels)

    assert result.iloc[0]["liquidity_class"] == "external"
    assert result.iloc[0]["liquidity_side"] == "BSL"
    assert result.iloc[1]["liquidity_class"] == "internal"
    assert result.iloc[1]["liquidity_side"] == "SSL"
