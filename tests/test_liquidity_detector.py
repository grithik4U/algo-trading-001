import pandas as pd

from trading_engine.liquidity.detector import detect_level_clusters


def test_clusters_equal_highs_into_buy_side_liquidity():
    levels = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="min"),
            "price": [105.00, 105.02, 105.01],
            "side": ["buy_side"] * 3,
            "source": ["swing_high"] * 3,
        }
    )

    candidates = detect_level_clusters(levels, tolerance=0.05, min_touches=2)

    assert len(candidates) == 1
    assert candidates[0].touches == 3
    assert 105.00 <= candidates[0].price <= 105.02
    assert candidates[0].strength == 3.0
