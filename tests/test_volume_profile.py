import pandas as pd

from trading_engine.volume_profile.profile import build_volume_profile


def test_profile_returns_poc_and_value_area():
    df = pd.DataFrame(
        {
            "high": [101, 102, 103, 102, 101],
            "low": [99, 100, 101, 100, 99],
            "close": [100, 101, 102, 101, 100],
            "volume": [10, 50, 100, 50, 10],
        }
    )

    profile = build_volume_profile(df, bins=20, value_area=0.70)

    assert len(profile.prices) == 20
    assert len(profile.volumes) == 20
    assert profile.val <= profile.poc <= profile.vah
    assert profile.volumes.sum() > 0


def test_typical_price_allocation_preserves_volume():
    df = pd.DataFrame(
        {
            "high": [10, 20],
            "low": [0, 10],
            "close": [5, 15],
            "volume": [100, 200],
        }
    )

    profile = build_volume_profile(df, bins=10, allocation="typical_price")

    assert abs(profile.volumes.sum() - 300.0) < 1e-9
