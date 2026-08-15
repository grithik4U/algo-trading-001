import pandas as pd

from trading_engine.volume_profile.confluence import score_profile_confluence


def test_profile_reference_is_attached_to_nearby_zone():
    zones = pd.DataFrame(
        {
            "zone_id": [1, 2],
            "price": [100.01, 105.0],
        }
    )

    result = score_profile_confluence(
        zones,
        {"POC": 100.0, "LVN": 105.0},
        tolerance=0.0005,
    )

    assert bool(result.loc[0, "near_POC"])
    assert bool(result.loc[1, "near_LVN"])
    assert result.loc[0, "profile_confluence_count"] == 1
    assert result.loc[1, "profile_confluence_count"] == 1
